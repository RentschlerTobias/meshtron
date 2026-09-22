"""train_grpo.py — GRPO-Spike auf der SFT-Cart-Policy (P2-2).

Gruppen-relativer Policy-Gradient (kein Critic) mit KL-Anker an eine
eingefrorene SFT-Referenz. Rollouts via generate.generate (KV-Cache + slot-Mask,
identischer Conditioning-Pfad conditioning.build_cloud blade_weight=3.0),
Scoring ueber rewards_hexarow. Die eigenen Samples werden mit demselben
own-slot-Forward wie im Decode re-gescored (token-level logp); PPO-Clip gegen
den Rollout-logp, 1 inneres Epoch = on-policy.

lr 5e-6 (~2 Groessenordnungen unter SFT 3e-4): RL-Gradienten sind um
Gruppenmittel zentriert und stark verrauscht; die SFT-Rate wuerde die
Policy in wenigen Steps kollabieren lassen. temp 1.0 (statt Eval 0.7) fuer
mehr Explorationsvarianz innerhalb der Gruppe; beta 0.04 haelt die Policy
nahe an SFT.

  uv run python train_grpo.py --smoke
  uv run python train_grpo.py --steps 300 --G 8
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np
import torch
import torch.nn.functional as F

from conditioning import build_cloud
from eval_family import load_model
from generate import generate
from hexa_row_tokenizer import HexaRowTokenizer
from rewards_hexarow import HexaRowRewardConfig, make_hexarow_reward
from train_hexarow_full import _slot_ids

CSV_FIELDS = ["step", "mean_R", "r_valid_share", "r_quality_mean",
              "r_conform_mean", "KL", "entropy", "grad_norm", "loss",
              "adv_mean", "peak_vram_gb"]


def credit_weights(seq: list[int], specials: set, sep: int, mode: str,
                   decay: float) -> np.ndarray:
    """Gewichte ueber die gesampelten Tokens (targets = seq[1:]).
    uniform: 1.0. sep: decay**(Tokens nach dem Token in derselben Row), d.h.
    maximal am Row-Ende (SEP) und faellt rueckwaerts Richtung Row-Start."""
    n = len(seq) - 1
    if mode == "uniform":
        return np.ones(n, dtype=np.float32)
    w = np.ones(n, dtype=np.float32)
    cnt = 0
    for j in range(n - 1, -1, -1):
        w[j] = decay ** cnt
        if seq[1 + j] == sep:
            cnt = 0
        elif seq[1 + j] not in specials:
            cnt += 1
    return w


def rescore(model, seqs: list[list[int]], pc, fc, specials: set, npt: int,
            pad_id: int, dev: str, dtype, use_slot: bool, grad: bool):
    """Batched own-slot forward ueber ganze Rollouts -> per-Token-logp + Maske."""
    B, L = len(seqs), max(len(s) for s in seqs)
    x = torch.full((B, L), pad_id, dtype=torch.long)
    slot = torch.zeros((B, L - 1), dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.as_tensor(s, dtype=torch.long)
        ids = _slot_ids(s, specials, npt)
        slot[i, :len(s) - 1] = torch.as_tensor(ids[:len(s) - 1], dtype=torch.long)
    inp, tgt = x[:, :-1].to(dev), x[:, 1:].to(dev)
    with torch.set_grad_enabled(grad):
        with torch.autocast(dev, dtype=dtype, enabled=dev == "cuda"):
            logits = model(inp, pc, fc, slot=slot.to(dev) if use_slot else None)
    logp = F.log_softmax(logits.float(), dim=-1).gather(
        -1, tgt.unsqueeze(-1)).squeeze(-1)
    mask = (tgt != pad_id)
    with torch.no_grad():
        lp = F.log_softmax(logits.float(), dim=-1)
        ent = float((-(lp.exp() * lp).sum(-1) * mask).sum() / mask.sum())
    return logp, mask, ent


def main() -> int:
    ap = argparse.ArgumentParser(description="GRPO-Spike HexaRow cart")
    ap.add_argument("--ckpt", default="data/hexarow_sft_cart_ep584.pt")
    ap.add_argument("--tokens", default="data/hexarow_tokens_family_cart.pt")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--items-per-step", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--credit", choices=["uniform", "sep"], default="uniform")
    ap.add_argument("--decay", type=float, default=0.9)
    ap.add_argument("--log-csv", default="data/grpo_cart_log.csv")
    ap.add_argument("--ckpt-prefix", default="data/grpo_cart")
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.steps, args.items_per_step, args.G = 3, 4, 4
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32

    ck, cfg, coords, npt, rb, zb, policy, max_len, missing = load_model(
        args.ckpt, dev)
    if missing.missing_keys:
        print(f"note: fehlende CKPT-Keys: {missing.missing_keys}")
    policy.eval()
    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    use_slot = "slot.weight" in ck["model"]
    cap = max_len - 1
    vocab, pad_id = int(ck["vocab"]), int(ck["pad_id"])

    tok = HexaRowTokenizer(r_bounds=rb, z_bounds=zb)
    core = tok.core
    specials = {core.start_token, core.end_token, core.sep_token,
                core.sep2_token, core.stop_token, core.pad_token}
    reward_cfg = HexaRowRewardConfig()
    reward_fn = make_hexarow_reward(tok, reward_cfg, coords=coords,
                                    stop_id=core.stop_token)

    ds = torch.load(args.tokens, weights_only=False)
    items = list(ds["train"])
    print(f"policy d={cfg['d']} L={cfg['layers']} H={cfg['heads']} coords={coords} "
          f"npt={npt} cap={cap} | {len(items)} train items | G={args.G} "
          f"items/step={args.items_per_step} steps={args.steps} lr={args.lr} "
          f"beta={args.beta} temp={args.temperature} credit={args.credit}")

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0)
    order_rng = np.random.default_rng(args.seed)
    cloud_rng = np.random.default_rng(args.seed + 1_000_000)
    torch.manual_seed(args.seed)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()

    new_csv = not os.path.exists(args.log_csv)
    logf = open(args.log_csv, "a", newline="")
    writer = csv.DictWriter(logf, fieldnames=CSV_FIELDS)
    if new_csv:
        writer.writeheader()

    t0 = time.time()
    order, cursor = order_rng.permutation(len(items)), 0
    for step in range(1, args.steps + 1):
        batch = []
        for _ in range(args.items_per_step):
            if cursor >= len(order):
                order, cursor = order_rng.permutation(len(items)), 0
            batch.append(items[int(order[cursor])])
            cursor += 1

        seqs, pcs, adv_all, stats = [], [], [], []
        for item in batch:
            pts, _ = build_cloud(item, cfg["n_points"], rb, zb, cloud_rng,
                                 blade_weight=3.0)
            pc = torch.as_tensor(pts[None], dtype=torch.float32, device=dev)
            fc = torch.tensor([float(item["blocks"])], device=dev)
            rolls = [generate(policy, pc, fc, core.start_token, core.stop_token,
                              core.sep_token, cap, args.temperature, 0, dev,
                              dtype, specials, use_slot, tok=tok,
                              constrained=True, coords=coords)
                     for _ in range(args.G)]
            terms = [reward_fn(s, item) for s in rolls]
            R = torch.tensor([[t.total for t in terms]], device=dev)
            A = (R - R.mean(1, keepdim=True)) / (R.std(1, unbiased=False,
                                                       keepdim=True) + 1e-6)
            seqs.extend(rolls)
            pcs.append(pc.expand(len(rolls), -1, -1))
            adv_all.extend(A[0].tolist())
            stats.extend(terms)

        pc_b = torch.cat(pcs, dim=0)
        fc_b = torch.tensor([float(it["blocks"]) for it in batch
                             for _ in range(args.G)], device=dev)
        adv = torch.tensor(adv_all, dtype=torch.float32, device=dev)
        logp_new, mask, ent = rescore(policy, seqs, pc_b, fc_b, specials, npt,
                                      pad_id, dev, dtype, use_slot, grad=True)
        with torch.no_grad():
            logp_old, _, _ = rescore(policy, seqs, pc_b, fc_b, specials, npt,
                                     pad_id, dev, dtype, use_slot, grad=False)
            logp_ref, _, _ = rescore(ref, seqs, pc_b, fc_b, specials, npt,
                                     pad_id, dev, dtype, use_slot, grad=False)

        w_np = np.zeros((len(seqs), mask.shape[1]), dtype=np.float32)
        for i, s in enumerate(seqs):
            w_np[i, :len(s) - 1] = credit_weights(s, specials, core.sep_token,
                                                  args.credit, args.decay)
        cw = torch.as_tensor(w_np, dtype=torch.float32, device=dev) * mask
        denom = cw.sum()

        ratio = torch.exp(logp_new - logp_old)
        a_t = adv[:, None]
        obj = torch.minimum(ratio * a_t,
                            torch.clamp(ratio, 1.0 - args.clip,
                                        1.0 + args.clip) * a_t)
        loss_pg = -(obj * cw).sum() / denom
        kl = ((logp_new - logp_ref) * mask).sum() / mask.sum()
        loss = loss_pg + args.beta * kl

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(),
                                               args.grad_clip)
        opt.step()

        valid_share = float(np.mean([t.valid for t in stats]))
        q_mean = float(np.mean([t.r_quality for t in stats]))
        c_mean = float(np.mean([t.r_conform for t in stats]))
        mean_R = float(np.mean([t.total for t in stats]))
        vram = (torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0.0)
        row = {"step": step, "mean_R": round(mean_R, 5),
               "r_valid_share": round(valid_share, 4),
               "r_quality_mean": round(q_mean, 5),
               "r_conform_mean": round(c_mean, 5),
               "KL": round(float(kl.detach()), 6),
               "entropy": round(ent, 5),
               "grad_norm": round(float(gnorm), 5),
               "loss": round(float(loss.detach()), 5),
               "adv_mean": round(float(adv.mean()), 6),
               "peak_vram_gb": round(vram, 3)}
        writer.writerow(row)
        logf.flush()
        print(f"step {step:4d}  R {mean_R:.4f}  valid {valid_share:.2f}  "
              f"q {q_mean:+.4f}  c {c_mean:.4f}  KL {row['KL']:+.5f}  "
              f"H {ent:.4f}  |g| {row['grad_norm']:.3f}  "
              f"loss {row['loss']:+.4f}  vram {vram:.2f}GB")

        if (not args.no_save and not args.smoke
                and step % args.ckpt_every == 0):
            cfg_out = dict(cfg)
            cfg_out.update({"npt": npt, "coords": coords, "grpo_step": step,
                            "grpo_seed": args.seed, "grpo_lr": args.lr,
                            "grpo_beta": args.beta,
                            "grpo_temperature": args.temperature})
            path = f"{args.ckpt_prefix}_step{step}.pt"
            torch.save({"model": policy.state_dict(), "cfg": cfg_out,
                        "vocab": vocab, "pad_id": pad_id, "r_bounds": rb,
                        "z_bounds": zb, "coords": coords, "npt": npt}, path)
            print(f"  checkpoint -> {path}")
    logf.close()
    vram = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0.0
    print(f"done: {args.steps} steps in {time.time() - t0:.0f}s | "
          f"peak VRAM {vram:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
