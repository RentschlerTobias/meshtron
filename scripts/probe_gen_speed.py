#!/usr/bin/env python
"""probe_gen_speed.py -- measure GRPO rollout throughput + detect generate() hangs.

Diagnostic ONLY: it never modifies the production code (`generate.py`,
`train_grpo.py` are left untouched). It answers two questions that a bare
`train_grpo.py` run cannot:

  1. How long does one rollout take (ms/token, tokens/s) as a function of the
     decode budget? A single `cap` number hides whether the cost is linear in
     the emitted token count or whether the model never emits `stop` and always
     burns the full budget.
  2. Does the token loop make progress at all, or does it stall? A watchdog
     prints a heartbeat every N tokens with the wall-clock delta, so an
     all-masked / NaN logits pathology is visible immediately instead of after
     36 minutes of silence.

Because `generate()` prints nothing and returns only after the whole sequence,
this wrapper reproduces its loop *shape* (KV cache, slot convention, constrained
slot mask, multinomial draw) while keeping the same code path via the real
`generate.forward_cached` and `generate.slot_mask`.

Usage (on the training PC):
  uv run python scripts/probe_gen_speed.py \
    --ckpt data/hexarow_h05_model.pt \
    --tokens data/hexarow_tokens_h05_family_cart.pt \
    --item 0 --budgets 128,256,512 --G 2 --watchdog-every 64

  # just measure one realistic full-budget rollout (slow, but definitive):
  uv run python scripts/probe_gen_speed.py --ckpt ... --tokens ... --full --G 1
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from conditioning import build_cloud  # noqa: E402
from eval_family import load_model  # noqa: E402
import generate as gen_mod  # noqa: E402
from generate import forward_cached, slot_mask  # noqa: E402
from hexa_row_tokenizer import HexaRowTokenizer  # noqa: E402


def _specials(tok: HexaRowTokenizer) -> set[int]:
    core = tok.core
    return {core.start_token, core.end_token, core.sep_token,
            core.sep2_token, core.stop_token, core.pad_token}


def rollout_with_heartbeat(model, pc, fc, tok, core, specials, cap, temperature,
                           dev, dtype, coords, use_slot, watchdog_every,
                           label):
    """Reproduce generate()'s loop with progress output + stall detection.

    Returns (seq, n_tokens, seconds, stop_seen, stalled_at).
    """
    global _SPECIAL_SET  # noqa: PLW0603 - same global the real generate() uses
    gen_mod._SPECIAL_SET = set(specials)

    kv: list = [[None, None] for _ in model.blocks]
    seq = [core.start_token]
    pos0 = 0
    cnt = 0
    t0 = time.time()
    last_t = t0
    last_len = 0
    stalled_at = None

    with torch.no_grad():
        while len(seq) < cap:
            x = torch.tensor([[seq[-1]]], dtype=torch.long, device=dev)
            s = torch.zeros((1, 1), dtype=torch.long, device=dev)
            if use_slot:
                s.fill_(0 if cnt == 0
                        else (cnt - 1) % (3 if coords == "cart" else 4))
            with torch.autocast(dev, dtype=dtype, enabled=dev == "cuda"):
                logits = forward_cached(model, x, pc, fc, kv, pos0,
                                        slot=s if use_slot else None)
            pos0 += 1
            nxt_l = (logits[:, -1, :] / max(1e-9, temperature)).float()
            finite = bool(torch.isfinite(nxt_l).all())
            if not finite:
                print(f"  [{label}] WARNING non-finite logits at token "
                      f"{len(seq)} -> NaN risk in multinomial")
            nxt_l = nxt_l + slot_mask(tok, seq, cnt, nxt_l.shape[-1],
                                      coords).to(nxt_l.device)
            masked_all = bool(torch.isinf(nxt_l).all())
            if masked_all:
                print(f"  [{label}] ALL-MASKED distribution at token {len(seq)} "
                      f"(no legal token) -> multinomial undefined, aborting")
                stalled_at = len(seq)
                break
            nxt = int(torch.multinomial(F.softmax(nxt_l, dim=-1), 1).item())
            pos_rows = model.pos.weight.shape[0]
            if pos0 >= pos_rows:
                print(f"  [{label}] positional limit {pos_rows} reached without "
                      f"stop, aborting")
                stalled_at = len(seq)
                break
            seq.append(nxt)
            if nxt in specials:
                cnt = 0
            else:
                cnt += 1
            if nxt == core.stop_token:
                break

            if watchdog_every and len(seq) % watchdog_every == 0:
                now = time.time()
                d_len = len(seq) - last_len
                d_t = now - last_t
                rate = d_len / d_t if d_t > 0 else 0.0
                print(f"  [{label}] {len(seq):5d}/{cap} tok  "
                      f"+{d_t:6.1f}s  {rate:6.1f} tok/s  "
                      f"total {now - t0:6.1f}s", flush=True)
                last_t, last_len = now, len(seq)

    gen_mod._SPECIAL_SET = set()
    return seq, len(seq), time.time() - t0, seq[-1] == core.stop_token, stalled_at


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="data/hexarow_h05_model.pt")
    ap.add_argument("--tokens", default="data/hexarow_tokens_h05_family_cart.pt")
    ap.add_argument("--item", type=int, default=0, help="train item index")
    ap.add_argument("--budgets", default="128,256,512",
                    help="comma-separated decode budgets to time")
    ap.add_argument("--full", action="store_true",
                    help="also run one full-cap rollout (cap = max_len-1)")
    ap.add_argument("--G", type=int, default=2,
                    help="rollouts per budget (to expose per-rollout variance)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--watchdog-every", type=int, default=64,
                    help="heartbeat every N tokens (0 = off)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    print(f"device={dev} dtype={dtype}")

    ck, cfg, coords, npt, rb, zb, policy, max_len, missing = load_model(
        args.ckpt, dev)
    policy.eval()
    if missing.missing_keys:
        print(f"note: missing ckpt keys: {missing.missing_keys}")
    use_slot = "slot.weight" in ck["model"]
    cap_full = max_len - 1
    print(f"policy d={cfg['d']} L={cfg['layers']} H={cfg['heads']} "
          f"n_points={cfg['n_points']} coords={coords} npt={npt} "
          f"max_len={max_len} cap={cap_full} use_slot={use_slot}")

    tok = HexaRowTokenizer(r_bounds=rb, z_bounds=zb)
    core = tok.core
    specials = _specials(tok)

    ds = torch.load(args.tokens, weights_only=False)
    items = list(ds["train"])
    item = items[int(args.item) % len(items)]
    print(f"item[{args.item}] name={item.get('name')} blocks={item.get('blocks')} "
          f"has_surface_points={item.get('surface_points') is not None}")

    rng = np.random.default_rng(args.seed)
    pts, _ = build_cloud(item, cfg["n_points"], rb, zb, rng, blade_weight=3.0)
    pc = torch.as_tensor(pts[None], dtype=torch.float32, device=dev)
    fc = torch.tensor([float(item["blocks"])], device=dev)
    print(f"conditioning cloud shape={tuple(pc.shape)}")

    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    if args.full:
        budgets.append(cap_full)

    torch.manual_seed(args.seed)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print("=" * 72)
    for budget in budgets:
        print(f"budget={budget}")
        for g in range(args.G):
            seq, n, sec, stopped, stalled = rollout_with_heartbeat(
                policy, pc, fc, tok, core, specials, budget, args.temperature,
                dev, dtype, coords, use_slot, args.watchdog_every,
                label=f"cap{budget}/r{g}")
            per_tok = (sec / n * 1000.0) if n else float("nan")
            print(f"  -> tokens={n}  seconds={sec:.1f}  {per_tok:.1f} ms/tok  "
                  f"{n / sec if sec else 0:.1f} tok/s  stopped={stopped}  "
                  f"stalled_at={stalled}", flush=True)

    if dev == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"peak VRAM (allocated): {peak:.2f} GB")

    # extrapolation to the real GRPO step cost
    print("=" * 72)
    print("Interpretation:")
    print("  - If stopped=False for the small budgets: the policy does NOT emit "
          "stop early; the real run burns the full cap -> cost is linear in cap.")
    print("  - ms/tok from a small budget extrapolates directly to cap="
          f"{cap_full}: ~{cap_full} tok/rollout.")
    print("  - GRPO step cost ~ G * (rollout ms/tok * cap) + rescore passes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
