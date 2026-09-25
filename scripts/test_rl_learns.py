"""test_rl_learns.py — does the reward-based stage actually improve generation?

The pretraining objective is next-token cross-entropy with the ground truth fed
in at every position. scripts/test_memorisation.py measured what that buys:
teacher-forced accuracy 0.675 on the training split against 0.166 on validation,
yet under free generation not one ground-truth block comes back, on either
split, and the sequence diverges at the first coordinate token. Predicting one
token well is not the same as producing a good blocking.

That is what the RL stage is for: it scores a WHOLE generated sequence -- the
blocking's validity, its cell quality, its conformity -- and pushes the policy
towards sequences that score better, with credit spread back over the tokens
that produced them (`--credit sep` decays backwards from each row separator, so
a row is the unit that carries blame).

This script measures whether that works, on a real checkpoint rather than on a
1.3 M model whose every reward is 0:

  1 baseline    generate on held-out geometries, score them
  2 GRPO        N steps, logging mean reward, valid share, KL, gradient norm
  3 after       the same geometries again with the updated policy
  4 verdict     did the reward rise, and did the generated output improve

Two things are deliberately separated. `mean_R` rising says the optimiser is
doing its job on its own objective. The generated meshes improving says that
objective was worth optimising. Only the second is the claim that matters.

    uv run python scripts/test_rl_learns.py --steps 20
    uv run python scripts/test_rl_learns.py --steps 20 --credit sep

Exit 0 always: a measurement, not a gate.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from meshtron.data import conditioning  # noqa: E402
from meshtron.data.hexa_row_tokenizer import HexaRowTokenizer  # noqa: E402
from meshtron.geometry.mesh_validation import validate_generated_mesh  # noqa: E402
from meshtron.training.generate import detokenize_safe, generate  # noqa: E402
from meshtron.training.rewards_hexarow import (  # noqa: E402
    HexaRowRewardConfig, make_hexarow_reward)
from scripts.compare_viz import _to_cart  # noqa: E402
from scripts.eval_family import load_model  # noqa: E402


def score_policy(ckpt: str, items: list, k: int, temperature: float,
                 seed: int, dev: str) -> dict:
    """Generate k rollouts per item and score them with the training reward.

    Uses the same reward function the GRPO loop uses, so "before" and "after"
    are measured on the objective that was optimised, and additionally reports
    structural validity, which is what a downstream mesh actually needs.
    """
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    ck, cfg, coords, npt, rb, zb, model, max_len, _ = load_model(ckpt, dev)
    tok = HexaRowTokenizer(r_bounds=rb, z_bounds=zb)
    core = tok.core
    specials = {core.start_token, core.end_token, core.sep_token,
                core.sep2_token, core.stop_token, core.pad_token}
    use_slot = "slot.weight" in ck["model"]
    reward_fn = make_hexarow_reward(tok, HexaRowRewardConfig(), coords=coords,
                                    stop_id=core.stop_token)
    rows = []
    for it in items:
        rng = np.random.default_rng(seed)
        pts, _ = conditioning.build_cloud(it, cfg["n_points"], rb, zb, rng,
                                          blade_weight=3.0)
        pc = torch.as_tensor(pts[None], dtype=torch.float32, device=dev)
        fc = torch.tensor([float(it["blocks"])], device=dev)
        for r in range(k):
            torch.manual_seed(seed + r)
            seq = generate(model, pc, fc, core.start_token, core.stop_token,
                           core.sep_token, max_len - 1, temperature,
                           0 if k > 1 else 1, dev, dtype, specials, use_slot,
                           tok=tok, constrained=True, coords=coords)
            # reward_fn takes the dataset item: it reads blocks, and
            # surface_points/is_blade for the conformity term.
            terms = reward_fn([int(t) for t in seq], it)
            res, _ = detokenize_safe(seq, tok, core.stop_token, coords=coords)
            valid, nb = False, 0
            if res is not None:
                vpt, blk = res
                nb = int(blk.shape[0])
                valid = bool(validate_generated_mesh(
                    _to_cart(vpt.numpy(), coords), blk.numpy()).valid)
            rows.append({"name": it["name"], "rollout": r,
                         "R": float(getattr(terms, "total", 0.0)),
                         "r_valid": float(getattr(terms, "r_valid", 0.0)),
                         "r_quality": float(getattr(terms, "r_quality", 0.0)),
                         "r_conform": float(getattr(terms, "r_conform", 0.0)),
                         "structurally_valid": valid, "n_blocks": nb,
                         "blocks_gt": int(it["blocks"])})
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    agg = {
        "n": len(rows),
        "mean_R": round(float(np.mean([r["R"] for r in rows])), 4),
        "max_R": round(float(np.max([r["R"] for r in rows])), 4),
        "valid_share": round(float(np.mean(
            [r["structurally_valid"] for r in rows])), 4),
        "count_exact_share": round(float(np.mean(
            [r["n_blocks"] == r["blocks_gt"] for r in rows])), 4),
        "mean_r_quality": round(float(np.mean(
            [r["r_quality"] for r in rows])), 4),
        "mean_r_conform": round(float(np.mean(
            [r["r_conform"] for r in rows])), 4),
    }
    return {"aggregate": agg, "rollouts": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description="does GRPO improve generation?")
    ap.add_argument("--ckpt", default="data/hexarow_sft_cart_ep584.pt")
    ap.add_argument("--tokens", default="data/hexarow_tokens_family_cart.pt")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--credit", choices=["uniform", "sep"], default="uniform")
    ap.add_argument("--kl-estimator", choices=["k3", "naive"], default="k3")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--eval-items", type=int, default=4)
    ap.add_argument("--eval-k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = args.out_dir or os.path.join(ROOT, "data", "rl_check")
    os.makedirs(out_dir, exist_ok=True)
    ds = torch.load(args.tokens, weights_only=False)
    # Held-out geometries: whether RL helps on what it trained on is not the
    # question, and the GRPO loop itself samples from the train split.
    eval_items = list(ds["val"])[:args.eval_items]
    print(f"device {dev} | ckpt {os.path.basename(args.ckpt)} | "
          f"{len(eval_items)} held-out geometries x {args.eval_k} rollouts | "
          f"GRPO {args.steps} steps G={args.G} credit={args.credit} "
          f"kl={args.kl_estimator} lr={args.lr}")

    print("\n[1] baseline")
    before = score_policy(args.ckpt, eval_items, args.eval_k,
                          args.temperature, args.seed, dev)
    b = before["aggregate"]
    print(f"    mean_R {b['mean_R']:.4f}  valid {b['valid_share']:.2f}  "
          f"count_exact {b['count_exact_share']:.2f}  "
          f"quality {b['mean_r_quality']:.4f}  conform {b['mean_r_conform']:.4f}")

    print(f"\n[2] GRPO {args.steps} steps")
    log = os.path.join(out_dir, f"grpo_{args.credit}.csv")
    prefix = os.path.join(out_dir, f"grpo_{args.credit}")
    cmd = [sys.executable, os.path.join(ROOT, "meshtron", "training",
                                        "train_grpo.py"),
           "--ckpt", args.ckpt, "--tokens", args.tokens,
           "--steps", str(args.steps), "--G", str(args.G),
           "--items-per-step", "1", "--lr", str(args.lr),
           "--credit", args.credit, "--kl-estimator", args.kl_estimator,
           "--temperature", "1.0", "--seed", str(args.seed),
           "--log-csv", log, "--ckpt-prefix", prefix,
           "--ckpt-every", str(args.steps)]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    secs = time.time() - t0
    if not os.path.exists(log):
        print(f"    GRPO produced no log (exit {p.returncode}):\n"
              f"    {p.stderr.strip()[-600:]}")
        return 0
    rows = list(csv.DictReader(open(log)))
    if len(rows) < args.steps:
        # Surface why. An earlier version printed stderr only when the log was
        # missing entirely, so a run killed by GPU contention after 4 of 40
        # steps looked like a completed 4-step run.
        print(f"    WARNING: {len(rows)} of {args.steps} steps logged, the run "
              f"did not finish (exit {p.returncode})")
        if p.stderr.strip():
            print(f"    stderr tail: {p.stderr.strip()[-500:]}")
    R = [float(r["mean_R"]) for r in rows]
    vs = [float(r["r_valid_share"]) for r in rows]
    kl = [float(r["KL"]) for r in rows]
    gn = [float(r["grad_norm"]) for r in rows]
    half = max(1, len(R) // 2)
    # This trend is NOT a learning signal. With --items-per-step 1 each step's
    # reward belongs to whichever single geometry was drawn, so the step series
    # mostly measures which geometries came up: a direct probe gave R of 0.88,
    # 1.25, 1.68, 1.64, 1.14, 0.88 over six consecutive steps on an unchanged
    # task. The before/after evaluation on FIXED held-out geometries is the
    # measurement; this line is reported only to show the loop is running.
    print(f"    {len(rows)} steps in {secs:.0f}s | mean_R per step (noisy, one "
          f"geometry each) first half {np.mean(R[:half]):.4f} -> second half "
          f"{np.mean(R[half:]):.4f}")
    print(f"    valid_share {np.mean(vs[:half]):.2f} -> {np.mean(vs[half:]):.2f}"
          f" | KL {min(kl):.2e}..{max(kl):.2e} | grad {max(gn):.3f}")
    if max(abs(v) for v in R) == 0.0:
        print("    every reward was 0: nothing to learn from, the update is a "
              "no-op by construction")

    ck_after = f"{prefix}_step{args.steps}.pt"
    report = {"args": vars(args), "before": before,
              "grpo": {"steps": len(rows), "seconds": round(secs, 1),
                       "mean_R": R, "valid_share": vs, "KL": kl,
                       "grad_norm": gn}}
    if os.path.exists(ck_after):
        print("\n[3] after")
        after = score_policy(ck_after, eval_items, args.eval_k,
                             args.temperature, args.seed, dev)
        a = after["aggregate"]
        report["after"] = after
        print(f"    mean_R {a['mean_R']:.4f}  valid {a['valid_share']:.2f}  "
              f"count_exact {a['count_exact_share']:.2f}  "
              f"quality {a['mean_r_quality']:.4f}  "
              f"conform {a['mean_r_conform']:.4f}")
        print(f"\n[4] {'metric':18s} {'before':>9s} {'after':>9s}   change")
        for k in ("mean_R", "valid_share", "count_exact_share",
                  "mean_r_quality", "mean_r_conform"):
            d = a[k] - b[k]
            print(f"    {k:18s} {b[k]:9.4f} {a[k]:9.4f}   {d:+.4f}")
        print("\n    mean_R is the objective GRPO optimised; valid_share and "
              "the quality/conform terms are what a mesh needs. A rise in the "
              "first without the others is the optimiser gaming its reward.")
    else:
        print(f"\n[3] no post-GRPO checkpoint at {ck_after}, skipping after")
    rp = os.path.join(out_dir, f"rl_check_{args.credit}.json")
    json.dump(report, open(rp, "w"), indent=2, default=float)
    print(f"\nsaved {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
