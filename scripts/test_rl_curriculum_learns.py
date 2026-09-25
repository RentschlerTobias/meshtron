#!/usr/bin/env python3
"""test_rl_curriculum_learns.py -- does a curriculum stage improve generation?

scripts/test_rl_curriculum.py proves the wiring works and the rewards rank a
good sequence above a corrupted one. That is not the claim that matters. This
measures the claim: after N RL steps in a stage, does the policy GENERATE
better blockings than before?

Everything is measured on the same held-out meshes, with the same seed, by a
yardstick that does not depend on which stage was trained:

    decode_share        the rollout decodes into at least one quad
    count_exact_share   it decodes into exactly the conditioned block count
    nondegenerate       mean fraction of quads with 4 distinct corners, area > 0
    chamfer             symmetric corner distance to the ground-truth blocking,
                        as a fraction of its bounding-box diagonal

Per-step `mean_R` is deliberately not the verdict: with one item per step it
belongs to whichever geometry was drawn, which is why HANDOFF_TRAINING_RL.md
section 4 warns against reading it as learning.

RL never starts from random weights -- a random policy has no reward to
improve on the structural terms -- so phase 1 trains a supervised checkpoint
first and caches it; later runs reuse it.

    uv run python scripts/test_rl_curriculum_learns.py --stages mesh
    uv run python scripts/test_rl_curriculum_learns.py --sft-epochs 60 \
        --rl-epochs 2 --meshes 256

Exit 0 always: a measurement, not a gate.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import meshtron  # noqa: E402,F401
from meshtron.data.quad_domain import load_quad_domain  # noqa: E402
from meshtron.training.config import PipelineConfig  # noqa: E402
from meshtron.training.rewards import (  # noqa: E402
    _quad_nondegenerate_fraction, _safe_detokenize, make_quadtron_reward)

STAGES = ("vertex", "face", "row", "mesh")


# --------------------------------------------------------------------------
# configuration -- one shape for every phase, so a checkpoint from the
# supervised run loads into the RL run and into the evaluator unchanged
# --------------------------------------------------------------------------

def make_cfg(args, data_path: str, log_dir: str, *, rl: bool = False,
             stage: str = "vertex", init_ckpt: str = "") -> PipelineConfig:
    return PipelineConfig(
        model_family="quadtron", dim=2, data_path=data_path,
        sorting_strategy=args.sorting_strategy, quantization=args.quantization,
        n_sample_points=args.n_sample_points, train_val_ratio=0.8,
        d_model=args.d_model, n_heads=args.n_heads,
        stage_layers=tuple(args.stage_layers), n_latents=args.n_latents,
        dropout=args.dropout, batch_size=1 if rl else args.batch_size,
        num_epochs=args.rl_epochs if rl else args.sft_epochs,
        learning_rate=args.rl_lr if rl else args.sft_lr,
        warmup_steps=10 if rl else 100,
        val_every_n_epochs=10 ** 6 if rl else 1,
        max_val_batches=1 if rl else 0,
        early_stopping_patience=10 ** 6,
        precision="fp32", rl_enabled=rl, rl_curriculum_stage=stage,
        rl_rollouts_per_condition=args.G, rl_max_length=args.max_length,
        rl_temperature=args.temperature, init_checkpoint=init_ckpt,
        log_dir=log_dir, seed=args.seed, save_last=True,
    )


def _build(cfg: PipelineConfig):
    from meshtron.training.trainer import Trainer
    return Trainer(cfg)


# --------------------------------------------------------------------------
# the yardstick
# --------------------------------------------------------------------------

def _chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric mean nearest-neighbour distance, both directions."""
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float(0.5 * (d.min(axis=1).mean() + d.min(axis=0).mean()))


def evaluate(trainer, args, stage_for_reward: str) -> dict:
    """K rollouts per held-out mesh, scored by the stage-independent yardstick
    (plus the stage's own reward, for reference only)."""
    device = trainer.device
    tok = trainer.tokenizer
    dataset = trainer.val_loader.dataset
    reward_fn = make_quadtron_reward(tok, stage_for_reward)
    torch.manual_seed(args.eval_seed)

    decodes, exact, nondeg, chamfers, rewards = [], [], [], [], []
    n_items = min(args.eval_items, len(dataset))
    for i in range(n_items):
        item = dataset[i]
        mesh = dataset.meshes[i]
        gt = mesh.x[:, :2].numpy().astype(float)
        diag = float(np.linalg.norm(gt.max(axis=0) - gt.min(axis=0))) or 1.0
        # detokenize reads the bounds of the LAST tokenized mesh, so re-tokenize
        # this one first -- otherwise every rollout is decoded in a neighbour's
        # coordinate frame and the chamfer is meaningless.
        tok.tokenize(mesh.x[:, :2], mesh.faces)

        n_faces = int(item["face_count"])
        pc = item["point_cloud"][None].to(device)
        fc = torch.tensor([float(n_faces)], device=device)
        start = item["input_tokens"][None, :8].to(device)
        K = args.eval_k
        rollouts = trainer.policy.sample(
            point_cloud=pc.expand(K, *pc.shape[1:]), face_count=fc.expand(K),
            start_tokens=start.expand(K, -1), max_length=args.max_length,
            temperature=args.temperature, eos_token=tok.end_token,
        )
        for k in range(K):
            ids = rollouts[k].tolist()
            rewards.append(reward_fn(ids, n_faces))
            decoded = _safe_detokenize(tok, ids)
            decodes.append(decoded is not None)
            if decoded is None:
                exact.append(False)
                nondeg.append(0.0)
                chamfers.append(float("nan"))
                continue
            verts, quads = decoded
            exact.append(int(quads.size(1)) == n_faces)
            nondeg.append(_quad_nondegenerate_fraction(verts, quads))
            chamfers.append(_chamfer(verts.numpy().astype(float), gt) / diag)

    finite = [c for c in chamfers if np.isfinite(c)]
    return {
        "per_rollout": {"decodes": [bool(d) for d in decodes],
                        "exact": [bool(e) for e in exact],
                        "nondegenerate": [float(n) for n in nondeg],
                        "chamfer": [float(c) for c in chamfers]},
        "n_rollouts": len(decodes),
        "decode_share": float(np.mean(decodes)),
        "count_exact_share": float(np.mean(exact)),
        "nondegenerate": float(np.mean(nondeg)),
        "chamfer": float(np.mean(finite)) if finite else float("nan"),
        "mean_R": float(np.mean(rewards)),
        "std_R": float(statistics.pstdev(rewards)),
    }


def sign_test(before: list, after: list) -> tuple:
    """Matched pairs, same item and same rollout index under the same eval seed.
    Returns (better, worse, same, two-sided p) -- exact binomial at q = 0.5 over
    the discordant pairs, which is what HANDOFF_TRAINING_RL.md section 4 reports
    for the hexarow run."""
    from math import comb
    better = sum(1 for b, a in zip(before, after) if a > b)
    worse = sum(1 for b, a in zip(before, after) if a < b)
    same = len(before) - better - worse
    n = better + worse
    if n == 0:
        return better, worse, same, 1.0
    k = min(better, worse)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return better, worse, same, min(1.0, 2 * tail)


# --------------------------------------------------------------------------

def phase_data(args, work: str) -> str:
    path = os.path.join(work, f"quad2d_{args.meshes}.pt")
    if not os.path.exists(path):
        meshes = load_quad_domain(args.corpus, limit=args.meshes) if args.corpus \
            else load_quad_domain(limit=args.meshes)
        torch.save(meshes, path)
        print(f"data: {len(meshes)} meshes -> {path}")
    else:
        print(f"data: reusing {path}")
    return path


def phase_sft(args, data_path: str, work: str) -> str:
    ckpt = os.path.join(work, "sft_last.pt")
    if os.path.exists(ckpt) and not args.retrain:
        print(f"sft: reusing {ckpt}")
        return ckpt
    cfg = make_cfg(args, data_path, os.path.join(work, "runs_sft"))
    t0 = time.time()
    trainer = _build(cfg)
    result = trainer.run()
    src = os.path.join(str(trainer.logger.run_dir), "last.pt")
    os.replace(src, ckpt)
    print(f"sft: {args.sft_epochs} epochs, best val bpt {result.best_val_bpt:.4f} "
          f"at epoch {result.best_epoch}, {time.time() - t0:.0f} s -> {ckpt}")
    return ckpt


def phase_eval_only(args, data_path: str, work: str, ckpt: str,
                    stage: str) -> dict:
    """Re-score the cached checkpoints -- no training, so the sample size can
    be raised without paying for another run."""
    rl_ckpt = os.path.join(work, f"rl_{stage}.pt")
    if not os.path.exists(rl_ckpt):
        raise FileNotFoundError(f"{rl_ckpt} -- run without --eval-only first")
    before = evaluate(_build(make_cfg(args, data_path,
                                      os.path.join(work, "runs_eval"),
                                      init_ckpt=ckpt)), args, stage)
    after = evaluate(_build(make_cfg(args, data_path,
                                     os.path.join(work, "runs_eval"),
                                     init_ckpt=rl_ckpt)), args, stage)
    return {"stage": stage, "steps": -1, "seconds": 0.0,
            "before": before, "after": after}


def phase_stage(args, data_path: str, work: str, ckpt: str, stage: str) -> dict:
    eval_cfg = make_cfg(args, data_path, os.path.join(work, "runs_eval"),
                        init_ckpt=ckpt)
    before = evaluate(_build(eval_cfg), args, stage)

    rl_cfg = make_cfg(args, data_path, os.path.join(work, f"runs_rl_{stage}"),
                      rl=True, stage=stage, init_ckpt=ckpt)
    t0 = time.time()
    trainer = _build(rl_cfg)
    steps_per_epoch = len(trainer.train_loader)
    trainer.run()
    rl_ckpt = os.path.join(work, f"rl_{stage}.pt")
    os.replace(os.path.join(str(trainer.logger.run_dir), "last.pt"), rl_ckpt)
    took = time.time() - t0

    after_cfg = make_cfg(args, data_path, os.path.join(work, "runs_eval"),
                         init_ckpt=rl_ckpt)
    after = evaluate(_build(after_cfg), args, stage)
    return {"stage": stage, "steps": steps_per_epoch * args.rl_epochs,
            "seconds": took, "before": before, "after": after}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--work", default=os.path.join(ROOT, "runs", "rl_curriculum"))
    ap.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    ap.add_argument("--meshes", type=int, default=256)
    ap.add_argument("--sft-epochs", type=int, default=60)
    ap.add_argument("--sft-lr", type=float, default=3e-4)
    ap.add_argument("--rl-epochs", type=int, default=2)
    ap.add_argument("--rl-lr", type=float, default=2e-5)
    ap.add_argument("--retrain", action="store_true",
                    help="retrain the supervised checkpoint even if cached")
    ap.add_argument("--eval-only", action="store_true",
                    help="score the cached checkpoints again, no training")
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-latents", type=int, default=32)
    ap.add_argument("--stage-layers", type=int, nargs="+", default=[2, 2, 2])
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--quantization", type=int, default=128)
    ap.add_argument("--sorting-strategy", type=int, default=2)
    ap.add_argument("--n-sample-points", type=int, default=256)
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--eval-items", type=int, default=8)
    ap.add_argument("--eval-k", type=int, default=6)
    ap.add_argument("--eval-seed", type=int, default=1234)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    os.makedirs(args.work, exist_ok=True)
    data_path = phase_data(args, args.work)
    ckpt = phase_sft(args, data_path, args.work)

    results = []
    for stage in args.stages:
        r = (phase_eval_only(args, data_path, args.work, ckpt, stage)
             if args.eval_only
             else phase_stage(args, data_path, args.work, ckpt, stage))
        results.append(r)
        if not args.eval_only:
            print(f"rl {stage}: {r['steps']} steps, {r['seconds']:.0f} s")

    keys = ("decode_share", "count_exact_share", "nondegenerate", "chamfer",
            "mean_R", "std_R")
    print(f"\n{'stage':<7} {'metric':<18} {'before':>9} {'after':>9} {'delta':>9}")
    for r in results:
        for k in keys:
            b, a = r["before"][k], r["after"][k]
            print(f"{r['stage']:<7} {k:<18} {b:>9.4f} {a:>9.4f} {a - b:>+9.4f}")
        print(f"{'':<7} {'(' + str(r['steps']) + ' steps)':<18}")

    print(f"\n{'stage':<7} {'paired on':<14} {'better':>7} {'worse':>6} "
          f"{'same':>6} {'p':>8}")
    for r in results:
        for k in ("exact", "nondegenerate"):
            b, w, sm, p = sign_test(r["before"]["per_rollout"][k],
                                    r["after"]["per_rollout"][k])
            print(f"{r['stage']:<7} {k:<14} {b:>7} {w:>6} {sm:>6} {p:>8.4f}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"args": vars(args), "results": results}, fh, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
