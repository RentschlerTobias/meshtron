#!/usr/bin/env python3
"""eval_quadtron_curve.py -- Phase 1 baseline: the trained Polytron CurveModel
applied to Quadtron block structures.

Quadtron (Tokenizer2D) emits straight-edged blocks; the edge shape is what
decides whether the transfinite fill folds. This script asks whether the edge
head trained for Polytron transfers to Quadtron structures, WITHOUT training
anything:

  --mode gt        the head on the ground-truth Polytron structures (sanity:
                   should reproduce the checkpoint's own val numbers)
  --mode quadtron  the same structures pushed through Quadtron's coordinate
                   representation (per-mesh min/max quantisation to
                   `quantization_levels` bins and back) before the head sees
                   them -- isolates the representation shift Quadtron adds

Per undirected block edge the head predicts 6 quantised Bezier offsets; we
report the bin error against the ground truth and the geometric curve error
(max chord-relative distance between predicted and GT cubic), plus the share of
edges whose six bins are exact.

    uv run python scripts/eval_quadtron_curve.py --mode gt
    uv run python scripts/eval_quadtron_curve.py --mode quadtron --quantization 256
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from meshtron.data.polytron_blocks import (PolytronSpec, bezier, build_cloud,  # noqa: E402
                                           offsets_to_ctrl)
from meshtron.training.polytron_sample import clear_cache  # noqa: E402
from meshtron.training.train_polytron import load_stage  # noqa: E402


def quadtron_roundtrip(V: np.ndarray, levels: int) -> np.ndarray:
    """Quadtron's 3D coordinate representation, exactly.

    `tokenizer_v2.Tokenizer2D._quantize_coords` (dim=3): per-column min/max over
    the mesh, scale to [0, levels-1], round, clamp; detokenize inverts it. We
    apply encode+decode without the token sequence because the round trip is
    deterministic and per-coordinate.
    """
    V = np.asarray(V, float)
    mins, maxs = V.min(0), V.max(0)
    rng = np.where(maxs != mins, maxs - mins, 1.0)
    q = np.round((V - mins) / rng * (levels - 1))
    q = np.clip(q, 0, levels - 1)
    return q / (levels - 1) * rng + mins


@torch.no_grad()
def predict(vq, edges, pc, nb, spec, model, dev):
    """CurveModel forward on an explicit undirected edge list -> [E, 6] bins."""
    clear_cache(model)
    vqt = torch.as_tensor(np.asarray(vq), device=dev)[None]
    vvalid = torch.ones((1, len(vq)), dtype=torch.bool, device=dev)
    et = torch.as_tensor(np.asarray(edges), device=dev)[None]
    evalid = torch.ones((1, len(edges)), dtype=torch.bool, device=dev)
    vnorm = torch.as_tensor(spec.norm_xyz(spec.dequant_xyz(np.asarray(vq))),
                            dtype=torch.float32, device=dev)[None]
    logits = model(vqt, vvalid, pc, torch.tensor([int(nb)], device=dev), et,
                   evalid, vnorm)
    return logits[0].argmax(-1).cpu().numpy()


def curve_error(V, e, off_pred, off_gt, n=33):
    """Max chord-relative distance between predicted and GT cubic, plus chord."""
    P0, P1 = V[e[0]], V[e[1]]
    chord = float(np.linalg.norm(P1 - P0))
    if chord < 1e-9:
        return None, chord
    B1p, B2p = offsets_to_ctrl(P0, P1, off_pred)
    B1g, B2g = offsets_to_ctrl(P0, P1, off_gt)
    cp = bezier(P0, P1, B1p, B2p, n)
    cg = bezier(P0, P1, B1g, B2g, n)
    return float(np.linalg.norm(cp - cg, axis=1).max() / chord), chord


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=os.path.join(ROOT, "data",
                                                   "polytron_blocks_clean.pt"))
    ap.add_argument("--curve-ckpt", default=os.path.join(
        ROOT, "data", "polytron_clean", "polytron_curve_best.pt"))
    ap.add_argument("--split", default="val", choices=("train", "val"))
    ap.add_argument("--mode", default="gt", choices=("gt", "quadtron"))
    ap.add_argument("--quantization", type=int, default=256,
                    help="Quadtron quantization_levels for --mode quadtron")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n-points", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.load(args.data, weights_only=False)
    spec = PolytronSpec.from_json(data["spec"])
    model, mspec, ck = load_stage(args.curve_ckpt, dev)
    # The checkpoint's own spec governs the quantisation of its inputs.
    spec = mspec
    print(f"curve: epoch {ck['epoch']} val_loss {ck.get('val_loss'):.4f} "
          f"q_curve {spec.q_curve} mode {args.mode} device {dev}")

    items = data[args.split][args.start:args.start + args.n]
    rng = np.random.default_rng(args.seed)
    rows = []
    bin_err, exact, curve_rel, chords = [], [], [], []
    n_skipped = 0
    for it in items:
        pc = torch.as_tensor(build_cloud(it["surface_points"], it["point_labels"],
                                         args.n_points, spec, rng))[None].to(dev)
        V = spec.dequant_xyz(it["vq"])
        vq = it["vq"]
        if args.mode == "quadtron":
            V = quadtron_roundtrip(V, args.quantization)
            vq = spec.quant_xyz(V)
        pred = predict(vq, it["edges"], pc, it["n_blocks"], spec, model, dev)
        gt = np.asarray(it["cq"])
        e = np.asarray(it["edges"])
        d = np.abs(pred - gt)
        bin_err.append(d.mean())
        exact.append(float((d.max(1) == 0).mean()))
        off_p, off_g = spec.dequant_curve(pred), spec.dequant_curve(gt)
        for k in range(len(e)):
            ce, chord = curve_error(V, e[k], off_p[k], off_g[k])
            if ce is None:
                n_skipped += 1
                continue
            curve_rel.append(ce)
            chords.append(chord)
        rows.append({"name": it["name"], "blocks": it["n_blocks"],
                     "edges": int(len(e)), "mean_bin_err": float(d.mean()),
                     "exact_share": float((d.max(1) == 0).mean())})

    summary = {
        "mode": args.mode, "split": args.split, "n_items": len(items),
        "quantization": args.quantization if args.mode == "quadtron" else None,
        "n_edges": int(len(curve_rel)), "n_degenerate_skipped": int(n_skipped),
        "mean_bin_err": float(np.mean(bin_err)) if bin_err else None,
        "exact_share": float(np.mean(exact)) if exact else None,
        "curve_rel_median": float(np.median(curve_rel)) if curve_rel else None,
        "curve_rel_mean": float(np.mean(curve_rel)) if curve_rel else None,
        "curve_rel_p90": float(np.percentile(curve_rel, 90)) if curve_rel else None,
        "curve_rel_max": float(np.max(curve_rel)) if curve_rel else None,
    }
    print(json.dumps(summary, indent=1))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump({"summary": summary, "rows": rows},
                  open(args.out, "w"), indent=1)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
