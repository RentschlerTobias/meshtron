#!/usr/bin/env python3
"""infer_gptcond_curve.py -- the full 3D chain: 3D block generator + curve head.

The 3D "Quadtron" is GPTCond + HexaRowTokenizer (hex blocks, linear edges).
This runs that generator, then curves the generated blocks with the Polytron
CurveModel, and fills both through the curved TFI. Three fills per rollout, so
the curve head's contribution is isolated:

  straight  edges kept straight          (what the generator emits today)
  model     CurveModel predicted curves  (the added edge stage)
  gt        ground-truth edge_ctrl       (the fill's upper bound)

Metrics per fill: watertight, inverted share, uncovered share, cells.

    export PYTHONPATH=…/domain_partition_3D
    uv run python scripts/infer_gptcond_curve.py --split val --n 10 --k 4
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

from meshtron.data.conditioning import build_cloud  # noqa: E402
from meshtron.data.hexa_row_tokenizer import HexaRowTokenizer  # noqa: E402
from meshtron.data.polytron_blocks import (PolySeq, build_cloud as poly_cloud,  # noqa: E402
                                           encode_sample, point_labels)
from meshtron.geometry.polytron_tfi import mesh_candidate, rank_key  # noqa: E402
from meshtron.training.generate import detokenize_safe, generate  # noqa: E402
from meshtron.training.polytron_sample import predict_curves  # noqa: E402
from meshtron.training.train_polytron import load_stage  # noqa: E402
from scripts.eval_family import load_model  # noqa: E402

DATA = os.path.join(ROOT, "data", "hex3d_algohex")


def load_npz(d: str) -> dict:
    with np.load(os.path.join(DATA, d, "sample.npz"), allow_pickle=True) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "data", "grpo_cart_step300.pt"))
    ap.add_argument("--tokens", default=os.path.join(ROOT, "data",
                                                     "hexarow_tokens_family_cart.pt"))
    ap.add_argument("--curve-ckpt", default=os.path.join(ROOT, "data", "polytron_clean",
                                                         "polytron_curve_best.pt"))
    ap.add_argument("--split", default="val", choices=("train", "val"))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--curve-n-points", type=int, default=2048)
    ap.add_argument("--target-h", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blade-weight", type=float, default=3.0)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "data", "infer_gptcond_curve"))
    ap.add_argument("--no-vtk", action="store_true")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32

    ck, cfg, coords, npt, rb, zb, gmodel, max_len, _ = load_model(args.ckpt, dev)
    gtok = HexaRowTokenizer(r_bounds=rb, z_bounds=zb)
    core = gtok.core
    specials = {core.start_token, core.end_token, core.sep_token, core.sep2_token,
                core.stop_token, core.pad_token}
    cmodel, spec, ck2 = load_stage(args.curve_ckpt, dev)
    print(f"generator: coords {coords} d {cfg['d']} L {cfg['layers']} | "
          f"curve head: epoch {ck2['epoch']} val {ck2.get('val_loss'):.4f} "
          f"q_curve {spec.q_curve}")

    ds = torch.load(args.tokens, weights_only=False)
    items = ds[args.split][args.start:args.start + args.n]
    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for it in items:
        raw = load_npz(it["dir"])
        rng = np.random.default_rng(args.seed)
        pc_gen, _ = build_cloud(it, cfg["n_points"], rb, zb, rng,
                                blade_weight=args.blade_weight)
        pc_gen = torch.as_tensor(pc_gen[None], dtype=torch.float32, device=dev)
        fc = torch.tensor([float(it["blocks"])], device=dev)
        lab = point_labels(len(raw["surface_points"]), raw["surface_tris"],
                           raw["surface_tri_label"])
        pc_curve = torch.as_tensor(np.array(poly_cloud(
            raw["surface_points"], lab, args.curve_n_points, spec,
            np.random.default_rng(args.seed)))[None], dtype=torch.float32, device=dev)

        cand = {"straight": [], "model": [], "gt": []}
        for i in range(max(1, args.k)):
            torch.manual_seed(args.seed + i)
            topk = 1 if i == 0 else 0
            seq = generate(gmodel, pc_gen, fc, core.start_token, core.stop_token,
                           core.sep_token, max_len - 1, args.temp, topk, dev, dtype,
                           specials, True, tok=gtok, constrained=True, coords=coords)
            res, _trim = detokenize_safe(seq, gtok, core.stop_token, coords=coords)
            if res is None:
                continue
            vpt, blk = res
            V = vpt.numpy() if coords == "cart" else np.stack(
                [vpt[:, 0].numpy() * np.cos(vpt[:, 1].numpy()),
                 vpt[:, 0].numpy() * np.sin(vpt[:, 1].numpy()), vpt[:, 2].numpy()], 1)
            B = blk.numpy()
            vq = spec.quant_xyz(V)
            edges, cq = predict_curves(cmodel, vq, B, pc_curve, len(B), spec)
            variants = {
                "straight": PolySeq(vq, B, edges, spec.quant_curve(np.zeros((len(edges), 6)))),
                "model": PolySeq(vq, B, edges, cq),
                "gt": encode_sample(raw["vertices"], raw["blocks"], raw["edges"],
                                    raw["edge_ctrl"], spec),
            }
            tmp = os.path.join(args.out_dir, f"_{it['name']}_c{i}_%s.vtk")
            for name, sq in variants.items():
                r = mesh_candidate(sq, spec, raw, args.target_h, tmp % name,
                                   project=True)
                r["rollout"] = i
                cand[name].append(r)
        row = {"name": it["name"], "blocks_gt": int(it["blocks"])}
        for name, rs in cand.items():
            ok = [r for r in rs if r.get("tfi")]
            if not ok:
                row[name] = None
                continue
            best = min(ok, key=rank_key)
            row[name] = {
                "blocks": best["structure"]["n_blocks"],
                "watertight": best["tfi"]["watertight"],
                "inverted": best["tfi"]["inverted_curved"],
                "cells": best["tfi"]["cells_after"],
                "inv_share": best["tfi"]["inverted_curved"] / max(1, best["tfi"]["cells_after"]),
                "uncovered": best["surface"]["uncovered_share"],
                "on_surface": best["surface"]["on_surface_median"],
            }
        rows.append(row)
        def fmt(name):
            v = row.get(name)
            return ("-" if v is None else
                    f"inv {v['inv_share']*100:.2f}% wt {v['watertight']} "
                    f"uncov {v['uncovered']:.3f} nb {v['blocks']}")
        print(f"{it['name']:40s} nbgt {it['blocks']:2d} | straight {fmt('straight')} "
              f"| model {fmt('model')} | gt {fmt('gt')}", flush=True)

    summary = {"split": args.split, "n": len(rows), "k": args.k, "target_h": args.target_h,
               "ckpt": os.path.basename(args.ckpt)}
    for name in ("straight", "model", "gt"):
        vals = [r[name] for r in rows if r.get(name)]
        summary[name] = {
            "items": len(vals),
            "watertight_share": float(np.mean([v["watertight"] for v in vals])) if vals else None,
            "inv_share_median": float(np.median([v["inv_share"] for v in vals])) if vals else None,
            "uncovered_median": float(np.median([v["uncovered"] for v in vals])) if vals else None,
        }
    print(json.dumps(summary, indent=1))
    json.dump({"summary": summary, "rows": rows},
              open(os.path.join(args.out_dir, "summary.json"), "w"), indent=1, default=float)
    print("wrote", os.path.join(args.out_dir, "summary.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
