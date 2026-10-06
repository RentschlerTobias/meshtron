#!/usr/bin/env python3
"""Which order of snap and learned curves gives the best mesh?

Extends meshtron's scripts/infer_gptcond_curve.py (same generator, curve head,
fill and metrics) with variants that differ only in what happens between the
generated linear blocks and the curved TFI, all on the SAME rollouts:

  straight        no snap, straight edges
  model           no snap, learned curves                  (report baseline)
  snap_all_model  snap_corners_v2 on ALL corners, then learned curves
  snap_bnd_model  snap only corners on boundary faces (face owned by one
                  block), interior corners untouched, then learned curves
  snap_feat_model snap only corners near a geometry feature (curve endpoint or
                  seam curve); corners that would only be projected onto a
                  surface stay put, then learned curves
  backmap         snap all corners + seam routing, no learned curves (report)
  gt              GT blocks + GT edge_ctrl (ceiling)

snap_corners_v2 projects every corner that is near neither a curve endpoint
nor a seam onto the nearest surface point -- interior corners included.
snap_bnd_model is the "snap boundary only" step of the intended chain.

    PYTHONPATH=<dp3d hex3d_algohex> python infer_chain_order.py \
        --curve-ckpt <curve head checkpoint> --n 20 --k 3 --out-dir out

Results: reports/chain_order_snap_vs_curve.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from meshtron.data.conditioning import build_cloud  # noqa: E402
from meshtron.data.hexa_row_tokenizer import HexaRowTokenizer  # noqa: E402
from meshtron.data.polytron_blocks import (PolySeq, build_cloud as poly_cloud,  # noqa: E402
                                           encode_sample, point_labels)
from meshtron.geometry.block_mapping import SnapConfigV2, snap_corners_v2  # noqa: E402
from meshtron.geometry.geometry_features import FeatureModelV2  # noqa: E402
from meshtron.geometry.polytron_tfi import mesh_candidate, rank_key  # noqa: E402
from meshtron.training.generate import detokenize_safe, generate  # noqa: E402
from meshtron.training.polytron_sample import predict_curves  # noqa: E402
from meshtron.training.train_polytron import load_stage  # noqa: E402
from scripts.eval_family import load_model  # noqa: E402
from scripts.infer_gptcond_curve import DATA, backmap_fill, load_npz  # noqa: E402

HEX_FACES = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
VARIANTS = ("straight", "model", "snap_all_model", "snap_bnd_model", "snap_feat_model",
            "backmap", "gt")


def boundary_vertices(B: np.ndarray) -> np.ndarray:
    """Vertex ids on faces owned by exactly one block."""
    own = {}
    for blk in B:
        for f in HEX_FACES:
            key = frozenset(int(blk[i]) for i in f)
            own[key] = own.get(key, 0) + 1
    return np.array(sorted({v for f, n in own.items() if n == 1 for v in f}), dtype=np.int64)


def snapped_vertices(V, B, fm, only, tiers=("vertex", "edge", "surface")):
    """Shared-vertex snap: corners -> snap_corners_v2 -> back to vertex array.
    only: vertex ids allowed to move (None = all); tiers: snap tiers applied
    ("vertex" = curve endpoint, "edge" = seam curve, "surface" = nearest patch)."""
    target = type("T", (), {})()
    target.curves, target.surface_nearest = fm.seam_curves, fm.surface_nearest
    C, rec = snap_corners_v2(target, V[B], SnapConfigV2(tol_v=0.06, tol_e=0.04))
    V2 = V.copy()
    flat_ids, flat_c = B.reshape(-1), C.reshape(-1, 3)
    mask = np.ones(len(V), bool) if only is None else np.isin(np.arange(len(V)), only)
    for vid, c, r in zip(flat_ids, flat_c, rec):
        if mask[vid] and r["tier"] in tiers:
            V2[vid] = c
    moved = np.linalg.norm(V2 - V, axis=1)
    return V2, {"moved_n": int((moved > 1e-12).sum()), "moved_max": float(moved.max()),
                "moved_mean": float(moved[moved > 1e-12].mean()) if (moved > 1e-12).any() else 0.0}


def summarize(rs):
    ok = [r for r in rs if r.get("tfi")]
    if not ok:
        return None
    b = min(ok, key=rank_key)
    return {"blocks": b["structure"]["n_blocks"], "watertight": b["tfi"]["watertight"],
            "inv_share": b["tfi"]["inverted_curved"] / max(1, b["tfi"]["cells_after"]),
            "uncovered": b["surface"]["uncovered_share"],
            "on_surface": b["surface"]["on_surface_median"], "rollout": b["rollout"],
            **({"snap": b["snap"]} if "snap" in b else {})}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "data", "grpo_cart_step300.pt"))
    ap.add_argument("--tokens", default=os.path.join(ROOT, "data", "hexarow_tokens_family_cart.pt"))
    ap.add_argument("--curve-ckpt", required=True)
    ap.add_argument("--split", default="val", choices=("train", "val"))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--curve-n-points", type=int, default=2048)
    ap.add_argument("--target-h", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blade-weight", type=float, default=3.0)
    ap.add_argument("--feature-cache", default=os.path.join(ROOT, "data", "feature_cache"))
    ap.add_argument("--no-project", action="store_true")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    _ck, cfg, coords, _npt, rb, zb, gmodel, max_len, _ = load_model(a.ckpt, dev)
    gtok = HexaRowTokenizer(r_bounds=rb, z_bounds=zb); core = gtok.core
    specials = {core.start_token, core.end_token, core.sep_token, core.sep2_token,
                core.stop_token, core.pad_token}
    cmodel, spec, ck2 = load_stage(a.curve_ckpt, dev)
    print(f"curve head: epoch {ck2['epoch']} val {ck2.get('val_loss'):.4f}", flush=True)
    os.makedirs(a.out_dir, exist_ok=True); os.makedirs(a.feature_cache, exist_ok=True)

    items = torch.load(a.tokens, weights_only=False)[a.split][a.start:a.start + a.n]
    rows = []
    for it in items:
        raw = load_npz(it["dir"])
        rng = np.random.default_rng(a.seed)
        pc_gen, _ = build_cloud(it, cfg["n_points"], rb, zb, rng, blade_weight=a.blade_weight)
        pc_gen = torch.as_tensor(pc_gen[None], dtype=torch.float32, device=dev)
        fc = torch.tensor([float(it["blocks"])], device=dev)
        lab = point_labels(len(raw["surface_points"]), raw["surface_tris"], raw["surface_tri_label"])
        pc_curve = torch.as_tensor(np.array(poly_cloud(raw["surface_points"], lab, a.curve_n_points,
                                   spec, np.random.default_rng(a.seed)))[None],
                                   dtype=torch.float32, device=dev)
        fm = FeatureModelV2(os.path.join(DATA, it["dir"], "sample.npz"), cache_dir=a.feature_cache)
        cand = {v: [] for v in VARIANTS}
        for i in range(max(1, a.k)):
            torch.manual_seed(a.seed + i)
            seq = generate(gmodel, pc_gen, fc, core.start_token, core.stop_token, core.sep_token,
                           max_len - 1, a.temp, 1 if i == 0 else 0, dev, dtype, specials, True,
                           tok=gtok, constrained=True, coords=coords)
            res, _ = detokenize_safe(seq, gtok, core.stop_token, coords=coords)
            if res is None:
                continue
            vpt, blk = res
            V = vpt.numpy().astype(np.float64) if coords == "cart" else np.stack(
                [vpt[:, 0].numpy() * np.cos(vpt[:, 1].numpy()),
                 vpt[:, 0].numpy() * np.sin(vpt[:, 1].numpy()), vpt[:, 2].numpy()], 1)
            B = blk.numpy()
            tmp = os.path.join(a.out_dir, f"_{it['name']}_c{i}_%s.vtk")

            def curved(Vx):
                vq = spec.quant_xyz(Vx)
                e, cq = predict_curves(cmodel, vq, B, pc_curve, len(B), spec)
                return vq, e, cq
            vq, edges, cq = curved(V)
            seqs = {"straight": PolySeq(vq, B, edges, spec.quant_curve(np.zeros((len(edges), 6)))),
                    "model": PolySeq(vq, B, edges, cq),
                    "gt": encode_sample(raw["vertices"], raw["blocks"], raw["edges"],
                                        raw["edge_ctrl"], spec)}
            snapinfo = {}
            for name, only, tiers in (("snap_all_model", None, ("vertex", "edge", "surface")),
                                      ("snap_bnd_model", boundary_vertices(B), ("vertex", "edge", "surface")),
                                      ("snap_feat_model", None, ("vertex", "edge"))):
                V2, info = snapped_vertices(V, B, fm, only, tiers)
                vq2, e2, cq2 = curved(V2)
                seqs[name] = PolySeq(vq2, B, e2, cq2); snapinfo[name] = info
            for name, sq in seqs.items():
                try:
                    r = mesh_candidate(sq, spec, raw, a.target_h, tmp % name, project=not a.no_project)
                except Exception as e:  # noqa: BLE001
                    r = {"tfi": None, "error": f"{type(e).__name__}: {e}"}
                r["rollout"] = i
                if name in snapinfo:
                    r["snap"] = snapinfo[name]
                cand[name].append(r)
            r = backmap_fill(V, B, raw, fm, a.target_h, tmp % "backmap", spec)
            r["rollout"] = i; cand["backmap"].append(r)
        row = {"name": it["name"], "blocks_gt": int(it["blocks"])}
        row.update({v: summarize(cand[v]) for v in VARIANTS})
        rows.append(row)
        print(f"{it['name']:34s} " + " | ".join(
            f"{v} " + ("-" if row[v] is None else
                       f"inv {row[v]['inv_share']*100:.2f}% unc {row[v]['uncovered']:.3f}")
            for v in VARIANTS), flush=True)

    summary = {"n": len(rows), "k": a.k, "project": not a.no_project}
    for v in VARIANTS:
        vals = [r[v] for r in rows if r.get(v)]
        summary[v] = {"items": len(vals),
                      "watertight": float(np.mean([x["watertight"] for x in vals])) if vals else None,
                      "inv_median_pct": 100 * float(np.median([x["inv_share"] for x in vals])) if vals else None,
                      "inv_mean_pct": 100 * float(np.mean([x["inv_share"] for x in vals])) if vals else None,
                      "uncovered_median": float(np.median([x["uncovered"] for x in vals])) if vals else None}
    print(json.dumps(summary, indent=1))
    json.dump({"summary": summary, "rows": rows}, open(os.path.join(a.out_dir, "summary.json"), "w"),
              indent=1, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
