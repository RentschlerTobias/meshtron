#!/usr/bin/env python3
"""Snap -> learned curves -> replace boundary edges that lie on a known
geometry edge (seam curve) by that exact seam segment, then curved TFI.

Compared on the SAME generated rollouts against snap -> learned curves only
(the best chain of reports/chain_order_snap_vs_curve.md). An edge is replaced only if
  * it is a boundary edge (on a block face owned by one block),
  * both corners snapped onto the SAME seam curve,
  * the seam route exists and is at most WALK x the chord long,
  * the route stays within DEV x chord of the learned curve (guards against
    an edge that cuts across a face between two points of one closed seam).

    PYTHONPATH=<dp3d hex3d_algohex> python seam_replace.py \
        --curve-ckpt <curve head checkpoint> --n 6 --k 2 --out-dir out
"""
import argparse
import json
import os
import sys
import time
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from infer_chain_order import DATA, HEX_FACES, load_npz  # noqa: E402
from meshtron.data.conditioning import build_cloud  # noqa: E402
from meshtron.data.hexa_row_tokenizer import HexaRowTokenizer  # noqa: E402
from meshtron.data.polytron_blocks import PolySeq, build_cloud as poly_cloud, decode_seq, point_labels  # noqa: E402
from meshtron.geometry.block_mapping import SnapConfigV2, snap_corners_v2  # noqa: E402
from meshtron.geometry.geometry_features import FeatureModelV2  # noqa: E402
from meshtron.geometry.polytron_tfi import read_hex_vtk, refill_polytron, surface_fit  # noqa: E402
from meshtron.training.generate import detokenize_safe, generate  # noqa: E402
from meshtron.training.polytron_sample import predict_curves  # noqa: E402
from meshtron.training.train_polytron import load_stage  # noqa: E402
from scripts.eval_family import load_model  # noqa: E402
from scripts.map_generated_blocks import _seam_path_fn  # noqa: E402

WALK, DEV = 1.5, 0.25


def boundary_edges(B):
    own = {}
    for blk in B:
        for f in HEX_FACES:
            own[frozenset(int(blk[i]) for i in f)] = own.get(frozenset(int(blk[i]) for i in f), 0) + 1
    out = set()
    for blk in B:
        for f in HEX_FACES:
            if own[frozenset(int(blk[i]) for i in f)] == 1:
                for k in range(4):
                    a, b = int(blk[f[k]]), int(blk[f[(k + 1) % 4]])
                    out.add((min(a, b), max(a, b)))
    return out


def fill(V, B, curves, raw, h, out_vtk):
    surf = (raw["surface_points"], raw["surface_tris"], raw["surface_tri_label"])
    rep = refill_polytron(V, B, curves, h, out_vtk, write_edges=True, surface=surf)
    P, _ = read_hex_vtk(out_vtk)
    s = surface_fit(P, rep["boundary_point_ids"], rep["boundary_quads"], raw["surface_points"], raw["surface_tris"])
    return {"inv_share": rep["inverted_curved"] / max(1, rep["cells_after"]), "watertight": rep["watertight"],
            "uncovered": s["uncovered_share"], "on_surface_p95": s["on_surface_p95"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "data", "grpo_cart_step300.pt"))
    ap.add_argument("--tokens", default=os.path.join(ROOT, "data", "hexarow_tokens_family_cart.pt"))
    ap.add_argument("--curve-ckpt", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--h", type=float, default=0.08)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    dev = "cuda"
    _, cfg, coords, _, rb, zb, gmodel, max_len, _ = load_model(a.ckpt, dev)
    gtok = HexaRowTokenizer(r_bounds=rb, z_bounds=zb); core = gtok.core
    specials = {core.start_token, core.end_token, core.sep_token, core.sep2_token, core.stop_token, core.pad_token}
    cmodel, spec, _ = load_stage(a.curve_ckpt, dev)
    items = torch.load(a.tokens, weights_only=False)["val"][a.start:a.start + a.n]
    rows = []
    for it in items:
        raw = load_npz(it["dir"])
        pc_gen, _ = build_cloud(it, cfg["n_points"], rb, zb, np.random.default_rng(0), blade_weight=3.0)
        pc_gen = torch.as_tensor(pc_gen[None], dtype=torch.float32, device=dev)
        fc = torch.tensor([float(it["blocks"])], device=dev)
        lab = point_labels(len(raw["surface_points"]), raw["surface_tris"], raw["surface_tri_label"])
        pc_c = torch.as_tensor(np.array(poly_cloud(raw["surface_points"], lab, 2048, spec,
                                                   np.random.default_rng(0)))[None], dtype=torch.float32, device=dev)
        fm = FeatureModelV2(os.path.join(DATA, it["dir"], "sample.npz"), cache_dir=os.path.join(ROOT, "data", "feature_cache"))
        target = types.SimpleNamespace(curves=fm.seam_curves, surface_nearest=fm.surface_nearest)
        for i in range(a.k):
            torch.manual_seed(i)
            seq = generate(gmodel, pc_gen, fc, core.start_token, core.stop_token, core.sep_token, max_len - 1,
                           0.7, 1 if i == 0 else 0, dev, torch.bfloat16, specials, True, tok=gtok,
                           constrained=True, coords=coords)
            res, _ = detokenize_safe(seq, gtok, core.stop_token, coords=coords)
            if res is None:
                continue
            V0, B = res[0].numpy().astype(float), res[1].numpy()
            C, rec = snap_corners_v2(target, V0[B], SnapConfigV2(tol_v=0.06, tol_e=0.04))
            V = V0.copy()
            for vid, c in zip(B.reshape(-1), C.reshape(-1, 3)):
                V[vid] = c
            vq = spec.quant_xyz(V)
            e, cq = predict_curves(cmodel, vq, B, pc_c, len(B), spec)
            Vd, Bd, curves = decode_seq(PolySeq(vq, B, e, cq), spec, n_edge_pts=64)
            row = {"name": it["name"], "rollout": i}
            try:
                row["snap_model"] = fill(Vd, Bd, curves, raw, a.h, os.path.join(a.out_dir, f"{it['name']}_{i}_snap_model.vtk"))
            except Exception as ex:  # noqa: BLE001
                row["snap_model"] = {"error": type(ex).__name__}
            # seam replacement
            recs = {}                                   # vertex id -> record (first occurrence)
            for vid, r in zip(B.reshape(-1), rec):
                recs.setdefault(int(vid), r)
            stats = {"routes": 0}
            path_fn = _seam_path_fn(fm.seam_curves, [recs[int(v)] for v in B.reshape(-1)], stats)
            cnt = {"boundary": 0, "same_seam": 0, "replaced": 0, "rej_long": 0, "rej_dev": 0}
            devs = []
            cs = fm.seam_curves

            def curves_at(v):
                """Seam curves through corner v: its record's curve, plus every
                curve with an endpoint at v (junction corners belong to all)."""
                r = recs.get(v)
                out = set()
                if r and r["curve_id"] >= 0:
                    out.add(int(r["curve_id"]))
                if len(cs.ep_pt):
                    d = np.linalg.norm(cs.ep_pt - Vd[v], axis=1)
                    out |= {int(c) for c in cs.ep_curve[d < 1e-3]}
                return out
            new = dict(curves)
            for (p, q) in sorted(boundary_edges(Bd)):
                cnt["boundary"] += 1
                if not (curves_at(p) & curves_at(q)):
                    continue
                cnt["same_seam"] += 1
                pr = path_fn(Vd[p], Vd[q], 64)
                if pr is None:
                    continue
                pts = np.asarray(pr[0] if isinstance(pr, tuple) else pr, float)
                chord = np.linalg.norm(Vd[q] - Vd[p])
                if np.linalg.norm(np.diff(pts, axis=0), axis=1).sum() > WALK * chord:
                    cnt["rej_long"] += 1; continue
                ref = curves.get((p, q))
                from scipy.spatial import cKDTree
                if ref is not None:
                    dev_ = max(cKDTree(ref).query(pts)[0].max(), cKDTree(pts).query(ref)[0].max())
                    if dev_ > DEV * chord:
                        cnt["rej_dev"] += 1; continue
                pts[0], pts[-1] = Vd[p], Vd[q]
                if ref is not None:
                    devs.append(float(cKDTree(pts).query(ref)[0].max() / chord))
                new[(p, q)] = pts; cnt["replaced"] += 1
            row["edges"] = cnt
            row["model_vs_seam_dev_rel"] = {"median": float(np.median(devs)) if devs else None,
                                            "max": float(np.max(devs)) if devs else None}
            try:
                row["snap_model_seam"] = fill(Vd, Bd, new, raw, a.h, os.path.join(a.out_dir, f"{it['name']}_{i}_snap_model_seam.vtk"))
            except Exception as ex:  # noqa: BLE001
                row["snap_model_seam"] = {"error": type(ex).__name__}
            rows.append(row)
            print(json.dumps(row), flush=True)
    json.dump(rows, open(os.path.join(a.out_dir, "rows.json"), "w"), indent=1)
    for v in ("snap_model", "snap_model_seam"):
        ok = [r[v] for r in rows if "uncovered" in r[v]]
        print(v, "n", len(ok), "uncovered median", np.median([x["uncovered"] for x in ok]) if ok else None,
              "inv median %", 100 * np.median([x["inv_share"] for x in ok]) if ok else None,
              "on-surface p95 median", np.median([x["on_surface_p95"] for x in ok]) if ok else None)


if __name__ == "__main__":
    main()
