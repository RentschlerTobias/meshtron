#!/usr/bin/env python3
"""Run the Polytron chain on dataset items and fill every result into a mesh.

    uv run python scripts/eval_polytron.py --split train --n 20 --k 4
    uv run python scripts/eval_polytron.py --split val --n 78 --which best

Per item: conditioning cloud from the npz surface alone, k rollouts of
vertices -> blocks -> curves, every well-formed rollout filled by curved TFI,
the best one kept (watertight, manifold, fewest inverted cells, best surface
fit). Against the ground truth it reports whether the vertex set and the block
topology came back EXACTLY -- the memorisation proof on the training split --
and, independent of the ground truth, whether the mesh is usable.

`--teacher vertices|blocks` feeds ground truth into the chain up to that
stage, so a failure can be attributed to the stage that made it.

Writes per item <name>_mesh.vtk (fine hex mesh, block_id), <name>_edges.vtk
(the curves the fill used) and <name>_structure.vtk (predicted AND ground
truth coarse blocks and curves in one file, scalar `source` 0=pred 1=gt), plus
summary.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from meshtron.data.polytron_blocks import (build_cloud, decode_seq, item_seq,  # noqa: E402
                                           load_npz, point_labels)
from meshtron.geometry.polytron_tfi import mesh_candidate, rank_key  # noqa: E402
from meshtron.training.polytron_sample import run_chain  # noqa: E402
from meshtron.training.train_polytron import load_stage  # noqa: E402

DATA = os.path.join(ROOT, "data", "hex3d_algohex")
HEX_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7))


def write_structure_vtk(path, parts, title):
    """parts: list of (V, blocks, curves, source). Hexes as straight VTK cells,
    curves as polylines; scalars source / kind (0 block, 1 edge) / block_id."""
    pts, cells, types, src, kind, bid = [], [], [], [], [], []
    for V, B, curves, s in parts:
        base = len(pts)
        pts.extend(np.asarray(V, float).tolist())
        for r, b in enumerate(B):
            cells.append([base + int(x) for x in b])
            types.append(12)
            src.append(s), kind.append(0), bid.append(r)
        for (a, b), Q in curves.items():
            o = len(pts)
            pts.extend(np.asarray(Q, float).tolist())
            cells.append(list(range(o, o + len(Q))))
            types.append(4)
            src.append(s), kind.append(1), bid.append(-1)
    with open(path, "w") as fh:
        fh.write(f"# vtk DataFile Version 2.0\n{title}\nASCII\n"
                 "DATASET UNSTRUCTURED_GRID\n")
        fh.write(f"POINTS {len(pts)} double\n")
        for p in pts:
            fh.write(f"{p[0]:.7f} {p[1]:.7f} {p[2]:.7f}\n")
        fh.write(f"CELLS {len(cells)} {sum(len(c) + 1 for c in cells)}\n")
        for c in cells:
            fh.write(f"{len(c)} " + " ".join(map(str, c)) + "\n")
        fh.write(f"CELL_TYPES {len(cells)}\n")
        for t in types:
            fh.write(f"{t}\n")
        fh.write(f"CELL_DATA {len(cells)}\n")
        for name, arr in (("source", src), ("kind", kind), ("block_id", bid)):
            fh.write(f"SCALARS {name} int 1\nLOOKUP_TABLE default\n")
            for v in arr:
                fh.write(f"{int(v)}\n")


def compare_to_gt(seq, gt, spec, close_bins: float = 2.0) -> dict:
    """Exactness against the ground truth, in the model's own quantisation."""
    pv = [tuple(v) for v in np.asarray(seq.vq).tolist()]
    gv = [tuple(v) for v in np.asarray(gt.vq).tolist()]
    out = {"vertices_exact": sorted(pv) == sorted(gv)}
    Vp, Vg = spec.dequant_xyz(seq.vq), spec.dequant_xyz(gt.vq)
    from scipy.spatial import cKDTree
    out["corner_chamfer"] = float(0.5 * (cKDTree(Vg).query(Vp)[0].mean()
                                         + cKDTree(Vp).query(Vg)[0].mean()))
    blocks_exact = False
    curve_bins = None
    if out["vertices_exact"] and len(pv) == len(gv):
        # identical sorted lists -> identical ids
        key = lambda B: {frozenset(b) for b in np.asarray(B).tolist()}  # noqa: E731
        blocks_exact = key(seq.blocks) == key(gt.blocks)
        if blocks_exact:
            cg = {tuple(e): c for e, c in zip(gt.edges.tolist(), gt.cq)}
            diff = [np.abs(np.asarray(c) - cg[tuple(e)])
                    for e, c in zip(seq.edges.tolist(), seq.cq) if tuple(e) in cg]
            curve_bins = float(np.mean(diff)) if diff else None
            out["curve_exact_share"] = float(np.mean([d.max() == 0 for d in diff]))
    out["blocks_exact"] = blocks_exact
    out["curve_mean_bin_error"] = curve_bins
    # Exactness up to a few bins: a corner one bin off is the same corner,
    # geometrically (the projection moves it more than that). Vertices are
    # matched one-to-one (Hungarian), then the block topology is compared
    # through that matching.
    out["vertices_close"] = False
    out["blocks_close"] = False
    if len(Vp) == len(Vg):
        from scipy.optimize import linear_sum_assignment
        step = (np.asarray(spec.hi) - np.asarray(spec.lo)) / (spec.q_vert - 1)
        D = np.abs(Vp[:, None] - Vg[None]) / step          # bins per axis
        cost = D.max(-1)
        r, c = linear_sum_assignment(cost)
        out["max_bin_error"] = float(cost[r, c].max())
        if cost[r, c].max() <= close_bins:
            out["vertices_close"] = True
            gid = np.empty(len(Vp), np.int64)
            gid[r] = c
            key = lambda B: {frozenset(b) for b in np.asarray(B).tolist()}  # noqa: E731
            out["blocks_close"] = key(gid[np.asarray(seq.blocks)]) == key(gt.blocks)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Polytron chain -> TFI evaluation")
    ap.add_argument("--prefix", default=os.path.join(ROOT, "data", "polytron_final"))
    ap.add_argument("--which", default="last", choices=("last", "best"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data",
                                                   "polytron_blocks_clean.pt"))
    ap.add_argument("--split", default="train", choices=("train", "val"))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--beam", type=int, default=8,
                    help="vertex beam width (0 = greedy + sampling)")
    ap.add_argument("--target-h", type=float, default=0.08)
    ap.add_argument("--teacher", default="none",
                    choices=("none", "vertices", "blocks"))
    ap.add_argument("--n-points", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "data", "polytron_eval"))
    ap.add_argument("--no-vtk", action="store_true")
    ap.add_argument("--oracle", action="store_true",
                    help="no models: fill the ground-truth tokens (the upper "
                         "bound any model can reach with this representation)")
    ap.add_argument("--no-project", action="store_true",
                    help="fill with the model's curves only, no mapping onto "
                         "the surface")
    for st in ("vertex", "block", "curve"):
        ap.add_argument(f"--{st}-ckpt", default=None,
                        help=f"override the {st} checkpoint of --prefix")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models, spec, specs = {}, None, {}
    data = torch.load(args.data, weights_only=False)
    if args.oracle:
        from meshtron.data.polytron_blocks import PolytronSpec
        spec = PolytronSpec.from_json(data["spec"])
        args.k = 1
    for st in () if args.oracle else ("vertex", "block", "curve"):
        path = getattr(args, f"{st}_ckpt") or f"{args.prefix}_{st}_{args.which}.pt"
        m, spec, ck = load_stage(path, dev)
        models[st] = m
        specs[st] = spec
        print(f"{st}: epoch {ck['epoch']} val {ck['val_loss']:.4f}")
    items = data[args.split][args.start:args.start + args.n]
    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    t_all = time.time()
    for it in items:
        t0 = time.time()
        raw = load_npz(os.path.join(DATA, it["dir"], "sample.npz"))
        lab = point_labels(len(raw["surface_points"]), raw["surface_tris"],
                           raw["surface_tri_label"])
        pc = {st: torch.as_tensor(build_cloud(
            raw["surface_points"], lab, args.n_points, sp,
            np.random.default_rng(args.seed)))[None].to(dev)
            for st, sp in specs.items()}
        gt = item_seq(it)
        cands = [gt] if args.oracle else run_chain(models, pc, it["n_blocks"], spec, k=args.k,
                          temp=args.temp, top_p=args.top_p, seed=args.seed, beam=args.beam,
                          gt_vertices=gt.vq if args.teacher != "none" else None,
                          gt_blocks=gt.blocks if args.teacher == "blocks" else None)
        results = []
        for ci, seq in enumerate(cands):
            r = {"cand": ci}
            if seq is None:
                results.append(r)
                continue
            tmp = os.path.join(args.out_dir, f"_{it['name']}_c{ci}.vtk")
            r.update(mesh_candidate(seq, spec, raw, args.target_h, tmp,
                                    project=not args.no_project))
            r["vs_gt"] = compare_to_gt(seq, gt, spec)
            r["_seq"] = seq
            r["_tmp"] = tmp
            results.append(r)
        ok = [r for r in results if "_seq" in r]
        best = min(ok, key=rank_key) if ok else None
        row = {"name": it["name"], "n_blocks": it["n_blocks"],
               "gt_vertices": int(len(gt.vq)), "k": args.k,
               "any_vertices_exact": any(r["vs_gt"]["vertices_exact"] for r in ok),
               "any_blocks_exact": any(r["vs_gt"]["blocks_exact"] for r in ok),
               "any_blocks_close": any(r["vs_gt"]["blocks_close"] for r in ok),
               "best_blocks_close": bool(best is not None and best["vs_gt"]["blocks_close"]),
               "n_tfi_ok": sum(r.get("tfi") is not None for r in ok)}
        if best is not None:
            row["best"] = {k: v for k, v in best.items() if not k.startswith("_")}
            if not args.no_vtk and best.get("tfi") is not None:
                stem = os.path.join(args.out_dir, it["name"])
                os.replace(best["_tmp"], stem + "_mesh.vtk")
                e = os.path.splitext(best["_tmp"])[0] + "_edges.vtk"
                if os.path.exists(e):
                    os.replace(e, stem + "_edges.vtk")
                Vg, Bg, Cg = decode_seq(gt, spec, n_edge_pts=32)
                Vp, Bp, Cp = decode_seq(best["_seq"], spec, n_edge_pts=32)
                write_structure_vtk(stem + "_structure.vtk",
                                    [(Vp, Bp, Cp, 0), (Vg, Bg, Cg, 1)],
                                    f"{it['name']} polytron pred (source 0) vs gt (1)")
        for r in results:
            for f in (r.get("_tmp"), os.path.splitext(r.get("_tmp", "x"))[0] + "_edges.vtk"):
                if f and os.path.exists(f) and os.path.basename(f).startswith("_"):
                    os.remove(f)
        rows.append(row)
        b = row.get("best", {})
        t = b.get("tfi") or {}
        s = b.get("surface") or {}
        print(f"{it['name']:40s} nb {it['n_blocks']:2d}  V-exact {row['any_vertices_exact']!s:5s} "
              f"B-exact {row['any_blocks_exact']!s:5s} B-close {row['any_blocks_close']!s:5s} tfi {row['n_tfi_ok']}/{args.k}  "
              f"wt {t.get('watertight')}  inv {t.get('inverted_curved')}/{t.get('cells_after')}  "
              f"on-surf {s.get('on_surface_median', float('nan')):.2e}  "
              f"uncov {s.get('uncovered_share', float('nan')):.3f}  {time.time() - t0:.1f}s",
              flush=True)

    def share(key):
        return float(np.mean([r[key] for r in rows])) if rows else 0.0

    good = [r for r in rows if (r.get("best") or {}).get("tfi")]
    summary = {
        "split": args.split, "n": len(rows), "k": args.k, "which": args.which,
        "teacher": "oracle" if args.oracle else args.teacher, "temp": args.temp, "target_h": args.target_h,
        "vertices_exact": share("any_vertices_exact"),
        "blocks_exact": share("any_blocks_exact"),
        "blocks_close_any": share("any_blocks_close"),
        "blocks_close_chosen": share("best_blocks_close"),
        "mesh_generated": len(good) / max(1, len(rows)),
        "watertight": float(np.mean([r["best"]["tfi"]["watertight"] for r in good])) if good else 0.0,
        "inverted_share_median": float(np.median([r["best"]["tfi"]["inverted_curved"]
                                                  / r["best"]["tfi"]["cells_after"]
                                                  for r in good])) if good else None,
        "on_surface_median": float(np.median([r["best"]["surface"]["on_surface_median"]
                                              for r in good])) if good else None,
        "uncovered_share_median": float(np.median([r["best"]["surface"]["uncovered_share"]
                                                   for r in good])) if good else None,
        "seconds": time.time() - t_all,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, indent=1, default=float)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
