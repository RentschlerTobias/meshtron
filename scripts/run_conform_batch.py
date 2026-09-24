"""run_conform_batch.py -- conform many samples, collect one numeric table.

Plan v4 S3/S4: the visual sign-off happens on a single sample; a batch run
answers whether the settings generalise.  Blockings with more than
--max-blocks blocks are skipped (same rule as scripts/clean_base_npz.py).

Writes <out>/batch_summary.json and prints a table sorted by boundary error.
VTK artifacts are written only for samples that fail a gate, so a full run
stays a few MB instead of a few GB.

Usage:
  uv run python scripts/run_conform_batch.py --samples a,b,c --out data/conform_batch
  uv run python scripts/run_conform_batch.py --all --out data/conform_batch
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from block_mapping import SnapConfigV2, snap_corners_v2  # noqa: E402
from curved_bridge import refill_curved  # noqa: E402
from geometry_features import FeatureModelV2  # noqa: E402
from patch_paths import (PatchPaths, blend_chord_edges,  # noqa: E402
                         make_boundary_face_test, make_face_projector,
                         max_kink_deg, snap_seam_path)
from scripts.conform_gt_blocks import (_boundary_edge_pred,  # noqa: E402
                                        collapsed_faces)
from scripts.map_generated_blocks import _seam_path_fn  # noqa: E402

BATCH = os.path.join(ROOT, "data", "hex3d_algohex", "batch")
BOUNDARY_TOL = 1e-3


def run_one(name: str, args) -> dict:
    npz = os.path.join(BATCH, name, "sample.npz")
    t0 = time.time()
    fm = FeatureModelV2(npz, cache_dir=args.feature_cache)
    target = types.SimpleNamespace(curves=fm.seam_curves,
                                   surface_nearest=fm.surface_nearest)
    C = fm.vertices[fm.blocks].astype(np.float64)
    nb = int(C.shape[0])
    C_snap, records = snap_corners_v2(target, C, SnapConfigV2())
    n_coll = collapsed_faces(fm.blocks, C_snap)
    if n_coll:
        return {"name": name, "blocks": nb, "skipped": True,
                "reason": f"{n_coll} collapsed block faces (degenerate block)",
                "gate_pass": None}
    stats = {"routes": 0, "edges_surface_projected": 0,
             "edges_walked_multi_patch": 0}
    _seam_raw = _seam_path_fn(fm.seam_curves, records, stats, tol=args.seam_tol)

    def seam_fn(p0, p1, n):
        res = _seam_raw(p0, p1, n)
        if res is None:
            return None
        return snap_seam_path(fm.seam_curves, fm, res[0]), res[1]
    geo = PatchPaths(fm, records=records, stats=stats,
                     is_boundary=_boundary_edge_pred(fm.blocks, C_snap),
                     clearance=args.blade_clearance,
                     clearance_chord_frac=args.clearance_chord_frac)

    def path_fn(p0, p1, n):
        res = seam_fn(p0, p1, n)
        return res if res is not None else geo(p0, p1, n)

    box: dict = {}

    def edge_post(st, corner_ids, C_s, blks):
        box["st"] = st
        blend_chord_edges(st, corner_ids, C_s, blks, stats=stats)

    out_vtk = os.path.join(args.out, f"{name}_refill.vtk")
    rep = refill_curved(
        C_snap, args.target_h, out_vtk, fm=target, path_fn=path_fn,
        write_edges=False,
        edge_post_fn=edge_post,
        face_project_fn=(make_face_projector(geo, stats)
                         if args.project_faces else None),
        is_boundary_face=make_boundary_face_test(fm))
    rejected = int(rep.get("single_owner_faces_rejected", 0))
    bids = rep.pop("boundary_point_ids")
    rep.pop("boundary_quads", None)
    pts = _read_points(out_vtk)
    d, _, _ = fm.surface_nearest(pts[bids], k=32)
    st = box.get("st")
    kinks = ([max_kink_deg(np.asarray(q, float))
              for q in st.edge_pts.values()] if st is not None else [0.0])
    row = {
        "name": name, "blocks": nb, "cells": int(rep["cells_after"]),
        "watertight": bool(rep["watertight"]),
        "inverted": int(rep["inverted_curved"]),
        "min_scaled_jacobian": float(rep["min_scaled_jacobian"]),
        "boundary_max": float(d.max()), "boundary_p99": float(np.percentile(d, 99)),
        "boundary_p50": float(np.percentile(d, 50)),
        "chord_edges": int(rep["buckets"]["edges_degenerate_chord"]),
        "seam_routes": int(stats["routes"]),
        "geodesic": int(stats.get("edges_geodesic", 0)),
        "blended": int(stats.get("edges_blended", 0)),
        "faces_projected": int(stats.get("faces_projected", 0)),
        "single_owner_faces_rejected": rejected,
        "max_kink_deg": float(max(kinks)),
        "kink_edges_over_20deg": int(sum(1 for k in kinks if k > 20.0)),
        "seconds": round(time.time() - t0, 1),
    }
    row["gate_pass"] = bool(row["boundary_max"] <= BOUNDARY_TOL
                            and row["chord_edges"] == 0 and row["watertight"])
    if not row["gate_pass"] and args.keep_failing_vtk:
        row["vtk"] = out_vtk
    else:
        os.remove(out_vtk)
    return row


def _read_points(path: str) -> np.ndarray:
    with open(path) as fh:
        lines = fh.read().split("\n")
    i = next(k for k, l in enumerate(lines) if l.startswith("POINTS"))
    n = int(lines[i].split()[1])
    return np.array([[float(x) for x in lines[i + 1 + k].split()]
                     for k in range(n)])


def main() -> int:
    ap = argparse.ArgumentParser(description="batch conform over samples")
    ap.add_argument("--samples", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--max-blocks", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "conform_batch"))
    ap.add_argument("--target-h", type=float, default=0.05)
    ap.add_argument("--blade-clearance", type=float, default=0.06)
    ap.add_argument("--clearance-chord-frac", type=float, default=0.25)
    ap.add_argument("--seam-tol", type=float, default=1e-9)
    ap.add_argument("--project-faces", action="store_true", default=True)
    ap.add_argument("--keep-failing-vtk", action="store_true")
    ap.add_argument("--feature-cache",
                    default=os.path.join(ROOT, "data", "features"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.all:
        names = []
        for d in sorted(os.listdir(BATCH)):
            p = os.path.join(BATCH, d, "sample.npz")
            if not os.path.exists(p):
                continue
            try:
                nb = int(np.load(p, allow_pickle=True)["blocks"].shape[0])
            except Exception:
                continue
            if nb <= args.max_blocks:
                names.append(d)
    else:
        names = [s for s in args.samples.split(",") if s]

    rows, failed, skipped = [], [], []
    for k, name in enumerate(names, 1):
        try:
            row = run_one(name, args)
        except Exception as exc:  # noqa: BLE001
            failed.append({"name": name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{k}/{len(names)}] {name}: ERROR {type(exc).__name__}: {exc}")
            continue
        if row.get("skipped"):
            skipped.append(row)
            print(f"[{k}/{len(names)}] {name:24s} SKIP {row['reason']}")
            continue
        rows.append(row)
        print(f"[{k}/{len(names)}] {name:24s} blocks={row['blocks']:3d} "
              f"bnd_max={row['boundary_max']:.2e} chords={row['chord_edges']} "
              f"inv={row['inverted']:5d} minSJ={row['min_scaled_jacobian']:7.3f} "
              f"{'PASS' if row['gate_pass'] else 'FAIL'}")

    rows.sort(key=lambda r: -r["boundary_max"])
    summary = {
        "n": len(rows), "failed": failed, "skipped": skipped,
        "gate_pass": int(sum(1 for r in rows if r["gate_pass"])),
        "settings": {"target_h": args.target_h, "seam_tol": args.seam_tol,
                     "blade_clearance": args.blade_clearance,
                     "clearance_chord_frac": args.clearance_chord_frac,
                     "project_faces": bool(args.project_faces),
                     "boundary_tol": BOUNDARY_TOL},
        "rows": rows,
    }
    path = os.path.join(args.out, "batch_summary.json")
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nsaved {path}")
    print(f"gate pass {summary['gate_pass']}/{len(rows)}, "
          f"skipped {len(skipped)}, errors {len(failed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
