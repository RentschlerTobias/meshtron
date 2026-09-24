"""compare_tfi_npz.py -- final VGL: TFI conform mesh vs original npz geometry.

User request (verbatim): final comparison -- conform pipeline (block structure,
mapping onto npz edges, TFI) at h=0.01, plus a comparison file between the TFI
h=0.01 mesh and the original npz geometry, to see how well the pipeline can
generate CFD meshes.

What it does:
  1. Loads the exported conform refill VTK (hex mesh produced by
     scripts/conform_gt_blocks.py) via its ASCII reader.
  2. Extracts the boundary quads of the hex mesh (faces shared by exactly one
     hex), vectorized.
  3. Computes per-boundary-point distances to
       a) the npz triangulated surface  (FeatureModelV2.surface_nearest)
       b) the npz seam curves           (CurveSet.nearest, vectorized)
  4. Writes ONE comparison VTK (UNSTRUCTURED_GRID, ASCII):
       part A: TFI boundary quads (type 9) with POINT_DATA scalars
               dist_npz_surface and dist_npz_edge
       part B: npz surface triangles (type 5)  [context, distance = 0]
       part C: npz seam polylines (type 4)     [context, distance = 0]
     plus a stats JSON (max/mean/percentiles/count-over-threshold for both
     distances).

Usage:
  uv run python scripts/compare_tfi_npz.py [--npz PATH] [--refill VTK]
      [--out VTK] [--k INT]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from meshtron.geometry.geometry_features import FeatureModelV2  # noqa: E402
from scripts.conform_gt_blocks import DEFAULT_NPZ, _read_vtk_mesh  # noqa: E402

# 6 quad faces of a VTK hex (local corner ids)
_HEX_FACES = np.array(
    [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
     [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]], dtype=np.int64)


def _boundary_quads(H: np.ndarray) -> np.ndarray:
    """Boundary quads of a (m, 8) hex connectivity, as sorted corner 4-tuples."""
    F = H[:, _HEX_FACES].reshape(-1, 4)          # (6m, 4) oriented faces
    K = np.sort(F, axis=1)
    void_dt = np.dtype((np.void, K.dtype.itemsize * 4))
    view = np.ascontiguousarray(K).view(void_dt).ravel()
    uniq, counts = np.unique(view, return_counts=True)
    U = uniq.view(K.dtype).reshape(-1, 4)
    return U[counts == 1]                        # faces owned by exactly 1 hex


def _dist_stats(d: np.ndarray) -> dict:
    return {
        "max": float(np.max(d)),
        "mean": float(np.mean(d)),
        "p50": float(np.percentile(d, 50)),
        "p90": float(np.percentile(d, 90)),
        "p99": float(np.percentile(d, 99)),
        "n_gt_0.01": int(np.sum(d > 0.01)),
        "n_gt_0.05": int(np.sum(d > 0.05)),
        "n": int(len(d)),
    }


def _write_vgl_vtk(path: str, title: str,
                   tfi_pts: np.ndarray, quads: np.ndarray,
                   d_surf: np.ndarray, d_edge: np.ndarray,
                   npz_pts: np.ndarray, npz_tris: np.ndarray,
                   seam_segments: list[np.ndarray]) -> None:
    """One ASCII VTK: TFI boundary + npz surface + npz seams, shared POINT_DATA."""
    n1 = len(tfi_pts)
    n2 = len(npz_pts)
    n3 = int(sum(len(s) for s in seam_segments))
    pts = np.vstack([tfi_pts, npz_pts] + seam_segments)

    # one record per line: "n id0 .. id{n-1}"; quad offset 0, tri offset n1,
    # seam polyline offsets after that
    off2 = n1
    off3 = n1 + n2
    q = np.column_stack([np.full(len(quads), 4, np.int64), quads])
    t = np.column_stack([np.full(len(npz_tris), 3, np.int64),
                         npz_tris + off2])
    types = [np.full(len(quads), 9, np.int64),
             np.full(len(npz_tris), 5, np.int64)]
    types += [np.array([4], np.int64) for _ in seam_segments]
    n_cells = len(quads) + len(npz_tris) + len(seam_segments)
    n_recs = int(q.size + t.size + sum(len(s) + 1 for s in seam_segments))
    types_arr = np.concatenate(types)

    with open(path, "w") as fh:
        fh.write("# vtk DataFile Version 3.0\n")
        fh.write(f"{title}\nASCII\nDATASET UNSTRUCTURED_GRID\n")
        fh.write(f"POINTS {len(pts)} double\n")
        np.savetxt(fh, pts, fmt="%.10g")
        fh.write(f"CELLS {n_cells} {n_recs}\n")
        np.savetxt(fh, q, fmt="%d")
        np.savetxt(fh, t, fmt="%d")
        for seg in seam_segments:
            ids = np.arange(off3, off3 + len(seg), dtype=np.int64)
            fh.write(str(len(seg)) + " "
                     + " ".join(map(str, ids.tolist())) + "\n")
            off3 += len(seg)
        fh.write(f"CELL_TYPES {len(types_arr)}\n")
        np.savetxt(fh, types_arr.reshape(-1, 1), fmt="%d")
        fh.write(f"POINT_DATA {len(pts)}\n")
        for name, arr in (("dist_npz_surface", d_surf),
                          ("dist_npz_edge", d_edge)):
            full = np.concatenate([arr, np.zeros(n2), np.zeros(n3)])
            fh.write(f"SCALARS {name} double 1\nLOOKUP_TABLE default\n")
            np.savetxt(fh, full, fmt="%.6g")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="VGL: TFI conform mesh boundary vs original npz geometry")
    ap.add_argument("--npz", default=DEFAULT_NPZ)
    ap.add_argument("--refill", default=None,
                    help="refill VTK from conform_gt_blocks (default: "
                         "data/features_debug/refined_h001/<stem>_gt_conform_refill.vtk)")
    ap.add_argument("--out", default=None,
                    help="output VGL VTK (default: <refill>_vgl.vtk / _vgl.json)")
    ap.add_argument("--k", type=int, default=32,
                    help="surface_nearest candidate triangles (default 32)")
    args = ap.parse_args()

    base = os.path.splitext(os.path.basename(args.npz))[0]
    stem = (os.path.basename(os.path.dirname(os.path.abspath(args.npz)))
            if base == "sample" else base)
    refill = args.refill or os.path.join(
        ROOT, "data", "features_debug", "refined_h001",
        f"{stem}_gt_conform_refill.vtk")
    out_vtk = args.out or refill.replace("_refill.vtk", "_refill_vgl.vtk")
    out_json = os.path.splitext(out_vtk)[0] + ".json"

    print(f"loading npz features: {args.npz}")
    fm = FeatureModelV2(args.npz, cache_dir=os.path.join(ROOT, "data", "features"))
    seam = fm.seam_curves

    print(f"parsing refill VTK (large, ASCII): {refill}")
    pts, H = _read_vtk_mesh(refill)
    print(f"  points={len(pts)} hexes={len(H)}")

    bnd = _boundary_quads(H)
    bpts_ids = np.unique(bnd)
    print(f"  boundary quads={len(bnd)} boundary points={len(bpts_ids)}")

    # remap quad corners into the compact boundary point list
    idx_of = np.full(len(pts), -1, np.int64)
    idx_of[bpts_ids] = np.arange(len(bpts_ids), dtype=np.int64)
    quads = idx_of[bnd]

    P = pts[bpts_ids]
    print("  distance to npz surface ...")
    d_surf, _, _ = fm.surface_nearest(P, k=args.k)
    print("  distance to npz seam curves ...")
    d_edge, _, _, _ = seam.nearest(P)

    print(f"writing VGL VTK: {out_vtk}")
    seam_segments = [seam.segment(ci) for ci in range(seam.n_curves)]
    _write_vgl_vtk(out_vtk,
                   f"meshtron {stem} TFI boundary vs npz "
                   f"(1=TFI quads, 2=npz surface, 3=npz seams)",
                   P, quads, np.asarray(d_surf), np.asarray(d_edge),
                   fm.surface_points, np.asarray(fm.surface_tris),
                   seam_segments)

    stats = {
        "refill_vtk": refill,
        "npz": args.npz,
        "n_hexes": int(len(H)),
        "n_boundary_quads": int(len(bnd)),
        "n_boundary_points": int(len(bpts_ids)),
        "dist_npz_surface": _dist_stats(np.asarray(d_surf)),
        "dist_npz_edge": _dist_stats(np.asarray(d_edge)),
    }
    with open(out_json, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"saved {out_vtk}")
    print(f"saved {out_json}")
    print(f"dist_npz_surface: max={stats['dist_npz_surface']['max']:.4g} "
          f"mean={stats['dist_npz_surface']['mean']:.4g} "
          f"p99={stats['dist_npz_surface']['p99']:.4g}")
    print(f"dist_npz_edge:    max={stats['dist_npz_edge']['max']:.4g} "
          f"mean={stats['dist_npz_edge']['mean']:.4g} "
          f"p99={stats['dist_npz_edge']['p99']:.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
