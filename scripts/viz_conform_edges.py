"""viz_conform_edges.py -- classify the refill edge polylines and emit a
problem-analysis VTK for visual inspection.

Classification of each unique block-edge polyline (from refill_edges.vtk):
  part 1  problem: boundary edge NOT on the npz surface (chord that should
           have merged but deviates -> the visual inspection target)
  part 2  chord: non-boundary (interior O-grid) or planar edge that stays a
           straight chord -- expected, not a problem
  part 3  conformal + on a npz seam curve (seam-routed edge)
  part 4  conformal + NOT on a seam (surface-projected edge)

Context -- part 5: original npz seam curves (reference geometry).

Usage:
  uv run python scripts/viz_conform_edges.py \
      [--npz NPZ] [--edges REFILL_EDGES_VTK] [--refill REFILL_VTK] [--out VTK]
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

from geometry_features import FeatureModelV2  # noqa: E402
from scripts.compare_viz import _write_parts_vtk  # noqa: E402

DEFAULT_NPZ = os.path.join(ROOT, "data", "hex3d_algohex", "batch",
                           "machine_0034_n2000", "sample.npz")
DEFAULT_EDGES = os.path.join(
    ROOT, "data", "features_debug", "refined_h001_sp",
    "machine_0034_n2000_gt_conform_refill_edges.vtk")
DEFAULT_REFILL = os.path.join(
    ROOT, "data", "features_debug", "refined_h001_sp",
    "machine_0034_n2000_gt_conform_refill.vtk")


def _read_lines_vtk(path: str):
    """POINTSk + CELLS (polylines: count>=2) -> (points, list-of-index-lists)."""
    with open(path) as fh:
        lines = fh.read().splitlines()
    i, pts, cells = 0, None, []
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("POINTS "):
            n = int(s.split()[1])
            pts = np.array([[float(x) for x in lines[i + 1 + j].split()]
                            for j in range(n)])
            i += 1 + n
            continue
        if s.startswith("CELLS "):
            n = int(s.split()[1])
            j = i + 1
            for _ in range(n):
                p = [int(x) for x in lines[j].split()]
                if p[0] >= 2:
                    cells.append(np.array(p[1:], dtype=np.int64))
                j += 1
            i = j
            continue
        i += 1
    if pts is None:
        raise RuntimeError(f"VTK parse failed (no POINTS): {path}")
    return pts, cells


def _boundary_ids(refill_path: str) -> np.ndarray:
    """Unique vertex ids used by exactly one hex face (vectorized)."""
    seen = open(refill_path).read().splitlines()
    i = 0
    while i < len(seen):
        s = seen[i].strip().split()
        if s and s[0] == "CELLS":
            n = int(s[1])
            hexes = np.array([[int(x) for x in seen[i + 1 + k].split()[1:]]
                              for k in range(n)])
            break
        i += 1
    HEXF = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
            (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
    keys = np.sort(hexes[:, HEXF], axis=2).reshape(-1, 4)
    voidv = np.dtype((np.void, keys.dtype.itemsize * 4))
    v = np.ascontiguousarray(keys).view(voidv).ravel()
    uq, first, cnt = np.unique(v, return_index=True, return_counts=True)
    flat = first[cnt == 1]
    return np.unique(keys[flat].reshape(-1))


def _map_to_refill(epts: np.ndarray, refill_path: str):
    """Edges-VTK vertex index -> refill-VTK vertex index (KD-tree)."""
    with open(refill_path) as fh:
        for line in fh:
            if line.startswith("POINTS "):
                n = int(line.split()[1])
                pts = np.array([[float(x) for x in next(fh).split()]
                                for _ in range(n)])
                break
    from scipy.spatial import cKDTree
    d, idx = cKDTree(pts).query(epts, k=1)
    return idx, pts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default=DEFAULT_NPZ)
    ap.add_argument("--edges", default=DEFAULT_EDGES)
    ap.add_argument("--refill", default=DEFAULT_REFILL,
                    help="refill mesh VTK (for boundary vertex detection)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if not args.out:
        stem = os.path.splitext(os.path.basename(args.edges))[0]
        argpart = stem.replace("_gt_conform_refill_edges", "")
        args.out = os.path.join(os.path.dirname(args.edges),
                                f"{argpart}_gt_conform_edges_problems.vtk")

    fm = FeatureModelV2(args.npz, cache_dir=os.path.join(ROOT, "data", "features"))
    epts, polys = _read_lines_vtk(args.edges)
    print(f"polylines: {len(polys)}, points: {len(epts)}")

    bset_ids, _ = _map_to_refill(epts, args.refill)
    bset_uid = _boundary_ids(args.refill)
    mapped = [bset_ids[p] for p in polys]           # refill ids per polyline
    bset = set(bset_uid.tolist())
    print(f"boundary vertices of refill: {len(bset)}")
    print(f"boundary vertices of refill: {len(bset)}")

    parts = {1: [], 2: [], 3: [], 4: []}   # part -> list of index-lists
    xyz = []
    stats = {"problem": [], "chord_other": [], "seam": [], "projected": []}

    for pi, pl in enumerate(polys):
        q = epts[pl]
        # off-surface test on up to 41 samples
        step = max(1, len(q) // 41)
        samp = q[::step]
        d_surf, _, _ = fm.surface_nearest(samp, k=32)
        bfrac = float(np.mean([int(v) in bset for v in mapped[pi]]))
        if d_surf.max() > 1e-3:
            if bfrac >= 0.95:
                part, key = 1, "problem"
            else:
                part, key = 2, "chord_other"
        else:
            qn = q[::step]
            d_seam = fm.seam_curves.nearest(qn)
            ds = float(np.max(d_seam[0]))
            part, key = (3, "seam") if ds <= 1e-3 else (4, "projected")
        base_idx = [(len(xyz) + j) for j in range(len(q))]
        xyz.append(q)
        parts[part].append(base_idx)
        stats[key].append({"poly": pi, "max_off_surface": float(d_surf.max()),
                           "boundary_frac": bfrac, "n_pts": len(q)})

    xyz = np.vstack(xyz)
    title = ("meshtron conform edge analysis: 1=problem boundary chords, "
             "2=interior/planar chords, 3=seam edges, 4=surface-projected")
    _write_parts_vtk(args.out, [
        (xyz, parts[1], 1, 4),   # polylines (LINE)
        (xyz, parts[2], 2, 4),
        (xyz, parts[3], 3, 4),
        (xyz, parts[4], 4, 4),
    ], title)
    print(f"saved {args.out}")
    print("counts: problem=%d chord=%d seam=%d projected=%d"
          % (len(parts[1]), len(parts[2]), len(parts[3]), len(parts[4])))
    jpath = os.path.splitext(args.out)[0] + ".json"
    with open(jpath, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"saved {jpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
