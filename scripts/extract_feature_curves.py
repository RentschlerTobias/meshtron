"""extract_feature_curves.py — HANDOFF step (a): feature curves from GEOMETRY.

The previous failure mapped generated block edges onto AlgoHex block-complex
curves (npz `edge_polyline`) instead of design-system curves. This script
extracts the REAL geometry feature curves (label seams of the wall
triangulation) so they can be visually checked in ParaView before any snapping:

  --source msh : DTOO ground-truth walls, import-only chain
                 dp3d.extraction.parse_msh -> tet_prep_v2.outer_boundary
                 -> to_physical (7 physical surfaces) -> seam chaining.
  --source npz : per-machine reduced-domain surface triangulation the
                 transformer conditions on (sample.npz surface_* keys).

Output: legacy ASCII VTK polylines (cell type 4) with a `part` scalar (one part
per seam label pair; ParaView: Threshold on `part`), feature vertices as point
cells (type 1), and an optional `gt_block_edges` overlay (npz only) to verify
that the real AlgoHex block edges lie exactly on the geometry curves.
A JSON report with counts, arc lengths and per-vertex kink angles accompanies
each file: a straight smooth curve has kink ~0; junk curves will show up
immediately as huge kinks.

Usage:
  uv run python scripts/extract_feature_curves.py --source msh
  uv run python scripts/extract_feature_curves.py --source npz --overlay-gt-edges
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from curve_model import extract_seam_curves  # noqa: E402
from scripts.compare_viz import _write_parts_vtk  # noqa: E402

DP3D_REPO = Path("/home/t1dde/hydrostack_pipeline/stack/domain_partition_3D")
HEX3D_REPO = DP3D_REPO / "experimentell" / "hex3d_algohex"
DEFAULT_MSH = DP3D_REPO / "data" / "T1_9" / "T1_9_ru_gridGmsh.msh"
DEFAULT_NPZ = ROOT / "data" / "hex3d_algohex" / "batch" / "machine_0034_n2000" / "sample.npz"

# npz reduced-domain labels (tet_prep_v5 scheme, see DATASET_PIPELINE.md)
NPZ_SURF_NAMES = {1: "inlet", 2: "outlet", 3: "periodic_A", 4: "periodic_B",
                  5: "bl_iface_hub", 6: "bl_iface_shroud", 7: "ogrid_iface"}


def curves_from_msh(path: Path):
    """DTOO .msh -> outer skin -> 7 physical surfaces -> seam CurveSet."""
    sys.path.insert(0, str(HEX3D_REPO))  # tet_prep_v2 sets up dp3d on import
    import tet_prep_v2 as t2  # noqa: E402 (external repo, import-only)

    nodes, elements = t2.parse_msh(path)
    tris, gid = t2.outer_boundary(nodes, elements)
    ids = t2.to_physical(nodes, tris, gid)
    used = sorted({v for tri in tris for v in tri})
    tag2i = {v: i for i, v in enumerate(used)}
    pts = np.array([nodes[v] for v in used], float)
    tris_i = np.array([[tag2i[v] for v in tri] for tri in tris], np.int64)
    return extract_seam_curves(pts, tris_i, ids), dict(t2.tp.SURF_NAMES)


def curves_from_npz(path: Path):
    """sample.npz surface triangulation -> seam CurveSet (+ optional overlay)."""
    z = np.load(path)
    cs = extract_seam_curves(z["surface_points"], z["surface_tris"],
                             z["surface_tri_label"])
    return cs, dict(NPZ_SURF_NAMES)


def kink_degrees(Q: np.ndarray) -> np.ndarray:
    """Deviation from straight (deg) at each interior vertex of a polyline."""
    if len(Q) < 3:
        return np.zeros(0)
    u = Q[:-2] - Q[1:-1]
    w = Q[2:] - Q[1:-1]
    nu, nw = np.linalg.norm(u, axis=1), np.linalg.norm(w, axis=1)
    ok = (nu > 1e-12) & (nw > 1e-12)
    cos = np.clip((u[ok] * w[ok]).sum(1) / (nu[ok] * nw[ok]), -1.0, 1.0)
    return 180.0 - np.degrees(np.arccos(cos))


def part_groups(cs, names: dict) -> dict:
    """Curve index -> (part_id, part_name) keyed by sorted seam label pair."""
    pairs = sorted({(int(cs.label_lo[c]), int(cs.label_hi[c]))
                    for c in range(cs.n_curves)})
    pid = {pr: i + 1 for i, pr in enumerate(pairs)}
    label = {pr: "seam_{}_{}".format(names.get(pr[0], f"L{pr[0]}"),
                                     names.get(pr[1], f"L{pr[1]}"))
             for pr in pairs}
    return pid, label


def build(cs, names: dict, overlay=None, overlay_name="gt_block_edges"):
    """Assemble (parts, report) for the VTK writer from one or two CurveSets."""
    pid, label = part_groups(cs, names)
    by_part = defaultdict(list)  # part_id -> [polyline, ...]
    for c in range(cs.n_curves):
        Q = cs.segment(c)
        key = pid[(int(cs.label_lo[c]), int(cs.label_hi[c]))]
        by_part[key].append(Q)

    report = {}
    parts = []
    for key in sorted(by_part):
        pr = [pr for pr, k in pid.items() if k == key][0]
        segs = by_part[key]
        pts = np.vstack(segs)
        cells, off = [], [0]
        kinks, lens = [], []
        for Q in segs:
            cells.append(np.arange(len(Q)) + off[-1])
            off.append(off[-1] + len(Q))
            kinks.append(kink_degrees(Q))
            lens.append(float(np.linalg.norm(np.diff(Q, axis=0), axis=1).sum()))
        k = np.concatenate(kinks) if kinks else np.zeros(0)
        parts.append((pts, cells, key, 4))
        report[label[pr]] = {"part": key, "n_curves": len(segs),
                             "total_length": round(sum(lens), 4),
                             "kink_p95_deg": round(float(np.percentile(k, 95)), 2) if len(k) else 0.0,
                             "kink_max_deg": round(float(k.max()), 2) if len(k) else 0.0}

    if overlay is not None:
        key = max(pid.values(), default=0) + 1
        segs = [overlay.segment(c) for c in range(overlay.n_curves)]
        pts = np.vstack(segs)
        cells, off = [], [0]
        for Q in segs:
            cells.append(np.arange(len(Q)) + off[-1])
            off.append(off[-1] + len(Q))
        parts.append((pts, cells, key, 4))
        report[overlay_name] = {"part": key, "n_curves": len(segs)}

    # unique feature vertices (curve endpoints) as point cells, last part
    eps = {}
    for p in cs.ep_pt:
        eps[tuple(np.round(p, 9))] = p
    if eps:
        key = max(p[2] for p in parts) + 1
        V = np.array(list(eps.values()), float)
        parts.append((V, [[i] for i in range(len(V))], key, 1))
        report["feature_vertices"] = {"part": key, "n_points": len(V)}
    return parts, report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["msh", "npz"], required=True)
    ap.add_argument("--msh", type=Path, default=DEFAULT_MSH)
    ap.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    ap.add_argument("--overlay-gt-edges", action="store_true",
                    help="npz only: add AlgoHex block-edge polylines as one part")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    overlay = None
    if args.source == "msh":
        cs, names = curves_from_msh(args.msh)
        stem = args.msh.stem
    else:
        cs, names = curves_from_npz(args.npz)
        stem = args.npz.parent.name
        if args.overlay_gt_edges:
            from curve_model import block_edge_curves
            z = np.load(args.npz)
            overlay = block_edge_curves(z["edges"], z["edge_polyline"],
                                        z["edge_polyline_offset"])

    out = args.out or ROOT / "data" / "features_debug" / f"{stem}_feature_curves.vtk"
    out.parent.mkdir(parents=True, exist_ok=True)
    parts, report = build(cs, names, overlay)
    _write_parts_vtk(out, parts, f"feature curves ({args.source}) from {stem}")
    with out.with_suffix(".json").open("w") as fh:
        json.dump({"source": args.source, "file": str(args.msh if args.source == "msh" else args.npz),
                   "n_curves": int(cs.n_curves), "parts": report}, fh, indent=2)

    print(f"[extract] {args.source}: {cs.n_curves} curves -> {out}")
    for name, r in sorted(report.items(), key=lambda kv: kv[1].get("part", 0)):
        print(f"  part {r.get('part'):3d}  {name:28s}  {json.dumps(r)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
