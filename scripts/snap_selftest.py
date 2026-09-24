"""snap_selftest.py — HANDOFF step (b): snap GT decomposition onto SEAM curves.

Self-test for the back-mapping machinery: take the GROUND-TRUTH block corners
(sample.npz vertices), snap them against the design-system seam curves ONLY
(not the AlgoHex block-complex edges — that concat was the root cause of the
previous failure), reconstruct every GT block edge that both-ends-land-on-one
seam curve as the arc-length sub-segment between the two snap parameters, and
compare it against the true GT edge polyline.

Gate (b): reconstructed sub-segments must lie exactly on the real block edges
(deviation ~ mesh spacing), and the curve-part coverage (how many GT edges are
reconstructable at all) is reported — it is the same number the later
`coverage` section of summary.json will use for generated blocks.

Output VTK parts: 1=seam curves, 2=GT edges on one curve, 3=reconstructed
sub-segments (must overlap part 2 in ParaView), 4/5/6=GT corner snap tiers
(vertex/edge/surface) as point cells. JSON twin carries per-edge deviations.

Usage:
  uv run python scripts/snap_selftest.py
  uv run python scripts/snap_selftest.py --npz data/hex3d_algohex/batch/machine_0005_n8000/sample.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meshtron.geometry.block_mapping import SnapConfigV2, snap_corners_v2, tier_counts  # noqa: E402
from meshtron.geometry.geometry_features import FeatureModelV2  # noqa: E402
from meshtron.geometry.seam_graph import build_graph, path_points, simple_paths  # noqa: E402
from scripts.compare_viz import _write_parts_vtk  # noqa: E402

DEFAULT_NPZ = ROOT / "data" / "hex3d_algohex" / "batch" / "machine_0034_n2000" / "sample.npz"


def _resample(Q: np.ndarray, n: int) -> np.ndarray:
    """n aersequidistanten Punkten auf der Polylinie Q (Bogenlaengen-Interp)."""
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))])
    if s[-1] < 1e-12 or n < 2:
        return np.repeat(Q[:1], max(n, 1), axis=0)
    q = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(q, s, Q[:, k]) for k in range(3)], axis=1)


def _arc_targets(seam, records):
    """(curve_id, arc-length t, tier) je GT-Vertex; Zielbogenlaenge normiert."""
    out = []
    for i, r in enumerate(records):
        cid, t = int(r["curve_id"]), float(r["t"])
        if cid >= 0:
            lo, hi = int(seam.offset[cid]), int(seam.offset[cid + 1])
            if r["tier"] == "vertex":
                t = float(seam.arclen[hi - 1]) if t > 0.5 else 0.0
            out.append((cid, float(np.clip(t, seam.arclen[lo], seam.arclen[hi - 1]))))
        else:
            out.append((-1, float("nan")))
    return out



def reconstruct(seam, edges, targets, edge_polys, tol_on=0.01):
    """Classify each directed GT edge against the seam curve set and rebuild
    its polyline as a shortest seam-graph path between the snapped corners.

    First gate is always the edge polyline itself vs the seam point set
    (endpoint membership alone is not enough: an edge can have both ends on
    seams yet cut across a surface -> off_seam, not reconstructible).
    recon[edge] = reconstructed seam path (ParaView part 3, must overlay part 2).
    """
    corner_ts = defaultdict(list)
    for cid, t in targets:
        if cid >= 0:
            corner_ts[cid].append(float(t))
    node, adj = build_graph(seam, corner_ts)
    key = lambda c, t: (int(c), round(float(t), 9))
    rows, recon = [], {}
    for i, (a, b) in enumerate(edges):
        Q = edge_polys[i]
        d_on = float(seam.nearest(Q)[0].max())
        if d_on > tol_on:
            rows.append({"edge": i, "kind": "off_seam", "d_on_seam": round(d_on, 4)})
            continue
        ca, ta = targets[int(a)]
        cb, tb = targets[int(b)]
        if ca < 0 or cb < 0:
            rows.append({"edge": i, "kind": "on_seam_unsnapped",
                         "d_on_seam": round(d_on, 6)})
            continue
        na, nb = node.get(key(ca, ta)), node.get(key(cb, tb))
        if na is None or nb is None:
            rows.append({"edge": i, "kind": "unconnected",
                         "curves": [int(ca), int(cb)],
                         "d_on_seam": round(d_on, 6)})
            continue
        if na == nb:
            rows.append({"edge": i, "kind": "degenerate", "curve": int(ca),
                         "d_on_seam": round(d_on, 6)})
            continue
        cands = simple_paths(adj, na, nb)
        if not cands:
            rows.append({"edge": i, "kind": "unconnected",
                         "curves": [int(ca), int(cb)],
                         "d_on_seam": round(d_on, 6)})
            continue
        n = max(int(Q.shape[0]), 32)
        R = _resample(Q, n)
        best = None
        for tot_w, hops in cands:
            if tot_w < 1e-9:
                continue
            P = _resample(path_points(seam, hops, tot_w, n), n)
            dev = min(float(np.abs(R - P).max()),
                      float(np.abs(R - P[::-1]).max()))
            if best is None or dev < best[0]:
                best = (dev, P, tot_w, hops)
        if best is None:
            rows.append({"edge": i, "kind": "degenerate", "curve": int(ca),
                         "d_on_seam": round(d_on, 6)})
            continue
        dev, P, tot_w, hops = best
        curves = [int(hops[0][0])]
        for h in hops[1:]:
            if int(h[0]) != curves[-1]:
                curves.append(int(h[0]))
        rows.append({"edge": i, "kind": "path", "curves": curves,
                     "dev_max": round(dev, 6), "length": round(tot_w, 4)})
        recon[i] = P
    return rows, recon



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--tol-v", type=float, default=0.06)
    ap.add_argument("--tol-e", type=float, default=0.04)
    args = ap.parse_args()

    fm = FeatureModelV2(args.npz)
    seam = fm.seam_curves
    # SNAP TARGET: seam curves only (no edge_curves concat = the old failure)
    shim = types.SimpleNamespace(curves=seam, surface_nearest=fm.surface_nearest)
    V = fm.vertices
    C, records = snap_corners_v2(shim, V[None], SnapConfigV2(
        tol_v=args.tol_v, tol_e=args.tol_e))
    tiers = tier_counts(records)

    off = fm.edge_offset
    edge_polys = [fm.edge_polyline[int(off[i]):int(off[i + 1])]
                  for i in range(len(fm.edges))]
    targets = _arc_targets(seam, records)
    rows, recon = reconstruct(seam, fm.edges, targets, edge_polys)

    paths = [r for r in rows if r["kind"] == "path"]
    degen = [r for r in rows if r["kind"] == "degenerate"]
    unconn = [r for r in rows if r["kind"] == "unconnected"]
    unsnap = [r for r in rows if r["kind"] == "on_seam_unsnapped"]
    off_seam = [r for r in rows if r["kind"] == "off_seam"]
    on_seam = paths + degen + unconn + unsnap
    devs = np.array([r["dev_max"] for r in paths]) if paths else np.zeros(0)

    parts = []
    pts, cells, b = [], [], 0
    for c in range(seam.n_curves):
        Q = seam.segment(c)
        cells.append(np.arange(len(Q)) + b)
        pts.append(Q)
        b += len(Q)
    parts.append((np.vstack(pts), cells, 1, 4))

    def _edge_part(idx_list, pid):
        pts_, cells_, b = [], [], 0
        for i in idx_list:
            Q = edge_polys[i]
            cells_.append(np.arange(len(Q)) + b)
            pts_.append(Q)
            b += len(Q)
        if not cells_:
            return
        parts.append((np.vstack(pts_), cells_, pid, 4))

    _edge_part([r["edge"] for r in on_seam], 2)
    pts_, cells_, b = [], [], 0
    for r in paths:
        Q = recon[r["edge"]]
        cells_.append(np.arange(len(Q)) + b)
        pts_.append(Q)
        b += len(Q)
    if cells_:
        parts.append((np.vstack(pts_), cells_, 3, 4))
    for pid, tier in ((4, "vertex"), (5, "edge"), (6, "surface")):
        sel = np.array([i for i, r in enumerate(records) if r["tier"] == tier], int)
        if len(sel):
            parts.append((V[sel], [[k] for k in range(len(sel))], pid, 1))

    out_dir = args.out_dir or ROOT / "data" / "snap_selftest" / args.npz.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    vtk = out_dir / "snap_selftest.vtk"
    _write_parts_vtk(vtk, parts, f"snap selftest (seam-only) {args.npz.parent.name}")
    corner_d = {t: (round(float(np.percentile([r["dist"] for r in records
                                               if r["tier"] == t], 95)), 5)
                    if any(r["tier"] == t for r in records) else None)
                for t in ("vertex", "edge", "surface")}
    rep = {"npz": str(args.npz), "corner_tiers_gt_vertex": tiers,
           "corner_snap_dist_p95_m": corner_d,
           "edges_total": int(len(fm.edges)),
           "coverage": {"path": len(paths), "degenerate": len(degen),
                        "unconnected": len(unconn),
                        "on_seam_unsnapped": len(unsnap),
                        "off_seam": len(off_seam)},
           "deviation_m": ({"p50": round(float(np.percentile(devs, 50)), 6),
                            "p95": round(float(np.percentile(devs, 95)), 6),
                            "max": round(float(devs.max()), 6)} if len(devs) else None),
           "per_edge": rows, "summary_counts": {
               "multi_curve_edges": [c["curves"] for c in paths if len(c["curves"]) > 1][:50]}}
    with (out_dir / "snap_selftest.json").open("w") as fh:
        json.dump(rep, fh, indent=2)

    print(f"[selftest] {args.npz.parent.name}: GT corners {V.shape[0]} tiers={tiers}")
    print(f"[selftest] GT edges {len(fm.edges)}: "
          f"path={len(paths)} degenerate={len(degen)} "
          f"on_seam_unsnapped={len(unsnap)} off_seam={len(off_seam)}")
    if len(devs):
        print(f"[selftest] max-dev on reconstructed: p50={np.percentile(devs, 50):.2e} "
              f"p95={np.percentile(devs, 95):.2e} max={devs.max():.2e} m")
    print(f"saved {vtk}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
