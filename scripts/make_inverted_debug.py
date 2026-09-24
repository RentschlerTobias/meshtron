"""make_inverted_debug.py -- artifacts for looking at the folded cells.

The conformed mesh holds the geometry to 1e-10 but carries folded cells that
the block structure itself does not have: refilled with its own edge polylines,
machine_0034_n2000 produces 9 inverted cells; routed onto the geometry by this
pipeline it produces a few hundred. The folds are ours, so this writes the two
meshes side by side plus the folded cells on their own.

  <name>_ours.vtk           our conformed mesh, cell arrays scaled_jacobian,
                            inverted, block_id, dist_to_boundary
  <name>_gtcurve.vtk        the same blocking refilled with its OWN edge
                            polylines -- the reference to beat
  <name>_inverted_only.vtk  just the folded cells of our mesh, so they can be
                            seen without thresholding
  <name>_edges.vtk          the routed block edges, scalar route_kind

Usage:
  uv run python scripts/make_inverted_debug.py --machine machine_0034_n2000
"""
from __future__ import annotations

import argparse
import os
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
HEX3D = ("/home/t1dde/hydrostack_pipeline/stack/domain_partition_3D/"
         "experimentell/hex3d_algohex")
if HEX3D not in sys.path:
    sys.path.insert(0, HEX3D)

import clean_blocks as cb  # noqa: E402  (extern, read-only)

from meshtron.geometry.block_mapping import SnapConfigV2, snap_corners_v2  # noqa: E402
from meshtron.geometry.curved_bridge import refill_curved  # noqa: E402
from meshtron.geometry.geometry_features import FeatureModelV2  # noqa: E402
from meshtron.geometry.patch_paths import (PatchPaths, make_face_projector,  # noqa: E402
                         max_kink_deg, snap_seam_path)
from scripts.conform_gt_blocks import _boundary_edge_pred  # noqa: E402
from scripts.map_generated_blocks import _seam_path_fn  # noqa: E402

BATCH = os.path.join(ROOT, "data", "hex3d_algohex", "batch")
HEX_FACES = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5),
             (2, 3, 7, 6), (3, 0, 4, 7))


def _resample(Q, n):
    s = np.concatenate([[0.0],
                        np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))])
    if s[-1] <= 1e-12:
        return np.repeat(Q[:1], n, axis=0)
    q = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(q, s, Q[:, k]) for k in range(3)], axis=1)


def _read(path):
    with open(path) as fh:
        L = fh.read().split("\n")
    i = next(k for k, l in enumerate(L) if l.startswith("POINTS"))
    n = int(L[i].split()[1])
    P = np.array([[float(x) for x in L[i + 1 + k].split()] for k in range(n)])
    j = next(k for k, l in enumerate(L) if l.startswith("CELLS"))
    m = int(L[j].split()[1])
    H = np.array([[int(x) for x in L[j + 1 + k].split()][1:] for k in range(m)])
    s = next((k for k, l in enumerate(L) if l.startswith("SCALARS block_id")),
             None)
    bid = (np.array([int(float(L[s + 2 + k])) for k in range(m)])
           if s is not None else np.zeros(m, int))
    return P, H, bid


def _write(path, P, H, arrays, title):
    with open(path, "w") as fh:
        fh.write(f"# vtk DataFile Version 2.0\n{title}\nASCII\n")
        fh.write("DATASET UNSTRUCTURED_GRID\n")
        fh.write(f"POINTS {len(P)} double\n")
        for q in P:
            fh.write("%.9f %.9f %.9f\n" % tuple(q))
        fh.write(f"CELLS {len(H)} {9 * len(H)}\n")
        for c in H:
            fh.write("8 " + " ".join(str(int(x)) for x in c) + "\n")
        fh.write(f"CELL_TYPES {len(H)}\n")
        for _ in H:
            fh.write("12\n")
        fh.write(f"CELL_DATA {len(H)}\n")
        for name, vals in arrays.items():
            v = np.asarray(vals)
            isint = np.issubdtype(v.dtype, np.integer)
            fh.write(f"SCALARS {name} {'int' if isint else 'double'} 1\n")
            fh.write("LOOKUP_TABLE default\n")
            for x in v:
                fh.write(("%d\n" % x) if isint else ("%.9e\n" % x))


def main() -> int:
    ap = argparse.ArgumentParser(description="folded-cell artifacts")
    ap.add_argument("--machine", default="machine_0034_n2000")
    ap.add_argument("--target-h", type=float, default=0.05)
    ap.add_argument("--out", default=os.path.join(ROOT, "data",
                                                  "inverted_debug"))
    args = ap.parse_args()
    name = args.machine
    os.makedirs(args.out, exist_ok=True)
    npz = os.path.join(BATCH, name, "sample.npz")
    z = np.load(npz, allow_pickle=True)
    V = np.asarray(z["vertices"], float)
    B = np.asarray(z["blocks"], np.int64)
    E, EP, OFF = z["edges"], z["edge_polyline"], z["edge_polyline_offset"]
    fm = FeatureModelV2(npz, cache_dir=os.path.join(ROOT, "data", "features"))
    target = types.SimpleNamespace(curves=fm.seam_curves,
                                   surface_nearest=fm.surface_nearest)
    C = V[B].astype(np.float64)
    C_snap, records = snap_corners_v2(target, C, SnapConfigV2())

    # ours
    stats = {"routes": 0, "edges_surface_projected": 0,
             "edges_walked_multi_patch": 0}
    raw = _seam_path_fn(fm.seam_curves, records, stats, tol=1e-9)

    def seam(p0, p1, n):
        r = raw(p0, p1, n)
        return None if r is None else (
            snap_seam_path(fm.seam_curves, fm, r[0]), r[1])

    geo = PatchPaths(fm, records=records, stats=stats,
                     is_boundary=_boundary_edge_pred(B, C_snap))

    def path_fn(p0, p1, n):
        r = seam(p0, p1, n)
        return r if r is not None else geo(p0, p1, n)

    ours = os.path.join(args.out, f"{name}_ours.vtk")
    edges = os.path.join(args.out, f"{name}_edges.vtk")
    box = {}
    refill_curved(C_snap, args.target_h, ours, fm=target, path_fn=path_fn,
                  write_edges=False,
                  edge_post_fn=lambda st, ci, cs, bl: box.update(st=st),
                  face_project_fn=make_face_projector(geo, stats))

    # GT curvature reference
    poly = {}
    for k, (a, b) in enumerate(E):
        Q = EP[OFF[k]:OFF[k + 1]]
        poly[(int(a), int(b))] = Q
        poly[(int(b), int(a))] = Q[::-1]
    km = {}
    for r in range(B.shape[0]):
        for c in range(8):
            km[np.round(C_snap[r, c], 9).tobytes()] = int(B[r, c])

    def gt_path(p0, p1, n):
        a = km.get(np.round(np.asarray(p0, float), 9).tobytes())
        b = km.get(np.round(np.asarray(p1, float), 9).tobytes())
        if a is None or b is None or (a, b) not in poly:
            return None
        R = _resample(np.asarray(poly[(a, b)], float), n)
        R[0], R[-1] = p0, p1
        return R, 700000

    shim = types.SimpleNamespace(curves=None, surface_nearest=None)
    gtc = os.path.join(args.out, f"{name}_gtcurve.vtk")
    refill_curved(C_snap, args.target_h, gtc, fm=shim, path_fn=gt_path,
                  write_edges=False)

    rows = []
    for tag, path in (("ours", ours), ("gtcurve", gtc)):
        P, H, bid = _read(path)
        sj = cb.scaled_jacobians(P, H)
        cnt = {}
        for c in H:
            for f in HEX_FACES:
                k = tuple(sorted(int(c[x]) for x in f))
                cnt[k] = cnt.get(k, 0) + 1
        bp = np.unique([x for k, v in cnt.items() if v == 1 for x in k])
        from scipy.spatial import cKDTree
        dist, _ = cKDTree(P[bp]).query(P[H].mean(axis=1))
        _write(path, P, H, {"block_id": bid.astype(int),
                            "scaled_jacobian": sj,
                            "inverted": (sj <= 0).astype(int),
                            "dist_to_boundary": dist},
               f"{name} {tag}: scaled_jacobian, inverted, block_id, "
               f"dist_to_boundary")
        rows.append((tag, len(H), int((sj <= 0).sum()), float(sj.min()),
                     float(np.median(dist[sj <= 0])) if (sj <= 0).any() else 0.0))
        if tag == "ours":
            sel = np.where(sj <= 0)[0]
            if len(sel):
                used = np.unique(H[sel])
                rm = {int(v): i for i, v in enumerate(used)}
                _write(os.path.join(args.out, f"{name}_inverted_only.vtk"),
                       P[used], [[rm[int(x)] for x in H[c]] for c in sel],
                       {"block_id": bid[sel].astype(int),
                        "scaled_jacobian": sj[sel],
                        "dist_to_boundary": dist[sel]},
                       f"{name}: the {len(sel)} folded cells of the conformed "
                       f"mesh")
            per = {}
            for c in sel:
                per[int(bid[c])] = per.get(int(bid[c]), 0) + 1
            print("folded cells per block:", dict(sorted(per.items())))

    st = box.get("st")
    if st is not None:
        polys = [np.asarray(q, float) for q in st.edge_pts.values()]
        kinds = [(1 if 0 <= int(st.edge_curve[k]) < 900000
                  else (2 if int(st.edge_curve[k]) >= 900000 else 0))
                 for k in st.edge_pts]
        from meshtron.geometry.patch_paths import write_debug_vtk
        write_debug_vtk(edges, polys,
                        {"route_kind": kinds,
                         "max_kink_deg": [max_kink_deg(q) for q in polys]},
                        f"{name} routed block edges "
                        f"(route_kind 0=chord 1=seam 2=geodesic)")

    print(f"\n{'variant':10s} {'cells':>8s} {'inverted':>9s} {'min SJ':>8s} "
          f"{'median dist to boundary':>24s}")
    for tag, n, inv, mn, dm in rows:
        print(f"{tag:10s} {n:8d} {inv:9d} {mn:8.3f} {dm:24.4f}")
    print(f"\nartifacts in {args.out}/")
    for f in sorted(os.listdir(args.out)):
        if f.startswith(name):
            print("   ", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
