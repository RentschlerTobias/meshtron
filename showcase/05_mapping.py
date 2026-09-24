"""05 -- mapping a blocking onto the geometry, low level.

Corners snap to features, edges are routed, boundary faces are projected, the
volume is filled by transfinite interpolation. Each stage writes a VTK and
prints what it changed, so the failure modes are visible rather than argued.
"""
# %% [0] setup
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

from block_mapping import SnapConfigV2, snap_corners_v2  # noqa: E402
from curved_bridge import refill_curved  # noqa: E402
from geometry_features import FeatureModelV2  # noqa: E402
from patch_paths import (PatchPaths, make_face_projector,  # noqa: E402
                         max_kink_deg, snap_seam_path, write_debug_vtk)
from scripts.conform_gt_blocks import _boundary_edge_pred  # noqa: E402
from scripts.map_generated_blocks import _seam_path_fn  # noqa: E402

npz = os.path.join(C.BATCH, C.MACHINE, "sample.npz")
z = np.load(npz, allow_pickle=True)
fm = FeatureModelV2(npz, cache_dir=os.path.join(C.DATA, "features"))
target = types.SimpleNamespace(curves=fm.seam_curves,
                               surface_nearest=fm.surface_nearest)
out = C.outdir()

# %% [1] the feature model
# Patches come from the triangle labels; the seam curves are their boundaries.
# Everything the mapping knows about the geometry is in here.
C.head("[1] feature model")
C.show("surface_points", fm.surface_points, 0)
C.show("surface_tris", fm.surface_tris, 0)
seam = fm.seam_curves
print(f"   {seam.n_curves} seam curves over {len(seam.pts)} polyline points")
for c in range(min(seam.n_curves, 8)):
    lo, hi = int(seam.label_lo[c]), int(seam.label_hi[c])
    a, b = int(seam.offset[c]), int(seam.offset[c + 1])
    print(f"      curve {c:2d}  {C.PATCHES.get(lo, lo):11s} | "
          f"{C.PATCHES.get(hi, hi):11s}  {b - a:4d} points  "
          f"closed={bool(seam.closed[c])}")

# %% [2] snapping the corners
# Every block corner is pulled onto the nearest feature: a seam junction, a
# seam curve, or a patch. The tier tells which of the three it was.
C.head("[2] snap")
corners = fm.vertices[fm.blocks].astype(float)
snapped, records = snap_corners_v2(target, corners, SnapConfigV2())
move = np.linalg.norm(snapped.reshape(-1, 3) - corners.reshape(-1, 3), axis=1)
from collections import Counter  # noqa: E402
print(f"   tiers {dict(Counter(r['tier'] for r in records))}")
print(f"   moved p50 {np.median(move):.2e}  max {move.max():.2e}")
d, _, _ = fm.surface_nearest(snapped.reshape(-1, 3), k=32)
print(f"   corners to surface afterwards: max {d.max():.2e}")

# %% [3] routing one edge, watched
# An edge whose ends sit on a seam follows that seam. Otherwise it is a
# shortest path on the patch it belongs to -- and picking THAT patch is where
# a generated blocking goes wrong if the choice is made by proximity alone.
C.head("[3] one edge")
stats = {"routes": 0, "edges_surface_projected": 0,
         "edges_walked_multi_patch": 0}
raw = _seam_path_fn(fm.seam_curves, records, stats, tol=1e-9)
geo = PatchPaths(fm, records=records, stats=stats,
                 is_boundary=_boundary_edge_pred(fm.blocks, snapped))
p0, p1 = snapped[0, 0], snapped[0, 1]
print(f"   endpoints {p0.round(4)} -> {p1.round(4)}")
print(f"   candidate patches {geo.candidates(p0, p1)}")
R, info = geo.route(p0, p1, 25)
for k in ("chosen", "tried", "arc_over_chord", "max_dist_patch",
          "max_kink_deg"):
    if k in info:
        print(f"   {k:16s} {info[k]}")

# %% [4] all edges
# Three kinds: seam curves, geodesics on a patch, and interior chords that no
# patch constrains.
C.head("[4] all edges")


def seam_fn(a, b, n):
    r = raw(a, b, n)
    return None if r is None else (snap_seam_path(fm.seam_curves, fm, r[0]),
                                   r[1])


def path_fn(a, b, n):
    r = seam_fn(a, b, n)
    return r if r is not None else geo(a, b, n)


box = {}
mesh = os.path.join(out, "15_mapped.vtk")
rep = refill_curved(snapped, 0.05, mesh, fm=target, path_fn=path_fn,
                    write_edges=False,
                    edge_post_fn=lambda st, ci, cs, bl: box.update(st=st),
                    face_project_fn=make_face_projector(geo, stats))
st = box["st"]
kinds, polys = [], []
for key, Q in st.edge_pts.items():
    cid = int(st.edge_curve[key])
    kinds.append(0 if cid < 0 else (1 if cid < 900000 else 2))
    polys.append(np.asarray(Q, float))
print(f"   {len(polys)} unique edges: "
      f"{kinds.count(1)} seam, {kinds.count(2)} geodesic, "
      f"{kinds.count(0)} chord")
dd = [float(fm.surface_nearest(q, k=32)[0].max()) for q in polys]
print(f"   distance to the surface: seam/geodesic max "
      f"{max(d for d, k in zip(dd, kinds) if k):.2e}, "
      f"chords max {max([d for d, k in zip(dd, kinds) if not k] or [0]):.2e}")
write_debug_vtk(os.path.join(out, "16_edges.vtk"), polys,
                {"route_kind": kinds,
                 "max_kink_deg": [max_kink_deg(q) for q in polys]},
                "routed block edges (0 chord, 1 seam, 2 geodesic)")

# %% [5] the mesh, and the two different quality questions
# "on the geometry" and "covers the geometry" are not the same, and neither is
# "the cells are valid". All three have to be asked separately.
C.head("[5] quality")
with open(mesh) as fh:
    L = fh.read().split("\n")
i = next(k for k, l in enumerate(L) if l.startswith("POINTS"))
n = int(L[i].split()[1])
P = np.array([[float(v) for v in L[i + 1 + k].split()] for k in range(n)])
j = next(k for k, l in enumerate(L) if l.startswith("CELLS"))
m = int(L[j].split()[1])
H = np.array([[int(v) for v in L[j + 1 + k].split()][1:] for k in range(m)])
sb = next(k for k, l in enumerate(L) if l.startswith("SCALARS block_id"))
bid = np.array([int(float(L[sb + 2 + k])) for k in range(m)])

bids = rep["boundary_point_ids"]
d, _, _ = fm.surface_nearest(P[bids], k=32)
print(f"   on the geometry    max {d.max():.2e}   (gate 1e-3)")

from scipy.spatial import cKDTree  # noqa: E402
tri_cen = fm.surface_points[fm.surface_tris].mean(axis=1)
cov, _ = cKDTree(P[bids]).query(tri_cen)
print(f"   covers it          max {cov.max():.4f}  uncovered(>0.15) "
      f"{int((cov > 0.15).sum())} of {len(cov)}")

sj = C.scaled_jacobians(P, H)
print(f"   cells valid        {int((sj <= 0).sum())} inverted of {len(H)} "
      f"({100 * (sj <= 0).mean():.2f}%)  min SJ {sj.min():.3f}")
C.write_hexes(mesh, P, H, {"block_id": bid, "scaled_jacobian": sj,
                           "inverted": (sj <= 0).astype(int)},
              "mapped mesh (colour by inverted)")

# %% [6] where the folds are, and that they are ours
# Refilled with the blocking's OWN edge polylines the same corners give an
# almost valid mesh. The difference is the edge shape we impose.
C.head("[6] the folds are introduced by the routing")
inv = np.where(sj <= 0)[0]
if len(inv):
    cen = P[H[inv]].mean(axis=1)
    _, tri, _ = fm.surface_nearest(cen, k=32)
    lab = Counter(int(v) for v in fm.surface_tri_label[tri])
    print(f"   {len(inv)} folded cells, nearest patch "
          f"{ {C.PATCHES.get(k, k): v for k, v in lab.items()} }")
    print(f"   per block {dict(sorted(Counter(bid[inv].tolist()).items()))}")

E, EP, OFF = z["edges"], z["edge_polyline"], z["edge_polyline_offset"]
poly = {}
for k, (a, b) in enumerate(E):
    Q = EP[OFF[k]:OFF[k + 1]]
    poly[(int(a), int(b))] = Q
    poly[(int(b), int(a))] = Q[::-1]
km = {}
for r in range(fm.blocks.shape[0]):
    for c in range(8):
        km[np.round(snapped[r, c], 9).tobytes()] = int(fm.blocks[r, c])


def _rs(Q, nn):
    s = np.concatenate([[0.0],
                        np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))])
    if s[-1] <= 1e-12:
        return np.repeat(Q[:1], nn, axis=0)
    q = np.linspace(0.0, s[-1], nn)
    return np.stack([np.interp(q, s, Q[:, k]) for k in range(3)], axis=1)


def gt_path(a, b, nn):
    ia = km.get(np.round(np.asarray(a, float), 9).tobytes())
    ib = km.get(np.round(np.asarray(b, float), 9).tobytes())
    if ia is None or ib is None or (ia, ib) not in poly:
        return None
    R = _rs(np.asarray(poly[(ia, ib)], float), nn)
    R[0], R[-1] = a, b
    return R, 700000


shim = types.SimpleNamespace(curves=None, surface_nearest=None)
ref = os.path.join(out, "17_mapped_gtcurve.vtk")
refill_curved(snapped, 0.05, ref, fm=shim, path_fn=gt_path, write_edges=False)
with open(ref) as fh:
    L2 = fh.read().split("\n")
i = next(k for k, l in enumerate(L2) if l.startswith("POINTS"))
n2 = int(L2[i].split()[1])
P2 = np.array([[float(v) for v in L2[i + 1 + k].split()] for k in range(n2)])
j = next(k for k, l in enumerate(L2) if l.startswith("CELLS"))
m2 = int(L2[j].split()[1])
H2 = np.array([[int(v) for v in L2[j + 1 + k].split()][1:] for k in range(m2)])
sj2 = C.scaled_jacobians(P2, H2)
C.write_hexes(ref, P2, H2, {"scaled_jacobian": sj2,
                            "inverted": (sj2 <= 0).astype(int)},
              "same corners, the blocking's own edge curves")
print(f"\n   our routing       {int((sj <= 0).sum()):5d} inverted, "
      f"boundary {d.max():.2e}")
print(f"   its own curves    {int((sj2 <= 0).sum()):5d} inverted, "
      f"boundary not conforming (~0.14)")
print("\n   conformity and cell validity are not yet available at the same")
print("   time -- that is the open end of the mapping work.")
