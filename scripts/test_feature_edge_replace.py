#!/usr/bin/env python
"""test_feature_edge_replace.py -- T1: GT coarse block edges -> npz feature sub-curves.

Takes the GROUND-TRUTH coarse block-structure of one npz sample (vertices/blocks
from the npz, exactly like the h=0.5 adaptive recipe), classifies every welded
block corner against the npz boundary labels (inlet/outlet/periodic/hub-band/
shroud-band/blade), snaps it onto the ANALYTIC surface of that label and
REPLACES every block edge that has a geometric counterpart by the exact feature
sub-curve between the two projected endpoints.

Ansatz 1a (user): the feature edge REPLACES the straight block edge -- the full
curve between the two projected endpoints, independent of node density. Curved
block edges are wanted; plain point-snapping is not.

Snap targets (derived from the npz label, never from the nearest triangle
position):
  L1 inlet        -> plane z = Z_INLET  (0.0)
  L2 outlet       -> plane z = Z_OUTLET (2.5)
  L3/L4 periodic  -> nearest triangle of the label (the T1_9 patches are curved,
                     so the nominal const-theta plane is enforced locally)
  L5 bl_iface_hub -> cylinder r = R_HUB     (0.5; --cyl-radius band -> 0.595)
  L6 bl_iface_sh -> cylinder r = R_SHROUD   (1.9; --cyl-radius band -> 1.801)
  L7 ogrid_iface  -> nearest L7 triangle (mesh-bound blade surface)

The npz carries L5/L6 only as thin bands (r~0.59/1.80) while the analytic walls
are r=0.5/1.9, so hub/shroud corners move ~0.1 radially in the default (wall)
mode; --cyl-radius band keeps them on the measured npz band instead.

Match rule (user): ONE endpoint inside the tolerance is enough. The tolerance is
RELATIVE by default (tol = --snap-frac * edge_length), overridable with
--snap-abs. It gates only whether a corner is *considered*, not the curve
replacement.

Output VTK (POLYLINE cells, VTK type 4) with a scalar `part`:
  1 = straight / kept (planar or meridional chord already on the surface)
  2 = feature-replaced (curved sub-curve)
  3 = partial-snap (only one endpoint covered)
  4 = uncovered (no endpoint covered / beyond tolerance)
  5 = original straight GT chords (reference)
  6 = npz seam curves (context)
plus a JSON report.

Usage:
  uv run python scripts/test_feature_edge_replace.py \
    --npz data/hex3d_algohex/batch/T1_9_n2000/sample.npz --target-h 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import curved_bridge  # noqa: E402
from edge_curves import local_edges  # noqa: E402
from geometry_features import FeatureModelV2  # noqa: E402
from scripts.compare_viz import _write_parts_vtk  # noqa: E402

HEX3D_REPO = curved_bridge.HEX3D_REPO
if HEX3D_REPO not in sys.path:
    sys.path.insert(0, HEX3D_REPO)
from clean_blocks import _closest_point_on_tris  # noqa: E402

R_HUB, R_SHROUD = 0.5, 1.9
Z_INLET, Z_OUTLET = 0.0, 2.5
LABEL_NAMES = {1: "inlet", 2: "outlet", 3: "periodic_A", 4: "periodic_B",
               5: "bl_iface_hub", 6: "bl_iface_shroud", 7: "ogrid_iface"}
CYL_R = {5: R_HUB, 6: R_SHROUD}
PLANE_Z = {1: Z_INLET, 2: Z_OUTLET}
DEFAULT_NPZ = ROOT / "data" / "hex3d_algohex" / "batch" / "T1_9_n2000" / "sample.npz"
DEFAULT_OUT = ROOT / "data" / "features_debug" / "t1_9_gt_gross_feature_edges_replace.vtk"
DEFAULT_JSON = ROOT / "data" / "features_debug" / "t1_9_gt_gross_feature_edges_replace.json"
N_CURVE = 24


def _rz(p: np.ndarray) -> tuple[float, float, float]:
    return float(np.hypot(p[0], p[1])), float(np.arctan2(p[1], p[0])), float(p[2])


def _lin(a: np.ndarray, b: np.ndarray, n: int = 2) -> np.ndarray:
    t = np.linspace(0.0, 1.0, max(int(n), 2))
    return a[None] + t[:, None] * (b - a)[None]


class LabelSurfaces:
    """Analytic snap targets + label-restricted triangle projection."""

    def __init__(self, fm: FeatureModelV2, cyl_radius: str = "wall") -> None:
        self.fm = fm
        self._trees: dict[int, tuple[np.ndarray, cKDTree]] = {}
        self.cyl = dict(CYL_R)
        if cyl_radius == "band":
            for label in CYL_R:
                tri, _tree = self._tree(label)
                cen = fm.surface_points[tri].mean(axis=1)
                self.cyl[label] = float(np.hypot(cen[:, 0], cen[:, 1]).mean())

    def _tree(self, label: int):
        if label not in self._trees:
            m = self.fm.surface_tri_label == label
            tri = self.fm.surface_tris[m]
            cen = self.fm.surface_points[tri].mean(axis=1)
            self._trees[label] = (tri, cKDTree(cen))
        return self._trees[label]

    def project(self, q: np.ndarray, label: int, k: int = 32
                ) -> tuple[np.ndarray, float]:
        """Nearest point of `q` on the label's triangles (mesh-bound)."""
        tri, tree = self._tree(label)
        _, cand = tree.query(q.reshape(1, 3), k=min(k, len(tri)))
        cand = np.atleast_2d(cand)
        T = tri[cand]
        P = self.fm.surface_points
        d2, p = _closest_point_on_tris(q.reshape(1, 3), P[T[..., 0]], P[T[..., 1]],
                                       P[T[..., 2]])
        best = int(np.argmin(d2[0]))
        return p[0, best].copy(), float(np.sqrt(d2[0, best]))

    def target(self, q: np.ndarray, label: int) -> np.ndarray:
        """Analytic snap target of corner `q` for its npz `label`."""
        _r, th, z = _rz(q)
        if label in CYL_R:
            R = self.cyl[label]
            return np.array([R * np.cos(th), R * np.sin(th), z])
        if label in PLANE_Z:
            return np.array([q[0], q[1], PLANE_Z[label]])
        if label in (3, 4):
            # L3/L4 are nominally const-theta, but the T1_9 patches are curved
            # (fitted-plane residual up to 0.18), so force the LOCAL plane: project
            # onto the nearest triangle of the same label (mesh-bound, move ~0).
            return self.project(q, label)[0]
        if label == 7:
            return self.project(q, 7)[0]
        return q.copy()

    def mesh_path(self, a: np.ndarray, b: np.ndarray, label: int, n: int
                  ) -> np.ndarray:
        """Sample a->b and project every sample onto the label's surface."""
        tri, tree = self._tree(label)
        Q = _lin(a, b, n)
        _, cand = tree.query(Q, k=min(32, len(tri)))
        cand = np.atleast_2d(cand)
        T = tri[cand]
        P = self.fm.surface_points
        d2, p = _closest_point_on_tris(Q, P[T[..., 0]], P[T[..., 1]], P[T[..., 2]])
        best = np.argmin(d2, axis=1)
        out = p[np.arange(len(Q)), best].copy()
        out[0], out[-1] = a, b
        return out


def classify_corners(fm: FeatureModelV2, P: np.ndarray, k: int = 32):
    """Label + raw (nearest-centroid) distance per welded corner.

    Label comes from the exact projection onto the nearest triangle (more stable
    at junctions than the centroid vote); the priority rule then forces the
    band/plane family by geometry. The raw distance reported is the distance to
    the nearest triangle CENTROID (goal step 1), used as the coverage gate.
    """
    dcent, j = fm.tri_tree.query(P, k=k)
    labs_cen = fm.surface_tri_label[j]
    _, tri, _ = fm.surface_nearest(P, k=k)
    labs_proj = fm.surface_tri_label[tri]
    out_lab = np.zeros(len(P), np.int64)
    for i in range(len(P)):
        lab = int(labs_proj[i])
        r, _th, z = _rz(P[i])
        if lab in CYL_R:
            lab = 5 if abs(r - R_HUB) <= abs(r - R_SHROUD) else 6
        elif lab in PLANE_Z:
            lab = 1 if abs(z - Z_INLET) <= abs(z - Z_OUTLET) else 2
        out_lab[i] = lab
    return out_lab, dcent[:, 0], labs_cen[:, 0], labs_proj


def seam_path(fm: FeatureModelV2, a: np.ndarray, b: np.ndarray, la: int, lb: int,
              tol: float, n: int):
    """Sub-arc of the npz seam curve that connects the two endpoint labels.

    Detected on the ORIGINAL corners (they lie on the npz surface); the seam's
    label pair must be exactly {la, lb} so a mid-band edge cannot grab an
    unrelated label boundary that happens to be within tolerance."""
    if la == lb:
        return None, -1
    seam = fm.seam_curves
    d, c, t, _pt = seam.nearest(np.stack([a, b]))
    ca, cb = int(c[0]), int(c[1])
    if ca != cb or d[0] > tol or d[1] > tol:
        return None, -1
    if {int(seam.label_lo[ca]), int(seam.label_hi[ca])} != {la, lb}:
        return None, -1
    seg = seam.sample_segment(ca, float(t[0]), float(t[1]), n)
    if np.linalg.norm(seg[0] - a) > np.linalg.norm(seg[-1] - a):
        seg = seg[::-1]
    return seg, ca


def sample_cylinder(a: np.ndarray, b: np.ndarray, R: float, n: int) -> np.ndarray:
    """Geodesic (helix) on the cylinder of radius R from a to b."""
    tha, thb = float(np.arctan2(a[1], a[0])), float(np.arctan2(b[1], b[0]))
    dth = (thb - tha + np.pi) % (2.0 * np.pi) - np.pi
    t = np.linspace(0.0, 1.0, max(int(n), 2))
    th = tha + t * dth
    z = a[2] + t * (b[2] - a[2])
    return np.stack([R * np.cos(th), R * np.sin(th), z], axis=1)


def _surface_dist(fm: FeatureModelV2, Q: np.ndarray) -> float:
    d, _, _ = fm.surface_nearest(Q, k=32)
    return float(np.max(d)) if len(d) else 0.0


def build_edges(H: np.ndarray):
    """Unique welded GT edges -> {key: (block, local_start, local_end, axis)}."""
    seen: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for r in range(H.shape[0]):
        for li, lj, ax in local_edges():
            a, b = int(H[r, li]), int(H[r, lj])
            key = (min(a, b), max(a, b))
            if key not in seen:
                seen[key] = (r, a, b, ax)
    return seen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    ap.add_argument("--target-h", type=float, default=0.5)
    ap.add_argument("--snap-frac", type=float, default=0.2)
    ap.add_argument("--snap-abs", type=float, default=None)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--limit", type=int, default=None, help="only first N blocks")
    ap.add_argument("--cyl-radius", choices=("wall", "band"), default="wall",
                    help="L5/L6 target radius: analytic wall (R_HUB/R_SHROUD) or "
                         "measured npz band radius")
    ap.add_argument("--feature-cache", type=str, default=str(ROOT / "data" / "features"))
    args = ap.parse_args()

    fm = FeatureModelV2(str(args.npz), cache_dir=args.feature_cache)
    surf = LabelSurfaces(fm, cyl_radius=args.cyl_radius)

    tfi, _ev, bc, _cb = curved_bridge._load()
    C = fm.vertices[fm.blocks].astype(np.float64)
    nb = C.shape[0]
    if args.limit is not None:
        nb = min(nb, int(args.limit))
        C = C[:nb]
    P, remap = tfi.weld(C.reshape(-1, 3))
    H = remap.reshape(nb, 8)
    B = np.arange(nb, dtype=np.int64)
    lat = {r: (np.ones(3, int), curved_bridge._lattice_vert(H[r], tfi.CORNER))
           for r in range(nb)}
    _f2h, classes, counts, mode, reason = curved_bridge._classes_counts(
        tfi, bc, lat, H, B, P, args.target_h)
    cof = tfi.class_of_axis(classes)
    dims = [tuple(int(counts[cof[(r, ax)]]) for ax in (0, 1, 2)) for r in range(nb)]
    n_cells = int(sum(int(np.prod(d)) for d in dims))

    lab, d_raw, lab_cen, lab_proj = classify_corners(fm, P)
    label_counts = {int(k): int((lab == k).sum()) for k in sorted(set(lab.tolist()))}

    # analytic snap targets per welded corner
    tgt = np.stack([surf.target(P[i], int(lab[i])) for i in range(len(P))])
    corner_records = []
    for i in range(len(P)):
        r, th, z = _rz(P[i])
        corner_records.append({
            "id": int(i), "xyz": [round(float(v), 6) for v in P[i]],
            "r": round(r, 6), "theta_deg": round(float(np.degrees(th)), 4),
            "z": round(z, 6),
            "label": int(lab[i]), "label_name": LABEL_NAMES.get(int(lab[i]), "?"),
            "label_centroid": int(lab_cen[i]), "label_proj": int(lab_proj[i]),
            "raw_dist": round(float(d_raw[i]), 6),
            "snap_target": [round(float(v), 6) for v in tgt[i]],
            "snap_move": round(float(np.linalg.norm(tgt[i] - P[i])), 6),
        })

    edges = build_edges(H)
    n_gt_edges = len(edges)

    parts: dict[int, list] = defaultdict(list)
    records: list[dict] = []
    counts = {"feature_replaced": 0, "partial_snap": 0,
              "chord_interior": 0, "boundary_uncovered": 0}
    seams_used: dict[int, int] = defaultdict(int)
    covered_ids: set[int] = set()

    for key, (blk, a, b, ax) in sorted(edges.items()):
        A, Bc = P[a], P[b]
        LA, LB = int(lab[a]), int(lab[b])
        length = float(np.linalg.norm(Bc - A))
        tol = float(args.snap_abs) if args.snap_abs is not None else args.snap_frac * length
        cov_a, cov_b = bool(d_raw[a] <= tol), bool(d_raw[b] <= tol)
        ncov = int(cov_a) + int(cov_b)
        Ta, Tb = tgt[a], tgt[b]
        source, replaced, pts = "chord", False, _lin(A, Bc, 2)

        if ncov == 2:
            seg, cid = seam_path(fm, A, Bc, LA, LB, tol, N_CURVE)
            if seg is not None:
                pts, source, replaced = seg, "seam", True
                seams_used[int(cid)] += 1
            elif LA == LB and LA in CYL_R:
                pts = sample_cylinder(Ta, Tb, surf.cyl[LA], N_CURVE)
                dth = abs(float(np.arctan2(Tb[1], Tb[0]) - np.arctan2(Ta[1], Ta[0])))
                source, replaced = "cylinder", bool(dth > 1e-6)
                if not replaced:
                    pts = _lin(Ta, Tb, 2)
            elif LA == LB and LA in PLANE_Z:
                pts, source, replaced = _lin(Ta, Tb, 2), "plane", False
            elif LA == LB and LA in (3, 4):
                pts, source, replaced = _lin(Ta, Tb, 2), "theta_plane", False
            elif LA == LB and LA == 7:
                pts = surf.mesh_path(Ta, Tb, 7, N_CURVE)
                source, replaced = "blade_mesh", bool(
                    np.max(np.linalg.norm(pts - _lin(Ta, Tb, N_CURVE), axis=1)) > 1e-4)
            else:
                pts, source, replaced = _lin(Ta, Tb, 2), "interior_chord", False
            part = 2 if replaced else 1
            counts["feature_replaced" if replaced else "chord_interior"] += 1
        elif ncov == 1:
            pts = _lin(Ta, Bc, 2) if cov_a else _lin(A, Tb, 2)
            source, part = "partial_snap", 3
            counts["partial_snap"] += 1
        else:
            pts, source, part = _lin(A, Bc, 2), "uncovered", 4
            counts["boundary_uncovered"] += 1

        pts = np.asarray(pts, float)
        parts[part].append(pts)
        if cov_a:
            covered_ids.add(int(a))
        if cov_b:
            covered_ids.add(int(b))
        records.append({
            "edge_key": [int(key[0]), int(key[1])],
            "block": int(blk), "local_axis": int(ax),
            "endpoints": [[round(float(v), 6) for v in A],
                          [round(float(v), 6) for v in Bc]],
            "labels": [LABEL_NAMES.get(LA, str(LA)), LABEL_NAMES.get(LB, str(LB))],
            "label_ids": [LA, LB],
            "length": round(length, 6), "tol": round(tol, 6),
            "raw_dist": [round(float(d_raw[a]), 6), round(float(d_raw[b]), 6)],
            "covered": [cov_a, cov_b],
            "replaced": bool(replaced), "category": source, "part": int(part),
            "n_samples": int(len(pts)),
            "max_surface_distance": round(_surface_dist(fm, pts), 6),
        })

    # reference part 5 (original chords) + context part 6 (npz seam curves)
    ref_pts: list = []
    for key, (blk, a, b, ax) in sorted(edges.items()):
        ref_pts.append(_lin(P[a], P[b], 2))
    seam_pts: list = []
    for c in range(fm.seam_curves.n_curves):
        seam_pts.append(fm.seam_curves.segment(c))

    parts_emit = []
    for pid in (1, 2, 3, 4):
        if parts.get(pid):
            allp = np.concatenate(parts[pid], axis=0)
            cells, base = [], 0
            for q in parts[pid]:
                cells.append(list(range(base, base + len(q))))
                base += len(q)
            parts_emit.append((allp, cells, pid, 4))
    for pid, coll in ((5, ref_pts), (6, seam_pts)):
        if coll:
            allp = np.concatenate(coll, axis=0)
            cells, base = [], 0
            for q in coll:
                cells.append(list(range(base, base + len(q))))
                base += len(q)
            parts_emit.append((allp, cells, pid, 4))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write_parts_vtk(args.out, parts_emit,
                     f"T1 GT coarse edges vs npz feature sub-curves ({args.npz.parent.name})")

    # verification of snapped endpoints on the analytic surfaces
    snapped: dict[int, list] = defaultdict(list)
    for i in sorted(covered_ids):
        snapped[int(lab[i])].append(tgt[i])
    ver: dict[str, float | None] = {}
    for label, key, val in ((5, "max_abs_r_minus_R_HUB", R_HUB),
                            (6, "max_abs_r_minus_R_SHROUD", R_SHROUD),
                            (1, "max_abs_z_minus_0", Z_INLET),
                            (2, "max_abs_z_minus_2p5", Z_OUTLET)):
        q = snapped.get(label)
        if not q:
            ver[key] = None
        elif label in CYL_R:
            ver[key] = round(float(np.max(np.abs(np.hypot(np.array(q)[:, 0],
                                                          np.array(q)[:, 1]) - val))), 6)
        else:
            ver[key] = round(float(np.max(np.abs(np.array(q)[:, 2] - val))), 6)
    q7 = snapped.get(7)
    ver["max_L7_mesh_residual"] = (round(float(np.max([surf.project(p, 7)[1]
                                                       for p in q7])), 6)
                                   if q7 else None)

    category_counts: dict[str, int] = defaultdict(int)
    for rec in records:
        category_counts[rec["category"]] += 1

    report = {
        "npz": str(args.npz), "target_h": args.target_h,
        "snap_frac": args.snap_frac, "snap_abs": args.snap_abs,
        "cyl_radius": args.cyl_radius,
        "cyl_radii": {int(k): round(float(v), 6) for k, v in surf.cyl.items()},
        "limit": args.limit,
        "n_blocks": int(nb), "n_cells_adaptive": n_cells,
        "tfi_mode": mode, "tfi_reason": reason,
        "n_welded_corners": int(len(P)),
        "label_counts": label_counts,
        "n_gt_edges": int(n_gt_edges),
        "n_gt_edges_raw_local": int(nb * 12),
        "n_feature_replaced": counts["feature_replaced"],
        "n_partial_snap": counts["partial_snap"],
        "n_chord_interior": counts["chord_interior"],
        "n_boundary_uncovered": counts["boundary_uncovered"],
        "category_counts": dict(sorted(category_counts.items())),
        "seam_curves_used": {int(k): int(v) for k, v in sorted(seams_used.items())},
        "verification": ver,
        "parts_vtk": {"1": "straight/kept", "2": "feature-replaced",
                      "3": "partial-snap", "4": "uncovered",
                      "5": "original chords (ref)", "6": "npz seam curves (context)"},
        "assumptions": [
            "analytic targets: L1 z=0, L2 z=2.5, L5 r=R_HUB=0.5, L6 r=R_SHROUD=1.9 "
            "(npz bands sit at r~0.59/1.80, so hub/shroud corners move ~0.1 radially)",
            "L3/L4 are near-planar but TILTED (measured nz~-0.28), so 'theta forced' "
            "is generalised to an exact projection onto the fitted label plane",
            "coverage gate uses the nearest-triangle-centroid distance (goal step 1); "
            "the exact surface distance is ~0 for every GT corner",
            "seam sub-arc is used only when it connects the two endpoint labels and "
            "both original corners project onto it within tol",
            "one endpoint inside tol is enough (user rule); the tolerance gates only "
            "whether a corner is considered",
        ],
        "corners": corner_records,
        "edges": records,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as fh:
        json.dump(report, fh, indent=2)

    print("=" * 72)
    print(f"T1 feature-edge replacement  ({args.npz.parent.name})")
    print("=" * 72)
    print(f"npz                 : {args.npz}")
    print(f"target_h            : {args.target_h}   blocks={nb}  cells={n_cells}  "
          f"mode={mode}")
    print(f"welded corners      : {len(P)}   labels: " +
          " ".join(f"L{k}({LABEL_NAMES.get(k, '?')})={v}"
                   for k, v in sorted(label_counts.items())))
    rule = (f"abs {args.snap_abs}" if args.snap_abs is not None
            else f"{args.snap_frac} * edge_length (relative)")
    print(f"snap tolerance      : {rule}")
    print(f"cyl radius mode     : {args.cyl_radius}  "
          f"({', '.join(f'L{k}={v:.4f}' for k, v in surf.cyl.items())})")
    print("-" * 72)
    print(f"n_gt_edges          : {n_gt_edges}  (unique welded; "
          f"{nb * 12} raw local block edges)")
    print(f"n_feature_replaced  : {counts['feature_replaced']}")
    print(f"n_partial_snap      : {counts['partial_snap']}")
    print(f"n_chord_interior    : {counts['chord_interior']}")
    print(f"n_boundary_uncovered: {counts['boundary_uncovered']}")
    print(f"sum                 : {sum(counts.values())}")
    print(f"by source           : {dict(sorted(category_counts.items()))}")
    print("-" * 72)
    print("verification (snapped endpoints on analytic surfaces):")
    for k, v in ver.items():
        print(f"  {k:28s}: {v}")
    print(f"seam curves used    : {dict(seams_used) if seams_used else '{}'}")
    print("-" * 72)
    print(f"vtk  : {args.out}  ({args.out.stat().st_size} bytes)")
    print(f"json : {args.out_json}  ({args.out_json.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
