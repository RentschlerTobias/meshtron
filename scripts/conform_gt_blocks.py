"""conform_gt_blocks.py -- plan v3: conform the GT coarse block structure onto the
npz geometry (machine_0034_n2000 or any npz path).

Goal (verbatim from user): "Wir wollen die Blockstruktur auf unsere npz-Geometrie
uebertragen. Mehr nicht."

Reuses the proven production path of scripts/map_generated_blocks.py
(--curve-target seam) but feeds it the GT coarse blocks instead of generated ones:

  fm = FeatureModelV2(npz)                     # cache data/features (format 2)
  target = SimpleNamespace(curves=seam, surface_nearest=...)   # SEAM-ONLY shim
  C = fm.vertices[fm.blocks]                   # (nb,8,3) VTK hex corner order
  C_snap, records = snap_corners_v2(target, C, SnapConfigV2())
  path_fn = _seam_path_fn(seam, records, stats)
  rep = refill_curved(C_snap, target_h=10.0, ..., fm=target, path_fn=path_fn)

CRITICAL constraints (deep research, plan v3):
  - NEVER pass the full fm as snap/refill target: _gt_edge_curve would return
    block-complex edge polylines up to 0.454 off the npz surface. The seam-only
    shim routes through fm.seam_curves instead (boundary dist 5.5e-8 measured).
  - No analytic snap constants (plan v3 amendment A1; Momus ranking: highest
    silent-failure risk).
  - target_h=10.0 (>> max GT edge 1.41) => 1x1x1 lattice => 12 coarse cells.

Artifacts (data/features_debug/<stem>_gt_conform_*):
  conform_refill.vtk   12 curved hexes (Coons+Gordon-Hall TFI)
  conform_edges.vtk    curved block-edge polylines (84 unique welded)
  conform_compare.vtk  part 1=GT corners, 2=snapped corners
  conform_summary.json numeric tripwires

Tripwires (plan v3 amendment A2, fail with exit 2 on violation):
  - max boundary->npz-surface distance <= 1e-3
  - watertight
  - seam routes reported (informational; 0 => path_fn produced no routes)

Known pre-existing dataset artifact (scripts/map_generated_blocks.py:241-244):
  GT coarse blocks contain folded cells (min det J < 0, documented for block
  [4,7] on this geometry); inverted cells are reported but NOT a tripwire.

Usage:
  python scripts/conform_gt_blocks.py [--npz PATH] [--out-dir DIR]
                                      [--geodesic] [--surface-project]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from meshtron.geometry.block_mapping import SnapConfigV2, snap_corners_v2  # noqa: E402
from meshtron.geometry.curved_bridge import refill_curved  # noqa: E402
from meshtron.geometry.geometry_features import FeatureModelV2  # noqa: E402
from scripts.compare_viz import _write_parts_vtk  # noqa: E402
from meshtron.geometry.patch_paths import (  # noqa: E402
    BLEND_ID_BASE, PatchPaths, blend_boundary_edges, blend_chord_edges,
    make_boundary_face_test, make_face_projector, max_kink_deg,
    snap_seam_path, write_debug_vtk)
from scripts.map_generated_blocks import _seam_path_fn  # noqa: E402

DEFAULT_NPZ = os.path.join(ROOT, "data", "hex3d_algohex", "batch",
                           "machine_0034_n2000", "sample.npz")
DEFAULT_OUT = os.path.join(ROOT, "data", "features_debug")
BOUNDARY_TOL = 1e-3
WALK_ARC_MULT = 1.5      # walked path may exceed the chord by this factor
WALK_ID_BASE = 910000    # synthetic curve_id for multi-patch walked edges


def _resample(Q: np.ndarray, n: int) -> np.ndarray:
    """Polyline auf n bogenlaengen-aequidistante Punkte (wie edge_curves)."""
    s = np.concatenate([[0.0],
                        np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))])
    if s[-1] <= 1e-12:
        return np.repeat(Q[:1], n, axis=0)
    q = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(q, s, Q[:, k]) for k in range(3)], axis=1)


def _boundary_dist(fm: FeatureModelV2, pts: np.ndarray, Hn: np.ndarray) -> float:
    """Max distance of boundary vertices to the npz surface (tripwire metric)."""
    faces = {}
    for c in Hn:
        # 12 quad faces of a VTK hex (local corner ids per face)
        for f in ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5),
                  (2, 3, 7, 6), (3, 0, 4, 7)):
            key = tuple(sorted(int(c[i]) for i in f))
            faces[key] = faces.get(key, 0) + 1
    bnd_ids = sorted({i for f, n in faces.items() if n == 1 for i in f})
    if not bnd_ids:
        return 0.0
    d, _, _ = fm.surface_nearest(pts[bnd_ids], k=32)
    return float(np.max(d)) if len(d) else 0.0


def _surface_path_fn(fm: FeatureModelV2, stats: dict, max_pull: float = 0.15,
                     gap_mult: float = 1.5, m_dec: int = 9,
                     records: list | None = None):
    """path_fn fallback: project a non-seam block edge onto the npz surface.

    For block edges that lie ON a single labeled patch (hub/shroud/blade) but
    are not npz seam curves, the chord is replaced by the projection of a
    straight resample onto the npz triangulation (synthetic positive
    curve_id so buckets count it as a curved edge).

    Endpoint labels for the decision come from the TRUE snapped corners
    (records of snap_corners_v2): the nearest-triangle label at the corner
    position plus, when the corner sits on a seam curve, both patches
    (label_lo/label_hi) of that seam curve.  If a corner position is not in
    records, only its projected nearest-triangle label is used; the polyline
    interior is never trusted for endpoint labels.

    Multi-patch walk: when the fine projected label sequence leaves the
    endpoint patch set mid-way, the edge is no longer declined.  A run of
    >= 2 fine samples with one foreign label nl, flanked by the SAME allowed
    patch la on both sides, is cut out of the projected path and replaced by
    the arc along the npz seam curve separating la and nl (ground-truth
    feature curve); the projection continues on la before and after.  Any
    ambiguity declines (None -> caller falls back to chord, status quo):
    mixed run labels/flanks, no seam curve for the patch pair (e.g. hub and
    shroud share none), crossing snap farther than max_pull, arc against the
    chord direction, arc covering most of a closed seam loop, walked length
    > WALK_ARC_MULT * chord, or pull bound violated on kept/resampled points.

    Guards (any failure -> None -> caller falls back to chord, status quo):
      1. label consistency: every decision sample's nearest-triangle label
         must be one of the corner-derived endpoint labels (kills hub->shroud
         verticals that would drag across the blade, and inlet-plane edges
         whose chord is already exact).
      2. pull bound: max distance chord->surface <= max_pull.
      3. gap guard: consecutive projected decision points must not jump more
         than gap_mult * chord step (protects against the projection sliding
         around folds onto a different region).
    """
    seam = fm.seam_curves
    pos: dict = {}
    for r in (records or []):
        pos[np.round(np.asarray(r["target"], float), 12).tobytes()] = r

    def endpoint_ok(p: np.ndarray) -> set[int]:
        d, tri, _ = fm.surface_nearest(np.stack([p, p]), k=32)
        labs = {int(fm.surface_tri_label[tri[0]])}
        rec = pos.get(np.round(p, 12).tobytes())
        if rec is not None and int(rec["curve_id"]) >= 0:
            cid = int(rec["curve_id"])
            labs |= {int(seam.label_lo[cid]), int(seam.label_hi[cid])}
        return labs

    def walkable_runs(labs: list[int], ok: set[int]) -> list[tuple]:
        """Interior maximal runs (len >= 2) of one foreign label nl, flanked
        by the same allowed label la on both sides: [(i0, i1, nl, la)]."""
        out = []
        m = len(labs)
        i = 1
        while i < m - 1:
            j = i
            while j + 1 < m - 1 and labs[j + 1] == labs[i]:
                j += 1
            if (j > i and labs[i] not in ok and labs[i] != labs[i - 1]
                    and labs[i] != labs[j + 1]
                    and labs[i - 1] == labs[j + 1] and labs[i - 1] in ok):
                out.append((i, j, labs[i], labs[i - 1]))
            i = j + 1
        return out

    def seam_between(la: int, nl: int) -> list[int]:
        want = {la, nl}
        return [c for c in range(seam.n_curves)
                if {int(seam.label_lo[c]), int(seam.label_hi[c])} == want]

    def snap_window(qw: np.ndarray, c: int):
        """Nearest point on seam curve c within a chord window qw."""
        d, cids, ts, ps = seam.nearest(qw)
        take = cids == c
        if not take.any():
            return None
        k = int(np.argmin(np.where(take, d, np.inf)))
        return float(d[k]), float(ts[k]), ps[k]

    def _walk(p0, p1, n, chord, q, d2, pts, runs):
        """Cut the projected path at the patch crossings and run the span
        along the ground-truth seam curve between the crossing points."""
        if len({(nl, la) for _i0, _i1, nl, la in runs}) > 1:
            return None
        nl, la = runs[0][2], runs[0][3]
        cands = seam_between(la, nl)
        if not cands:
            return None  # patches share no seam curve: decline, no invention
        i0 = min(r[0] for r in runs)
        i1 = max(r[1] for r in runs)
        wl = q[max(0, i0 - 1):min(n, i0 + 2)]
        wr = q[max(0, i1 - 1):min(n, i1 + 2)]
        best = None
        for c in cands:
            left = snap_window(wl, c)
            right = snap_window(wr, c)
            if left is None or right is None:
                continue
            if best is None or left[0] + right[0] < best[0]:
                best = (left[0] + right[0], c, left, right)
        if best is None:
            return None
        _tot, c, (dl, t0, _pl), (dr, t1, _pr) = best
        if max(dl, dr) > max_pull:
            return None
        lo, hi = int(seam.offset[c]), int(seam.offset[c + 1])
        L = float(seam.arclen[hi - 1])
        direct = abs(t1 - t0)
        wrap = bool(seam.closed[c]) and direct > 0.5 * L
        arc_len = (L - direct) if wrap else direct
        if arc_len <= 1e-9 or (bool(seam.closed[c]) and arc_len > 0.9 * L):
            return None
        step = chord / max(n - 1, 1)
        k = int(np.clip(arc_len / step, 8, n))
        if wrap:
            t1w = t1 - (L if t1 > t0 else -L)
            qq = np.mod(np.linspace(t0, t1w, k), L)
        else:
            qq = np.linspace(t0, t1, k)
        sarr = seam.arclen[lo:hi]
        P_arc = np.stack([np.interp(qq, sarr, seam.pts[lo:hi, kk])
                          for kk in range(3)], axis=1)
        if (P_arc[-1] - P_arc[0]) @ (p1 - p0) <= 0:
            return None  # arc would run against the chord direction
        keep = np.ones(n, bool)
        keep[i0:i1 + 1] = False
        if float(d2[keep].max()) > max_pull:
            return None
        P = np.vstack([pts[:i0], P_arc, pts[i1 + 1:]])
        if float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) \
                > WALK_ARC_MULT * chord:
            return None
        R = _resample(P, n)
        R[0], R[-1] = p0, p1  # corners stay the input coordinates
        d3, _, _ = fm.surface_nearest(R, k=32)
        if float(d3.max()) > max_pull:
            return None
        stats["edges_walked_multi_patch"] = \
            stats.get("edges_walked_multi_patch", 0) + 1
        return R, WALK_ID_BASE + stats["edges_walked_multi_patch"]

    def seam_neighbors(ok: set[int]) -> set[int]:
        """Patches directly seam-connected to any allowed patch: a block edge
        may run across the blade O-grid face between hub and shroud without
        being cut at the blade intersection or inventing a new corner."""
        ext = set()
        for la in ok:
            for cur in seam_neighbors_cache.get(la, ()):
                ext |= {int(seam.label_lo[cur]), int(seam.label_hi[cur])}
        return ext

    seam_neighbors_cache: dict[int, list[int]] = {}
    for c in range(seam.n_curves):
        for lab in (int(seam.label_lo[c]), int(seam.label_hi[c])):
            seam_neighbors_cache.setdefault(lab, []).append(c)

    def fn(p0: np.ndarray, p1: np.ndarray, n: int):
        p0 = np.asarray(p0, float)
        p1 = np.asarray(p1, float)
        chord = float(np.linalg.norm(p1 - p0))
        if not np.isfinite(chord) or chord <= 1e-12:
            return None
        ok = (endpoint_ok(p0) | endpoint_ok(p1)) | seam_neighbors(
            endpoint_ok(p0) | endpoint_ok(p1))
        s = np.linspace(p0, p1, m_dec)
        d, tri, proj = fm.surface_nearest(s, k=32)
        clean = {int(x) for x in fm.surface_tri_label[tri]} <= ok
        if clean:
            if float(d.max()) > max_pull:
                return None
            step = chord / (m_dec - 1)
            gaps = np.linalg.norm(np.diff(proj, axis=0), axis=1)
            if gaps.size and float(gaps.max()) > gap_mult * step:
                return None
        q = np.linspace(p0, p1, n)
        d2, tri2, pts = fm.surface_nearest(q, k=32)
        labs2 = [int(x) for x in fm.surface_tri_label[tri2]]
        runs = walkable_runs(labs2, ok)
        if not runs:
            if (not clean or not set(labs2) <= ok
                    or float(d2.max()) > max_pull):
                return None
            pts[0], pts[-1] = p0, p1  # corners stay the input coordinates
            stats["edges_surface_projected"] = \
                stats.get("edges_surface_projected", 0) + 1
            return pts, 900000 + stats["edges_surface_projected"]
        return _walk(p0, p1, n, chord, q, d2, pts, runs)

    return fn


HEX_FACES = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5),
             (2, 3, 7, 6), (3, 0, 4, 7))


def collapsed_faces(blocks, C_snap: np.ndarray) -> int:
    """Number of block faces with fewer than 4 distinct welded corners.

    A collapsed face means a degenerate block (a prism or worse). refill_curved
    cannot orient such a face and raises "keine D4-Orientierung". These are
    exactly the 12 of 692 samples that scripts/clean_base_npz.py already drops
    as degenerate (|signed volume| <= 1e-9), so they are not in the training
    corpus either -- skipping them adds no new restriction.
    """
    import tfi as _tfi
    blocks = np.asarray(blocks, np.int64)
    _, remap = _tfi.weld(np.asarray(C_snap, float).reshape(-1, 3))
    H = remap.reshape(blocks.shape[0], 8)
    n = 0
    for row in H:
        for f in HEX_FACES:
            if len({int(row[i]) for i in f}) < 4:
                n += 1
    return n


def _boundary_edge_pred(blocks, C_snap: np.ndarray):
    """(p0, p1) -> True when the block edge lies on a DOMAIN BOUNDARY face.

    A quad face shared by two blocks is interior; a face owned by exactly one
    block bounds the domain.  Only the edges of those faces may be routed on
    an npz patch -- interior edges run through the volume and must stay
    chords, otherwise they get dragged onto a surface.

    `blocks` is the (nb, 8) corner-index array of the block structure -- GT
    (`fm.blocks`) or transformer-generated alike; nothing here reads GT data.
    """
    blocks = np.asarray(blocks, np.int64)
    key_of = {}
    for r in range(blocks.shape[0]):
        for c in range(8):
            key_of[np.round(C_snap[r, c], 9).tobytes()] = int(blocks[r, c])
    count: dict = {}
    for row in blocks:
        for f in HEX_FACES:
            k = frozenset(int(row[i]) for i in f)
            count[k] = count.get(k, 0) + 1
    bnd_edges = set()
    for row in blocks:
        for f in HEX_FACES:
            if count[frozenset(int(row[i]) for i in f)] != 1:
                continue
            for a, b in zip(f, f[1:] + f[:1]):
                ia, ib = int(row[a]), int(row[b])
                bnd_edges.add((ia, ib) if ia < ib else (ib, ia))

    def pred(p0, p1) -> bool:
        a = key_of.get(np.round(np.asarray(p0, float), 9).tobytes())
        b = key_of.get(np.round(np.asarray(p1, float), 9).tobytes())
        if a is None or b is None:
            return False
        return ((a, b) if a < b else (b, a)) in bnd_edges

    return pred


def main() -> int:
    ap = argparse.ArgumentParser(
        description="conform GT coarse block structure onto npz geometry")
    ap.add_argument("--npz", default=DEFAULT_NPZ)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--target-h", type=float, default=10.0,
                    help=">> max GT edge length => 1x1x1 lattice (coarse)")
    ap.add_argument("--feature-cache", default=os.path.join(ROOT, "data", "features"))
    ap.add_argument("--geodesic", action="store_true",
                    help="route non-seam block edges as shortest paths ON "
                         "their associated npz patch (plan v4, D4). A patch "
                         "path cannot cross the blade footprint, so edges run "
                         "AROUND the blade; no arc/chord guard. Writes "
                         "<prefix>_geodesic_edges.vtk + _routing.json")
    ap.add_argument("--reject-interior-faces", action="store_true",
                    help="try to reject single-owner faces that lie inside the "
                         "domain (block-level T-junctions). OFF by default: by "
                         "the time the test runs, the edges of such a wall have "
                         "already been routed onto the geometry, so it neither "
                         "catches machine_0387_n8000's two T-junctions nor "
                         "avoids false positives on clean samples -- and a "
                         "wrongly rejected face is neither projected nor "
                         "measured, which puts a hole in the conformity number. "
                         "Use scripts/detect_block_tjunctions.py on blocks.vtk "
                         "to gate the dataset instead.")
    ap.add_argument("--no-seam-snap", action="store_true",
                    help="keep the seam navigator's raw path instead of "
                         "snapping it back onto the seam polyline and the npz "
                         "surface (measured 4.9e-03 off both).")
    ap.add_argument("--allow-collapsed", action="store_true",
                    help="attempt samples with collapsed block faces; they "
                         "normally fail in refill_curved with "
                         "'keine D4-Orientierung'.")
    ap.add_argument("--seam-tol", type=float, default=1e-9,
                    help="a corner without a seam record in the snap may only "
                         "be routed along a seam when it lies this close to "
                         "one. The legacy 0.12 fallback started the arc up to "
                         "0.078 away from the block corner, which distorts the "
                         "Coons boundary row; with the geodesic router those "
                         "edges belong on a patch instead.")
    ap.add_argument("--project-faces", action="store_true",
                    help="pull the interior of every domain boundary face onto "
                         "its npz patch before Gordon-Hall (plan v4 S2). Coons "
                         "faces only interpolate their four edges, so on a "
                         "curved patch the interior cuts the chord.")
    ap.add_argument("--blade-clearance", type=float, default=0.0,
                    help="keep geodesic edges this far away from the blade "
                         "patch (label 7). 0 = plain shortest path, which "
                         "hugs the blade root exactly as the GT edges do.")
    ap.add_argument("--clearance-chord-frac", type=float, default=0.25,
                    help="cap the effective blade clearance of an edge at this "
                         "fraction of its own chord; a short edge cannot bow a "
                         "full clearance away without creasing.")
    ap.add_argument("--blend-boundary", type=int, default=0, metavar="N",
                    help="pull routed boundary edges towards the shape of "
                         "their parallel rails, N passes. A shortest path is "
                         "the wrong shape for a block edge; four passes cut "
                         "inverted cells by a third to two thirds with the "
                         "boundary error unchanged. Only edges belonging "
                         "unambiguously to one patch are touched.")
    ap.add_argument("--blend-interior", action="store_true",
                    help="give interior chord edges a shape blended from the "
                         "parallel rails of their direction class (plan v4). "
                         "Boundary edges untouched -- their patch already "
                         "determines them.")
    ap.add_argument("--surface-project", action="store_true",
                    help="project non-seam block edges lying on a single npz "
                         "patch onto that patch; edges crossing onto another "
                         "patch mid-way are walked along the separating seam "
                         "curve (guarded; default off)")
    args = ap.parse_args()

    base = os.path.splitext(os.path.basename(args.npz))[0]
    stem = (os.path.basename(os.path.dirname(os.path.abspath(args.npz)))
            if base == "sample" else base)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, f"{stem}_gt_conform")

    fm = FeatureModelV2(args.npz, cache_dir=args.feature_cache)
    seam = fm.seam_curves
    # Seam-only shim (plan v3): full fm would bypass seams via _gt_edge_curve.
    target = types.SimpleNamespace(curves=seam, surface_nearest=fm.surface_nearest)

    C = fm.vertices[fm.blocks].astype(np.float64)  # (nb, 8, 3) VTK hex order
    nb = int(C.shape[0])

    C_snap, records = snap_corners_v2(target, C, SnapConfigV2())
    from meshtron.geometry import curved_bridge  # noqa: F401  (inserts the hex3d path for tfi)
    curved_bridge._load()
    n_coll = collapsed_faces(fm.blocks, C_snap)
    if n_coll and not args.allow_collapsed:
        print(f"skip {stem}: {n_coll} collapsed block faces (degenerate block; "
              f"clean_base_npz.py drops these samples too). "
              f"Use --allow-collapsed to try anyway.", file=sys.stderr)
        return 3
    tier = {}
    for rec in records:
        tier[rec["tier"]] = tier.get(rec["tier"], 0) + 1

    route_stats = {"routes": 0, "edges_surface_projected": 0,
                   "edges_walked_multi_patch": 0}
    _seam_raw = _seam_path_fn(seam, records, route_stats, tol=args.seam_tol)

    def seam_fn(p0, p1, n):
        res = _seam_raw(p0, p1, n)
        if res is None or args.no_seam_snap:
            return res
        return snap_seam_path(seam, fm, res[0]), res[1]
    surf_fn = (_surface_path_fn(fm, route_stats, records=records)
               if args.surface_project else None)
    is_bnd = _boundary_edge_pred(fm.blocks, C_snap)
    geo = (PatchPaths(fm, records=records, stats=route_stats,
                      is_boundary=is_bnd, clearance=args.blade_clearance,
                      clearance_chord_frac=args.clearance_chord_frac)
           if args.geodesic else None)
    # One record per block edge, in build_structures creation order, so the
    # debug VTK shows ALL 84 edges -- seam, geodesic and chord alike.
    edge_log: list = []

    def path_fn(p0, p1, n):
        p0a, p1a = np.asarray(p0, float), np.asarray(p1, float)
        chord = float(np.linalg.norm(p1a - p0a))
        res = seam_fn(p0, p1, n)
        if res is not None:
            pts = np.asarray(res[0], float)
            arc = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
            edge_log.append({"pts": pts, "kind": 1, "label": -1,
                             "arc_over_chord": arc / chord if chord else 0.0})
            return res
        if geo is not None:
            res = geo(p0, p1, n)
            if res is not None:
                pts = np.asarray(res[0], float)
                edge_log.append({"pts": pts, "kind": 2,
                                 "label": int(geo.debug[-1]["chosen"]),
                                 "arc_over_chord": float(
                                     geo.debug[-1]["arc_over_chord"])})
                return res
            edge_log.append({"pts": np.linspace(p0a, p1a, n), "kind": 0,
                             "label": -1, "arc_over_chord": 1.0})
            return None
        res = surf_fn(p0, p1, n) if surf_fn is not None else None
        pts = (np.asarray(res[0], float) if res is not None
               else np.linspace(p0a, p1a, n))
        arc = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        edge_log.append({"pts": pts, "kind": 3 if res is not None else 0,
                         "label": -1,
                         "arc_over_chord": arc / chord if chord else 0.0})
        return res

    # edge_post_fn also hands us the finished CurvedStructure, so the debug
    # VTK can be written from the FINAL edges (path_fn alone runs before the
    # blend and would show the pre-blend chords).
    final_st: list = []

    def edge_post_fn(st, corner_ids, C_s, blks):
        final_st.append(st)
        if args.blend_boundary and geo is not None:
            blend_boundary_edges(st, corner_ids, C_s, blks, geo,
                                 passes=args.blend_boundary,
                                 stats=route_stats)
        if args.blend_interior:
            blend_chord_edges(st, corner_ids, C_s, blks, stats=route_stats)

    face_fn = (make_face_projector(geo, route_stats)
               if (args.project_faces and geo is not None) else None)

    curved_path = prefix + "_refill.vtk"
    rep = refill_curved(C_snap, args.target_h, curved_path, fm=target,
                        path_fn=path_fn, edge_post_fn=edge_post_fn,
                        face_project_fn=face_fn,
                        is_boundary_face=(make_boundary_face_test(fm)
                                          if args.reject_interior_faces
                                          else None))

    # Boundary conformity tripwire on the exported mesh.
    import export_vtk  # noqa: E402  (curved_bridge inserted the path)
    # re-read exported mesh cheaply: refill returns counts but not points, so
    # rebuild via the same weld path is overkill; instead measure on the VTK.
    # Simplest: re-derive boundary points from rep is not possible -> parse VTK.
    pts_w, Hn = _read_vtk_mesh(curved_path)
    bquads = rep.pop("boundary_quads", None)
    bids = rep.pop("boundary_point_ids", None)
    if bids is not None and len(bids):
        # Boundary from the block topology (robust where cells fold), not from
        # the fine mesh facet count.
        d, _, _ = fm.surface_nearest(pts_w[bids], k=32)
        max_bnd = float(np.max(d))
        bnd_stats = {"n": int(len(bids)), "max": max_bnd,
                     "mean": float(np.mean(d)),
                     "p50": float(np.percentile(d, 50)),
                     "p99": float(np.percentile(d, 99))}
    else:
        max_bnd = _boundary_dist(fm, pts_w, Hn)
        bnd_stats = {"max": max_bnd}

    if bquads is not None and len(bquads):
        # The domain boundary as quads, coloured by distance to the npz
        # surface -- the same point set the gate measures, so the picture and
        # the number cannot disagree.
        d_all, _, _ = fm.surface_nearest(pts_w, k=32)
        bpath = prefix + "_boundary.vtk"
        with open(bpath, "w") as fh:
            fh.write("# vtk DataFile Version 2.0\n"
                     f"meshtron {stem} domain boundary vs npz surface\n"
                     "ASCII\nDATASET UNSTRUCTURED_GRID\n")
            fh.write(f"POINTS {len(pts_w)} double\n")
            for q in pts_w:
                fh.write(f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f}\n")
            fh.write(f"CELLS {len(bquads)} {5 * len(bquads)}\n")
            for q in bquads:
                fh.write(f"4 {q[0]} {q[1]} {q[2]} {q[3]}\n")
            fh.write(f"CELL_TYPES {len(bquads)}\n")
            for _ in bquads:
                fh.write("9\n")
            fh.write(f"POINT_DATA {len(pts_w)}\n"
                     "SCALARS dist_npz_surface double 1\nLOOKUP_TABLE default\n")
            for v in d_all:
                fh.write(f"{float(v):.12e}\n")
        print(f"saved {bpath}  ({len(bquads)} boundary quads, "
              f"scalar dist_npz_surface)")

    compare_path = prefix + "_compare.vtk"
    gt_blocks = [[int(j) for j in b] for b in fm.blocks]
    snapped_v = np.array(fm.vertices, dtype=np.float64)
    snapped_v[fm.blocks] = C_snap  # welded vertex j gets corner C_snap[r,c]
    _write_parts_vtk(compare_path,
                     [(fm.vertices, gt_blocks, 1, 12),
                      (snapped_v, gt_blocks, 2, 12)],
                     f"meshtron {stem} GT conform (1=GT corners, 2=snapped)")

    if edge_log and final_st:
        # Debug artifact: EVERY block edge in its FINAL shape, so nothing
        # looks "missing".  route_kind 0=chord (interior, untouched),
        # 1=seam curve, 2=geodesic on a patch, 3=legacy surface projection,
        # 4=interior chord reshaped by blend_chord_edges.  patch_label is -1
        # unless the edge was routed geodesically.
        st_fin = final_st[0]
        keys = list(st_fin.edge_pts.keys())
        polys, kinds, labels, aoc, kinkd = [], [], [], [], []
        for i, key in enumerate(keys):
            Q = np.asarray(st_fin.edge_pts[key], float)
            rec = edge_log[i] if i < len(edge_log) else {
                "kind": 0, "label": -1, "arc_over_chord": 1.0}
            kind = rec["kind"]
            if int(st_fin.edge_curve.get(key, -1)) >= BLEND_ID_BASE:
                kind = 4
            chord = float(np.linalg.norm(Q[-1] - Q[0]))
            arc = float(np.linalg.norm(np.diff(Q, axis=0), axis=1).sum())
            polys.append(Q)
            kinds.append(int(kind))
            labels.append(int(rec["label"]))
            aoc.append(arc / chord if chord > 0 else 0.0)
            kinkd.append(max_kink_deg(Q))
        write_debug_vtk(prefix + "_edges_debug.vtk", polys,
                        {"route_kind": kinds, "patch_label": labels,
                         "arc_over_chord": aoc, "max_kink_deg": kinkd},
                        f"meshtron {stem} block edge routing (route_kind "
                        f"0=chord 1=seam 2=geodesic 3=surfproj 4=blended)")
    if geo is not None:
        with open(prefix + "_routing.json", "w") as fh:
            json.dump(geo.debug, fh, indent=2)

    kink_all = kinkd if (edge_log and final_st) else []
    snap_moves = np.linalg.norm(C_snap.reshape(-1, 3) - C.reshape(-1, 3), axis=1)
    summary = {
        "npz": args.npz,
        "n_blocks": nb,
        "snap": {"max_move": float(snap_moves.max()),
                 "mean_move": float(snap_moves.mean()),
                 "tier_counts": {str(k): v for k, v in sorted(tier.items())}},
        "seam_routes": route_stats["routes"],
        "surface_projected": {
            "enabled": bool(args.surface_project),
            "edges": int(route_stats["edges_surface_projected"])},
        "multi_patch_walk": {
            "enabled": bool(args.surface_project),
            "edges": int(route_stats["edges_walked_multi_patch"])},
        "single_owner_faces_rejected": int(
            rep.get("single_owner_faces_rejected", 0)),
        "project_faces": {
            "enabled": bool(args.project_faces),
            "faces": int(route_stats.get("faces_projected", 0)),
            "by_label": route_stats.get("faces_projected_labels", {})},
        "kink": {
            "max_deg": float(max(kink_all, default=0.0)),
            "p90_deg": float(np.percentile(kink_all, 90)) if kink_all else 0.0,
            "edges_over_20deg": int(sum(1 for k in kink_all if k > 20.0))},
        "blend_boundary": {
            "passes": int(args.blend_boundary),
            "edges_touched": int(route_stats.get("edges_blend_boundary", 0))},
        "blend_interior": {
            "enabled": bool(args.blend_interior),
            "edges": int(route_stats.get("edges_blended", 0))},
        # Edges routed ON the blade patch itself sit at distance 0 by
        # definition; they are excluded from the clearance statistic.
        "blade_clearance": {
            "value": float(args.blade_clearance),
            "min_median_off_blade_edges": float(min(
                [d["blade_dist_med"] for d in (geo.debug if geo else [])
                 if d.get("blade_dist_med") is not None
                 and d.get("chosen") not in (7,)], default=-1.0))},
        "geodesic": {
            "enabled": bool(args.geodesic),
            "edges": int(route_stats.get("edges_geodesic", 0)),
            "failed": int(sum(1 for d in (geo.debug if geo else [])
                              if d["chosen"] is None)),
            "max_arc_over_chord": float(max(
                [d["arc_over_chord"] for d in (geo.debug if geo else [])
                 if d.get("arc_over_chord") is not None], default=0.0)),
            "max_dist_patch": float(max(
                [d["max_dist_patch"] for d in (geo.debug if geo else [])
                 if d.get("max_dist_patch") is not None], default=0.0))},
        "refill": rep,
        "boundary": bnd_stats,
        "tripwires": {
            "max_boundary_surface_dist": max_bnd,
            "tolerance": BOUNDARY_TOL,
            "pass": bool(max_bnd <= BOUNDARY_TOL and rep["watertight"]),
            "watertight": bool(rep["watertight"]),
        },
        "known_artifact": {
            "inverted_cells_gt_dataset": int(rep["inverted_curved"]),
            "note": "GT coarse blocks contain folded cells (documented, "
                    "scripts/map_generated_blocks.py:241-244); not a tripwire"},
    }
    summary_path = prefix + "_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"blocks={nb} snap_max_move={snap_moves.max():.6f} "
          f"tiers={summary['snap']['tier_counts']} routes={route_stats['routes']} "
          f"surface_projected={route_stats['edges_surface_projected']} "
          f"walked_multi_patch={route_stats['edges_walked_multi_patch']} "
          f"geodesic={route_stats.get('edges_geodesic', 0)} "
          f"blended={route_stats.get('edges_blended', 0)}")
    print(f"refill: watertight={rep['watertight']} cells={rep['cells_after']} "
          f"inverted={rep['inverted_curved']} (known GT dataset artifact) "
          f"buckets={rep.get('buckets')}")
    print(f"tripwire boundary_dist={max_bnd:.3e} (tol {BOUNDARY_TOL:.0e}) "
          f"pass={summary['tripwires']['pass']}")
    print(f"saved {curved_path}")
    print(f"saved {rep.get('out_edges_vtk')}")
    print(f"saved {compare_path}  (part: 1=GT corners, 2=snapped corners)")
    print(f"saved {summary_path}")
    return 0 if summary["tripwires"]["pass"] else 2


def _read_vtk_mesh(path: str):
    """Minimal ASCII VTK UNSTRUCTURED_GRID reader -> (points, hexa connectivity)."""
    with open(path) as fh:
        lines = fh.read().splitlines()
    i = 0
    pts = None
    cells = None
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
            cells = []
            for _ in range(n):
                parts = lines[j].split()
                if int(parts[0]) == 8:
                    cells.append([int(p) for p in parts[1:]])
                j += 1
            i = j
            continue
        i += 1
    if pts is None or cells is None:
        raise RuntimeError(f"VTK parse failed (points/cells missing): {path}")
    return pts, np.asarray(cells, dtype=np.int64)


if __name__ == "__main__":
    raise SystemExit(main())
