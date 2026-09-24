"""conform.py — put a coarse hexa block structure onto the real npz geometry.

This is the mapping stage as a library, so the two callers share one
implementation instead of two copies of the same router wiring:

  scripts/conform_gt_blocks.py   ground-truth blocks of one sample (the gate)
  scripts/infer.py               blocks the transformer generated

Corners come in already snapped (block_mapping.snap_corners_v2). What happens
here is everything between those corners and a CFD mesh:

  1. every block edge gets a route:  npz seam curve  >  geodesic on one
     labeled patch  >  projection onto the surface  >  chord
  2. routed boundary edges are pulled towards the shape of their parallel
     rails (blend_boundary_edges), which is what keeps the first cell layer
     at the blade from folding
  3. boundary faces are projected onto their patch
  4. transfinite interpolation fills the volume (curved_bridge.refill_curved)
  5. the exported mesh is measured against the npz surface -- the tripwire,
     because a mesh that misses the geometry is worthless however valid its
     cells are

Only edges on the domain boundary are routed. An interior edge has no patch
to lie on, and routing it anyway moved the boundary error to 0.311.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from meshtron.geometry.curved_bridge import refill_curved  # noqa: E402
from meshtron.geometry.geometry_features import FeatureModelV2  # noqa: E402
from meshtron.geometry.patch_paths import (  # noqa: E402
    BLEND_ID_BASE, PatchPaths, blend_boundary_edges, blend_chord_edges,
    make_boundary_face_test, make_face_projector, max_kink_deg,
    snap_seam_path, write_debug_vtk)
from meshtron.geometry.seam_graph import SeamNavigator  # noqa: E402

HEX_FACES = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5),
             (2, 3, 7, 6), (3, 0, 4, 7))
WALK_ARC_MULT = 1.5      # walked path may exceed the chord by this factor
WALK_ID_BASE = 910000    # synthetic curve_id for multi-patch walked edges


def _arc_targets(seam, records):
    """(curve_id, t) per snap record, or (-1, 0.0) when it is not on a curve."""
    from scripts.snap_selftest import _arc_targets as _impl
    return _impl(seam, records)


def seam_path_fn(seam, records, stats, tol=1e-9):
    """Route a block edge along the npz seam curves it connects.

    Keyed by the exact snapped corner positions. tol is deliberately tiny:
    a loose tol (0.12 was the old default) routes corners up to 7.8e-02 away
    onto seams they do not lie on, which is how 24 of 25 samples used to fail
    the conformity gate.
    """
    arc = _arc_targets(seam, records)
    corner_ts: dict[int, list[float]] = defaultdict(list)
    pos: dict[bytes, tuple[int, float]] = {}
    for (cid, t), r in zip(arc, records):
        if cid < 0:
            continue
        corner_ts[cid].append(t)
        pos[np.round(np.asarray(r["target"], float), 12).tobytes()] = (cid, t)
    nav = SeamNavigator(seam, dict(corner_ts))

    def target_of(p):
        p = np.asarray(p, float)
        hit = pos.get(np.round(p, 12).tobytes())
        if hit is not None:
            return hit
        d, c, t, _ = seam.nearest(p[None])
        if float(d[0]) > tol:
            return None
        cid, t = int(c[0]), float(t[0])
        nav.add_point(cid, t)
        return cid, t

    def path_fn(p0, p1, n):
        a, b = target_of(p0), target_of(p1)
        if a is None or b is None:
            return None
        res = nav.path_between(p0, a[0], a[1], p1, b[0], b[1], n)
        if res is not None:
            stats["routes"] += 1
        return res
    return path_fn


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


@dataclass
class ConformOptions:
    """Mapping knobs. The defaults are the configuration the measurements in
    docs/decisions/2026-09-23-blocking-geometry-mapping.md were taken with:
    geodesic routing on, faces projected, four blend passes, seams only for
    corners that actually lie on them.

    Off by default because measurement said so, not by oversight:
      blade_clearance      pushing edges off the blade hurt in general
      blend_interior       reshaping interior chords hurt
      surface_project      superseded by the geodesic router
      reject_interior_faces  its ray cast has false positives on faces that
                             lie exactly on the surface
    """
    target_h: float = 0.05
    geodesic: bool = True
    project_faces: bool = True
    blend_boundary: int = 4
    seam_tol: float = 1e-9
    no_seam_snap: bool = False
    blade_clearance: float = 0.0
    clearance_chord_frac: float = 0.0
    blend_interior: bool = False
    surface_project: bool = False
    reject_interior_faces: bool = False


def collapsed_blocks_ok(blocks, C_snap: np.ndarray) -> int:
    """Alias kept for callers that only want the count."""
    return collapsed_faces(blocks, C_snap)


def conform_blocks(fm: FeatureModelV2, blocks, C_snap: np.ndarray,
                   records: list, prefix: str, opt: ConformOptions,
                   stem: str = "sample", write_debug: bool = True) -> dict:
    """Route the block edges, fill the volume, measure the result.

    fm       the npz feature model (geometry)
    blocks   (nb, 8) block topology, welded vertex ids
    C_snap   (nb, 8, 3) corners already snapped onto the geometry
    records  snap_corners_v2 records for those corners
    prefix   output path prefix; writes <prefix>_refill.vtk and friends

    Returns the refill report plus 'boundary' (distance of the domain
    boundary to the npz surface), 'route_stats' and the written paths.
    refill_curved only ever sees the seam-only shim, never the full fm --
    handing it the full model lets it snap to GT block edge curves, which
    would make the conformity number meaningless.
    """
    import types

    seam = fm.seam_curves
    target = types.SimpleNamespace(curves=seam,
                                   surface_nearest=fm.surface_nearest)
    route_stats = {"routes": 0, "edges_surface_projected": 0,
                   "edges_walked_multi_patch": 0}
    _seam_raw = seam_path_fn(seam, records, route_stats, tol=opt.seam_tol)

    def seam_fn(p0, p1, n):
        res = _seam_raw(p0, p1, n)
        if res is None or opt.no_seam_snap:
            return res
        return snap_seam_path(seam, fm, res[0]), res[1]

    surf_fn = (_surface_path_fn(fm, route_stats, records=records)
               if opt.surface_project else None)
    is_bnd = _boundary_edge_pred(blocks, C_snap)
    geo = (PatchPaths(fm, records=records, stats=route_stats,
                      is_boundary=is_bnd, clearance=opt.blade_clearance,
                      clearance_chord_frac=opt.clearance_chord_frac)
           if opt.geodesic else None)
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

    final_st: list = []

    def edge_post_fn(st, corner_ids, C_s, blks):
        final_st.append(st)
        if opt.blend_boundary and geo is not None:
            blend_boundary_edges(st, corner_ids, C_s, blks, geo,
                                 passes=opt.blend_boundary, stats=route_stats)
        if opt.blend_interior:
            blend_chord_edges(st, corner_ids, C_s, blks, stats=route_stats)

    face_fn = (make_face_projector(geo, route_stats)
               if (opt.project_faces and geo is not None) else None)
    curved_path = prefix + "_refill.vtk"
    rep = refill_curved(C_snap, opt.target_h, curved_path, fm=target,
                        path_fn=path_fn, edge_post_fn=edge_post_fn,
                        face_project_fn=face_fn,
                        is_boundary_face=(make_boundary_face_test(fm)
                                          if opt.reject_interior_faces
                                          else None))

    pts_w, Hn = _read_vtk_mesh(curved_path)
    bquads = rep.pop("boundary_quads", None)
    bids = rep.pop("boundary_point_ids", None)
    if bids is not None and len(bids):
        d, _, _ = fm.surface_nearest(pts_w[bids], k=32)
        bnd = {"n": int(len(bids)), "max": float(np.max(d)),
               "mean": float(np.mean(d)),
               "p50": float(np.percentile(d, 50)),
               "p99": float(np.percentile(d, 99))}
    else:
        bnd = {"max": _boundary_dist(fm, pts_w, Hn)}

    out = {**rep, "boundary": bnd, "route_stats": dict(route_stats),
           "out_vtk": curved_path}
    if bquads is not None and len(bquads) and write_debug:
        out["out_boundary_vtk"] = _write_boundary_vtk(
            prefix + "_boundary.vtk", fm, pts_w, bquads, stem)
    out["kink_deg"] = []
    if write_debug and edge_log and final_st:
        path_dbg, out["kink_deg"] = _write_edge_debug(
            prefix + "_edges_debug.vtk", final_st[0], edge_log, stem)
        out["out_edges_debug_vtk"] = path_dbg
    out["routing_debug"] = geo.debug if geo is not None else []
    return out


def _write_boundary_vtk(path, fm, pts_w, bquads, stem) -> str:
    """The domain boundary as quads, coloured by distance to the npz surface,
    over the same point set the gate measures -- so picture and number cannot
    disagree."""
    d_all, _, _ = fm.surface_nearest(pts_w, k=32)
    with open(path, "w") as fh:
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
    return path


def _write_edge_debug(path, st_fin, edge_log, stem):
    """EVERY block edge in its FINAL shape, so nothing looks 'missing'.
    route_kind 0=chord (interior, untouched), 1=seam curve, 2=geodesic on a
    patch, 3=legacy surface projection, 4=interior chord reshaped by
    blend_chord_edges."""
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
    write_debug_vtk(path, polys,
                    {"route_kind": kinds, "patch_label": labels,
                     "arc_over_chord": aoc, "max_kink_deg": kinkd},
                    f"meshtron {stem} block edge routing (route_kind "
                    f"0=chord 1=seam 2=geodesic 3=surfproj 4=blended)")
    return path, kinkd
