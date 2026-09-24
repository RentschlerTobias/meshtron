"""curved_bridge.py — gesnappte Bloecke + Kurven -> gekruemmtes TFI-Volumen.

Import-only gegenueber `hex3d_algohex` (`tfi`, `export_vtk`). Aufbau:
  weld -> build_topology -> 1x1x1-Lattice -> direction_classes
       -> solve_block_divisions(target_h)  (wie tfi_bridge, damit identische
       Divisionszahlen/Conformity)
  je Block: 6 Randflaechen kanonisch als Coons aus den Kurvenkanten
       (edge_curves), Innenraum per Gordon-Hall (`tfi.tfi`)
  -> weld -> check_watertight -> scaled_jacobian -> export_vtk

`refill_cfd` (tfi_bridge, Chord-Modus) bleibt der benannte Fallback. Der Chord-
Vergleich wird im selben Report mitgefuehrt (invertierte Zellen Seite an Seite).
"""
from __future__ import annotations

import os
import sys

import numpy as np

from meshtron.geometry.edge_curves import (OTHER, build_structures, face_cycle, local_edges,
                         orient_face)

HEX3D_REPO = ("/home/t1dde/hydrostack_pipeline/stack/domain_partition_3D/"
              "experimentell/hex3d_algohex")


def _load():
    if HEX3D_REPO not in sys.path:
        sys.path.insert(0, HEX3D_REPO)
    import clean_blocks as cb
    import export_vtk as ev
    import tfi
    import base_complex as bc
    return tfi, ev, bc, cb


def _lattice_vert(row, corner):
    v = np.empty((2, 2, 2), dtype=np.int64)
    for li, (di, dj, dk) in enumerate(corner):
        v[di, dj, dk] = row[li]
    return v


def _classes_counts(tfi, bc, lat, H, B, P, target_h):
    """(classes, counts, mode, reason) — identisch zur tfi_bridge-Logik."""
    f2h, _ = bc.build_topology(H)
    try:
        classes = tfi.direction_classes(lat, f2h, H, B, verbose=False)
        counts = tfi.solve_block_divisions(lat, classes, P, target_h,
                                           verbose=False)
        linked = any(len(c) > 1 for c in classes)
        if linked:
            return f2h, classes, counts, "conforming", None
        return f2h, classes, counts, "independent_fallback", "no face-sharing links"
    except Exception as exc:
        classes = [[(r, ax)] for r in sorted(lat) for ax in (0, 1, 2)]
        counts = tfi.solve_block_divisions(lat, classes, P, target_h, verbose=False)
        return (bc.build_topology(H)[0], classes, counts, "independent_fallback",
                f"{type(exc).__name__}: {exc}")


def _block_curved_mesh(tfi, r, dims, H, C_snap, st, pts_of,
                       face_project_fn=None, bnd_faces=None):
    """Ein Block: 6 Coons-Randflaechen + Gordon-Hall-Innenraum.

    `face_project_fn(Gloc, ids, pts_of) -> Gloc` pulls the INTERIOR of a domain
    boundary face onto its geometry patch before the volume is filled: a Coons
    patch only interpolates its four curve edges, so on a curved patch its
    interior cuts the chord.  That is the missing "faces" level of the
    corners -> edges -> faces -> volume hierarchy.  Corners and edges stay
    untouched, so neighbouring blocks still match exactly.
    """
    ni, nj, nk = dims
    X = np.zeros((ni + 1, nj + 1, nk + 1, 3))
    for axis in (0, 1, 2):
        for side in (0, 1):
            cyc = face_cycle(axis, side)
            ids = [int(H[r, c]) for c in cyc]
            G, cids = st.faces[frozenset(ids)]
            o0, o1 = OTHER[axis]
            shp = (dims[o0] + 1, dims[o1] + 1)
            Gloc = orient_face(G, cids, ids, pts_of, shp)
            if (face_project_fn is not None and bnd_faces is not None
                    and frozenset(ids) in bnd_faces):
                Gloc = face_project_fn(Gloc, ids, pts_of)
            sl = [slice(None)] * 3
            sl[axis] = 0 if side == 0 else -1
            X[tuple(sl)] = Gloc
    return tfi.tfi(X)


def refill_curved(C_snap: np.ndarray, target_h: float, out_vtk: str,
                  blocks: np.ndarray | None = None, fm=None,
                  write_edges: bool = True, path_fn=None,
                  edge_post_fn=None, face_project_fn=None,
                  is_boundary_face=None) -> dict:
    """(nb,8,3) gesnappte Ecken (+ Kurvenmodell `fm`) -> CFD-VTK, Kurven-TFI.

    `path_fn(p0, p1, n) -> (pts, curve_id) | None` forwards the seam-graph
    fallback to build_structures/sample_on_curve (default None: legacy)."""
    tfi, ev, bc, cb = _load()
    C = np.asarray(C_snap, dtype=np.float64)
    if C.ndim != 3 or C.shape[1:] != (8, 3):
        raise ValueError(f"corners shape {C.shape} != (nb,8,3)")
    nb = C.shape[0]
    if blocks is None:
        blocks = np.arange(nb * 8).reshape(nb, 8)
    P, remap = tfi.weld(C.reshape(-1, 3))
    H = remap.reshape(nb, 8)
    B = np.arange(nb, dtype=np.int64)
    lat = {r: (np.ones(3, int), _lattice_vert(H[r], tfi.CORNER))
           for r in range(nb)}
    f2h, classes, counts, mode, reason = _classes_counts(tfi, bc, lat, H, B, P,
                                                         target_h)
    report: dict = {"mode": mode, "curved": True, "fallback_reason": reason,
                    "n_blocks": nb, "target_h": float(target_h),
                    "n_points_welded": int(len(P)), "n_classes": len(classes)}
    if fm is None:
        raise ValueError("refill_curved braucht ein FeatureModelV2 (fm=...)")

    from tfi import class_of_axis
    cof = class_of_axis(classes)
    st = build_structures(fm, H, C, blocks, counts, cof, path_fn=path_fn,
                          edge_post_fn=edge_post_fn)

    # Domain boundary from the BLOCK topology: a block face owned by exactly
    # one block bounds the domain.  Deriving it from the fine mesh instead is
    # unreliable where cells fold -- collapsed points there make interior
    # facets look like boundary facets.
    #
    # One owner is NECESSARY but not sufficient: where two blocks touch across
    # only part of a side (a T-junction, present in 17% of the corpus), the
    # four-corner face format cannot express the interface, so both sides are
    # recorded with one owner each while lying INSIDE the domain. Projecting
    # those onto the geometry drags interior walls outward. `is_boundary_face`
    # (ids, grid) -> bool lets the caller reject them; without it the old
    # behaviour is kept.
    face_owners: dict = {}
    for r in range(nb):
        for axis in (0, 1, 2):
            for side in (0, 1):
                key = frozenset(int(H[r, c]) for c in face_cycle(axis, side))
                face_owners[key] = face_owners.get(key, 0) + 1

    bnd_faces = {k for k, v in face_owners.items() if v == 1}
    interior_single = 0
    if is_boundary_face is not None:
        keep = set()
        for r in range(nb):
            for axis in (0, 1, 2):
                for side in (0, 1):
                    ids = [int(H[r, c]) for c in face_cycle(axis, side)]
                    key = frozenset(ids)
                    if key not in bnd_faces or key in keep:
                        continue
                    G, cids = st.faces[key]
                    if is_boundary_face(ids, G):
                        keep.add(key)
        interior_single = len(bnd_faces) - len(keep)
        bnd_faces = keep
    report["single_owner_faces_rejected"] = int(interior_single)

    chunks, cells, bid, bnd_ids, bnd_quads = [], [], [], [], []
    pts_of = {int(H[r, c]): C[r, c] for r in range(nb) for c in range(8)}
    for r in range(nb):
        Xi = _block_curved_mesh(tfi, r, st.dims[r], H, C, st, pts_of,
                                face_project_fn=face_project_fn,
                                bnd_faces=bnd_faces)
        base = sum(len(c) for c in chunks)
        ids = np.arange(base, base + Xi[..., 0].size).reshape(Xi.shape[:3])
        for axis in (0, 1, 2):
            for side in (0, 1):
                key = frozenset(int(H[r, c]) for c in face_cycle(axis, side))
                if key not in bnd_faces:
                    continue
                sl = [slice(None)] * 3
                sl[axis] = 0 if side == 0 else -1
                grid = ids[tuple(sl)]
                bnd_ids.append(grid.reshape(-1))
                bnd_quads.append(np.stack([grid[:-1, :-1], grid[1:, :-1],
                                           grid[1:, 1:], grid[:-1, 1:]],
                                          axis=-1).reshape(-1, 4))
        chunks.append(Xi.reshape(-1, 3))
        cells.append(tfi.block_cells(ids))
        bid.append(np.full(len(cells[-1]), r, int))
    pts = np.vstack(chunks)
    Hn = np.vstack(cells)
    Bn = np.concatenate(bid)
    pts_w, remap2 = tfi.weld(pts)
    Hn = remap2[Hn]
    boundary_ids = (np.unique(remap2[np.concatenate(bnd_ids)])
                    if bnd_ids else np.empty(0, np.int64))
    boundary_quads = (remap2[np.vstack(bnd_quads)] if bnd_quads
                      else np.empty((0, 4), np.int64))
    ok, bnd = tfi.check_watertight(pts_w, Hn, verbose=False)
    sj = cb.scaled_jacobians(pts_w, Hn)
    report.update({"cells_after": int(len(Hn)), "points_after": int(len(pts_w)),
                   "watertight": bool(ok), "boundary_faces": int(bnd),
                   "inverted_curved": int((sj <= 0).sum()),
                   "min_scaled_jacobian": float(sj.min())})
    ev.write_vtk(out_vtk, pts_w, Hn, [12] * len(Hn), Bn, "block_id",
                 "meshtron generated->CFD (curved Coons+TFI)")
    report["out_vtk"] = os.path.abspath(out_vtk)
    if write_edges:
        epath = os.path.splitext(out_vtk)[0] + "_edges.vtk"
        write_curved_edges_vtk(epath, st)
        report["out_edges_vtk"] = os.path.abspath(epath)
    report["buckets"] = _bucket_hist(st)
    report["boundary_point_ids"] = boundary_ids
    report["boundary_quads"] = boundary_quads
    return report


def chord_baseline(C_snap: np.ndarray, target_h: float, out_vtk: str) -> dict:
    """Chord-Modus ueber tfi_bridge (benannter Fallback) + invertierte Zellen."""
    from meshtron.geometry.tfi_bridge import refill_cfd
    rep = refill_cfd(C_snap, target_h, out_vtk)
    tfi, ev, bc, cb = _load()
    import meshio
    m = meshio.read(out_vtk)
    H = np.vstack([b.data for b in m.cells if b.type == "hexahedron"])
    P = np.asarray(m.points, float)
    sj = cb.scaled_jacobians(P, H)
    rep["inverted_chord"] = int((sj <= 0).sum())
    rep["min_scaled_jacobian"] = float(sj.min())
    return rep


def _bucket_hist(st) -> dict:
    one = sum(1 for v in st.edge_curve.values() if v >= 0)
    deg = sum(1 for v in st.edge_curve.values() if v < 0)
    return {"edges_1d_curve": int(one), "edges_degenerate_chord": int(deg),
            "faces_coons": int(len(st.faces)),
            "edges_total": int(len(st.edge_curve))}


def write_curved_edges_vtk(path: str, st) -> None:
    """Gewaehlte Kurvenkanten als Polyline-Zellen (VTK-Typ 4)."""
    pts, cells = [], []
    for key, Q in st.edge_pts.items():
        base = len(pts)
        pts.extend(np.asarray(Q, float).tolist())
        cells.append((base, len(Q)))
    with open(path, "w") as fh:
        fh.write("# vtk DataFile Version 2.0\ncurved block edges\n")
        fh.write("ASCII\nDATASET UNSTRUCTURED_GRID\n")
        fh.write(f"POINTS {len(pts)} double\n")
        for p in pts:
            fh.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        fh.write(f"CELLS {len(cells)} {sum(n + 1 for _b, n in cells)}\n")
        for b, n in cells:
            fh.write(f"{n} " + " ".join(str(b + i) for i in range(n)) + "\n")
        fh.write(f"CELL_TYPES {len(cells)}\n")
        for _ in cells:
            fh.write("4\n")
