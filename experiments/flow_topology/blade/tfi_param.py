"""Local variant of meshtron.geometry.polytron_tfi.refill_polytron (copied, the repo stays untouched) with
`param_keys`: edges whose polyline is resampled by its own point index instead of arc length, so that point
correspondences survive into the fill (a wall edge = projection of the cut edge point by point; graded radial
edges for wall clustering)."""
import types

import numpy as np

import meshtron.geometry.curved_bridge as _cb
from meshtron.geometry.curved_bridge import refill_curved

_RADIAL = {}                       # block index -> fixed cell count along its local axis 2 (VTK 0 -> 4)
_orig_classes_counts = _cb._classes_counts


def _classes_counts_radial(tfi, bc, lat, H, B, P, target_h):
    f2h, classes, counts, mode, reason = _orig_classes_counts(tfi, bc, lat, H, B, P, target_h)
    if _RADIAL:
        for ci, members in enumerate(classes):
            fixed = [_RADIAL[r] for r, ax in members if ax == 2 and r in _RADIAL]
            if fixed:
                counts[ci] = max(fixed)
    return f2h, classes, counts, mode, reason


_cb._classes_counts = _classes_counts_radial
from meshtron.geometry.polytron_tfi import SurfaceProjector, _arclength, _boundary_labels  # noqa: F401


def refill_polytron(V: np.ndarray, blocks: np.ndarray, curves: dict,
                    target_h: float, out_vtk: str,
                    write_edges: bool = True, surface=None,
                    move_tol: float = 0.1, relax_iters: int = 0,
                    edge_resample: bool = False, face_tol: float = 0.03, param_keys=frozenset(), radial_counts=None) -> dict:
    """V [M,3], blocks [B,8] (VTK order, ids into V), curves {(a,b): [k,3]}
    with a < b (dense polyline from V[a] to V[b]). Missing curve -> chord.

    `surface=(points, tris, labels)` turns on the mapping onto the geometry,
    corners -> edges -> faces, each level before the one it bounds:
      corners  boundary corners projected onto the patches of their faces
      edges    the model's curve follows its moved endpoints (linear blend of
               the two displacements), then its interior is projected onto
               the patch(es) of its boundary faces
      faces    boundary face interiors projected and relaxed on their patch
    and Gordon-Hall fills the volume from those faces. Nothing moves further
    than `move_tol`, and a single-owner face whose edge midpoints sit further
    than `face_tol` from every patch is left alone: that is an interior
    T-junction wall (measured 0.06 to 0.09 off the surface in the corpus),
    not domain boundary."""
    _RADIAL.clear(); _RADIAL.update(radial_counts or {})
    V = np.asarray(V, float).copy()
    blocks = np.asarray(blocks, np.int64)
    curves = {k: np.asarray(v, float) for k, v in curves.items()}
    stats = {"curve": 0, "chord": 0}
    face_lab = {}
    bnd_edges = {}
    proj = None
    if surface is not None:
        proj = SurfaceProjector(*surface, move_tol=move_tol,
                                relax_iters=relax_iters, face_tol=face_tol)
        face_lab, cyc_of = _boundary_labels(V, blocks, curves, proj)
        vlab, elab = {}, {}
        for key, lab in face_lab.items():
            if lab is None:
                continue
            ids = cyc_of[key]
            for a, b in zip(ids, ids[1:] + ids[:1]):
                vlab.setdefault(a, set()).add(lab)
                elab.setdefault((min(a, b), max(a, b)), set()).add(lab)
        V0 = V.copy()
        bnd_edges = {}
        for i, labs in vlab.items():
            q = proj.onto(V[i], labs)[0]
            if np.linalg.norm(q - V[i]) <= move_tol:
                V[i] = q
        for (a, b), Q in list(curves.items()):
            t = np.linspace(0.0, 1.0, len(Q))[:, None]
            Q = Q + (1 - t) * (V[a] - V0[a]) + t * (V[b] - V0[b])
            labs = elab.get((a, b))
            if labs and len(Q) > 2:
                Q[1:-1] = proj.onto(Q[1:-1], labs)
                if edge_resample:
                    Q = _arclength(Q, len(Q))
                    Q[1:-1] = proj.onto(Q[1:-1], labs)
                Q[0], Q[-1] = V[a], V[b]
                bnd_edges[(a, b)] = labs
            curves[(a, b)] = Q
        stats["faces_on_patch"] = sum(l is not None for l in face_lab.values())
        stats["faces_left_alone"] = sum(l is None for l in face_lab.values())
        stats["corners_projected"] = int((np.linalg.norm(V - V0, axis=1) > 0).sum())
    key_of = {np.round(V[i], 9).tobytes(): i for i in range(len(V))}

    def path_fn(p0, p1, n):
        a = key_of.get(np.round(np.asarray(p0, float), 9).tobytes())
        b = key_of.get(np.round(np.asarray(p1, float), 9).tobytes())
        if a is None or b is None:
            return None
        Q = curves.get((a, b))
        if Q is None and (b, a) in curves:
            Q = np.asarray(curves[(b, a)])[::-1]
        if Q is None:
            stats["chord"] += 1
            return np.linspace(p0, p1, n), -1
        if (min(a, b), max(a, b)) in param_keys:      # keep the curve's own parametrisation (point i <-> i)
            Qa = np.asarray(Q, float)
            u = np.linspace(0.0, 1.0, len(Qa)); t = np.linspace(0.0, 1.0, n)
            R = np.stack([np.interp(t, u, Qa[:, d]) for d in range(3)], 1)
        else:
            R = _arclength(np.asarray(Q, float), n)
        labs = bnd_edges.get((min(a, b), max(a, b)))
        if labs and n > 2:          # resampled points sit on chords: back on
            R[1:-1] = proj.onto(R[1:-1], labs)
        R[0], R[-1] = p0, p1
        stats["curve"] += 1
        return R, 800000

    face_fn = None
    if proj is not None:
        def face_fn(G, ids, pts_of):
            # refill_curved numbers corners by its own weld, not by our ids
            own = [key_of.get(np.round(np.asarray(pts_of[int(i)], float), 9)
                              .tobytes()) for i in ids]
            lab = face_lab.get(frozenset(own)) if None not in own else None
            stats["faces_projected"] = stats.get("faces_projected", 0) + (lab is not None)
            return G if lab is None else proj.relax_face(G, lab)

    shim = types.SimpleNamespace(curves=None, surface_nearest=None)
    rep = refill_curved(V[blocks], target_h, out_vtk, fm=shim,
                        path_fn=path_fn, write_edges=write_edges,
                        face_project_fn=face_fn)
    rep["edges_from_model"] = stats.pop("curve")
    rep["edges_chord"] = stats.pop("chord")
    rep["projection"] = stats if proj is not None else None
    return rep


