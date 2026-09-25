"""polytron_tfi.py -- a Polytron block structure straight into a CFD mesh.

The Polytron emits what the transfinite fill needs and nothing it has to
guess: corners, the hex topology as pointers, and a curve for every block
edge. So the fill is `refill_curved` with a `path_fn` that hands back the
model's own edge instead of routing one over the geometry:

    edges  (Bezier, sampled by arc length)
      -> faces   Coons patch from the four edges, built once per shared face
      -> volume  Gordon-Hall per block, then welded

Divisions per direction class come from `solve_block_divisions`, exactly as
for every other refill in this repo, so neighbouring blocks stay conforming.
"""
from __future__ import annotations

import types

import numpy as np

from meshtron.geometry.curved_bridge import refill_curved


def _arclength(Q: np.ndarray, n: int) -> np.ndarray:
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Q, axis=0),
                                                        axis=1))])
    if s[-1] <= 1e-12:
        return np.repeat(Q[:1], n, axis=0)
    q = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(q, s, Q[:, k]) for k in range(3)], axis=1)


class SurfaceProjector:
    """Projection onto the labelled surface, one patch at a time.

    A point is projected onto the patches it belongs to: one patch for a
    face interior, two for a seam edge, up to three at a corner, where the
    alternating projection converges onto the patches' common curve or point.
    Unrestricted nearest-point projection would let a hub face slide onto the
    blade where the two meet."""

    def __init__(self, SP, ST, SL, move_tol: float = 0.1, relax_iters: int = 30,
                 face_tol: float = 0.03):
        from meshtron.geometry.patch_paths import Patch
        SP, ST, SL = (np.asarray(SP, float), np.asarray(ST, np.int64),
                      np.asarray(SL, np.int64))
        self.labels = [int(l) for l in np.unique(SL)]
        self.patches = {l: Patch(SP, ST[SL == l], l) for l in self.labels}
        self.move_tol = float(move_tol)
        self.face_tol = float(face_tol)
        self.relax_iters = int(relax_iters)
        # candidate triangles per query: the blade hull carries long slivers
        # whose centroid is far from a point that lies on them; 16 missed one
        self.k = 48

    def dists(self, q):
        """[n,3] -> [n, n_labels] distance to every patch."""
        q = np.asarray(q, float).reshape(-1, 3)
        return np.stack([self.patches[l].project(q)[0] for l in self.labels], 1)

    def onto(self, q, labs, rounds: int = 8):
        q = np.asarray(q, float).reshape(-1, 3).copy()
        labs = list(labs)
        if len(labs) == 1:
            return self.patches[labs[0]].project(q, k=self.k)[1]
        for _ in range(rounds):
            for l in labs:
                q = self.patches[l].project(q, k=self.k)[1]
        return q

    def relax_face(self, G, lab):
        out = np.array(G, float, copy=True)
        if out.shape[0] < 3 or out.shape[1] < 3:
            return out
        patch = self.patches[lab]
        sh = (out.shape[0] - 2, out.shape[1] - 2, 3)
        out[1:-1, 1:-1] = patch.project(out[1:-1, 1:-1], k=self.k)[1].reshape(sh)
        for _ in range(self.relax_iters):
            lap = 0.25 * (out[:-2, 1:-1] + out[2:, 1:-1]
                          + out[1:-1, :-2] + out[1:-1, 2:])
            out[1:-1, 1:-1] = 0.5 * out[1:-1, 1:-1] + 0.5 * lap
            out[1:-1, 1:-1] = patch.project(out[1:-1, 1:-1], k=self.k)[1].reshape(sh)
        return out


def _boundary_labels(V, blocks, curves, proj: SurfaceProjector):
    """Which patch every single-owner block face lies on, or None when it
    sits too far from every patch to be domain boundary (a T-junction wall).

    Candidates come from the corners (labels within slack of the nearest
    patch, intersected over the four); the face midpoint, taken from the
    model's own edge curves, decides among them."""
    from meshtron.geometry.edge_curves import face_cycle
    owners = {}
    cyc_of = {}
    for b in blocks:
        for axis in (0, 1, 2):
            for side in (0, 1):
                ids = [int(b[c]) for c in face_cycle(axis, side)]
                key = frozenset(ids)
                owners[key] = owners.get(key, 0) + 1
                cyc_of[key] = ids
    vd = proj.dists(V)
    near = {}
    for i, row in enumerate(vd):
        near[i] = {proj.labels[j] for j in np.where(row <= row.min() + 0.02)[0]}
    face_lab = {}
    for key, n in owners.items():
        if n != 1:
            continue
        ids = cyc_of[key]
        mids = []
        for a, b in zip(ids, ids[1:] + ids[:1]):
            Q = curves.get((min(a, b), max(a, b)))
            mids.append(Q[len(Q) // 2] if Q is not None else 0.5 * (V[a] + V[b]))
        probe = np.vstack([np.mean(mids, 0)[None], np.asarray(mids)])
        pd = proj.dists(probe).mean(0)
        cands = set.intersection(*(near[i] for i in ids))
        pool = [j for j, l in enumerate(proj.labels) if l in cands] or \
            list(range(len(proj.labels)))
        j = min(pool, key=lambda j: pd[j])
        face_lab[key] = proj.labels[j] if pd[j] <= proj.face_tol else None
    return face_lab, cyc_of


def refill_polytron(V: np.ndarray, blocks: np.ndarray, curves: dict,
                    target_h: float, out_vtk: str,
                    write_edges: bool = True, surface=None,
                    move_tol: float = 0.1, relax_iters: int = 0,
                    edge_resample: bool = False, face_tol: float = 0.03) -> dict:
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


def read_hex_vtk(path: str):
    """(points [N,3], hexes [C,8]) of a refill VTK."""
    import meshio
    m = meshio.read(path)
    H = np.vstack([b.data for b in m.cells if b.type == "hexahedron"])
    return np.asarray(m.points, float), np.asarray(H, np.int64)


def _point_to_tris(q, A, B, C):
    """Exact distance from each q to the nearest of the triangles (A,B,C)."""
    from scipy.spatial import cKDTree
    from meshtron.geometry.curved_bridge import HEX3D_REPO
    import sys
    if HEX3D_REPO not in sys.path:
        sys.path.insert(0, HEX3D_REPO)
    from clean_blocks import _closest_point_on_tris
    cen = (A + B + C) / 3.0
    _, cand = cKDTree(cen).query(q, k=min(48, len(cen)))
    cand = np.atleast_2d(cand)
    d2, _ = _closest_point_on_tris(q, A[cand], B[cand], C[cand])
    return np.sqrt(d2.min(axis=1))


def surface_fit(P: np.ndarray, boundary_ids: np.ndarray, boundary_quads: np.ndarray,
                SP: np.ndarray, ST: np.ndarray, uncovered_tol: float = 0.02) -> dict:
    """Two questions, kept apart because they fail independently:

    on_surface  how far each mesh boundary point is from the labelled surface
    coverage    how far each surface triangle centroid is from the mesh
                boundary (its quads); a blocking that misses the blade passes
                the first and fails this

    Both are exact point-to-triangle distances, so neither depends on the
    mesh density.
    """
    SP = np.asarray(SP, float)
    ST = np.asarray(ST, np.int64)
    Q = np.asarray(boundary_quads, np.int64)
    d_on = _point_to_tris(P[np.asarray(boundary_ids, np.int64)],
                          SP[ST[:, 0]], SP[ST[:, 1]], SP[ST[:, 2]])
    T = np.concatenate([Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]])
    d_cov = _point_to_tris(SP[ST].mean(axis=1), P[T[:, 0]], P[T[:, 1]], P[T[:, 2]])
    return {"on_surface_median": float(np.median(d_on)),
            "on_surface_p95": float(np.percentile(d_on, 95)),
            "on_surface_max": float(d_on.max()),
            "coverage_median": float(np.median(d_cov)),
            "coverage_p95": float(np.percentile(d_cov, 95)),
            "coverage_max": float(d_cov.max()),
            "uncovered_share": float((d_cov > uncovered_tol).mean())}


def mesh_candidate(seq, spec, raw: dict, target_h: float, out_vtk: str,
                   project: bool = True, write_edges: bool = True) -> dict:
    """One Polytron sequence -> filled mesh + every number needed to rank it.
    `raw` needs surface_points / surface_tris / surface_tri_label. Never
    raises: a sequence the fill cannot take comes back with tfi=None."""
    from meshtron.data.polytron_blocks import decode_seq
    from meshtron.training.polytron_sample import structure_report
    r = {"structure": structure_report(seq)}
    V, B, curves = decode_seq(seq, spec, n_edge_pts=64)
    surf = (raw["surface_points"], raw["surface_tris"],
            raw["surface_tri_label"]) if project else None
    import time
    try:
        t0 = time.perf_counter()
        rep = refill_polytron(V, B, curves, target_h, out_vtk,
                              write_edges=write_edges, surface=surf)
        r["seconds_fill"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        P, _H = read_hex_vtk(out_vtk)
        r["tfi"] = {k: rep[k] for k in ("cells_after", "watertight",
                                        "boundary_faces", "inverted_curved",
                                        "min_scaled_jacobian", "mode",
                                        "projection")}
        r["surface"] = surface_fit(P, rep["boundary_point_ids"],
                                   rep["boundary_quads"], raw["surface_points"],
                                   raw["surface_tris"])
        r["seconds_metrics"] = time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001
        r["tfi"] = None
        r["error"] = f"{type(e).__name__}: {e}"
    return r


def rank_key(r: dict):
    """Lower is better. Gates first -- a mesh at all, watertight, manifold
    blocks, the geometry covered -- then the model's own preference (the
    candidate index, which is beam order = likelihood), inverted share last.
    Ranking by inverted cells before likelihood picked a DIFFERENT structure
    than the memorised one in 13 of 40 training items."""
    t = r.get("tfi")
    if t is None:
        return (1, 1, 1, 1e9, 1e9)
    s = r.get("surface") or {}
    st = r["structure"]
    return (0 if t["watertight"] else 1,
            st["nonmanifold_faces"] + st["duplicate_blocks"],
            s.get("uncovered_share", 1.0) > 0.02,
            r.get("cand", 0),
            t["inverted_curved"] / max(1, t["cells_after"]))
