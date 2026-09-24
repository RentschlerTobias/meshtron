"""patch_paths.py -- geodesic routing of block edges on labelled npz patches.

Replaces the projection+guard machinery of scripts/conform_gt_blocks.py
(_surface_path_fn): instead of projecting a straight chord and rejecting the
result when the projection misbehaves, a block edge is first ASSOCIATED with
one npz label patch and then routed as a shortest path ON that patch.

Why this is different in kind:
  - A path on a patch cannot leave the patch.  The blade footprint is a hole
    in the hub patch, so a hub path runs AROUND the blade by construction.
    Cutting an edge at the blade (the old _walk) produced edges that ran
    THROUGH the blade region; those block structures were invalid.
  - No arc/chord guard.  Running around the blade legitimately makes the arc
    much longer than the chord; the old 1.15*chord guard forbade exactly the
    correct answer.

Pipeline per edge:
  1. candidate labels from both endpoints (nearest triangle + seam labels)
  2. Dijkstra on the patch's triangle-edge graph, per candidate
  3. pick the candidate with the shortest path
  4. smooth the polyline (Laplacian + reprojection onto the same patch)
  5. resample to n points, endpoints pinned to the input corners

Debug artifacts: every routed edge is recorded (candidates, chosen label,
raw/smoothed length, chord, max distance to the patch) and can be dumped as
VTK polylines via write_debug_vtk().
"""
from __future__ import annotations

import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

HEX3D_REPO = ("/home/t1dde/hydrostack_pipeline/stack/domain_partition_3D/"
              "experimentell/hex3d_algohex")

GEODESIC_ID_BASE = 920000


def _closest_point_on_tris():
    if HEX3D_REPO not in sys.path:
        sys.path.insert(0, HEX3D_REPO)
    from clean_blocks import _closest_point_on_tris as fn
    return fn


class Patch:
    """One npz label patch: its triangles, its vertex graph, its projection."""

    def set_clearance(self, dist_to_obstacle: np.ndarray, clearance: float,
                      alpha: float = 8.0) -> None:
        """Re-weight the graph so shortest paths keep away from an obstacle.

        `dist_to_obstacle` is one distance per patch vertex.  An edge costs its
        length times 1 + alpha*(1 - d/clearance)^2 while it is closer than
        `clearance`, so a path bows away from the blade instead of hugging it,
        yet stays the plain shortest path wherever the blade is far.
        """
        if clearance <= 0:
            return
        d = np.asarray(dist_to_obstacle, float)
        m = 1.0 + alpha * np.clip(1.0 - d / clearance, 0.0, 1.0) ** 2
        g = self.graph.tocoo()
        w = g.data * 0.5 * (m[g.row] + m[g.col])
        self.graph = coo_matrix((w, (g.row, g.col)),
                                shape=self.graph.shape).tocsr()
        self.clearance_d = d
        self.clearance = float(clearance)

    def __init__(self, points: np.ndarray, tris: np.ndarray, label: int):
        self.label = int(label)
        self.tris = np.asarray(tris, np.int64)
        vids = np.unique(self.tris)
        self.vids = vids                       # global vertex ids of the patch
        self.g2l = {int(v): i for i, v in enumerate(vids)}
        self.pts = np.asarray(points, float)[vids]   # local vertex coords
        loc = np.vectorize(self.g2l.__getitem__)(self.tris)
        self.ltris = loc
        # Unique undirected edges.  coo_matrix SUMS duplicate entries on
        # tocsr(), so feeding every triangle edge twice (interior edges appear
        # in two triangles, boundary edges in one) would double the weight of
        # interior edges and make Dijkstra hug the patch boundary.
        pairs = np.concatenate([loc[:, [0, 1]], loc[:, [1, 2]], loc[:, [2, 0]]])
        pairs = np.unique(np.sort(pairs, axis=1), axis=0)
        i, j = pairs[:, 0], pairs[:, 1]
        w = np.linalg.norm(self.pts[i] - self.pts[j], axis=1)
        n = len(vids)
        self.graph = coo_matrix(
            (np.concatenate([w, w]),
             (np.concatenate([i, j]), np.concatenate([j, i]))),
            shape=(n, n)).tocsr()
        self.vtree = cKDTree(self.pts)
        self.ttree = cKDTree(self.pts[loc].mean(axis=1))
        self._cpt = _closest_point_on_tris()
        self.clearance = 0.0
        self.clearance_d = None

    # -- projection -------------------------------------------------------
    def project(self, q: np.ndarray, k: int = 16):
        """(...,3) -> (dist, point) onto THIS patch only."""
        q = np.asarray(q, float).reshape(-1, 3)
        _, cand = self.ttree.query(q, k=min(k, len(self.ltris)))
        cand = np.atleast_2d(cand)
        T = self.ltris[cand]
        d2, p = self._cpt(q, self.pts[T[..., 0]], self.pts[T[..., 1]],
                          self.pts[T[..., 2]])
        best = np.argmin(d2, axis=1)
        rows = np.arange(len(q))
        return np.sqrt(d2[rows, best]), p[rows, best]

    # -- shortest path ----------------------------------------------------
    def path(self, p0: np.ndarray, p1: np.ndarray):
        """Dijkstra polyline between the patch vertices nearest to p0 / p1."""
        _, i0 = self.vtree.query(np.asarray(p0, float))
        _, i1 = self.vtree.query(np.asarray(p1, float))
        i0, i1 = int(i0), int(i1)
        if i0 == i1:
            return np.stack([self.pts[i0]]), 0.0
        dist, pred = dijkstra(self.graph, indices=i0, return_predecessors=True)
        if not np.isfinite(dist[i1]):
            return None, np.inf          # different connected components
        chain = [i1]
        while chain[-1] != i0:
            nxt = int(pred[chain[-1]])
            if nxt < 0:
                return None, np.inf
            chain.append(nxt)
        chain.reverse()
        return self.pts[chain], float(dist[i1])

    # -- smoothing --------------------------------------------------------
    def smooth(self, P: np.ndarray, iters: int = 60, w: float = 0.5):
        """Laplacian smoothing with reprojection; endpoints stay fixed."""
        P = np.asarray(P, float).copy()
        if len(P) < 3:
            return P
        for _ in range(iters):
            mid = 0.5 * (P[:-2] + P[2:])
            P[1:-1] = (1.0 - w) * P[1:-1] + w * mid
            _, P[1:-1] = self.project(P[1:-1])
        return P


def max_kink_deg(Q: np.ndarray) -> float:
    """Largest turning angle between consecutive segments, in degrees.

    A crease in a block edge propagates into a bad cell layer through the
    transfinite interpolation, so it is worth reporting per edge."""
    Q = np.asarray(Q, float)
    if len(Q) < 3:
        return 0.0
    d = np.diff(Q, axis=0)
    nn = np.linalg.norm(d, axis=1, keepdims=True)
    u = d / np.maximum(nn, 1e-15)
    c = np.clip(np.einsum('ij,ij->i', u[:-1], u[1:]), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)).max())


def resample(Q: np.ndarray, n: int) -> np.ndarray:
    """Polyline -> n arc-length equidistant points."""
    Q = np.asarray(Q, float)
    if len(Q) == 1:
        return np.repeat(Q, n, axis=0)
    s = np.concatenate([[0.0],
                        np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))])
    if s[-1] <= 1e-12:
        return np.repeat(Q[:1], n, axis=0)
    q = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(q, s, Q[:, k]) for k in range(3)], axis=1)


class PatchPaths:
    """path_fn(p0, p1, n) -> (points, curve_id) | None, geodesic on a patch."""

    def __init__(self, fm, records: list | None = None, smooth_iters: int = 300,
                 stats: dict | None = None, is_boundary=None,
                 clearance: float = 0.0, obstacle_labels=(7,),
                 clearance_alpha: float = 8.0,
                 clearance_chord_frac: float = 0.25):
        self.fm = fm
        self.seam = fm.seam_curves
        self.stats = stats if stats is not None else {}
        self.smooth_iters = int(smooth_iters)
        # Only edges that bound a DOMAIN BOUNDARY face may be routed on a
        # patch.  Interior edges (e.g. the radial hub->shroud edges shared by
        # five blocks) run through the volume; forcing them onto a surface
        # drags the block structure off the geometry.
        self.is_boundary = is_boundary
        self.debug: list[dict] = []
        self.patches: dict[int, Patch] = {}
        labels = np.unique(fm.surface_tri_label)
        for lab in labels:
            sel = fm.surface_tri_label == lab
            self.patches[int(lab)] = Patch(fm.surface_points,
                                           fm.surface_tris[sel], int(lab))
        self.clearance = float(clearance)
        # A short edge cannot bow a full clearance away from the blade without
        # kinking: pushing an edge of chord 0.081 out by 0.06 bends it by 68
        # degrees, and the transfinite interpolation turns that crease into a
        # bad cell layer. The effective clearance is therefore capped at a
        # fraction of the edge's own chord.
        self.clearance_chord_frac = float(clearance_chord_frac)
        self.obstacle = None
        if self.clearance > 0:
            sel = np.isin(fm.surface_tri_label, list(obstacle_labels))
            if sel.any():
                otris = fm.surface_tris[sel]
                self.obstacle = (fm.surface_points, otris,
                                 cKDTree(fm.surface_points[otris].mean(axis=1)))
                for lab, patch in self.patches.items():
                    if lab in obstacle_labels:
                        continue        # the obstacle patch itself: no bias
                    patch.set_clearance(self.obstacle_dist(patch.pts),
                                        self.clearance, clearance_alpha)
        self.rec: dict = {}
        for r in (records or []):
            key = np.round(np.asarray(r["target"], float), 12).tobytes()
            self.rec[key] = r

    def obstacle_dist(self, Q: np.ndarray, k: int = 24, with_point=False):
        """Distance (and optionally closest point) to the obstacle (blade)."""
        Q = np.asarray(Q, float).reshape(-1, 3)
        if self.obstacle is None:
            d = np.full(len(Q), np.inf)
            return (d, Q.copy()) if with_point else d
        P, tris, tree = self.obstacle
        _, c = tree.query(Q, k=min(k, len(tris)))
        c = np.atleast_2d(c)
        tt = tris[c]
        cpt = _closest_point_on_tris()
        d2, pp = cpt(Q, P[tt[..., 0]], P[tt[..., 1]], P[tt[..., 2]])
        best = np.argmin(d2, axis=1)
        rows = np.arange(len(Q))
        d = np.sqrt(d2[rows, best])
        return (d, pp[rows, best]) if with_point else d

    def _smooth_clear(self, patch, P: np.ndarray, iters: int,
                      clearance: float | None = None) -> np.ndarray:
        """Laplacian smoothing that keeps the blade clearance.

        Plain smoothing minimises length and would pull the path straight back
        onto the blade it was routed around, undoing the clearance weighting.
        """
        P = np.asarray(P, float).copy()
        c = self.clearance if clearance is None else float(clearance)
        if len(P) < 3 or c <= 0:
            return patch.smooth(P, iters=iters)
        for _ in range(iters):
            mid = 0.5 * (P[:-2] + P[2:])
            P[1:-1] = 0.5 * P[1:-1] + 0.5 * mid
            _, P[1:-1] = patch.project(P[1:-1])
            d, near = self.obstacle_dist(P[1:-1], with_point=True)
            bad = d < c
            if bad.any():
                v = P[1:-1][bad] - near[bad]
                nv = np.linalg.norm(v, axis=1, keepdims=True)
                v = np.divide(v, np.where(nv > 1e-12, nv, 1.0))
                P[1:-1][bad] += (c - d[bad])[:, None] * v
                _, P[1:-1] = patch.project(P[1:-1])
        return P

    # -- association ------------------------------------------------------
    def endpoint_labels(self, p: np.ndarray) -> set[int]:
        """Labels a corner may belong to: nearest triangle + seam patch pair."""
        d, tri, _ = self.fm.surface_nearest(np.stack([p, p]), k=32)
        labs = {int(self.fm.surface_tri_label[tri[0]])}
        r = self.rec.get(np.round(np.asarray(p, float), 12).tobytes())
        if r is not None and int(r["curve_id"]) >= 0:
            c = int(r["curve_id"])
            labs |= {int(self.seam.label_lo[c]), int(self.seam.label_hi[c])}
        return labs

    def candidates(self, p0, p1) -> list[int]:
        a, b = self.endpoint_labels(p0), self.endpoint_labels(p1)
        both = sorted(a & b)
        return both if both else sorted(a | b)

    def nearest_candidate(self, p0, p1, cands, m: int = 9):
        """Of several candidate patches, the one the EDGE actually lies on.

        Picking by shortest path is wrong when a corner sits slightly off the
        feature it belongs to -- a generated corner near the blade root reads
        as hub, the path then runs on the hub, and the block face is dragged
        away from the blade. Scoring by how far the straight edge is from each
        patch keeps it where it belongs: on machine_0034_n2000's generated
        blocking this took blade-hull triangles with no mesh boundary within
        0.15 from 714 down to 28 and inverted cells from 1157 to 545, while the
        GT blocking stayed identical.
        """
        if len(cands) <= 1:
            return cands
        q = np.linspace(np.asarray(p0, float), np.asarray(p1, float), m)
        score = {}
        for lab in cands:
            patch = self.patches.get(lab)
            if patch is None:
                continue
            d, _ = patch.project(q)
            score[lab] = float(d.max())
        return [min(score, key=score.get)] if score else cands

    # -- routing ----------------------------------------------------------
    def route(self, p0: np.ndarray, p1: np.ndarray, n: int):
        p0 = np.asarray(p0, float)
        p1 = np.asarray(p1, float)
        chord = float(np.linalg.norm(p1 - p0))
        cands = self.nearest_candidate(p0, p1, self.candidates(p0, p1))
        best = None
        tried = {}
        for lab in cands:
            patch = self.patches.get(lab)
            if patch is None:
                continue
            raw, length = patch.path(p0, p1)
            tried[lab] = float(length)
            if raw is None:
                continue
            if best is None or length < best[1]:
                best = (lab, length, raw)
        if best is None:
            return None, {"chord": chord, "candidates": cands, "tried": tried,
                          "chosen": None, "reason": "no path"}
        lab, raw_len, raw = best
        patch = self.patches[lab]
        # The Dijkstra path only fixes the homotopy class (which side of the
        # blade footprint the edge passes).  Smoothing with reprojection then
        # relaxes it to the geodesic INSIDE that class.  Resample first: on the
        # raw vertex chain a Laplacian diffuses far too slowly to straighten a
        # zigzag, which left planar patches at arc/chord 1.6.
        P = np.vstack([p0[None], raw, p1[None]])
        P = resample(P, max(int(n), 40))
        P[0], P[-1] = p0, p1
        c_eff = min(self.clearance, self.clearance_chord_frac * chord)
        if c_eff > 0 and self.obstacle is not None and lab not in (7,):
            P = self._smooth_clear(patch, P, self.smooth_iters, clearance=c_eff)
        else:
            P = patch.smooth(P, iters=self.smooth_iters)
        P[0], P[-1] = p0, p1
        R = resample(P, n)
        # resample interpolates BETWEEN projected points, which lifts the
        # samples off the patch again -> project once more, endpoints pinned.
        _, R[1:-1] = patch.project(R[1:-1])
        R[0], R[-1] = p0, p1
        d, _ = patch.project(R)
        info = {"chord": chord, "candidates": cands, "tried": tried,
                "chosen": lab, "raw_len": float(raw_len),
                "len": float(np.linalg.norm(np.diff(R, axis=0), axis=1).sum()),
                "max_dist_patch": float(d.max()), "reason": None}
        info["arc_over_chord"] = info["len"] / chord if chord > 0 else np.inf
        info["clearance_eff"] = float(c_eff)
        info["max_kink_deg"] = float(max_kink_deg(R))
        if self.obstacle is not None:
            dob = self.obstacle_dist(R)
            info["blade_dist_min"] = float(dob.min())
            info["blade_dist_med"] = float(np.median(dob))
        return R, info

    def __call__(self, p0, p1, n):
        chord = float(np.linalg.norm(np.asarray(p1) - np.asarray(p0)))
        if not np.isfinite(chord) or chord <= 1e-12:
            return None
        if self.is_boundary is not None and not self.is_boundary(p0, p1):
            self.debug.append({"poly": len(self.debug), "chord": chord,
                               "candidates": [], "tried": {}, "chosen": None,
                               "reason": "interior edge"})
            return None
        R, info = self.route(p0, p1, n)
        info["poly"] = len(self.debug)
        self.debug.append(info)
        if R is None:
            return None
        self.stats["edges_geodesic"] = self.stats.get("edges_geodesic", 0) + 1
        return R, GEODESIC_ID_BASE + self.stats["edges_geodesic"]


def write_debug_vtk(path: str, polys: list[np.ndarray],
                    scalars: dict | list,
                    title: str = "meshtron edge routing debug") -> None:
    """Polylines as VTK polyline cells with one or more cell scalar arrays.

    `scalars` is {name: values} (int -> int array, float -> double array).
    A bare list is accepted as the legacy single "patch_label" array."""
    if not isinstance(scalars, dict):
        scalars = {"patch_label": list(scalars)}
    pts, cells = [], []
    for Q in polys:
        base = len(pts)
        pts.extend(np.asarray(Q, float).tolist())
        cells.append((base, len(Q)))
    with open(path, "w") as fh:
        fh.write(f"# vtk DataFile Version 2.0\n{title}\nASCII\n")
        fh.write("DATASET UNSTRUCTURED_GRID\n")
        fh.write(f"POINTS {len(pts)} double\n")
        for p in pts:
            fh.write(f"{p[0]:.9f} {p[1]:.9f} {p[2]:.9f}\n")
        fh.write(f"CELLS {len(cells)} {sum(n + 1 for _b, n in cells)}\n")
        for b, n in cells:
            fh.write(f"{n} " + " ".join(str(b + i) for i in range(n)) + "\n")
        fh.write(f"CELL_TYPES {len(cells)}\n")
        for _ in cells:
            fh.write("4\n")
        fh.write(f"CELL_DATA {len(cells)}\n")
        for name, vals in scalars.items():
            isint = all(isinstance(v, (int, np.integer)) for v in vals)
            fh.write(f"SCALARS {name} {'int' if isint else 'double'} 1\n")
            fh.write("LOOKUP_TABLE default\n")
            for v in vals:
                fh.write(f"{int(v)}\n" if isint else f"{float(v):.9f}\n")


BLEND_ID_BASE = 930000


def blend_chord_edges(st, corner_ids: np.ndarray, C_snap: np.ndarray,
                      blocks: np.ndarray, stats: dict | None = None) -> int:
    """Give interior chord edges a shape blended from their parallel rails.

    An interior block edge runs through the volume, so no patch constrains it
    and it stays a straight chord.  The block topology already provides the
    "parallel transport" a frame field would have to solve for: within a block,
    the three other edges of the same direction class are the rails parallel to
    this edge.  Their deviation from their own chord, averaged, is added to our
    chord.  Boundary edges are left alone -- their patch already determines
    them, and any added bow is removed again by the reprojection.

    Runs as `edge_post_fn` of build_structures, i.e. BEFORE the Coons faces are
    built, so the new shape propagates into faces and volume.
    Returns the number of edges rewritten.
    """
    from edge_curves import CORNERS, local_edges

    occ: dict = {}
    for r in range(len(blocks)):
        for li, lj, ax in local_edges():
            a, b = int(corner_ids[r, li]), int(corner_ids[r, lj])
            occ.setdefault((a, b) if a < b else (b, a), []).append((r, ax, li, lj))

    def canon(r, li, lj, ax):
        """Local corner pair oriented from side 0 to side 1 of the axis."""
        return (li, lj) if CORNERS[li][ax] == 0 else (lj, li)

    done = 0
    for key in list(st.edge_pts.keys()):
        if st.edge_curve.get(key, -1) >= 0:
            continue                      # already an exact curve
        pts = np.asarray(st.edge_pts[key], float)
        devs = []
        for (r, ax, li, lj) in occ.get(key, []):
            for (li2, lj2, ax2) in local_edges():
                if ax2 != ax:
                    continue
                a2, b2 = int(corner_ids[r, li2]), int(corner_ids[r, lj2])
                k2 = (a2, b2) if a2 < b2 else (b2, a2)
                if k2 == key or st.edge_curve.get(k2, -1) < 0:
                    continue
                lo2, hi2 = canon(r, li2, lj2, ax)
                Q2 = np.asarray(st.get_edge(int(corner_ids[r, lo2]),
                                            int(corner_ids[r, hi2])), float)
                if len(Q2) != len(pts):
                    continue
                devs.append(Q2 - np.linspace(Q2[0], Q2[-1], len(Q2)))
        if not devs:
            continue
        D = np.mean(devs, axis=0)
        r, ax, li, lj = occ[key][0]
        lo, hi = canon(r, li, lj, ax)
        a0 = int(corner_ids[r, lo])
        p0, p1 = C_snap[r, lo], C_snap[r, hi]
        new = np.linspace(p0, p1, len(pts)) + D
        new[0], new[-1] = p0, p1
        st.edge_pts[key] = new if a0 == key[0] else new[::-1]
        st.edge_curve[key] = BLEND_ID_BASE + done
        done += 1
    if stats is not None:
        stats["edges_blended"] = stats.get("edges_blended", 0) + done
    return done


def make_face_projector(pp: "PatchPaths", stats: dict | None = None,
                        relax_iters: int = 200):
    """face_project_fn for refill_curved: pull a boundary face onto its patch.

    Only the INTERIOR of the face grid moves; the four edge rows stay as the
    routed curves, so neighbouring blocks keep matching exactly and the mesh
    stays watertight.  The patch is chosen from the labels the four corners
    agree on; when that is ambiguous, the majority label of the projected
    interior decides (restricted to the corner candidates when possible).

    Plain nearest-point projection moves the grid points LATERALLY, which
    bunches up the face parametrisation and folds the first cell layer behind
    the wall: measured 2 -> 595 inverted cells on machine_0252_n8000 with the
    boundary quads themselves barely changing (aspect p99 3.0 -> 3.1). After
    projecting, `relax_iters` sweeps of a surface-constrained Laplacian
    redistribute the interior points on the patch, with the four edge rows held
    fixed. Set relax_iters=0 for the raw projection.
    """
    fm = pp.fm

    def fn(G: np.ndarray, ids, pts_of) -> np.ndarray:
        if G.shape[0] < 3 or G.shape[1] < 3:
            return G
        inner = G[1:-1, 1:-1].reshape(-1, 3)
        cands: set | None = None
        for i in ids:
            labs = pp.endpoint_labels(np.asarray(pts_of[int(i)], float))
            cands = labs if cands is None else (cands & labs)
        if cands is not None and len(cands) == 1:
            lab = int(next(iter(cands)))
        else:
            _, tri, _ = fm.surface_nearest(inner, k=32)
            labs = np.asarray(fm.surface_tri_label[tri], np.int64)
            vals, counts = np.unique(labs, return_counts=True)
            if cands:
                keep = np.isin(vals, list(cands))
                if keep.any():
                    vals, counts = vals[keep], counts[keep]
            lab = int(vals[int(np.argmax(counts))])
        patch = pp.patches.get(lab)
        if patch is None:
            return G
        _, proj = patch.project(inner)
        out = np.array(G, float, copy=True)
        out[1:-1, 1:-1] = proj.reshape(G.shape[0] - 2, G.shape[1] - 2, 3)
        for _ in range(int(relax_iters)):
            lap = 0.25 * (out[:-2, 1:-1] + out[2:, 1:-1]
                          + out[1:-1, :-2] + out[1:-1, 2:])
            out[1:-1, 1:-1] = 0.5 * out[1:-1, 1:-1] + 0.5 * lap
            _, q = patch.project(out[1:-1, 1:-1].reshape(-1, 3))
            out[1:-1, 1:-1] = q.reshape(G.shape[0] - 2, G.shape[1] - 2, 3)
        if stats is not None:
            stats["faces_projected"] = stats.get("faces_projected", 0) + 1
            stats.setdefault("faces_projected_labels", {})
            k = str(lab)
            stats["faces_projected_labels"][k] = \
                stats["faces_projected_labels"].get(k, 0) + 1
        return out

    return fn


def snap_seam_path(seam, fm, R: np.ndarray) -> np.ndarray:
    """Pull a seam-routed path back onto the feature curve and the surface.

    The seam navigator returns a resampled path that is neither on the seam
    polyline nor on the npz surface (measured 4.9e-03 on machine_0387_n8000).
    Seam polyline VERTICES lie exactly on the surface, but some segments are
    long chords across it, so snapping to the polyline alone is not enough:
    snap to the seam first (keeps the edge on the label boundary), then project
    onto the surface (removes the residual chord error). Endpoints stay put --
    they are the welded block corners.
    """
    R = np.asarray(R, float).copy()
    if len(R) < 3:
        return R
    inner = R[1:-1]
    _, _, _, ps = seam.nearest(inner)
    ps = np.asarray(ps, float)
    d, _, proj = fm.surface_nearest(ps, k=32)
    R[1:-1] = proj
    return R


def make_boundary_face_test(fm, surface_points=None, surface_tris=None,
                            delta_frac: float = 0.25, seed: int = 0,
                            min_dist_cells: float = 1.0):
    """is_boundary_face(ids, G) -> bool for refill_curved.

    A block face owned by a single block is not necessarily on the domain
    boundary: where two blocks touch across only part of a side (a T-junction),
    both sides are recorded with one owner each while lying INSIDE the domain.
    Offset the face grid's centre to both sides along its normal and ray cast
    against the closed npz surface -- a real boundary face has exactly one side
    inside, an interior wall has both.

    The ray cast alone is not enough: on T-junction-free samples it still
    rejected faces whose grid sits exactly ON the npz surface (median distance
    0.0000), and a wrongly rejected face is neither projected nor measured, so
    the conformity number would keep a hole. A face is therefore only rejected
    when it ALSO sits clearly off the geometry -- median distance above
    `min_dist_cells` times its own cell size. Real T-junction walls measured
    0.06 to 0.09 against a cell size of 0.048 (1.3 to 1.9 cells); the false
    positives measured 0.000 to 0.020.

    `delta_frac` is the offset as a fraction of the face's shorter side, so the
    test scales with the block rather than with the mesh.
    """
    P = np.asarray(fm.surface_points if surface_points is None
                   else surface_points, float)
    T = np.asarray(fm.surface_tris if surface_tris is None
                   else surface_tris, np.int64)
    rng = np.random.default_rng(seed)
    d = rng.normal(size=3)
    d /= np.linalg.norm(d)
    v0, v1, v2 = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    e1, e2 = v1 - v0, v2 - v0
    hcr = np.cross(d, e2)
    a = np.einsum('ij,ij->i', e1, hcr)
    ok = np.abs(a) > 1e-12
    inv = np.zeros_like(a)
    inv[ok] = 1.0 / a[ok]

    def _inside(q):
        sv = q[None] - v0
        u = np.einsum('ij,ij->i', sv, hcr) * inv
        qv = np.cross(sv, e1)
        v = (qv @ d) * inv
        t = np.einsum('ij,ij->i', qv, e2) * inv
        hit = ok & (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (t > 1e-9)
        return bool(hit.sum() % 2 == 1)

    def fn(ids, G) -> bool:
        G = np.asarray(G, float)
        ni, nj = G.shape[0], G.shape[1]
        c = G[ni // 2, nj // 2]
        du = G[-1, nj // 2] - G[0, nj // 2]
        dv = G[ni // 2, -1] - G[ni // 2, 0]
        n = np.cross(du, dv)
        ln = np.linalg.norm(n)
        if ln < 1e-15:
            return True
        n = n / ln
        delta = delta_frac * min(np.linalg.norm(du), np.linalg.norm(dv))
        if delta <= 1e-12:
            return True
        if _inside(c + delta * n) != _inside(c - delta * n):
            return True                      # one side out: real boundary
        d, _, _ = fm.surface_nearest(G.reshape(-1, 3), k=32)
        cell = max(np.linalg.norm(du) / max(ni - 1, 1),
                   np.linalg.norm(dv) / max(nj - 1, 1))
        return float(np.median(d)) <= min_dist_cells * cell

    return fn
