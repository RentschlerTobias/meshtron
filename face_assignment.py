"""face_assignment.py -- assign boundary block faces to geometry patches as a
whole, instead of deciding each edge by proximity.

Why this exists. Mapping a block structure onto a geometry needs an
ASSOCIATION: which patch every boundary face belongs to, which feature curve
every boundary edge follows. Ground truth carries it implicitly, because its
corners sit exactly on the features. A generated blocking does not, and 32 of
machine_0034_n2000's 40 corners have more than one patch within 0.06 -- corners
of a block structure sit where features meet, so per-entity proximity is a coin
flip exactly where it matters. Deciding an edge by "shortest path on the
nearest patch" dragged faces off the blade until the mesh no longer wrapped it.

The association is not free information, but it is heavily constrained:

  - a boundary face lies on ONE patch;
  - two boundary faces sharing a block edge either lie on the same patch, or on
    two patches whose seam curve that edge follows;
  - every patch of the geometry must be covered by at least one face.

That makes it a labelling problem on the face adjacency graph, with a unary
cost (how far the face is from the patch), a pairwise cost (how far the shared
edge is from the seam between the two patches, infinite when no such seam
exists) and a coverage penalty. Small enough to solve by iterated conditional
modes with restarts: 40 to 64 faces over at most 7 labels.

The result determines the routing outright -- an edge between two faces with
different labels follows their seam, an edge inside one patch is routed on that
patch -- so no proximity heuristic is left in the loop.

STATUS: the labelling works, the routing built on it does NOT yet beat the
simpler rule it was meant to replace. Measured on machine_0034_n2000 at h=0.05,
blade-hull triangles with no mesh boundary within 0.15:

    generated, shortest path                 714 uncovered   1157 inverted
    generated, nearest patch (patch_paths)    28 uncovered    545 inverted
    generated, this assignment               120 uncovered    686 inverted
    GT, nearest patch                          0 uncovered    218 inverted
    GT, this assignment                        0 uncovered    250 inverted

The assignment itself is sound -- it solves in under a second, covers every
patch and violates no adjacency constraint on both blockings. The weak part is
the unary cost: it measures the BILINEAR face through the four corners against
each patch, and on a strongly curved face that surface sits far enough off its
own patch for a wrong label to win. Recomputing the unary from the real Coons
faces (a two-pass scheme) is the obvious next step and is not done here.

Until that is resolved the production path stays with
PatchPaths.nearest_candidate; this module is kept for the labelling, which is
useful on its own as a consistency check on a blocking.
"""
from __future__ import annotations

import numpy as np

HEX_FACES = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5),
             (2, 3, 7, 6), (3, 0, 4, 7))
FORBIDDEN = 1e6


def _bilinear(quad: np.ndarray, m: int = 5) -> np.ndarray:
    """m x m samples of the bilinear face through the four corners."""
    a, b, c, d = quad
    u = np.linspace(0.0, 1.0, m)[:, None, None]
    v = np.linspace(0.0, 1.0, m)[None, :, None]
    return ((1 - u) * (1 - v) * a + u * (1 - v) * b
            + u * v * c + (1 - u) * v * d).reshape(-1, 3)


def boundary_faces(blocks: np.ndarray, welded: np.ndarray):
    """(faces, owners): faces owned by exactly one block, as corner id cycles."""
    count: dict = {}
    order: dict = {}
    for row in welded:
        for f in HEX_FACES:
            ids = [int(row[i]) for i in f]
            key = frozenset(ids)
            count[key] = count.get(key, 0) + 1
            order.setdefault(key, ids)
    return [order[k] for k, v in count.items() if v == 1]


class FaceAssignment:
    """Global patch labelling of a blocking's boundary faces."""

    def __init__(self, fm, welded_ids: np.ndarray, points: np.ndarray,
                 patch_labels=None, coverage_weight: float = 50.0,
                 samples: int = 5):
        self.fm = fm
        self.seam = fm.seam_curves
        self.pts = np.asarray(points, float)
        self.labels = (sorted(set(int(x) for x in fm.surface_tri_label))
                       if patch_labels is None else list(patch_labels))
        self.faces = boundary_faces(None, np.asarray(welded_ids, np.int64))
        self.coverage_weight = float(coverage_weight)
        self.samples = int(samples)
        self._unary()
        self._adjacency()
        self._pairwise()

    # -- costs ------------------------------------------------------------
    def _patch_dist(self, Q, lab):
        sel = self.fm.surface_tri_label == lab
        tris = self.fm.surface_tris[sel]
        from scipy.spatial import cKDTree
        key = ("_tree", int(lab))
        tree = getattr(self, "_trees", {}).get(lab)
        if tree is None:
            tree = cKDTree(self.fm.surface_points[tris].mean(axis=1))
            if not hasattr(self, "_trees"):
                self._trees = {}
            self._trees[lab] = tree
            self._tris = getattr(self, "_tris", {})
            self._tris[lab] = tris
        import sys
        hexrepo = ("/home/t1dde/hydrostack_pipeline/stack/domain_partition_3D/"
                   "experimentell/hex3d_algohex")
        if hexrepo not in sys.path:
            sys.path.insert(0, hexrepo)
        from clean_blocks import _closest_point_on_tris
        tris = self._tris[lab]
        _, cand = tree.query(Q, k=min(24, len(tris)))
        cand = np.atleast_2d(cand)
        T = tris[cand]
        d2, _ = _closest_point_on_tris(Q, self.fm.surface_points[T[..., 0]],
                                       self.fm.surface_points[T[..., 1]],
                                       self.fm.surface_points[T[..., 2]])
        return np.sqrt(d2.min(axis=1))

    def _unary(self):
        self.U = np.zeros((len(self.faces), len(self.labels)))
        for i, ids in enumerate(self.faces):
            Q = _bilinear(self.pts[np.asarray(ids, int)], self.samples)
            for j, lab in enumerate(self.labels):
                d = self._patch_dist(Q, lab)
                self.U[i, j] = float(d.mean() + d.max())

    def _adjacency(self):
        edge_faces: dict = {}
        for i, ids in enumerate(self.faces):
            for a, b in zip(ids, ids[1:] + ids[:1]):
                k = (a, b) if a < b else (b, a)
                edge_faces.setdefault(k, []).append(i)
        self.pairs = [(v[0], v[1], k) for k, v in edge_faces.items()
                      if len(v) == 2]

    def _pairwise(self):
        """cost[(i,j)][a,b]: shared edge against the seam between labels a,b."""
        self.P = {}
        for i, j, edge in self.pairs:
            Q = np.linspace(self.pts[edge[0]], self.pts[edge[1]], 7)
            M = np.zeros((len(self.labels), len(self.labels)))
            for a, la in enumerate(self.labels):
                for b, lb in enumerate(self.labels):
                    if la == lb:
                        M[a, b] = 0.0
                        continue
                    cands = [c for c in range(self.seam.n_curves)
                             if {int(self.seam.label_lo[c]),
                                 int(self.seam.label_hi[c])} == {la, lb}]
                    if not cands:
                        M[a, b] = FORBIDDEN
                        continue
                    d, cid, _, _ = self.seam.nearest(Q)
                    best = np.inf
                    for c in cands:
                        take = cid == c
                        best = min(best, float(d[take].max())
                                   if take.any() else np.inf)
                    M[a, b] = best if np.isfinite(best) else FORBIDDEN
            self.P[(i, j)] = M
        return self.P

    # -- solve ------------------------------------------------------------
    def energy(self, x):
        e = float(self.U[np.arange(len(x)), x].sum())
        for (i, j), M in self.P.items():
            e += float(M[x[i], x[j]])
        used = set(int(v) for v in x)
        e += self.coverage_weight * (len(self.labels) - len(used))
        return e

    def solve(self, restarts: int = 8, sweeps: int = 40, seed: int = 0):
        rng = np.random.default_rng(seed)
        best_x, best_e = None, np.inf
        for r in range(restarts):
            x = (self.U.argmin(axis=1) if r == 0
                 else rng.integers(0, len(self.labels), len(self.faces)))
            x = np.asarray(x, int)
            for _ in range(sweeps):
                moved = False
                for i in rng.permutation(len(self.faces)):
                    cur, bl, be = x[i], x[i], np.inf
                    for a in range(len(self.labels)):
                        x[i] = a
                        e = self.energy(x)
                        if e < be:
                            be, bl = e, a
                    x[i] = bl
                    if bl != cur:
                        moved = True
                if not moved:
                    break
            e = self.energy(x)
            if e < best_e:
                best_e, best_x = e, x.copy()
        self.x = best_x
        self.e = best_e
        return {frozenset(ids): self.labels[int(best_x[i])]
                for i, ids in enumerate(self.faces)}

    # -- report -----------------------------------------------------------
    def report(self):
        lab = [self.labels[int(v)] for v in self.x]
        from collections import Counter
        viol = sum(1 for (i, j), M in self.P.items()
                   if M[self.x[i], self.x[j]] >= FORBIDDEN)
        return {"faces": len(self.faces), "energy": float(self.e),
                "by_label": dict(Counter(lab)),
                "uncovered_labels": sorted(set(self.labels) - set(lab)),
                "forbidden_pairs": int(viol)}


def edge_roles(assignment, faces):
    """(patch_of_edge, seam_of_edge) from a face->patch assignment.

    An edge between two boundary faces carrying the same label lies inside that
    patch; between different labels it follows their seam. That decides the
    routing outright -- no proximity left in the loop.
    """
    edge_faces: dict = {}
    for ids in faces:
        key = frozenset(ids)
        for a, b in zip(ids, ids[1:] + ids[:1]):
            k = (a, b) if a < b else (b, a)
            edge_faces.setdefault(k, []).append(assignment.get(key))
    patch_of, seam_of = {}, {}
    for k, labs in edge_faces.items():
        labs = [l for l in labs if l is not None]
        if len(labs) < 2:
            if labs:
                patch_of[k] = labs[0]
            continue
        if labs[0] == labs[1]:
            patch_of[k] = labs[0]
        else:
            seam_of[k] = (min(labs), max(labs))
    return patch_of, seam_of


class AssignedRouter:
    """path_fn driven by a face assignment instead of by proximity."""

    def __init__(self, fm, patch_paths, patch_of, seam_of, welded_ids,
                 corners, stats=None, seam_fn=None):
        self.fm = fm
        self.pp = patch_paths
        self.seam = fm.seam_curves
        self.patch_of = patch_of
        self.seam_of = seam_of
        self.stats = stats if stats is not None else {}
        # The assignment says WHICH seam an edge follows; the existing seam
        # navigator is still the better router along it (junctions, direction,
        # wrap). Sampling the arc here is only the fallback.
        self.seam_fn = seam_fn
        self.key = {}
        w = np.asarray(welded_ids, np.int64)
        C = np.asarray(corners, float)
        for r in range(w.shape[0]):
            for c in range(8):
                self.key[np.round(C[r, c], 9).tobytes()] = int(w[r, c])

    def _edge_key(self, p0, p1):
        a = self.key.get(np.round(np.asarray(p0, float), 9).tobytes())
        b = self.key.get(np.round(np.asarray(p1, float), 9).tobytes())
        if a is None or b is None:
            return None
        return (a, b) if a < b else (b, a)

    def _seam_path(self, p0, p1, n, pair):
        cands = [c for c in range(self.seam.n_curves)
                 if {int(self.seam.label_lo[c]),
                     int(self.seam.label_hi[c])} == set(pair)]
        best = None
        for c in cands:
            d, cid, t, _ = self.seam.nearest(np.stack([p0, p1]))
            take = cid == c
            if not take.all():
                d2, _, t2, _ = self.seam.nearest(np.stack([p0, p1]))
                t = t2
            lo, hi = int(self.seam.offset[c]), int(self.seam.offset[c + 1])
            s = self.seam.arclen[lo:hi]
            t0 = float(s[np.argmin(np.linalg.norm(
                self.seam.pts[lo:hi] - p0, axis=1))])
            t1 = float(s[np.argmin(np.linalg.norm(
                self.seam.pts[lo:hi] - p1, axis=1))])
            if abs(t1 - t0) < 1e-12:
                continue
            q = np.linspace(t0, t1, n)
            P = np.stack([np.interp(q, s, self.seam.pts[lo:hi, k])
                          for k in range(3)], axis=1)
            cost = (np.linalg.norm(P[0] - p0) + np.linalg.norm(P[-1] - p1))
            if best is None or cost < best[0]:
                best = (cost, P)
        if best is None:
            return None
        P = best[1].copy()
        P[0], P[-1] = p0, p1
        d, _, proj = self.fm.surface_nearest(P[1:-1], k=32)
        P[1:-1] = proj
        return P

    def __call__(self, p0, p1, n):
        k = self._edge_key(p0, p1)
        if k is None:
            return None
        if k in self.seam_of:
            if self.seam_fn is not None:
                res = self.seam_fn(p0, p1, n)
                if res is not None:
                    self.stats["edges_seam_navigator"] = \
                        self.stats.get("edges_seam_navigator", 0) + 1
                    return res
            P = self._seam_path(np.asarray(p0, float), np.asarray(p1, float),
                                n, self.seam_of[k])
            if P is not None:
                self.stats["edges_seam_assigned"] = \
                    self.stats.get("edges_seam_assigned", 0) + 1
                return P, 940000 + self.stats["edges_seam_assigned"]
        lab = self.patch_of.get(k)
        if lab is None:
            return None
        patch = self.pp.patches.get(int(lab))
        if patch is None:
            return None
        saved = self.pp.candidates
        self.pp.candidates = lambda a, b, _l=int(lab): [_l]
        try:
            R, info = self.pp.route(p0, p1, n)
        finally:
            self.pp.candidates = saved
        if R is None:
            return None
        self.stats["edges_patch_assigned"] = \
            self.stats.get("edges_patch_assigned", 0) + 1
        return R, 950000 + self.stats["edges_patch_assigned"]


def make_assigned_face_projector(pp, assignment, stats=None):
    """face_project_fn that uses the assignment instead of guessing a patch."""
    def fn(G, ids, pts_of):
        if G.shape[0] < 3 or G.shape[1] < 3:
            return G
        lab = assignment.get(frozenset(int(i) for i in ids))
        patch = pp.patches.get(int(lab)) if lab is not None else None
        if patch is None:
            return G
        inner = G[1:-1, 1:-1].reshape(-1, 3)
        _, proj = patch.project(inner)
        out = np.array(G, float, copy=True)
        out[1:-1, 1:-1] = proj.reshape(G.shape[0] - 2, G.shape[1] - 2, 3)
        if stats is not None:
            stats["faces_projected"] = stats.get("faces_projected", 0) + 1
        return out
    return fn
