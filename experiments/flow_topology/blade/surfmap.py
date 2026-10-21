"""Harmonic map of a cut surface (triangulated, from the sample surface) onto its wall: boundary vertices get
their images by the blocking rules (Dirichlet), the interior solves the cotangent Laplace equation for the 3D
target positions, and any point on the cut surface maps by barycentric interpolation (then onto the wall).
Curves that do not cross on the cut do not cross on the wall (relative positions are kept)."""
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

import msh_walls as MW


def refine(P, T):
    """1:4 subdivision (midpoints)."""
    P = list(P); mid = {}
    def m(a, b):
        k = (min(a, b), max(a, b))
        if k not in mid:
            mid[k] = len(P); P.append(0.5 * (P[a] + P[b]))
        return mid[k]
    out = []
    for a, b, c in T:
        ab, bc, ca = m(a, b), m(b, c), m(c, a)
        out += [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
    return np.array(P), np.array(out)


class SurfaceMap:
    def __init__(self, S, T, boundary_images, levels=2):
        """S [n,3] points, T [m,3] triangles of the patch, boundary_images(points) -> [k,3] images."""
        used = np.unique(T); ix = {int(v): i for i, v in enumerate(used)}
        X = S[used]; Tl = np.vectorize(ix.get)(T)
        for _ in range(levels):
            X, Tl = refine(X, Tl)
        self.X, self.T = X, Tl
        n = len(self.X)
        cnt = defaultdict(int)
        for t in self.T:
            for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                cnt[(min(a, b), max(a, b))] += 1
        bnd = sorted({v for e, c in cnt.items() if c == 1 for v in e})
        self.bnd = np.array(bnd)
        Y = np.zeros((n, 3)); Y[self.bnd] = boundary_images(self.X[self.bnd])
        # cotangent Laplacian
        I, J, W = [], [], []
        for t in self.T:
            for k in range(3):
                i, j, o = t[k], t[(k + 1) % 3], t[(k + 2) % 3]
                u, v = self.X[i] - self.X[o], self.X[j] - self.X[o]
                c = float(u @ v / max(np.linalg.norm(np.cross(u, v)), 1e-14))
                c = max(c, 1e-3)                                      # keep the weights positive
                I += [i, j]; J += [j, i]; W += [0.5 * c, 0.5 * c]
        Lm = sp.coo_matrix((W, (I, J)), shape=(n, n)).tocsr()
        Lm = sp.diags(np.asarray(Lm.sum(1)).ravel()) - Lm
        free = np.setdiff1d(np.arange(n), self.bnd)
        A = Lm[free][:, free].tocsc(); Bm = Lm[free][:, self.bnd]
        for d in range(3):
            Y[free, d] = spla.spsolve(A, -Bm @ Y[self.bnd, d])
        self.Y = Y
        self.tree = cKDTree(self.X[self.T].mean(1))

    def __call__(self, Q, k=24):
        Q = np.asarray(Q, float).reshape(-1, 3)
        _, idx = self.tree.query(Q, k=min(k, len(self.T)))
        A, B, C = (self.X[self.T[idx, i]] for i in range(3))
        P, bary = MW._closest_on_tris(Q, A, B, C)
        j = np.argmin(np.linalg.norm(P - Q[:, None], axis=2), 1); r = np.arange(len(Q))
        tri = self.T[idx[r, j]]; b = bary[r, j]
        return (b[:, :, None] * self.Y[tri]).sum(1)
