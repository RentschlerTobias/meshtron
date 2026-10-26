"""Parametrisation of a tube-like wall patch (e.g. the blade between root and tip profile): u = harmonic
function on the patch, 0 on one boundary loop and 1 on the other; s = normalised arc length along the iso-lines
of u (start points aligned from level to level). A curve between two points of the patch is a straight line in
(u, s) mapped back to the surface -- it stays on the surface and runs smoothly from one profile to the other,
without the side jumps of a nearest-point projection on a thin blade."""
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import msh_walls as MW


def _loops(T):
    cnt = defaultdict(int)
    for t in T:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            cnt[(min(a, b), max(a, b))] += 1
    adj = defaultdict(list)
    for (a, b), c in cnt.items():
        if c == 1:
            adj[a].append(b); adj[b].append(a)
    loops, seen = [], set()
    for v0 in adj:
        if v0 in seen:
            continue
        lp, prev = [v0], None; seen.add(v0)
        while True:
            nx = [w for w in adj[lp[-1]] if w != prev and w not in seen]
            if not nx:
                break
            prev = lp[-1]; lp.append(nx[0]); seen.add(nx[0])
        loops.append(lp)
    return loops


def _resample_closed(Q, m):
    Q = np.vstack([Q, Q[:1]])
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]
    t = np.linspace(0, s[-1], m, endpoint=False)
    return np.stack([np.interp(t, s, Q[:, k]) for k in range(3)], 1)


class TubeParam:
    def __init__(self, P, T, levels=81, m=720):
        used = np.unique(T); ix = {int(v): i for i, v in enumerate(used)}
        self.X = P[used]; self.T = np.vectorize(ix.get)(T)
        loops = _loops(self.T)
        if len(loops) != 2:
            raise ValueError(f"patch has {len(loops)} boundary loops, not a tube")
        n = len(self.X)
        I, J, W = [], [], []
        for t in self.T:
            for k in range(3):
                i, j, o = t[k], t[(k + 1) % 3], t[(k + 2) % 3]
                a, b = self.X[i] - self.X[o], self.X[j] - self.X[o]
                c = max(float(a @ b / max(np.linalg.norm(np.cross(a, b)), 1e-14)), 1e-3)
                I += [i, j]; J += [j, i]; W += [0.5 * c, 0.5 * c]
        Lm = sp.coo_matrix((W, (I, J)), shape=(n, n)).tocsr(); Lm = sp.diags(np.asarray(Lm.sum(1)).ravel()) - Lm
        u = np.zeros(n); fixed = np.r_[loops[0], loops[1]]; u[loops[1]] = 1.0
        free = np.setdiff1d(np.arange(n), fixed)
        u[free] = spla.spsolve(Lm[free][:, free].tocsc(), -Lm[free][:, fixed] @ u[fixed])
        self.u = u
        # iso-lines
        self.lev = np.linspace(0, 1, levels); secs = []
        for k, ul in enumerate(self.lev):
            if k == 0 or k == levels - 1:
                lp = loops[0] if k == 0 else loops[1]
                secs.append(_resample_closed(self.X[lp], m)); continue
            segs = []
            for t in self.T:
                pts = []
                for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                    ua, ub = u[a] - ul, u[b] - ul
                    if ua * ub < 0:
                        w = ua / (ua - ub); pts.append(((min(a, b), max(a, b)), self.X[a] + w * (self.X[b] - self.X[a])))
                if len(pts) == 2:
                    segs.append(pts)
            adj = defaultdict(list); pos = {}
            for (ka, pa), (kb, pb) in segs:
                adj[ka].append(kb); adj[kb].append(ka); pos[ka] = pa; pos[kb] = pb
            start = next(iter(adj)); lp, prev = [start], None
            while True:
                nx = [w for w in adj[lp[-1]] if w != prev]
                if not nx or nx[0] == start or len(lp) > len(adj):
                    break
                prev = lp[-1]; lp.append(nx[0])
            secs.append(_resample_closed(np.array([pos[v] for v in lp]), m))
        # align start and orientation level by level
        for k in range(1, levels):
            best = None
            for S in (secs[k], secs[k][::-1]):
                j = int(np.argmin(np.linalg.norm(S - secs[k - 1][0], axis=1)))
                R = np.roll(S, -j, axis=0)
                err = np.linalg.norm(R - secs[k - 1], axis=1).mean()
                if best is None or err < best[0]:
                    best = (err, R)
            secs[k] = best[1]
        self.S = np.array(secs)                                  # [levels, m, 3]
        self.m = m

    def to_param(self, Q):
        """(u, s) of points on the patch."""
        Q = np.asarray(Q, float).reshape(-1, 3); out = np.zeros((len(Q), 2))
        flat = self.S.reshape(-1, 3)
        from scipy.spatial import cKDTree
        tr = cKDTree(flat)
        _, j = tr.query(Q)
        out[:, 0] = self.lev[j // self.m]; out[:, 1] = (j % self.m) / self.m
        return out

    def from_param(self, us):
        us = np.asarray(us, float).reshape(-1, 2)
        u = np.clip(us[:, 0], 0, 1) * (len(self.lev) - 1); k = np.minimum(np.floor(u).astype(int), len(self.lev) - 2)
        w = u - k; s = (us[:, 1] % 1.0) * self.m; i = np.floor(s).astype(int) % self.m; ws = s - np.floor(s)
        def at(kk):
            return (1 - ws)[:, None] * self.S[kk, i] + ws[:, None] * self.S[kk, (i + 1) % self.m]
        return (1 - w)[:, None] * at(k) + w[:, None] * at(k + 1)

    def curve(self, A, B, n):
        """Straight line in (u, s) from A to B (shortest way round in s), on the surface."""
        pa, pb = self.to_param(np.array([A, B]))
        ds = (pb[1] - pa[1] + 0.5) % 1.0 - 0.5
        t = np.linspace(0, 1, n)[:, None]
        us = np.c_[pa[0] + t[:, 0] * (pb[0] - pa[0]), pa[1] + t[:, 0] * ds]
        C = self.from_param(us); C[0], C[-1] = A, B
        return C


def _poly_s(Sk, X):
    """normalised arc-length parameter of the nearest point on closed polyline Sk [m,3] for points X [n,3]."""
    A = Sk; Bq = np.roll(Sk, -1, axis=0); d = Bq - A; L = np.linalg.norm(d, axis=1); s0 = np.r_[0, np.cumsum(L)][:-1]
    tot = L.sum()
    out = np.zeros(len(X))
    for i0 in range(0, len(X), 256):
        Xi = X[i0:i0 + 256]
        t = np.clip(np.einsum("nkj,kj->nk", Xi[:, None] - A[None], d) / (L ** 2).clip(1e-30), 0, 1)
        D = np.linalg.norm(A[None] + t[..., None] * d[None] - Xi[:, None], axis=2)
        k = np.argmin(D, 1)
        out[i0:i0 + 256] = (s0[k] + t[np.arange(len(Xi)), k] * L[k]) / tot
    return out


def params_on_tube(tp, Q, u):
    """s for points Q with known u (e.g. the tube's own vertices): blend of the two bracketing iso-lines."""
    Q = np.asarray(Q, float).reshape(-1, 3); u = np.asarray(u, float)
    x = np.clip(u, 0, 1) * (len(tp.lev) - 1); k = np.minimum(np.floor(x).astype(int), len(tp.lev) - 2); w = x - k
    s = np.zeros(len(Q))
    for kk in np.unique(k):
        m = k == kk
        s0 = _poly_s(tp.S[kk], Q[m]); s1 = _poly_s(tp.S[kk + 1], Q[m])
        ds = (s1 - s0 + 0.5) % 1.0 - 0.5
        s[m] = (s0 + w[m] * ds) % 1.0
    return s


def vertex_u(tp, P_global, used_global):
    """u of the tube's own vertices, by their global indices (as passed in T)."""
    return tp.u
