"""seam_graph.py — 1D connectivity graph along geometry seam curves.

Nodes are junction points (curve endpoints clustered by coincidence, shared
across curves) plus queried arc positions on each curve. Arcs are the
sub-intervals between adjacent positions on a curve, including the wrap arc
of closed curves (blade foot loops).

`SeamNavigator.path_between()` finds a polyline between two queried arc
positions: it enumerates loopless candidate paths, keeps only routes that
behave sanely versus the chord (arc/chord <= 1.15, forward direction, endpoint
tolerance) and returns the shortest surviving route resampled to `n` points.
Used by the seam-only back-mapping (scripts/map_generated_blocks.py) and by
the step-(b) selftest (scripts/snap_selftest.py).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def curve_L(seam, c):
    lo, hi = int(seam.offset[c]), int(seam.offset[c + 1])
    return float(seam.arclen[hi - 1] - seam.arclen[lo])


def segment_wrap(seam, c, t0, t1, n):
    """sample_segment with arc-length parameters allowed to exceed the curve
    length (wrap-around on closed curves). Width t1-t0 must be <= L."""
    lo, hi = int(seam.offset[c]), int(seam.offset[c + 1])
    L = float(seam.arclen[hi - 1] - seam.arclen[lo])
    s0, width = float(t0) % L, float(t1) - float(t0)
    if width >= L:
        return seam.sample_segment(c, 0.0, L, n)
    if s0 + width <= L + 1e-9:
        return seam.sample_segment(c, s0, min(s0 + width, L), n)
    n1 = max(2, int(round(n * (L - s0) / width)))
    A = seam.sample_segment(c, s0, L, n1)[:-1]
    B = seam.sample_segment(c, 0.0, s0 + width - L, n - n1 + 1)
    return np.concatenate([A, B])


def sample_arc(seam, c, start, width, dir_, n):
    if dir_ > 0:
        return segment_wrap(seam, c, start, start + width, n)
    return segment_wrap(seam, c, start - width, start, n)[::-1]


def path_points(seam, hops, tot_w, n):
    segs = [sample_arc(seam, c, st, w, d, max(2, int(round(n * w / tot_w))))
            for c, st, w, d in hops]
    return np.concatenate([s[:-1] for s in segs[:-1]] + [segs[-1]])


def ep_clusters(seam, tol=1e-4):
    """Union-find over curve endpoints: one junction id per coincident group."""
    M = len(seam.ep_curve)
    parent = list(range(M))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(M):
        for j in range(i + 1, M):
            if np.linalg.norm(seam.ep_pt[i] - seam.ep_pt[j]) < tol:
                parent[find(j)] = find(i)
    return [find(i) for i in range(M)]


def simple_paths(adj, s, g, max_hops=4, cap=400):
    """All loopless paths s->g up to max_hops arcs (tiny graphs)."""
    out = []
    stack = [(s, [], 0.0, frozenset({s}))]
    while stack and len(out) < cap:
        u, hops, w, seen = stack.pop()
        if u == g:
            out.append((w, hops))
            continue
        if len(hops) >= max_hops:
            continue
        for v, wc, arc in adj[u]:
            if v not in seen:
                stack.append((v, hops + [arc], w + wc, seen | {v}))
    return out


def build_graph(seam, corner_ts):
    """1D connectivity along the seam curves. Nodes: junction points (shared
    across curves) plus queried arc positions. Arcs: curve sub-intervals
    between adjacent positions, incl. the wrap arc on closed curves.
    Adjacency entry: (nbr, weight, (curve, start_t, width, dir)).
    Returns (node_by_(curve,rounded_t), adjacency)."""
    Ls = {c: curve_L(seam, c) for c in range(seam.n_curves)}
    clusters = ep_clusters(seam)
    marks = {}
    for m in range(len(seam.ep_curve)):
        c = int(seam.ep_curve[m])
        t = float(seam.ep_t[m]) * Ls[c]
        marks.setdefault(c, {})[round(t, 9)] = int(clusters[m])
    tsets = {c: {round(float(t), 9) for t in corner_ts.get(c, ())}
             for c in range(seam.n_curves)}
    for c, mm in marks.items():
        tsets.setdefault(c, set()).update(mm)
    node, node_j, nid = {}, {}, 0
    for c in range(seam.n_curves):
        for tr in sorted(tsets[c]):
            jt = marks.get(c, {}).get(tr)
            if jt is not None and jt in node_j:
                node[(c, tr)] = node_j[jt]
                continue
            node[(c, tr)] = nid
            if jt is not None:
                node_j[jt] = nid
            nid += 1
    adj = defaultdict(list)
    for c in range(seam.n_curves):
        ts = sorted(tsets[c])
        pairs = [(ts[k], ts[k + 1]) for k in range(len(ts) - 1)]
        if bool(seam.closed[c]) and len(ts) >= 2:
            pairs.append((ts[-1], ts[0]))
        for ta, tb in pairs:
            w = tb - ta if tb > ta else Ls[c] - ta + tb
            a, b = node[(c, ta)], node[(c, tb)]
            adj[a].append((b, w, (c, ta, w, +1)))
            adj[b].append((a, w, (c, tb, w, -1)))
    return node, adj


def resample_len(P, n):
    """Arc-length resample of a polyline to exactly n points (endpoints kept).

    `path_points` splits its sample budget across hops by rounding, which can
    drift +/-1; consumers (Coons faces) need the requested count exactly."""
    P = np.asarray(P, float)
    if P.shape[0] == int(n):
        return P
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    q = np.linspace(0.0, s[-1], int(n))
    return np.column_stack([np.interp(q, s, P[:, k]) for k in range(P.shape[1])])


class SeamNavigator:
    """Shortest sane seam route between two queried arc positions.

    Constructed once per block candidate from the set of snapped corner
    positions (curve id -> arc-length values). `path_between` returns
    (points (n,3) p0->p1, first curve id) or None when no route survives
    the chord-sanity guards (= caller keeps the chord)."""

    def __init__(self, seam, corner_ts):
        self.seam = seam
        self.node, self.adj = build_graph(seam, corner_ts)

    def _node_of(self, c, t):
        return self.node.get((int(c), round(float(t), 9)))

    def add_point(self, c, t):
        """Insert an interior node on curve c at arc-length t (graphs are tiny,
        so rebuild-adjacent arcs locally instead of a full rebuild)."""
        c, tr = int(c), round(float(t), 9)
        if (c, tr) in self.node or not (0.0 <= tr <= curve_L(self.seam, c) + 1e-9):
            return self._node_of(c, tr)
        nid = max(self.node.values()) + 1
        self.node[(c, tr)] = nid
        ts = sorted(tr_ for (cc, tr_) in self.node if cc == c)
        fresh = defaultdict(list)
        L = curve_L(self.seam, c)
        pairs = [(ts[k], ts[k + 1]) for k in range(len(ts) - 1)]
        if bool(self.seam.closed[c]) and len(ts) >= 2:
            pairs.append((ts[-1], ts[0]))
        for ta, tb in pairs:
            w = tb - ta if tb > ta else L - ta + tb
            a, b = self.node[(c, ta)], self.node[(c, tb)]
            fresh[a].append((b, w, (c, ta, w, +1)))
            fresh[b].append((a, w, (c, tb, w, -1)))
        for a, arcs in fresh.items():
            self.adj[a] = [e for e in self.adj[a]
                           if len(e) == 3 and e[2][0] != c]
            self.adj[a].extend(arcs)
        return nid

    def path_between(self, p0, ca, ta, p1, cb, tb, n):
        na, nb = self._node_of(ca, ta), self._node_of(cb, tb)
        if na is None or nb is None or na == nb:
            return None
        chord = float(np.linalg.norm(p1 - p0))
        if chord <= 1e-9:
            return None
        best = None
        for w, hops in simple_paths(self.adj, na, nb):
            if w < 1e-9:
                continue
            # a route that turns 90 deg at a junction needs arc/chord ~sqrt(2);
            # such edges are straight chords in the block decomposition
            if w > 1.15 * chord:
                continue
            P = path_points(self.seam, hops, w, max(int(n), 8))
            if float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) > 1.15 * chord:
                continue
            if (P[-1] - P[0]) @ (p1 - p0) <= 0:
                continue
            if max(float(np.linalg.norm(P[0] - p0)),
                   float(np.linalg.norm(P[-1] - p1))) > 0.5 * chord + 1e-9:
                continue
            if np.linalg.norm(P[0] - p1) < np.linalg.norm(P[-1] - p1):
                P = P[::-1]
            if best is None or w < best[0]:
                best = (w, P, int(hops[0][0]))
        if best is None:
            return None
        return resample_len(best[1], int(n)), best[2]
