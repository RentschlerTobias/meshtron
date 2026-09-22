"""curve_model.py — Kurven-Primitive: CurveSet, Blockkanten-, Seam-Extraktion.

Herausgeloest aus `geometry_features.py`, weil dieses sonst die 250-LOC-Grenze
reisst (Rollen: hier Datenstruktur + Quellen, dort Frames + FeatureModel v2).

`block_edge_curves` liest den `_edge_records`-Vertrag (`edges`/`edge_polyline`/
`offset`) — die AlgoHex-Blockkanten (Provenienz-Befund im Report).
`extract_seam_curves` liest Label-Grenzen der Design-System-Triangulierung.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

AXIS = np.array([0.0, 0.0, 1.0])


@dataclass
class CurveSet:
    """Kurven als Sample-Arrays: pts/curve_of/offset/arclen + Endpunkte."""

    pts: np.ndarray
    curve_of: np.ndarray
    offset: np.ndarray
    arclen: np.ndarray
    closed: np.ndarray
    label_lo: np.ndarray
    label_hi: np.ndarray
    ep_pt: np.ndarray
    ep_curve: np.ndarray
    ep_t: np.ndarray
    ep_vertex: np.ndarray
    _tree: cKDTree = field(default=None, repr=False)
    _ep_tree: cKDTree = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._tree = cKDTree(self.pts) if len(self.pts) else None
        self._ep_tree = cKDTree(self.ep_pt) if len(self.ep_pt) else None

    @property
    def n_curves(self) -> int:
        return len(self.offset) - 1

    def segment(self, curve: int) -> np.ndarray:
        lo, hi = int(self.offset[curve]), int(self.offset[curve + 1])
        return self.pts[lo:hi]

    def sample_segment(self, curve: int, t0: float, t1: float, n: int) -> np.ndarray:
        """`n` Punkte entlang Kurve `curve` zwischen Bogenlaengen t0..t1."""
        lo, hi = int(self.offset[curve]), int(self.offset[curve + 1])
        s = self.arclen[lo:hi]
        if len(s) < 2:
            return np.repeat(self.pts[lo:lo + 1], n, axis=0)
        q = np.linspace(float(t0), float(t1), n)
        P = self.pts[lo:hi]
        return np.stack([np.interp(q, s, P[:, k]) for k in range(3)], axis=1)

    def nearest(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(...,3) -> (dist, curve_id, bogenlaenge, punkt) auf naechster Kurve."""
        q = np.asarray(q, float).reshape(-1, 3)
        _, j = self._tree.query(q, k=min(8, len(self.pts)))
        j = np.atleast_2d(j)
        best = np.full(len(q), np.inf)
        bc = np.zeros(len(q), np.int64)
        bt = np.zeros(len(q))
        bp = np.zeros((len(q), 3))
        for col in range(j.shape[1]):
            idx = j[:, col]
            cur = self.curve_of[idx]
            lo = self.offset[cur]
            hi = self.offset[cur + 1]
            nxt = np.where(idx + 1 < hi, idx + 1, idx)
            pr = np.where(idx > lo, idx - 1, idx)
            for a, b in ((pr, idx), (idx, nxt)):
                A, B = self.pts[a], self.pts[b]
                AB = B - A
                L2 = np.maximum((AB * AB).sum(-1), 1e-30)
                s = np.clip(((q - A) * AB).sum(-1) / L2, 0.0, 1.0)
                p = A + s[:, None] * AB
                d = np.linalg.norm(q - p, axis=1)
                take = d < best
                if take.any():
                    best[take] = d[take]
                    bc[take] = cur[take]
                    Ar = self.arclen[a][take]
                    bt[take] = Ar + s[take] * (self.arclen[b][take] - Ar)
                    bp[take] = p[take]
        return best, bc, bt, bp

    def nearest_endpoint(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(q, float).reshape(-1, 3)
        d, i = self._ep_tree.query(q, k=1)
        return np.atleast_1d(d), np.atleast_1d(i)


def _chain_curves(pts, nodes_list, labels_list) -> CurveSet:
    P, co, off, arc, closed, lo, hi = [], [], [0], [], [], [], []
    ep_p, ep_c, ep_t = [], [], []
    for ci, (nodes, labs) in enumerate(zip(nodes_list, labels_list)):
        Q = pts[np.asarray(nodes, int)]
        d = np.linalg.norm(np.diff(Q, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(d)])
        P.append(Q)
        co.append(np.full(len(Q), ci, np.int64))
        off.append(off[-1] + len(Q))
        arc.append(s)
        is_closed = bool(np.linalg.norm(Q[0] - Q[-1]) < 1e-9 and len(Q) > 2)
        closed.append(is_closed)
        lo.append(labs[0] if labs else -1)
        hi.append(labs[1] if labs else -1)
        if not is_closed:
            ep_p.append(Q[0]); ep_c.append(ci); ep_t.append(0.0)
            ep_p.append(Q[-1]); ep_c.append(ci); ep_t.append(1.0)
    return CurveSet(
        pts=np.concatenate(P) if P else np.zeros((0, 3)),
        curve_of=np.concatenate(co) if co else np.zeros(0, np.int64),
        offset=np.asarray(off, np.int64),
        arclen=np.concatenate(arc) if arc else np.zeros(0),
        closed=np.asarray(closed, bool),
        label_lo=np.asarray(lo, np.int64),
        label_hi=np.asarray(hi, np.int64),
        ep_pt=np.asarray(ep_p, float).reshape(-1, 3),
        ep_curve=np.asarray(ep_c, np.int64),
        ep_t=np.asarray(ep_t, float),
        ep_vertex=np.full(len(ep_c), -1, np.int64),
    )


def block_edge_curves(edges: np.ndarray, poly: np.ndarray, off: np.ndarray) -> CurveSet:
    """Kurvensatz aus dem `_edge_records`-Vertrag (edges/polyline/offset)."""
    P, co, o, arc, closed, lo, hi = [], [], [0], [], [], [], []
    ep_p, ep_c, ep_t, ep_v = [], [], [], []
    for i in range(len(edges)):
        a, b = int(edges[i, 0]), int(edges[i, 1])
        Q = poly[int(off[i]):int(off[i + 1])]
        d = np.linalg.norm(np.diff(Q, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(d)])
        ci = len(o) - 1
        P.append(Q); co.append(np.full(len(Q), ci, np.int64))
        o.append(o[-1] + len(Q)); arc.append(s); closed.append(False)
        lo.append(-1); hi.append(-1)
        ep_p.append(Q[0]); ep_c.append(ci); ep_t.append(0.0); ep_v.append(a)
        ep_p.append(Q[-1]); ep_c.append(ci); ep_t.append(1.0); ep_v.append(b)
    return CurveSet(
        pts=np.concatenate(P) if P else np.zeros((0, 3)),
        curve_of=np.concatenate(co) if co else np.zeros(0, np.int64),
        offset=np.asarray(o, np.int64),
        arclen=np.concatenate(arc) if arc else np.zeros(0),
        closed=np.asarray(closed, bool),
        label_lo=np.asarray(lo, np.int64), label_hi=np.asarray(hi, np.int64),
        ep_pt=np.asarray(ep_p, float).reshape(-1, 3),
        ep_curve=np.asarray(ep_c, np.int64), ep_t=np.asarray(ep_t, float),
        ep_vertex=np.asarray(ep_v, np.int64),
    )


def extract_seam_curves(pts: np.ndarray, tris: np.ndarray, labels: np.ndarray) -> CurveSet:
    """Label-Grenzen der Design-System-Triangulierung, zu Kurven gruppiert.

    Grenzkante = Dreieckskante mit ungleichen Labels an den inzidenten
    Dreiecken. Grad != 2 trennt (Junction), sonst verkettet; Grad-2-Zyklen
    werden geschlossen gefuehrt. Analytischer Wuerfel mit 6 Labels -> 12 Kurven.
    """
    emap: dict = collections.defaultdict(set)
    for t, (a, b, c) in enumerate(tris):
        for u, v in ((a, b), (b, c), (c, a)):
            emap[(min(u, v), max(u, v))].add(int(labels[t]))
    bnd = {e for e, ls in emap.items() if len(ls) > 1}
    deg: dict = collections.Counter()
    adj: dict = collections.defaultdict(list)
    for u, v in bnd:
        deg[u] += 1; deg[v] += 1; adj[u].append(v); adj[v].append(u)
    used: set = set()
    chains: list = []

    def walk(a, b):
        ch = [a, b]; used.add((min(a, b), max(a, b)))
        prev, cur = a, b
        while deg[cur] == 2:
            nxt = [w for w in adj[cur] if w != prev]
            if not nxt:
                break
            w = nxt[0]; e = (min(cur, w), max(cur, w))
            if e in used:
                break
            used.add(e); ch.append(w); prev, cur = cur, w
        return ch

    for n in sorted(x for x in deg if deg[x] != 2):
        for w in adj[n]:
            if (min(n, w), max(n, w)) not in used:
                chains.append(walk(n, w))
    for e in sorted(bnd):
        if e not in used:
            chains.append(walk(e[0], e[1]))
    chains = [c for c in chains if len(c) >= 2]
    nodes_list, labels_list = [], []
    for ch in chains:
        labs = tuple(sorted(set().union(
            *[emap[(min(a, b), max(a, b))] for a, b in zip(ch[:-1], ch[1:])])))
        nodes_list.append(ch)
        labels_list.append((labs[0] if labs else -1, labs[-1] if labs else -1))
    return _chain_curves(pts, nodes_list, labels_list)


def concat(a: CurveSet, b: CurveSet) -> CurveSet:
    off = np.concatenate([[0], a.offset[1:], a.offset[-1] + b.offset[1:]])
    return CurveSet(
        pts=np.vstack([a.pts, b.pts]),
        curve_of=np.concatenate([a.curve_of, b.curve_of + a.n_curves]),
        offset=off, arclen=np.concatenate([a.arclen, b.arclen]),
        closed=np.concatenate([a.closed, b.closed]),
        label_lo=np.concatenate([a.label_lo, b.label_lo]),
        label_hi=np.concatenate([a.label_hi, b.label_hi]),
        ep_pt=np.vstack([a.ep_pt, b.ep_pt]) if len(b.ep_pt) else a.ep_pt,
        ep_curve=np.concatenate([a.ep_curve, b.ep_curve + a.n_curves]),
        ep_t=np.concatenate([a.ep_t, b.ep_t]),
        ep_vertex=np.concatenate([a.ep_vertex, b.ep_vertex]))
