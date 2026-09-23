"""edge_curves.py — Blockkanten als Kurven + konforme Coons-Flaechen.

Das Kurvenmodell kommt aus `geometry_features.FeatureModelV2` (Blockkanten-
Polylinien + Seam-Kurven). Eine generierte Blockkante wird als Kurven-Segment
zwischen ihren beiden (gesnappten) Endpunkten gesampelt; findet sich keine
gemeinsame Kurve, bleibt die Sehne (degenerierter Fall).

Flaechen: jede Blockflaeche ist durch ihre 4 Randkanten begrenzt und wird als
Coons-Patch (`curved_refill._coons`, Import-only) gebaut. Damit zwei Bloecke
dieselbe geteilte Flaeche bit-identisch sehen, wird jede Flaeche global unter
einem kanonischen Schluessel (4 Eckpunkt-Ids) genau EINMAL gebaut und fuer den
lokalen Block nur reorientiert (D4-Symmetrie des Gitters).

Emission: `emit_edge_records` erzeugt exakt den `export_sample._edge_records`-
Vertrag (gerichtete Kanten, edge_ctrl [E,2,3], edge_polyline + offset,
dir_class, params).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np

HEX3D_REPO = ("/home/t1dde/hydrostack_pipeline/stack/domain_partition_3D/"
              "experimentell/hex3d_algohex")
if HEX3D_REPO not in sys.path:
    sys.path.insert(0, HEX3D_REPO)
from curved_refill import _coons  # noqa: E402  (extern, read-only)

CORNERS = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
           (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))
OTHER = {0: (1, 2), 1: (0, 2), 2: (0, 1)}


def local_edges() -> list[tuple[int, int, int]]:
    """Die 12 Hex-Kanten als (lokaler Start, lokales Ende, Achse)."""
    out = []
    for i in range(8):
        for j in range(i + 1, 8):
            diff = [k for k in range(3) if CORNERS[i][k] != CORNERS[j][k]]
            if len(diff) == 1:
                out.append((i, j, diff[0]))
    return out


def face_cycle(axis: int, side: int) -> list[int]:
    """Die 4 lokalen Ecken einer Flaeche als (0,0),(1,0),(1,1),(0,1)-Zyklus."""
    o0, o1 = OTHER[axis]
    lc = [i for i in range(8) if CORNERS[i][axis] == side]
    lc.sort(key=lambda i: (CORNERS[i][o0], CORNERS[i][o1]))
    # Gitterpositionen (0,0),(1,0),(1,1),(0,1) in (o0,o1): Permutation aus sortiert
    return [lc[0], lc[2], lc[3], lc[1]]


# --------------------------------------------------------------------------
# Kurven-Sampling
# --------------------------------------------------------------------------

def _resample(Q: np.ndarray, n: int) -> np.ndarray:
    """Polyline auf `n` aequidistante Bogenlaengen-Punkte."""
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))])
    if s[-1] <= 1e-12:
        return np.repeat(Q[:1], n, axis=0)
    q = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(q, s, Q[:, k]) for k in range(3)], axis=1)


def _gt_edge_curve(fm, p0: np.ndarray, p1: np.ndarray, n: int, tol: float):
    """Exakte GT-Blockkante, falls beide Endpunkte GT-Vertices sind."""
    if not hasattr(fm, "edge_lookup"):
        return None
    dv, iv = fm.vtree.query(np.stack([p0, p1]), k=1)
    if dv[0] > tol or dv[1] > tol or int(iv[0]) == int(iv[1]):
        return None
    c = fm.edge_lookup.get((int(iv[0]), int(iv[1])))
    if c is None:
        return None
    Q = fm.edge_curves.segment(int(c))
    return _resample(Q, n), int(c)


def sample_on_curve(fm, p0: np.ndarray, p1: np.ndarray, n: int,
                    tol: float = 0.06, path_fn=None) -> tuple[np.ndarray, int]:
    """`n` Punkte von p0 nach p1 entlang der naechsten gemeinsamen Kurve.

    Optional `path_fn(p0, p1, n) -> (points, curve_id) | None` is a graph
    fallback (seam navigator) for endpoints on different curves; it is tried
    before the chord. Rueckgabe (Punkte, curve_id);
    curve_id=-1 bedeutet Sehne (degeneriert)."""
    exact = _gt_edge_curve(fm, p0, p1, n, tol)
    if exact is not None:
        return exact
    if path_fn is not None:
        res = path_fn(p0, p1, n)
        if res is not None:
            return res
    q = np.stack([p0, p1])[None]
    d, c, t, _ = fm.curves.nearest(q)
    d0, c0, t0, d1, c1, t1 = d[0], int(c[0]), t[0], d[1], int(c[1]), t[1]
    if c0 == c1 and d0 <= tol and d1 <= tol and abs(t1 - t0) > 1e-9:
        seg = _orient_segment(fm.curves.sample_segment(c0, t0, t1, n), p0)
        if _gentle(seg, p0, p1):
            return seg, c0
    de, ie = fm.curves.nearest_endpoint(q)
    ce0, ce1 = int(fm.curves.ep_curve[ie[0]]), int(fm.curves.ep_curve[ie[1]])
    if ce0 == ce1 and de[0] <= tol and de[1] <= tol:
        ta, tb = float(fm.curves.ep_t[ie[0]]), float(fm.curves.ep_t[ie[1]])
        if abs(tb - ta) > 1e-9:
            seg = _orient_segment(fm.curves.sample_segment(ce0, ta, tb, n), p0)
            if _gentle(seg, p0, p1):
                return seg, ce0
    return np.linspace(p0, p1, n), -1


def _orient_segment(seg: np.ndarray, p0: np.ndarray) -> np.ndarray:
    if np.linalg.norm(seg[0] - p0) > np.linalg.norm(seg[-1] - p0):
        return seg[::-1]
    return seg


def _gentle(seg: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> bool:
    """Kurvensegment nur akzeptieren, wenn es die Sehne nicht wild umfaehrt."""
    chord = float(np.linalg.norm(p1 - p0))
    if chord <= 1e-9:
        return False
    arc = float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum())
    # sqrt(2): a 90-degree detour needs ratio ~1.41; anything beyond 1.15
    # cannot be a gentle wall-hugging arc (matches the seam-router guard).
    if arc / chord > 1.15:
        return False
    if (seg[-1] - seg[0]) @ (p1 - p0) <= 0:
        return False
    return max(float(np.linalg.norm(seg[0] - p0)),
               float(np.linalg.norm(seg[-1] - p1))) <= 0.5 * chord + 1e-9


def _reversed(P: np.ndarray) -> np.ndarray:
    return P[::-1]


# --------------------------------------------------------------------------
# Feste Kurven aufloesen (global geteilt, kanonisch)
# --------------------------------------------------------------------------

@dataclass
class CurvedStructure:
    """Globale Kurvenkanten + kanonische Flaechengitter eines Blocksatzes."""

    edge_pts: dict = field(default_factory=dict)       # (min,max) -> (n,3)
    edge_curve: dict = field(default_factory=dict)     # (min,max) -> curve_id
    edge_dir: dict = field(default_factory=dict)       # (min,max) -> dir_class
    edge_len: dict = field(default_factory=dict)
    faces: dict = field(default_factory=dict)          # frozenset(ids) -> (G, cycle)
    dims: dict = field(default_factory=dict)           # block -> (ni,nj,nk)

    def get_edge(self, a: int, b: int) -> np.ndarray:
        key = (a, b) if a < b else (b, a)
        P = self.edge_pts[key]
        return P if a < b else _reversed(P)


def build_structures(fm, corner_ids: np.ndarray, C_snap: np.ndarray,
                     blocks: np.ndarray, counts: list[int],
                     cof: dict, path_fn=None,
                     edge_post_fn=None) -> CurvedStructure:
    """Kurven fuer jede Blockkante + kanonische Coons-Flaechengitter.

    `path_fn` is forwarded to sample_on_curve (seam-graph fallback).
    `edge_post_fn(st, corner_ids, C_snap, blocks)` runs after all edges are
    sampled and BEFORE the Coons faces are built, so an edge rewritten there
    still propagates into the faces."""
    st = CurvedStructure()
    for r in range(len(blocks)):
        st.dims[int(r)] = tuple(int(counts[cof[(int(r), ax)]]) for ax in (0, 1, 2))
    for r in range(len(blocks)):
        for li, lj, axis in local_edges():
            a, b = int(corner_ids[r, li]), int(corner_ids[r, lj])
            key = (a, b) if a < b else (b, a)
            if key in st.edge_pts:
                continue
            cls = cof[(int(r), axis)]
            n = int(counts[cls]) + 1
            pts, cid = sample_on_curve(fm, C_snap[r, li], C_snap[r, lj], n,
                                       path_fn=path_fn)
            if a > b:
                pts = _reversed(pts)
            st.edge_pts[key] = pts
            st.edge_curve[key] = cid
            st.edge_dir[key] = int(cls)
            st.edge_len[key] = n
    if edge_post_fn is not None:
        edge_post_fn(st, corner_ids, C_snap, blocks)
    for r in range(len(blocks)):
        for axis in (0, 1, 2):
            for side in (0, 1):
                cyc = face_cycle(axis, side)
                ids = [int(corner_ids[r, c]) for c in cyc]
                key = frozenset(ids)
                if key in st.faces:
                    continue
                canon, cids = _canon_cycle(ids)
                G = _canonical_face(st, canon)
                st.faces[key] = (G, cids)
    return st


def _canon_cycle(ids: list[int]) -> tuple[list[int], list[int]]:
    """Kanonischer Zyklus: kleinste Id zuerst, Richtung ueber kleineren Nachbarn."""
    i = int(np.argmin(ids))
    fwd = [ids[(i + k) % 4] for k in range(4)]
    bwd = [ids[(i - k) % 4] for k in range(4)]
    cyc = fwd if fwd[1] <= fwd[3] else bwd
    return cyc, cyc


def _canonical_face(st: CurvedStructure, cyc: list[int]) -> np.ndarray:
    """Coons-Gitter aus 4 kanonisch geordneten Kurvenkanten.

    `e0`/`e1` laufen in u, `e2`/`e3` in v; Kanten mit abweichender Sample-Zahl
    (nicht-konforme Nachbarbloecke) werden auf die Zielkantenlaenge resampled."""
    e0 = st.get_edge(cyc[0], cyc[1])
    e2 = st.get_edge(cyc[0], cyc[3])
    e1 = _resample(st.get_edge(cyc[3], cyc[2]), len(e0))
    e3 = _resample(st.get_edge(cyc[1], cyc[2]), len(e2))
    return _coons(e0, e1, e2, e3)


_TRANSFORMS = (lambda A: A, lambda A: A[::-1], lambda A: A[:, ::-1],
               lambda A: A[::-1, ::-1])


def orient_face(G: np.ndarray, canon: list[int], local: list[int],
                pts_of: dict, shape: tuple[int, int]) -> np.ndarray:
    """Gitter G (kanonische Ecken `canon`) auf lokale Eckenreihenfolge `local`.

    Die Zuordnung geschieht ueber die ECKEN-IDS (nicht Koordinaten), damit
    gesnappte Ecken, die nicht exakt auf der Kurve liegen, keine Rolle spielen;
    anschliessend werden die 4 Gitterecken exakt auf die gesnappten Punkte
    gesetzt. Shape-Filter trennt u/v (Transponierte)."""
    cid = np.full(G.shape[:2], -1, np.int64)
    cid[0, 0], cid[-1, 0], cid[-1, -1], cid[0, -1] = canon
    T = np.swapaxes(G, 0, 1)
    TC = np.swapaxes(cid, 0, 1)
    for A, Ac in ((G, cid), (T, TC)):
        for tf in _TRANSFORMS:
            B, Bc = tf(A), tf(Ac)
            if tuple(B.shape[:2]) != tuple(shape):
                continue
            if [int(Bc[0, 0]), int(Bc[-1, 0]), int(Bc[-1, -1]),
                    int(Bc[0, -1])] == list(local):
                B = np.array(B, float, copy=True)
                for pos, lid in zip(((0, 0), (-1, 0), (-1, -1), (0, -1)),
                                    local):
                    B[pos] = pts_of[lid]
                return B
    raise RuntimeError(f"keine D4-Orientierung: G{G.shape} shape{shape} "
                       f"canon{canon} local{local}")


# --------------------------------------------------------------------------
# Bezier-Kontrolle + Datensatz-Vertrag
# --------------------------------------------------------------------------

def bezier_ctrl(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """Kubischer Bezier (B1,B2) durch `pts` mit festen Endpunkten.

    Identisch zu `block_edges.fit_cubic_bezier` (Chord-Parametrisierung,
    quadratischer/heuristischer Fallback)."""
    from block_edges import fit_cubic_bezier  # extern, read-only
    B1, B2, mode = fit_cubic_bezier(pts[0], pts[-1], pts)
    return np.asarray(B1, float), np.asarray(B2, float), mode


def emit_edge_records(fm, corner_ids: np.ndarray, C_snap: np.ndarray,
                      blocks: np.ndarray, st: CurvedStructure,
                      params: dict | None = None) -> dict:
    """Datensatz-Format wie `export_sample.build` (edge_records-Vertrag).

    Gerichtete Kanten (beide Traversen), edge_ctrl je Richtung umgedreht,
    edge_polyline + offset, dir_class, params (JSON-String)."""
    import json
    P_parts, edges, ctrls, poly, poff, dcls, eidx = [], [], [], [], [0], [], {}
    seen = set()
    for r in range(len(blocks)):
        for li, lj, axis in local_edges():
            a, b = int(corner_ids[r, li]), int(corner_ids[r, lj])
            key = (a, b) if a < b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            q = st.get_edge(a, b)  # a -> b
            B1, B2, _m = bezier_ctrl(q)
            d = st.edge_dir[key]
            eidx[key] = len(edges) // 2
            P_parts.append(q)
            edges.append([a, b]); ctrls.append([B1, B2]); poly.append(q)
            poff.append(poff[-1] + len(q)); dcls.append(d)
            edges.append([b, a]); ctrls.append([B2, B1]); poly.append(q[::-1])
            poff.append(poff[-1] + len(q)); dcls.append(d)
    ctrl_by_pair = {}
    for key, i in eidx.items():
        ctrl_by_pair[key] = (np.asarray(ctrls[2 * i], float),
                             np.asarray(ctrls[2 * i + 1], float))
    return {
        "edge_index": ctrl_by_pair,
        "edges": np.asarray(edges, np.int64),
        "edge_ctrl": np.asarray(ctrls, float),
        "edge_polyline": np.concatenate(poly) if poly else np.zeros((0, 3)),
        "edge_polyline_offset": np.asarray(poff, np.int64),
        "dir_class": np.asarray(dcls, np.int64),
        "params": json.dumps(params or {}),
    }


def bucket_stats(fm, st: CurvedStructure) -> dict:
    """Histogramm: 1D (auf einer Kurve), 2D (Coons), degeneriert (Sehne)."""
    one_d = sum(1 for v in st.edge_curve.values() if v >= 0)
    deg = sum(1 for v in st.edge_curve.values() if v < 0)
    return {"buckets_1d": int(one_d), "buckets_2d": int(st.faces.__len__()),
            "degenerate_edges": int(deg), "edges_total": int(len(st.edge_curve))}
