"""
blade_inject.py

Re-Injektion der BEKANNTEN Blade-Geometrie nach dem Detokenize.

Idee: die Blade-Form ist der CFD-Input und damit exakt bekannt. Der Transformer
muss sie nicht reproduzieren. Nach Detokenize werden:
  1. die Blade-Knoten in der rekonstruierten Blockstruktur erkannt (Naehe zur
     bekannten Blade-Polyline),
  2. auf die Bogenlaenge der Blade projiziert/gesnappt,
  3. die Blade-Block-Kanten durch die EXAKTEN Teilboegen der Blade ersetzt.
Danach Coons-TFI -> Wand exakt.

Ansatz #1 (post-hoc, kein Retraining).
"""

import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict, List, Optional, Tuple

from reconstruct_domain import (polar_to_cartesian, merge_duplicate_vertices,
                                build_hermite_edges, coons_patch_2d,
                                quadratic_bezier_2d, _resample_curve)


# ----------------------------------------------------------------------
# Bekannte Blade extrahieren (aus den Mesh-Daten; ersetzt spaeter CFD-Input)
# ----------------------------------------------------------------------
def _poly_area(loop: np.ndarray) -> float:
    x, y = loop[:, 0], loop[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def get_known_blade(mesh: Dict, min_area_frac: float = 1e-3) -> Optional[np.ndarray]:
    """
    Liefert die bekannte Blade als geschlossene Polyline [N, 2] (CCW, ohne
    Duplikat am Ende) oder None, falls keine gefunden.

    Robust ueber beide Repraesentationen:
      - grob: anti-paralleles Streamline-Paar (u,v)&(v,u) das Flaeche einschliesst
      - fein: Kette von Wand-Randkanten (Randkante, nicht auf aeusserem Rechteck),
              zu geschlossenem Loop verkettet
    """
    bn = mesh['vertices_cartesian'].numpy()
    faces = mesh['faces'].numpy()
    e2s = {k: np.asarray(v, float) for k, v in mesh['edge_to_streamline'].items()}
    xmin, ymin = bn.min(0); xmax, ymax = bn.max(0)
    dom_area = max((xmax - xmin) * (ymax - ymin), 1e-9)
    tol = 1e-3 * max(xmax - xmin, ymax - ymin)

    def on_rect(p):
        return ((np.abs(p[:, 0] - xmin) < tol) | (np.abs(p[:, 0] - xmax) < tol) |
                (np.abs(p[:, 1] - ymin) < tol) | (np.abs(p[:, 1] - ymax) < tol))

    # --- grob: anti-paralleles Paar mit groesster eingeschlossener Flaeche ---
    best = None; best_area = min_area_frac * dom_area
    for (u, v), a in e2s.items():
        if (v, u) in e2s and u < v:
            b = e2s[(v, u)]
            loop = np.vstack([a, b])
            ar = _poly_area(loop)
            if ar > best_area:
                best_area = ar; best = loop
    if best is not None:
        return _dedup_closed(best)

    # --- fein: Wand-Randkanten verketten ---
    from collections import Counter
    ec = Counter()
    for fi in range(faces.shape[1]):
        f = faces[:, fi]
        for i in range(4):
            ec[frozenset((int(f[i]), int(f[(i + 1) % 4])))] += 1
    wall_edges = []
    for e, c in ec.items():
        if c != 1:
            continue
        u, v = tuple(e)
        pts = e2s.get((u, v))
        rev = False
        if pts is None:
            pts = e2s.get((v, u)); rev = True
        if pts is None:
            continue
        if on_rect(pts).mean() < 0.5:
            wall_edges.append((v, u, pts[::-1]) if rev else (u, v, pts))
    if not wall_edges:
        return None
    loop = _chain_edges(wall_edges)
    return _dedup_closed(loop) if loop is not None else None


def _chain_edges(edges: List[Tuple[int, int, np.ndarray]]) -> Optional[np.ndarray]:
    """
    Verkette Wand-Kanten (u,v,pts) UNGERICHTET zu einer geschlossenen Polyline.
    Streamline-Orientierung wird beim Laufen head-to-tail angepasst.
    """
    from collections import defaultdict
    adj = defaultdict(list)   # node -> list of (other, pts oriented node->other, edge_id)
    for eid, (u, v, pts) in enumerate(edges):
        pts = np.asarray(pts, float)
        adj[u].append((v, pts, eid))
        adj[v].append((u, pts[::-1], eid))

    # Start: Knoten mit Grad 1 (offene Kette) sonst beliebig (geschlossen)
    start = None
    for nprt in adj:
        if len(adj[nprt]) == 1:
            start = nprt; break
    if start is None:
        start = edges[0][0]

    loop = []; used = set(); cur = start
    for _ in range(len(edges) + 1):
        nxt = None
        for other, pts, eid in adj[cur]:
            if eid not in used:
                nxt = (other, pts, eid); break
        if nxt is None:
            break
        other, pts, eid = nxt
        used.add(eid)
        loop.append(pts if not loop else pts[1:])
        cur = other
        if cur == start:
            break
    if not loop:
        return None
    return np.vstack(loop)


def _dedup_closed(poly: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Aufeinanderfolgende Duplikate entfernen; nicht schliessen (offen halten)."""
    keep = [poly[0]]
    for p in poly[1:]:
        if np.linalg.norm(p - keep[-1]) > tol:
            keep.append(p)
    out = np.array(keep)
    if len(out) > 1 and np.linalg.norm(out[0] - out[-1]) < tol:
        out = out[:-1]
    return out


# ----------------------------------------------------------------------
# Arc-length Parametrisierung der Blade
# ----------------------------------------------------------------------
class BladeArc:
    """Geschlossene Blade-Polyline mit Bogenlaengen-Parametrisierung s in [0,1)."""

    def __init__(self, poly: np.ndarray):
        self.poly = np.asarray(poly, float)
        closed = np.vstack([self.poly, self.poly[:1]])
        seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        self.total = float(seg.sum())
        self.s_nodes = np.concatenate([[0.0], np.cumsum(seg)]) / max(self.total, 1e-12)
        self._closed = closed

    def project(self, pt: np.ndarray) -> float:
        """Naechster Punkt auf Polyline -> s in [0,1)."""
        pt = np.asarray(pt, float)
        C = self._closed
        best_s = 0.0; best_d = 1e18
        for i in range(len(C) - 1):
            a, b = C[i], C[i + 1]
            ab = b - a; L2 = float(ab @ ab)
            t = 0.0 if L2 < 1e-18 else float(np.clip((pt - a) @ ab / L2, 0, 1))
            proj = a + t * ab
            d = float(np.linalg.norm(pt - proj))
            if d < best_d:
                best_d = d
                best_s = (self.s_nodes[i] + t * (self.s_nodes[i + 1] - self.s_nodes[i]))
        return best_s % 1.0

    def point(self, s: float) -> np.ndarray:
        s = s % 1.0
        return np.array([np.interp(s, self.s_nodes, self._closed[:, 0]),
                         np.interp(s, self.s_nodes, self._closed[:, 1])])

    def subarc_dir(self, s0: float, s1: float, sign: int, n: int) -> np.ndarray:
        """Teilbogen von s0 nach s1 in Richtung sign (+1 vorwaerts, -1 rueckwaerts)."""
        s0 %= 1.0; s1 %= 1.0
        if sign > 0:
            d = (s1 - s0) % 1.0
        else:
            d = -((s0 - s1) % 1.0)
        ss = (s0 + np.linspace(0, d, n)) % 1.0
        return np.array([self.point(s) for s in ss])


# ----------------------------------------------------------------------
# Reconstruct mit exakter Blade-Injektion
# ----------------------------------------------------------------------
def reconstruct_domain_coons_blade(tokenizer_output: Dict, center: np.ndarray,
                                   blade_poly: np.ndarray, n: int = 11,
                                   snap_tol_frac: float = 0.06,
                                   merge_tol: float = 1e-4) -> Tuple[Data, dict]:
    """
    Wie reconstruct_domain_coons, aber Blade-Knoten werden auf die bekannte Blade
    gesnappt und Blade-Kanten durch exakte Teilboegen ersetzt.

    Returns: (quad_mesh Data, info dict mit Diagnostik)
    """
    from collections import defaultdict
    from torch_geometric.data import Data
    places = tokenizer_output['vertex_places']
    faces_as_places = tokenizer_output['faces_as_places']
    cart = polar_to_cartesian(places, center)
    _, p2u = merge_duplicate_vertices(cart, threshold=1e-3)

    arc = BladeArc(blade_poly)
    bbox = float(np.linalg.norm(blade_poly.max(0) - blade_poly.min(0)))
    snap_tol = snap_tol_frac * bbox

    # Blade-Knoten: unique Knoten nahe der Blade
    n_unique = max(p2u.values()) + 1
    upos = np.zeros((n_unique, 2))
    for p, u in p2u.items():
        upos[u] = cart[p]
    blade_node = {}
    for u in range(n_unique):
        s = arc.project(upos[u])
        d = np.linalg.norm(upos[u] - arc.point(s))
        if d < snap_tol:
            blade_node[u] = s
            upos[u] = arc.point(s)   # snap Position exakt auf Blade

    # Wand-Kanten = KONSEKUTIVE Blade-Knoten entlang der Bogenlaenge (geschlossen).
    # Nur diese Node-Paare duerfen durch exakte Blade-Teilboegen ersetzt werden
    # (verhindert, dass Diagonalen quer ueber die Blade faelschlich gesnappt werden).
    ordered = sorted(blade_node, key=lambda u: blade_node[u])
    wall_pairs = set()
    if len(ordered) >= 2:
        for i in range(len(ordered)):
            a, b = ordered[i], ordered[(i + 1) % len(ordered)]
            wall_pairs.add(frozenset((a, b)))

    # deduped Kantenkurven (quadratisch) als Fallback fuer Nicht-Blade-Kanten
    ei, ep = build_hermite_edges(places, cart, faces_as_places, p2u)
    ded = defaultdict(list)
    for i in range(ei.shape[1]):
        a, b = int(ei[0, i]), int(ei[1, i])
        ded[frozenset((a, b))].append(np.asarray(ep[i], float))

    def edge_curve(p0, p1):
        a, b = p2u[p0], p2u[p1]
        # eigene rekonstruierte Quadratik (kodiert ueber Tangenten die richtige Seite)
        d0 = np.array([np.cos(places[p0]['alpha_out']), np.sin(places[p0]['alpha_out'])])
        d2 = np.array([np.cos(places[p1]['alpha_in']), np.sin(places[p1]['alpha_in'])])
        own = quadratic_bezier_2d(cart[p0], cart[p1], d0, d2, n)
        ownmid = own[len(own) // 2]

        # Blade-Kante? nur konsekutive Blade-Knoten -> exakter Teilbogen.
        # Richtungswahl: der Bogen, der KEINEN anderen Blade-Knoten enthaelt
        # (= direkte Wand-Segment). Bei Gleichstand (2-Knoten-Blade: Saug/Druck)
        # per Match auf die eigene rekonstruierte Kanten-Mitte disambiguieren.
        if frozenset((a, b)) in wall_pairs:
            s0, s1 = blade_node[a], blade_node[b]
            others = [s for u, s in blade_node.items() if u not in (a, b)]

            def interior_count(sign):
                d = (s1 - s0) % 1.0 if sign > 0 else -((s0 - s1) % 1.0)
                lo, hi = (s0, (s0 + d) % 1.0) if sign > 0 else ((s0 + d) % 1.0, s0)
                cnt = 0
                for s in others:
                    inside = (lo < s < hi) if lo < hi else (s > lo or s < hi)
                    cnt += inside
                return cnt

            nf, nb = interior_count(+1), interior_count(-1)
            if nf != nb:
                sign = +1 if nf < nb else -1
            else:
                fwd = arc.subarc_dir(s0, s1, +1, n); bwd = arc.subarc_dir(s0, s1, -1, n)
                sign = +1 if (np.linalg.norm(fwd[len(fwd)//2] - ownmid)
                              <= np.linalg.norm(bwd[len(bwd)//2] - ownmid)) else -1
            return arc.subarc_dir(s0, s1, sign, n)

        # Fallback: deduped Quadratik, orientiert a->b
        cands = ded.get(frozenset((a, b)), [own])
        c = min(cands, key=lambda cc: np.linalg.norm(cc[len(cc)//2] - ownmid))
        c = np.asarray(c, float)
        if np.linalg.norm(c[0] - upos[a]) > np.linalg.norm(c[-1] - upos[a]):
            c = c[::-1]
        c = _resample_curve(c, n).copy()
        c[0] = upos[a]; c[-1] = upos[b]
        return c

    node_map = {}; nodes = []
    def node_id(pt):
        key = (round(float(pt[0]) / merge_tol), round(float(pt[1]) / merge_tol))
        if key not in node_map:
            node_map[key] = len(nodes); nodes.append([float(pt[0]), float(pt[1])])
        return node_map[key]

    quads = []
    for face in faces_as_places:
        e01 = edge_curve(face[0], face[1]); e12 = edge_curve(face[1], face[2])
        e23 = edge_curve(face[2], face[3]); e30 = edge_curve(face[3], face[0])
        grid = coons_patch_2d(e01, e23[::-1], e30[::-1], e12, n)
        ids = np.empty((n, n), dtype=np.int64)
        for iv in range(n):
            for iu in range(n):
                ids[iv, iu] = node_id(grid[iv, iu])
        for iv in range(n - 1):
            for iu in range(n - 1):
                quads.append([ids[iv, iu], ids[iv, iu + 1],
                              ids[iv + 1, iu + 1], ids[iv + 1, iu]])

    x = torch.tensor(nodes, dtype=torch.float32)
    faces_t = torch.tensor(quads, dtype=torch.long).T
    info = {'n_blade_nodes': len(blade_node), 'blade_total_len': arc.total}
    return Data(x=x, faces=faces_t), info


def arc_len(c: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(np.asarray(c, float), axis=0), axis=1).sum())
