"""
augment_subdivide.py

Daten-Augmentation fuer die 6-Face-Blockstrukturen: jeder higher-order Block wird
per transfiniter Coons-Interpolation isotrop in n x n Sub-Zellen unterteilt. Erzeugt
neue, gueltige Mesh-Topologien (Facecount 6 -> 6*n^2) auf DERSELBEN Domain.

Wichtig (Blade):
  - Die Blade-Lens (Druck/Saug, z.B. Kante v10-v11) ist KEIN Block, sondern ein Loch.
    Sie wird nie unterteilt -> das Loch bleibt automatisch erhalten.
  - Die an das Blade grenzenden Bloecke (F2/F5) DUERFEN unterteilt werden; dabei wird
    ihre Blade-Randkurve (Druck- bzw. Saugseite) feiner aufgeloest. Druck und Saug sind
    verschiedene Kurven (Loch dazwischen) -> ihre inneren Sub-Punkte fallen NICHT
    zusammen und werden nicht dedupliziert.

Konformitaet (kein haengender Knoten):
  Jede Blockkante wird gleichmaessig in n Segmente (nach BOGENLAENGE der echten Kurve)
  geteilt. Eine von zwei Bloecken geteilte Innenkante ist dieselbe Kurve (nur umgekehrt)
  -> die n Teilungspunkte fallen zusammen. Globale Vertex-Deduplizierung per Toleranz.

Dieses Modul baut zuerst die Geometrie (Coons + Subdivision) und ein Validierungsbild.
Der volle mesh_data-Aufbau (edge_index/tangents/streamline) + Batch folgt in
build_subdivided_mesh / augment_dataset.
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# Kurven-Auswertung nach Bogenlaenge
# ------------------------------------------------------------------
def eval_curve(pts, t):
    """Kurve pts [N,2] an Parameter t in [0,1] (nach Bogenlaenge) auswerten.
    t skalar oder [K] -> Punkt(e) [2] bzw [K,2]."""
    pts = np.asarray(pts, float)
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] < 1e-12:
        t = np.atleast_1d(t)
        return np.repeat(pts[:1], len(t), axis=0) if len(t) > 1 else pts[0]
    s /= s[-1]
    t = np.atleast_1d(np.asarray(t, float))
    x = np.interp(t, s, pts[:, 0])
    y = np.interp(t, s, pts[:, 1])
    out = np.stack([x, y], axis=1)
    return out[0] if out.shape[0] == 1 else out


def block_boundary(e2s, quad, vc):
    """Vier Randkurven eines Blocks [p0,p1,p2,p3] in Coons-Konvention zurueckgeben:
       Cb (p0->p1), Ct (p3->p2), Cl (p0->p3), Cr (p1->p2)  -- jeweils Param 0..1.

    MULTIGRAPH-TREU: die Randkurven werden aus den GERICHTETEN Face-Loop-Kanten
    (p0->p1->p2->p3->p0) geholt. An der Blade-Lens existieren beide Richtungen
    (u,v)/(v,u) als VERSCHIEDENE Kurven (Druck/Saug) -> es MUSS die vom Face-Loop
    traversierte Richtung genommen werden, sonst greift der Block ueber die Lens ins
    Loch (fuellt es). Frueher: naives e2s.get((u,v)) -> beide Blade-Bloecke nahmen die
    falsche (gegenueberliegende) Lens-Kurve -> Loch gefuellt."""
    p0, p1, p2, p3 = [int(x) for x in quad]

    def dedge(a, b):
        """Kurve entlang gerichteter Face-Kante a->b (bevorzugt die in a->b gespeicherte)."""
        c = e2s.get((a, b))
        if c is not None:
            return np.asarray(c, float)
        c = e2s.get((b, a))
        if c is not None:
            return np.asarray(c, float)[::-1]
        return np.stack([vc[a], vc[b]])          # Fallback Gerade

    Cb = dedge(p0, p1)              # bottom v=0  (Loop-Kante p0->p1)
    Cr = dedge(p1, p2)             # right  u=1  (Loop-Kante p1->p2)
    Ct = dedge(p2, p3)[::-1]       # top    v=1  (Loop-Kante p2->p3, gedreht -> p3->p2)
    Cl = dedge(p3, p0)[::-1]       # left   u=0  (Loop-Kante p3->p0, gedreht -> p0->p3)
    return Cb, Ct, Cl, Cr, (vc[p0], vc[p1], vc[p2], vc[p3])


def coons_grid(Cb, Ct, Cl, Cr, corners, nu, nv):
    """Transfinite Coons-Interpolation -> Gitter [(nv+1),(nu+1),2] an gleichmaessigen
    (u,v). nu/nv = Anzahl Segmente in u/v."""
    P00, P10, P11, P01 = corners
    us = np.linspace(0, 1, nu + 1)
    vs = np.linspace(0, 1, nv + 1)
    cb = eval_curve(Cb, us); ct = eval_curve(Ct, us)     # [(nu+1),2]
    cl = eval_curve(Cl, vs); cr = eval_curve(Cr, vs)     # [(nv+1),2]
    U, V = np.meshgrid(us, vs)                            # [(nv+1),(nu+1)]
    grid = (
        (1 - V)[..., None] * cb[None, :, :] +
        V[..., None] * ct[None, :, :] +
        (1 - U)[..., None] * cl[:, None, :] +
        U[..., None] * cr[:, None, :] -
        ((1 - U) * (1 - V))[..., None] * P00 -
        (U * (1 - V))[..., None] * P10 -
        (U * V)[..., None] * P11 -
        ((1 - U) * V)[..., None] * P01
    )
    return grid, us, vs


def coons_edge_curve(Cb, Ct, Cl, Cr, corners, along, fixed, t0, t1, m=16):
    """Kurve einer Sub-Zellkante: entlang 'u' (v=fixed) oder 'v' (u=fixed),
    Parameter von t0..t1, m Punkte -> [m,2] (gekruemmt via Coons)."""
    P00, P10, P11, P01 = corners
    tt = np.linspace(t0, t1, m)
    if along == 'u':
        v = fixed; us = tt; vs = np.full(m, v)
    else:
        u = fixed; vs = tt; us = np.full(m, u)
    cb = eval_curve(Cb, us); ct = eval_curve(Ct, us)
    cl = eval_curve(Cl, vs); cr = eval_curve(Cr, vs)
    U = us; V = vs
    pts = (
        (1 - V)[:, None] * cb + V[:, None] * ct +
        (1 - U)[:, None] * cl + U[:, None] * cr -
        ((1 - U) * (1 - V))[:, None] * P00 - (U * (1 - V))[:, None] * P10 -
        (U * V)[:, None] * P11 - ((1 - U) * V)[:, None] * P01
    )
    return pts


# ------------------------------------------------------------------
# Volles mesh_data bauen (Dedup + faces + streamline + tangents + polar)
# ------------------------------------------------------------------
DEDUP_TOL = 1e-6


def _dedup_points(pts, tol=DEDUP_TOL):
    """Rundungs-Hash-Dedup. Geteilte Randpunkte sind numerisch identisch (dieselbe
    Bogenlaengen-Auswertung) -> gleicher Bucket; Blade Druck/Saug sind >>tol getrennt.
    Returns (uniq [U,2], idxmap [len(pts)])."""
    scale = 1.0 / tol
    key2idx = {}
    uniq = []
    idxmap = np.empty(len(pts), dtype=np.int64)
    for i, p in enumerate(pts):
        key = (int(round(p[0] * scale)), int(round(p[1] * scale)))
        gi = key2idx.get(key)
        if gi is None:
            gi = len(uniq); key2idx[key] = gi; uniq.append(p)
        idxmap[i] = gi
    return np.asarray(uniq, float), idxmap


def _curve_tangents(curve):
    """Hermite-Tangenten (alpha,tn) an beiden Enden via cubic-Bezier-Fit -> [4]."""
    from meshtron.data.polytron_tokenizer import PolytronTokenizer
    P0, P1 = curve[0], curve[-1]
    B1, B2 = PolytronTokenizer._fit_cubic_bezier(P0, P1, curve)
    T0 = 3.0 * (B1 - P0); T1 = 3.0 * (P1 - B2)
    return np.array([np.arctan2(T0[1], T0[0]), np.linalg.norm(T0),
                     np.arctan2(T1[1], T1[0]), np.linalg.norm(T1)], dtype=np.float32)


# ------------------------------------------------------------------
# C1-Kanten-Glaettung (lokal, minimal-invasiv).
# Blockweise Coons-TFI erzeugt an den Block-Grenzen Tangenten-KNICKE: die inneren
# Gitterlinien von Block A und Block B treffen sich am geteilten Sub-Vertex mit
# verschiedenen Tangenten. Fix OHNE Vertices/Rand zu veraendern: nur die INNEREN
# Sub-Kanten (nicht auf Original-Block-Grenzen) als kubische Hermite mit gemeinsamer
# Tangente an jedem Durchlauf neu ziehen -> C1. Rand-Kanten (Blade-Kontur, Domain-Box,
# Original-Block-Grenzen = echte Streamlines) bleiben UNVERAENDERT.
# ------------------------------------------------------------------
def _hermite_curve(P0, P1, T0, T1, m=16):
    """Kubische Hermite P0->P1 mit End-Tangenten T0,T1 -> Punktkette [m,2] (via Bezier)."""
    B1 = P0 + T0 / 3.0; B2 = P1 - T1 / 3.0
    t = np.linspace(0, 1, m)[:, None]; mt = 1 - t
    return mt**3 * P0 + 3 * mt**2 * t * B1 + 3 * mt * t**2 * B2 + t**3 * P1


def _c1_align(uniq, new_e2s, boundary, m_curve=16):
    """Innere Sub-Kanten tangentenstetig machen. boundary = set(frozenset({a,b})) der
    Sub-Kanten auf Original-Block-Grenzen (fix). Vertices unveraendert."""
    pos = np.asarray(uniq, float)
    from collections import defaultdict
    outg = defaultdict(list)
    for (a, b) in new_e2s:
        outg[a].append(b)

    def sdir(a, b):
        c = np.asarray(new_e2s[(a, b)], float)
        t = c[1] - c[0]; L = np.linalg.norm(t)
        if L > 1e-12:
            return t / L
        d = pos[b] - pos[a]; return d / max(np.linalg.norm(d), 1e-12)

    # Ziel-Einheitstangente je gerichteter Kante (an Startvertex, Richtung Ziel)
    tdir = {}
    for v, ws in outg.items():
        dirs = {w: sdir(v, w) for w in ws}
        for w in ws:
            if frozenset((v, w)) in boundary:
                tdir[(v, w)] = dirs[w]; continue          # Rand -> unveraendert
            best = 1.0; opp = None
            for x in ws:
                if x == w:
                    continue
                dp = float(np.dot(dirs[w], dirs[x]))
                if dp < best:
                    best = dp; opp = x
            if opp is not None and best < -0.5:            # echte durchlaufende Linie
                if frozenset((v, opp)) in boundary:
                    tdir[(v, w)] = -dirs[opp]              # Rand-Tangente glatt fortsetzen
                else:
                    d = pos[w] - pos[opp]; L = np.linalg.norm(d)
                    tdir[(v, w)] = d / L if L > 1e-12 else dirs[w]   # Catmull-Rom
            else:
                tdir[(v, w)] = dirs[w]                     # kein Durchlauf -> behalten

    done = set()
    for (a, b) in list(new_e2s):
        key = frozenset((a, b))
        if key in boundary or key in done:
            continue
        done.add(key)
        chord = np.linalg.norm(pos[b] - pos[a])
        cab = _hermite_curve(pos[a], pos[b], tdir[(a, b)] * chord, -tdir[(b, a)] * chord, m_curve)
        new_e2s[(a, b)] = cab.astype(np.float32)
        new_e2s[(b, a)] = cab[::-1].astype(np.float32)
    return new_e2s


def build_subdivided_mesh(d, n, m_curve=16, smooth=False):
    """Ein 6-Face-mesh_data isotrop n x n unterteilen -> neues mesh_data (gleiches
    Format/keys). Domain (center/bounds/tri_coordinates) bleibt unveraendert.
    smooth=True: innere TFI-Kanten C1-glaetten (Vertices + Blade/Box fix)."""
    vc = d['vertices_cartesian'].numpy()
    faces = d['faces'].numpy().T
    e2s = d['edge_to_streamline']
    center = d['center'].numpy()

    # 1) pro Block Coons-Gitter, alle Punkte sammeln
    all_pts = []
    blocks = []            # (gi [n+1,n+1], coons)
    for quad in faces:
        coons = block_boundary(e2s, quad, vc)     # (Cb,Ct,Cl,Cr,corners)
        grid, _, _ = coons_grid(*coons, n, n)     # [n+1(v), n+1(u), 2]
        gi = np.empty((n + 1, n + 1), np.int64)
        for a in range(n + 1):
            for b in range(n + 1):
                gi[a, b] = len(all_pts); all_pts.append(grid[a, b])
        blocks.append((gi, coons))
    uniq, idxmap = _dedup_points(all_pts)

    # 2) Sub-Faces + gerichtete Kanten-Kurven; Block-Rand-Sub-Kanten merken (Original-Grenzen)
    new_faces = []
    new_e2s = {}
    boundary = set()
    for gi, coons in blocks:
        Cb, Ct, Cl, Cr, corners = coons
        for j in range(n):            # v
            for i in range(n):        # u
                g00 = int(idxmap[gi[j,     i]]);     g10 = int(idxmap[gi[j,     i + 1]])
                g11 = int(idxmap[gi[j + 1, i + 1]]); g01 = int(idxmap[gi[j + 1, i]])
                quad = [g00, g10, g11, g01]
                new_faces.append(quad)
                u0, u1 = i / n, (i + 1) / n
                v0, v1 = j / n, (j + 1) / n
                segs = [
                    (g00, g10, coons_edge_curve(*coons, 'u', v0, u0, u1, m_curve), j == 0),
                    (g10, g11, coons_edge_curve(*coons, 'v', u1, v0, v1, m_curve), i == n - 1),
                    (g11, g01, coons_edge_curve(*coons, 'u', v1, u1, u0, m_curve), j == n - 1),
                    (g01, g00, coons_edge_curve(*coons, 'v', u0, v1, v0, m_curve), i == 0),
                ]
                for a, b, cv, on_bd in segs:
                    new_e2s[(a, b)] = cv.astype(np.float32)
                    if on_bd:
                        boundary.add(frozenset((a, b)))

    if smooth:
        new_e2s = _c1_align(uniq, new_e2s, boundary, m_curve)

    # 3) Tensoren im Originalformat
    uniq = np.asarray(uniq, float)
    M = len(uniq)
    faces_t = torch.tensor(np.asarray(new_faces).T, dtype=torch.long)     # [4,F']
    rel = uniq - center
    r = np.linalg.norm(rel, axis=1)
    th = np.arctan2(rel[:, 1], rel[:, 0])
    vpol = torch.tensor(np.stack([r, th], axis=1), dtype=torch.float64)
    ekeys = list(new_e2s.keys())
    edge_index = torch.tensor(np.asarray(ekeys).T, dtype=torch.long)       # [2,E]
    etan = torch.tensor(np.stack([_curve_tangents(new_e2s[k]) for k in ekeys]),
                        dtype=torch.float32)                               # [E,4]

    out = {
        'vertices_cartesian': torch.tensor(uniq, dtype=torch.float64),
        'vertices_polar': vpol,
        'faces': faces_t,
        'edge_index': edge_index,
        'edge_tangents': etan,
        'edge_to_streamline': {k: np.asarray(v, np.float32) for k, v in new_e2s.items()},
        'center': d['center'].clone(),
        'bounds': d['bounds'].clone(),
        'tri_coordinates': d['tri_coordinates'].clone(),   # Domain unveraendert
        'subdiv_n': n,
    }
    return out


def augment_dataset(data, ns=(2, 3, 4), keep_original=True, only6=True, limit=None,
                    smooth=False):
    """6-Face-Meshes isotrop unterteilen. Returns Liste neuer mesh_data."""
    src = [d for d in data if (not only6 or d['faces'].shape[1] == 6)]
    if limit:
        src = src[:limit]
    out = []
    for d in src:
        if keep_original:
            out.append(d)
        for n in ns:
            out.append(build_subdivided_mesh(d, n, smooth=smooth))
    return out


# ------------------------------------------------------------------
# Validierungs-Plot: Original + Subdivision n=2,3,4
# ------------------------------------------------------------------
def _plot_subdiv(d, ns, path):
    vc = d['vertices_cartesian'].numpy()
    faces = d['faces'].numpy().T
    e2s = d['edge_to_streamline']
    fig, axes = plt.subplots(1, len(ns), figsize=(5.2 * len(ns), 5.2))
    if len(ns) == 1:
        axes = [axes]
    cmap = plt.get_cmap('tab10')
    for ax, n in zip(axes, ns):
        n_cells = 0
        for fi, quad in enumerate(faces):
            Cb, Ct, Cl, Cr, corners = block_boundary(e2s, quad, vc)
            # gekruemmte Sub-Gitterlinien zeichnen
            for j in range(n + 1):           # u-Linien (v = j/n)
                seg = coons_edge_curve(Cb, Ct, Cl, Cr, corners, 'u', j / n, 0, 1, m=6 * n + 4)
                ax.plot(seg[:, 0], seg[:, 1], color=cmap(fi % 10), lw=0.7)
            for i in range(n + 1):           # v-Linien (u = i/n)
                seg = coons_edge_curve(Cb, Ct, Cl, Cr, corners, 'v', i / n, 0, 1, m=6 * n + 4)
                ax.plot(seg[:, 0], seg[:, 1], color=cmap(fi % 10), lw=0.7)
            n_cells += n * n
        # Blade-Lens rot druebermalen
        for (u, v) in e2s:
            if (v, u) in e2s:
                a = np.asarray(e2s[(u, v)], float)
                b = np.asarray(e2s[(v, u)], float)[::-1]
                if a.shape == b.shape and np.max(np.linalg.norm(a - b, axis=1)) > 0.02:
                    ax.plot(a[:, 0], a[:, 1], color='crimson', lw=2.6, zorder=5)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(f"n={n}  ->  {n_cells} Faces", fontsize=11)
    fig.suptitle("Coons-Subdivision (isotrop n x n) — rot = Blade-Lens (nie unterteilt)",
                 fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches='tight')
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='domain_data.pt')
    ap.add_argument('--idx', type=int, default=None)
    args = ap.parse_args()
    data = torch.load(args.data, weights_only=False)
    idx = args.idx if args.idx is not None else \
        next(i for i, d in enumerate(data) if d['faces'].shape[1] == 6)
    _plot_subdiv(data[idx], [2, 3, 4], "figures/augment/01_subdivision.png")


if __name__ == '__main__':
    main()
