"""
augment_subdivide_3d.py -- Geometrie-Kern (v1, getestet).

Subdividiert Hex-Block-sample.npz (VTK-Eckenordnung) per
Gordon-Hall-Coons-Volumen-TFI in n x n x n Sub-Hexes.

Konformitaet: Jede Kante des Komplexes wird EINMAL kanonisch (kleinere
Vertex-id -> groessere) nach Boegenlaenge gleichmaessig (n+1 Punkte)
gesampelt; jede Face des Komplexes wird EINMAL per 2D-Coons aus denselben 4
Kantkurven (Cache ueber frozenset der 4 Corner-ids) gebaut. Zwei Bloecke,
die eine Face teilen, lesen dieselben Gitterwerte (dihedral umindiziert).
Beweis (m0019): Face-Borders == Kantkurven-Samples exakt; Coons-Face ==
Blockface-Ebene. Dedup: cKDTree + Union-Find, rep = KLEINSTER Index
(Port von tfi.weld, kein round-hash-Bug).

Handedness: traege Block-Ordnung (negativer Volumen-Jacobian wird anhand der
KORNER-Koordinaten erkannt) wird DURCH SPIEGELUNG DES AX-0-INDEX BEHOBEN
(wie tfi._fix_handedness) -- dh vor dem slotting, so dass Face-Cache und
Kantkurven (unabhaengig von Block-Indexkonvention) unveraendert bleiben.
"""

import numpy as np

DEDUP_TOL = 1e-7

CORNER = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
          (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))
CORNER_IDX = {c: i for i, c in enumerate(CORNER)}


# ------------------------------------------------------------------
# Kurven nach Boegenlaenge
# ------------------------------------------------------------------
def arc_resample(poly, k):
    """Polyline [M,3] an k aequidistanten Boegenlaengen-Stellen -> [k,3]."""
    poly = np.asarray(poly, float)
    if len(poly) < 3:
        t = np.linspace(0, 1, k)[:, None]
        return (1 - t) * poly[0] + t * poly[1]
    d = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] < 1e-12:
        return np.repeat(poly[:1], k, axis=0)
    s /= s[-1]
    t = np.linspace(0, 1, k)
    return np.stack([np.interp(t, s, poly[:, i]) for i in range(3)], axis=1)


def canonical_edge_lines(edges, edge_polyline, edge_polyline_offset):
    """undirected (u,v), u<v -> rohe Polyline, orientiert u -> v.
    CSR-Zeile e speichert Kurve im Richtung (edges[e,0] -> edges[e,1])."""
    out = {}
    for e in range(len(edges)):
        a, b = int(edges[e, 0]), int(edges[e, 1])
        key = (min(a, b), max(a, b))
        if key in out:
            continue
        lo, hi = int(edge_polyline_offset[e]), int(edge_polyline_offset[e + 1])
        poly = np.asarray(edge_polyline[lo:hi], float)
        if a > b:
            poly = poly[::-1]
        out[(u_pos := key)] = poly
    return out


class EdgeCurves:
    """Arc-laengen-gleichmaessige Samples (n+1 Punkte), pro ungerichteter
    Kant einmal gesampelt; Aufruf mit gewuenschter Richtung."""

    def __init__(self, raw, n):
        self.raw = raw
        self.n = n
        self.cache = {}

    def __call__(self, u, v):
        key = (min(u, v), max(u, v))
        c = self.cache.get(key)
        if c is None:
            if key not in self.raw:
                raise KeyError(f"edge {key} hat keine gespeicherte polyline")
            c = arc_resample(self.raw[key], self.n + 1)
            self.cache[key] = c
        return c[::-1].copy() if u > v else c


# ------------------------------------------------------------------
# 2D-Coons auf 3D-Punkten + Dihedral-Mapping
# ------------------------------------------------------------------
def coons_face(p00, p10, p01, p11, cur_b, cur_t, cur_l, cur_r, n):
    """Coons-Gitter (n+1, n+1, 3), grid[iu, iv]. Randkurven:
    cur_b: p00->p10 (u), cur_t: p01->p11 (u), cur_l: p00->p01 (v),
    cur_r: p10->p11 (v). Ecken: grid[0,0]=p00, grid[n,0]=p10,
    grid[0,n]=p01, grid[n,n]=p11."""
    cb = arc_resample(cur_b, n + 1)
    ct = arc_resample(cur_t, n + 1)
    cl = arc_resample(cur_l, n + 1)
    cr = arc_resample(cur_r, n + 1)
    U = np.linspace(0, 1, n + 1)[:, None, None]          # iu
    V = np.linspace(0, 1, n + 1)[None, :, None]          # iv
    return ((1 - V) * cb[:, None, :] + V * ct[:, None, :]
            + (1 - U) * cl[None, :, :] + U * cr[None, :, :]
            - ((1 - U) * (1 - V)) * p00
            - (U * (1 - V)) * p10
            - (U * V) * p11
            - ((1 - U) * V) * p01)


def dihedral_solve(canon, caller, n):
    """caller(g0,g1) -> canon(au,av) als (swap, mx, my):
    c0,c1 = (g0,g1) bzw (g1,g0); au = c0 (+1) bzw n-c0 (-1); analog av.
    Prueft alle 4 Corners."""
    for swap in (False, True):
        for mx in (1, -1):
            for my in (1, -1):
                ok = True
                for vid, (au, av) in canon.items():
                    g0, g1 = caller[vid]
                    c0, c1 = (g0, g1) if not swap else (g1, g0)
                    if (c0 if mx == 1 else n - c0,
                            c1 if my == 1 else n - c1) != (au, av):
                        ok = False
                        break
                if ok:
                    return (swap, mx, my)
    raise RuntimeError("dihedral: keine konsistente Map")


def caller_to_canon(dih, pts, n):
    """[(g0,g1)...] -> [(au,av)...]"""
    swap, mx, my = dih
    out = []
    for g0, g1 in pts:
        c0, c1 = (g0, g1) if not swap else (g1, g0)
        au = c0 if mx == 1 else n - c0
        av = c1 if my == 1 else n - c1
        out.append((au, av))
    return out


def canon_to_from(dih, pts, n):
    """[(au,av)...] -> block caller [(g0,g1)...] (inverse)."""
    swap, mx, my = dih
    out = []
    for au, av in pts:
        g0 = au if mx == 1 else n - au
        g1 = av if my == 1 else n - av
        out.append((g0, g1) if not swap else (g1, g0))
    return out


# der inverse: block caller coords (g0,g1) -> canon = caller_to_canon.
# canon -> block: wie oben. Beide duerfen nicht verwechselt werden.


def block_caller(v8, ax, side, n):
    """Die 4 Face-Corner des Blocks als {vid: (g0,g1)} in 0/n:
    g0 laeuft entlang der ersten Nicht-ax-Achse."""
    o0 = [a for a in (0, 1, 2) if a != ax][0]
    o1 = [a for a in (0, 1, 2) if a != ax][1]
    out = {}
    for c in CORNER:
        if c[ax] != side:
            continue
        out[v8[CORNER_IDX[c]]] = (c[o0] * n, c[o1] * n)
    return out


def corner_loop(caller, n):
    """Loop (p00,p10,p11,p01) ans caller dict."""
    return tuple(next(v for v, p in caller.items() if p == pt)
                 for pt in ((0, 0), (n, 0), (n, n), (0, n)))


def face_sl(ax, side):
    """Numpy-Slice: Face (ax, side) des Blockgitters."""
    sl = [slice(None)] * 3
    sl[ax] = 0 if side == 0 else -1
    return tuple(sl)


# ------------------------------------------------------------------
# Dedup (Port tfi.weld) + Gordon-Hall (Port)
# ------------------------------------------------------------------
def weld_points(pts, tol=DEDUP_TOL):
    """Coincidentes -> Union-Find, rep = KLEINSTER Index (torch cdist,
    chunked). Returns (uniq [U,3], idxmap [len(pts)])."""
    import torch
    par = np.arange(len(pts))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[par[x]]
        return x

    t = torch.as_tensor(pts, dtype=torch.float64)
    CH = 2048
    for i in range(0, len(pts), CH):
        d = torch.cdist(t[i:i + CH], t)
        for a, b in torch.nonzero(d < tol).tolist():
            if b <= a + i:
                continue  # strikt obere Dreieckshaelfte, einmal pro Paar
            ra, rb = find(int(a) + i), find(int(b))
            if ra != rb:
                par[max(ra, rb)] = min(ra, rb)
    root = np.array([find(i) for i in range(len(pts))])
    keep = np.unique(root)
    remap = np.zeros(len(pts), int)
    remap[keep] = np.arange(len(keep))
    return pts[keep], remap[root]


def tfi_fill(X):
    """X (ni,nj,nk,3) mit 6 Rand-Faces gefuellt -> Gordon-Hall."""
    ni, nj, nk = X.shape[:3]
    u = np.linspace(0, 1, ni)[:, None, None, None]
    v = np.linspace(0, 1, nj)[None, :, None, None]
    w = np.linspace(0, 1, nk)[None, None, :, None]
    F = ((1 - u) * X[0][None, :, :, :] + u * X[-1][None, :, :, :]
         + (1 - v) * X[:, 0][:, None, :, :] + v * X[:, -1][:, None, :, :]
         + (1 - w) * X[:, :, 0][:, :, None, :] + w * X[:, :, -1][:, :, None, :])
    bu, bv, bw = [(1 - u), u], [(1 - v), v], [(1 - w), w]
    iu = [0, -1]
    E = np.zeros_like(F)
    for a in (0, 1):
        for b in (0, 1):
            E += bu[a] * bv[b] * X[iu[a], iu[b], :][None, None, :, :]
            E += bu[a] * bw[b] * X[iu[a], :, iu[b]][None, :, None, :]
            E += bv[a] * bw[b] * X[:, iu[a], iu[b]][:, None, None, :]
    V = np.zeros_like(F)
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                V += bu[a] * bv[b] * bw[c] * X[iu[a], iu[b], iu[c]]
    return F - E + V


# ------------------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------------------
def lattice_pos(ax, t, p, q):
    """(i,j,k) entlang ax bei t, fixe (p, q) auf den beiden anderen."""
    pos = [0, 0, 0]
    o = [a for a in (0, 1, 2) if a != ax]
    pos[o[0]], pos[o[1]] = p, q
    pos[ax] = t
    return (pos[0], pos[1], pos[2])


# ------------------------------------------------------------------
# Geometrie-Kern
# ------------------------------------------------------------------
def subdivide_core(s: dict, n: int, name: str) -> dict:
    """sample.npz dict -> {'vertices','hexes','quads','dir_class','e2s',
                           'n','report','surface_points'}"""
    P0 = np.asarray(s['vertices'], float)
    NT = len(P0)
    blocks0 = np.asarray(s['blocks'])
    edges = np.asarray(s['edges'])
    lut = {(int(a), int(b)): e for e, (a, b) in enumerate(edges)}

    curves = EdgeCurves(canonical_edge_lines(edges, s['edge_polyline'],
                                             s['edge_polyline_offset']), n)
    face_cache = {}   # frozenset(4 corner ids) -> (grid, cpos)

    def face_of(corners):
        key = frozenset(int(c) for c in corners)
        ent = face_cache.get(key)
        if ent is None:
            p00, p10, p11, p01 = (int(c) for c in corners)
            grid = coons_face(P0[p00], P0[p10], P0[p01], P0[p11],
                              curves(p00, p10), curves(p01, p11),
                              curves(p00, p01), curves(p10, p11), n)
            ent = (grid, {p00: (0, 0), p10: (n, 0),
                          p11: (n, n), p01: (0, n)})
            face_cache[key] = ent
        return ent

    pool = [P0]
    all_cells = []
    lattices = []        # (ids (n+1)^3 int64, X)
    edge_jobs = []       # [(ra, rb, kind, payload)]
    face_regs = {}       # frozenset -> (dih_b, ax, side, lix)

    for blk in blocks0:
        v8 = list(int(x) for x in blk)

        # Handedness aus den G radiusCorner-Koordinaten (VTK-Jacobian)
        cA = P0[v8]
        hv = np.dot(np.cross(cA[1] - cA[0], cA[3] - cA[0]), cA[4] - cA[0])
        if hv < 0:
            # Achse 0 der (i,j,k)-Konvention spiegeln: CORNER (ci,cj,ck)
            # <-> (1-ci, cj, ck)
            v8 = [v8[CORNER_IDX[(1 - c[0], c[1], c[2])]] for c in CORNER]

        # local lattice Index (raw pool) und corner pos
        lix = len(lattices)
        base = NT + lix * (n + 1) ** 3
        ids = np.arange(base, base + (n + 1) ** 3,
                        dtype=np.int64).reshape(n + 1, n + 1, n + 1)
        corner_pos = {(c[0] * n, c[1] * n, c[2] * n): v8[CORNER_IDX[c]]
                      for c in CORNER}

        # (a) Coons-Faces slotten; dihedral pro Face merken
        X = np.zeros((n + 1, n + 1, n + 1, 3))
        solved = {}
        for ax in range(3):
            for side in (0, 1):
                caller = block_caller(v8, ax, side, n)
                q = corner_loop(caller, n)
                key = frozenset(q)
                ent = face_of(q)
                grid, cpos = ent
                dih = dihedral_solve(cpos, caller, n)
                solved[key] = (dih, ax, side)
                for g0 in range(n + 1):
                    for g1 in range(n + 1):
                        au, av = caller_to_canon(dih, [(g0, g1)], n)[0]
                        pos = list(lattice_pos(ax, n if side == 1 else 0, 0, 0))
                        o = [a for a in (0, 1, 2) if a != ax]
                        pos[o[0]], pos[o[1]] = g0, g1
                        X[tuple(pos)] = grid[au, av]
                if key not in face_regs:
                    face_regs[key] = (dih, ax, side, lix)

        # (b) Gordon-Hall-Innenraum
        X = tfi_fill(X)
        pool.append(X.reshape(-1, 3))
        lattices.append((ids, X))

        # Sub-Hexes
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    all_cells.append((int(ids[i, j, k]), int(ids[i + 1, j, k]),
                                      int(ids[i + 1, j + 1, k]),
                                      int(ids[i, j + 1, k]),
                                      int(ids[i, j, k + 1]),
                                      int(ids[i + 1, j, k + 1]),
                                      int(ids[i + 1, j + 1, k + 1]),
                                      int(ids[i, j + 1, k + 1])))

        # (c) Kanten-Jobs: fuer jede Gitterlinie je n Sub-Kanten ...
        for ax in range(3):
            for p in range(n + 1):
                for q_ in range(n + 1):
                    on_curve = (p in (0, n)) and (q_ in (0, n))
                    for t in range(n):
                        ra = int(ids[lattice_pos(ax, t, p, q_)])
                        rb = int(ids[lattice_pos(ax, t + 1, p, q_)])
                        if on_curve:
                            u0 = corner_pos[lattice_pos(ax, 0, p, q_)]
                            u1 = corner_pos[lattice_pos(ax, n, p, q_)]
                            edge_jobs.append((ra, rb, 'curve', (u0, u1, t)))
                        else:
                            edge_jobs.append((ra, rb, 'straight', None))

    # (d) dedup + remap
    pts = np.vstack(pool)
    uniq, idxmap = weld_points(pts)
    n_welded = int(len(pts) - len(uniq))
    Lr = [idxmap[ids] for (ids, _X) in lattices]
    hexes = np.array([[idxmap[i] for i in cell] for cell in all_cells],
                     dtype=np.int64)

    # (e) Sub-Kanten materialisieren
    e2s = {}
    seen = set()
    for ra, rb, kind, payload in edge_jobs:
        ia, ib = int(idxmap[ra]), int(idxmap[rb])
        key = (ia, ib) if ia < ib else (ib, ia)
        if key in seen:
            continue
        seen.add(key)
        if kind == 'straight':
            pa, pb = uniq[ia], uniq[ib]
            seg = np.array([pa, 0.5 * (pa + pb), pb])
        else:
            u0, u1, t = payload
            uu, vv = (min(u0, u1), max(u0, u1))
            c = curves(uu, vv)
            k = t if u0 <= u1 else (n - 1) - t
            seg = np.array([c[k], 0.5 * (c[k] + c[k + 1]), c[k + 1]])
        _store(e2s, ia, ib, seg)

    # (f) Surface-Quads: gleicher face-cache; id und dir_class ueber
    # Face-Record des einzigen Blocks
    quad0 = np.asarray(s['quad_faces'])
    dir0 = np.asarray(s['dir_class'])
    sub_quads, sub_dir_class = [], []
    for q in quad0:
        q = (int(q[0]), int(q[1]), int(q[2]), int(q[3]))
        key = frozenset(q)
        ent = face_cache.get(key)
        if ent is None:
            ent = _face_entry_direct(q, P0, curves, n, face_cache)
        _grid, cpos = ent
        dih_b, ax, side, lix = face_regs[key]
        ql = {q[0]: (0, 0), q[1]: (n, 0), q[2]: (n, n), q[3]: (0, n)}
        di2 = dihedral_solve(cpos, ql, n)
        ids_face = Lr[lix][face_sl(ax, side)]
        e = lut.get((q[0], q[1]))
        if e is None:
            e = lut[(q[1], q[0])]
        dcl = int(dir0[e])
        for jv in range(n):
            for iu in range(n):
                cells = [(iu, jv), (iu + 1, jv), (iu + 1, jv + 1), (iu, jv + 1)]
                cnq = caller_to_canon(di2, cells, n)
                bg = canon_to_from(dih_b, cnq, n)
                sub_quads.append([int(ids_face[g0, g1]) for g0, g1 in bg])
                sub_dir_class.append(dcl)
    sub_quads = np.asarray(sub_quads, dtype=np.int64)
    sub_dir_class = np.asarray(sub_dir_class, dtype=np.int64)

    return {"vertices": uniq, "hexes": hexes, "n": n, "name": name,
            "quads": sub_quads, "dir_class": sub_dir_class, "e2s": e2s,
            "surface_points": np.asarray(s['surface_points'], float),
            "report": {"welded": n_welded, "vertices": int(len(uniq)),
                       "hexes": int(len(hexes)),
                       "quads": int(len(sub_quads))}}


def _face_entry_direct(quad, P0, curves, n, face_cache):
    """Surface-Face, das kein Block-Face war (sollte nicht vorkommen,
    aber sicher ist sicher): per Coons bauen und cachen."""
    p00, p10, p11, p01 = quad
    grid = coons_face(P0[p00], P0[p10], P0[p01], P0[p11],
                      curves(p00, p10), curves(p01, p11),
                      curves(p00, p01), curves(p10, p11), n)
    ent = (grid, {p00: (0, 0), p10: (n, 0), p11: (n, n), p01: (0, n)})
    face_cache[frozenset(quad)] = ent
    return ent


def _store(e2s, ia, ib, seg):
    seg32 = np.asarray(seg, np.float32)
    e2s[(ia, ib)] = seg32
    e2s[(ib, ia)] = seg32[::-1].copy()


# ------------------------------------------------------------------
# Smoke
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import glob as _g

    srcs = _g.glob(sys.argv[1] if len(sys.argv) > 1 else
                   "/tmp/opencode/quad3d_smoke_src/*/sample.npz")
    for src in sorted(srcs):
        s = dict(np.load(src, allow_pickle=True))
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
        r = subdivide_core(s, n, src.split("/")[-2])
        blk0 = np.asarray(s['blocks']).shape[0]
        q0 = np.asarray(s['quad_faces']).shape[0]
        assert r['hexes'].shape[0] == blk0 * n ** 3, (r['hexes'].shape, blk0)
        assert r['quads'].shape[0] == q0 * n ** 2, (r['quads'].shape, q0)
        assert r['report']['welded'] >= 0
        # Hex-Volumina positiv?
        H = r['vertices'][r['hexes']]
        vols = np.einsum('ij,ij->i', np.cross(H[:, 1] - H[:, 0], H[:, 3] - H[:, 0]),
                         H[:, 4] - H[:, 0])
        # echte Orientierung: scalar triple (P1-P0)x(P3-P0) . (P4-P0)
        print(r['name'], r['report'], "min_hex_vol=%.3e" % vols.min(),
              "max=%.3e" % vols.max())


# ------------------------------------------------------------------
# Packaging: Polytron-Format + Quadtron-Data + CLI
# ------------------------------------------------------------------
def build_polytron_aug(r, rng, max_tri_points=768):
    """subdivide_core-Result -> dict wie domain_extractor_3d
    .build_polytron_sample (plus 'subdiv_n')."""
    import torch
    from domain_extractor_3d import (to_cylindrical, subsample_points)
    from polytron_tokenizer import PolytronTokenizer

    uniq = np.asarray(r['vertices'], np.float64)
    e2s = r['e2s']
    edges = np.array(sorted(e2s), np.int64)          # beide Richtungen
    E = len(edges)
    ctrl = np.zeros((E, 2, 3), np.float32)
    for e, (u, v) in enumerate(edges):
        poly = np.asarray(e2s[(int(u), int(v))], np.float64)
        B1, B2 = PolytronTokenizer._fit_cubic_bezier(
            np.asarray(poly[0]), np.asarray(poly[-1]), poly)
        ctrl[e, 0] = B1
        ctrl[e, 1] = B2
    quad_faces = np.asarray(r['quads'], np.int64)

    return {
        'vertices_polar': torch.tensor(to_cylindrical(uniq), dtype=torch.float32),
        'vertices_cartesian': torch.tensor(uniq, dtype=torch.float64),
        'faces': torch.tensor(np.asarray(r['hexes']).T, dtype=torch.long),
        'edge_index': torch.tensor(edges.T, dtype=torch.long),
        'edge_ctrl': torch.tensor(ctrl, dtype=torch.float32),
        'edge_to_streamline': {
            (int(u), int(v)): np.asarray(e2s[(int(u), int(v))], np.float32)
            for (u, v) in e2s},
        'center': torch.tensor([0.0, 0.0, 0.0]),
        'quad_faces': torch.tensor(quad_faces.T, dtype=torch.long),
        'tri_coordinates': torch.tensor(
            subsample_points(np.asarray(r['surface_points'], np.float64),
                             max_tri_points, rng), dtype=torch.float32),
        'surface_points': torch.tensor(np.asarray(r['surface_points'], np.float32)),
        'subdiv_n': int(r['n']),
    }


def build_quadtron_aug(r):
    """subdivide_core-Result -> torch_geometric Data."""
    import torch
    from torch_geometric.data import Data
    uniq = np.asarray(r['vertices'], np.float64)
    quad_faces = np.asarray(r['quads'], np.int64)
    return Data(x=torch.tensor(uniq, dtype=torch.float32),
                faces=torch.tensor(quad_faces.T, dtype=torch.long),
                tri_coordinates=torch.tensor(
                    np.asarray(r['surface_points'], np.float32)),
                dir_class=torch.tensor(np.asarray(r['dir_class'], np.int64)))


def _proc_one(p_str, n_vals, n3_maxblocks, max_tri_points):
    """Pro-Sample-Worker: load npz, Base+Subdivision-Builds. Returns
    (name, [(kind, obj)...], msgs) kind in 'q'/'p'.
    Deterministischer rng-Seed pro Sample (Subsampling)."""
    import numpy as _np
    from pathlib import Path
    from domain_extractor_3d import build_quadtron_sample, build_polytron_sample
    p = Path(p_str)
    s = dict(_np.load(p, allow_pickle=True))
    name = p.parent.name
    import hashlib as _h
    rng = _np.random.default_rng(int(_h.md5(name.encode()).hexdigest()[:8], 16))
    hits = []
    try:
        hits.append(('q', build_quadtron_sample(s)))
        base = build_polytron_sample(s, rng)
        base['name'], base['subdiv_n'] = name, None
        hits.append(('p', base))
        for n in n_vals:
            if n == 3 and n3_maxblocks and int(s['blocks'].shape[0]) > n3_maxblocks:
                continue
            r = subdivide_core(s, n, f"{name}_n{n}")
            print(name, "n=%d" % n, r['report'], flush=True)
            hits.append(('q', build_quadtron_aug(r)))
            pa = build_polytron_aug(r, rng, max_tri_points=max_tri_points)
            pa['name'] = f"{name}_n{n}"
            hits.append(('p', pa))
    except Exception as e:
        print(f"SKIP {name}: {type(e).__name__}: {e}", flush=True)
        return name, hits, f"SKIP {name}: {type(e).__name__}: {e}"
    return name, hits, ""


def main():
    import argparse, glob as _g, multiprocessing as mp
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=Path, required=True)
    ap.add_argument('--n', default='2,3')
    ap.add_argument('--n3-maxblocks', type=int, default=0,
                    help='n3 nur auf Original-Samples mit <= dieser Blockzahl (0: alle)')
    ap.add_argument('--max-tri-points', type=int, default=768)
    ap.add_argument('--jobs', type=int, default=min(24, mp.cpu_count()))
    ap.add_argument('--out-quadtron', type=Path, default=Path('data/quadtron_data_3d_aug.pt'))
    ap.add_argument('--out-polytron', type=Path, default=Path('data/polytron_data_3d_aug.pt'))
    args = ap.parse_args()

    import torch
    srcs = sorted(_g.glob(str(args.src / '*/sample.npz')))
    n_vals = [int(x) for x in args.n.split(',')]
    print(f"{len(srcs)} samples", flush=True)
    quadtron, polytron = [], []
    nsusp = 0
    with mp.get_context('fork').Pool(args.jobs) as pool:
        asyncs = [pool.apply_async(_proc_one, (p, n_vals, args.n3_maxblocks,
                                               args.max_tri_points))
                  for p in srcs]
        for a in asyncs:
            _name, hits, msg = a.get()
            if msg:
                print(msg, flush=True)
            for kind, obj in hits:
                (quadtron if kind == 'q' else polytron).append(obj)
            nsusp += 1
            if nsusp % 50 == 0:
                print(f"... {nsusp} samples", flush=True)
    torch.save(quadtron, args.out_quadtron)
    torch.save(polytron, args.out_polytron)
    print("wrote", args.out_quadtron, args.out_polytron, len(quadtron), len(polytron))


if __name__ == '__main__' and '--src' in __import__('sys').argv:
    main()
