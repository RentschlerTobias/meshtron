"""Boundary surface of the full dtOO domain (case/mesh.msh, gmsh 2.2 ASCII) as labelled triangles.

The physical groups are not attached to the elements in these files (phys tag 0 everywhere), so the boundary is
taken from the volume itself: faces owned by exactly one 3D cell (tet / hex / prism / pyramid). Each boundary face
carries the geom tag of the tagged 2D element with the same nodes. Geom tags are merged into patches by smooth
connectivity: two tags touching along an edge with a dihedral below `smooth_deg` belong to the same patch
(the hub wall under the prism layer and under the O-grid, ...). Only the boundary (geometry) is used; the dtOO
volume cells (O-grid, boundary layers) are not kept."""
from collections import Counter, defaultdict

import numpy as np

FACES = {4: [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)],
         5: [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)],
         6: [(0, 1, 2), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)],
         7: [(0, 1, 2, 3), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)]}
NV = {4: 4, 5: 8, 6: 6, 7: 5}


def read_msh(path):
    nodes, el, names = {}, [], {}
    sec = None
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("$End"):
                sec = None; continue
            if s.startswith("$"):
                sec = s[1:]; continue
            p = s.split()
            if sec == "PhysicalNames" and len(p) == 3:
                names[int(p[1])] = (int(p[0]), p[2].strip('"'))
            elif sec == "Nodes" and len(p) == 4:
                nodes[int(p[0])] = (float(p[1]), float(p[2]), float(p[3]))
            elif sec == "Elements" and len(p) >= 5:
                et, nt = int(p[1]), int(p[2])
                geom = int(p[4]) if nt >= 2 else 0
                el.append((et, geom, [int(x) for x in p[3 + nt:]]))
    return nodes, el, names


def load_boundary(path, smooth_deg=30.0):
    """-> P [n,3], T [m,3] triangles, patch [m] (0..k-1), info: per patch the geom tags and triangle count."""
    nodes, el, _ = read_msh(path)
    own = Counter(); face_nodes = {}
    for et, _g, nd in el:
        if et in FACES:
            for f in FACES[et]:
                key = frozenset(nd[i] for i in f)
                own[key] += 1; face_nodes[key] = [nd[i] for i in f]
    tag2d = {frozenset(nd[:3] if et == 2 else nd[:4]): g for et, g, nd in el if et in (2, 3)}
    bnd = [k for k, c in own.items() if c == 1]
    geoms = np.array([tag2d.get(k, -1) for k in bnd])
    # triangles of the boundary faces
    tris, tgeom = [], []
    for k, g in zip(bnd, geoms):
        f = face_nodes[k]
        for t in ([f] if len(f) == 3 else [[f[0], f[1], f[2]], [f[0], f[2], f[3]]]):
            tris.append(t); tgeom.append(g)
    used = sorted({n for t in tris for n in t}); idx = {n: i for i, n in enumerate(used)}
    P = np.array([nodes[n] for n in used], float)
    T = np.array([[idx[n] for n in t] for t in tris], np.int64)
    tgeom = np.array(tgeom)
    # merge geom tags by smooth adjacency across tag boundaries
    nrm = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    e2t = defaultdict(list)
    for i, t in enumerate(T):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            e2t[(min(a, b), max(a, b))].append(i)
    ug = sorted(set(tgeom.tolist())); gi = {g: i for i, g in enumerate(ug)}
    parent = list(range(len(ug)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    cos_lim = np.cos(np.radians(smooth_deg))
    votes = Counter()
    for ts in e2t.values():
        if len(ts) == 2 and tgeom[ts[0]] != tgeom[ts[1]]:
            pair = (gi[tgeom[ts[0]]], gi[tgeom[ts[1]]])
            votes[(min(pair), max(pair), abs(float(nrm[ts[0]] @ nrm[ts[1]])) > cos_lim)] += 1
    for (a, b, smooth), n in votes.items():
        n_sharp = votes.get((a, b, not smooth), 0) if smooth else 0
        if smooth and n > n_sharp:                                  # mostly smooth contact -> same patch
            parent[find(a)] = find(b)
    roots = sorted({find(i) for i in range(len(ug))})
    patch = np.array([roots.index(find(gi[g])) for g in tgeom])
    info = [{"geoms": sorted(g for g in ug if roots.index(find(gi[g])) == p), "n_tris": int((patch == p).sum())}
            for p in range(len(roots))]
    return P, T, patch, info


class SeamProjector:
    """Nearest point on the seam polyline between patches (mesh edges with one triangle in each patch).
    Continuous along the seam -- alternating projection onto the two coarse triangulations snaps to their
    vertices and collapses neighbouring points."""

    def __init__(self, P, T, L):
        from scipy.spatial import cKDTree
        e2t = defaultdict(list)
        for i, t in enumerate(T):
            for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                e2t[(min(a, b), max(a, b))].append(i)
        self.seg = defaultdict(list)
        for e, ts in e2t.items():
            labs = sorted({int(L[t]) for t in ts})
            for i in range(len(labs)):                    # non-manifold edges (3 surfaces) -> every pair
                for j in range(i + 1, len(labs)):
                    self.seg[(labs[i], labs[j])].append(e)
        self.P = P
        self.tree = {k: (np.array(v), cKDTree(P[np.array(v)].mean(1))) for k, v in self.seg.items()}

    def has(self, labs):
        return tuple(sorted(labs)) in self.tree

    def onto(self, X, labs, k=16):
        E, tr = self.tree[tuple(sorted(labs))]
        X = np.asarray(X, float).reshape(-1, 3)
        _, idx = tr.query(X, k=min(k, len(E)))
        A, B = self.P[E[idx, 0]], self.P[E[idx, 1]]                 # [n,k,3]
        d = B - A
        t = np.clip(np.einsum("nkj,nkj->nk", X[:, None] - A, d) / np.einsum("nkj,nkj->nk", d, d).clip(1e-30), 0, 1)
        Q = A + t[..., None] * d
        j = np.argmin(np.linalg.norm(Q - X[:, None], axis=2), 1)
        return Q[np.arange(len(X)), j]


class SmoothProjector:
    """Continuous projection onto a faceted patch: the nearest facet point gives a smooth (vertex-averaged,
    barycentrically interpolated) normal n, then the line x + s n is intersected with the facets. Plain
    nearest-point projection maps the whole normal cone of a convex facet edge / vertex onto that edge / vertex
    and collapses neighbouring points (coarse hub cylinder, blade nose)."""

    def __init__(self, P, T, L, k=48):
        from collections import deque
        from scipy.spatial import cKDTree
        self.P, self.k, self.pat = P, k, {}
        for l in np.unique(L):
            Tl = T[L == l]
            n = np.cross(P[Tl[:, 1]] - P[Tl[:, 0]], P[Tl[:, 2]] - P[Tl[:, 0]])
            # consistent orientation by propagation over edge neighbours
            e2t = defaultdict(list)
            for i, t in enumerate(Tl):
                for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                    e2t[(min(a, b), max(a, b))].append(i)
            sign = np.zeros(len(Tl)); 
            for s0 in range(len(Tl)):
                if sign[s0]:
                    continue
                sign[s0] = 1; dq = deque([s0])
                while dq:
                    i = dq.popleft(); t = Tl[i]
                    for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                        for j in e2t[(min(a, b), max(a, b))]:
                            if not sign[j]:
                                sign[j] = sign[i] if n[i] @ n[j] > 0 else -sign[i]; dq.append(j)
            n = n * sign[:, None]
            vn = np.zeros_like(P)
            for c in range(3):
                np.add.at(vn, Tl[:, c], n)
            vn /= np.linalg.norm(vn, axis=1, keepdims=True).clip(1e-30)
            self.pat[int(l)] = (Tl, vn, cKDTree(P[Tl].mean(1)))

    def onto(self, X, lab):
        Tl, vn, tr = self.pat[int(lab)]
        X = np.asarray(X, float).reshape(-1, 3)
        _, idx = tr.query(X, k=min(self.k, len(Tl)))
        A, B, C = (self.P[Tl[idx, i]] for i in range(3))            # [n,k,3]
        # nearest facet point -> barycentric -> smooth normal
        Q, bary = _closest_on_tris(X, A, B, C)
        dist = np.linalg.norm(Q - X[:, None], axis=2); j = np.argmin(dist, 1); r = np.arange(len(X))
        b = bary[r, j]; tri = Tl[idx[r, j]]
        nrm = (b[:, :, None] * vn[tri]).sum(1); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True).clip(1e-30)
        # line x + s n against the candidate facets (Moller-Trumbore), smallest |s|
        e1, e2 = B - A, C - A
        pv = np.cross(nrm[:, None], e2); det = np.einsum("nkj,nkj->nk", e1, pv)
        ok = np.abs(det) > 1e-14
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tv = X[:, None] - A
        u = np.einsum("nkj,nkj->nk", tv, pv) * inv
        qv = np.cross(tv, e1)
        v = np.einsum("nj,nkj->nk", nrm, qv) * inv
        s = np.einsum("nkj,nkj->nk", e2, qv) * inv
        tol = 1e-9
        hit = ok & (u >= -tol) & (v >= -tol) & (u + v <= 1 + tol)
        s_abs = np.where(hit, np.abs(s), np.inf)
        jj = np.argmin(s_abs, 1)
        out = Q[r, j].copy()
        good = np.isfinite(s_abs[r, jj])
        out[good] = X[good] + s[r, jj][good, None] * nrm[good]
        return out


def _closest_on_tris(X, A, B, C):
    """Closest points of X [n,3] on triangles [n,k,3] and their barycentric coordinates [n,k,3]."""
    P = X[:, None]
    ab, ac, ap = B - A, C - A, P - A
    d1, d2 = np.einsum("nkj,nkj->nk", ab, ap), np.einsum("nkj,nkj->nk", ac, ap)
    bp = P - B; d3, d4 = np.einsum("nkj,nkj->nk", ab, bp), np.einsum("nkj,nkj->nk", ac, bp)
    cp_ = P - C; d5, d6 = np.einsum("nkj,nkj->nk", ab, cp_), np.einsum("nkj,nkj->nk", ac, cp_)
    va = d3 * d6 - d5 * d4; vb = d5 * d2 - d1 * d6; vc = d1 * d4 - d3 * d2
    den = (va + vb + vc); den = np.where(np.abs(den) < 1e-30, 1e-30, den)
    v = vb / den; w = vc / den
    # clamp to the triangle (simple: clip barycentrics and renormalise, then exact edge handling)
    u = 1 - v - w
    bary = np.stack([u, v, w], -1)
    inside = (bary >= 0).all(-1)
    Q = A + v[..., None] * ab + w[..., None] * ac
    # outside: best of the three edge projections
    best = np.full(u.shape, np.inf); Qe = np.zeros_like(Q); Be = np.zeros_like(bary)
    for (S0, S1, i0, i1) in ((A, B, 0, 1), (B, C, 1, 2), (C, A, 2, 0)):
        d = S1 - S0
        t = np.clip(np.einsum("nkj,nkj->nk", P - S0, d) / np.einsum("nkj,nkj->nk", d, d).clip(1e-30), 0, 1)
        q = S0 + t[..., None] * d
        dd = np.linalg.norm(q - P, axis=-1)
        m = dd < best
        best = np.where(m, dd, best); Qe[m] = q[m]
        bb = np.zeros_like(bary); bb[..., i0] = 1 - t; bb[..., i1] = t
        Be[m] = bb[m]
    Q = np.where(inside[..., None], Q, Qe); bary = np.where(inside[..., None], bary, Be)
    return Q, bary


class SmoothSeamProjector(SeamProjector):
    """Seam polyline with vertex tangents interpolated along each segment; a point maps to the polyline point
    whose interpolated tangent is perpendicular to the connecting line (continuous at convex polyline vertices,
    where plain nearest-point projection snaps)."""

    def __init__(self, P, T, L):
        super().__init__(P, T, L)
        self.tan = {}
        for key, (E, _tr) in self.tree.items():
            tv = {}
            for a, b in E:
                d = P[b] - P[a]; d = d / np.linalg.norm(d)
                for v in (a, b):
                    if v in tv and tv[v] @ d < 0:
                        tv[v] = tv[v] - d
                    else:
                        tv[v] = tv.get(v, 0) + d
            self.tan[key] = {v: t / np.linalg.norm(t) for v, t in tv.items()}

    def onto(self, X, labs, k=16):
        key = tuple(sorted(labs)); E, tr = self.tree[key]; tv = self.tan[key]
        X = np.asarray(X, float).reshape(-1, 3)
        base = SeamProjector.onto(self, X, labs, k)
        _, idx = tr.query(X, k=min(k, len(E)))
        out = base.copy()
        for n, x in enumerate(X):
            best, bd = None, np.inf
            for e in idx[n]:
                a, b = E[e]; A, Bp = self.P[a], self.P[b]; d = Bp - A
                tA, tB = tv[a], tv[b]
                if tA @ tB < 0:
                    tB = -tB
                dt = tB - tA; w = x - A
                c0, c1, c2 = w @ tA, w @ dt - d @ tA, -(d @ dt)
                roots = [(-c0 / c1)] if abs(c2) < 1e-14 and abs(c1) > 1e-14 else (
                    [r.real for r in np.roots([c2, c1, c0]) if abs(r.imag) < 1e-12] if abs(c2) >= 1e-14 else [])
                for u in roots:
                    if -1e-9 <= u <= 1 + 1e-9:
                        q = A + u * d; dd = np.linalg.norm(x - q)
                        if dd < bd:
                            best, bd = q, dd
            if best is not None and bd < 1.5 * np.linalg.norm(x - base[n]) + 1e-12:
                out[n] = best
        return out


GROUP = {4: "core", 7: "core", 5: "hex", 6: "prism"}


def load_regions(path, smooth_deg=30.0):
    """Boundary AND internal region interfaces of the dtOO mesh as labelled triangles (geometry only).

    Regions: the core (tets + pyramids) and the connected components of the hex cells (O-grid) and of the prism
    cells (boundary layers, one per wall). Labels:
      kind "phys"  : boundary faces, split by physical patch (load_boundary's merge) x owning region
      kind "iface" : faces between two different regions (core | O-grid = ogrid cut, core | layer = layer cuts,
                     layer | O-grid = the O-grid's outer side inside the boundary layers)
    Returns P, T, lab [m], info [{kind, phys, regions, n_tris}], region_type [region] ('core'/'hex'/'prism')."""
    nodes, el, _ = read_msh(path)
    cells = [(et, nd) for et, _g, nd in el if et in FACES]
    # region components of hex / prism cells (face adjacency), core = one region
    face_cells = defaultdict(list)
    for ci, (et, nd) in enumerate(cells):
        for f in FACES[et]:
            face_cells[frozenset(nd[i] for i in f)].append(ci)
    parent = list(range(len(cells)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for fc in face_cells.values():
        if len(fc) == 2 and GROUP[cells[fc[0]][0]] == GROUP[cells[fc[1]][0]]:
            parent[find(fc[0])] = find(fc[1])
    roots, region, rtype = {}, [], []
    for ci, (et, nd) in enumerate(cells):
        key = "core" if GROUP[et] == "core" else find(ci)
        if key not in roots:
            roots[key] = len(rtype); rtype.append(GROUP[et])
        region.append(roots[key])
    # boundary physical patch per boundary face (reuse load_boundary's smooth merge by geom tags)
    Pb, Tb, patch_b, _info = load_boundary(path, smooth_deg)
    idx_b = {tuple(sorted(np.round(Pb[t].mean(0), 9))): p for t, p in zip(Tb, patch_b)}
    tris, labkey, used = [], [], {}
    face_nodes = {}
    for key, fc in face_cells.items():
        if len(fc) == 1:
            reg = region[fc[0]]
            et, nd = cells[fc[0]]
            k = ("phys", None, (reg,))
        else:
            r0, r1 = region[fc[0]], region[fc[1]]
            if r0 == r1:
                continue
            k = ("iface", None, tuple(sorted((r0, r1))))
        # node order from the owning cell
        et, nd = cells[fc[0]]
        for f in FACES[et]:
            if frozenset(nd[i] for i in f) == key:
                fn = [nd[i] for i in f]; break
        for t in ([fn] if len(fn) == 3 else [[fn[0], fn[1], fn[2]], [fn[0], fn[2], fn[3]]]):
            tris.append(t); labkey.append(k)
    un = sorted({n for t in tris for n in t}); idx = {n: i for i, n in enumerate(un)}
    P = np.array([nodes[n] for n in un], float)
    T = np.array([[idx[n] for n in t] for t in tris], np.int64)
    # physical patch of boundary triangles: nearest boundary triangle of load_boundary
    from scipy.spatial import cKDTree
    tr = cKDTree(Pb[Tb].mean(1))
    cen = P[T].mean(1)
    phys = patch_b[tr.query(cen)[1]]
    keys = [(k[0], int(ph) if k[0] == "phys" else None, k[2]) for k, ph in zip(labkey, phys)]
    uniq = sorted(set(keys), key=lambda k: (k[0], -1 if k[1] is None else k[1], k[2]))
    li = {k: i for i, k in enumerate(uniq)}
    lab = np.array([li[k] for k in keys])
    info = [{"kind": k[0], "phys": k[1], "regions": k[2], "n_tris": int((lab == i).sum())} for i, k in enumerate(uniq)]
    return P, T, lab, info, rtype
