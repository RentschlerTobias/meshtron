"""Slide the core corners on the cut surfaces along their seams so that the blocks built from them (core blocks,
layer blocks, O-ring blocks) are well shaped -- before the blocking. Generic: corners that lie on a seam between
two cut surfaces (e.g. O-grid cut | layer cut) move along that seam only; the objective is the minimum scaled
Jacobian of the straight blocks touching the corner, with their wall corners computed by the blocking rules
(blocking.py: descend within side surfaces, level constraint on the blade, nearest otherwise)."""
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

import blade_insert as BI
import blocking as BK
import close_layers as CL
import diag
from meshtron.geometry.polytron_tfi import SurfaceProjector, _boundary_labels


def seam_polyline(geo, labs):
    """Ordered (closed or open) polyline of the seam between two geometry labels."""
    E, _ = geo.seam.tree[tuple(sorted(labs))]
    adj = defaultdict(list)
    for a, b in E:
        adj[a].append(b); adj[b].append(a)
    ends = [v for v in adj if len(adj[v]) == 1]
    start = ends[0] if ends else next(iter(adj))
    order, prev = [start], None
    while True:
        nxt = [w for w in adj[order[-1]] if w != prev and w not in order[-2:]]
        if not nxt or (nxt[0] == start):
            break
        prev = order[-1]; order.append(nxt[0])
        if len(order) > len(adj):
            break
    Q = geo.P[order]
    closed = not ends
    if closed:
        Q = np.vstack([Q, Q[:1]])
    return Q, closed


def relocate(V, B, curves, raw, geo, n_steps=41, span=0.25, sweeps=3, verbose=True):
    S, T, L = raw["surface_points"], raw["surface_tris"], raw["surface_tri_label"]
    core_proj = SurfaceProjector(S, T, L)
    V = np.asarray(V, float).copy(); B = np.asarray(B)
    # core labels -> geometry labels
    cls, glab = {}, {}
    for l in np.unique(L):
        c = S[T[L == l]].mean(1)
        d = np.median(geo.near.dists(c), 0); lab = geo.near.labels[int(np.argmin(d))]; inf = geo.info[lab]
        cls[int(l)] = ("phys", inf["phys"]) if inf["kind"] == "phys" else ("cut", [r for r in inf["regions"] if r][0])
        glab[int(l)] = lab
    face_lab, cyc_of = _boundary_labels(V, B, curves, core_proj)
    faces = [(cyc_of[k], int(l)) for k, l in face_lab.items() if l is not None]
    v2l = defaultdict(set)
    for cyc, l in faces:
        for v in cyc:
            v2l[v].add(l)
    cut_labels = [l for l, c in cls.items() if c[0] == "cut"]
    # movable: corners on a seam between two cut labels
    movable = {v: sorted(ls & set(cut_labels)) for v, ls in v2l.items() if len(ls & set(cut_labels)) >= 2}
    seams = {}
    for v, ls in movable.items():
        key = tuple(sorted(glab[l] for l in ls[:2]))
        if key not in seams:
            seams[key] = seam_polyline(geo, key)
    # which region is "enclosed" (cut touching no physical patch) -> ring; the others are layers
    def touches_phys(l):
        return any(cls[m][0] == "phys" for cyc, ll in faces if ll == l for v in cyc for m in v2l[v])
    ring_l = [l for l in cut_labels if not touches_phys(l)]
    wall_of = {}
    for l in cut_labels:
        R = cls[l][1]
        cand = [i for i, d in enumerate(geo.info) if d["kind"] == "phys" and d["regions"] == (R,)]
        cen = np.array([np.mean([V[v] for v in cyc], 0) for cyc, ll in faces if ll == l])
        dd = np.median(geo.near.dists(cen)[:, [geo.near.labels.index(i) for i in cand]], 0)
        wall_of[l] = cand[int(np.argmin(dd))]

    def images(x, v):
        """wall corners of corner v at position x: {cut label: image}."""
        out = {}
        for l in cut_labels:
            if l not in v2l[v]:
                continue
            R = cls[l][1]; wall = wall_of[l]
            others = [m for m in v2l[v] if m != l]
            side = sorted({geo.iface(R, cls[m][1]) if cls[m][0] == "cut" else geo.phys_part(cls[m][1], R)
                           for m in others} - {None})
            if l in ring_l:                               # O-ring corner on a layer seam: keep the level
                m = [mm for mm in others if cls[mm][0] == "cut"][0]
                ph = geo.info[wall_of[m]]["phys"]
                pl = [i for i, d in enumerate(geo.info) if d["kind"] == "phys" and d["phys"] == ph]
                dd = geo.near.dists(x[None])[0, [geo.near.labels.index(i) for i in pl]].min()
                out[l] = BK.level_onto(geo, x[None], wall, pl, dd)[0]
            elif side:
                out[l] = BK.descend(geo, x[None], side, wall)[-1, 0]
            else:
                out[l] = geo.near.onto(x[None], [wall])[0]
        return out

    def blocks_of(v, Vx):
        """straight blocks touching corner v: core blocks + extruded blocks of its cut faces."""
        hexes = [Vx[b] for b in B if v in b]
        img = {u: images(Vx[u], u) for cyc, l in faces if v in cyc and l in cut_labels for u in cyc}
        for cyc, l in faces:
            if v in cyc and l in cut_labels:
                h = np.array([Vx[u] for u in cyc] + [img[u][l] for u in cyc])
                if BI.block_volume(h) < 0:
                    h = np.array([Vx[u] for u in cyc[::-1]] + [img[u][l] for u in cyc[::-1]])
                hexes.append(h)
        return hexes

    def score(v, Vx):
        H = np.array(blocks_of(v, Vx))
        return float(diag.cell_sj(H).min())

    moved = {}
    for sw in range(sweeps):
        for v, ls in sorted(movable.items()):
            key = tuple(sorted(glab[l] for l in ls[:2]))
            Q, closed = seams[key]
            s0, tot = CL.poly_param(Q, V[v][None])
            best = (score(v, V), V[v].copy())
            for ds in np.linspace(-span, span, n_steps):
                s = s0[0] + ds
                if closed:
                    s = s % tot
                elif not (0 <= s <= tot):
                    continue
                x = CL.resample_at(Q, [s / tot])[0]
                Vx = V.copy(); Vx[v] = x
                sc = score(v, Vx)
                if sc > best[0] + 1e-9:
                    best = (sc, x)
            moved[v] = moved.get(v, 0.0) + float(np.linalg.norm(best[1] - V[v]))
            V[v] = best[1]
        if verbose:
            worst = min(score(v, V) for v in movable)
            print(f"[relocate] sweep {sw}: worst min-sJ over the moved corners' blocks {worst:.3f}", flush=True)
    return V, moved, movable


def update_curves(V_old, V_new, curves, raw, geo, B):
    S, T, L = raw["surface_points"], raw["surface_tris"], raw["surface_tri_label"]
    core_proj = SurfaceProjector(S, T, L)
    glab = {}
    for l in np.unique(L):
        c = S[T[L == l]].mean(1)
        glab[int(l)] = geo.near.labels[int(np.argmin(np.median(geo.near.dists(c), 0)))]
    face_lab, cyc_of = _boundary_labels(V_old, B, curves, core_proj)
    e2l = defaultdict(set)
    for k, l in face_lab.items():
        if l is None:
            continue
        cyc = cyc_of[k]
        for a, b in zip(cyc, cyc[1:] + cyc[:1]):
            e2l[(min(a, b), max(a, b))].add(glab[int(l)])
    out = {}
    for (a, b), Q in curves.items():
        Q = np.asarray(Q, float)
        da, db = V_new[a] - V_old[a], V_new[b] - V_old[b]
        if not (np.any(da) or np.any(db)):
            out[(a, b)] = Q; continue
        t = np.linspace(0, 1, len(Q))[:, None]
        Q = Q + (1 - t) * da + t * db
        labs = sorted(e2l.get((a, b), set()))
        if labs and len(Q) > 2:
            Q[1:-1] = geo.onto(Q[1:-1], labs[:2])
        Q[0], Q[-1] = V_new[a], V_new[b]
        out[(a, b)] = Q
    return out


def widen_gaps(V, B, curves, raw, geo, frac=0.5, verbose=True):
    """Push core corners / edge points on a cut surface away from its seams with OTHER cut surfaces (e.g. the
    layer cut near the O-grid cut seam) to at least `frac` x the layer thickness, sliding on their own geometry.
    Returns (V, curves, report)."""
    S, T, L = raw["surface_points"], raw["surface_tris"], raw["surface_tri_label"]
    core_proj = SurfaceProjector(S, T, L)
    V = np.asarray(V, float).copy(); B = np.asarray(B)
    cls, glab = {}, {}
    for l in np.unique(L):
        c = S[T[L == l]].mean(1)
        d = np.median(geo.near.dists(c), 0); lab = geo.near.labels[int(np.argmin(d))]; inf = geo.info[lab]
        cls[int(l)] = ("phys", inf["phys"]) if inf["kind"] == "phys" else ("cut", [r for r in inf["regions"] if r][0])
        glab[int(l)] = lab
    face_lab, cyc_of = _boundary_labels(V, B, curves, core_proj)
    v2l, e2l = defaultdict(set), defaultdict(set)
    for k, l in face_lab.items():
        if l is None:
            continue
        cyc = cyc_of[k]
        for a, b in zip(cyc, cyc[1:] + cyc[:1]):
            v2l[a].add(int(l)); e2l[(min(a, b), max(a, b))].add(int(l))
    cuts = [l for l, c in cls.items() if c[0] == "cut"]
    # only cuts that touch physical patches (layers) are widened; the enclosed cut (O-ring) keeps its edges
    phys_l = {l for l, c in cls.items() if c[0] == "phys"}
    layer_cuts = [c for c in cuts if any(c in ls and ls & phys_l for ls in e2l.values())]
    V0 = V.copy(); rep = []
    pushed_edges = {}
    for c in layer_cuts:
        for c2 in cuts:
            key = tuple(sorted((glab[c], glab[c2])))
            if c2 == c or key not in geo.seam.tree:
                continue
            E, _ = geo.seam.tree[key]
            seam = np.concatenate([np.linspace(geo.P[a], geo.P[b], 8) for a, b in E])
            stree = cKDTree(seam)
            # layer thickness of cut c: distance from its surface points to the wall of its region
            R = cls[c][1]
            cand = [i for i, d in enumerate(geo.info) if d["kind"] == "phys" and d["regions"] == (R,)]
            pts = S[np.unique(T[L == c])]
            g = frac * float(np.median(geo.near.dists(pts)[:, [geo.near.labels.index(i) for i in cand]].min(1)))
            nrm_tris = geo.T[geo.L == glab[c]]
            ntree = cKDTree(geo.P[nrm_tris].mean(1))
            n_all = np.cross(geo.P[nrm_tris[:, 1]] - geo.P[nrm_tris[:, 0]], geo.P[nrm_tris[:, 2]] - geo.P[nrm_tris[:, 0]])
            n_all /= np.linalg.norm(n_all, axis=1, keepdims=True)

            def push(X, w=None):
                X = np.asarray(X, float).reshape(-1, 3)
                d, j = stree.query(X)
                dirv = X - seam[j]
                nn = n_all[ntree.query(X)[1]]
                dirv = dirv - (dirv * nn).sum(1, keepdims=True) * nn
                dirv /= np.linalg.norm(dirv, axis=1, keepdims=True).clip(1e-12)
                amt = np.clip(g - d, 0, None) * (1 if w is None else w)
                return X + amt[:, None] * dirv, d
            for v, ls in v2l.items():
                if c in ls and c2 not in ls:
                    x2, d = push(V[v][None])
                    if d[0] < g:
                        labs = sorted({glab[m] for m in ls})
                        V[v] = geo.onto(x2, labs[:2] if len(labs) > 1 else labs)[0]
                        rep.append({"corner": int(v), "gap_before": float(d[0]), "gap_min": g})
            for e, ls in e2l.items():
                if c in ls and c2 not in ls:
                    pushed_edges[e] = (c, c2, push, sorted({glab[m] for m in ls}))
    curves = update_curves(V0, V, curves, raw, geo, B)
    seam_v = {v for v, ls in v2l.items() if len([m for m in ls if m in cuts]) >= 2}
    for e, (c, c2, push, labs) in pushed_edges.items():
        Q = np.asarray(curves[e], float).copy()
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]; tot = s[-1]
        w = np.ones(len(Q))
        g = rep[0]["gap_min"] if rep else 0.0
        for end, se in ((0, s), (1, tot - s)):                    # fade out towards an end on the seam
            if e[end] in seam_v and g > 0:
                w = np.minimum(w, np.clip(se / (2 * g), 0, 1))
        Q2, _ = push(Q, w)
        Q2[1:-1] = geo.onto(Q2[1:-1], labs[:2] if len(labs) > 1 else labs)
        Q2[0], Q2[-1] = V[e[0]], V[e[1]]
        curves[e] = Q2
    if verbose:
        print(f"[gaps] moved {len(rep)} corners:", [(r["corner"], round(r["gap_before"], 4), round(r["gap_min"], 4)) for r in rep],
              f"| pushed {len(pushed_edges)} edges", flush=True)
    return V, curves, rep
