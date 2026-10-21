#!/usr/bin/env python3
"""Close the core mesh to the full domain with ONE structured layer region per cut surface (O-ring around the
blade, hub layer, shroud layer) -- no subdivision into blocks, no dtOO volume cells.

1. Core: joint curved TFI of the core block structure (12 blocks), projected onto the core surface.
2. Per cut surface (core patch that is not on the full-domain boundary; its wall = nearest full-domain patch),
   in the order "fewest real neighbours first" (O-ring before hub / shroud layers):
     - the outer side is the core's own boundary quads on that cut -> conforming by construction;
     - every outer node gets its partner ON the wall directly below it (nearest point = foot of the wall normal;
       nodes on a seam with a real patch onto wall AND that patch);
     - n layers along the straight node -> wall lines, geometric grading towards the wall;
     - side quads of the new layer inherit the label of the neighbouring patch (real -> domain boundary,
       another cut -> extruded with that cut later, e.g. the O-ring's hub/shroud sides belong to the hub/shroud
       layers).
3. Boundary patches of the result from the full-domain surface (case/mesh.msh boundary, msh_walls.py).

    python close_layers.py --name tistos_s1_0131 --out out/layers_0131
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, "/opt/stack/meshtron")
sys.path.insert(0, "/work/meshtron_v2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import meshio  # noqa: E402
import numpy as np  # noqa: E402

import blade_insert as BI  # noqa: E402
import diag  # noqa: E402
import msh_walls as MW  # noqa: E402
from meshtron.geometry.polytron_tfi import SurfaceProjector, refill_polytron  # noqa: E402

HEX_FACES = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]


def boundary_quads(H):
    own = defaultdict(list)
    for ci, h in enumerate(H):
        for f in HEX_FACES:
            q = [int(h[i]) for i in f]
            own[frozenset(q)].append((ci, q))
    return [v[0][1] for v in own.values() if len(v) == 1]


def grading(n, growth):
    """t_0 = 0 (outer) ... t_n = 1 (wall), cells shrinking geometrically towards the wall."""
    h = growth ** np.arange(n)[::-1]                    # outermost cell largest
    return np.r_[0.0, np.cumsum(h) / h.sum()]


def seam_path(seamP, edges, a_pt, b_pt, closed_start=None):
    """Ordered polyline along the wall seam graph `edges` from the vertex nearest a_pt to the one nearest b_pt
    (shortest path); a_pt / b_pt are prepended / appended."""
    import heapq
    adj = defaultdict(list)
    for u, v in edges:
        w = float(np.linalg.norm(seamP[u] - seamP[v])); adj[u].append((v, w)); adj[v].append((u, w))
    vs = np.array(sorted(adj))
    s0 = int(vs[np.argmin(np.linalg.norm(seamP[vs] - a_pt, axis=1))])
    t0 = int(vs[np.argmin(np.linalg.norm(seamP[vs] - b_pt, axis=1))])
    dist, prev, pq = {s0: 0.0}, {}, [(0.0, s0)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == t0:
            break
        if d > dist.get(u, np.inf):
            continue
        for v, w in adj[u]:
            if d + w < dist.get(v, np.inf):
                dist[v] = d + w; prev[v] = u; heapq.heappush(pq, (d + w, v))
    path = [t0]
    while path[-1] != s0:
        path.append(prev[path[-1]])
    path = path[::-1]
    return np.vstack([a_pt[None], seamP[path], b_pt[None]])


def resample_at(Q, u):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]
    t = np.asarray(u) * s[-1]
    return np.stack([np.interp(t, s, Q[:, d]) for d in range(3)], 1)


def map_patch(P, F, quads, nodes, groups, bnd_nodes, wall, full_proj, seam, relax=20, verbose=False):
    """Outer patch nodes -> wall points: corners (>= 3 patches) by alternating projection, seam nodes
    distributed by arc length along the wall seam between their chain ends, the rest nearest-point projected
    and Laplace-relaxed on the wall with the boundary fixed."""
    lab_of = {v: labs for labs, vs in groups.items() for v in vs}
    W = {}
    for v in nodes:                                       # first guess: nearest point on the wall (+ seam patches)
        labs = lab_of[v]
        W[v] = full_proj.onto(P[v][None], list(labs))[0] if len(labs) >= 3 else \
            full_proj.patches[wall].project(P[v][None])[1][0]
    # seam chains: boundary edges whose two nodes share the same 2-label set (wall + one side)
    nb = defaultdict(set)
    for k in F:
        q = quads[k][0]
        for a, b in zip(q, q[1:] + q[:1]):
            nb[a].add(b); nb[b].add(a)
    seam_nodes = {v for v in nodes if len(lab_of[v]) == 2}
    done = set()
    for v0 in sorted(seam_nodes):
        if v0 in done:
            continue
        key = lab_of[v0]
        # collect the connected chain of nodes with this label set (through boundary neighbours)
        comp, stack = {v0}, [v0]
        while stack:
            u = stack.pop()
            for w in nb[u]:
                if w in bnd_nodes and w not in comp and lab_of.get(w) == key:
                    comp.add(w); stack.append(w)
        done |= comp
        ends = [u for u in comp if sum(1 for w in nb[u] if w in comp) < 2]
        corner_nb = {u: [w for w in nb[u] if w in bnd_nodes and len(lab_of.get(w, ())) >= 3 and set(key) <= set(lab_of[w])]
                     for u in ends}
        if not seam.has(key):
            continue
        E = seam.tree[tuple(sorted(key))][0]
        if len(ends) == 2:                                # open chain: corner -> ... -> corner
            a = ends[0]; order = [a]
            while len(order) < len(comp):
                nxt = [w for w in nb[order[-1]] if w in comp and w not in order]
                if not nxt:
                    break
                order.append(nxt[0])
            ca = corner_nb[order[0]]; cb = corner_nb[order[-1]]
            chain = ([ca[0]] if ca else []) + order + ([cb[0]] if cb else [])
            X = np.array([P[u] for u in chain]); s_ = np.r_[0, np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))]
            u = s_ / s_[-1]
            A = W[chain[0]] if ca else seam.onto(P[chain[0]][None], key)[0]
            B = W[chain[-1]] if cb else seam.onto(P[chain[-1]][None], key)[0]
            path = seam_path(seam.P, E, A, B)
            sn, tot = poly_param(path, X)
            sn[0], sn[-1] = 0.0, tot
            Wq = resample_at(path, regular_params(sn, u * tot, tot, False) / tot)
            near = seam.onto(np.array([P[uu] for uu in chain]), key)
            dev = np.linalg.norm(Wq - near, axis=1)
            print(f"[layers]   open seam {key}: {len(chain)} nodes, |W - nearest| median {np.median(dev):.4f} max {dev.max():.4f}", flush=True)
            for uu, x in zip(chain, Wq):
                if len(lab_of[uu]) == 2:
                    W[uu] = x
        elif not ends:                                    # closed loop (e.g. blade root ring)
            order = [v0]
            while len(order) < len(comp):
                nxt = [w for w in nb[order[-1]] if w in comp and w not in order]
                if not nxt:
                    break
                order.append(nxt[0])
            X = np.array([P[u] for u in order + order[:1]])
            s_ = np.r_[0, np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))]; u = s_ / s_[-1]
            A = seam.onto(P[order[0]][None], key)[0]; Bm = seam.onto(P[order[len(order) // 2]][None], key)[0]
            half1 = seam_path(seam.P, E, A, Bm)
            # second half: the other way round (remove the first half's interior vertices from the graph)
            used = {tuple(np.round(p, 12)) for p in half1[1:-1]}
            E2 = [e for e in E if tuple(np.round(seam.P[e[0]], 12)) not in used and tuple(np.round(seam.P[e[1]], 12)) not in used]
            half2 = seam_path(seam.P, E2, Bm, A) if E2 else half1[::-1]
            loop = np.vstack([half1, half2[1:]])
            # orientation: compare the guess of the second node with both directions
            Xo = np.array([P[uu] for uu in order])
            best = None
            for lp in (loop, loop[::-1]):
                sn, tot = poly_param(lp, Xo)
                sp = regular_params(sn, u * tot, tot, True)
                cand = resample_at(lp, (sp + sn[0]) % tot / tot)
                err = np.linalg.norm(cand[:-1] - seam.onto(Xo, key), axis=1).mean()
                if best is None or err < best[0]:
                    best = (err, cand)
            Wq = best[1]
            near = seam.onto(np.array([P[uu] for uu in order]), key)
            dev = np.linalg.norm(Wq[:-1] - near, axis=1)
            print(f"[layers]   closed seam {key}: {len(order)} nodes, loop length {np.linalg.norm(np.diff(loop, axis=0), axis=1).sum():.3f}, "
                  f"outer {s_[-1]:.3f}, |W - nearest| median {np.median(dev):.4f} max {dev.max():.4f}", flush=True)
            for uu, x in zip(order, Wq[:-1]):
                W[uu] = x
    # column points of earlier layers: same parameter between the images of their line ends
    cols = [v for v in nodes if v in COLPT and COLPT[v][0] in W and COLPT[v][1] in W and v not in seam_nodes]

    def interp_cols():
        if cols:
            R = np.array([(1 - COLPT[v][2]) * W[COLPT[v][0]] + COLPT[v][2] * W[COLPT[v][1]] for v in cols])
            R = full_proj.patches[wall].project(R)[1]
            for v, x in zip(cols, R):
                W[v] = x
    interp_cols()
    # interior: Laplace relaxation on the wall, boundary and column points fixed (re-interpolated)
    colset = set(cols)
    free = [v for v in nodes if v not in bnd_nodes and v not in colset]
    if relax and free:
        fi = {v: i for i, v in enumerate(free)}
        X = np.array([W[v] for v in free])
        nbl = [list(nb[v]) for v in free]
        for _ in range(relax):
            M = np.array([np.mean([X[fi[w]] if w in fi else W[w] for w in ns], 0) for ns in nbl])
            X = full_proj.patches[wall].project(0.5 * X + 0.5 * M)[1]
            for v, x in zip(free, X):
                W[v] = x
            interp_cols()
    return W


def poly_param(Q, X):
    """Arc-length parameter of the nearest point on polyline Q for each X."""
    A, B = Q[:-1], Q[1:]; d = B - A; L = np.linalg.norm(d, axis=1); s0 = np.r_[0, np.cumsum(L)]
    t = np.clip(np.einsum("nkj,kj->nk", X[:, None] - A[None], d) / (L ** 2).clip(1e-30), 0, 1)
    D = np.linalg.norm(A[None] + t[..., None] * d[None] - X[:, None], axis=2)
    k = np.argmin(D, 1)
    return s0[k] + t[np.arange(len(X)), k] * L[k], s0[-1]


def regular_params(s, s_ref, total, closed, min_frac=0.3, smooth=3):
    """Monotone, non-collapsing parameters close to the nearest-point parameters s (ends fixed; closed loops
    unwrapped from the first node). Minimum gap = min_frac x mean gap; a few light smoothing passes."""
    s = np.asarray(s, float).copy()
    if closed:
        s = np.r_[np.mod(s - s[0], total), total]
    n = len(s); gap = min_frac * total / (n - 1)
    s = np.maximum.accumulate(s)
    for i in range(1, n):
        s[i] = max(s[i], s[i - 1] + gap)
    if s[-1] > total:                                     # squeeze back into [0, total]
        s = s * total / s[-1]
    s[-1] = total
    for _ in range(smooth):
        s[1:-1] = 0.75 * s[1:-1] + 0.125 * (s[:-2] + s[2:])
    return s


NLAB = {}
COLPT = {}
COLS = {}


def close_layers(Pm, H, raw, full, n_layers=8, growth=1.2, relax=20, verbose=True):
    S, T, L = raw["surface_points"], raw["surface_tris"], raw["surface_tri_label"]
    diagl = float(np.linalg.norm(np.ptp(S, 0)))
    core_proj = SurfaceProjector(S, T, L); full_proj = SurfaceProjector(*full); seam = MW.SeamProjector(*full); smooth = MW.SmoothProjector(*full)
    cls = BI.classify_core_patches(raw, full_proj, 2e-3 * diagl)
    real_full = {l: c[1] for l, c in cls.items() if c[0] == "real"}
    cuts = [l for l, c in cls.items() if c[0] == "cut"]
    Q = boundary_quads(H)
    cen = np.array([Pm[q].mean(0) for q in Q])
    lab = np.array(core_proj.labels)[np.argmin(core_proj.dists(cen), 1)]
    quads = {frozenset(q): (q, int(l)) for q, l in zip(Q, lab)}
    P = [p for p in Pm]; H = [list(map(int, h)) for h in H]; part = [0] * len(H)
    for w in {c[1] for c in cls.values()}:
        real_full[f"wall{w}"] = w

    def edge_labels():
        e2l = defaultdict(set); v2l = defaultdict(set)
        for q, l in quads.values():
            for a, b in zip(q, q[1:] + q[:1]):
                e2l[(min(a, b), max(a, b))].add(l); v2l[a].add(l)
        return e2l, v2l

    def n_real_nb(c):
        e2l, _ = edge_labels()
        return len({m for ls in e2l.values() if c in ls for m in ls if m in real_full and m != c})
    order = sorted(cuts, key=n_real_nb)
    tk = grading(n_layers, growth)
    rep = {"classes": {str(k): v for k, v in cls.items()}, "order": order, "cuts": []}
    for ci, c in enumerate(order, 1):
        wall = cls[c][1]
        e2l, v2l = edge_labels()
        F = [k for k, (q, l) in quads.items() if l == c]
        nodes = sorted({v for k in F for v in quads[k][0]})
        # wall partner per node; nodes on a seam with a real patch also onto that patch
        groups = defaultdict(list)
        for v in nodes:
            labs = tuple(sorted({wall} | {real_full[m] for m in v2l[v] - {c} if m in real_full}))
            groups[labs].append(v)
        bnd_nodes = {v for (a_, b_), ls in e2l.items() if c in ls and len(ls) > 1 for v in (a_, b_)}
        for labs, vs in groups.items():
            for v in vs:
                NLAB[v] = "|".join(map(str, labs)) + f"@{c}"
        W = map_patch(P, F, quads, nodes, groups, bnd_nodes, wall, full_proj, seam, relax=relax)
        col = {}                                          # node -> ids along its line, k = 0 .. n_layers
        for labs, vs in groups.items():
            side = [l for l in labs if l != wall]         # seam column: stays on the neighbouring patch(es)
            X0 = np.array([P[v] for v in vs]); X1 = np.array([W[v] for v in vs])
            for j, t in enumerate(tk[1:], 1):
                X = (1 - t) * X0 + t * X1
                if side and j < n_layers:
                    X = smooth.onto(X, side[0]) if len(side) == 1 else (
                        seam.onto(X, side) if len(side) == 2 and seam.has(side) else full_proj.onto(X, side))
                for v, x in zip(vs, X):
                    col.setdefault(v, [v]).append(len(P)); P.append(x)
        # columns of earlier-layer column points follow the bend of their line-end columns (e.g. the O-ring
        # band under the hub layer leans with the blade): add the ends' deviations from straight, blended
        dev = {}
        for u in nodes:
            dev[u] = [P[col[u][j]] - ((1 - tk[j]) * P[u] + tk[j] * W[u]) for j in range(n_layers + 1)]
        for v in nodes:
            if v in COLPT and COLPT[v][0] in dev and COLPT[v][1] in dev:
                rt, tp, tv = COLPT[v]
                for j in range(1, n_layers):
                    P[col[v][j]] = P[col[v][j]] + (1 - tv) * dev[rt][j] + tv * dev[tp][j]
        for v in nodes:
            COLS.setdefault(v, {})[ci] = col[v]
        for v in nodes:                                   # column points remember their line (root, tip, t)
            for j in range(1, n_layers):
                COLPT[col[v][j]] = (v, col[v][-1], float(tk[j]))
        thick = np.array([np.linalg.norm(W[v] - P[v]) for v in nodes])
        n0 = len(H)
        for k in F:
            q = quads[k][0]
            for j in range(n_layers):
                lo = [col[v][j] for v in q]; hi = [col[v][j + 1] for v in q]
                h = lo + hi
                if BI.block_volume(np.array([P[i] for i in h])) < 0:
                    h = lo[::-1] + hi[::-1]
                H.append(h); part.append(ci)
            for a, b in zip(q, q[1:] + q[:1]):          # side quads -> label of the neighbouring patch
                others = e2l[(min(a, b), max(a, b))] - {c}
                if len(others) == 1:
                    o = next(iter(others))
                    for j in range(n_layers):
                        s = [col[a][j], col[b][j], col[b][j + 1], col[a][j + 1]]
                        quads[frozenset(s)] = (s, o)
            del quads[k]
            wq = [col[v][-1] for v in q]
            quads[frozenset(wq)] = (wq, f"wall{wall}")
        rep["cuts"].append({"label": c, "wall": wall, "quads": len(F), "cells": len(H) - n0,
                            "thickness_min": float(thick.min()), "thickness_med": float(np.median(thick))})
        if verbose:
            print(f"[layers] cut {c} -> wall {wall}: {len(F)} quads x {n_layers} layers, thickness "
                  f"{thick.min():.3f}..{np.median(thick):.3f}..{thick.max():.3f}", flush=True)
    return np.asarray(P), np.asarray(H, np.int64), np.asarray(part), quads, rep


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", default="tistos_s1_0131")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-h", type=float, default=0.05)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--growth", type=float, default=1.2)
    ap.add_argument("--relax", type=int, default=20)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    raw = dict(np.load(f"{BI.RUNS}/{a.name}/sample.npz", allow_pickle=True))
    full = MW.load_boundary(f"{BI.RUNS}/{a.name}/case/mesh.msh")[:3]
    V, B, curves = BI.gt_post_structure(a.name)
    surf = (raw["surface_points"], raw["surface_tris"], raw["surface_tri_label"])
    refill_polytron(V, B, curves, a.target_h, f"{a.out}/core.vtk", write_edges=False, surface=surf)
    m = meshio.read(f"{a.out}/core.vtk")
    P, H, part, quads, rep = close_layers(m.points, m.cells[0].data, raw, full, a.layers, a.growth, a.relax)
    s = np.nan_to_num(diag.cell_sj(P[H]), nan=-1.0)
    rep["cells"] = int(len(H)); rep["inverted"] = int((s <= 0).sum())
    rep["parts"] = {int(p): {"cells": int((part == p).sum()), "inverted": int((s[part == p] <= 0).sum()),
                             "sj_p1": float(np.percentile(s[part == p], 1)), "sj_min": float(s[part == p].min())}
                    for p in np.unique(part)}
    meshio.Mesh(P, [("hexahedron", H)], cell_data={"part": [part], "sj": [s], "inverted": [(s <= 0).astype(np.int32)]}
                ).write(f"{a.out}/mesh_full.vtk", binary=True)
    print(f"[layers] cells {len(H)} inverted {rep['inverted']} | " +
          " ".join(f"part {p}: {v['cells']} cells inv {v['inverted']} sj_p1 {v['sj_p1']:.2f} min {v['sj_min']:.2f}"
                   for p, v in rep["parts"].items()), flush=True)
    json.dump(rep, open(f"{a.out}/report.json", "w"), indent=1, default=str)
    json.dump({str(k): v for k, v in NLAB.items()}, open(f"{a.out}/node_labs.json", "w"))
    np.save(f"{a.out}/colpt_ids.npy", np.array(sorted(COLPT), np.int64))
    rc = [(r, c) for r in sorted(COLS) for c in sorted(COLS[r])]
    np.savez(f"{a.out}/columns.npz", roots=np.array([r for r, _ in rc]), cut=np.array([c for _, c in rc]),
             cols=np.array([COLS[r][c] for r, c in rc]), P=P, order=np.array(rep["order"]))


if __name__ == "__main__":
    main()
