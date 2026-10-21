#!/usr/bin/env python3
"""Full-domain BLOCKING (core + boundary-layer blocks + O-ring blocks) at block level, then ONE curved TFI.

Topology as in the dtOO mesh: the boundary layers lie under the core only, the O-grid reaches hub and shroud.
The dtOO volume cells are discarded; its boundary AND internal region interfaces are used as geometry
(msh_walls.load_regions): walls per region (hub under the layer / under the O-grid, ...), the cut surfaces
(core | region) and the O-grid's outer side inside the layers (layer | O-grid).

  1. every core patch is either on the physical boundary or on a core | region interface (a cut, region R);
  2. cuts touching physical patches first (layers), then the cut enclosed by other cuts (O-ring); every block face
     on the cut becomes one new block towards the wall of R (nearest wall sub-patch of R); side faces inherit the
     neighbouring label, so the O-ring also extrudes the layers' side faces and reaches hub and shroud;
  3. corners: wall + the geometry of their neighbours (physical patch -> its part owned by R; another cut R' ->
     the R | R' interface), 1 patch smooth projection, 2 patches seam polyline, more alternating;
  4. edges: arc-length sampled, mapped point by point (smooth / seam projection), smoothed on the target
     geometry, monotone non-collapsing parameters -> point i of the image corresponds to point i of the source;
     radial edges graded towards the wall (fixed number of layers in the TFI);
  5. one joint TFI (blade/tfi_param.py) over all blocks, projected onto the physical boundary.

    python blocking.py --name tistos_s1_0131 --out out/blocking_0131
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
from scipy.spatial import cKDTree  # noqa: E402

import blade_insert as BI  # noqa: E402
import close_layers as CL  # noqa: E402
import diag  # noqa: E402
import msh_walls as MW  # noqa: E402
import tfi_param as TP  # noqa: E402
from meshtron.geometry.polytron_tfi import SurfaceProjector, _arclength, _boundary_labels  # noqa: E402

NS = 129                                                  # samples per mapped edge


class Geo:
    """Projection onto the labelled region geometry."""

    def __init__(self, P, T, L, info):
        self.P, self.T, self.L, self.info = P, T, L, info
        self.near = SurfaceProjector(P, T, L)
        self.smooth = MW.SmoothProjector(P, T, L)
        self.seam = MW.SeamProjector(P, T, L)

    def onto(self, X, labs):
        X = np.asarray(X, float).reshape(-1, 3); labs = sorted(set(labs))
        if len(labs) == 1:
            return self.smooth.onto(X, labs[0])
        if len(labs) == 2 and self.seam.has(labs):
            return self.seam.onto(X, labs)
        return self.near.onto(X, labs)

    def phys_part(self, p, R):
        """label of physical patch p owned by region R (None if R does not touch p)."""
        for i, d in enumerate(self.info):
            if d["kind"] == "phys" and d["phys"] == p and d["regions"] == (R,):
                return i
        return None

    def iface(self, R0, R1):
        for i, d in enumerate(self.info):
            if d["kind"] == "iface" and d["regions"] == tuple(sorted((R0, R1))):
                return i
        return None


def descend(geo, X, side, wall, K=24):
    """Walk points WITHIN the side geometry towards the wall; returns the paths [K+1, n, 3] (last = on wall+side)."""
    X = np.asarray(X, float).reshape(-1, 3).copy(); path = [X.copy()]
    for k in range(K):
        Wp = geo.near.onto(X, [wall])
        X = geo.onto(X + (Wp - X) / (K - k), side)
        path.append(X.copy())
    path[-1] = geo.onto(path[-1], sorted(set(side) | {wall}))
    return np.stack(path)


def graded_path(Pth, tk):
    """Resample a column path (outer -> wall) at the grading fractions tk (by arc length)."""
    return CL.resample_at(Pth, tk)


def foot_union(geo, X, labs):
    """Nearest point on the union of several patches (not their intersection)."""
    X = np.asarray(X, float).reshape(-1, 3)
    best = None; bd = None
    for l in labs:
        d, q = geo.near.patches[l].project(X)[:2]
        if best is None:
            best, bd = q.copy(), d.copy()
        else:
            m = d < bd; best[m] = q[m]; bd[m] = d[m]
    return best


def level_onto(geo, X, target, ref_labs, d, iters=12):
    """Points on `target` at distance d (per point) from the reference wall `ref_labs` (keeps a seam between two
    cuts at its level when it is mapped onto a third wall)."""
    X = np.asarray(X, float).reshape(-1, 3); d = np.broadcast_to(np.asarray(d, float), (len(X),))
    p = geo.near.onto(X, [target])
    for _ in range(iters):
        D = geo.near.dists(p)[:, [geo.near.labels.index(l) for l in ref_labs]]
        j = np.argmin(D, 1); h = D[np.arange(len(p)), j]
        foot = np.stack([geo.near.patches[ref_labs[jj]].project(p[i:i + 1])[1][0] for i, jj in enumerate(j)])
        g = (p - foot) / np.maximum(h, 1e-12)[:, None]
        p = geo.near.onto(p + (d - h)[:, None] * g, [target])
    return p


def smooth_on(geo, W, labs, iters=20):
    """Laplace smoothing of a polyline on its target geometry, ends fixed."""
    W = W.copy()
    for _ in range(iters):
        W[1:-1] = geo.onto(0.5 * W[1:-1] + 0.25 * (W[:-2] + W[2:]), labs)
    return W


def map_curve(geo, Q, labs, A, B, wall=None, shift=None, level=None, foot=None):
    """Source polyline Q (NS points, correspondence by index) -> image polyline on `labs` from A to B.
    shift: tangential offset field (lean of nearby side columns); level: (ref_labs, distances) constraint."""
    side = [t for t in labs if t != wall]
    if side and wall is not None:
        W0 = descend(geo, Q, side, wall)[-1]
    elif level is not None:
        W0 = level_onto(geo, Q, wall, level[0], level[1])
    else:
        F = foot(Q) if foot is not None else geo.near.onto(Q, [wall])
        W0 = geo.onto(F + (shift(Q) if shift is not None else 0), labs)
    W0[0], W0[-1] = A, B
    # smoothing only where the curve is guided by a seam (1D); on a surface Laplace smoothing shortens the
    # curve and pulls it across neighbouring curves (e.g. onto the O-ring footprint)
    Ws = smooth_on(geo, W0, labs) if (side and wall is not None) or level is not None or len(labs) > 1 else W0
    s, tot = CL.poly_param(Ws, W0)
    s[0], s[-1] = 0.0, tot
    W = CL.resample_at(Ws, CL.regular_params(s, None, tot, False) / tot)
    W[1:-1] = geo.onto(W[1:-1], labs)
    W[0], W[-1] = A, B
    return W


def index_resample(Q, n):
    Q = np.asarray(Q, float); u = np.linspace(0, 1, len(Q)); t = np.linspace(0, 1, n)
    return np.stack([np.interp(t, u, Q[:, d]) for d in range(3)], 1)


def build(V, B, curves, raw, geo, n_layers=8, growth=1.2, verbose=True):
    S, T, L = raw["surface_points"], raw["surface_tris"], raw["surface_tri_label"]
    diagl = float(np.linalg.norm(np.ptp(S, 0)))
    core_proj = SurfaceProjector(S, T, L)
    # classify core patches: nearest region-geometry label (median)
    cls = {}
    for l in np.unique(L):
        c = S[T[L == l]].mean(1)
        d = np.median(geo.near.dists(c), 0); j = int(np.argmin(d)); lab = geo.near.labels[j]; inf = geo.info[lab]
        if inf["kind"] == "phys":
            cls[int(l)] = ("phys", inf["phys"])
        else:
            R = [r for r in inf["regions"] if r != 0][0]          # region 0 = core
            cls[int(l)] = ("cut", R)
    face_lab, cyc_of = _boundary_labels(V, B, curves, core_proj)
    faces = {}
    for k, l in face_lab.items():
        if l is None:
            continue
        kind, x = cls[int(l)]
        faces[k] = (cyc_of[k], (kind, x))
    V = [np.asarray(v, float) for v in V]; B = [list(map(int, b)) for b in B]; part = [0] * len(B)
    curves = dict(curves); param = set(); radial_blocks = {}; LEVEL = {}; LEVEL_E = {}

    def adjacency():
        e2l = defaultdict(set); v2l = defaultdict(set)
        for cyc, l in faces.values():
            for a, b in zip(cyc, cyc[1:] + cyc[:1]):
                e2l[(min(a, b), max(a, b))].add(l); v2l[a].add(l)
        return e2l, v2l
    cuts = sorted({l for _, l in faces.values() if l[0] == "cut"})

    def n_phys_nb(c):
        e2l, _ = adjacency()
        return len({m for ls in e2l.values() if c in ls for m in ls if m[0] == "phys"})
    order = sorted(cuts, key=lambda c: -n_phys_nb(c))
    layer_cuts = {c for c in cuts if n_phys_nb(c) > 0}             # decided before any extrusion
    tk = CL.grading(n_layers, growth)
    done = set()
    rep = {"classes": {str(k): v for k, v in cls.items()}, "order": [str(c) for c in order], "cuts": []}
    for ci, c in enumerate(order, 1):
        R = c[1]
        # wall of R: nearest physical part owned by R
        Fk = [k for k, (cyc, l) in faces.items() if l == c]
        cen = np.array([np.mean([V[v] for v in faces[k][0]], 0) for k in Fk])
        cand = [i for i, d in enumerate(geo.info) if d["kind"] == "phys" and d["regions"] == (R,)]
        dd = np.median(geo.near.dists(cen)[:, [geo.near.labels.index(i) for i in cand]], 0)
        wall = cand[int(np.argmin(dd))]; wall_phys = geo.info[wall]["phys"]
        e2l, v2l = adjacency()

        def target(lbl):
            if lbl == c:
                return None
            if lbl[0] == "phys":
                return geo.phys_part(lbl[1], R)
            if lbl[0] == "cut":
                return geo.iface(R, lbl[1])
            return None

        def labs_of(ls):
            return sorted({wall} | {t for t in (target(m) for m in ls) if t is not None})
        corners = sorted({v for k in Fk for v in faces[k][0]})
        phys_labs = [i for i, d in enumerate(geo.info) if d["kind"] == "phys" and d["phys"] == wall_phys]
        # seams with cuts processed later keep their level relative to this wall
        for v in corners:
            if any(m[0] == "cut" and m != c and m not in done for m in v2l[v]):
                LEVEL.setdefault(v, (phys_labs, float(geo.near.dists(V[v][None])[0, [geo.near.labels.index(l) for l in phys_labs]].min())))
        for k in Fk:
            cyc = faces[k][0]
            for a, b in zip(cyc, cyc[1:] + cyc[:1]):
                key = (min(a, b), max(a, b))
                if any(m[0] == "cut" and m != c and m not in done for m in e2l[key]):
                    LEVEL_E.setdefault(key, phys_labs)
        wmap, moved, shifts = {}, [], []
        free = []
        for v in corners:
            labs = labs_of(v2l[v]); side = [t for t in labs if t != wall]
            if side:
                pth = descend(geo, V[v][None], side, wall)[:, 0]
                q = pth[-1]
                shifts.append((V[v], q - geo.near.onto(V[v][None], [wall])[0]))
            elif v in LEVEL:
                q = level_onto(geo, V[v][None], wall, LEVEL[v][0], LEVEL[v][1])[0]
                pth = np.linspace(V[v], q, 25)
            else:
                free.append(v); continue
            wmap[v] = len(V); V.append(q); moved.append(float(np.linalg.norm(q - V[v])))
            X = graded_path(pth, tk); X[0], X[-1] = V[v], q            # radial edge, graded towards the wall
            curves[(v, wmap[v])] = X; param.add((v, wmap[v]))
        # edges in two passes: constrained ones (seams with side geometry, level) give dense displacement
        # samples; free corners / edges then get the same displacement field interpolated (relative distances
        # to the seams are kept -- the core ring around the O-ring and the O-ring footprint cannot collide)
        samp_x, samp_d = [], []
        # harmonic surface map of this cut (if it is a core patch of the sample surface)
        smap = None
        core_l = [l for l, cc in cls.items() if cc == c]
        if core_l and c in layer_cuts:                # layer cuts only; the enclosed cut keeps its level rules
            import surfmap as SM
            Tc = T[L == core_l[0]]
            vlab_s = defaultdict(set)
            for t, l in zip(T, L):
                if int(l) != core_l[0]:
                    for v in t:
                        vlab_s[int(v)].add(cls[int(l)])

            # boundary labels of the patch's boundary, by nearest vertex of the original triangulation
            bverts = np.unique(Tc); btree = cKDTree(S[bverts])

            def bimg(X0):
                X0 = np.asarray(X0, float)
                _, j = btree.query(X0)
                # a refined boundary point takes the labels common to the two nearest original boundary vertices
                _, j2 = btree.query(X0, k=2)
                out = np.zeros((len(X0), 3)); groups = defaultdict(list)
                for i in range(len(X0)):
                    a, b = int(bverts[j2[i, 0]]), int(bverts[j2[i, 1]])
                    ls = (vlab_s[a] & vlab_s[b]) or vlab_s[a]
                    groups[tuple(sorted(labs_of(ls)))].append(i)
                for labs, ii in groups.items():
                    side = [t for t in labs if t != wall]
                    if side:
                        out[ii] = descend(geo, X0[ii], side, wall)[-1]
                    else:
                        out[ii] = geo.onto(foot_union(geo, X0[ii], phys_labs), [wall])
                return out
            smap = SM.SurfaceMap(S, Tc, bimg)
        for v in corners:
            if v in wmap:
                samp_x.append(V[v]); samp_d.append(V[wmap[v]] - foot_union(geo, V[v][None], phys_labs)[0])

        def idw(X, p=4.0):
            X = np.asarray(X, float).reshape(-1, 3)
            if not samp_x:
                return np.zeros_like(X)
            S0 = np.array(samp_x); Sd = np.array(samp_d)
            out = np.zeros_like(X)
            for i0_ in range(0, len(X), 512):
                D = np.linalg.norm(X[i0_:i0_ + 512, None] - S0[None], axis=2)
                w = 1.0 / np.maximum(D, 1e-9) ** p
                out[i0_:i0_ + 512] = (w @ Sd) / w.sum(1, keepdims=True)
            return out
        edges = []
        for k in Fk:
            cyc = faces[k][0]
            for a, b in zip(cyc, cyc[1:] + cyc[:1]):
                key = (min(a, b), max(a, b))
                if key not in [e for e, _ in edges]:
                    side = [t for t in labs_of(e2l[key] - {c}) if t != wall]
                    edges.append((key, bool(side) or key in LEVEL_E))
        for pass_constrained in (True, False):
            if not pass_constrained:
                for v in free:                                # free corners: interpolated displacement
                    q = geo.onto(smap(V[v][None]) if smap is not None else
                                 foot_union(geo, V[v][None], phys_labs) + idw(V[v][None]), [wall])[0]
                    wmap[v] = len(V); V.append(q); moved.append(float(np.linalg.norm(q - V[v])))
                    X = (1 - tk[:, None]) * V[v] + tk[:, None] * q
                    curves[(v, wmap[v])] = X; param.add((v, wmap[v]))
            for key, constrained in edges:
                if constrained != pass_constrained:
                    continue
                wk = (min(wmap[key[0]], wmap[key[1]]), max(wmap[key[0]], wmap[key[1]]))
                if wk in curves:
                    continue
                Q = curves.get(key)
                Q = np.linspace(V[key[0]], V[key[1]], NS) if Q is None else np.asarray(Q, float)
                Q = index_resample(Q, NS) if key in param else _arclength(Q, NS)
                lv = None
                if key in LEVEL_E and not [t for t in labs_of(e2l[key] - {c}) if t != wall]:
                    rl = LEVEL_E[key]
                    dq = geo.near.dists(Q)[:, [geo.near.labels.index(l) for l in rl]].min(1)
                    lv = (rl, dq)
                if smap is not None and not constrained:          # free edge on the cut: harmonic map
                    W = geo.onto(smap(Q), [wall]); W[0], W[-1] = V[wmap[key[0]]], V[wmap[key[1]]]
                else:
                    W = map_curve(geo, Q, labs_of(e2l[key] - {c}), V[wmap[key[0]]], V[wmap[key[1]]], wall,
                                  shift=idw, level=lv, foot=lambda X: foot_union(geo, X, phys_labs))
                curves[wk] = W if wmap[key[0]] < wmap[key[1]] else W[::-1]
                param.add(wk)
                if pass_constrained:
                    samp_x.extend(Q[1:-1]); samp_d.extend(W[1:-1] - foot_union(geo, Q[1:-1], phys_labs))
        for k in Fk:
            cyc = faces[k][0]
            blk = list(cyc) + [wmap[v] for v in cyc]
            if BI.block_volume(np.asarray([V[i] for i in blk])) < 0:
                blk = list(cyc[::-1]) + [wmap[v] for v in cyc[::-1]]
            radial_blocks[len(B)] = n_layers
            B.append(blk); part.append(ci)
            for a, b in zip(cyc, cyc[1:] + cyc[:1]):
                others = e2l[(min(a, b), max(a, b))] - {c}
                if len(others) == 1:
                    side = [a, b, wmap[b], wmap[a]]
                    faces[frozenset(side)] = (side, next(iter(others)))
            del faces[k]
            wq = [wmap[v] for v in cyc]
            faces[frozenset(wq)] = (wq, ("phys", wall_phys))
        done.add(c)
        rep["cuts"].append({"cut": str(c), "wall": int(wall), "faces": len(Fk),
                            "corner_move_med": float(np.median(moved)), "corner_move_max": float(max(moved))})
        if verbose:
            print(f"[blocking] cut {c} -> wall label {wall} (phys {wall_phys}): {len(Fk)} faces -> {len(Fk)} blocks, "
                  f"corner move median {np.median(moved):.3f} max {max(moved):.3f}", flush=True)
    return (np.asarray(V), np.asarray(B, np.int64), curves, np.asarray(part), frozenset(param), radial_blocks, rep)


def write_block_vtk(path, V, B, curves, part, geo):
    pts, lines, sets = [], [], []
    edge_part = {}
    for bi, b in enumerate(B):
        for i, j in BI.M4.HEX_EDGES:
            k = (min(int(b[i]), int(b[j])), max(int(b[i]), int(b[j])))
            edge_part[k] = min(edge_part.get(k, 99), int(part[bi]))
    base = 0
    for k, Q in curves.items():
        if k not in edge_part:
            continue
        Q = np.asarray(Q); n = len(Q); pts.append(Q)
        lines.append(np.stack([base + np.arange(n - 1), base + np.arange(1, n)], 1)); sets.append(np.full(n - 1, edge_part[k]))
        base += n
    tri = []; tset = []
    for i, d in enumerate(geo.info):
        if d["kind"] == "phys" and d["phys"] in (4, 5, 6) or d["kind"] == "iface":
            tri.append(geo.T[geo.L == i] + base); tset.append(np.full(int((geo.L == i).sum()), 100 + i))
    pts.append(geo.P)
    meshio.Mesh(np.concatenate(pts), [("line", np.concatenate(lines)), ("triangle", np.concatenate(tri))],
                cell_data={"set": [np.concatenate(sets).astype(np.int32), np.concatenate(tset).astype(np.int32)]}
                ).write(path, binary=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", default="tistos_s1_0131")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-h", type=float, default=0.05)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--growth", type=float, default=1.2)
    ap.add_argument("--relocate", action="store_true", help="slide cut-seam corners for block quality first")
    ap.add_argument("--min-gap", type=float, default=0.0, help="push core corners/edges away from cut|cut seams "
                    "to this fraction of the layer thickness before the blocking")
    ap.add_argument("--opt", type=int, default=0, help="iterations of the surrogate optimisation of the full blocking")
    ap.add_argument("--opt-n", type=int, default=9, help="surrogate samples per edge")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    raw = dict(np.load(f"{BI.RUNS}/{a.name}/sample.npz", allow_pickle=True))
    P, T, L, info, rtype = MW.load_regions(f"{BI.RUNS}/{a.name}/case/mesh.msh")
    geo = Geo(P, T, L, info)
    V, B, curves = BI.gt_post_structure(a.name)
    if a.min_gap > 0:
        import relocate as RL
        V, curves, grep_ = RL.widen_gaps(V, B, curves, raw, geo, a.min_gap)
    if a.relocate:
        import relocate as RL
        V0 = np.asarray(V, float).copy()
        V, moved, movable = RL.relocate(V, B, curves, raw, geo)
        curves = RL.update_curves(V0, V, curves, raw, geo, B)
        print("[blocking] relocated corners:", {int(k): round(v, 3) for k, v in moved.items()}, flush=True)
    V2, B2, cur2, part, param, radial, rep = build(V, B, curves, raw, geo, a.layers, a.growth)
    if a.opt:
        import opt_blocking as OB
        keep_l = [i for i, d in enumerate(info) if d["kind"] == "phys"]
        V2, cur2, olog, oinfo = OB.optimise_blocking(V2, B2, cur2, param, geo, keep_l, n=a.opt_n, iters=a.opt)
        rep["opt"] = {"log": olog, **oinfo}
        print("[blocking] opt", oinfo, flush=True)
    vols = np.array([BI.block_volume(V2[b]) for b in B2])
    rep["negative_blocks"] = int((vols <= 0).sum())
    print(f"[blocking] blocks {len(B)} -> {len(B2)} (parts {np.bincount(part).tolist()}), negative-volume blocks "
          f"{rep['negative_blocks']}", flush=True)
    np.savez(f"{a.out}/blocks_full.npz", V=V2, B=B2, part=part,
             curve_keys=np.array(list(cur2.keys())), curve_off=np.cumsum([0] + [len(q) for q in cur2.values()]),
             curve_pts=np.concatenate(list(cur2.values())))
    write_block_vtk(f"{a.out}/block_edges_blade.vtk", V2, B2, cur2, part, geo)
    keep = [i for i, d in enumerate(info) if d["kind"] == "phys"]
    m = np.isin(L, keep)
    tfi = TP.refill_polytron(V2, B2, cur2, a.target_h, f"{a.out}/mesh_full.vtk", write_edges=False,
                             surface=(P, T[m], L[m]), param_keys=param, radial_counts=radial)
    rep["tfi"] = {k: v for k, v in tfi.items() if isinstance(v, (int, float, bool, str))}
    mm = meshio.read(f"{a.out}/mesh_full.vtk"); H = mm.cells[0].data; bid = mm.cell_data["block_id"][0].ravel()
    s = np.nan_to_num(diag.cell_sj(mm.points[H]), nan=-1.0); pc = part[bid]
    rep["parts"] = {int(p): {"cells": int((pc == p).sum()), "inverted": int((s[pc == p] <= 0).sum()),
                             "sj_p1": float(np.percentile(s[pc == p], 1))} for p in np.unique(pc)}
    meshio.Mesh(mm.points, [("hexahedron", H)], cell_data={"block_id": [bid], "part": [pc], "sj": [s],
                "inverted": [(s <= 0).astype(np.int32)]}).write(f"{a.out}/mesh_full.vtk", binary=True)
    print(f"[blocking] tfi cells {len(H)} inverted {(s <= 0).sum()} | " + " ".join(
        f"part {p}: {v['cells']} inv {v['inverted']} sj_p1 {v['sj_p1']:.2f}" for p, v in rep["parts"].items()), flush=True)
    json.dump(rep, open(f"{a.out}/report.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
