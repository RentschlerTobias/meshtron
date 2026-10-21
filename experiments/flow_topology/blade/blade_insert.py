#!/usr/bin/env python3
"""Close the reduced (core) block structure to the full domain BEFORE the TFI, at block level.

The core domain is bounded by real boundary patches (inlet, outlet, periodic, ...) and by CUT surfaces where the
dtOO O-grid and boundary layers were removed. The dtOO volume cells are discarded; only its boundary surface
(walls, inlet, outlet, periodic, from case/mesh.msh) is used as geometry. Every core block face on a cut surface is
extruded to the wall behind it as one new block:
  - cut vs. real: a core patch whose triangles lie on the full-domain boundary is real, otherwise it is a cut;
    its wall is the full-domain patch closest to it (median distance) -- no label numbers;
  - order: cuts that touch fewer real patches first (the blade ring before the hub / shroud layers, whose faces
    then include the side faces of the ring blocks);
  - corners and edge points are projected onto the wall, on seams with a real patch onto the wall AND that patch;
    radial edges are straight from the cut corner to its wall point; neighbouring faces share corners and edges,
    so the new blocks are conforming;
  - one joint curved TFI (meshtron polytron_tfi.refill_polytron) over core + new blocks, projected onto the full
    boundary.

    python blade_insert.py --name tistos_s1_0131 --out out/tistos_s1_0131
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, "/opt/stack/meshtron")
sys.path.insert(0, "/work/meshtron_v2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from infer_chain_v2 import snap_all  # noqa: E402,F401  (import order)
import curve_opt as CO  # noqa: E402
import infer_chain_m4 as M4  # noqa: E402
import msh_walls as MW  # noqa: E402
from meshtron.geometry.edge_curves import face_cycle  # noqa: E402
from meshtron.geometry.polytron_tfi import SurfaceProjector, _arclength, _boundary_labels  # noqa: E402
from tfi_param import refill_polytron  # noqa: E402

cp = M4.cp
RUNS = "/work/datagen_full/datagen/runs"
TETS = M4.TETS


def block_volume(X):
    return sum(np.dot(np.cross(X[b] - X[a], X[c] - X[a]), X[d] - X[a]) / 6 for a, b, c, d in TETS)


def gt_post_structure(name):
    """GT block structure with the post-processed curves (seam arcs + G1 cubics): V, B, curves {(a,b): poly}."""
    smp = np.load(f"{RUNS}/{name}/sample.npz", allow_pickle=True)
    post = np.load(f"/work/curves_post/{name}_curves.npz")
    B = smp["blocks"]; V = post["post_vertices"]
    Eu, _, dmap = cp.undirected({k: smp[k] for k in smp.files})
    pc = np.zeros((len(Eu), 2, 3)); arcs = {}
    ao, apl = post["post_arc_offset"], post["post_arc_polyline"]
    for d, (k, rev) in enumerate(dmap):
        c = post["post_edge_ctrl"][d]
        pc[k] = (c[1], c[0]) if rev else (c[0], c[1])
        if ao[d + 1] > ao[d] and k not in arcs:
            P = apl[ao[d]:ao[d + 1]]; arcs[k] = P[::-1] if rev else P
    cur = CO.to_curves(V, Eu, pc[:, 0], pc[:, 1], arcs)
    return V, B, {(min(a, b), max(a, b)): (Q if a < b else Q[::-1]) for (a, b), Q in cur.items()}


def classify_core_patches(raw, full_proj, tol):
    """core label -> ('real', full patch) or ('cut', wall patch)."""
    S, T, L = raw["surface_points"], raw["surface_tris"], raw["surface_tri_label"]
    out = {}
    for l in np.unique(L):
        c = S[T[L == l]].mean(1)
        d = np.median(full_proj.dists(c), 0)
        j = int(np.argmin(d))
        out[int(l)] = ("real" if d[j] < tol else "cut", full_proj.labels[j], float(d[j]))
    return out


def close_domain(V, B, curves, raw, full, verbose=True):
    """Extrude every core block face on a cut surface to its wall. Returns V', B', curves', part [B'] (0 core,
    k = k-th cut), report."""
    S, T, L = raw["surface_points"], raw["surface_tris"], raw["surface_tri_label"]
    diag = float(np.linalg.norm(np.ptp(S, 0)))
    core_proj = SurfaceProjector(S, T, L)
    full_proj = SurfaceProjector(*full)
    cls = classify_core_patches(raw, full_proj, 2e-3 * diag)
    face_lab, cyc_of = _boundary_labels(V, B, curves, core_proj)
    # boundary faces: key -> (cycle, label); labels of cut faces stay core labels, real faces get full labels
    faces = {k: (cyc_of[k], lab) for k, lab in face_lab.items() if lab is not None}
    cuts = [l for l, c in cls.items() if c[0] == "cut"]
    real_full = {l: c[1] for l, c in cls.items() if c[0] == "real"}

    def neighbours(lab):
        """labels of boundary faces sharing an edge with faces of label lab."""
        e2l = defaultdict(set)
        for cyc, l in faces.values():
            for a, b in zip(cyc, cyc[1:] + cyc[:1]):
                e2l[(min(a, b), max(a, b))].add(l)
        return {m for ls in e2l.values() if lab in ls for m in ls} - {lab}
    order = sorted(cuts, key=lambda l: len([m for m in neighbours(l) if m in real_full]))
    V = [np.asarray(v, float) for v in V]; B = [list(map(int, b)) for b in B]; part = [0] * len(B)
    curves = dict(curves); param = set()
    for w in {c_[1] for c_ in cls.values()}:
        real_full[f"wall{w}"] = w
    rep = {"classes": {str(k): v for k, v in cls.items()}, "order": order, "cuts": []}
    for ci, c in enumerate(order, 1):
        wall = cls[c][1]
        F = [k for k, (cyc, l) in faces.items() if l == c]
        # seam labels of corners / edges with real patches (full labels)
        e2l = defaultdict(set); v2l = defaultdict(set)
        for cyc, l in faces.values():
            for a, b in zip(cyc, cyc[1:] + cyc[:1]):
                e2l[(min(a, b), max(a, b))].add(l); v2l[a].add(l)

        def full_labs(ls):
            return sorted({real_full[m] for m in ls if m in real_full})
        wmap, moved = {}, []
        corners = sorted({v for k in F for v in faces[k][0]})
        for v in corners:
            labs = sorted({wall} | set(full_labs(v2l[v] - {c})))
            q = full_proj.onto(V[v], labs)[0]
            wmap[v] = len(V); V.append(q); moved.append(float(np.linalg.norm(q - V[v])))
            curves[(v, wmap[v])] = np.linspace(V[v], q, 33)                      # radial edge (v < new id)
            param.add((v, wmap[v]))
        for k in F:
            cyc = faces[k][0]
            for a, b in zip(cyc, cyc[1:] + cyc[:1]):
                key = (min(a, b), max(a, b)); wk = (min(wmap[a], wmap[b]), max(wmap[a], wmap[b]))
                if wk in curves:
                    continue
                Q = curves.get(key)
                Q = np.linspace(V[key[0]], V[key[1]], 33) if Q is None else np.asarray(Q, float)
                if key not in param:                 # the fill samples this edge by arc length: so do we
                    Q = _arclength(Q, 129)
                labs = sorted({wall} | set(full_labs(e2l[key] - {c})))
                W = full_proj.onto(Q, labs)
                W[0], W[-1] = V[wmap[key[0]]], V[wmap[key[1]]]
                curves[wk] = W if wmap[key[0]] < wmap[key[1]] else W[::-1]
                param.add(wk)                        # point i of the wall edge = projection of point i
        n_new = 0
        for k in F:
            cyc = faces[k][0]
            blk = list(cyc) + [wmap[v] for v in cyc]
            if block_volume(np.asarray([V[i] for i in blk])) < 0:
                blk = list(cyc[::-1]) + [wmap[v] for v in cyc[::-1]]
            B.append(blk); part.append(ci); n_new += 1
            # side faces of the new block inherit the seam label (another cut -> extruded later, real -> boundary)
            for a, b in zip(cyc, cyc[1:] + cyc[:1]):
                others = e2l[(min(a, b), max(a, b))] - {c}
                if len(others) == 1:
                    side = [a, b, wmap[b], wmap[a]]
                    faces[frozenset(side)] = (side, next(iter(others)))
            del faces[k]
            wq = [wmap[v] for v in cyc]                                  # wall face: real boundary of the new block
            faces[frozenset(wq)] = (wq, f"wall{wall}")
        rep["cuts"].append({"label": c, "wall": wall, "faces": len(F), "new_blocks": n_new,
                            "corner_move_max_rel": max(moved) / diag, "corner_move_med_rel": float(np.median(moved)) / diag})
        if verbose:
            print(f"[close] cut {c} -> wall {wall}: {len(F)} faces -> {n_new} blocks, corner move "
                  f"median {np.median(moved)/diag:.4f} max {max(moved)/diag:.4f} x diag", flush=True)
    return np.asarray(V), np.asarray(B, np.int64), curves, np.asarray(part), rep, frozenset(param)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", default="tistos_s1_0131")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-h", type=float, default=0.05)
    ap.add_argument("--relax-iters", type=int, default=0, help="Laplace relaxation of boundary faces on their patch")
    ap.add_argument("--edge-resample", action="store_true")
    ap.add_argument("--no-param", action="store_true", help="arc-length sampling for all edges (old behaviour)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    raw = dict(np.load(f"{RUNS}/{a.name}/sample.npz", allow_pickle=True))
    P, T, Lf, info = MW.load_boundary(f"{RUNS}/{a.name}/case/mesh.msh")
    V, B, curves = gt_post_structure(a.name)
    V2, B2, cur2, part, rep, param = close_domain(V, B, curves, raw, (P, T, Lf))
    vols = [block_volume(V2[b]) for b in B2]
    print(f"[close] blocks {len(B)} -> {len(B2)}, corners {len(V)} -> {len(V2)}, negative-volume blocks "
          f"{sum(v <= 0 for v in vols)}", flush=True)
    np.savez(f"{a.out}/blocks_full.npz", V=V2, B=B2, part=part,
             curve_keys=np.array(list(cur2.keys())), curve_off=np.cumsum([0] + [len(q) for q in cur2.values()]),
             curve_pts=np.concatenate(list(cur2.values())))
    tfi = refill_polytron(V2, B2, cur2, a.target_h, f"{a.out}/mesh_full.vtk", write_edges=True, surface=(P, T, Lf),
                          relax_iters=a.relax_iters, edge_resample=a.edge_resample,
                          param_keys=frozenset() if a.no_param else param)
    rep["tfi"] = {k: v for k, v in tfi.items() if isinstance(v, (int, float, bool, str))}
    print("[close] tfi", rep["tfi"], flush=True)
    json.dump(rep, open(f"{a.out}/report.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
