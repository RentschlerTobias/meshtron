#!/usr/bin/env python3
"""O-ring as ONE structured tube after the core TFI (conforming by construction).

The core block structure (with post-processed curves) is filled by the curved TFI, projected onto the physical
patches and onto the O-grid cut. The core's boundary quads on the cut form a structured quad tube between hub and
shroud; every cut node gets its partner on the blade through the two tube parametrisations (tubeparam: u = span
0..1, s = around), not by projection: target = blade(u_cut, sigma * s_cut + ds) with orientation sigma and offset
ds fitted once. The ring is n graded layers of hexahedra between the cut nodes and their targets; columns of nodes
on hub / shroud stay on that wall. No ring block corners or edges are defined.

    python oring.py --name tistos_s1_0131 --sample S.npz --post C.npz --msh case/mesh.msh --out out/oring_0131
"""
import argparse
import json
import os
import sys

sys.path.insert(0, "/opt/stack/meshtron")
sys.path.insert(0, "/opt/stack/meshtron/experiments/flow_topology")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import meshio  # noqa: E402
import numpy as np  # noqa: E402

import blade_insert as BI  # noqa: E402
import close_layers as CL  # noqa: E402
import diag  # noqa: E402
import msh_walls as MW  # noqa: E402
import tubeparam as TPm  # noqa: E402
from meshtron.geometry.polytron_tfi import SurfaceProjector, refill_polytron  # noqa: E402


def quads_to_tris(Q):
    Q = np.asarray(Q)
    return np.r_[Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", default="tistos_s1_0131")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--post", required=True)
    ap.add_argument("--msh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-h", type=float, default=0.05)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--growth", type=float, default=1.2)
    ap.add_argument("--opt", type=int, default=0, help="surrogate optimisation of the core blocking (iterations)")
    ap.add_argument("--opt-n", type=int, default=17)
    ap.add_argument("--smooth", type=int, default=30, help="smoothing passes of the nearest-point correction")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    P, T, L, info, rtype = MW.load_regions(a.msh)
    og = rtype.index("hex")
    # the cut = every interface of the O-grid with a region that stays in the domain (core and, with the boundary
    # layer kept, the layers): one label for all of them
    og_if = [i for i, d in enumerate(info) if d["kind"] == "iface" and og in d["regions"]]
    cut = og_if[0]
    L = L.copy(); L[np.isin(L, og_if)] = cut
    phys = [i for i, d in enumerate(info) if d["kind"] == "phys"]
    blade = next(i for i in phys if info[i]["regions"] == (og,) and
                 not any(info[j]["phys"] == info[i]["phys"] and j != i for j in phys))       # wall only owned by O-grid
    # 1-2 core TFI, projected onto the physical patches and the cut
    V, B, curves = BI.gt_post_structure(a.name, a.sample, a.post)
    m = np.isin(L, phys + [cut])
    rep = {}
    if a.opt:                                                 # core corners + curves on the TFI surrogate
        import blocking as BK
        import opt_blocking as OB
        geo = BK.Geo(P, T, L, info)
        V, curves, olog, oinfo = OB.optimise_blocking(V, B, curves, frozenset(), geo, phys + [cut], n=a.opt_n,
                                                      iters=a.opt)
        rep["opt"] = {"log": olog, **oinfo}
        print("[oring] core optimisation", oinfo, "final", olog[-1], flush=True)
    tfi = refill_polytron(V, B, curves, a.target_h, f"{a.out}/core.vtk", write_edges=False, surface=(P, T[m], L[m]))
    rep["core_tfi"] = {k: v for k, v in tfi.items() if isinstance(v, (int, float, bool, str))}
    core = meshio.read(f"{a.out}/core.vtk"); Pm = core.points; Hm = core.cells[0].data
    # 3 cut quads
    Q = np.array(CL.boundary_quads(Hm))
    proj = SurfaceProjector(P, T[m], L[m])
    # a quad is on the cut when all its nodes lie on it (nearest-patch labelling fails next to the blade / hub)
    dn = proj.dists(Pm)[:, proj.labels.index(cut)]
    tol = 1e-3 * float(np.linalg.norm(np.ptp(P, 0)))
    Qc = Q[(dn[Q] < tol).all(1)]
    nodes = np.unique(Qc)
    print(f"[oring] core {len(Hm)} cells, cut quads {len(Qc)}, cut nodes {len(nodes)}", flush=True)
    # 4 tube parametrisations and the node -> blade map
    tb = TPm.TubeParam(P, T[L == blade])
    tc = TPm.TubeParam(Pm, quads_to_tris(Qc))
    used = np.unique(quads_to_tris(Qc))                       # TubeParam's vertex order
    uc = tc.u.copy()
    # orient u: cut loop with u = 0 must face the blade loop with u = 0
    d00 = np.linalg.norm(tc.S[0].mean(0) - tb.S[0].mean(0)); d01 = np.linalg.norm(tc.S[0].mean(0) - tb.S[-1].mean(0))
    if d01 < d00:
        uc = 1.0 - uc
    sc = TPm.params_on_tube(tc, Pm[used], tc.u)
    near = proj.patches[blade].project(Pm[used])[1]
    sb = TPm.params_on_tube(tb, near, uc)
    best = None
    for sg in (1.0, -1.0):
        ang = 2 * np.pi * (sb - sg * sc)
        z = np.exp(1j * ang).mean()
        if best is None or abs(z) > best[0]:
            best = (abs(z), sg, (np.angle(z) / (2 * np.pi)) % 1.0)
    _, sg, ds = best
    print(f"[oring] s orientation {sg:+.0f}, offset {ds:.3f}, alignment |mean| {best[0]:.3f}", flush=True)
    # local correction: towards the nearest-point position (columns along the blade normal), smoothed over the
    # cut grid so neighbouring targets keep their order
    st = (sg * sc + ds) % 1.0
    delta = ((sb - st) + 0.5) % 1.0 - 0.5
    nb = [set() for _ in used]; ix = {int(v): i for i, v in enumerate(used)}
    for q in Qc:
        for i in range(4):
            a_, b_ = ix[int(q[i])], ix[int(q[(i + 1) % 4])]; nb[a_].add(b_); nb[b_].add(a_)
    nb = [np.array(sorted(x)) for x in nb]
    for _ in range(a.smooth):
        delta = 0.5 * delta + 0.5 * np.array([delta[n].mean() for n in nb])
    W = tb.from_param(np.c_[uc, (st + delta) % 1.0])
    W = proj.patches[blade].project(W)[1]
    tgt = {int(v): W[i] for i, v in enumerate(used)}
    # 5 ring layers
    tk = CL.grading(a.layers, a.growth)
    wall_side = {}
    hub_shroud = [i for i in phys if i != blade and info[i]["regions"] == (og,)]   # walls under the O-grid
    for v in nodes:
        d = proj.dists(Pm[v][None])[0]
        cands = [l for l in hub_shroud if d[proj.labels.index(l)] < 1e-6 * 1e3]
        if cands:
            wall_side[int(v)] = min(cands, key=lambda l: d[proj.labels.index(l)])
    smooth = MW.SmoothProjector(P, T, L)
    Pout = [p for p in Pm]; col = {}
    for v in nodes:
        v = int(v); X = (1 - tk[:, None]) * Pm[v] + tk[:, None] * tgt[v]
        if v in wall_side:
            X[1:] = smooth.onto(X[1:], wall_side[v])
        ids = [v]
        for x in X[1:]:
            ids.append(len(Pout)); Pout.append(x)
        col[v] = ids
    H = [list(map(int, h)) for h in Hm]; part = [0] * len(H)
    for q in Qc:
        for j in range(a.layers):
            lo = [col[int(v)][j] for v in q]; hi = [col[int(v)][j + 1] for v in q]
            h = lo + hi
            if BI.block_volume(np.array([Pout[i] for i in h])) < 0:
                h = lo[::-1] + hi[::-1]
            H.append(h); part.append(1)
    Pout = np.array(Pout); H = np.array(H); part = np.array(part)
    s = np.nan_to_num(diag.cell_sj(Pout[H]), nan=-1.0)
    rep["parts"] = {int(p): {"cells": int((part == p).sum()), "inverted": int((s[part == p] <= 0).sum()),
                             "sj_p1": float(np.percentile(s[part == p], 1)), "sj_min": float(s[part == p].min())}
                    for p in (0, 1)}
    rep["blade_dist_max"] = float(proj.patches[blade].project(W)[0].max())
    rep["wall_nodes"] = len(wall_side)
    meshio.Mesh(Pout, [("hexahedron", H)], cell_data={"part": [part], "sj": [s], "inverted": [(s <= 0).astype(np.int32)]}
                ).write(f"{a.out}/mesh_full.vtk", binary=True)
    np.savez(f"{a.out}/ring.npz", cut_quads=Qc, used=used, u=uc, s_cut=sc, targets=W, sigma=sg, ds=ds)
    print("[oring] " + " | ".join(f"part {p}: {v['cells']} cells, inverted {v['inverted']}, sj p1 {v['sj_p1']:.2f}, "
                                  f"min {v['sj_min']:.2f}" for p, v in rep["parts"].items()), flush=True)
    json.dump(rep, open(f"{a.out}/report.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
