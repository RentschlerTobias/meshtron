#!/usr/bin/env python3
"""Cubic edge curves fitted to the own tet frame field (frame_field_tet.py) instead of the AlgoHex polyline.

Per non-feature block edge (feature edges keep their exact seam arcs from curve_postprocess): endpoints fixed (the
post-processed corners), the two inner control points minimise
    alignment   mean over the curve of 1 - (sum_i (t . a_i)^4 - 1/3) / (2/3)    (0 = tangent on a frame axis;
                octahedral-invariant, frame a_i of the nearest tet vertex, held fixed per step)
  + lam_data  * mean squared distance to the AlgoHex polyline / chord^2          (keeps the AlgoHex structure)
  + lam_surf  * mean squared distance to the surface / chord^2                  (boundary edges only)
Evaluated with the real fill (curve_opt.mesh_eval) on the GT block structure against the exported cubics and the
post-processed targets.

    PYTHONPATH=<dp3d hex3d_algohex> python fit_field_cubics.py --name tistos_s1_0131 --out viz/tistos_s1_0131_field
"""
import argparse
import json
import os
import sys

sys.path.insert(0, "/opt/stack/meshtron")
sys.path.insert(0, "/opt/stack/meshtron/experiments/flow_topology")
sys.path.insert(0, "/opt/stack/domain_partition_3D/experimentell/hex3d_algohex")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from infer_chain_v2 import snap_all  # noqa: E402,F401  (import order)
import curve_opt as CO  # noqa: E402
import curve_postprocess as cp  # noqa: E402
from frame_field_tet import TetFrameField  # noqa: E402

RUNS = "/opt/stack/meshtron/experiments/flow_topology/datagen/runs"


def bez_t(P0, B1, B2, P1, t):
    t = t[None, :, None]
    return ((1 - t) ** 3 * P0[:, None] + 3 * (1 - t) ** 2 * t * B1[:, None] + 3 * (1 - t) * t ** 2 * B2[:, None]
            + t ** 3 * P1[:, None])


def fit(ff, V, E, c1, c2, polys, bmask, surf_tree, S, lam_data=1.0, lam_surf=10.0, iters=300, lr=3e-3, n=33):
    dt = torch.float64
    P0, P1 = torch.tensor(V[E[:, 0]], dtype=dt), torch.tensor(V[E[:, 1]], dtype=dt)
    L = (P1 - P0).norm(dim=1)
    B1 = torch.tensor(c1, dtype=dt, requires_grad=True); B2 = torch.tensor(c2, dtype=dt, requires_grad=True)
    t = torch.linspace(0, 1, n, dtype=dt)
    Rall = torch.tensor(ff.R, dtype=dt)
    polyt = [torch.tensor(p, dtype=dt) for p in polys]
    bm = torch.tensor(bmask)
    opt = torch.optim.Adam([B1, B2], lr=lr * float(L.mean()))
    log = []
    for it in range(iters + 1):
        C = bez_t(P0, B1, B2, P1, t)                                    # [E,n,3]
        with torch.no_grad():
            mid = 0.5 * (C[:, 1:] + C[:, :-1])
            idx = torch.as_tensor(ff.tree.query(mid.reshape(-1, 3).numpy())[1]).reshape(mid.shape[:2])
            Rm = Rall[idx]                                              # [E,n-1,3,3]
        tg = C[:, 1:] - C[:, :-1]; tg = tg / tg.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        c = torch.einsum("eni,enij->enj", tg, Rm)
        align = (1 - ((c ** 4).sum(-1) - 1 / 3) / (2 / 3)).mean(1)    # [E]
        dd = []
        for k, Pp in enumerate(polyt):                                  # curve -> polyline distance
            dd.append(torch.cdist(C[k], Pp).min(1).values.pow(2).mean() / L[k] ** 2)
        ld = torch.stack(dd)
        with torch.no_grad():
            sp = torch.tensor(S[surf_tree.query(C.reshape(-1, 3).numpy())[1]], dtype=dt).reshape(C.shape)
        ls = ((C - sp).norm(dim=-1).pow(2).mean(1) / L ** 2) * bm
        loss = (align + lam_data * ld + lam_surf * ls).mean()
        if it % 100 == 0 or it == iters:
            log.append({"it": it, "align": float(align.mean()), "data": float(ld.mean()), "surf": float(ls.mean())})
        if it == iters:
            break
        opt.zero_grad(); loss.backward(); opt.step()
    return B1.detach().numpy(), B2.detach().numpy(), log


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lams", default="0.1,1,10", help="lam_data values to try")
    ap.add_argument("--target-h", type=float, default=0.08)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = f"{RUNS}/{a.name}"
    s = cp.__dict__["np"].load(f"{R}/sample.npz")
    import export_sample as es
    smp = es.load(f"{R}/sample.npz")
    post = np.load(f"/opt/stack/meshtron/experiments/flow_topology/curves_post/{a.name}_curves.npz")
    raw = {k: smp[k] for k in ("surface_points", "surface_tris", "surface_tri_label")}
    ff = TetFrameField(f"{R}/tet.vtk", iters=200, verbose=False)
    V0, B = smp["vertices"], smp["blocks"]
    Vp = post["post_vertices"]
    Eu, ctrl_raw, dmap = cp.undirected(smp)
    post_ctrl = post["post_edge_ctrl"]
    featd = post["post_feature_edge"]
    # undirected views of the post data
    pc = np.zeros((len(Eu), 2, 3)); feat = np.zeros(len(Eu), bool); arcs = {}
    ao, ap_ = post["post_arc_offset"], post["post_arc_polyline"]
    for d, (k, rev) in enumerate(dmap):
        c = post_ctrl[d]
        pc[k] = (c[1], c[0]) if rev else (c[0], c[1])
        feat[k] |= bool(featd[d])
        if ao[d + 1] > ao[d] and k not in arcs:
            P = ap_[ao[d]:ao[d + 1]]
            arcs[k] = P[::-1] if rev else P
    poly, off = smp["edge_polyline"], smp["edge_polyline_offset"]
    polys = {}
    for d, (p, q) in enumerate(smp["edges"]):
        k, rev = dmap[d]
        if k not in polys:
            P = poly[off[d]:off[d + 1]]
            polys[k] = P[::-1] if rev else P
    faces, nown = cp.faces_of(B)
    bq = [f for f, n in zip(faces, nown) if n == 1]
    share2 = cp.edge_face_sets(Eu, bq)
    bmask_all = np.array([bool(x) for x in share2])
    free = np.flatnonzero(~feat)
    S = smp["surface_points"]; stree = cKDTree(S)

    def curves_of(c1, c2, V, use_arcs):
        cur = {}
        for k, (p, q) in enumerate(Eu):
            if use_arcs and k in arcs:
                cur[(int(p), int(q))] = arcs[k]
            else:
                cur[(int(p), int(q))] = cp.bezier(V[p], c1[k], c2[k], V[q], 257)
        return cur

    def align_stats(cur):
        m = [ff.misalign(cur[(int(p), int(q))]).mean() for k, (p, q) in enumerate(Eu) if k in set(free)]
        return float(np.median(m)), float(np.percentile(m, 90))

    def sag(cur):
        out = []
        for k in free:
            p, q = Eu[k]; P = cur[(int(p), int(q))]; ch = np.linalg.norm(P[-1] - P[0]); u = (P[-1] - P[0]) / ch
            d = P - P[0]; out.append(np.linalg.norm(d - np.outer(d @ u, u), axis=1).max() / ch)
        return float(np.max(out)), float(np.percentile(out, 90))

    res = {}
    variants = {"export cubics (AlgoHex fit)": curves_of(ctrl_raw[:, 0], ctrl_raw[:, 1], V0, False),
                "post targets (seam arcs + G1)": curves_of(pc[:, 0], pc[:, 1], Vp, True)}
    for lam in [float(x) for x in a.lams.split(",")]:
        c1, c2, log = fit(ff, Vp, Eu[free], pc[free, 0], pc[free, 1], [polys[k] for k in free], bmask_all[free], stree, S,
                          lam_data=lam)
        f1, f2 = pc[:, 0].copy(), pc[:, 1].copy(); f1[free], f2[free] = c1, c2
        variants[f"field fit, lam_data={lam:g}"] = curves_of(f1, f2, Vp, True)
        res[f"log_lam{lam:g}"] = log
    print(f"{'variant':34s} | align free edges med/p90 | sagitta max/p90 | uncovered  inverted  sj_p1  nonortho_max")
    for name, cur in variants.items():
        V = V0 if name.startswith("export") else Vp
        tag = name.split()[0] + name.split("=")[-1] if "=" in name else name.split()[0]
        m = CO.mesh_eval(V, B, cur, raw, a.target_h, f"{a.out}/mesh_{tag.replace('.', 'p')}.vtk")
        al, sg = align_stats(cur), sag(cur)
        res[name] = {"align": al, "sagitta": sg, **{k: m.get(k) for k in ("uncovered", "inv_share", "sj_p1", "nonortho_max")}}
        print(f"{name:34s} | {al[0]:9.1f} / {al[1]:5.1f}      | {sg[0]:5.2f} / {sg[1]:4.2f}   | {100*m['uncovered']:7.2f} % "
              f"{100*m['inv_share']:7.3f} % {m['sj_p1']:6.2f} {m['nonortho_max']:8.1f}")
        CO_write = os.path.join(a.out, f"curves_{tag.replace('.', 'p')}.vtk")
        pts, lines, base = [], [], 0
        for (p, q), P in cur.items():
            pts.append(P); idx = base + np.arange(len(P)); lines.append(np.stack([idx[:-1], idx[1:]], 1)); base += len(P)
        import meshio
        meshio.Mesh(np.concatenate(pts), [("line", np.concatenate(lines))]).write(CO_write, binary=True)
    ff.write_vtk(f"{a.out}/tet_framefield.vtk")
    json.dump(res, open(f"{a.out}/summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
