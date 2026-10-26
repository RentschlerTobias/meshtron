#!/usr/bin/env python3
"""Round 2 of the field fit (fit_field_cubics.py made the meshes worse): field alignment as an extra term of the
TFI curve optimisation (curve_opt.CurveProblem: scaled-Jacobian hinge + surface + data), with
  - ONE axis per edge: chosen at the curve midpoint (frame axis closest to the chord), carried to every sample
    point by matching that point's frame to the midpoint frame with the 24 cube symmetries -> no axis switching;
  - no field term near the block corners (20 % of the curve at each end) and where the tet field is not smooth
    (vertex misfit > --misfit-deg, i.e. near singularities);
  - the surface term of CurveProblem (boundary edges onto their own patch, re-projected).
Feature edges stay the exact seam arcs. Compared on the GT block structure of one sample with the exported cubics,
the post-processed targets and the hinge optimisation without field term.

    PYTHONPATH=<dp3d hex3d_algohex> python fit_field_cubics2.py --name tistos_s1_0131 --out viz/tistos_s1_0131_field2
"""
import argparse
import json
import os
import sys

sys.path.insert(0, "/opt/stack/meshtron")
sys.path.insert(0, "/opt/stack/meshtron/experiments/flow_topology")
sys.path.insert(0, "/opt/stack/domain_partition_3D/experimentell/hex3d_algohex")

import meshio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from infer_chain_v2 import snap_all  # noqa: E402,F401  (import order)
import continuity as C  # noqa: E402
import curve_opt as CO  # noqa: E402
import curve_postprocess as cp  # noqa: E402
import export_sample as es  # noqa: E402
from build_flow_dataset import surface_multihot  # noqa: E402
from frame_field_tet import TetFrameField, cube_group  # noqa: E402
from meshtron.geometry.geometry_features import FeatureModelV2  # noqa: E402
from meshtron.geometry.polytron_tfi import SurfaceProjector  # noqa: E402

RUNS = "/opt/stack/meshtron/experiments/flow_topology/datagen/runs"


def field_axes(ff, C_, G, misfit_deg):
    """Per sample point of every curve the field axis that continues the midpoint axis; weights [E,n-1]."""
    E, n = C_.shape[:2]
    mid = 0.5 * (C_[:, 1:] + C_[:, :-1])
    idx = ff.tree.query(mid.reshape(-1, 3))[1].reshape(E, n - 1)
    R = ff.R[idx]                                                  # [E,n-1,3,3]
    m = (n - 1) // 2
    chord = C_[:, -1] - C_[:, 0]; chord /= np.linalg.norm(chord, axis=1, keepdims=True)
    Rref = R[:, m]
    j = np.abs(np.einsum("ei,eij->ej", chord, Rref)).argmax(1)    # axis at the midpoint
    M = np.einsum("eki,enkj->enij", Rref, R)                       # Rref^T R_s
    g = np.einsum("gab,enab->eng", G, M).argmax(-1)                # best symmetry per sample
    Ra = np.einsum("enab,enbc->enac", R, np.transpose(G[g], (0, 1, 3, 2)))   # R_s aligned to Rref
    A = Ra[np.arange(E)[:, None], np.arange(n - 1)[None], :, j[:, None]]     # [E,n-1,3]
    s = (np.arange(n - 1) + 0.5) / (n - 1)
    w = np.clip(np.minimum(s, 1 - s) / 0.2, 0, 1)[None].repeat(E, 0)
    w = w * (ff.vertex_misfit_deg[idx] <= misfit_deg)
    return A, w


def run(pr, ff, w, G, misfit_deg, iters=300, lr=2e-3, reproj=25, n=33):
    """Adam on CurveProblem variables: hinge + surface + data (+ field)."""
    opt = torch.optim.Adam([pr.x1, pr.x2], lr=lr * float(pr.L.mean()))
    free = pr.free
    tt = torch.linspace(0, 1, n, dtype=torch.float64)
    log = []
    for it in range(iters + 1):
        if it % reproj == 0:
            pr.reproject()
        X, dense, c1t, c2t = pr.mesh()
        sj = CO.cell_sj(X).min(1).values
        lq = (torch.relu(w["tau"] - sj) ** 2).mean() + 10 * (torch.relu(-sj) ** 2).mean()
        ls, lb, ld = pr.side_terms(dense, c1t, c2t)
        lf = torch.zeros((), dtype=torch.float64)
        if w["field"] > 0:
            Cf = CO.bezier_t(pr.Vt, pr.Et, c1t, c2t, n)[free]          # [Ef,n,3]
            if it % reproj == 0:
                A, wt = field_axes(ff, Cf.detach().numpy(), G, misfit_deg)
                A, wt = torch.tensor(A), torch.tensor(wt)
            tg = Cf[:, 1:] - Cf[:, :-1]; tg = tg / tg.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            lf = ((1 - (tg * A).sum(-1) ** 2) * wt).sum() / wt.sum().clamp(min=1)
        loss = w["quality"] * lq + w["surface"] * ls + w["data"] * ld + w["field"] * lf
        if it % 100 == 0 or it == iters:
            log.append({"it": it, "q": float(lq.detach()), "s": float(ls.detach()), "d": float(ld.detach()),
                        "f": float(lf.detach()), "sj_min": float(sj.min().detach())})
        if it == iters:
            break
        opt.zero_grad(); loss.backward(); opt.step()
    return (*pr.result(), log)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fields", default="0,0.3,1,3", help="field weights to try (0 = hinge only)")
    ap.add_argument("--misfit-deg", type=float, default=15.0)
    ap.add_argument("--target-h", type=float, default=0.08)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = f"{RUNS}/{a.name}"
    smp = es.load(f"{R}/sample.npz")
    post = np.load(f"/opt/stack/meshtron/experiments/flow_topology/curves_post/{a.name}_curves.npz")
    raw = {k: smp[k] for k in ("surface_points", "surface_tris", "surface_tri_label")}
    ff = TetFrameField(f"{R}/tet.vtk", iters=200, verbose=False)
    G = cube_group()
    B = smp["blocks"]; V0 = smp["vertices"]; Vp = post["post_vertices"]
    Eu, ctrl_raw, dmap = cp.undirected(smp)
    pc = np.zeros((len(Eu), 2, 3)); arcs = {}
    ao, apl = post["post_arc_offset"], post["post_arc_polyline"]
    for d, (k, rev) in enumerate(dmap):
        c = post["post_edge_ctrl"][d]
        pc[k] = (c[1], c[0]) if rev else (c[0], c[1])
        if ao[d + 1] > ao[d] and k not in arcs:
            P = apl[ao[d]:ao[d + 1]]; arcs[k] = P[::-1] if rev else P
    S = smp["surface_points"]; tol = 6e-3 * np.linalg.norm(np.ptp(S, 0))
    vpatch = [set((np.flatnonzero(l) + 1).tolist()) for l in surface_multihot(Vp, raw, tol)]
    fm = FeatureModelV2(f"{R}/sample.npz", cache_dir="/opt/stack/meshtron/experiments/flow_topology/cache/features_v2")
    det = C.detect(Vp, B, Eu, (pc[:, 0], pc[:, 1]), fm, vpatch)
    proj = SurfaceProjector(S, smp["surface_tris"], smp["surface_tri_label"])
    cnt, per = C.block_faces(B)
    uf = list({tuple(sorted(f)): f for fs in per for f in fs}.values())
    bq = [f for f in uf if cnt[tuple(sorted(f))] == 1]
    qlab = C.quad_patch(bq, Vp, Eu, (pc[:, 0], pc[:, 1]), fm, vpatch)
    share2 = C.edge_face_sets(Eu, bq)
    epat = [set(int(qlab[q]) for q in share2[k]) for k in range(len(Eu))]

    def curves(c1, c2, V, use_arcs=True):
        return {(int(p), int(q)): (arcs[k] if use_arcs and k in arcs else cp.bezier(V[p], c1[k], c2[k], V[q], 257))
                for k, (p, q) in enumerate(Eu)}

    free = [k for k in range(len(Eu)) if k not in arcs]

    def stats(cur):
        al = [ff.misalign(cur[(int(Eu[k, 0]), int(Eu[k, 1]))]).mean() for k in free]
        sg = []
        for k in free:
            P = cur[(int(Eu[k, 0]), int(Eu[k, 1]))]; ch = np.linalg.norm(P[-1] - P[0]); u = (P[-1] - P[0]) / ch
            dd = P - P[0]; sg.append(np.linalg.norm(dd - np.outer(dd @ u, u), axis=1).max() / ch)
        return float(np.median(al)), float(np.percentile(al, 90)), float(np.max(sg)), float(np.percentile(sg, 90))

    variants = [("export cubics (AlgoHex fit)", curves(ctrl_raw[:, 0], ctrl_raw[:, 1], V0, False), V0),
                ("post targets (seam arcs + G1)", curves(pc[:, 0], pc[:, 1], Vp), Vp)]
    logs = {}
    for wf in [float(x) for x in a.fields.split(",")]:
        pr = CO.CurveProblem(Vp, B, Eu, pc[:, 0], pc[:, 1], arcs, det, proj, epat)
        w = {"quality": 1.0, "surface": 10.0, "bending": 0.0, "data": 0.1, "tau": 0.6, "field": wf}
        c1, c2, log = run(pr, ff, w, G, a.misfit_deg)
        logs[wf] = log
        variants.append((f"hinge opt + field {wf:g}" if wf else "hinge opt (no field)", curves(c1, c2, Vp), Vp))
    res = {}
    print(f"{'variant':32s} | align free med/p90 | sagitta max/p90 | uncovered  inverted  sj_p1  nonortho_max")
    for name, cur, V in variants:
        tag = name.split("(")[0].strip().replace(" ", "_").replace("+", "").replace(".", "p")
        m = CO.mesh_eval(V, B, cur, raw, a.target_h, f"{a.out}/mesh_{tag}.vtk")
        st = stats(cur)
        res[name] = {"align_med": st[0], "align_p90": st[1], "sag_max": st[2], "sag_p90": st[3],
                     **{k: m.get(k) for k in ("uncovered", "inv_share", "sj_p1", "nonortho_max")}}
        print(f"{name:32s} | {st[0]:7.1f} / {st[1]:5.1f}   | {st[2]:5.2f} / {st[3]:4.2f}   | {100*m['uncovered']:7.2f} % "
              f"{100*m['inv_share']:7.3f} % {m['sj_p1']:6.2f} {m['nonortho_max']:8.1f}")
        pts, lines, base = [], [], 0
        for P in cur.values():
            pts.append(P); idx = base + np.arange(len(P)); lines.append(np.stack([idx[:-1], idx[1:]], 1)); base += len(P)
        meshio.Mesh(np.concatenate(pts), [("line", np.concatenate(lines))]).write(f"{a.out}/curves_{tag}.vtk", binary=True)
    json.dump({"results": res, "logs": {str(k): v for k, v in logs.items()}}, open(f"{a.out}/summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
