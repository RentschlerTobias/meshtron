#!/usr/bin/env python3
"""VTK export of every step of the flow -> pointer -> flow chain for one geometry (ParaView + slides).

  00_surface.vtk          labelled input surface (cell data 'label')
  01_pointcloud.vtk       conditioning point cloud
  02_stage1_t{00..32}.vtk stage-1 corner set during the flow (noise at t=0 -> corners at t=1)
  03_stage1_snapped.vtk   corners after the block-free snap (point data 'role': patches touched)
  04_blocks.vtk           pointer blocks, straight edges (hexahedra, cell data 'block')
  05_curves_stage3.vtk    stage-3 Bezier edges (+ refined corners), cell data 'feature'
  06_curves_arcs.vtk      feature edges replaced by the exact seam arcs, cell data 'kind'
  07_mesh_tfi.vtk         curved TFI mesh of 06 (refill_polytron, projection on)
  08_mesh_opt.vtk         TFI mesh after the curve optimisation (curve_opt.optimise, hinge)
  09_gt_blocks.vtk / 10_gt_mesh.vtk   AlgoHex ground truth for comparison
  summary.json            metrics per step

    PYTHONPATH=<dp3d hex3d_algohex> python viz_pipeline.py --name tistos_s1_0131 --out viz/tistos_s1_0131
"""
import argparse
import json
import os
import sys

sys.path.insert(0, "/opt/stack/meshtron")
sys.path.insert(0, "/work/meshtron_v2")
sys.path.insert(0, "/work/rl_reward")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from infer_chain_v2 import snap_all  # noqa: E402,F401  (import order: before meshtron.geometry)
import continuity as C  # noqa: E402
import curve_opt as CO  # noqa: E402
import flow_geo_var as FV  # noqa: E402
import flow_vertex as FVX  # noqa: E402
import meshio  # noqa: E402
import pointer_conn as PC  # noqa: E402
import pointer_conn2 as P2  # noqa: E402
from build_flow_dataset import surface_multihot  # noqa: E402
from meshtron.data.polytron_blocks import bezier, offsets_to_ctrl  # noqa: E402
from meshtron.geometry.geometry_features import FeatureModelV2  # noqa: E402
from meshtron.geometry.polytron_tfi import SurfaceProjector  # noqa: E402
from train import batch_points  # noqa: E402

HEX_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))


def hex_edges(B):
    return np.array(sorted({tuple(sorted((int(b[i]), int(b[j])))) for b in B for i, j in HEX_EDGES}), np.int64)


def write_points(path, P, **pdata):
    meshio.Mesh(P, [("vertex", np.arange(len(P))[:, None])], point_data=pdata).write(path, binary=True)


def write_curves(path, V, E, c1, c2, arcs=None, cdata=None, n=17):
    pts, lines, base, cd = [V], [], len(V), {k: [] for k in (cdata or {})}
    for k, (a, b) in enumerate(E):
        P = arcs[k] if arcs is not None and k in arcs else bezier(V[a], V[b], c1[k], c2[k], n)
        pts.append(P)
        idx = base + np.arange(len(P)); base += len(P)
        seg = np.stack([idx[:-1], idx[1:]], 1); lines.append(seg)
        for key, vals in (cdata or {}).items():
            cd[key] += [vals[k]] * len(seg)
    L = np.concatenate(lines)
    meshio.Mesh(np.concatenate(pts), [("vertex", np.arange(len(V))[:, None]), ("line", L)],
                cell_data={k: [np.full(len(V), -1), np.asarray(v)] for k, v in cd.items()}).write(path, binary=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="val_full")
    ap.add_argument("--data", default="/work/meshtron_v2/data/flow_canon_full.pt")
    ap.add_argument("--post-data", default="/work/meshtron_v2/data/flow_canon_full_post.pt")
    ap.add_argument("--vtx-ckpt", default="/work/meshtron_v2/runs/vtx_full_s0/best.pt")
    ap.add_argument("--ptr-ckpt", default="/work/meshtron_v2/runs/ptr_s0/last.pt")
    ap.add_argument("--var-ckpt", default="/work/meshtron_v2/runs/var_post_s0/last.pt")
    ap.add_argument("--target-h", type=float, default=0.08)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    os.makedirs(a.out, exist_ok=True)
    data = torch.load(a.data, weights_only=False)
    it = next(x for x in data[a.split] if x["name"] == a.name)
    npz = P2.npz_of(a.name)
    raw = {k: np.asarray(v) for k, v in np.load(npz, allow_pickle=False).items()}
    fm = FeatureModelV2(npz, cache_dir=P2.FEAT_CACHE)
    S, T, L = raw["surface_points"], raw["surface_tris"], raw["surface_tri_label"]
    summ = {"name": a.name}
    gen = torch.Generator(device=dev); gen.manual_seed(1234)

    # 00 / 01 input
    meshio.Mesh(S, [("triangle", T)], cell_data={"label": [L]}).write(f"{a.out}/00_surface.vtk", binary=True)
    pts, pmask = batch_points([it], dev)
    pf = it["points"].float().numpy()
    write_points(f"{a.out}/01_pointcloud.vtk", S, label=np.argmax(pf[:, 7:14], 1) + 1)

    # 02 stage 1 during the flow
    vm, vn, va = FVX.load(a.vtx_ckpt, data)
    with torch.no_grad():
        zv = vm.enc(pts, pmask)
        n = int(vm.count_logits(zv).argmax(-1)[0])
        vmask = torch.ones(1, n, dtype=torch.bool, device=dev)
        xv = torch.randn(1, n, 3, device=dev, generator=gen)
        steps = 32
        for i in range(steps + 1):
            if i % 8 == 0:
                write_points(f"{a.out}/02_stage1_t{i:02d}.vtk", vn.dec_v(xv)[0].cpu().double().numpy())
            if i == steps:
                break
            xv = xv + vm(xv, torch.full((1,), i / steps, device=dev), zv, vmask) / steps
    P = vn.dec_v(xv)[0].cpu().double().numpy()
    summ["n_corners_pred"] = n
    summ["n_corners_gt"] = len(it["verts"])

    # 03 snap
    Ps = P2.snap_free(P, fm)
    tol = 6e-3 * np.linalg.norm(np.ptp(S, 0))
    lab = surface_multihot(Ps, raw, tol)
    write_points(f"{a.out}/03_stage1_snapped.vtk", Ps, role=lab.sum(1).astype(np.int64),
                 moved=np.linalg.norm(Ps - P, axis=1))

    # 04 pointer blocks
    ck = torch.load(a.ptr_ckpt, weights_only=False); pa = ck["args"]
    nrm = FV.Norm(data, data["train"]).to(dev)
    labelled = "labels" in pa or "slide" in pa
    ptr = (P2.SetPointer2 if labelled else PC.SetPointer)(pa["d"], pa["layers"], pa["heads"], ck["max_len"]).to(dev)
    ptr.load_state_dict(ck["model"]); ptr.eval()
    with torch.no_grad():
        x = nrm.enc_v(torch.tensor(Ps, dtype=torch.float32, device=dev))[None]
        vm1 = torch.ones(1, len(Ps), dtype=torch.bool, device=dev)
        if labelled:
            lt = torch.tensor(lab, device=dev)[None]
            if pa.get("labels") == "none":
                lt = torch.zeros_like(lt)
            e, z = ptr.memory(x, vm1, pts, pmask, lt)
        else:
            e, z = ptr.memory(x, vm1, pts, pmask)
        blocks = PC.decode(ptr, e, z, vm1)
    B = np.asarray(blocks, np.int64)
    used = sorted({int(c) for b in B for c in b})
    remap = {c: i for i, c in enumerate(used)}
    V = Ps[used]; B = np.vectorize(remap.get)(B)
    lab = lab[used]
    meshio.Mesh(V, [("hexahedron", B)], cell_data={"block": [np.arange(len(B))]}).write(f"{a.out}/04_blocks.vtk", binary=True)
    import networkx as nx
    from networkx.algorithms.isomorphism import categorical_node_match
    from eval_ptr_iso import complex_graph
    summ["n_blocks"] = len(B)
    summ["topology_matches_gt"] = bool(len(B) == len(it["conn"]) and nx.is_isomorphic(
        complex_graph(B), complex_graph(it["conn"].long().numpy()), node_match=categorical_node_match("kind", None)))

    # 05 stage 3 curves (+ corner refinement)
    pdata = torch.load(a.post_data, weights_only=False)
    var, vnrm, vargs = FV.load(a.var_ckpt, pdata)
    E = hex_edges(B)
    mid = (V[E[:, 0]] + V[E[:, 1]]) / 2
    item = {"verts": torch.tensor(V, dtype=torch.float32), "conn": torch.tensor(B), "edges": torch.tensor(E),
            "vlab": torch.tensor(lab), "elab": torch.tensor(surface_multihot(mid, raw, tol)),
            "eoff": torch.zeros(len(E), 6), "points": it["points"]}
    b = FV.collate([item], dev)
    with torch.no_grad():
        zz = var.enc(pts, pmask)
        xv3, xo3 = FV.sample(var, zz, b, vnrm.enc_v(torch.tensor(V, dtype=torch.float32, device=dev))[None], vargs["steps"], gen)
    V3 = vnrm.dec_v(xv3)[0].cpu().double().numpy(); O3 = vnrm.dec_o(xo3)[0].cpu().double().numpy()
    V3 = P2.snap_free(V3, fm)
    c1, c2 = offsets_to_ctrl(V3[E[:, 0]], V3[E[:, 1]], O3)
    vpatch = [set((np.flatnonzero(l > 0.5) + 1).tolist()) for l in lab]
    det = C.detect(V3, B, E, (c1, c2), fm, vpatch)
    feat = [int(k in det["feat"]) for k in range(len(E))]
    write_curves(f"{a.out}/05_curves_stage3.vtk", V3, E, c1, c2, cdata={"feature": feat})

    # 06 exact seam arcs for the feature edges
    seams = CO.SmoothSeams(fm.seam_curves)
    V6, arcs = CO.feature_arcs(V3, E, c1, c2, seams, det)
    kind = [2 if k in arcs else (1 if feat[k] else 0) for k in range(len(E))]
    write_curves(f"{a.out}/06_curves_arcs.vtk", V6, E, c1, c2, arcs=arcs, cdata={"kind": kind})
    cur = CO.to_curves(V6, E, c1, c2, arcs)
    summ["kinks_stage3"] = {k: float(np.median(v)) for k, v in CO.pair_kinks(CO.to_curves(V3, E, c1, c2, {}), E, det).items() if v}
    summ["kinks_arcs"] = {k: float(np.median(v)) for k, v in CO.pair_kinks(cur, E, det).items() if v}

    # 07 TFI, 08 TFI after the curve optimisation
    summ["mesh_tfi"] = CO.mesh_eval(V6, B, cur, raw, a.target_h, f"{a.out}/07_mesh_tfi.vtk")
    proj = SurfaceProjector(S, T, L)
    cnt, per = C.block_faces(B)
    uf = list({tuple(sorted(f)): f for fs in per for f in fs}.values())
    bq = [f for f in uf if cnt[tuple(sorted(f))] == 1]
    qlab = C.quad_patch(bq, V6, E, (c1, c2), fm, vpatch)
    share2 = C.edge_face_sets(E, bq)
    epat = [set(int(qlab[q]) for q in share2[k]) for k in range(len(E))]
    o1, o2, log = CO.optimise(V6, B, E, c1, c2, arcs, det, proj, epat)
    cur8 = CO.to_curves(V6, E, o1, o2, arcs)
    summ["mesh_opt"] = CO.mesh_eval(V6, B, cur8, raw, a.target_h, f"{a.out}/08_mesh_opt.vtk")
    write_curves(f"{a.out}/08_curves_opt.vtk", V6, E, o1, o2, arcs=arcs, cdata={"kind": kind})

    # 09 / 10 ground truth
    Vg, Bg = raw["vertices"], raw["blocks"]
    Eg, ctrl, _ = None, None, None
    meshio.Mesh(Vg, [("hexahedron", Bg)], cell_data={"block": [np.arange(len(Bg))]}).write(f"{a.out}/09_gt_blocks.vtk", binary=True)
    curg = {}
    for (p, q), (g1, g2) in zip(raw["edges"], raw["edge_ctrl"]):
        if p < q:
            curg[(int(p), int(q))] = bezier(Vg[p], Vg[q], g1, g2, 257)
    summ["mesh_gt"] = CO.mesh_eval(Vg, Bg, curg, raw, a.target_h, f"{a.out}/10_gt_mesh.vtk")
    json.dump(summ, open(f"{a.out}/summary.json", "w"), indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main()
