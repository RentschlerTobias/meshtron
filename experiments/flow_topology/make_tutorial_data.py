#!/usr/bin/env python3
"""Compact JSON of every chain step for the interactive tutorial page (reads viz_pipeline.py output and re-runs
the two flow stages to record their full trajectories with the same seeds and checkpoints).

    PYTHONPATH=<dp3d hex3d_algohex> python make_tutorial_data.py --name tistos_s1_0131 --viz viz/tistos_s1_0131 \
        --out viz/tutorial_data.json
"""
import argparse
import json
import sys
from collections import Counter

sys.path.insert(0, "/opt/stack/meshtron")
sys.path.insert(0, "/opt/stack/meshtron/experiments/flow_topology")

import meshio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import flow_geo_var as FV  # noqa: E402
import flow_vertex as FVX  # noqa: E402
from meshtron.data.polytron_blocks import bezier, offsets_to_ctrl  # noqa: E402
from train import batch_points  # noqa: E402

HEXF = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
HEX_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))


def r(a, d=4):
    return np.round(np.asarray(a, float), d).tolist()


def lines_of(m):
    """Polylines of a meshio line mesh as lists of point arrays, with their cell data."""
    L = np.concatenate([c.data for c in m.cells if c.type == "line"])
    out, cur, cd = [], [L[0, 0], L[0, 1]], {}
    keys = [k for k in m.cell_data]
    vals = {k: np.concatenate([d for d, c in zip(m.cell_data[k], m.cells) if c.type == "line"]) for k in keys}
    meta = []
    for i in range(1, len(L)):
        if L[i, 0] == cur[-1]:
            cur.append(L[i, 1])
        else:
            out.append(cur); meta.append({k: int(vals[k][i - 1]) for k in keys}); cur = [L[i, 0], L[i, 1]]
    out.append(cur); meta.append({k: int(vals[k][len(L) - 1]) for k in keys})
    return [r(m.points[np.asarray(c)]) for c in out], meta


def boundary_quads(m):
    H = np.concatenate([c.data for c in m.cells if c.type == "hexahedron"])
    cnt = Counter(tuple(sorted(h[list(f)])) for h in H for f in HEXF)
    quads = [tuple(h[list(f)]) for h in H for f in HEXF if cnt[tuple(sorted(h[list(f)]))] == 1]
    used = sorted({v for q in quads for v in q}); remap = {v: i for i, v in enumerate(used)}
    Q = np.array([[remap[v] for v in q] for q in quads])
    return {"pts": r(m.points[used]), "quads": Q.tolist()}


def mesh_edges_on_boundary(m, every=1):
    """Unique boundary quad edges (wireframe) as one flat list with None separators."""
    bq = boundary_quads(m)
    P = np.asarray(bq["pts"]); E = set()
    for q in bq["quads"]:
        for i in range(4):
            E.add(tuple(sorted((q[i], q[(i + 1) % 4]))))
    xs, ys, zs = [], [], []
    for a, b in sorted(E)[::every]:
        xs += [P[a, 0], P[b, 0], None]; ys += [P[a, 1], P[b, 1], None]; zs += [P[a, 2], P[b, 2], None]
    return bq | {"wire": [xs, ys, zs]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--viz", required=True)
    ap.add_argument("--split", default="val_full")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    V = a.viz
    out = {"name": a.name, "summary": json.load(open(f"{V}/summary.json"))}
    s = meshio.read(f"{V}/00_surface.vtk")
    tri = np.concatenate([c.data for c in s.cells if c.type == "triangle"])
    out["surface"] = {"pts": r(s.points), "tris": tri.tolist(),
                      "label": np.concatenate(s.cell_data["label"]).astype(int).tolist()}
    pc = meshio.read(f"{V}/01_pointcloud.vtk")
    sel = np.random.default_rng(0).choice(len(pc.points), min(3000, len(pc.points)), replace=False)
    out["cloud"] = {"pts": r(pc.points[sel]), "label": pc.point_data["label"][sel].astype(int).tolist()}

    # stage-1 trajectory (same seed / order of random draws as viz_pipeline.py)
    data = torch.load("/opt/stack/meshtron/experiments/flow_topology/data/flow_canon_full.pt", weights_only=False)
    it = next(x for x in data[a.split] if x["name"] == a.name)
    vm, vn, _ = FVX.load("/opt/stack/meshtron/experiments/flow_topology/runs/vtx_full_s0/best.pt", data)
    gen = torch.Generator(device=dev); gen.manual_seed(1234)
    pts, pmask = batch_points([it], dev)
    traj = []
    with torch.no_grad():
        zv = vm.enc(pts, pmask)
        n = int(vm.count_logits(zv).argmax(-1)[0])
        probs = torch.softmax(vm.count_logits(zv)[0], -1)
        vmask = torch.ones(1, n, dtype=torch.bool, device=dev)
        xv = torch.randn(1, n, 3, device=dev, generator=gen)
        for i in range(33):
            traj.append(r(vn.dec_v(xv)[0].cpu().numpy()))
            if i == 32:
                break
            xv = xv + vm(xv, torch.full((1,), i / 32, device=dev), zv, vmask) / 32
    top = torch.topk(probs, 5)
    out["stage1"] = {"traj": traj, "count": n, "count_top": [[int(i), float(p)] for p, i in zip(top.values, top.indices)],
                     "gt": r(it["verts"].numpy())}
    sn = meshio.read(f"{V}/03_stage1_snapped.vtk")
    out["snap"] = {"pts": r(sn.points), "role": sn.point_data["role"].astype(int).tolist(),
                   "moved": r(sn.point_data["moved"], 5)}

    b = meshio.read(f"{V}/04_blocks.vtk")
    H = np.concatenate([c.data for c in b.cells if c.type == "hexahedron"])
    out["blocks"] = {"pts": r(b.points), "hex": H.tolist(), "edges": [list(e) for e in HEX_EDGES]}
    g = meshio.read(f"{V}/09_gt_blocks.vtk")
    out["gt_blocks"] = {"pts": r(g.points), "hex": np.concatenate([c.data for c in g.cells if c.type == "hexahedron"]).tolist()}

    # stage-3 trajectory: corners + curves while the flow runs
    pdata = torch.load("/opt/stack/meshtron/experiments/flow_topology/data/flow_canon_full_post.pt", weights_only=False)
    var, vnrm, vargs = FV.load("/opt/stack/meshtron/experiments/flow_topology/runs/var_post_s0/last.pt", pdata)
    Vb = np.asarray(b.points, float)
    E = np.array(sorted({tuple(sorted((int(h[i]), int(h[j])))) for h in H for i, j in HEX_EDGES}), np.int64)
    from build_flow_dataset import surface_multihot
    raw = np.load(f"/opt/stack/meshtron/experiments/flow_topology/datagen/runs/{a.name}/sample.npz")
    tol = 6e-3 * np.linalg.norm(np.ptp(np.asarray(raw["surface_points"]), 0))
    item = {"verts": torch.tensor(Vb, dtype=torch.float32), "conn": torch.tensor(H), "edges": torch.tensor(E),
            "vlab": torch.tensor(surface_multihot(Vb, raw, tol)),
            "elab": torch.tensor(surface_multihot((Vb[E[:, 0]] + Vb[E[:, 1]]) / 2, raw, tol)),
            "eoff": torch.zeros(len(E), 6), "points": it["points"]}
    bb = FV.collate([item], dev)
    s3 = []
    with torch.no_grad():
        zz = var.enc(pts, pmask)
        xv = vnrm.enc_v(torch.tensor(Vb, dtype=torch.float32, device=dev))[None]
        xo = torch.randn(1, len(E), 6, device=dev, generator=gen)
        steps = vargs["steps"]
        for i in range(steps + 1):
            if i % 4 == 0 or i == steps:
                Vt = vnrm.dec_v(xv)[0].cpu().double().numpy(); Ot = vnrm.dec_o(xo)[0].cpu().double().numpy()
                c1, c2 = offsets_to_ctrl(Vt[E[:, 0]], Vt[E[:, 1]], Ot)
                s3.append({"t": i / steps, "curves": [r(bezier(Vt[p], Vt[q], c1[k], c2[k], 13)) for k, (p, q) in enumerate(E)]})
            if i == steps:
                break
            vv, vo = var(xv, xo, torch.full((1,), i / steps, device=dev), zz, bb)
            xv, xo = xv + vv / steps, xo + vo / steps
    out["stage3"] = s3
    for key, f in (("curves_arcs", "06_curves_arcs.vtk"), ("curves_opt", "08_curves_opt.vtk"), ("curves_s3", "05_curves_stage3.vtk")):
        pl, meta = lines_of(meshio.read(f"{V}/{f}"))
        out[key] = {"lines": pl, "meta": meta}
    for key, f in (("mesh_tfi", "07_mesh_tfi.vtk"), ("mesh_opt", "08_mesh_opt.vtk"), ("mesh_gt", "10_gt_mesh.vtk")):
        m = meshio.read(f"{V}/{f}")
        bq = mesh_edges_on_boundary(m)
        out[key] = {"wire": bq["wire"]}
    s = json.dumps(out, separators=(",", ":"))
    open(a.out, "w").write(s)
    print(f"wrote {a.out}: {len(s) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
