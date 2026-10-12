#!/usr/bin/env python3
"""Inference of a flow_geo checkpoint on a validation split, written as a ParaView-friendly
VTK set: one folder per geometry plus index.csv.

Per item (<out>/<split>/<name>/):
  surface.vtk      input geometry: surface triangles, cell data 'label' (1..7, 7 = blade / O-grid cut)
  points.vtk       the point cloud the model is conditioned on (all surface points, vertex cells);
                   point data 'label_1'..'label_7' (multi-hot of the incident surface labels) and
                   'label_code' (bitmask, bit k-1 = label k)
  pred_blocks.vtk  predicted block structure: 12 coarse hex blocks (corners, snapped) + the cubic
                   Bezier edges as polylines; cell data 'kind' (0 block, 1 edge), 'id'
  pred_mesh.vtk    curved TFI mesh of the prediction; cell data 'block_id', 'scaled_jacobian'
                   (min over the 8 corners, <= 0 = inverted)
  gt_blocks.vtk, gt_mesh.vtk   the same for the GT block structure
index.csv: corner error, uncovered surface share, inverted cells, sj_min for prediction and GT.

The prediction is one sample per item (fixed seed), corners snapped as in infer_chain_flow.py.

    PYTHONPATH=<dp3d hex3d_algohex> python export_vtk_val.py --data data/flow_canon_papaya.pt \
        --ckpt runs/flowp_flow_all_s0/last.pt --split val_new --src-root /work/datagen_papaya/runs \
        --out vtk_val
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, "/opt/stack/meshtron")
sys.path.insert(0, "/work/meshtron_v2")
sys.path.insert(0, "/work/rl_reward")

import meshio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import train as T  # noqa: E402
from flow_geo import FlowGeo, Topology, sample  # noqa: E402
from infer_chain_flow import RELABEL, to_seq  # noqa: E402
from infer_chain_v2 import snap_all  # noqa: E402
from meshtron.data.polytron_blocks import bezier, offsets_to_ctrl  # noqa: E402
from meshtron.geometry.geometry_features import FeatureModelV2  # noqa: E402
from meshtron.geometry.polytron_tfi import mesh_candidate  # noqa: E402
from meshtron.training.train_polytron import load_stage  # noqa: E402
from rewards_v2 import scaled_jacobian_corners  # noqa: E402


def write_blocks(path, V, B, edges, off, n_pts=17):
    """Coarse hex blocks + Bezier edge polylines in one file."""
    c1, c2 = offsets_to_ctrl(V[edges[:, 0]], V[edges[:, 1]], off)
    pts, lines = [V], []
    base = len(V)
    for k, (a, b) in enumerate(edges):
        P = bezier(V[a], V[b], c1[k], c2[k], n_pts)
        pts.append(P)
        idx = base + np.arange(n_pts)
        lines.append(np.stack([idx[:-1], idx[1:]], 1)); base += n_pts
    L = np.concatenate(lines)
    eid = np.repeat(np.arange(len(edges)), n_pts - 1)
    meshio.write(path, meshio.Mesh(np.concatenate(pts), [("hexahedron", B), ("line", L)],
                                   cell_data={"kind": [np.zeros(len(B), np.int32), np.ones(len(L), np.int32)],
                                              "id": [np.arange(len(B), dtype=np.int32), eid.astype(np.int32)]}),
                 binary=True)


def annotate_mesh(path):
    """Add the per-cell scaled Jacobian to a TFI mesh in place; returns (sj_min, n_inverted)."""
    m = meshio.read(path)
    hx = [i for i, c in enumerate(m.cells) if c.type == "hexahedron"]
    sjs = []
    for i, c in enumerate(m.cells):
        sjs.append(scaled_jacobian_corners(m.points[c.data]).min(1) if i in hx else np.zeros(len(c.data)))
    m.cell_data["scaled_jacobian"] = sjs
    meshio.write(path, m, binary=True)
    s = np.concatenate([sjs[i] for i in hx])
    return float(s.min()), int((s <= 0).sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="dataset the checkpoint was trained on")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--eval-data", default=None, help="take the --split items from this file (default --data)")
    ap.add_argument("--src-root", default=RELABEL, help="directory with <name>/sample.npz")
    ap.add_argument("--spec-from", default="/work/meshtron_v2/runs/curve_order_last.pt")
    ap.add_argument("--target-h", type=float, default=0.08)
    ap.add_argument("--feature-cache", default="/work/curve/features_v2")
    ap.add_argument("--out", default="/work/meshtron_v2/vtk_val")
    ap.add_argument("--n", type=int, default=None, help="only the first n items")
    a = ap.parse_args()
    dev = "cuda"
    data = torch.load(a.data, weights_only=False)
    ck = torch.load(a.ckpt, weights_only=False); fa = ck["args"]
    rmax = float(data["r_bounds"][1]); z0, z1 = (float(v) for v in data["z_bounds"])
    lo, hi = torch.tensor([-rmax, -rmax, z0]), torch.tensor([rmax, rmax, z1])
    tp = Topology(data["train"], lo, hi).to(dev)
    model = FlowGeo(tp, fa["d"], fa["layers"], fa["heads"], fa["n_latent"], 0.0, curves=fa["parts"] == "all").to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    _, spec, _ = load_stage(a.spec_from, dev)
    edges = tp.edges.cpu().numpy()
    items = (torch.load(a.eval_data, weights_only=False) if a.eval_data else data)[a.split][:a.n]
    root = os.path.join(a.out, a.split); os.makedirs(root, exist_ok=True)
    gen = torch.Generator(device=dev); gen.manual_seed(1234)
    rows = []
    for it in items:
        name = it["name"]; D = os.path.join(root, name); os.makedirs(D, exist_ok=True)
        npz = os.path.join(a.src_root, name, "sample.npz")
        with np.load(npz, allow_pickle=True) as zf:
            raw = {k: np.asarray(zf[k]) for k in zf.files}
        fm = FeatureModelV2(npz, cache_dir=a.feature_cache)
        B = it["conn"].numpy().astype(np.int64)
        with torch.no_grad():
            pts, pmask = T.batch_points([it], dev)
            V, O = tp.decode(*sample(model, model.enc(pts, pmask), fa["mode"], fa["steps"], gen))
        V = V[0].cpu().numpy().astype(np.float64); O = O[0].cpu().numpy().astype(np.float64)
        Vs = snap_all(V, B, fm)
        Vg, Og = it["verts"].numpy().astype(np.float64), it["eoff"].numpy().astype(np.float64)
        meshio.write(os.path.join(D, "surface.vtk"),
                     meshio.Mesh(raw["surface_points"], [("triangle", raw["surface_tris"])],
                                 cell_data={"label": [raw["surface_tri_label"].astype(np.int32)]}), binary=True)
        lab = it["points"][:, -7:].float().numpy()               # same points / order as the model input
        P = raw["surface_points"]; assert len(P) == len(lab)
        pd = {f"label_{k + 1}": lab[:, k].astype(np.float32) for k in range(7)}
        pd["label_code"] = (lab * (2 ** np.arange(7))).sum(1).astype(np.int32)
        meshio.write(os.path.join(D, "points.vtk"),
                     meshio.Mesh(P, [("vertex", np.arange(len(P))[:, None])], point_data=pd), binary=True)
        row = {"name": name, "corner_err_x1000": round(float(1000 * np.linalg.norm(V - Vg, axis=1).mean()), 2)}
        for tag, (VV, OO) in (("pred", (Vs, O)), ("gt", (Vg, Og))):
            write_blocks(os.path.join(D, f"{tag}_blocks.vtk"), VV, B, edges, OO)
            vtk = os.path.join(D, f"{tag}_mesh.vtk")
            r = mesh_candidate(to_seq(VV, B, edges, OO, spec), spec, raw, a.target_h, vtk, write_edges=False)
            if r.get("tfi") is None:
                row.update({f"{tag}_meshed": False}); continue
            sj_min, n_inv = annotate_mesh(vtk)
            row.update({f"{tag}_meshed": True, f"{tag}_watertight": bool(r["tfi"]["watertight"]),
                        f"{tag}_uncovered_pct": round(100 * float(r["surface"]["uncovered_share"]), 3),
                        f"{tag}_inverted_cells": n_inv, f"{tag}_cells": int(r["tfi"]["cells_after"]),
                        f"{tag}_sj_min": round(sj_min, 4)})
        rows.append(row)
        print(name, {k: v for k, v in row.items() if k != "name"}, flush=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(root, "index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} items to {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
