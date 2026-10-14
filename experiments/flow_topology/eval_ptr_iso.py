#!/usr/bin/env python3
"""M3 check: is the block structure the pointer builds on stage-1 corners topologically the GT one?

Per item: stage-1 corners (flow_vertex.py, predicted count) -> pointer blocks (pointer_conn.decode). Compared to
GT by
  exact_id   block corner sets equal after Hungarian matching of predicted to GT corners (pointer_conn.evaluate)
  iso        the block complexes are isomorphic: graph of corner nodes + block nodes, corner-corner edges for the
             12 hex edges of every block, corner-block incidence edges, node type as attribute (networkx VF2);
             independent of which predicted corner is matched to which GT corner
  iso_lab    as iso, corners additionally labelled with their surface patch set (predicted corners: patches
             within the on-surface tolerance of the input surface; GT: the dataset multi-hot)
plus validity of the predicted complex: every face in <= 2 blocks, positive scaled Jacobian at all block corners.

    python eval_ptr_iso.py --ptr-ckpt runs/ptr_s0/last.pt --vtx-ckpt runs/vtx_full_s0/best.pt --out runs/ptr_s0/iso.json
"""
import argparse
import json
import os
import sys
from collections import Counter

import networkx as nx
import numpy as np
import torch
from networkx.algorithms.isomorphism import categorical_node_match
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, "/work/meshtron_v2")
import flow_vertex as FVX  # noqa: E402
import pointer_conn as PC  # noqa: E402
from build_flow_dataset import surface_multihot  # noqa: E402
from flow_geo_var import Norm  # noqa: E402
from train import batch_points  # noqa: E402

HEX_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
NB = ((1, 3, 4), (2, 0, 5), (3, 1, 6), (0, 2, 7), (7, 5, 0), (4, 6, 1), (5, 7, 2), (6, 4, 3))
SRC = {"tistos": "/work/datagen_full/datagen/runs", "machine": "/work/relabel/out"}


def complex_graph(blocks, labels=None):
    G = nx.Graph()
    used = sorted({int(c) for b in blocks for c in b})
    for c in used:
        G.add_node(("v", c), kind="v" if labels is None else "v" + labels[c])
    for bi, b in enumerate(blocks):
        G.add_node(("b", bi), kind="b")
        for c in b:
            G.add_edge(("b", bi), ("v", int(c)))
        for i, j in HEX_EDGES:
            G.add_edge(("v", int(b[i])), ("v", int(b[j])))
    return G


def validity(P, blocks):
    faces = Counter(tuple(sorted(int(b[i]) for i in f)) for b in blocks for f in PC.FACES)
    manifold = all(n <= 2 for n in faces.values())
    sj = []
    for b in blocks:
        X = P[np.asarray(b)]
        e = X[np.asarray(NB)] - X[:, None]
        e = e / np.maximum(np.linalg.norm(e, axis=-1, keepdims=True), 1e-30)
        sj.append(np.linalg.det(e).min())
    return manifold, float(min(sj)) if sj else -1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="/work/meshtron_v2/data/flow_canon_full.pt")
    ap.add_argument("--ptr-ckpt", required=True)
    ap.add_argument("--vtx-ckpt", required=True)
    ap.add_argument("--splits", default="val,val_new,val_full")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    data = torch.load(a.data, weights_only=False)
    ck = torch.load(a.ptr_ckpt, weights_only=False); pa = ck["args"]
    ptr = PC.SetPointer(pa["d"], pa["layers"], pa["heads"], ck["max_len"]).to(dev)
    ptr.load_state_dict(ck["model"]); ptr.eval()
    nrm = Norm(data, data["train"]).to(dev)
    vm, vn, _ = FVX.load(a.vtx_ckpt, data)
    gen = torch.Generator(device=dev); gen.manual_seed(1234)
    out = {}
    for sp in a.splits.split(","):
        rows = []
        for it in data[sp]:
            G = it["verts"].double().numpy()
            pts, pmask = batch_points([it], dev)
            with torch.no_grad():
                zv = vm.enc(pts, pmask)
                k = int(vm.count_logits(zv).argmax(-1)[0])
                P = vn.dec_v(FVX.sample(vm, zv, [k], 32, gen)[0]).cpu().double().numpy()
                x = nrm.enc_v(torch.tensor(P, dtype=torch.float32, device=dev))[None]
                vmask = torch.ones(1, len(P), dtype=torch.bool, device=dev)
                e, z = ptr.memory(x, vmask, pts, pmask)
                blocks = PC.decode(ptr, e, z, vmask)
            D = np.linalg.norm(P[:, None] - G[None], axis=2)
            r, c = linear_sum_assignment(D)
            match = {int(i): int(j) for i, j in zip(r, c)}
            gt = it["conn"].long().numpy()
            raw = np.load(os.path.join(SRC[it["name"].split("_")[0]], it["name"], "sample.npz"), allow_pickle=False)
            tol = 6e-3 * np.linalg.norm(np.ptp(np.asarray(raw["surface_points"]), 0))
            lp = surface_multihot(P, raw, tol)
            lab_p = ["".join(str(int(v)) for v in row) for row in lp]
            lab_g = ["".join(str(int(v)) for v in row.numpy()) for row in it["vlab"]]
            Gp, Gg = complex_graph(blocks), complex_graph(gt)
            iso = len(blocks) == len(gt) and nx.is_isomorphic(Gp, Gg, node_match=categorical_node_match("kind", None))
            iso_lab = iso and nx.is_isomorphic(complex_graph(blocks, lab_p), complex_graph(gt, lab_g),
                                               node_match=categorical_node_match("kind", None))
            manifold, sjmin = validity(P, blocks)
            rows.append({"name": it["name"], "n_corners": len(P), "n_blocks": len(blocks),
                         "exact_id": PC.topo_exact(blocks, gt, match), "iso": bool(iso), "iso_lab": bool(iso_lab),
                         "manifold": manifold, "block_sj_min": sjmin})
        n = len(rows)
        out[sp] = {"n": n, "exact_id": sum(r["exact_id"] for r in rows) / n, "iso": sum(r["iso"] for r in rows) / n,
                   "iso_lab": sum(r["iso_lab"] for r in rows) / n, "manifold": sum(r["manifold"] for r in rows) / n,
                   "positive_blocks": sum(r["block_sj_min"] > 0 for r in rows) / n,
                   "blocks": dict(Counter(r["n_blocks"] for r in rows)), "rows": rows}
        print(f"{sp:8s} n {n:3d} | exact(id) {out[sp]['exact_id']:.2f} | isomorphic {out[sp]['iso']:.2f} | "
              f"iso+labels {out[sp]['iso_lab']:.2f} | manifold {out[sp]['manifold']:.2f} | all blocks sJ>0 "
              f"{out[sp]['positive_blocks']:.2f} | block counts {out[sp]['blocks']}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
