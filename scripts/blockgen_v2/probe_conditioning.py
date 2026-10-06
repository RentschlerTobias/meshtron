#!/usr/bin/env python3
"""Does the vertex model use its point-cloud conditioning?

For every val item, generate corners (greedy) with (a) its own surface points and
(b) the points of another val item. Reports the corner error to the item's GT for
both, the spread of the predictions across items, and the mean-template baseline.
Usage: probe_conditioning.py <vertex run>/best.pt <v2 dataset>.pt
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths  # noqa: E402,F401  (puts the repo root on sys.path)
from meshtron.model.blockgen import BlockGen  # noqa: E402
from meshtron.training import train_blockgen as T  # noqa: E402
from meshtron.data.hexa_row_tokenizer import HexaRowTokenizer  # noqa: E402

ck = torch.load(sys.argv[1], weights_only=False); a = ck["args"]
data = torch.load(sys.argv[2], weights_only=False)
tr, va = data["train"], data["val"]
va = [i for i in va if i["rows"] == "3-2-3-3-1"]
sp = data["special"]; specials = set(sp.values())
max_len = max(len(it["vtok"]) for it in tr + data["val"]) + 64
m = BlockGen(data["vocab"], a["d"], a["layers"], a["heads"], max_len, sp["pad"], 0.0, a["n_latent"]).cuda()
m.load_state_dict(ck["model"]); m.eval()
tok = HexaRowTokenizer(r_bounds=tuple(data["r_bounds"]), z_bounds=tuple(data["z_bounds"]))


@torch.no_grad()
def gen(items):
    pts, pm = T.batch_points(items, "cuda")
    out = T.generate(m, pts, pm, sp["start"], sp["stop"], specials, max(len(i["vtok"]) for i in items) + 40)
    V = []
    for seq in out.tolist():
        if sp["stop"] in seq[1:]:
            seq = seq[:seq.index(sp["stop"], 1)]
        body = [t for t in seq[1:] if t not in specials]
        body = body[:len(body) // 3 * 3]
        V.append(T.dequant_vertices(tok, body))
    return V


own = gen(va)
rolled = gen(va[1:] + va[:1])            # item k gets the points of item k+1
mean = np.stack([i["verts"].numpy() for i in tr if i["rows"] == "3-2-3-3-1"]).mean(0)


def err(V, gt):
    if V is None or len(V) != len(gt):
        return np.nan
    return 1000 * np.linalg.norm(V - gt, axis=1).mean()


e_own = [err(v, i["verts"].numpy()) for v, i in zip(own, va)]
e_rol = [err(v, i["verts"].numpy()) for v, i in zip(rolled, va)]
e_rol_to_src = [err(v, i["verts"].numpy()) for v, i in zip(rolled, va[1:] + va[:1])]
e_mean = [err(mean, i["verts"].numpy()) for i in va]
P = np.stack([v for v in own if v is not None and len(v) == 40])
G = np.stack([i["verts"].numpy() for i in va])
print(f"items {len(va)}  ({sys.argv[1]})")
print(f"corner error x1000, median: own points {np.nanmedian(e_own):.1f} | other item's points {np.nanmedian(e_rol):.1f}"
      f" | (vs that other item's GT {np.nanmedian(e_rol_to_src):.1f}) | mean template {np.median(e_mean):.1f}")
print(f"spread across items x1000 (median over corners of the std): predictions {1000*np.median(np.linalg.norm(P.std(0),axis=1)):.1f}"
      f" | GT {1000*np.median(np.linalg.norm(G.std(0),axis=1)):.1f}")
