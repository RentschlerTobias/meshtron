#!/usr/bin/env python3
"""Compare reward v1 (rewards_hexarow) and the v2 prototype under controlled
perturbations of GT meshes (30 val items of hexarow_tokens_family_cart.pt).
Besides the means, reports the within-item spread that GRPO actually sees:
for each item the rank correlation of the reward with perturbation severity."""
import os
import sys
import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from meshtron.data.hexa_row_tokenizer import HexaRowTokenizer
from meshtron.training.generate import detokenize_safe
from meshtron.training.rewards_hexarow import chamfer_symmetric
from meshtron.geometry.mesh_validation import hex_min_jacobian, hex_signed_volumes, validate_generated_mesh
from meshtron.training.rewards_v2 import score, local_spacing, boundary_faces

D = torch.load(os.path.join(ROOT, "data", "hexarow_tokens_family_cart.pt"), weights_only=False)
tok = HexaRowTokenizer(r_bounds=tuple(D["r_bounds"]), z_bounds=tuple(D["z_bounds"]))
rng = np.random.default_rng(0)


def v1(v, blk, it):
    gt = it["surface_points"].numpy().astype(np.float64)
    w = np.where(it["is_blade"].numpy().astype(bool), 2.0, 1.0)
    diag = float(np.linalg.norm(gt.max(0) - gt.min(0)))
    rc = max(0.0, 1.0 - min(1.0, chamfer_symmetric(v, gt, w) / diag))
    cells = v[blk]; jac = hex_min_jacobian(cells); vol = hex_signed_volumes(cells)
    ori = jac[vol > 0]
    rq = (ori.mean() if ori.size else 0.0) - np.maximum(0, -jac).mean() - (0.5 if ori.size == 0 else 0)
    valid = validate_generated_mesh(v, blk, expected_blocks=int(it["blocks"])).valid
    gtb = it["blocks"]; rcount = max(0, 1 - abs(len(blk) - gtb) / gtb)
    return 0.5 * rcount + 0.5 * rq + 0.1 * rc + 1.0 * valid


def mesh(it):
    (v, b), _ = detokenize_safe(it["tokens"].tolist(), tok, tok.core.stop_token, coords="cart")
    return v.numpy().astype(np.float64), b.numpy()


val = D["val"]
items = [val[i] for i in rng.choice(len(val), 30, replace=False)]
names = ["GT", "1 bnd corner +0.3h", "noise 0.1h", "noise 0.3h", "shift 5% diag",
         "scale 0.8", "other geometry"]
severity = [0, 1, 2, 3, 3, 3, 4]
R = {k: [] for k in ("v1", "v2", "gate", "fid", "lab", "q", "cov_blade", "sj_p5")}
rho1, rho2 = [], []
for it in items:
    v, b = mesh(it)
    surf = it["surface_points"].numpy().astype(np.float64)
    lab = np.where(it["is_blade"].numpy().astype(bool), 7, 0)
    diag = float(np.linalg.norm(surf.max(0) - surf.min(0)))
    h = np.nan_to_num(local_spacing(v, b), nan=np.nanmedian(local_spacing(v, b)))
    c = v.mean(0)
    bnd, _ = boundary_faces(b)
    k = int(rng.choice(np.unique(bnd)))
    v_one = v.copy(); v_one[k] += 0.3 * h[k] * (c - v[k]) / np.linalg.norm(c - v[k])
    other = next(o for o in val if o["geom_id"] != it["geom_id"] and abs(o["blocks"] - it["blocks"]) <= 2)
    variants = [(v, b), (v_one, b),
                (v + rng.normal(0, 1, v.shape) * 0.1 * h[:, None], b),
                (v + rng.normal(0, 1, v.shape) * 0.3 * h[:, None], b),
                (v + 0.05 * diag / np.sqrt(3), b), (c + 0.8 * (v - c), b), mesh(other)]
    r1s, r2s = [], []
    for vv, bb in variants:
        r1 = v1(vv, bb, it); r2 = score(vv, bb, surf, lab)
        r1s.append(r1); r2s.append(r2.total)
        for key, val_ in (("v1", r1), ("v2", r2.total), ("gate", r2.gate), ("fid", r2.r_fid),
                          ("lab", r2.r_lab), ("q", r2.r_q), ("cov_blade", r2.detail["cov_blade"]),
                          ("sj_p5", r2.detail["sj_p5"])):
            R[key].append(val_)
    rho1.append(spearmanr(severity, r1s)[0]); rho2.append(spearmanr(severity, r2s)[0])

n = len(names)
print(f"{'variant':20s}" + "".join(f"{k:>10s}" for k in R))
for i, nm in enumerate(names):
    print(f"{nm:20s}" + "".join(f"{np.nanmean(R[k][i::n]):10.3f}" for k in R))
print(f"\nSpearman(reward, severity) per item, mean: v1 {np.nanmean(rho1):+.3f}   v2 {np.nanmean(rho2):+.3f}")
gt1, gt2 = np.array(R["v1"][0::n]), np.array(R["v2"][0::n])
ot1, ot2 = np.array(R["v1"][6::n]), np.array(R["v2"][6::n])
print(f"GT beats other-geometry mesh: v1 {np.mean(gt1 > ot1):.2f}   v2 {np.mean(gt2 > ot2):.2f}")
