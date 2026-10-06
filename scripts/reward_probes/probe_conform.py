#!/usr/bin/env python3
"""Sensitivity probe for the HexaRow GRPO reward terms (rewards_hexarow.py).

Takes ground-truth meshes of val items, applies controlled perturbations and
reports how much each reward term moves. A term that barely moves between the
GT mesh and a clearly wrong mesh cannot discriminate rollouts inside a GRPO
group (saturated / insensitive).
"""
import os
import sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from meshtron.data.hexa_row_tokenizer import HexaRowTokenizer
from meshtron.training.generate import detokenize_safe
from meshtron.training.rewards_hexarow import chamfer_symmetric
from meshtron.geometry.mesh_validation import hex_min_jacobian, hex_signed_volumes, validate_generated_mesh

D = torch.load(os.path.join(ROOT, "data", "hexarow_tokens_family_cart.pt"), weights_only=False)
tok = HexaRowTokenizer(r_bounds=tuple(D["r_bounds"]), z_bounds=tuple(D["z_bounds"]))
stop = tok.core.stop_token
rng = np.random.default_rng(0)


def terms(v, blk, it):
    gt = it["surface_points"].numpy().astype(np.float64)
    w = np.where(it["is_blade"].numpy().astype(bool), 2.0, 1.0)
    diag = float(np.linalg.norm(gt.max(0) - gt.min(0)))
    d = chamfer_symmetric(v, gt, w)
    rc = max(0.0, 1.0 - min(1.0, d / diag))
    cells = v[blk]
    jac = hex_min_jacobian(cells); vol = hex_signed_volumes(cells)
    ori = jac[vol > 0]
    rq = (ori.mean() if ori.size else 0.0) - np.maximum(0, -jac).mean() - (0.5 if ori.size == 0 else 0)
    valid = validate_generated_mesh(v, blk, expected_blocks=int(it["blocks"])).valid
    # nearest-surface distance of the generated vertices only (one direction)
    from meshtron.training.rewards_hexarow import _nn_distances
    d_ab = float(_nn_distances(v, gt).mean()) / diag
    return rc, rq, float(valid), d / diag, d_ab


def mesh(it):
    res, _ = detokenize_safe(it["tokens"].tolist(), tok, stop, coords="cart")
    v, b = res
    return v.numpy().astype(np.float64), b.numpy()


val = D["val"]
items = [val[i] for i in rng.choice(len(val), 30, replace=False)]
cases = {}
for it in items:
    v, b = mesh(it)
    gt = it["surface_points"].numpy()
    diag = float(np.linalg.norm(gt.max(0) - gt.min(0)))
    c = v.mean(0)
    other = next(o for o in val if o["geom_id"] != it["geom_id"] and abs(o["blocks"] - it["blocks"]) <= 2)
    vo, bo = mesh(other)
    variants = {
        "GT": (v, b),
        "noise 1% diag": (v + rng.normal(0, 0.01 * diag, v.shape), b),
        "noise 3% diag": (v + rng.normal(0, 0.03 * diag, v.shape), b),
        "noise 10% diag": (v + rng.normal(0, 0.10 * diag, v.shape), b),
        "shift 5% diag": (v + 0.05 * diag / np.sqrt(3), b),
        "scale 0.8": (c + 0.8 * (v - c), b),
        "other geometry": (vo, bo),
    }
    for k, (vv, bb) in variants.items():
        cases.setdefault(k, []).append(terms(vv, bb, it))

print(f"{'variant':16s} {'r_conform':>10s} {'0.1*r_conf':>10s} {'r_quality':>10s} {'valid':>6s} {'chamf/diag':>10s} {'gen->srf/diag':>13s}")
for k, rows in cases.items():
    a = np.array(rows)
    m = a.mean(0)
    print(f"{k:16s} {m[0]:10.3f} {0.1*m[0]:10.4f} {m[1]:10.3f} {m[2]:6.2f} {m[3]:10.4f} {m[4]:13.4f}")
