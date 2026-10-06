"""Diagnose GT meshes: corner scaled Jacobians, degenerate edges, boundary faces off the surface."""
import os
import sys

import numpy as np
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from meshtron.data.hexa_row_tokenizer import HexaRowTokenizer
from meshtron.training.generate import detokenize_safe
from meshtron.geometry.mesh_validation import validate_generated_mesh
from meshtron.geometry.tfi_mesh_quality import CORNER_NB as _CORNER_NB, scaled_jacobian_corners
from meshtron.training.rewards_v2 import boundary_faces, local_spacing
from scipy.spatial import cKDTree
D = torch.load(os.path.join(ROOT, "data", "hexarow_tokens_family_cart.pt"), weights_only=False)
tok = HexaRowTokenizer(r_bounds=tuple(D["r_bounds"]), z_bounds=tuple(D["z_bounds"]))
rng = np.random.default_rng(0); val = D["val"]
for i in rng.choice(len(val), 12, replace=False):
    it = val[i]
    (v, b), _ = detokenize_safe(it["tokens"].tolist(), tok, tok.core.stop_token, coords="cart")
    v = v.numpy().astype(float); b = b.numpy()
    sj = scaled_jacobian_corners(v[b])
    e = np.linalg.norm(v[b][:, np.asarray(_CORNER_NB)] - v[b][:, :, None], axis=-1)
    h = local_spacing(v, b)
    bnd, man = boundary_faces(b)
    surf = it["surface_points"].numpy()
    d, _ = cKDTree(surf).query(v[bnd].mean(1))
    off = d > 0.1 * np.median(h[bnd], 1)
    ok = validate_generated_mesh(v, b, expected_blocks=it["blocks"]).valid
    print(f"{it['name'][:28]:28s} B={len(b):3d} valid={ok:d} sj_min={sj.min():+.3f} sj<=0 corners={int((sj<=0).sum()):3d} "
          f"zero-edges={int((e < 1e-9).sum()):3d} bnd={len(bnd):3d} off-surface faces={off.mean():.2f} blade frac={it['is_blade'].float().mean():.2f}")
