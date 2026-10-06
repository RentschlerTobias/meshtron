#!/usr/bin/env python3
"""Which chain step amplifies small corner errors? (overfit models, train items)

On GT corners + GT connectivity, add isotropic noise of a given size, then run
  snap -> curve head -> TFI  |  curve head only  |  snap -> straight edges
and compare with the generated corners. Metrics as infer_chain_v2.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths  # noqa: E402
import infer_chain_v2 as C  # noqa: E402
from infer_chain_v2 import load_raw, snap_all  # noqa: E402

# usage: probe_chain_sensitivity.py <v2 canon set>.pt <curve head>_last.pt <chain train dir>/summary.json

dev = "cuda"
data = torch.load(sys.argv[1], weights_only=False)
cm, spec, _ = C.load_stage(sys.argv[2], dev)
gen = {r["name"]: r for r in __import__("json").load(open(sys.argv[3]))["rows"]}
out = os.path.join(_paths.WORK, "sensitivity")
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
res = {}
for it in data["train"][:10]:
    raw = load_raw(it["name"])
    fm = C.FeatureModelV2(os.path.join(_paths.need_samples(), it["name"], "sample.npz"),
                         cache_dir=_paths.FEATURE_CACHE)
    lab = C.point_labels(len(raw["surface_points"]), raw["surface_tris"], raw["surface_tri_label"])
    pc = torch.as_tensor(np.array(C.poly_cloud(raw["surface_points"], lab, 2048, spec, np.random.default_rng(0)))[None],
                         dtype=torch.float32, device=dev)
    V0, B = it["verts"].numpy().astype(float), it["conn"].numpy().astype(np.int64).reshape(-1, 8)

    def mesh(V, snap, curves, tag):
        V2 = snap_all(V, B, fm) if snap else V
        vq = spec.quant_xyz(V2)
        e, cq = C.predict_curves(cm, vq, B, pc, len(B), spec)
        if not curves:
            cq = spec.quant_curve(np.zeros((len(e), 6)))
        r = C.mesh_candidate(C.PolySeq(vq, B, e, cq), spec, raw, 0.08, f"{out}/{tag}.vtk")
        if r.get("tfi") is None:
            return (np.nan, np.nan)
        return (r["surface"]["uncovered_share"], 100 * r["tfi"]["inverted_curved"] / max(1, r["tfi"]["cells_after"]))

    for s in (0.0, 0.005, 0.01, 0.02, 0.05):
        Vn = V0 + rng.normal(0, s / np.sqrt(3), V0.shape)          # mean displacement ~ s
        for name, snap, curves in (("snap+curve", True, True), ("curve only", False, True), ("snap+straight", True, False)):
            res.setdefault((f"GT+noise {s:.3f}", name), []).append(mesh(Vn, snap, curves, "x"))
for k in sorted(res):
    a = np.array(res[k])
    print(f"{k[0]:18s} {k[1]:14s} uncovered median {np.nanmedian(a[:, 0]):.3f}  inverted median {np.nanmedian(a[:, 1]):.3f}%")
print("generated corners (err median 18.5e-3), snap+curve: uncovered median "
      f"{np.median([gen[it['name']]['gen']['uncovered'] for it in data['train'][:10]]):.3f}")
