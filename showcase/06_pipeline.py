"""06 -- the full inference pipeline, live: npz -> cloud -> transformer ->
mapping -> TFI. The one chain scripts/infer.py runs and the other tutorials
assume. Every stage writes what it produced, so a bad mesh can be taken
apart stage by stage.

This is the middle ground 01 skips: 01 loads a pre-computed generation
artifact and maps the ground-truth blocks. Here the checkpoint actually
samples, the generated blocks are validated and snapped, and the refilled
CFD mesh is built from them.

This tutorial is designed in and for nvim using a python repl (iron.nvim).
You can also run the whole file non-interactive: uv run python showcase/06_pipeline.py
"""

# %% [0] setup
import os
import sys
import types

import numpy as np
import torch

# Run as a file and __file__ gives the location. Paste a cell into a REPL and
# the code is stdin, so there is no __file__ -- then locate showcase/ from the
# working directory instead. Start the REPL in the repo root or in showcase/.
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.path.abspath("showcase" if os.path.isdir("showcase") else ".")

if not os.path.isfile(os.path.join(HERE, "_common.py")):
    raise RuntimeError(f"showcase/_common.py not found from {os.getcwd()!r} -- "
                       "start the REPL in the meshtron repo root")

if HERE not in sys.path:
    sys.path.insert(0, HERE)

import _common as C  # noqa: E402

from meshtron.geometry.block_mapping import (SnapConfigV2,  # noqa: E402
                                             score_candidate,
                                             snap_corners_v2, tier_counts)
from meshtron.geometry.conform import (ConformOptions,  # noqa: E402
                                       collapsed_faces, conform_blocks)
from meshtron.geometry.geometry_features import FeatureModelV2  # noqa: E402
from meshtron.geometry.mesh_validation import validate_generated_mesh  # noqa: E402
from meshtron.training.generate import detokenize_safe, generate  # noqa: E402
from scripts.infer import (cloud_from_npz, write_cloud_vtk,  # noqa: E402
                           write_surface_vtk)
from scripts.compare_viz import _to_cart  # noqa: E402

npz = os.path.join(C.BATCH, C.MACHINE, "sample.npz")
# The production RL checkpoint (the one _common.CKPT predates: it has slot
# ids, coords=cart and was refined by GRPO).
CKPT = os.path.join(C.DATA, "grpo_cart_step300.pt")

C.head("meshtron inference pipeline, live")

# %% [1] geometry -- the only thing the pipeline is given
fm = FeatureModelV2(npz, cache_dir=os.path.join(C.DATA, "features"))
g1 = write_surface_vtk(os.path.join(C.outdir(), "08_geometry.vtk"), fm,
                       C.MACHINE)
print(f"{g1['points']} points, {g1['tris']} triangles")
for k, v in g1["labels"].items():
    print(f"   patch {k:10s} {v:6d} triangles")

# %% [2] the model, loaded exactly as inference loads it
# _common.load_model uses the old CKPT; load the RL checkpoint directly so the
# cfg (n_points, bounds, slot flag) comes from the checkpoint that made the
# meshes.
from scripts.eval_family import load_model  # noqa: E402
from meshtron.data.hexa_row_tokenizer import HexaRowTokenizer  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if dev == "cuda" else torch.float32
ck, cfg, coords, npt, rb, zb, model, max_len, missing = load_model(CKPT, dev)
tok = HexaRowTokenizer(r_bounds=rb, z_bounds=zb)
core = tok.core
specials = {core.start_token, core.end_token, core.sep_token,
            core.sep2_token, core.stop_token, core.pad_token}
use_slot = "slot.weight" in ck["model"]
C.show("cfg", cfg, 0)
print(f"coords={coords} npt={npt} max_len={max_len} slot={use_slot} "
      f"dev={dev} missing={len(missing.missing_keys)}")

# %% [3] the conditioning cloud, built from the npz alone
# A new machine needs no dataset entry: surface_points and is_blade both come
# out of the geometry. blade_weight oversamples the blade hull because that is
# where the blocking has to agree with geometry.
rng = np.random.default_rng(0)
item, pts, blade_mask = cloud_from_npz(fm, int(cfg["n_points"]), rb, zb, rng,
                                       blade_weight=3.0)
pc = torch.as_tensor(pts[None], dtype=torch.float32, device=dev)
C.show("cloud", pts, 2)
print(f"blade {np.mean(item['is_blade']):.1%} of surface -> "
      f"{np.mean(blade_mask):.1%} of samples")
write_cloud_vtk(os.path.join(C.outdir(), "09_cloud.vtk"),
                item["surface_points"][:len(pts)], item["is_blade"][:len(pts)],
                C.MACHINE)

# %% [4] the transformer: k rollouts of the block structure
# --blocks is an INPUT: the block count conditions the model (FiLM) and the
# npz does not carry it. A known sample hides it in the npz blocking.
k_roll, temp = 4, 0.9
fc = torch.tensor([float(len(fm.blocks))], device=dev)
print(f"conditioning on {len(fm.blocks)} blocks (from the npz), k={k_roll} "
      f"temperature={temp}")

cands = []
for i in range(k_roll):
    torch.manual_seed(2 + i)
    seq = generate(model, pc, fc, core.start_token, core.stop_token,
                   core.sep_token, max_len - 1, temp, 0, dev, dtype,
                   specials, use_slot, tok=tok, constrained=True,
                   coords=coords)
    res, trim = detokenize_safe(seq, tok, core.stop_token, coords=coords)
    if res is None:
        print(f"   r{i}: detok FAILED ({trim})")
        continue
    vpt, blk = res
    vcart = _to_cart(vpt.numpy(), coords)
    val = validate_generated_mesh(vcart, blk.numpy())
    print(f"   r{i}: tok={len(seq)} blocks={blk.shape[0]} "
          f"{'valid' if val.valid else 'INVALID: ' + str(val.errors[:1])}")
    if val.valid:
        cands.append({"r": i, "v": vcart, "b": blk.numpy().astype(int)})

print(f"{len(cands)} valid rollouts survive")

# %% [5] snap the generated corners onto the geometry
# The raw generation is on the quantised lattice; snapping pulls corners onto
# seams and patches. The candidate with the smallest mean snap distance wins.
target = types.SimpleNamespace(curves=fm.seam_curves,
                               surface_nearest=fm.surface_nearest)
for r in cands:
    Cns = r["v"][r["b"]]
    Csnap, records = snap_corners_v2(target, Cns, SnapConfigV2())
    mean_d, min_j = score_candidate(Csnap, records)
    r["C"] = Cns
    r["Csnap"] = Csnap
    r["rec"] = records
    r["mean_snap"] = float(mean_d)
    r["min_j"] = float(min_j)
    r["collapsed"] = int(collapsed_faces(r["b"], Csnap))
    print(f"   r{r['r']}: mean_snap={mean_d:.5f} minJ={min_j:.5f} "
          f"collapsed={r['collapsed']} tiers={tier_counts(records)}")

ok = [r for r in cands if r["collapsed"] == 0]
if not ok:
    raise SystemExit("every candidate has collapsed block faces -- refill would fail")

best = min(ok, key=lambda r: r["mean_snap"])
v_snap = best["v"].copy()
v_snap[best["b"]] = best["Csnap"]
print(f"chosen r{best['r']} ({best['b'].shape[0]} blocks)")

used = sorted({int(j) for b in best["b"] for j in b})
remap = {v: i for i, v in enumerate(used)}
blocks_idx = [[remap[int(j)] for j in b] for b in best["b"]]
C.write_hexes(os.path.join(C.outdir(), "10_blocks_generated.vtk"),
              best["v"][used],
              blocks_idx,
              {"block_id": np.arange(len(blocks_idx))},
              "generated blocking, snapped")

# %% [6] map onto the geometry: routes, projections, TFI refill
opt = ConformOptions(target_h=0.05, geodesic=True, project_faces=True,
                     blend_boundary=4)
out = conform_blocks(fm, best["b"], best["Csnap"], best["rec"],
                     os.path.join(C.outdir(), "11_cfd"), opt,
                     stem=C.MACHINE)
bnd = out["boundary"]
C.head("[6] cfd refill")
print(f"cells={out['cells_after']} watertight={out['watertight']} "
      f"inverted={out['inverted_curved']} "
      f"({100.0 * out['inverted_curved'] / max(1, out['cells_after']):.2f}%)")
print(f"boundary vs npz surface: max={bnd['max']:.3e} "
      f"p99={bnd.get('p99', float('nan')):.3e} n={bnd.get('n')}")

# %% [7] verdict
C.head("[7] verdict")
print(f"sits_on_geometry {bnd['max'] <= 1e-3}  "
      f"watertight {out['watertight']}")
print("inverted cells are a known open item of the mapping (edge shape), "
      "not of the geometry -- colour cell 11 by nothing and look at the "
      "first cell layer at the blade")

