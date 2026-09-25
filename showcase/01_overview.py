"""01 -- the whole chain, high level.

Eight interactive steps from the parametric geometry to a CFD mesh.
Run a cell, look at the numbers, the function under the hood, look at the VTK it wrote, move on.

Nothing here is deep -- this is an high level overview

This tutorial is design in and for nvim using an python repl (iron.nvim)
You can can the whole plain file non interactive: uv run python -i showcase/01_overview.py
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

C.head("meshtron pipeline, high level")
print(f"machine  {C.MACHINE}")
print(f"out      {C.outdir()}")

# %% [1] the geometry, as the pipeline sees it
# The parametric definition (30 cV_ru values) is meshed by gmsh into tets; the
# boundary of that tet mesh, with a patch label per triangle, IS the geometry
# every later step works against.
npz = os.path.join(C.BATCH, C.MACHINE, "sample.npz")
z = np.load(npz, allow_pickle=True)

C.head("[1] geometry")
C.show("surface_points", z["surface_points"], 2)
C.show("surface_tris", z["surface_tris"], 2)
labels, counts = np.unique(z["surface_tri_label"], return_counts=True)
for lab, n in zip(labels, counts):
    print(f"   label {lab}  {C.PATCHES.get(int(lab), '?'):11s} {n:6d} triangles")

# %% [2] the conditioning cloud the transformer is given
# Not the mesh, not the geometry: a fixed number of points drawn from the
# surface, in (r, sin(theta), cos(theta), z), normalised into the bounds the
# model was trained with.
from meshtron.data import conditioning  # noqa: E402

src = torch.load(C.SRC, weights_only=False)
sample = next(s for s in src["samples"] if s["name"] == C.MACHINE)
ck, cfg, coords, npt, rb, zb, model, max_len, _ = C.load_model()
cloud = conditioning.build_cloud(
    sample, cfg["n_points"], rb, zb, np.random.default_rng(0)
)
cloud = cloud[0] if isinstance(cloud, tuple) else cloud

C.head("[2] conditioning cloud")
C.show("cloud", cloud, 3)
print(f"   r bounds {rb}   z bounds {zb}   coords {coords}")
xyz = C.cloud_to_xyz(cloud, rb, zb)
C.write_points(
    os.path.join(C.OUT, "02_cloud.vtk"),
    xyz,
    {"r_norm": cloud[:, 0], "z_norm": cloud[:, 3]},
    "conditioning cloud, back in xyz",
)

# %% [3] the ground-truth block structure
# What AlgoHex produced and the export kept: corner coordinates plus one row of
# eight corner indices per block.
C.head("[3] ground-truth blocking")
V = np.asarray(z["vertices"], float)
B = np.asarray(z["blocks"], np.int64)
C.show("vertices", V, 2)
C.show("blocks", B, 2)
print(f"   {len(B)} blocks over {len(V)} corners")
C.write_hexes(
    os.path.join(C.OUT, "03_blocks_gt.vtk"),
    V,
    B,
    {"block_id": np.arange(len(B))},
    "GT block structure",
)

# %% [4] the same structure as a token sequence
# The tokenizer walks the blocks in a canonical order and emits quantised
# coordinates. This is what the transformer is trained on.
tokens = torch.load(C.TOKENS, weights_only=False)
item = next(
    (it for it in tokens["train"] + tokens["val"] if it["name"] == C.MACHINE),
    tokens["train"][0],
)

C.head("[4] tokens")
print(
    f"   sample {item['name']}, {item['blocks']} blocks, "
    f"{len(item['tokens'])} tokens, vocab {tokens['vocab']}"
)
C.show("tokens[:24]", np.asarray(item["tokens"][:24]), 1)

# %% [5] one forward pass
# Tokens in, next-token logits out, conditioned on the cloud and the block
# count. 40 M parameters, 12 layers, 8 heads.
C.head("[5] forward pass")
x = torch.as_tensor(np.asarray(item["tokens"][:64]))[None]
pc = torch.as_tensor(np.asarray(cloud), dtype=torch.float32)[None]
fc = torch.tensor([float(item["blocks"])])
with torch.no_grad():
    logits = model(x, pc, fc)

C.show("x (tokens)", x, 1)
C.show("pc (cloud)", pc, 0)
C.show("logits", logits, 0)
print(f"   parameters {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")

# %% [6] the generated blocking
# An earlier generation run for this machine; script 03 shows how the sampling
# loop produces it.
C.head("[6] generated blocking")
gen_dir = os.path.join(C.DATA, f"map_batch__{C.MACHINE}")
cmp_path = os.path.join(gen_dir, "compare.vtk")
if os.path.exists(cmp_path):
    with open(cmp_path) as fh:
        L = fh.read().split("\n")
    i = next(k for k, l in enumerate(L) if l.startswith("POINTS"))
    n = int(L[i].split()[1])
    P = np.array([[float(v) for v in L[i + 1 + k].split()] for k in range(n)])
    j = next(k for k, l in enumerate(L) if l.startswith("CELLS"))
    m = int(L[j].split()[1])
    cells = [[int(v) for v in L[j + 1 + k].split()][1:] for k in range(m)]
    s = next(k for k, l in enumerate(L) if l.startswith("SCALARS"))
    part = np.array([int(float(L[s + 2 + k])) for k in range(m)])
    gen = [c for c, p in zip(cells, part) if p == 2 and len(c) == 8]
    used = sorted({int(v) for c in gen for v in c})
    rm = {v: i for i, v in enumerate(used)}
    Cgen = np.stack([P[np.asarray(c, int)] for c in gen])
    print(f"   {len(gen)} generated blocks against {len(B)} ground truth")
    C.write_hexes(
        os.path.join(C.OUT, "06_blocks_generated.vtk"),
        P[used],
        [[rm[int(v)] for v in c] for c in gen],
        {"block_id": np.arange(len(gen))},
        "generated blocking",
    )
else:
    Cgen = None
    print("   no generation artifacts for this machine")

# %% [7] map a blocking onto the geometry
# Corners snap to features, edges are routed along seams or as geodesics on
# their patch, boundary faces are projected. Script 05 opens this up.
from meshtron.geometry.block_mapping import SnapConfigV2, snap_corners_v2  # noqa: E402
from meshtron.geometry.curved_bridge import refill_curved  # noqa: E402
from meshtron.geometry.geometry_features import FeatureModelV2  # noqa: E402
from meshtron.geometry.patch_paths import (
    PatchPaths,
    make_face_projector,  # noqa: E402
    snap_seam_path,
)
from meshtron.geometry.conform import _boundary_edge_pred  # noqa: E402
from scripts.map_generated_blocks import _seam_path_fn  # noqa: E402

C.head("[7] mapping onto the geometry")
fm = FeatureModelV2(npz, cache_dir=os.path.join(C.DATA, "features"))
target = types.SimpleNamespace(
    curves=fm.seam_curves, surface_nearest=fm.surface_nearest
)
corners = fm.vertices[fm.blocks].astype(float)
snapped, records = snap_corners_v2(target, corners, SnapConfigV2())
move = np.linalg.norm(snapped.reshape(-1, 3) - corners.reshape(-1, 3), axis=1)
print(f"   snap moved corners by at most {move.max():.2e}")

stats = {"routes": 0, "edges_surface_projected": 0, "edges_walked_multi_patch": 0}
raw = _seam_path_fn(fm.seam_curves, records, stats, tol=1e-9)


def seam(p0, p1, n):
    r = raw(p0, p1, n)
    return None if r is None else (snap_seam_path(fm.seam_curves, fm, r[0]), r[1])


geo = PatchPaths(
    fm,
    records=records,
    stats=stats,
    is_boundary=_boundary_edge_pred(fm.blocks, snapped),
)


def path_fn(p0, p1, n):
    r = seam(p0, p1, n)
    return r if r is not None else geo(p0, p1, n)


mesh = os.path.join(C.OUT, "07_cfd_mesh.vtk")
rep = refill_curved(
    snapped,
    0.05,
    mesh,
    fm=target,
    path_fn=path_fn,
    write_edges=False,
    face_project_fn=make_face_projector(geo, stats),
)
print(
    f"   seam-routed edges {stats['routes']}, geodesic {stats.get('edges_geodesic', 0)}"
)
print(f"   cells {rep['cells_after']}, watertight {rep['watertight']}")

# %% [8] how good is it
# Two different questions: does the boundary sit ON the geometry, and are the
# cells valid. The first is solved, the second is not.
C.head("[8] quality")
P, H = None, None
with open(mesh) as fh:
    L = fh.read().split("\n")

i = next(k for k, l in enumerate(L) if l.startswith("POINTS"))
n = int(L[i].split()[1])
P = np.array([[float(v) for v in L[i + 1 + k].split()] for k in range(n)])
j = next(k for k, l in enumerate(L) if l.startswith("CELLS"))
m = int(L[j].split()[1])
H = np.array([[int(v) for v in L[j + 1 + k].split()][1:] for k in range(m)])
sb = next(k for k, l in enumerate(L) if l.startswith("SCALARS block_id"))
block_id = np.array([int(float(L[sb + 2 + k])) for k in range(m)])
bids = rep["boundary_point_ids"]
d, _, _ = fm.surface_nearest(P[bids], k=32)
sj = C.scaled_jacobians(P, H)
print(f"   boundary to geometry   max {d.max():.2e}   (gate 1e-3)")
print(
    f"   inverted cells         {int((sj <= 0).sum())} of {len(H)} "
    f"({100 * (sj <= 0).mean():.2f}%)   min scaled Jacobian {sj.min():.3f}"
)
C.write_hexes(
    mesh,
    P,
    H,
    {"block_id": block_id, "scaled_jacobian": sj, "inverted": (sj <= 0).astype(int)},
    "conformed CFD mesh (colour by inverted or scaled_jacobian)",
)
print(
    "\nthe folds sit in the first cell layer at the blade -- script 05 "
    "takes that apart."
)
