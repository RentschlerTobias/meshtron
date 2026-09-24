"""02 -- data, low level: npz to tokens and back.

Every transformation the data goes through before the model sees it, with the
arrays in between. Run cell by cell; each one is meant to be poked at in the
REPL afterwards.
"""
# %% [0] setup
import os
import sys

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

npz_path = os.path.join(C.BATCH, C.MACHINE, "sample.npz")
z = np.load(npz_path, allow_pickle=True)
C.head("[0] what one sample.npz holds")
for k in z.files:
    C.show(k, z[k], 0)

# %% [1] the block structure, decoded
# blocks[i] are eight indices into vertices, in VTK hex corner order: bottom
# face 0-1-2-3 as a cycle, top face 4-5-6-7 the same way round.
C.head("[1] blocks")
V = np.asarray(z["vertices"], float)
B = np.asarray(z["blocks"], np.int64)
C.show("vertices", V, 3)
C.show("blocks", B, 3)
print("\nblock 0, its eight corners:")
for slot, vid in enumerate(B[0]):
    p = V[vid]
    print(f"   slot {slot}  vertex {vid:3d}  xyz {p}  "
          f"r {np.hypot(p[0], p[1]):.4f}  phi {np.arctan2(p[1], p[0]):+.4f}  "
          f"z {p[2]:.4f}")

# %% [2] the block edges carry their own curve
# A block edge is not a straight line: edge_polyline holds the polyline, and
# refilling with these instead of chords is the difference between a valid and
# an invalid mesh.
C.head("[2] edge curvature")
E = np.asarray(z["edges"])
EP = np.asarray(z["edge_polyline"], float)
OFF = np.asarray(z["edge_polyline_offset"], np.int64)
C.show("edges", E, 3)
C.show("edge_polyline", EP, 2)
lens = np.diff(OFF)
print(f"   {len(E)} directed edges, polyline lengths "
      f"{lens.min()}..{lens.max()} points")


def sagitta(Q):
    Q = np.asarray(Q, float)
    c = Q[-1] - Q[0]
    L = np.linalg.norm(c)
    if L < 1e-12:
        return 0.0
    u = c / L
    d = Q - Q[0]
    return float(np.linalg.norm(d - np.outer(d @ u, u), axis=1).max())


sag = np.array([sagitta(EP[OFF[k]:OFF[k + 1]]) for k in range(len(E))])
print(f"   bow off the chord: p50 {np.median(sag):.4f}  max {sag.max():.4f}")

# %% [3] the cleaned training sample
# clean_base_npz.py flips negatively oriented blocks, drops degenerate ones and
# converts to the polytron sample format. Note what it does NOT check: folded
# cells, which have positive volume and negative corner Jacobians.
C.head("[3] cleaned sample")
src = torch.load(C.SRC, weights_only=False)
sample = next(s for s in src["samples"] if s["name"] == C.MACHINE)
for k in sorted(sample):
    C.show(k, sample[k], 0)

# %% [4] the conditioning cloud
# build_cloud draws n_points from the surface in (r, theta, z), normalises them
# into the training bounds and appends a blade flag. This is the model's only
# view of the geometry.
from meshtron.data import conditioning  # noqa: E402

C.head("[4] conditioning cloud")
ck, cfg, coords, npt, rb, zb, model, max_len, _ = C.load_model()
rng = np.random.default_rng(0)
cloud = conditioning.build_cloud(sample, cfg["n_points"], rb, zb, rng)
cloud = cloud[0] if isinstance(cloud, tuple) else cloud
C.show("cloud", cloud, 4)
print("   columns are (r', sin(theta), cos(theta), z') -- the angle enters as a"
      "\n   sin/cos pair so the seam at +-pi is not a jump, and r and z are"
      "\n   normalised into the bounds stored in the checkpoint.")
xyz = np.stack([cloud[:, 0] * (rb[1] - rb[0]) + rb[0], cloud[:, 1],
                cloud[:, 3] * (zb[1] - zb[0]) + zb[0]], axis=-1)
xyz = np.stack([xyz[:, 0] * cloud[:, 2], xyz[:, 0] * cloud[:, 1], xyz[:, 2]],
               axis=-1)
C.write_points(os.path.join(C.outdir(), "12_cloud.vtk"), xyz,
               {"r_norm": cloud[:, 0], "z_norm": cloud[:, 3]},
               "conditioning cloud, back in xyz")

# %% [5] tokenisation
# HexaRowTokenizer walks the blocks in a canonical order and emits, per corner,
# npt quantised coordinates. Same structure in, same sequence out.
from meshtron.data.hexa_row_tokenizer import HexaRowTokenizer  # noqa: E402

C.head("[5] tokenizer")
tok = HexaRowTokenizer(r_bounds=rb, z_bounds=zb)
print(f"   quantisation Qr={tok.Qr} Qa={tok.Qa}, npt={npt}, coords={coords}")
tokens_file = torch.load(C.TOKENS, weights_only=False)
item = next((it for it in tokens_file["train"] + tokens_file["val"]
             if it["name"] == C.MACHINE), tokens_file["train"][0])
seq = np.asarray(item["tokens"])
C.show("token sequence", seq, 1)
print(f"   {len(seq)} tokens for {item['blocks']} blocks "
      f"= {len(seq) / item['blocks']:.1f} per block")
vals, cnt = np.unique(seq, return_counts=True)
print(f"   {len(vals)} distinct ids, most frequent "
      f"{vals[cnt.argmax()]} appears {cnt.max()} times (separator)")

# %% [6] round trip
# Detokenising the sequence must give the blocks back. This is the contract the
# whole training rests on.
C.head("[6] detokenise")
vpt, blk = tok.detokenize(list(map(int, seq)), coords=coords)
C.show("vertices back", vpt, 2)
C.show("blocks back", blk, 2)
print(f"   {len(blk)} blocks recovered, original {len(B)}")

# %% [7] how the corpus is split
# By geometry, so the same machine never appears on both sides.
C.head("[7] dataset")
print(f"   train {len(tokens_file['train'])}  val {len(tokens_file['val'])}")
lens = np.array([len(it["tokens"]) for it in tokens_file["train"]])
blocks = np.array([it["blocks"] for it in tokens_file["train"]])
print(f"   sequence length p50 {int(np.median(lens))} max {lens.max()}")
print(f"   blocks per sample p50 {int(np.median(blocks))} max {blocks.max()}")
names = {it["name"].rsplit("_n", 1)[0] for it in tokens_file["train"]}
vnames = {it["name"].rsplit("_n", 1)[0] for it in tokens_file["val"]}
print(f"   machines: {len(names)} train, {len(vnames)} val, "
      f"{len(names & vnames)} shared")
