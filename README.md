# meshtron

A transformer that generates hexahedral **block structures** for turbine
passages, a second network that bends the generated block edges onto the
geometry, and the machinery that fills such a structure into a CFD mesh by
transfinite interpolation.

The block structure is the hard part: a coarse decomposition of the flow
passage into curvilinear hexahedra whose faces follow the geometry. Once it
exists, refining it into a mesh of any density is mechanical. The training
data — minimal block structures for Sobol-sampled machines — comes from the
AlgoHex route in [`domain_partition_3D`](../domain_partition_3D).

Abbreviations used below:

| | |
|---|---|
| CFD | computational fluid dynamics |
| TFI | transfinite interpolation — fills a block with cells from its boundary curves and faces (Coons patches, Gordon–Hall) |
| dtOO | our parametric turbomachinery design system |
| GPT | generative pre-trained transformer — here: a decoder-only, causal transformer |
| GPTCond | our GPT conditioned on the geometry and the block count |
| HexaRow | our row-encoded tokenizer for hexahedral block structures |
| MLP | multi-layer perceptron |
| FiLM | feature-wise linear modulation — a condition vector scales and shifts every channel: `h ← h·(1+tanh c) + c` |
| SFT | supervised fine-tuning — here the first, supervised training stage |
| GRPO | group relative policy optimisation — the reinforcement-learning stage: several rollouts per geometry, each rewarded relative to its group |
| KL | Kullback–Leibler divergence, the anchor that keeps GRPO close to the supervised model |
| VTK / npz | ParaView mesh files / NumPy archives (one `sample.npz` per machine) |
| val | the validation split: geometries the models never trained on |

## The pipeline at a glance

![The pipeline in five steps](docs/images/polytron/00_pipeline_steps.png)

1. **Geometry surface** — the same dtOO geometry the numerical route starts
   from, as a labelled surface (7 patches).
2. **Surface point cloud** — 1000 points sampled from that surface, the whole
   conditioning signal.
3. **Block structure** — `GPTCond` generates it token by token, with straight
   edges: the corners are exact, the edges are chords.
4. **Curved block edges** — the edge network (`CurveModel`) turns every block
   edge into a cubic Bézier curve on the geometry (dashed: the straight
   chords).
5. **CFD mesh** — every block is filled by curved TFI; neighbouring blocks
   agree on their shared faces, so the hex mesh is conforming.

Steps 3–5 show the dataset structure of base_a as the stand-in for a
generated one — the exact object the models are trained to produce.

## The model

![The model: generator and edge network](docs/images/polytron/00_model.png)

**The generator, `GPTCond`** (`meshtron/model/gpt_cond.py`), 46.9 M
parameters, 12 layers, 8 heads, width 512.

- **Tokens.** HexaRow writes a block structure as rows of blocks: the first
  block of a row emits all 8 corners (entry and exit ring), each following
  block that shares a face only its 4 exit corners; `EOR` closes a row, `STOP`
  the structure. Cartesian quantisation, 512 bins, 3 tokens per corner
  (x, y, z).
- **Embedding.** Each token is the sum of three learned embeddings: its value,
  its absolute position and its slot (which of x, y, z it carries).
- **Conditioning.** The point encoder lifts every point with an MLP, lets 16
  learned queries cross-attend into the set and pools them into one vector;
  the block count goes through its own MLP. The two vectors add into one
  condition vector, applied once by FiLM before the decoder stack — the
  geometry sets a scale and a shift per channel, and the residual stream
  carries it through all 12 layers.
- **Slot mask** (`slot_mask` in `meshtron/training/generate.py`). Before
  every sampling step, every token that cannot legally stand at this position
  gets −∞: coordinates only from the 512 coordinate bins, `EOR`/`STOP` only
  when a row closes on a whole ring of 4 corners (the first row needs both
  rings), after `STOP` only `END`. Every generated stream is grammatical and
  terminates. What the mask cannot guarantee is geometric validity — that is
  what the GRPO reward is spent on.

**The edge network, `CurveModel`** (`meshtron/model/polytron.py`), trained
separately.

- **Input:** the generated corners and blocks, the block count and the point
  cloud (xyz with Fourier features and the patch label).
- The cloud is encoded into 128 latent vectors (Perceiver-style
  cross-attention) and kept as a set, so an edge can look up *where* on the
  geometry it lies; a bidirectional encoder runs over the corners.
- Each undirected block edge is described by its two corner features, their
  product, its chord direction and length; 6 layers let all edges attend to
  each other and to the cloud at once — **not autoregressive**.
- **Output:** per edge the two inner control points of a cubic Bézier curve,
  as offsets from the chord's third points in chord lengths, μ-law companded
  into 256 bins — a classification (cross-entropy in training, argmax at
  inference). Offset 0 is exactly the straight edge.

On 20 val geometries (`reports/quadtron_curve_phase2.md`):

| fill of the generated blocks | geometry surface not reached | inverted cells |
|---|---|---|
| straight edges | 43.5 % | 0.49 % |
| non-learned back-mapping (snap corners, route edges along seams) | 43.4 % | 0.62 % |
| **edge network** | **25.6 %** | **0.29 %** |
| ground-truth blocks and curves (ceiling) | 0.0 % | 0.04 % |

"Not reached" is the share of geometry surface triangles farther than 0.02
from the mesh boundary. The remaining quarter is the placement of the
generated blocks, which no edge stage can repair.

## Start here

```
uv run python showcase/01_overview.py
```

Eight cells, the whole chain on one machine, each printing what it produced and
writing a VTK. `showcase/02` to `05` open one box each — the data and the
tokenizer, the model, the training loop, the mapping. They are written as
`# %%` cells so they can be stepped through from an editor into a REPL. See
`showcase/README.md`.

## Inference

Generator plus learned edge network, with the straight and the ground-truth
fills next to it for comparison:

```
export PYTHONPATH=…/domain_partition_3D
uv run python scripts/infer_gptcond_curve.py --split val --n 20 --k 3 \
    --curve-ckpt data/hexa_curve_best.pt
```

Generator plus the non-learned mapping, from the geometry alone — labelled
surface, conditioning cloud, generated blocking, snapping, edge routing, TFI —
one VTK per stage (`01_geometry` … `05_cfd_refill`) and a `report.json` with
every number of every stage:

```
uv run python scripts/infer.py \
    --npz data/hex3d_algohex/batch/machine_0034_n2000/sample.npz \
    --ckpt data/grpo_cart_step300.pt --blocks 12 --k 8
```

`--blocks` is an input, not a derived quantity: the block count conditions the
model and the geometry does not carry it. `--blocks-sweep 12,16,20` tries
several and keeps the best mesh. `--k` is the number of rollouts.

On four held-out geometries, at cell size h=0.08 with 4 rollouts each:

| geometry | blocks | cells | watertight | boundary max | inverted |
|---|---|---|---|---|---|
| machine_0034_n2000 | 12 | 12660 | yes | 2.4e-10 | 1.30% |
| machine_0005_n2000 | 22 | 12165 | yes | 8.5e-11 | 1.59% |
| machine_0026_n2000 | 12 | 11985 | yes | 7.7e-11 | 4.79% |
| machine_0034_n8000 | 16 | 12390 | yes | 3.1e-10 | 2.62% |

## Training

The generator: supervised (SFT) by `meshtron/training/train_hexarow_full.py`,
then GRPO by `train_grpo.py`; the production checkpoint is
`data/grpo_cart_step300.pt`.

The edge network, on the curves of the same dataset structures (683 train /
78 val):

```
uv run python scripts/build_hexa_curve_dataset.py
uv run python -m meshtron.training.train_polytron --stage curve \
    --data data/hexa_curve_blocks.pt --out data/hexa_curve \
    --epochs 300 --bs 16 --lr 3e-4 --d 256 --heads 8 --layers 6 \
    --dropout 0.1 --n-points 2048 --eval-every 10 --weight-label 5
```

End-to-end check of the training path:

```
uv run python scripts/test_training_e2e.py        # four phases, ~2 min
```

Supervised run, resume, real GRPO steps, then inference with the checkpoint it
just trained. Each phase asserts the claim its stage has to make — the loss
falls, the resumed run continues instead of restarting, the best-val checkpoint
loads and not merely exists, and GRPO does not move the policy when no reward
says to. It runs at width 128 (1.3 M parameters) because it tests the path.

`scripts/verify_pipeline.py` covers the rest: tokenisation in 2D and 3D,
Cartesian and polar, a real forward and training step for both model families,
the GRPO entry point, the mapping, and inference end to end. Its first run
found four real defects, among them a KL anchor that was the plain log-ratio
mean rather than a divergence; `--kl-estimator` now defaults to `k3` (the
unbiased, always non-negative estimator), `naive` reproduces the existing
checkpoints. `docs/decisions/2026-09-24-training-end-to-end.md` has the
measurements.

## Model families

- **GPTCond + HexaRow** — the active generator, above. The reports call this
  pairing the "3D Quadtron".
- **Polytron** (`meshtron/model/polytron.py`, `scripts/infer_polytron.py`) —
  the PolyGen recipe lifted to hexahedra: a vertex model, a pointer-network
  block model and the curve model, chained at inference. Its curve model is the
  edge network used above.
- **2D Quadtron** (`meshtron/model/quadtron.py`, `meshtron/training/trainer.py`)
  — the older quad-mesh model. It does not yet have the slot embedding or the
  polar conditioning the 3D path gained.

## Mapping a block structure onto geometry (non-learned)

```
uv run python scripts/conform_gt_blocks.py \
    --target-h 0.05 --geodesic --project-faces \
    --out-dir data/features_debug/run
```

Corners snap onto seam junctions, seam curves or patches. Every boundary edge
is routed: along a feature curve where its ends sit on one, otherwise as a
shortest path on the patch it belongs to — which is why the blade footprint,
being a hole in the hub patch, is walked *around* rather than cut through.
Boundary face interiors are projected onto their patch, and Gordon–Hall fills
the volume. The code lives in `meshtron/geometry/conform.py`, shared by this
gate and `scripts/infer.py`; `ConformOptions` documents which knobs are off by
default because measurement said so. Batch it with
`scripts/run_conform_batch.py`, check a blocking first with
`scripts/detect_block_tjunctions.py`.

## Where it stands

- **Conformity is solved.** Over the whole corpus at h=0.05, 679 of 680
  samples pass a gate of 1e-3, median boundary error 1.26e-10.
- **Coverage is the open gap.** With the edge network a generated structure
  still leaves about a quarter of the geometry surface unreached; the cause is
  where the generator places the blocks, not the edges.
- **Cell validity** of the non-learned mapping: median 0.42 % inverted cells,
  introduced by the edge routing rather than inherited — the same corners
  refilled with the blocking's own edge polylines give 9 folded cells where the
  routed ones give 218, all in the first cell layer at the blade
  (`scripts/make_inverted_debug.py` writes both).
- **"Sits on the geometry" and "covers the geometry" are different
  questions.** A blocking that fails to wrap the blade scores 1e-10 on the
  first and leaves 20 % of the blade uncovered on the second;
  `scripts/make_pipeline_demo.py` reports both.
- **17 % of the corpus carries block-level T-junctions**: two blocks touching
  across only part of a side, which the four-corner face format cannot
  express, leaving an unmeshed slit.

`docs/decisions/2026-09-23-blocking-geometry-mapping.md` carries the
measurements behind the mapping statements.

## Layout

```
meshtron/
  geometry/   feature model, seam curves, edge routing, transfinite refill
  model/      GPTCond, Polytron (incl. the edge network), Quadtron, encoders
  data/       tokenizers, conditioning clouds, datasets, augmentation
  training/   supervised and GRPO training, generation, rewards, the 2D stack
  viz/        plotting, tokenisation animations, terminal UI
  legacy/     earlier generations, kept for reference
scripts/      one-shot tooling: dataset builds, batch runs, diagnostics, gates
showcase/     a five-script walkthrough of the whole repo
reports/      measured results, one file per study
docs/         decisions, proposals, notes, README figures
```

`meshtron/__init__.py` puts every subpackage directory on `sys.path` so that
modules still using bare imports keep working. New code should use the full
path: `from meshtron.geometry import patch_paths`.

## Gates

Five checks that have to stay exit 0:

```
uv run python scripts/test_slot_parity.py
uv run python scripts/smoke_mesh_validation.py
uv run python scripts/test_conditioning_parity.py
uv run python scripts/test_rewards_hexarow.py
uv run python scripts/test_block_mapping.py
```

`scripts/repo_inventory.py` shows which modules still hang off the active
paths.

## Environment

`uv run` for everything; the system Python lacks scipy. One RTX 4060 with
8 GB, runs are serial. `data/` is not versioned.
