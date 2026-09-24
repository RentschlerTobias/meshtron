# meshtron

A transformer that generates hexahedral **block structures** for turbine
passages, and the machinery that maps such a structure onto the real geometry
and fills it into a CFD mesh by transfinite interpolation.

The block structure is the hard part: a coarse decomposition of the flow
passage into curvilinear hexahedra whose faces follow the geometry. Once it
exists, refining it into a mesh of any density is mechanical.

```
parametric geometry          30 cV_ru values per machine
   -> gmsh                   tetrahedral volume mesh
   -> labelled surface       7 patches: inlet, outlet, 2x periodic, hub, shroud, blade hull
   -> conditioning cloud     what the transformer is given
   -> transformer            a block structure, as a token sequence
   -> mapping                corners snapped, edges routed, faces projected
   -> transfinite refill     the CFD mesh
```

## Layout

```
meshtron/
  geometry/   feature model, seam curves, edge routing, transfinite refill
  model/      GPTCond over block tokens, Quadtron, their encoders
  data/       tokenizers, conditioning clouds, datasets, augmentation
  training/   supervised and GRPO training, generation, rewards, the 2D stack
  viz/        plotting, tokenisation animations, terminal UI
  legacy/     earlier generations, kept for reference
scripts/      one-shot tooling: dataset builds, batch runs, diagnostics, gates
showcase/     a five-script walkthrough of the whole repo
docs/         decisions, proposals, notes
```

`meshtron/__init__.py` puts every subpackage directory on `sys.path` so that
modules still using bare imports keep working. New code should use the full
path: `from meshtron.geometry import patch_paths`.

## Start here

```
uv run python showcase/01_overview.py
```

Eight cells, the whole chain on one machine, each printing what it produced and
writing a VTK. `showcase/02` to `05` open one box each -- the data and the
tokenizer, the model, the training loop, the mapping. They are written as
`# %%` cells so they can be stepped through from an editor into a REPL. See
`showcase/README.md`.

## The two model families

**3D block structures** — the active path. `GPTCond`
(`meshtron/model/gpt_cond.py`) is a decoder-only transformer over quantised
block-structure tokens, conditioned on a surface point cloud and the block
count: both are encoded into one vector per sample and applied to every token
position as a FiLM scale and shift. 40 M parameters, 12 layers, 8 heads.
Trained by `meshtron/training/train_hexarow_full.py`, refined with GRPO by
`train_grpo.py`, sampled by `generate.py` under a structural mask that makes a
syntactically broken sequence impossible.

**2D quads (Quadtron)** — `meshtron/model/quadtron.py` with
`meshtron/training/trainer.py`. Older, and it does not yet have the coordinate
slot embedding or the polar conditioning the 3D path gained. Bringing it up to
the same footing is open work.

## Mapping a block structure onto geometry

```
uv run python scripts/conform_gt_blocks.py \
    --target-h 0.05 --geodesic --project-faces \
    --out-dir data/features_debug/run
```

Corners snap onto seam junctions, seam curves or patches. Every boundary edge is
routed: along a feature curve where its ends sit on one, otherwise as a shortest
path on the patch it belongs to -- which is why the blade footprint, being a
hole in the hub patch, is walked *around* rather than cut through. Boundary face
interiors are projected onto their patch, and Gordon-Hall fills the volume.

Batch it with `scripts/run_conform_batch.py`, and check a blocking before
mapping it with `scripts/detect_block_tjunctions.py`.

## Where it stands

Conformity is solved. Over the whole corpus at h=0.05, 679 of 680 samples pass
a gate of 1e-3, with a median boundary error of 1.26e-10.

Cell validity is not. Median 0.42% inverted cells, and they are introduced by
the mapping rather than inherited: the same corners refilled with the
blocking's own edge polylines give 9 folded cells where the routed ones give
218, all in the first cell layer at the blade. `scripts/make_inverted_debug.py`
writes both meshes side by side.

Two further things worth knowing before trusting a number here:

- **"Sits on the geometry" and "covers the geometry" are different questions.**
  A blocking that fails to wrap the blade scores 1e-10 on the first and leaves
  20% of the blade uncovered on the second. `scripts/make_pipeline_demo.py`
  reports both.
- **17% of the corpus carries block-level T-junctions**: two blocks touching
  across only part of a side, which the four-corner face format cannot express,
  leaving an unmeshed slit. The AlgoHex mesh underneath is conforming, so
  regenerating would reproduce it.

`docs/decisions/2026-09-23-blocking-geometry-mapping.md` carries every
measurement behind those statements.

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

`uv run` for everything; the system Python lacks scipy. One RTX 4060 with 8 GB,
runs are serial. `data/` is not versioned.
