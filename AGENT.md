# AGENT.md

Guide for an AI agent working in this repo. Read `README.md` first for the
what and why; this file is the how and the traps.

## What

Two networks and a mesher, for turbomachinery flow passages:

1. **The generator, `GPTCond`** — a decoder-only, causal transformer over
   HexaRow tokens of a hexahedral block structure, conditioned on a surface
   point cloud and the block count (FiLM: feature-wise linear modulation, one
   condition vector scales and shifts every channel before the stack).
   Supervised training (SFT), then GRPO (group relative policy optimisation,
   reinforcement learning on geometric rewards).
2. **The edge network, `CurveModel`** — non-autoregressive; per block edge the
   two inner control points of a cubic Bézier curve.
3. **The mesher** — maps a block structure onto the geometry and fills it by
   transfinite interpolation (TFI) into a conforming hex CFD mesh.

The training data comes from `../domain_partition_3D` (AlgoHex route, one
`sample.npz` per machine).

## Environment

- `uv run python …` for everything; the system Python lacks scipy.
- One RTX 4060, 8 GB — runs are serial, never two GPU jobs at once.
- `data/` is not versioned: datasets, token files and checkpoints live there.
  Production checkpoints: `data/grpo_cart_step300.pt` (generator, Cartesian),
  `data/hexa_curve_best.pt` (edge network).
- Scripts that read hex3d code need `PYTHONPATH=…/domain_partition_3D`.

## Where things are (active paths only)

| stage | code |
|---|---|
| tokens | `meshtron/data/hexa_row_tokenizer.py` (`coords='cart'`: 3 tokens per corner, `'polar'`: 4) |
| conditioning cloud | `meshtron/data/conditioning.py` |
| generator | `meshtron/model/gpt_cond.py` |
| SFT / GRPO | `meshtron/training/train_hexarow_full.py`, `train_grpo.py`, rewards in `rewards_hexarow.py` |
| sampling + slot mask | `meshtron/training/generate.py` (`slot_mask`, `generate`) |
| edge network | `meshtron/model/polytron.py` (`CurveModel`), data `meshtron/data/polytron_blocks.py`, training `python -m meshtron.training.train_polytron --stage curve` |
| non-learned mapping | `meshtron/geometry/conform.py` (`ConformOptions`) |
| curved refill + projection | `meshtron/geometry/curved_bridge.py`, `meshtron/geometry/polytron_tfi.py` |
| inference | `scripts/infer_gptcond_curve.py` (generator + edge network), `scripts/infer.py` (generator + non-learned mapping) |
| walkthrough | `showcase/01`–`06` |

`meshtron/legacy/` and `deprecated/` are not on any active path;
`scripts/repo_inventory.py` shows what still hangs off the active ones.
`meshtron/__init__.py` puts every subpackage directory on `sys.path` so old
bare imports keep working — new code uses full paths
(`from meshtron.geometry import patch_paths`).

## Gates — must stay exit 0

```
uv run python scripts/test_slot_parity.py
uv run python scripts/smoke_mesh_validation.py
uv run python scripts/test_conditioning_parity.py
uv run python scripts/test_rewards_hexarow.py
uv run python scripts/test_block_mapping.py
uv run python scripts/test_training_e2e.py      # four phases, ~2 min
uv run python scripts/verify_pipeline.py        # 12 checks
```

Run the gate that covers what you touched before claiming it works.

## Conventions

- Everything written into files is **English**: code, comments, docstrings,
  docs, commit messages. Older modules still carry German comments; leave them
  unless you are rewriting that code anyway.
- Measured results go to `reports/<study>.md`; decisions, with the rejected
  options and their numbers, to `docs/decisions/<date>-<topic>.md`. Both are
  records: do not rewrite them later, write a new one.
- Name a number with what it was measured on (split, h, rollouts, checkpoint).

## Traps

- **Slot convention.** Training feeds each position the slot of the token it
  *carries*; inference must do the same. A mismatch once gave a perfect loss
  curve and gibberish generation. `test_slot_parity.py` guards it.
- **Cartesian vs polar.** The production generator is Cartesian (`npt=3`);
  the slot mask, detokenisation and the slot embedding all depend on `npt`.
  Polar (`npt=4`) checkpoints exist — check `cfg['coords']` before decoding.
- **Old checkpoints.** The one in `showcase/_common.py` predates the slot
  embedding (no `npt`/`coords` in its `cfg`): feed it `slot=None`, or it loses
  ~30 points of accuracy.
- **KL estimator.** `--kl-estimator` defaults to `k3`; the existing GRPO
  checkpoints were trained with `naive`, which reproduces them.
- **The slot mask guarantees grammar, not geometry.** Every stream terminates
  and decodes; whether the hexahedra are valid is what the GRPO reward and the
  mesh validation measure.
- **"On the geometry" ≠ "covers the geometry".** Boundary points can sit on the
  surface to 1e-10 while 20 % of the blade is not covered. Report both
  (`polytron_tfi.surface_fit`: `on_surface_*` and `uncovered_share`).
- **The refill covers the core passage only.** The blade O-grid and the
  hub/shroud boundary layer are not re-inserted by any meshtron chain; that
  lives in `domain_partition_3D` (`reattach.py`, `ogrid_extrude.py`).
- **Block-level T-junctions** (17 % of the corpus) leave unmeshed slits; the
  four-corner face format cannot express them.
- **The surface projection after the curves is a trade.** It lowers the
  uncovered share a little and raises the inverted share
  (`reports/quadtron_curve_phase2.md`, projection on/off).
