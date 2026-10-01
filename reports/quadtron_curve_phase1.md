# Quadtron curve stage — Phase 1: transfer baseline

Date: 2026-10-01
Branch: `polytron-main` (un-pushed `polytron` commit `95b0364` revived from the
NAS bundle and merged with `main`)

## Question

Quadtron (Tokenizer2D) emits straight-edged blocks; the edge shape is what
decides whether the transfinite fill folds (`meshtron/data/polytron_blocks.py`,
module docstring). The Polytron `CurveModel` predicts per-block-edge cubic
Bezier curves. Phase 1 asks: **does the already-trained Polytron curve head
transfer to Quadtron structures, without any retraining?**

## Setup

- Revived the un-pushed `polytron` branch (`95b0364`, "PolyGen-style curvilinear
  hex blocks, from cloud to TFI mesh", 2026-09-25) from
  `…/archive/repo-bundles/hydrostack_pipeline__stack__meshtron.bundle`. It adds
  `meshtron/model/polytron.py` (`CurveModel`), `meshtron/data/polytron_blocks.py`,
  `meshtron/training/{train_polytron,polytron_sample}.py`,
  `meshtron/geometry/polytron_tfi.py` and the `scripts/*polytron*` entry points.
- Edge head: `data/polytron_clean/polytron_curve_best.pt` (epoch 30, val
  cross-entropy 3.1022). It is cloud-conditioned: `CloudEncoder` + vertex
  encoder cross-attend the condition cloud, and the edge transformer does too.
  Per **undirected** block edge it predicts 6 quantised Bezier offsets
  (`q_curve=256`, mu-law companded) as classification, not regression.
- Structures: `data/polytron_blocks_clean.pt` (407 train / 46 val, same
  geometry-disjoint split as the Quadtron family tokens).
- TFI fill: `scripts/eval_polytron.py`, `--target-h 0.08`, projection on.

## Results

Curve metric over the 46 val items (5 292 edges), `scripts/eval_quadtron_curve.py`:

| mode | mean bin error | exact share | curve rel. median | mean | p90 | max |
|---|---|---|---|---|---|---|
| `gt` (Polytron structures) | 10.84 | 12.8 % | 1.55 % | 3.27 % | 8.3 % | 49 % |
| `quadtron` (256-bin, per-mesh) | 10.89 | 12.7 % | 1.55 % | 3.28 % | 8.0 % | 57 % |

`quadtron` pushes every structure through Quadtron's coordinate representation
(per-column min/max → 256 bins → back, `tokenizer_v2.Tokenizer2D._quantize_coords`).

End-to-end TFI fill over the same 46 val items:

| run | watertight | inverted median | uncovered median | on-surface median |
|---|---|---|---|---|
| oracle (GT curves, no model) | 100 % | 0.077 % | 0.0 % | 1.2e-8 |
| curve head on GT topology | 100 % | 0.386 % | 11.6 % | 1.5e-8 |

`inverted_share_median` is inverted cells / cells of the chosen candidate;
`uncovered_share_median` is the share of the surface the blocking does not wrap.

## Reading

- The curve head transfers to Quadtron's coordinate representation with **no
  measurable loss** (bin error 10.84 → 10.89, curve median 1.55 % → 1.55 %).
  The head is not sensitive to the per-mesh quantisation Quadtron applies.
- On GT topology the head reaches 100 % watertight meshes with a median
  **0.39 %** inverted cells, ~5× the oracle's 0.077 %. So the curve head is
  usable but leaves a real gap to GT curves.
- The **11.6 % uncovered** surface is the bigger effect: predicted curves hug
  the geometry less tightly than the GT curves, so more of the blade surface is
  left outside the blocking. This is the number to move in Phase 2.
- No 3D Quadtron checkpoint exists (only 2D RL-curriculum checkpoints), so
  "Quadtron structures" here are the ground-truth structures Quadtron is trained
  to reproduce. The generated-structure leg is gated on training a 3D Quadtron.

## Reproduce

```bash
uv run python scripts/eval_quadtron_curve.py --mode gt       --n 46
uv run python scripts/eval_quadtron_curve.py --mode quadtron --quantization 256 --n 46

export PYTHONPATH=…/domain_partition_3D   # for the external TFI modules
uv run python scripts/eval_polytron.py --split val --n 46 --oracle --no-vtk \
    --out-dir data/polytron_eval_oracle46
uv run python scripts/eval_polytron.py --split val --n 46 --k 1 --teacher blocks --no-vtk \
    --out-dir data/polytron_eval_curve_gt46 \
    --vertex-ckpt data/polytron_clean/polytron_vertex_last.pt \
    --block-ckpt  data/polytron_clean/polytron_block_best.pt \
    --curve-ckpt  data/polytron_clean/polytron_curve_best.pt
```

## Next — Phase 2

Train a Quadtron-specific curve head and compare on the same items and metrics:
- `train_polytron.py --stage curve` on `data/polytron_blocks_clean.pt`, then
  evaluate on Quadtron's decoded structures.
- Optional ablation: the same head without the cloud cross-attention, to isolate
  how much of the uncovered-share gap the conditioning closes.
- Generated-structure leg: train a 3D Quadtron
  (`train.py --model-family quadtron --dim 3 --data-path data/quadtron_data_3d_full_aug.pt`)
  so the curve head can be fed sampled structures, not only GT.
