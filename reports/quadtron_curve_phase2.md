# Quadtron curve stage — Phase 2: Quadtron-specific head

Date: 2026-10-01
Branch: `polytron-main`
Continues `reports/quadtron_curve_phase1.md`.

## What Quadtron actually emits (this changes the target)

Quadtron 3D is `Tokenizer2D(dim=3)`, so it models **4-corner faces with 3
coordinates** — 12 tokens per face. It cannot represent a hexahedron (8
corners). `meshtron/data/domain_extractor_3d.build_quadtron_sample` therefore
feeds it `faces = quad_faces.T`, which `sample.npz` defines as the **boundary
quad faces** of the hex block structure (verified on `machine_0009_n2000`:
`quad_faces` == the 64 faces owned by exactly one block; the 22 blocks have 98
unique faces, 33 interior).

So Quadtron's edge stage must curve the **boundary shell**, not the full
hex-block edge set Polytron's `CurveModel` was trained on.

## Setup

- `scripts/build_quadtron_curve_dataset.py` → `data/quadtron_curve_blocks.pt`:
  407 train / 46 val, spec fitted from the sample vertices, `edges` = unique
  undirected edges of the boundary quad faces (80–144 per sample), `cq` = GT
  control-point offsets (every edge has a record). Same geometry-disjoint split
  as the Polytron dataset, by `dir`.
- Head: `train_polytron.py --stage curve` on that dataset, same architecture as
  Phase 1 (13.3 M params). Best val at epoch 30 (3.2499); trains to accuracy 1.0
  by epoch 300, val flat at ~0.26 — the documented overfit regime.

## Results — direct curve metric, 46 val items

`scripts/eval_quadtron_curve.py`. `median/mean/p90` are max chord-relative
distance between predicted and GT cubic.

| data (edge set) | head | mean bin | exact | median | mean | p90 |
|---|---|---|---|---|---|---|
| full block edges | Polytron (transfer) | 10.84 | 12.8 % | 1.55 % | 3.27 % | 8.3 % |
| full block edges | Quadtron-specific (boundary-trained) | 16.77 | 14.0 % | 2.60 % | 5.01 % | 13.6 % |
| boundary edges | Polytron (transfer) | 17.96 | 10.3 % | 2.91 % | 5.50 % | 14.6 % |
| boundary edges | **Quadtron-specific, ep30 best** | **13.71** | **14.2 %** | **1.52 %** | **4.42 %** | **13.3 %** |
| boundary edges | Quadtron-specific, ep300 last | 13.93 | 13.1 % | 1.82 % | 4.50 % | 12.6 % |

## Reading

- On Quadtron's own boundary edges the Quadtron-specific head **halves the
  median curve error** against the transferred Polytron head (2.91 % → 1.52 %)
  and cuts the mean bin error by 24 % (17.96 → 13.71). Training on the right
  structure matters.
- Boundary edges are genuinely harder than interior edges: the same
  transferred head is at 1.55 % median on full block edges but 2.91 % on the
  boundary subset.
- The Quadtron-specific head does **not** transfer back to the full block edge
  set: applied to all block edges it is 2.60 % median with p90 13.6 %
  (vs 1.55 % / 8.3 % for the Polytron head). Interior edges are out of its
  training distribution.
- The tail (p90 ≈ 13 %) is unchanged in every run: the hard edges are hard for
  both heads, and no head is near the oracle's boundary behaviour.

## Consequence for the fill

Quadtron's output (boundary shell) does not contain the interior edges a
transfinite fill needs, so the Phase 2 head cannot be scored by TFI inverted
cells the way Phase 1 was. Phase 1 remains the fill-relevant number:

| run | watertight | inverted median | uncovered median |
|---|---|---|---|
| oracle (GT curves) | 100 % | 0.077 % | 0.0 % |
| Polytron head on GT block topology | 100 % | 0.386 % | 11.6 % |

## Open decision

1. **Boundary-shell only** — the Phase 2 head is the complete answer for what
   Quadtron emits; interior topology comes from the mapping/assembly side.
2. **Full fill** — the curve head must be trained on all block edges (i.e. the
   Polytron Phase-1 head), or Quadtron must be changed to emit hex blocks, which
   `Tokenizer2D(dim=3)` cannot do (12 tokens/face = 4 corners).
3. **Next experiment if (1)** — add the cloud ablation (same head without
   cloud cross-attention) to see how much of the 11.6 % uncovered share the
   conditioning closes.

## Reproduce

```bash
uv run python scripts/build_quadtron_curve_dataset.py
uv run python -m meshtron.training.train_polytron --stage curve \
    --data data/quadtron_curve_blocks.pt --out data/quadtron_curve \
    --epochs 300 --bs 16 --lr 3e-4 --d 256 --heads 8 --layers 6 \
    --dropout 0.1 --n-points 2048 --eval-every 10 --weight-label 5

uv run python scripts/eval_quadtron_curve.py --data data/quadtron_curve_blocks.pt \
    --curve-ckpt data/quadtron_curve_best.pt --mode gt --n 46
uv run python scripts/eval_quadtron_curve.py --data data/quadtron_curve_blocks.pt \
    --curve-ckpt data/polytron_clean/polytron_curve_best.pt --mode gt --n 46   # transfer
uv run python scripts/eval_quadtron_curve.py --data data/polytron_blocks_clean.pt \
    --curve-ckpt data/quadtron_curve_best.pt --mode gt --n 46                  # cross-check
```
