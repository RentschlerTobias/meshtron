# Block generator v2 — canonical-topology best case (2026-10-06)

Data: `build_dataset.py --topo 49a0142abb` = only topo 12bl/49a0142abb (337 samples, train 303 / val 34).
Models trained to overfit on purpose (last checkpoint). Chain: `scripts/blockgen_v2/infer_chain_v2.py`; driver `scripts/blockgen_v2/drivers/best_case.sh`
(vertex -> pointer conn -> snap -> curve head -> curved TFI, h = 0.08), 34 train + 34 val items.

## Bug found and fixed: curve head vertex/block order
The curve head was trained on the raw sample order (`build_item` on sample.npz), the generator
emits the canonical order. Same GT corners, only renumbered: 11.4 % uncovered instead of 0 %.
Fix: `scripts/blockgen_v2/build_curve_dataset.py --canonical` (exact coordinate match, both edge directions kept)
(curve head trained on that set).

## Results (median over 34 items; uncovered = surface share not covered, inv = inverted cells)
| stage-1 model | split | corner err x1000 | chain uncovered | chain inv | GT corners + curve head |
|---|---|---|---|---|---|
| vertex_S 400 ep, raw-order curve head | train | 18.5 | 27.6 % | 1.00 % | 11.4 % (order bug) |
| vertex_S 400 ep, canonical curve head | train | 18.5 | 10.8 % | 0.34 % | 0.0 % / 0.015 % |
| vertex_S 1500 ep, canonical curve head | train | 3.1 | **1.9 %** (p90 10.9 %) | 0.22 % | 0.0 % / 0.015 % |
| vertex_S 1500 ep, canonical curve head | val | 86.9 | 18.7 % (p90 27.5 %) | 0.80 % | 15.0 % / 0.58 % |
| mean template + GT conn | val | 128.7 | 23.9 % | 0.42 % | |
| GT blocks + GT curves (ceiling) | val | 0 | 0.0 % | 0.007 % | |

- conn (pointer net): exact 34/34 on train and val in every run.
- Snap helps once the order is right (train, 400 ep: 15.1 % -> 10.8 %; 1500 ep: 6.8 % -> 1.9 %).
- Longer training lowers the val corner error (102.6 -> 86.9) although val loss rises
  (3.2 -> 6.5): val token loss is the wrong checkpoint criterion for geometry.
- Learning curve (vertex_S, best-val ckpt, val corner err): 76 -> 171, 152 -> 135, 303 -> 118.
- HexaRow (400 ep) memorises corners to 3.3 but gets the val structure right only 19/34.

## Reading
- Best case (train): the chain reproduces GT-quality meshes once stage 1 is converged
  (1.9 % uncovered, 0.22 % inverted vs 0 % / 0.015 % for GT). Residual = corner quantisation.
- Val: two generalisation gaps. Corners (86.9 vs 3.1 on train) and the curve head (15 %
  uncovered even on exact GT corners; val acc 0.17-0.18, pure memorisation).
