# Block generator v2 — night grid 2026-10-05 (results and diagnostics)

Setup: model meshtron/model/blockgen.py, trainer meshtron/training/train_blockgen.py, dataset
scripts/blockgen_v2/build_dataset.py on the beam-relabelled n2000 samples (domain_partition_3D
experimentell/hex3d_algohex/relabel/), driver scripts/blockgen_v2/drivers/grid.sh.

## Relabel / audit
- 369/369 n2000 samples relabelled (beam collapse), 0 lane failures; 351 ok, 18 rejected
  (beam:inverted 10, beam:1_non_cuboid 7, beam:excess_faces 7, topo:ValueError 4,
  blocks:mixed_orientation 3; a sample can carry several reasons).
- Blocks before: 22 x223, 12 x70, 16 x55, 75 x10, rest scattered (36..66).
  After (ok): 12 x338, 10 x5, 42 x4, 16 x3, 36 x1.
- 8 distinct topologies; canonical 12bl/49a0142abb (rows 3-2-3-3-1) = 337/351.
- Dataset v2_cart.pt: train 316 / val 35 (34 val items canonical).

## Training grid (val n = 35)
| run | params M | valid | struct ok | corner mean x1000 (median) |
|---|---|---|---|---|
| vertex S/M/L | 1.5 / 7.3 / 60.9 | 35/35/35 | 34/34/34 | 124 / 138 / 133 |
| hexarow S/M/L | 1.6 / 7.4 / 61.1 | 35/35/35 | 8/17/0 | 123 / 151 / - |
| conn S/M | 1.3 / 7.2 | exact 35/35 | | |

## Diagnostics (scripts/blockgen_v2/probe_conditioning.py)
- Mean-template baseline (mean of the canonical train corners): corner error 128.7;
  nearest-train-sample oracle: 75.6. The vertex models (124-138) are NOT better than the
  mean template.
- The models do use the conditioning: fed another item's points, the prediction matches
  that item's GT exactly as well as the own prediction does (vertex_S 123.7 both), and the
  error to its own GT rises to 168. Predictions are shrunk towards the mean: spread
  across items 102 (S) / 75 (M) vs 153 in the GT.
- conn exact 35/35 is near-trivial: 34/35 val items share one connectivity.
- Training stops early (best epoch 30-135, 2-3 min per run): 316 samples overfit fast.
- Curve head (weight-label 7): val loss best 3.79 at the first evaluations, then rises to
  9.45 while train loss -> 0.0007 and train acc -> 1.00; val acc flat at 0.17. Pure
  memorisation, no generalisation. (This curve head was trained in the RAW sample vertex order;
  see blockgen_v2_canonical_best_case.md for the order bug.)

## Raw grid table

| run | params M | best epoch | val loss | val n | valid | struct ok | corner mean (median) | corner max (median) | exact conn |
|---|---|---|---|---|---|---|---|---|---|
| conn_M | 7.21 | 70 | 0.000 | 35 |  |  |  |  | 35 |
| conn_S | 1.32 | 125 | 0.000 | 35 |  |  |  |  | 35 |
| hexarow_L | 61.12 | 45 | 1.934 | 35 | 35 | 0 |  |  |  |
| hexarow_M | 7.39 | 50 | 1.629 | 35 | 35 | 17 | 151.4833984375 | 382.7200012207031 |  |
| hexarow_S | 1.57 | 135 | 1.642 | 35 | 35 | 8 | 123.38855743408203 | 388.2192687988281 |  |
| smoke_conn | 1.30 | 10 | 3.191 | 5 |  |  |  |  | 0 |
| smoke_hexarow | 7.28 | 75 | 3.681 | 5 | 5 | 0 |  |  |  |
| smoke_vertex | 7.25 | 75 | 3.531 | 5 | 5 | 5 | 133.93822703119932 | 339.52778784775114 |  |
| vertex_L | 60.91 | 30 | 2.479 | 35 | 35 | 34 | 133.00751906769753 | 361.35916021434275 |  |
| vertex_M | 7.28 | 40 | 2.392 | 35 | 35 | 34 | 138.13960356466566 | 388.2809577128514 |  |
| vertex_S | 1.52 | 85 | 2.377 | 35 | 35 | 34 | 123.6661046963461 | 337.1312722930742 |  |

corner values: distance generated vs GT corner in canonical order, dataset units x 1000.
