# Night run 2026-10-05 -- results and diagnostics

Raw numbers: `SUMMARY.md`, `run_all.log`, `runs_curve_v2.log`, `/work/relabel/audit.csv`.

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

## Diagnostics (probe_conditioning.py)
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
  memorisation, no generalisation. Checkpoints: runs/curve_v2_best.pt / _last.pt.
