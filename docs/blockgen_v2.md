# Block generator v2 (no block count, cross-attention to the surface)

Generates the hex block structure of a tistos passage from its labelled surface point cloud.
Training data: beam-relabelled samples (domain_partition_3D `experimentell/hex3d_algohex/relabel/`),
core blocks only — the blade is always inserted after generation with the exact dtOO geometry.

| file | purpose |
|---|---|
| `meshtron/model/blockgen.py` | `BlockGen`: decoder with cross-attention to Perceiver latents of ALL surface points (+7 label channels, blade = 7); no block-count input |
| `meshtron/training/train_blockgen.py` | tasks `hexarow` (HexaRow tokens, topo order), `vertex` (Polytron stage 1: each corner once), `conn` (stage 2: pointer net); `--select best/last`, `--train-frac`, `--eval-train N` |
| `scripts/blockgen_v2/build_dataset.py` | relabelled samples -> dataset (topo-canonical tokens, `vtok`, `conn`, all surface points); `--topo <code>` keeps one topology |
| `scripts/blockgen_v2/build_curve_dataset.py` | curve-head dataset on the same split; `--canonical` = generator's vertex order (required for the chain) |
| `scripts/blockgen_v2/infer_chain_v2.py` | vertex -> conn -> snap -> curve head -> curved TFI, with GT / mean-template / curve-only variants; mesh metrics + R_mesh |
| `scripts/blockgen_v2/probe_conditioning.py` | does the vertex model use its point cloud |
| `scripts/blockgen_v2/probe_chain_sensitivity.py` | chain response to corner noise |
| `scripts/blockgen_v2/runs/grid.sh`, `best_case.sh` | the two experiment drivers |
| `scripts/blockgen_v2/_paths.py` | env defaults: `BLOCKGEN_SAMPLES`, `BLOCKGEN_AUDIT`, `BLOCKGEN_WORK`, `BLOCKGEN_FEATURE_CACHE` |

Results: `reports/blockgen_v2_night_grid.md` (all 351 audit-ok samples) and
`reports/blockgen_v2_canonical_best_case.md` (one topology, overfit best case, curve-head order bug).

Key numbers (median over 34 items): best case on train 1.9 % surface uncovered / 0.22 % inverted
cells (GT 0 % / 0.015 %); val 18.7 % uncovered — corner error 86.9e-3 vs mean template 128.7e-3,
and the curve head leaves 15 % uncovered even on exact GT corners. More geometries are the lever
(learning curve 171 -> 135 -> 118 for 76 -> 152 -> 303 samples). Select checkpoints by corner /
mesh metrics, not by val token loss.
