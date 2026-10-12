# meshtron v2 sandbox — block generation without block count (2026-10-05/06)

Prototype, not yet ported to /opt/stack/meshtron. Data: the beam-relabelled samples of
/work/relabel; tokenizer: topology-canonical HexaRow row plan of /work/meshtron_topo.
Reports: `NIGHT_REPORT.md` (night grid, all 351 samples), `CANON_REPORT.md` (canonical
topology only, overfit best case, chain evaluation, curve-head order bug), `SUMMARY.md`
(raw grid table).

## Code
| file | purpose |
|---|---|
| `build_dataset.py` | relabelled samples -> `data/*.pt` (topo order, all surface points + 7 label channels, `vtok` for Polytron stage 1, `conn`); `--topo <code>` keeps one topology |
| `build_curve_dataset.py` | curve-head dataset, same split; `--canonical` renumbers into the generator's vertex/block order (REQUIRED for the chain) |
| `model.py` | `BlockGen`: decoder with cross-attention to Perceiver latents of the point cloud (no block count) |
| `train.py` | tasks `hexarow`, `vertex` (Polytron stage 1, each corner once), `conn` (pointer net, stage 2); `--select best/last`, `--train-frac`, `--eval-train N` |
| `infer_chain_v2.py` | vertex -> conn -> snap -> curve head -> curved TFI, variants gen / gen_nosnap / gen_gtconn / mean / gt_canon_curves / gt_curves / gt; mesh metrics + R_mesh |
| `probe_conditioning.py` | does the vertex model use its point cloud (own vs. other item's points, mean-template baseline) |
| `probe_chain_sensitivity.py` | chain response to corner noise (found the curve-head order bug) |
| `*.bak_20261006` | versions before the 2026-10-06 options were added |

## Data and runs
| path | content |
|---|---|
| `data/v2_cart.pt` | all 351 audit-ok samples (train 316 / val 35) |
| `data/v2_canon.pt` | only topo 49a0142abb (337: train 303 / val 34) |
| `data/curve_v2.pt`, `data/curve_canon.pt` | curve sets in RAW sample order (do not use with the generator) |
| `data/curve_canon_order.pt` | curve set in canonical order (use this) |
| `runs/{vertex,hexarow,conn}_{S,M,L}` | night grid (best-val checkpoint) |
| `runs/of_*` | overfit runs on v2_canon (last checkpoint); `of_vertex_S_long` = 1500 epochs |
| `runs/lc_vertex_S_{0.25,0.5,1.0}` | learning curve |
| `runs/curve_v2_*`, `runs/curve_canon_*` | curve heads in raw order (obsolete) |
| `runs/curve_order_{best,last}.pt` | curve head in canonical order |
| `chain/<run>/summary.json` | chain evaluations (of_*, order_*, long_* = train/val) |

Shell drivers: `run_all2.sh` (night chain), `resume_night.sh`, `run_curve.sh`, `run_canon.sh`,
`run_curve_canon.sh`, `run_chain_canon.sh`, `run_curve_order.sh`, `run_vertex_long.sh`;
markers `*_DONE`, logs `runs_*.log`, `chain_*.log`.

## Key numbers (median over 34 items)
- Best case on train (vertex_S 1500 ep, corner err 3.1e-3): chain 1.9 % surface uncovered,
  0.22 % inverted cells (GT: 0 % / 0.015 %).
- Val: corner err 86.9e-3 (mean template 128.7e-3), chain 18.7 % uncovered; the curve head
  leaves 15 % uncovered even on exact GT corners (memorises, val acc 0.17).
- Learning curve vertex_S (val corner err): 76 samples 171, 152 -> 135, 303 -> 118.
- conn exact 34/34 everywhere (trivial with one topology).

## Flow-matching geometry model (2026-10-11, `FLOW_REPORT.md`)
| file | purpose |
|---|---|
| `build_flow_dataset.py` | `data/flow_canon.pt`: continuous corners + Bezier edge offsets, surface labels per vertex/edge |
| `flow_geo.py` | set transformer (DiT + PointPerceiver), `--mode flow/regress`, `--parts verts/all` |
| `infer_chain_flow.py` | snap + curved TFI chain for flow_geo checkpoints (GT topology) |
| `probe_flow_conditioning.py` | own vs foreign point cloud |
| `train_cont.py` | continuous / ordinal AR heads for the vertex task (side result) |

Key numbers (val, 34 items): flow verts+curves corner err 50.8 (3 seeds; AR 86.9, template 124.7);
chain uncovered 1.5 % mean / 0.02 % median, 17/34 meshes without inverted cells
(AR chain 20.2 % / 18.7 %, 0/34; GT 0.06 % / 0 %, 16/34).
