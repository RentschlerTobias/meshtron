# experiments/flow_topology

Prototypes of the flow-matching block generator and its path to variable block topology
(sessions 2026-10-05 to 2026-10-12, copied from the local working directory `/work/meshtron_v2`).
The scripts still use the absolute paths of that working directory (`/work/meshtron_v2`, `/work/rl_reward`,
`/work/datagen_full`, `/work/relabel/out`, `/work/curve/features_v2`) and expect
`PYTHONPATH=<domain_partition_3D>/experimentell/hex3d_algohex` for the TFI chain. Data, checkpoints and chain
outputs are not part of the repository. `README_v2.md` describes the older AR / pointer files of the same
directory (Polytron stages, night grid).

## Plans and reports

| file | content |
|---|---|
| `VARIABLE_TOPOLOGY_PLAN.md` | approved plan: flow (corners) -> pointer (blocks) -> flow (curves), milestones M1-M4 |
| `FLOW_REPORT.md` | flow-matching geometry model vs. AR Polytron (fixed topology) |
| `BLADE_INSERTION_PLAN.md` | blade O-grid insertion before TFI (planned) |
| `reports/`, `research_notes/` | literature research on block-structured hex optimisation (German report) |

## Code

| file | role |
|---|---|
| `flow_geo.py` | fixed-topology flow model (corners + Bezier offsets, mean-template residuals) |
| `flow_vertex.py` | stage 1: corner SET of predicted size (count head, OT coupling, no template) |
| `pointer_conn.py` | stage 2: set pointer -> blocks (8 pointers per block, validity masks) |
| `flow_geo_var.py` | stage 3: topology-agnostic curves + corner refinement (RWPE, hop-biased attention) |
| `continuity.py` | continuity detection from topology + patches (feature / 2d / 3d pairs, corners) |
| `curve_opt.py` | seam arcs + differentiable edge-based Gordon-Hall TFI optimisation (hinge; Garanzha variant) |
| `infer_chain_flow.py`, `infer_chain_var.py` | end-to-end chains (snap, curved TFI, mesh metrics, ring kinks) |
| `build_flow_dataset.py`, `build_flow_dataset_post.py` | flow datasets (raw / post-processed curve targets) |
| `summarize_curve_opt.py` | tables for the curve optimisation runs |
| `run_*.sh` | drivers of the runs listed in the reports |
| `rl_reward/` | mesh quality metrics used by the chains |

The numpy-only part of the curve post-processing (continuity, exact feature arcs, G1 cubics) lives in
domain_partition_3D (`experimentell/hex3d_algohex/curve_postprocess.py`, branch `feat/curve-postprocess`).

## State (2026-10-12)

- M1 reached: stage 3 on the full data (995 train geometries) beats the fixed-topology model in the chain
  (uncovered 0.29-0.40 % vs 0.55-0.90 %); trained on post-processed targets the O-rings are smooth
  (ring kink median 2.5 deg vs 25-29 deg).
- Stage 1: corner error ~60 is mostly sliding along seams / surfaces (5-9 off the geometry); judge it by role
  and by the chain, not by the distance to the AlgoHex corners.
- M3 (pointer on stage-1 corners) in training.
