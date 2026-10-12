# Flow-matching geometry model for curved hex block structures (2026-10-11)

## Question
Can continuous coordinates replace the 512-bin coordinate tokens of Polytron stage 1, and is a
flow-matching transformer the right architecture for point cloud -> minimal hex block structure
with cubic Bezier edges -> TFI?

## Setup
- Data `data/flow_canon.pt` (`build_flow_dataset.py`): the 337 samples of the canonical topology
  (train 303 / val 34, same split as `v2_canon.pt`, val = unseen machines). Per sample 40 corners,
  12 blocks, 84 edges with continuous chord-relative Bezier offsets (GT `edge_ctrl` of the relabelled
  samples, not the quantised curve-head targets), 14-dim point features (xyz, r/sin/cos/z, 7 labels).
- Topology features per token (no GT geometry): majority surface labels of the vertex / the edge
  midpoint over the train set (on-surface tolerance 6e-3 x bbox diagonal: the distance histogram has
  a gap at ~10^-2.2 because corners lie on the exact geometry, not on the coarse triangulation;
  1e-4 marked 34 % of the corners as off-surface), Laplacian eigenvectors of the block-edge graph,
  Fourier-encoded mean-template position.
- Model `flow_geo.py` (2.1 M params): set transformer over 40 vertex + 84 edge tokens (residuals to
  the mean template, normalised), 4 DiT blocks (adaLN-Zero time conditioning, self-attention over all
  tokens, cross-attention to 32 PointPerceiver latents of the point cloud). Same network in two modes:
  - `flow`: rectified flow, logit-normal t, 32 Euler steps at sampling;
  - `regress`: zero input state, direct MSE on the data (control).
  `--parts verts` predicts corners only (mean-template curves), `all` corners + curves jointly.
- Training: 1500 epochs, bs 16, AdamW 3e-4, cosine, EMA 0.999; 3 seeds per configuration
  (`run_flow.sh`, ~49 min per run on the RTX PRO 5000, 3 runs in parallel).
- Units: dataset units x 1000, median over the 34 val items unless stated.

## Results: geometry (val, 3 seeds)
| configuration | corner s0 / s1 / s2 | corner mean | ensemble of 8 | curve | curve shape (GT corners) | train corner |
|---|---|---|---|---|---|---|
| **flow, verts + curves** | 52.3 / 51.1 / 49.1 | **50.8** | **49.6** | **50.1** | **9.7** | 3.5 |
| flow, verts only | 53.1 / 49.9 / 58.2 | 53.7 | 53.8 | 57.0 | 14.6 (template) | 2.7 |
| regress, verts only | 58.3 / 58.4 / 55.7 | 57.5 | - | 56.9 | 14.6 (template) | 0.5 |
| regress, verts + curves | 61.1 / 64.1 / 57.7 | 61.0 | - | 59.6 | 12.5 | 0.5 |
| AR vertex model (of_vertex_S_long, 512 bins) | 86.9 | | | | | 3.1 |
| mean template | 124.7 | | | 116.3 | 14.6 | |

- `curve`: distance of predicted to GT Bezier at t = 1/4, 1/2, 3/4 (corners + shape);
  `curve shape`: predicted offsets on GT corners. The `curve` mean of flow/verts includes seed 0,
  which was evaluated before a fix (verts-only flow left the unused curve state as noise); its
  corner values are unaffected.
- Flow samples are nearly unimodal: spread between 8 samples 2-3 (vs error ~50); best of 8 against
  GT (oracle) 45.1 for flow/all s0.
- Regression fits train almost exactly (0.5) and generalises worse; flow with joint curves is the
  most consistent configuration across seeds.

## Results: conditioning probe (`probe_flow_conditioning.py`, all 12 runs)
Every model follows its point cloud. Example flow/all s2: own cloud 48.6 | another item's cloud
176.6 against the own GT, but 50.0 against the other item's GT | mean template 124.7. Spread of the
predictions across items 129-133 (GT 142); the AR model stayed close to the template.

## Results: end-to-end chain (`infer_chain_flow.py`, seed 0, GT topology, snap + curved TFI, val)
| variant | uncovered mean / median / max | inverted cells mean | meshes without inverted cells |
|---|---|---|---|
| **flow, verts + curves** | **1.50 % / 0.02 % / 8.2 %** | **0.050 %** | **17 / 34** |
| flow, verts + curves, best of 4 (GT-free score) | 1.23 % / 0.01 % / 8.2 % | 0.047 % | 16 / 34 |
| regress, verts + curves | 7.37 % / 7.15 % | 0.053 % | 7 / 34 |
| flow, verts only (template curves) | 9.17 % / 9.84 % | 0.048 % | 12 / 34 |
| mean template corners + curves | 22.2 % / 23.1 % | 0.194 % | 3 / 34 |
| AR chain (vertex AR + pointer conn + curve head, chain/long_val) | 20.2 % / 18.7 % / 42.8 % | 1.103 % | 0 / 34 |
| AR chain, GT corners + curve head | 17.1 % / 15.0 % | 0.834 % | 1 / 34 |
| GT | 0.06 % / 0.00 % / 1.0 % | 0.029 % | 16 / 34 |

- All 34 meshes watertight in every variant. The flow meshes differ from GT (different point
  counts, corner errors 30-65) but cover the surface nearly as well.
- Curves dominate surface coverage: the same corners with template curves leave ~10 % uncovered,
  with jointly predicted curves 1.5 %. The corner error is a weak proxy for mesh quality.
- `R_mesh` (`/work/rl_reward/mesh_quality.py`) is 0 for any mesh with an inverted cell, so its median
  is ~0 for GT as well; not usable as a ranking metric here.
- `flow_nosnap` is close to `flow` (uncovered median 0.3 %): snapping matters little once the
  curves are right.

## Side result: continuous autoregressive heads (`train_cont.py`)
512-bin quantisation error of the AR tokens: mean 3.09e-3, i.e. the AR train best case (3.12e-3)
is at the quantisation floor. Overfit check (15 samples, 2000 steps): per-vertex 512-bin heads reach
3.09 (floor), HL-Gauss and bins+offset learn worse, GMM / regression heads did not yet decode the
right vertex count. Not pursued further: the set formulation above removes the sequence altogether.

## Limitations
- One topology (stage A = GT topology); conn / topology generation not part of this test.
- 34 val geometries; seed spread of the corner error up to ~8 points. Chain on seed 0 only.
- No auxiliary surface loss, no learning curve yet (plan steps 3c / 3d).

## Next steps
1. Learning curve (0.25 / 0.5 / 1.0) and the 1000 new tistos geometries once datagen delivers them.
2. Auxiliary surface losses (corner / curve on its labelled surface).
3. Reward check (GT vs flow vs template) before any GRPO / reward-weighted fine-tuning; the reward
   only needs the geometry, so RL could also use geometries without a GT block structure.

## Files
`build_flow_dataset.py`, `flow_geo.py`, `infer_chain_flow.py`, `probe_flow_conditioning.py`,
`train_cont.py`; drivers `run_flow.sh`, `run_flow_chain.sh`, `run_flow_probe.sh`; runs
`runs/flow_{flow,regress}_{verts,all}_s{0,1,2}/`, smoke runs `runs/flow_smoke/`, `runs/cont_smoke/`;
chains `chain/flow_*_val/`; probe `probe_flow_conditioning.log`.

## Update 2026-10-11 late: 141 new tistos_s1 geometries (papaya datagen)
- Snapshot of 148 finished samples of `/opt/stack/papaya_datagen/runs` -> `/work/datagen_papaya/runs`
  (sample.npz, beam.log, tfi.log, params.json); audit with the unchanged relabel checks
  (`/work/datagen_papaya/audit_papaya.py`): 144 ok, 142 in the canonical topology. One more dropped by
  the 512-bin tokenizer (`tistos_s1_0020`: two corners in one quantisation cell). Same topology hash,
  identical majority surface labels, inside the old normalisation bounds.
- `build_dataset.py` got `--src-root` / `--bounds-from` and skips tokenizer collisions (backup
  `.bak_20261011`; rebuild of `v2_canon.pt` without the new flags is bit-identical).
  `data/flow_canon_papaya.pt`: train 416 (303 old + 113 new), val 34 (old, unchanged), val_new 28.
- `flowp_flow_all_s{0,1,2}`: flow, verts + curves, same settings; old models evaluated on val_new with
  `eval_flow.py`.

| corner err x1000 (s0 / s1 / s2, mean) | old val (34) | new val_new (28) | curve shape old val / val_new |
|---|---|---|---|
| trained on 303 (old) | 52.3 / 51.1 / 49.1 = 50.8 | 43.2 / 45.3 / 42.8 = 43.8 | 9.7 / 8.8 |
| trained on 416 (old + new) | 55.3 / 52.7 / 49.3 = 52.4 | 44.3 / 40.4 / 40.1 = 41.6 | 8.9 / 8.6 |

Chain (seed 0; uncovered mean / median / max, inverted mean, meshes without inverted cells):

| | old val (34) | val_new (28) |
|---|---|---|
| trained on 303 | 1.50 / 0.02 / 8.2 %, 0.050 %, 17/34 | 0.67 / 0.00 / 4.0 %, 0.046 %, 13/28 |
| trained on 416 | 0.64 / 0.00 / 5.9 %, 0.050 %, 11/34 | 0.68 / 0.00 / 5.1 %, 0.029 %, 14/28 |
| GT | 0.06 / 0.00 / 1.0 %, 0.029 %, 16/34 | 0.14 / 0.00 / 1.6 %, 0.051 %, 12/28 |

- +37 % training data: corner error unchanged within the seed spread (~5 points); curve shape
  consistently slightly better; coverage on the old val improves (1.50 -> 0.64 % mean), on val_new
  equal. Chain numbers are one seed each.
- On the new geometries both models are at GT level after TFI (all watertight, median 0 % uncovered,
  as many or more meshes without inverted cells than GT).
- The models trained on old data generalise to freshly drawn geometries: 42-45 on val_new. val_new is
  only slightly easier for the mean template (118.8 vs 124.7 on the old val).
