# Stage-1 duplicate corners and the M4 chain (2026-10-18 .. 2026-10-20)

**Finding.** Remaining M4 failures (wrong topology although the corner count is right) come from stage 1, not from
the pointer: on GT corners the pointer is exact on the hard cases (4/4). Stage 1 sometimes puts two corners on one
GT corner (another one is missing) or misplaces one; the old snap could stack near-duplicates exactly.
Duplicate rate (pair < 4.8e-3 x diagonal, 8 draws): val 7.0 %, val_full 2.7 % at 32 Euler steps (3.7 / 1.6 % at 128,
but corners less accurate: 57 -> 61). The training loss (per-token velocity vs. the OT-matched target) never sees
the end result, so a doubled corner is not penalised directly.

**Training attempts on stage 1 (all start from runs/vtx_full_s0; none adopted):**

| run | method | result |
|---|---|---|
| vtx_x1_s0 | + two-sided Chamfer on x1_hat, late-t sampling | worse: dup val_full 2.7 -> 12.3 %; the pred->GT term rewards duplicates |
| vtx_grpo_s0 | Flow-GRPO (SDE a=0.3), chain reward, lr 1e-5, beta 0.01 | drifts: chain-ok 0.65 -> 0.32, corners worse, stopped at 76 |
| vtx_grpo_s1 | same, fixed pointer seed, G=16, lr 2e-6, beta 0.1 | stable but flat over 100 updates |
| vtx_bp_s0 | reward backprop through the ODE: Hungarian match + repulsion | dup val 7.0 -> 3.3 %, corners 57 -> 65, stopped at 100 |
| vtx_bp_s1 | weaker repulsion, lr 5e-6 | dup val ~5 %, corners slowly worse, stopped at 175 |

In every fine-tuning run the optimised loss itself rose: from a near-optimal start, per-batch gradients are mostly
noise and Adam random-walks uphill (hypothesis; large-batch test not done).

**Adopted fix, no training (infer_chain_m4.py):** 128 Euler steps, `snap_unique` (seam junctions one-to-one, no
snap-made collisions), and resampling of stage 1 (`--chain-tries 8`) until the corner set has no duplicate AND the
best pointer candidate is oriented, feature-complete and uses all corners (GT-free criterion). On the 12 val_full
M4 failures: isomorphic 8/12 -> 12/12, feature-complete 7 -> 12, meshed 9 -> 12, uncovered mean 0.79 -> 0.23 %,
at most 3 tries.
