# GRPO reward analysis and v2 proposal (2026-10-05)

Scope: `meshtron/training/rewards_hexarow.py` (v1) as used by `train_grpo.py`, and a
prototype replacement in `meshtron/training/rewards_v2.py` +
`meshtron/geometry/tfi_mesh_quality.py` (not wired into `train_grpo.py` yet).
Probes: `scripts/reward_probes/{probe_conform,probe_v2,diag_gt}.py` (30 or 12 val items of
`hexarow_tokens_family_cart.pt`, GT meshes perturbed in a controlled way).

## 1. What the existing GRPO runs show

| run (data/…_log.csv) | valid share (50-step windows) | mean R | verdict |
|---|---|---|---|
| `grpo_cart` (SFT ep584) | 0.67 → 0.66 → 0.64 → 0.75 → 0.74 → 0.73 | 0.78 → 0.84 | learned; gain came from `r_valid` + `r_quality` |
| `grpo_h05_ep254` | 0.00 in all windows | flat 0.27–0.32 | no learning; dense terms did not lift any rollout to valid, KL drifts up to 0.06 |
| `grpo_h05_fixed` | reward exactly 0 for all 300 steps | 0 | every rollout hard-invalid; KL −0.65 → −267, entropy → 0: policy collapse driven by the old naive KL term (now fixed: `--kl-estimator k3` is the default) |

`r_conform` sits at 0.6–0.7 in the logs and contributes `0.1 * r_conform` ≈ 0.07 to R.

## 2. Defects of v1

1. **`r_conform` is saturated.** Symmetric Chamfer normalised by the bbox diagonal:

   | variant of the GT mesh | r_conform | Chamfer / diag |
   |---|---|---|
   | GT | 0.964 | 0.036 |
   | noise 1 % diag | 0.961 | 0.039 |
   | noise 10 % diag | 0.924 | 0.076 |
   | shift 5 % diag | 0.952 | 0.048 |
   | scale 0.8 | 0.963 | 0.037 |
   | mesh of a different geometry | 0.962 | 0.038 |

   The GT→gen direction (dense surface → a few coarse corners) has a floor of ~0.035 diag
   that hides everything else. With weight 0.1 the whole term moves R by < 0.001
   between a correct mesh and a shifted / shrunken one.
2. **`r_quality` is not scale-invariant.** `hex_min_jacobian` is the unscaled det(J)
   (volume units): scaling a mesh by 0.8 drops it from 0.111 to 0.057 (= 0.8³). The policy
   is rewarded for larger blocks, not better-shaped ones. The same holds for
   `block_scores` (per-block advantage).
3. **The gate and `r_count` use the GT block count** (`expected_blocks=item["blocks"]`),
   and the policy is conditioned on it (`fc = item["blocks"]`). The relabelled data have
   one canonical 12-block topology, and the v2 model has no block count input. A mesh
   that fits the geometry but has a different (valid) block count fails the gate.
4. **No label term.** Nothing checks that boundary faces sit on the right patch (blade = 7).
5. **No mesh-level term.** The quantity the CFD needs — the quality of the refilled
   mesh after curve head / seam replacement / TFI — is not rewarded at all.

## 3. Prototype v2 (`meshtron/training/rewards_v2.py`)

`total = gate * weighted mean(R_fid, R_lab, R_q[, R_mesh])`, every term in [0, 1].

| term | definition | why |
|---|---|---|
| gate | 1 if `validate_generated_mesh` passes **without** GT block count, else 0.3 × share of positive-volume blocks (× 0.5 if non-manifold) | consistent with GT (corner scaled Jacobians are NOT a gate: 10/12 sampled GT meshes have negative corner SJ, down to −0.96 — coarse blocks on curved patches) |
| R_fid | ½ (corner term + coverage); corner: boundary corners → surface, `exp(-(d / (0.1 h))²)`, h = median incident edge length; coverage: share of surface samples within 0.15 h of a boundary quad (point–triangle distance), blade samples ×2 | local spacing instead of bbox diagonal |
| R_lab | share of on-surface boundary faces whose corners share a surface label (label sets within reach, so patch-edge corners are fine) | blade label 7 must be on blade faces |
| R_q | clip(p5 of corner scaled Jacobian / 0.4) | scale-invariant, tail-sensitive |
| R_mesh | optional, post-TFI, `mesh_quality.py` | see section 4 |

Same perturbations, v1 vs v2 (30 val items; weights 0.45/0.15/0.25, no R_mesh):

| variant | v1 total | v2 total | gate | R_fid | R_lab | R_q |
|---|---|---|---|---|---|---|
| GT | 1.519 | 0.788 | 0.90 | 0.91 | 0.92 | 0.73 |
| 1 boundary corner moved 0.3 h | 1.517 | 0.758 | 0.90 | 0.90 | 0.91 | 0.63 |
| noise 0.1 h | 1.220 | 0.378 | 0.71 | 0.70 | 0.87 | 0.01 |
| noise 0.3 h | 0.487 | 0.120 | 0.30 | 0.55 | 0.68 | 0.00 |
| shift 5 % diag | **1.518** | **0.573** | 0.90 | 0.50 | 0.82 | 0.73 |
| scale 0.8 | **1.491** | **0.525** | 0.90 | 0.40 | 0.82 | 0.73 |
| mesh of a different geometry | 1.037 | 0.779 | 1.00 | 0.81 | 0.89 | 0.66 |

- v1 cannot tell a shifted or shrunken mesh from GT (1.518 vs 1.519); v2 drops by 0.2–0.26.
- v1 punishes the foreign mesh only through the GT block count (gate + `r_count`). v2
  scores it close to GT because these are neighbouring geometries of the same machine
  family whose meshes really do fit within 0.1 h — a conforming foreign decomposition is
  not wrong for this application.
- On a relabelled sample with the real 7 labels (`machine_0414_n2000`): gate 1,
  R_fid 0.95, R_lab 1.00, R_q 0.54, total 0.84; 44 ms per rollout (1 core).

## 4. Post-TFI mesh quality (`meshtron/geometry/tfi_mesh_quality.py`)

checkMesh-style metrics on the refilled hex mesh: corner scaled Jacobian (min, p1, inverted
cells), max/mean non-orthogonality, skewness, aspect ratio.
`R_mesh = 0` if any cell is inverted, else mean of `clip(sj_p1/0.5)`,
`clip((80 − nonortho_max)/40)`, `clip((1.5 − skew_max)/1.0)`.

On 20 GT `tfi.vtk` of the relabelled n2000 set: sj_p1 0.62–0.83, nonortho_max 51–72°,
skew_max 0.44–0.73, no inverted cells; R_mesh 0.69–0.91. Not saturated: it spreads over the GT set.
Cost: `tfi.py` 0.65 s per sample at target h 0.1 (1.7 s at h 0.05), single core.

**Limitation:** `tfi.py` refills an existing block structure with lattices
(`--require-lattices`). For rollouts the chain behind the generator (snap → curve head →
seam-edge replacement → TFI) has to produce that input, so R_mesh can only enter RL once
that chain is trained and frozen.

## 5. Recommendation

1. Replace v1 by v2 in `train_grpo.py` for the v2 / relabelled data: needs the per-point
   labels in the RL items (already in `build_dataset.py` output) and dropping the
   block-count conditioning (`fc`).
2. Use scaled Jacobians for the per-block advantage (`block_scores`) as well.
3. Start RL with gate + R_fid + R_lab + R_q (cheap, available now); add R_mesh
   (weight ~0.15–0.3) once the curve head is frozen. Optionally a fidelity term on the
   curved edges (curve head output → surface) in place of the face-centre distance,
   which at block level is dominated by chord sag (reported only as `detail['mid']`).
4. Calibrate the weights on real rollouts of the trained v2 model (group std per term,
   share of groups with zero advantage) before a long run, not on perturbations alone.
5. Open: the per-item Spearman of reward vs. perturbation severity is −0.35 (v2) vs −0.48
   (v1) on this synthetic ladder. v1 gains here only through its block-count gate on the
   foreign mesh. The severity ladder is hand-made and not a target.
