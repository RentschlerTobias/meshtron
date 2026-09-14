# Decision Log — Meshtron 2D/3D Refactor, RL Curriculum, Long-Context Attention

Session date: 2026-09-13. Grilled via the `grilling` skill (tutor mode). Covers the scoping
session for a repo-wide refactor: a swappable 2D/3D pipeline across the two active model
families, a staged reinforcement-learning curriculum, and long-context attention.

## Final Design-Tree

```
Meshtron (project umbrella name)
├── A. Pipeline/config architecture   [full implementation]
│   ├── x Naming: Quadtron (row-encoded raw-quad-face family, formerly "Meshtron" model)
│   ├── x Naming: Polytron (two-stage family, formerly "Plan-B"/"PolyGen")
│   ├── x MeshtronDomain: deprecated — failed experiment, superseded by Polytron
│   ├── x 3D vertex coordinates: cylindrical (r, θ, z)
│   ├── x Entry point: single train.py, config-driven (architecture × tokenizer × dimension)
│   └── x Config format: frozen dataclass + JSON (existing TrainingConfig pattern), not YAML/TOML
├── B. RL curriculum: vertex → face → row → mesh (4 stages; "coordinate" dropped — already
│   │  covered by phase-A teacher-forcing pretraining)
│   ├── x Sequencing: starts only after A+C reach a validated baseline checkpoint (RL fine-tunes
│   │     a pretrained policy, it does not train from scratch)
│   ├── x Algorithm: pluggable AdvantageEstimator interface inside a new RLObjective
│   │     (extends the existing Objective ABC in objectives.py); default = GRPO (group-relative,
│   │     no critic network), PPO (critic-based) retrofittable behind the same interface
│   ├── x Reward: weighted sum of sub-metrics per stage, weights as config fields
│   └── x Scope: full curriculum for Quadtron; for Polytron, RL applies ONLY to Stage 1
│         (VertexGen) — Stage 2 (PointerFaceModel) already reaches 300/300 exact topology
│         via supervised teacher-forcing alone (documented round-trip result); Stage 3
│         (GeomHeadModel) is a non-autoregressive regression head (SmoothL1 on 4 real-valued
│         Bezier control-point scalars) with no token-level probability distribution to take
│         a policy gradient over — would require a separate continuous-action (Gaussian)
│         policy formulation, and there is no measured problem it would fix (median 0.7%
│         geometric round-trip error already achieved via supervised regression alone)
└── C. Sliding-window / flash attention   [full implementation]
    ├── x Sequencing: built in parallel with A (both touch the same HourglassTransformer /
    │     model backbone classes; sequencing B after this avoids retraining the RL policy
    │     twice if the backbone changes mid-way)
    ├── x Config: two independent boolean/int flags — `use_flash_attention: bool` (kernel-level,
    │     no behavior change) and `sliding_window_size: int` (0 = disabled/full attention,
    │     >0 = window size in tokens) — mirrors the existing "0 = disabled" convention already
    │     used in config.py (`max_val_batches: int = 0`)
    ├── x Window granularity: plain token-count window for the initial implementation (reuses
    │     standard flash-attention/SWA kernels directly); row-aligned or face-aligned windowing
    │     (Quadtron rows vs. Polytron faces — the two families have no shared semantic windowing
    │     unit) is deferred as a documented follow-up refinement, not built now
    └── x Older-context representation: reuse the existing `PerceiverPointEncoder` (already
          handles variable-length point sets → fixed latent count) to compress tokens/rows that
          fall out of the sliding window into the same `latent_condition` cross-attention slot
          already used for point-cloud conditioning — structurally the same pattern as
          DeepSeek-V4.1-Flash's CED (Causal Encoder-Decoder) architecture, which projects
          decoder global KV from encoder outputs instead of recomputing it
D. DeepSeek-V4.1-Flash cross-layer KV reuse / CSA2 — SCOPED OUT as its own workstream.
   Researched from the primary source (DeepSeek_V41_Tech_Report.pdf). Verdict: the paper's
   target regime (552B backbone parameters, up to 1M-token contexts, agentic serving cost)
   does not match this project's regime (≈5M-parameter models, sequences in the hundreds to
   low thousands of tokens). The transferable idea — compress/reuse distant context via a
   projected representation instead of recomputing it per layer — is structurally identical
   to the project's own point-encoder-conditioning pattern, so it was folded into C's
   "older-context representation" decision instead of being tracked separately.
```

## Decisions in detail

### Naming: Quadtron / Polytron / Meshtron umbrella
- **Options considered:** keep "Plan-B" name (rejected by user as awkward); "PolyGen" (initial
  assistant suggestion, accepted then later superseded); "Polytron" (chosen, matches a
  "-tron" family convention alongside the newly introduced "Quadtron").
- **Why:** user wants "Meshtron" to remain the umbrella project name, with the two active model
  families renamed consistently: **Quadtron** = the row-encoded raw-quad-face family (was
  informally "Meshtron" the model, `tokenizer_v2.py`), **Polytron** = the two-stage
  vertex→pointer→geometry family (was "Plan-B"/PolyGen, `prototype_twostage.py` and friends).
- **Consequence:** all future plan/doc/code work should use Quadtron/Polytron, not
  Meshtron-the-model or PolyGen/Plan-B, to avoid ambiguity with the project umbrella name.

### MeshtronDomain: deprecated
- **Options considered:** deprecate and exclude from the active refactor (chosen); leave
  untouched as a third, unmaintained family; extend it to 3D as a third full architecture.
- **Why:** user confirmed it was a failed experiment (`meshtron_domain.py` /
  `tokenizer_domain.py`) — documented result was severe overfitting (train ppl 2.8 vs. val
  ppl 17.8 on the original 100-mesh set) and invalid/empty generated output, which motivated
  building Polytron as the structural fix. It is superseded, not a peer architecture.
- **Consequence:** the unified 2D/3D config only needs to support two model families
  (Quadtron, Polytron), not three. `train_domain.py` and `meshtron_domain.py` are out of
  scope for the refactor; a repo-cleanup pass can mark them deprecated.

### 3D vertex coordinates: cylindrical (r, θ, z)
- **Options considered:** cylindrical (r, θ, z) — chosen; cartesian (x, y, z).
- **Why:** natural extension of the existing 2D polar-coordinate pattern (`DomainTokenizer`,
  `TwoStageTokenizer`) for axially-symmetric turbomachinery passage geometry (the tistos
  dataset).
- **Consequence:** low-risk, reversible — coordinate representation is a tokenizer-internal
  choice.

### Single entry point, config-driven family/tokenizer/dimension selection
- **Options considered:** single `train.py` selecting architecture × tokenizer × dimension via
  config (chosen); keep separate scripts per family.
- **Why:** current repo has three disconnected entry points (`train.py`, `train_domain.py`,
  `chain_e2e.py`) with inconsistent config mechanisms (argparse+dataclass vs. hardcoded vs.
  flat argparse flags); user wants one script to control model family, tokenizer strategy, and
  2D/3D via config.
- **Consequence:** cross-cutting — touches config.py, trainer.py, and all family-specific
  training loops. De-risked by keeping Polytron's inherently different 3-model training loop as
  its own dispatched function under the same config/CLI surface, rather than forcing it into
  the single-model `Trainer` abstraction.

### Config format: dataclass + JSON (not YAML/TOML)
- **Options considered:** dataclass + JSON (chosen, extends existing `TrainingConfig`/
  `DomainTrainingConfig` pattern in config.py); YAML; TOML.
- **Why:** matches the existing repo convention ("All hyperparameters live in TrainingConfig",
  per README); YAML/TOML's main advantage (human-editable, comments, natural fit for a future
  TUI) is only needed once the TUI is actually built, which the user explicitly deferred
  ("TUI wäre cool, aber das erst danach").
- **Consequence:** cheap to migrate later — a format switch is a mechanical addition
  (`to_yaml`/`from_yaml` next to existing `to_json`/`from_dict`), not a rewrite.

### A+C sequencing, B gated after
- **Options considered:** strict sequential with two gates; fully parallel (A, B, C all at
  once); A+C together → one gate → B (chosen).
- **Why:** A and C both modify the same backbone/model classes (`hourglass_transformer.py`);
  building C after B would force re-running RL fine-tuning once the backbone changes. RL
  fundamentally needs a pretrained policy checkpoint as a starting point (a randomly
  initialized policy gets near-zero reward on structural-validity rewards — an exploration
  problem), so B cannot meaningfully start before A+C produce a validated baseline.
- **Consequence:** big, cross-cutting, moderately reversible. Determines plan document
  structure: one integrated A+C milestone, then a B milestone gated on it.

### Attention backend as independent config flags
- **Options considered:** single combined flag; two independent flags (chosen).
- **Why:** flash-attention (kernel-level implementation detail, same math) and sliding-window
  (changes the actual receptive field) are orthogonal and independently toggleable.
- **Consequence:** `use_flash_attention: bool`, `sliding_window_size: int` (0 = full attention).

### RL algorithm: pluggable AdvantageEstimator, default GRPO
- **Options considered:** PPO (critic network + clipping); GRPO (group-relative baseline, no
  critic) — chosen as default; REINFORCE+baseline (single greedy-rollout baseline);
  rejection-sampling finetuning (no policy gradient, cross-entropy on best-of-N samples).
- **Why:** user asked whether the training structure could support switching between PPO and
  GRPO. Both share the same clipped-policy-gradient skeleton (`loss = -advantage × log P(...)`)
  and differ only in how the baseline/advantage is computed (learned critic vs. group-sample
  mean/std). This was generalized into a single `RLObjective` (extending the existing
  `Objective` ABC in objectives.py) with a swappable `AdvantageEstimator` component
  (`GroupRelativeAdvantage` = GRPO, `CriticAdvantage` = PPO). Default is GRPO because it needs
  no second network to train/stabilize, which matters given the small model sizes (~5M
  parameters) and small dataset (132 tistos meshes).
- **Consequence:** originally scoped as a large, hard-to-reverse decision; the pluggable
  design reduces it to a low-risk config default (`advantage_estimator: "group_relative" |
  "critic"`), consistent with the project's "everything is a config flag" pattern.

### RL curriculum stages and scope: vertex/face/row/mesh; Quadtron full, Polytron Stage-1-only
- **Options considered (stage count):** originally proposed 5 stages including "coordinate";
  user corrected — coordinate-level prediction is already what phase-A teacher-forcing
  pretraining does, not a separate RL stage. Settled on 4 stages: vertex, face, row, mesh.
- **Options considered (family scope):** RL applied identically to both families; RL for
  Quadtron fully + Polytron Stage 1 only (chosen); RL for Quadtron only.
- **Why:** Quadtron has no structural validity guarantee for generated token sequences, so RL
  has clear, measurable problems to fix. Within Polytron: Stage 2 (PointerFaceModel) already
  reaches 300/300 exact topology via supervised training alone (documented round-trip test in
  `07_polygen_walkthrough.md`) — no measured problem for RL to solve there, even though it is
  technically RL-compatible (its pointer choice is a softmax over existing vertices, i.e. a
  real discrete probability distribution). Stage 1 (VertexGen) does have a measured, unsolved
  problem: free-run generation does not always reproduce the correct vertex count
  (`vertex_eval`'s `count_ok` metric is not consistently 1.0). Stage 3 (GeomHeadModel) is a
  non-autoregressive regression head with no discrete action distribution — the current
  discrete-token GRPO/PPO design does not apply to it without a separate continuous-action
  (Gaussian) policy reformulation, and supervised regression already achieves a low measured
  error (median 0.7% relative to chord length), so there is no clear reward signal to add.
- **Consequence:** B's implementation surface is smaller and better justified by evidence
  rather than architectural symmetry. Revisit Stage 2/3 RL only if a future measured problem
  emerges there.

### Sliding-window granularity: token-count window (not row/face-aligned)
- **Options considered:** plain token-count window (chosen for initial implementation);
  semantic row/face window (row boundaries for Quadtron via `eor` tokens, face boundaries for
  Polytron — the two families have no shared semantic windowing unit); token window
  rounded up to the nearest row boundary (documented as a later refinement).
- **Why:** token-count windows can reuse standard flash-attention/SWA kernels directly, letting
  the sliding-window concept be validated with A before investing in custom per-family
  boundary-aware masking logic.
- **Consequence:** may cut a row/face mid-sequence in the initial implementation; the
  "older-context via PerceiverPointEncoder" mechanism is independent of window type and
  unaffected by this choice.

### DeepSeek-V4.1-Flash relevance: scoped out as a separate workstream
- **Research performed:** fetched and read the actual tech report
  (`DeepSeek_V41_Tech_Report.pdf`, via the user-provided Hugging Face link) rather than
  guessing. Confirmed mechanisms: CSA2 (Compressed Sparse Attention 2, cross-layer reuse of
  global KV/indexer-K across layers in Full/Reindex/Reuse modes), CED (Causal Encoder-Decoder,
  decoder global KV projected from encoder outputs), SWA Bounded Replay (reconstructs sliding-
  window KV by replaying only the most recent window instead of persisting the full history).
- **Why scoped out:** built for 552B-parameter models serving up to 1M-token agentic contexts;
  this project's models and sequence lengths are multiple orders of magnitude smaller, so the
  specific compression kernels are very unlikely to pay off at this scale. The transferable
  *idea* (compress/project distant context instead of recomputing it) was preserved by folding
  it into C's "older rows via PerceiverPointEncoder" decision, which is structurally the same
  pattern as CED.

## Follow-up needed

The formal refactor plan document was never written to the plan-mode plan file — plan mode was
exited externally mid-session, before the Phase 4 (final plan) / Phase 5 (ExitPlanMode) steps
of the planning workflow. All decisions above are settled; what remains is turning them into a
concrete, file-level implementation plan.

## Addendum — scope decisions made during implementation (Part A + C)

Recorded after the plan above was approved and executed, since two real scope boundaries were
discovered while implementing that the original plan text did not anticipate.

### Older-context representation (C): deferred, not built

**What was planned:** tokens/rows falling outside the sliding window get compressed by a second
`PerceiverPointEncoder` instance into extra latents, concatenated onto `latent_condition`.

**What was actually found during implementation:** this doesn't fit a single parallel
teacher-forcing forward pass over a full sequence the way `Trainer._epoch` currently works.
Sliding-window attention is *positional* — every token position `i` has its own window boundary
`[i-W, i]` — there is no single "current position" past which everything is uniformly "older",
so there is no single set of "older tokens" to summarize once per forward pass. A working version
of this idea needs the sequence processed in **chunks**, carrying a running summary from chunk to
chunk (Transformer-XL/Compressive-Transformer-style recurrence) — a materially different training
loop, not a drop-in module addition to `Quadtron.forward()`.

**Decision:** built and verified the sliding-window mask (`HourglassTransformerBlock._causal_mask`)
and the flash-attention path (`MultiHeadAttention.scaled_dot_product_attention` via
`F.scaled_dot_product_attention`) — both tested numerically (flash vs. manual attention match to
~1e-7 for identical weights/mask; windowed mask boundaries verified exactly). Did **not** build the
older-context summarizer module, to avoid shipping dead/untested code that nothing calls — per
"no half-finished implementations." Follow-up: design the chunked training loop first (how big are
chunks, how is the running summary carried across `DataLoader` batches), then add the summarizer.

### Polytron model-level dim=3 support: tokenizer done, models not yet

**What was planned:** "Polytron's Stage 2 pointer count changes... PolytronPointerModel already
generalizes trivially" implied the whole Polytron model stack was close to dim=3-ready.

**What was actually found:** `PolytronTokenizer` (`polytron_tokenizer.py`) *is* fully dim=3-ready
and tested (cylindrical `(r,θ,z)` vertices, `corners_per_block=8`, `edge_ctrl`-based 3D chord-frame
geometry — round-trip-tested against a synthetic hex block). But the neural model that consumes
Stage-1 tokens, `polytron_vertex_model.py`'s `VertexGen`, has three 2D-only assumptions baked in:
`group_mask`'s r/sin/cos 3-token cycle (needs a 4-cycle for r/sin/cos/z), `s1_generate`'s stride-3
decode loop, and the fixed `FACECOUNTS`/`FC2N` 6-block template (`F = 6n²`) that doesn't apply to
tistos's variable block counts — the last of which was already flagged as an open question in the
plan's B section (Stage 2 needs its own stop-criterion or external conditioning for non-templated
block counts). Fixing `group_mask` alone without resolving the block-count question would give a
false sense of completeness. Flagged clearly in `polytron_vertex_model.py`'s module docstring
rather than silently left broken. Follow-up: resolve the block-count/stop-criterion question first
(this blocks `PolytronPointerModel`/`GeomHeadModel`'s dim=3 wiring too, not just `VertexGen`), then
extend the three model files together.

### Hex-block edge topology bug found and fixed during A6

While building `domain_extractor_3d.py` and validating `PolytronTokenizer` against real tistos
data, found that the dim=3 Stage-3 half-edge loop (`for k in range(cpb): p0=faces_g[k,fi];
p1=faces_g[(k+1)%cpb,fi]`) silently assumed all `corners_per_block` corners form a single cyclic
ring -- true for a planar quad (4 corners) but **wrong** for a hex block (8 corners, VTK_HEXAHEDRON
order: 0-3 bottom face CCW, 4-7 top face CCW, 12 edges, not 8). The loop was producing 8 fake
"edges" per block including non-edges like corner 3 -> corner 4 (bottom-to-top diagonal-ish, not a
real hex edge) instead of the true 12. Fixed with an explicit `_QUAD_EDGES`/`_HEX_EDGES` topology
table (`PolytronTokenizer._face_edge_pairs()`) rather than deriving edge pairs from
`corners_per_block` arithmetic. Verified against a synthetic unit-cube hex block: reconstructed
edge set now matches the true 12-edge hex topology exactly (previously produced 8 wrong pairs).

### Real-data validation results (A6 extractor + tokenizers, full 132-sample tistos set)

- **Axis convention for cartesian -> cylindrical**: confirmed against the sibling
  `domain_partition_3D/dp3d/unwrap_surface.py:8` (`theta = atan2(y, x)`, axis = z) rather than
  guessed -- `sample.npz`'s own `params` dict is empty, so this is inherited from the established
  upstream convention, not invented.
- **`domain_extractor_3d.py`**: 132/132 samples extracted with zero failures.
- **`PolytronTokenizer` (dim=3) on real data**: 132/132 exact topology. Stage-3 geometry error
  (median 2.2%, mean 3.7% relative to chord) after excluding edges with near-zero chord length (8
  out of 7312, genuinely collapsed/degenerate hex edges in the source data -- confirmed by direct
  inspection, not assumed) from the aggregate, since dividing by a near-zero chord explodes the
  relative-error metric by construction and is the same documented "Bug 2" pattern as the 2D
  pipeline's mini-edge outliers (`docs/ho_quad_transformer/06_edge_geometry_study.md`). Also fixed
  `round_trip_report`'s geometry-error resampling, which hardcoded 2 columns
  (`c[:,0]`/`c[:,1]`) and crashed outright on 3D curves.
- **`Tokenizer2D` (dim=3, Quadtron) on real data**: found and fixed a second, unrelated bug in
  `testing()`'s round-trip self-check: it aligned original vs. reconstructed vertex sets by
  `lexsort`-ing both and comparing row-by-row, which is fragile whenever two vertices are close in
  the primary sort key -- a tiny quantization nudge can flip their tie-break order, comparing
  non-corresponding vertices and reporting a large false error (observed: MSE ~0.14 via lexsort
  alignment vs. an actual nearest-neighbour distance of ~0.002, exactly the expected quantization
  step). Replaced with nearest-neighbour distance matching, which is correct regardless of
  dimension. After the fix: 114/132 exact at `quantization_levels=1024`, 120/132 at 16384 -- the
  remainder is a small number of genuinely near-coincident vertices in specific meshes (consistent
  with the same degenerate-block characteristic found in the Polytron edge-length check above, not
  a code defect), where quantization collisions merge two distinct original vertices into one after
  detokenizing. Documented rather than chased further; a production run should measure per-mesh
  vertex spacing and pick `quantization_levels` accordingly (or accept the small loss rate).

### `PerceiverPointEncoder` dormant bug found and fixed (blocks all of dim=3, not just Polytron)

Running an actual end-to-end Quadtron dim=3 training smoke test (`train.py --model-family quadtron
--dim 3`, real extracted tistos data) crashed on the very first forward pass:
`RuntimeError: mat1 and mat2 shapes cannot be multiplied (512x51 and 35x32)`. Root cause:
`point_encoder.py`'s `PerceiverPointEncoder.__init__` computed
`fourier_dim = input_dim + 4 * n_freqs`, but the `fourier_features()` function it feeds actually
produces `input_dim * (1 + 2*n_freqs)` columns (`x` concatenated with `sin_feat`/`cos_feat`, each
sized `input_dim*n_freqs`). These two formulas are algebraically **equal only when `input_dim=2`**
(`2+4n = 2(1+2n)`) -- which is the only value this code ever ran with before this refactor added
`input_dim=3`, so the bug was invisible until now. This is the earlier "A4: `PerceiverPointEncoder`
is already dimension-agnostic, verified by reading the source" claim turning out to be almost but
not quite right -- the *interface* (`input_dim` parameter) was dimension-agnostic, but one internal
formula silently assumed `input_dim=2`. Fixed in `point_encoder.py` (both the docstring/shape
comment and the `__init__` computation now match the function exactly); verified the fix is
numerically identical for `input_dim=2` (`2*(1+2n)=2+4n`, same result) before re-running.

### Verification results (Task 7)

- **3D smoke run, Quadtron**: `train.py --model-family quadtron --dim 3` (`sorting_strategy=0`,
  `use_flash_attention=True`, `sliding_window_size=64`, real 132-sample tistos data, tiny model)
  completed 2 full epochs end-to-end -- tokenization, dataset, point encoder, attention (flash +
  windowed), training loop, validation, JSONL logging, all exercised together on real data.
  `train_bpt`/`val_bpt` decreased epoch over epoch (9.53 train / 9.32 val), i.e. the model is
  actually learning, not just running without crashing.
- **3D smoke run, Polytron**: not attempted as a full `run_polytron()` call. Already known-blocked
  by the `FACECOUNTS`/`FC2N` gap documented above: `build_vertex_examples` filters to
  `Fc in {6,24,54,96}` (the 6-block template's `F=6n²` face counts), and tistos samples have 16
  blocks each -- every real 3D sample would be silently filtered out before training starts. This
  is the same open item flagged in the plan's Part B (Stage 2 needs its own stop-criterion or
  external conditioning for non-templated block counts), not a new finding; re-attempt once that's
  resolved.
- **2D regression check**: not run -- no 2D dataset file (`centered_blades_cleaned.pt`) or built
  `openmesh` extra available in this sandbox (`openmesh` needs a C++/cmake build that fails here;
  confirmed pre-existing/environment-only, unrelated to the refactor). Covered instead by
  behavior-preserving unit checks on every touched 2D code path: `Tokenizer2D` dim=2 round-trip
  still passes (`testing()` regression check), `_causal_mask` at `sliding_window_size=0` produces
  the identical mask as before the flag existed, flash-attention output matches manual attention to
  ~1e-7, and `PerceiverPointEncoder`'s fixed formula is algebraically identical for `input_dim=2`.
  A real 2D training-curve comparison against a pre-refactor run is still recommended before
  trusting this in production, once a machine with the 2D dataset + built `openmesh` is available.
- **Attention flags**: covered above (windowed mask verified exact against hand-computed expected
  visibility per position; flash vs. manual attention numerically match; both active together in
  the successful Quadtron 3D smoke run).
- **Round-trip tests**: covered above (Polytron 132/132, Quadtron 120/132 at `quantization_levels
  =16384`, both on the full real tistos set, not synthetic data).

## Task 8 — A+C gate assessment (per-family, not all-or-nothing)

The plan's gate criterion ("Quadtron and Polytron both reach a working, config-switchable 2D+3D
baseline... before any Part B work starts") is **not uniformly met** -- assessed honestly per
family rather than declared passed wholesale:

| Family | dim | Status |
|---|---|---|
| Quadtron | 3 | **Gate passed.** Real end-to-end training run (tokenize -> dataset -> point encoder -> attention incl. flash+windowed -> train/val loop -> logging), loss decreasing. |
| Quadtron | 2 | Code paths verified behavior-preserving (unit-level), but no real training run -- no 2D dataset file in this sandbox. Not a code-confidence gap, an environment/data-availability gap. |
| Polytron | 3 | **Tokenizer gate passed** (132/132 topology, real data). **Model gate not passed** -- `run_polytron()`'s Stage 1 (`build_vertex_examples`) filters to the fixed `FACECOUNTS={6,24,54,96}` template and would silently drop every tistos sample (16 blocks each, not in that set). No checkpoint can exist to fine-tune with RL. |
| Polytron | 2 | Not attempted -- no 2D domain-partition dataset file in this sandbox either. |

**Decision:** proceed with Part B (B1: `RLObjective`/`AdvantageEstimator` infrastructure), but treat
the gate as passed for **Quadtron only**. This is a refinement of, not a deviation from, the
original per-family RL scope already decided (Quadtron full curriculum, Polytron Stage-1-only) --
Polytron's Stage-1 RL was always going to need a working Stage-1 checkpoint first, and that
checkpoint now has a concretely identified, already-documented blocker (the `FACECOUNTS` template)
independent of anything RL-specific. Building `RLObjective` as family-agnostic infrastructure (it
only depends on `Policy.sample()` + a reward function, not on which model produced the policy)
means the Polytron path activates automatically once its separate, already-flagged model-level gap
is fixed -- no rework of B1 needed when that happens.

## Part B implementation results (B1-B3)

- **B1 (`RLObjective` + `AdvantageEstimator`, `objectives.py`)**: real gradient-flow test on a
  Quadtron model -- both `GroupRelativeAdvantage` (default) and `CriticAdvantage` produce finite
  losses and nonzero gradients across multiple optimizer steps, confirming the swap really is just
  a config choice (`PipelineConfig.advantage_estimator`), not a different code path.
- **B2 (`rewards.py`)**: all four Quadtron curriculum-stage reward functions (vertex/face/row/mesh)
  correctly rank a real ground-truth token sequence above garbage/short sequences on real tistos
  data. One correction to the original plan text made here: `hex_hex_metrics_*.json` (one per
  tistos machine) holds the *ground-truth* mesh's own solve quality, constant with respect to
  whatever the model generates -- using it as a live reward term would carry zero learning signal.
  Kept as `load_ground_truth_quality()` for optional example-difficulty weighting instead of a
  reward term; every actual reward is computed from the generated sequence itself.
- **B3 (`train.py`/`trainer.py` RL dispatch)**: full CLI path verified end-to-end --
  `train.py --rl-enabled --init-checkpoint <path> ...` correctly builds an RL-enabled
  `PipelineConfig`, `Trainer` loads the checkpoint (weights verified byte-identical to the saved
  state dict) and constructs `RLObjective` instead of `TeacherForcingObjective`, and a real batch
  from the real `DataLoader` flows through `RLObjective.compute()` end-to-end. A full-epoch CLI run
  was attempted but is slow on this sandbox's CPU-only setup (`Policy.sample()` has no KV-cache --
  every generation step recomputes the full forward pass over the growing sequence, O(L²) per
  rollout; with `rollouts_per_condition` sequences per batch item this adds up over a full epoch).
  Not a correctness issue -- a single `.compute()` call through the identical real-Trainer wiring
  completed in ~23s and produced a finite loss; a full epoch just needs either a GPU or a smaller
  `rl_max_length`/`rollouts_per_condition` for CPU-only smoke testing. Left as a known performance
  characteristic rather than a blocker -- KV-cached generation would be the natural fix, out of
  scope for this pass.

## Follow-up: Polytron model-level dim=3 gap closed (user-requested)

The "Polytron model-level dim=3 support: tokenizer done, models not yet" gap noted above was
closed. User question: "kann man den facecount bei Polytron nicht variable gestalten?" -- yes; the
`FACECOUNTS=[6,24,54,96]`/`FC2N` template (`F=6n²`) was specific to a 2D 6-block experiment and had
no structural reason to exist (`FaceCountEncoder` already accepts any raw count, exactly like
Quadtron/MeshtronDomain use it). Removed, and while in this code, generalized the whole Stage
1/2/3 model stack to dim=3 (not just the facecount template):

- `polytron_vertex_model.py`: `build_vertex_examples` no longer filters by a fixed facecount set;
  conditions on the raw face/block count directly. `group_mask`/`s1_generate` generalized from a
  hardcoded 3-token (r,sin,cos) cycle to a `dim`-aware 3-or-4-token cycle (+z for dim=3).
  `vertex_eval`/`fmt_eval` bucket by whatever facecounts are actually present instead of the fixed
  list.
- `polytron_pointer_model.py` / `polytron_geom_model.py`: `vert_proj`/`GeomHeadModel.head` input/
  output widths made configurable (`vert_feat_dim`, `geom_out_dim`) instead of hardcoded 3/4.
- **A second instance of the same hex-edge-topology bug** (see the A6 entry above) found in
  `polytron_chain.py`'s `faces_to_edges` and `polytron_geom_model.py`'s `build_geom_examples`:
  both assumed `corners_per_block` corners form a single ring via `(k+1) % corners_per_block`.
  Fixed by reusing `PolytronTokenizer._face_edge_pairs()` (the same topology table from the
  tokenizer fix) instead of a second, independently-wrong implementation.
- `polytron_chain.py::chain_one`: `Fexp = 6*n*n` replaced with `Fexp = n` (n is now the raw
  count, not a resolution-level proxy); the whole function made `dim`/`corners_per_block`-aware
  (cylindrical vertex reconstruction, 3D chord-frame Bézier curves for dim=3).
- **New bug found via real-data smoke testing (not caught by unit tests alone)**: `VertexGen`
  runs full self-attention directly over `tri_coordinates` with no subsampling of its own (unlike
  Quadtron's `dataset.py`, which subsamples dynamically at load time) -- fed tistos's raw
  `surface_points` (up to ~12k points/sample), this allocated 43GB on the very first training
  batch (O(N²) self-attention memory). Fixed in `domain_extractor_3d.py`: added
  `subsample_points()`, applied to Polytron's `tri_coordinates` at extraction time (default cap
  768 points, matching the 2D pipeline's natural point-cloud scale). Quadtron's copy of
  `tri_coordinates` is left at full resolution since its dataset loader already subsamples.
- **Two more hardcoded-3/4 spots found only by actually running training**, not by reading the
  code: `train_pointer.py::collate` and `polytron_geom_model.py::collate` both padded vertex
  feature tensors to a fixed width (3, or 4 for targets) instead of reading it from the batch.
  Fixed by inferring the width from `batch[0]` instead.

**Verified against real data**: ran the full 3-head chain (`run_polytron`) on real tistos samples
with previously-rejected facecounts (12, 15, 16, 19, 22, 42 blocks -- none divisible into the old
`F=6n²` template) end-to-end -- S1, S2, S3 all train (real decreasing loss), chain inference and
evaluation run without shape errors. This is a 1-epoch/tiny-model smoke test (proves the pipeline
is structurally correct, not that quality is good yet -- `count-ok 0.00` on 1 held-out sample after
1 epoch is expected, not a regression).

**Not fixed, minor**: `train_pointer.py`'s own standalone `main()` CLI (a secondary debug entry
point, not on the `polytron_chain.py` integration path) still constructs `PolytronTokenizer`/
`PointerFaceModel` without `dim`/`corners_per_block` args. `collate`/`run_epoch`, which
`polytron_chain.py` actually uses, are fixed.

## TUI (`tui.py`, user-requested)

Built with `textual` (added to `pyproject.toml`). Deliberately not a general editor for every
`PipelineConfig` field (~35 total) -- exposes the ones grouped in `config.py`'s own comments that
someone actually changes between runs (pipeline/data/model/attention/optim/RL/runtime), organized
the same way. Anything else: load a JSON config built elsewhere, or edit after "Save Config".

Training runs in a background thread; `sys.stdout`/`sys.stderr` are redirected to a thread-safe
queue during the run (captures both `print()` epoch summaries and tqdm's progress bars, which
default to stderr) that a Textual timer drains into the log pane every 200ms — chosen over wiring
`Trainer.run()`'s `on_epoch` callback because `run_polytron()` has no equivalent callback and
redirecting stdout/stderr works uniformly for both without adding new plumbing to either.

**Verified with headless Textual tests** (`App.run_test()` + `Pilot`, not just import-checked):
default form values match `PipelineConfig()` exactly; switching model family/dimension and editing
numeric/bool/`stage_layers` fields all coerce to the right Python types; save-as-JSON and load-back
round-trip correctly; clicking "Start Training" with a real tiny Quadtron dim=3 config against the
real extracted tistos data actually starts a background thread that trains and streams live log
output into the pane. "Stop" resets the UI's buttons but isn't wired to cooperatively cancel the
training thread (`Trainer`/`run_polytron` run to completion or error, no cancellation hook exists
to attach to) -- documented in the button's own log message rather than pretending it stops the run.

### Follow-up: Stop button wired to real cooperative cancellation (user-requested)

User asked to actually cancel a running (real, TUI-started) training run and hit the documented gap
above directly. Fixed rather than just explained:

- New `TrainingCancelled` exception (`trainer.py`) -- raise it from `Trainer.run()`'s existing
  `on_epoch` callback (already designed for this: its docstring says "May raise to abort early,
  e.g. optuna.TrialPruned", just never used for it) to cancel Quadtron runs between epochs. The
  `finally` block still writes the partial `RunResult`/closes the logger before it propagates.
- `run_polytron()` (`polytron_chain.py`) got an analogous but coarser `stop_check: () -> bool`
  parameter, polled between S1/S2/S3's full `train_head` calls and per held-out item in the
  chain-inference eval loop -- coarser than per-epoch because `train_head`/the three heads' own
  `run_epoch` functions have no per-epoch hook of their own without changing all three
  independently (documented as a real, known limitation in the function's own docstring, not
  glossed over).
- `tui.py`: added a `threading.Event`; Stop sets it, the `on_epoch`/`stop_check` closures check it.

**Verified the core mechanism in isolation** (tiny synthetic 8-mesh dataset, so epochs are
near-instant rather than fighting real training speed): `Trainer.run(on_epoch=...)` requested for
20 epochs, cancelled via `TrainingCancelled` exactly after epoch 2 as instructed -- confirms the
exception propagates correctly through the `finally` block and stops the run where intended. The
combined "click Stop on a live real-data TUI run" end-to-end test hit an unrelated Textual
timer/teardown race when the test harness exited before the background thread's final log line was
drained (`_drain_log_queue` firing after `run_test()`'s context closed) -- a test-harness timing
issue with tearing down while a daemon thread is still finishing, not a defect in the cancellation
logic itself (which the isolated test above proves independently), left as a known rough edge in
automated testing rather than chased further.
