# Progress: Meshtron Curved Domain-Partition Extension

## Last Updated
2026-06-30

## Completed

### 1. domain_extractor.py ✅
- **Status:** Working
- **Location:** `/root/repos/meshtron/domain_extractor.py`
- **What it does:**
  - Loads raw meshes from `checkpoint_mesh_100.pt`
  - Extracts tangents from `edge_to_streamline` via linear fit over K=3 points
  - Computes `alpha_start`, `alpha_end`, `t_norm_start`, `t_norm_end` per directed edge
  - Converts vertices to polar coordinates `(r, θ)` relative to mesh centroid
  - Saves to `/root/repos/meshtron/domain_data.pt`
- **Test result:** 100 meshes processed successfully; sample mesh has 12 verts, 6 faces, 24 edges

### 2. tokenizer_domain.py ✅
- **Status:** Implemented; bug fixed
- **Location:** `/root/repos/meshtron/tokenizer_domain.py`
- **What it does:**
  - **3 sorting strategies:** 0=no compression, 1=row-compressed, 2=vertex-first
  - **3 embedding modes:** 0=split vocab ranges, 1=shared vocab, 2=separate embeddings
  - **Sincos encoding:** 2 tokens per angle (wrap-around safe)
  - **8 tokens per vertex place:** `[r, θ_sin, θ_cos, t_norm, α_in_sin, α_in_cos, α_out_sin, α_out_cos]`
  - Special tokens: start, end, pad, eor, sep
- **Bug fixed:** `_order_faces` had wrong centroid shape (faces [4,6] vs [6,4] confusion). Fixed by using `faces.T` before indexing.
- **Test results:**
  - Strategy 0 (no compression): 208 tokens, vocab=325 (mode 0), vocab=69 (mode 1), vocab=101 (mode 2)
  - Strategy 1 (row-compressed): 163 tokens (3 rows), saves ~20%
  - Detokenize returns 24 vertex places (correct: 6 faces × 4 vertices = 24)

### 3. dataset_domain.py ✅
- **Status:** Working
- **Location:** `/root/repos/meshtron/dataset_domain.py`
- **What changed:** Added `_sample_point_cloud` with boundary-priority + noisy interior fill (same logic as `dataset.py`). Fixed `max_seq_length` placement bug.
- **Test result:** 100 meshes tokenized; batch shape `[8, 655]` tokens, `[8, 1500, 2]` point clouds.

### 4. domain_extractor.py — Edge Tangents Refactor ✅
- **Status:** Refactored to store per-edge tangents instead of aggregated per-vertex
- **What changed:** Replaced `edges_polar` dict with `edge_index` [2, n_edges] and `edge_tangents` [n_edges, 4] tensors
- **Why:** Old version overwrote `alpha_out`/`alpha_in` when a vertex had multiple outgoing edges (bug)
- **Test result:** 100 meshes reprocessed successfully; n_edges=17-24 per mesh

### 5. tokenizer_domain.py — Edge Lookup Fix ✅
- **Status:** Updated to use `edge_index`/`edge_tangents`
- **What changed:** `_get_alpha_in`, `_get_alpha_out`, `_get_t_norm` now search directed edges with correct angle reversal for reversed edges
- **Test result:** Tokenization works (208 tokens for strategy 0); detokenize returns 24 places + 6 faces

### 6. reconstruct_domain.py ✅
- **Status:** Working end-to-end
- **Location:** `/root/repos/meshtron/reconstruct_domain.py`
- **What it does:**
  - Polar → Cartesian conversion
  - Merge duplicate vertices via proximity clustering (threshold 1e-3)
  - Build unique faces from place indices
  - Generate Hermite spline points per edge (50 points)
  - Create `blocked_mesh` Data object
  - Call `Transfinite_Interpolation` → quad mesh
- **Test result:** Round-trip successful: 12 verts → 17 edges → blocked mesh → 532 verts, 486 faces (transfinite divisions=5)

### 7. domain_embedding.py ✅
- **Status:** Implemented
- **Location:** `/root/repos/meshtron/domain_embedding.py`
- **Modes:** 0/1 = single `nn.Embedding`; 2 = separate scalar + angle embeddings
- **Test result:** Forward pass OK with mode 0, 1, 2

### 8. meshtron_domain.py ✅
- **Status:** Implemented
- **Location:** `/root/repos/meshtron/meshtron_domain.py`
- **Features:** DomainEmbedding, smaller face-count range (1-50), shorter max_seq_length (~1000)
- **Test result:** Forward pass OK, logits shape [batch, seq_len, vocab_size]

### 9. train_domain.py ✅
- **Status:** Implemented
- **Location:** `/root/repos/meshtron/train_domain.py`
- **Features:** TrainerDomain class with CosineAnnealingLR, checkpointing, best-model tracking
- **Test result:** Loss computed successfully (5.22 on first batch)

### 10. First Training Run ✅
- **Status:** Completed (10 epochs, small model for smoke test)
- **Model:** d_model=128, n_heads=4, stage_layers=(2,4,6,8,10), batch=4
- **Loss curve:**
  - Epoch 1: Train 3.59 | Val 2.72
  - Epoch 5: Train 1.29 | Val 1.51
  - Epoch 10: Train 0.95 | Val 1.37
- **Convergence:** Yes — loss drops steadily. Val loss plateaus ~1.4 (expected with only 100 meshes)

### 11. Generation Pipeline Smoke Test ✅
- **Status:** Pipeline works end-to-end
- **Observation:** Model generates ~207 tokens (GT=208) — length roughly correct
- **Issue:** Generated geometry is degenerate (69 nodes, no quads) after only 10 epochs
- **Root cause:** Insufficient training; needs more epochs + larger model + GPU

## Next Immediate Tasks (Priority Order)
1. **Longer training run** (100+ epochs) with larger model (d_model=512) on GPU
2. **Implement geometric validation metrics** (Chamfer distance, edge curvature error)
3. **Optuna sweep** over strategies (0,1,2) and embedding modes (0,1,2)
4. **Compare generated vs ground-truth meshes** qualitatively and quantitatively
5. **Add curriculum learning** — start with coarse quantization, increase resolution

## Blockers
- None currently — all core infrastructure is in place
- GPU would speed up training (currently CPU-only)
- Some meshes in dataset have 16-18 faces instead of 6 (domain partition outliers)

## Key Design Docs
- `/root/repos/meshtron/plan.md` — architecture plan
- `/root/repos/meshtron/AGENT.md` — original repo guide
- `/root/repos/meshtron/progress.md` — this file

## Key Design Docs
- `/root/repos/meshtron/plan.md` — this plan
- `/root/repos/meshtron/AGENT.md` — original repo guide
- Handoff v2: `/tmp/meshtron_domain_partition_handoff_v2.md`

## Branches
- Current checkout: `main`
- Upstream: `origin/tokenizer-sorting` (more recent, has sorting strategies)
- Recommendation: work on `main`, cherry-pick from `tokenizer-sorting` if needed
