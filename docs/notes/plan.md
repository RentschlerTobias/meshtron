# Plan: Meshtron Curved Domain-Partition Extension

## Goal
Extend Meshtron to predict **coarse domain partitions with curved edges** (not fine linear quads). Use Hermite splines for reconstruction, then apply transfinite interpolation for final mesh.

## Architecture

### Data Flow
```
domain_partition mesh (.pt)
    ├─ blocking_nodes (12, 2) ──► domain_extractor.py ──► polar coords + tangents
    ├─ blocking_faces (4, 6)   ──► tokenizer_domain.py ──► token sequence (~150-300 tok)
    ├─ edge_to_streamline      ──► tangent extraction (K=3 fit)
    └─ tri_coordinates         ──► point cloud conditioning

Token sequence ──► Meshtron (adapted) ──► logits ──► autoregressive generation
Generated tokens ──► detokenize_domain.py ──► Hermite splines ──► transfinite interpolation
```

## Design Decisions (Resolved)

| Q | Topic | Decision |
|---|-------|----------|
| Q1 | Validation | A — token-level cross-entropy only; geometric check at inference |
| Q2 | Sorting | All 3 strategies: A (no compression), B (row-compressed), C (vertex-first + explicit faces) |
| Q3/Q4 | Embedding modes | All 3: A (split vocab ranges), B (shared vocab), C (separate embeddings) |
| Q5 | `transfinite_divisions` | External/fixed, not in model |
| Q6/Q7 | Tangent computation | Fit K=3 points; tokenize with length (norm + sincos angle) |
| Q8 | Face indices | Test both explicit (A) and implicit (B) |

## Token Schema (per vertex place)
`[r, theta_sin, theta_cos, t_norm, alpha_in_sin, alpha_in_cos, alpha_out_sin, alpha_out_cos]` = 8 tokens

## Files

### New
1. `domain_extractor.py` — Preprocess raw domain_partition data ✅
2. `tokenizer_domain.py` — Tokenize domain partitions (polar + sincos + strategies) ✅
3. `dataset_domain.py` — Dataset with point-cloud padding ⚠️ (needs bugfix)
4. `detokenize_domain.py` / `reconstruct_domain.py` — Reverse pipeline
5. `domain_embedding.py` — Embedding wrapper supporting 3 modes
6. `train_domain.py` / `sweep_domain.py` — Training scripts

### Modified
7. `meshtron.py` — Swap `nn.Embedding` for `DomainEmbedding`, adjust `vocab_size`
8. `trainer.py` — Adapt for short sequences, no window_size

## Milestones
- [x] Data extraction working
- [x] Tokenizer implemented (3 strategies, 3 modes, sincos)
- [x] Centroid bug fixed in `_order_faces`
- [ ] Dataset point-cloud padding fix
- [ ] Round-trip reconstruction test (tokenize → detokenize → Hermite → mesh)
- [ ] DomainEmbedding wrapper
- [ ] First training run (Strategy 0, Mode 0)
- [ ] Optuna sweep over strategies & modes

## Dependencies
- `torch`, `torch_geometric` (available in `/root/venv/`)
- `openmesh` **NOT** installed; pure-Python face ordering used instead
- Gmsh used only in `domain_partition` for transfinite interpolation

## Data
- Raw: `/root/repos/meshtron/checkpoint_mesh_100.pt`
- Preprocessed: `/root/repos/meshtron/domain_data.pt`
