Goal
- Extend Meshtron autoregressive transformer to predict coarse domain partitions with curved edges (Hermite splines + transfinite interpolation) instead of fine linear quad meshes.
Constraints & Preferences
- Tokenize vertices and tangent vectors in polar coordinates for unified vocabulary.
- Use sinus/cosinus pair encoding for angles to avoid wrap-around.
- Tangent extraction: linear fit over K=3 streamline points with length tokenization.
- Test 3 sorting strategies (0=no compression, 1=row-compressed, 2=vertex-first+explicit faces) and 3 embedding modes (0=split vocab ranges, 1=shared vocab, 2=separate embeddings).
- Training: pure token-level cross-entropy; geometric validation only at inference.
- transfinite_divisions external/fixed, not part of transformer sequence.
- openmesh unavailable in /root/venv/ — pure-Python face ordering required.
Progress
Done
- domain_extractor.py: extracts tangents from edge_to_streamline, converts vertices+tangents to polar, saves edge_index/edge_tangents tensors (per-edge, not aggregated per-vertex). Reprocessed 100 meshes successfully.
- tokenizer_domain.py: all 3 strategies, 3 embedding modes, sincos encoding, polar token schema (8 tokens per vertex place). Uses edge_index/edge_tangents with correct angle reversal for reversed edges.
- dataset_domain.py → DomainMeshData: point-cloud sampling with boundary-priority + noisy interior fill, normalization to [-1, 1]. Batch shapes verified: [8, 655] tokens, [8, 1500, 2] point clouds.
- reconstruct_domain.py: end-to-end round-trip working — polar→cartesian, duplicate-vertex merge (threshold 1e-3), unique face building, Hermite splines (50 points/edge), blocked_mesh Data object, transfinite interpolation via Gmsh.
- domain_embedding.py: 3 modes (0/1=single nn.Embedding, 2=separate scalar+angle embeddings). Forward pass verified for all modes.
- meshtron_domain.py: adapted for short sequences (~1000 max), small face-count range (1–50), uses DomainEmbedding.
- config.py: extended with DomainTrainingConfig dataclass (compatible with TrainingConfig pattern).
- domain_trainer.py: modern trainer matching trainer.py API — Policy wrapper, TeacherForcingObjective, TokenLossAccumulator, JSONLLogger, bf16/fp16 autocast, linear-warmup+cosine scheduler, checkpointing.
- plot_domain_pipeline.py: 6-subplot figures (input, polar, tokens, blocked mesh + Hermite, transfinite, GT vs reconstructed).
- plot_training.py: token distribution, sequence length histogram, training loss curves.
- First training smoke test: 10 epochs, d_model=128, CPU. Loss: 3.59→0.95 (train), 2.72→1.37 (val). Model learned sequence length (~207 vs GT 208 tokens).
- Round-trip test successful: 12 verts → 17 edges → blocked mesh → 532 verts, 486 faces (transfinite divisions=5).
In Progress
- domain_trainer.py smoke test on domain_partition branch (just written, not yet executed).
Blocked
- (none) — all core infrastructure complete. GPU would accelerate training but is not required.
Key Decisions
- Validation: A (token-level only) — geometric reconstruction deferred to inference.
- Face encoding: A and B — both explicit indices and implicit row-traversal will be tested.
- Angle encoding: sincos pair (2 tokens) — eliminates wrap-around problem; test direct quantization as alternative in sweep.
- Embedding mode default: 0 (split vocab ranges) — r/t_norm in 0, Qr-1, angles in Qr, ....
- Tangent computation: per-edge storage — edge_index 2, n_edges + edge_tangents n_edges, 4 instead of dict; prevents overwrite when vertex has multiple edges.
- Branch: domain_partition — based on origin/tokenizer-sorting (most recent branch with tokenizer_v2.py, half_edge.py, sweep.py, config.py, metrics.py, objectives.py, policy.py, logger.py).
Next Steps
1. Run domain_trainer.py smoke test to verify compatibility with modern Trainer pattern.
2. Longer training run (100+ epochs, d_model=512) on GPU.
3. Implement geometric validation metrics (Chamfer distance, edge curvature error).
4. Optuna sweep over strategies (0,1,2) and embedding modes (0,1,2) using modern sweep.py framework.
5. Test autoregressive generation → reconstruct_domain round-trip with trained model.
6. Add curriculum learning: start with coarse quantization, increase resolution.
Critical Context
- Active branch: domain_partition (from origin/tokenizer-sorting).
- origin/tokenizer-sorting has modern infrastructure: config.py, sweep.py, logger.py, metrics.py, objectives.py, policy.py, trainer.py, inference.py, tokenizer_v2.py, half_edge.py, compare_strategies.py.
- domain_partition branch reuses all modern infrastructure; only domain-specific files are new.
- Sample data: /root/repos/meshtron/checkpoint_mesh_100.pt (100 meshes, 12 verts, 6 faces typical, some outliers with 16–18 faces).
- Preprocessed data: /root/repos/meshtron/domain_data.pt (polar verts + edge tangents + point clouds).
- Token schema per vertex place: 8 tokens = [r, θsin, θcos, t_norm, α_in sin, α_in cos, α_out sin, α_out cos].
- Special tokens: start, end, pad, eor, sep.
- Sequence lengths: median 208 tokens, mean 256 (outliers up to ~600 tokens for 18-face meshes).
- Venv: /root/venv/bin/python (torch + PyG available, no CUDA in current environment).
- Transfinite_Interpolation.__init__ calls generate() and get_mesh() automatically; must not call generate() twice (Gmsh not re-initializable).
Relevant Files
- /root/repos/meshtron/domain_extractor.py: extracts polar coordinates and per-edge tangents from raw domain_partition meshes.
- /root/repos/meshtron/tokenizer_domain.py: core domain tokenizer with 3 strategies, 3 embedding modes, sincos angle encoding, edge-based tangent lookup.
- /root/repos/meshtron/dataset_domain.py: DomainMeshData dataset with point-cloud sampling and [-1, 1] normalization.
- /root/repos/meshtron/reconstruct_domain.py: full reconstruction pipeline — polar→cartesian, vertex merge, Hermite splines, transfinite interpolation.
- /root/repos/meshtron/domain_embedding.py: embedding layer supporting 3 modes (split/shared/separate).
- /root/repos/meshtron/meshtron_domain.py: Meshtron variant for domain partitions (short sequences, small face counts).
- /root/repos/meshtron/config.py: DomainTrainingConfig dataclass added to modern config system.
- /root/repos/meshtron/domain_trainer.py: modern trainer compatible with trainer.py API.
- /root/repos/meshtron/plot_domain_pipeline.py: 6-phase pipeline visualizations.
- /root/repos/meshtron/plot_training.py: training metrics and token distribution plots.
- /root/repos/meshtron/domain_data.pt: preprocessed 100-mesh dataset.
- /root/repos/domain_partition/tools/transfinite_interpolation.py: Gmsh-based transfinite interpolation.
- /root/repos/meshtron/plan.md: architecture plan and design decisions.
- /root/repos/meshtron/progress.md: detailed progress tracking.
