# AGENT.md

Guide for AI Agent in this repo.

## What

Autoregressive gen model for **2D quad meshes**. Adapts NVIDIA MeshTron paper
(`literatur/2412.09548v1.pdf`, <https://arxiv.org/html/2412.09548v1>) from 3D tri → 2D quad.
Mesh → token seq, predict token-by-token. Conditioned on point cloud + target face count.

## Commands

No build, no test framework, no requirements file. Scripts run from repo root.

```bash
python main.py        # train 1 config (hyperparams inline in main.py)
python param_test.py  # grid sweep: quantization × n_latents
python testing.py     # load checkpoint, generate mesh, plot
python validation.py  # scan checkpoints/, plot loss + param/loss heatmap
```

Deps (install manual): `torch`, `torch_geometric` (only `lexsort`), `numpy`, `matplotlib`, `seaborn`,
`pandas`, `tqdm`. GPU auto (`cuda` else `cpu`).

HPC: `queuing.sh` = PBS job, activates venv, runs `main.py`.

### Data

Lives **outside repo**: `../data/*.pt` (e.g. `structured_quad_meshes_pre_selected.pt`), via `torch.load`.
Each = PyG mesh object: `mesh.x` (vert coords, use `[:, 0:2]`), `mesh.faces` (`[4, n_quads]`),
`mesh.tri_coordinates` (`[N, 3]`; col 2 = boundary/interior flag, `2` = interior). Files not provided.

### Checkpoints

`checkpoints/<notation>/epoch_N.pt`. `<notation>` encodes all hyperparams
(`q_..._d_model_..._n_latents_..._stage_layers_4_4_4_4_4`). `checkpoints/` gitignored.
`validation.py` + `testing.py` parse hyperparams back out of dir name → naming in `Trainer.notation`
load-bearing. Keep in sync across 3 files.

## Architecture (data flow)

```
mesh (PyG) ─ Tokenizer2D ─► token seq ─┐
mesh ─ point cloud sampling ───────────┤
mesh ─ face count ─────────────────────┤
                                        ▼
                            Meshtron (meshtron.py)
   Embedding ─► HourglassTransformer ◄─ latent_condition ─► output_head ─► logits over vocab
                         ▲
   PerceiverPointEncoder + FaceCountEncoder make latent_condition
```

- **`Trainer` (trainer.py)** = real orchestrator. Builds tokenizer + data + model, runs train/val loop,
  early stop (`max_patience`), cosine LR, grad accumulation, checkpointing.
  `main.py` / `param_test.py` / `testing.py` only instantiate `Trainer`.
- **`Tokenizer2D` (tokenizer.py)** = core serialization. Per quad: 8 tokens (4 verts × 2 coords).
  Quads canonically ordered (lexsort + CCW around centroid), coords quantized to `[0, quantization_levels-1]`.
  Special: `start=Q`, `end=Q+1`, `pad=Q+2` → `vocab_size = quantization + 3`.
  **Tokens emitted `(y, x)` order**; `detokenize` swaps back — watch when editing either.
  `tokenizer.testing()` = tokenize→detokenize roundtrip check.
- **`MeshData` (dataset.py)** pre-tokenizes all meshes at init, samples point clouds (boundary kept,
  interior padded/repeated w/ noise to `n_sample_points`), returns shifted `input_tokens`/`target_tokens`.
- **`HourglassTransformer` (hourglass_transformer.py)** = U-shaped 5-stage. Shorten ×2 (coord→vertex),
  shorten ×4 (vertex→face), upsample ×4, ×2 back, residuals across hourglass. Shortening picks **last**
  element of each block (causal); upsampling repeat-interleaves then causal-shifts right. Each stage =
  self-attn blocks (`HourglassStage`) + `CrossAttentionCondition` injecting `latent_condition`.
- **Attention (attention.py + positional_encoder.py)** = RoPE. RoPE **off for cross-attn**
  (point-cloud K/V different positional space). `position_ids` subsampled in lockstep with shortening
  factors → positions stay correct per level.

## Conventions & gotchas

- Code/comments/docstrings mostly **German**. Match surrounding language.
- `is_casual` = misspelled "causal". Means causal mask, not optional.
- **`window_size`**: when set, training crops random fixed-length window, aligned to mult of 8
  (`tokens_per_face`) → never splits quad mid-face. `None` = full seq.
- Loss = `cross_entropy`, `ignore_index=pad_token`.
- **Legacy/alt files — NOT on active path.** Live path = `tokenizer.py`, `attention.py`,
  `positional_encoder.py`. Stale: `train.py` (broken legacy, undefined names),
  `tokenizer_v1.py`/`tokenizer_v2.py`, `attention_v2.py`, `positional_encoder_v2.py`,
  `rotary_positional_embedding.py`, `transformer.py`, `half_edge.py`, `quatization.py`,
  `detoken.py` (empty), `hourglass_transformer.py.bak2`. Confirm imports before editing.
</content>
