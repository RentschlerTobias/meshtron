# Meshtron

Autoregressive transformer for 2D quadrilateral mesh generation. Conditioned on a point cloud and a target face count, it generates a token sequence describing quad faces.

## What lives here

"Meshtron" is the umbrella project name. Two model families are actively
maintained here, independent of each other (neither imports the other's
training code):

| Family | Entry point | What it does | Selector |
|---|---|---|---|
| **Quadtron** (2D/3D quads) | `train.py` | Point cloud + face count -> flat quad-token sequence, row-encoded | `--sorting-strategy {0,1,2,3}` |
| **Polytron** (two-stage) | `polytron_chain.py` | Topology split from geometry: S1 vertices -> S2 pointer faces -> S3 HO geometry | `--ep1/--ep2/--ep3`, `--load-s1`, `--init` |

Polytron is deliberately standalone: `polytron_tokenizer.py` and the
`polytron_*_model.py` heads do not touch Quadtron's tokenizer classes, so the
two pipelines keep working unchanged independently.

**MeshtronDomain** (`meshtron_domain.py`, `tokenizer_domain.py`,
`train_domain.py`, `domain_trainer.py`, `dataset_domain.py`,
`inference_domain.py`, `domain_embedding.py`, and the `plot_domain_*.py`
scripts) is **deprecated**: a failed experiment (severe overfitting, invalid
generated output — see
`docs/ho_quad_transformer/01_current_model_and_diagnosis.md`) that motivated
building Polytron as the structural fix. Kept in the repo for reference, not
maintained or extended; each of those files carries a `DEPRECATED` note at the
top.

Both families are 2D+3D, one config (`PipelineConfig`, `config.py`) — see
`docs/decisions/2026-09-13-meshtron-refactor-plan.md` for the full rationale,
what was found/fixed along the way, and known gaps.

## Running

No package build. Run scripts directly.

**TUI** (`tui.py`, needs `textual` — `uv sync`): pick model family/dimension,
edit the config fields that matter day to day, pick a dataset file, save/load
config as JSON, start training with a live log pane.

```bash
python tui.py
```

**CLI:**

```bash
# train.py is the single entry point for BOTH families, 2D and 3D:
python train.py --model-family quadtron --dim 2                  # defaults
python train.py --model-family quadtron --dim 3 --data-path quadtron_data_3d.pt
python train.py --model-family polytron --dim 3 --corners-per-block 8 \
    --data-path polytron_data_3d.pt
python train.py --config x.json                                  # CLI flags override fields
python train.py --model-family quadtron --sorting-strategy 2 \
    --use-flash-attention --sliding-window-size 512
python train.py --model-family quadtron --rl-enabled \
    --init-checkpoint runs/<hash>/best.pt --rl-curriculum-stage mesh

python validation.py            # inspect checkpoints and plot training history

# polytron_chain.py also still works standalone (same flags as before):
python polytron_chain.py --ep1 20 --ep2 25 --ep3 25
python polytron_chain.py --dim 3 --corners-per-block 8 --load-s1 <checkpoint>

# domain_extractor_3d.py: sample.npz (tistos) -> quadtron_data_3d.pt / polytron_data_3d.pt
python domain_extractor_3d.py --src ../domain_partition_3D/data/tistos_domain_partition
```

Dependencies are pinned in `pyproject.toml` (uv, CUDA 12.8 wheels): `uv sync`.
`openmesh` needs a C++ build and is optional -- only the `half_edge` modules use
it, not the training paths: `uv sync --extra mesh`.

## Datasets

Datasets are **not** stored in this repository. `.gitignore` excludes `*.pt`,
and the entry points expect the files next to the scripts:

| File | Used by | Config field |
|---|---|---|
| `centered_blades_cleaned.pt` | `train.py` | `TrainingConfig.data_path` |
| `domain_data_aug.pt` | `polytron_chain.py` | `--data` |
| `meta_mesh.pt` | `testing.py` | -- |

(`domain_data_10k.pt` / `train_domain.py` was MeshtronDomain-only; deprecated, see above.)

Point `--data-path` at wherever you keep them, or drop them into the working
directory before starting a run.

## Module layout

### Training pipeline
- `config.py` — `TrainingConfig` dataclass: single source of truth for all hyperparameters; hashable, JSON-serializable.
- `reproducibility.py` — `set_seed`, DataLoader generator and worker init for deterministic runs.
- `metrics.py` — `TokenLossAccumulator` and `EpochMetrics`. Computes NLL, bits-per-token and perplexity weighted by valid (non-pad) tokens, so values are comparable across batch sizes and sequence lengths.
- `policy.py` — `Policy` wrapper around `Quadtron`. `logits()` for teacher forcing today, `sample()` is in place for a future RL phase.
- `objectives.py` — `Objective` ABC + `TeacherForcingObjective` (cross-entropy, sum-reduction, pad-ignored). Returns `(loss, loss_sum, n_tokens)` so the per-step gradient is on per-token scale while logging stays unbiased.
- `logger.py` — `JSONLLogger`. Per run: `runs/<config-hash>/{config.json, metrics.jsonl, result.json}`.
- `trainer.py` — `Trainer(cfg).run() -> RunResult`. Linear warmup + cosine schedule, bf16/fp16 autocast, correct gradient accumulation, opt-in checkpointing.
- `train.py` — CLI entry point. Builds `TrainingConfig` from defaults / JSON / flags and calls `Trainer.run()`.

### Model
- `quadtron.py` — `Quadtron` model: token embedding + point encoder + face-count encoder, fed into the transformer with causal self-attention and cross-attention to the latent condition.
- `hourglass_transformer.py` — currently a flat transformer (despite the historical name): each stage runs at full sequence length, followed by cross-attention conditioning. No shortening / upsampling, so no information leaks through downsampling.
- `attention.py`, `positional_encoder.py` — multi-head attention with RoPE; `is_causal` flag controls masking.
- `point_encoder.py` — `PerceiverPointEncoder`: cross-attention over sampled boundary + interior points to a fixed-size latent set, with Fourier features and pre-norm.
- `faceCount_encoder.py` — sinusoidal encoding of the target face count.
- `embedding.py` — token embedding (positional encoding currently disabled there; positions handled inside attention via RoPE).

### Data
- `tokenizer_v2.py` — current tokenizer. 8 tokens per quad face (4 vertices × 2 coordinates), each quantized into discrete levels (default 256). Vocabulary size = `quantization_levels + 3` (BOS, EOS, PAD). `tokenizer.py` is the legacy v1.

  Four ordering strategies are implemented and selectable via `--sorting-strategy` (`_order_quads`); these are the `s0`–`s3` arms of the AIFLUIDS sorting study in `runs/`:

  | Strategy | Ordering | Emission |
  |---|---|---|
  | `0` | lexicographical (baseline) | uncompressed |
  | `1` | adjacent-face directed row ordering (default) | uncompressed |
  | `2` | adjacent rows | row-compressed |
  | `3` | adjacent rows, left-to-right | row-compressed |

- `tokenizer_domain.py` — **deprecated**, MeshtronDomain-only. Independent `sorting_strategy` axis: `0` = no compression, `1` = row-compressed, `2` = vertex-first, combined with `embedding_mode` (`0` split vocab, `1` shared, `2` separate).
- `polytron_tokenizer.py` — `PolytronTokenizer` for Polytron. Emits unique block corners once as quantized `(r, theta)`, then each quad as four pointers into that vertex list, so face validity holds by construction.
- `dataset.py` — `MeshData`. Tokenizes meshes, samples a fixed-size point cloud (boundary first, then interior, with noise replication if interior is small), and produces shifted `(input_tokens, target_tokens)` pairs for next-token prediction.

## Loss and metrics

Cross-entropy is computed with `reduction='sum'` and divided by the number of non-pad target tokens. Two consequences:

1. The backward signal is on per-token scale, so the optimal learning rate is largely portable across `batch_size` and `accumulation_steps`.
2. Logged metrics are token-weighted means (`Σ nll_i / Σ tokens`), not means-of-batch-means. They are comparable across runs with different batch sizes, sequence lengths and padding ratios.

Reported per epoch:
- `nll_per_token` — natural-log NLL per valid token.
- `bits_per_token` — same in bits (`nll / ln 2`). Useful upper bound: `log2(vocab_size)` ≈ 8.0 for `quantization=256`.
- `perplexity` — `exp(nll)`.

The tqdm postfix (`bpt=…`) shows the running token-weighted bits-per-token from the start of the current epoch. The outer bar tracks epochs against `cfg.num_epochs` and stops early if no validation improvement for `early_stopping_patience` epochs.

## Run artefacts

```
runs/<config-hash>/
    config.json     # full TrainingConfig snapshot
    metrics.jsonl   # one JSON record per logged epoch
    result.json     # best_val_bpt, best_epoch, total runtime, ...
    best.pt         # only if cfg.save_best
    last.pt         # only if cfg.save_last
```

The config hash is a deterministic 8-char digest of all fields, so identical configs share a directory and accidental duplicates are obvious.

## Conventions

- All hyperparameters live in `TrainingConfig`. Don't hardcode them inside `Trainer`, `Quadtron` or sweeps.
- Don't average per-batch loss values across batches; use `TokenLossAccumulator`.
- New training objectives plug in by implementing `Objective.compute(batch, policy) -> ObjectiveOutput`. The trainer is objective-agnostic.
- Checkpointing is off by default; turn on explicitly per run via `--save-best` / `--save-last`.

## Known broken

These were committed mid-edit on the Plan-B branch and do not parse. They came
across unchanged in the merge and need a fix before use:

- `test.py:15` — `for mesh i meshes:` (missing `in`), and the following `if` is mis-indented
- `testing.py` — unterminated f-string, reported at line 61
- `validation.py:155` — empty `else:` block, body not indented

## Hardware notes

`precision="bf16"` is the default and works on Ampere (RTX 3090) and newer. Use `"fp16"` only on hardware without bf16 (the `GradScaler` is wired up for this case). `"fp32"` for debugging.
