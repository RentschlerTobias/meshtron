# Full Optimization Pipeline

## Overview

For each sorting method (0, 1, 3):
1. Run 3-stage Optuna hyperparameter optimization (A → B → C)
2. Take the best config, train with 3 seeds
3. Save best model for inference testing
4. Final comparison across all sorting methods

## Sorting Methods

| Strategy | Description |
|----------|-------------|
| 0 | Baseline (lexicographical) |
| 1 | Row traversal with repetitive tokens |
| 3 | Row traversal without repetitive tokens (most efficient) |

## Optimization Stages

### Stage A: Architecture
Optimizes: `d_model, n_heads, n_stages, stage_layers, n_latents`

| Parameter | Search Space | Default |
|-----------|--------------|---------|
| d_model | [256, 384, 512, 768] | 512 |
| n_heads | d_model // 64 | auto |
| n_stages | [2, 3, 4, 5] | 3 |
| stage_layers | [2, 4, 6] per stage | [8,8,8] |
| n_latents | [4, 8, 16, 32, 64] | 64 |

**Note on n_stages:** Optuna internally optimizes `n_stages` and `layer_{i}` to determine the architecture.
However, `TrainingConfig` does not have an `n_stages` parameter - the number of stages is implicitly
determined by `len(stage_layers)`. When merging Optuna's `best_params` into a `TrainingConfig`,
`n_stages` and individual `layer_{i}` parameters are converted to a single `stage_layers` tuple.

**Trials:** 30
**Epochs per trial:** 15
**Early stopping patience:** 5
**Expected time:** ~67 min per sorting method (with 2 parallel)

### Stage B: Hyperparameters
Optimizes: `learning_rate, warmup_steps, dropout, weight_decay`

| Parameter | Search Space |
|-----------|--------------|
| learning_rate | [1e-5, 5e-4] (log scale) |
| warmup_steps | [0, 2000] (step=100) |
| dropout | [0.0, 0.3] |
| weight_decay | [0.0, 0.1] |

**Trials:** 30
**Epochs per trial:** 15
**Early stopping patience:** 5
**Expected time:** ~67 min per sorting method

### Stage C: Training Dynamics
Optimizes: `batch_size, accumulation_steps, learning_rate`

| Parameter | Search Space |
|-----------|--------------|
| batch_size | [8, 16, 24] |
| accumulation_steps | [1, 2, 4] |
| learning_rate | [1e-5, 5e-4] (log scale) |

**Trials:** 20
**Epochs per trial:** 25
**Early stopping patience:** 7
**Expected time:** ~75 min per sorting method

## Optuna Settings

- **Pruner:** MedianPruner (n_warmup_steps=2)
- **Sampler:** TPESampler (seed=0)
- **Direction:** minimize (val_bpt)

## Storage Locations

```
/sandbox/
├── sweep_s{0,1,3}_{a,b,c}.db     # Optuna databases
├── sweep_final_compare.db        # Final comparison study
├── best_configs/                 # Best configs per sorting
│   ├── s0_best.json
│   ├── s1_best.json
│   └── s3_best.json
├── final_models/                 # Final trained models (3 seeds each)
│   ├── s0/
│   ├── s1/
│   └── s3/
├── reports/                      # JSON summaries + plots
└── logs/                         # Execution logs
```

## Execution Timeline

```
Stage A: S0 → S1 → S3           (~67 min each)
Stage B: S0 → S1 → S3           (~67 min each)
Stage C: S0 → S1 → S3           (~75 min each)
Final Comparison: 3 seeds × 3 sortings  (~40 min)
---
Total: ~16 hours (2 parallel trials)
```

## Final Comparison

- **Study:** `meshtron-final-comparison`
- **Design:** 3 configurations (S0_best, S1_best, S3_best) × 3 seeds = 9 runs
- **Metrics:**
  - Val BPT (bits per token)
  - Bits per mesh (compression)
  - Inference time (ms)
  - Training time (hours)
  - Peak GPU memory (MB)
  - Model parameters

## Reports

Generated reports include:
- JSON summary per stage and sorting method
- Convergence plots (Optuna study visualization)
- Final comparison table
- Best model configurations

## Hardware

- **GPU:** RTX 3090 (24GB VRAM)
- **Parallel Trials:** 2
- **Storage:** Local disk (/sandbox/)
