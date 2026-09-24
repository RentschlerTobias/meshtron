# Training, end to end

2026-09-24. What "the training works" was allowed to mean, and what it means now.

## The gap

Every training check in this repo ran **one step** and reported that the loss
came out finite. That answers *does the code execute*, not *does training
work*. Nothing checked that a run converges, that an interrupted run can be
continued, that GRPO moves the policy for the right reason, or that a
checkpoint this repo trained can then mesh a geometry.

`scripts/test_training_e2e.py` closes that: four phases, each asserting the
claim its stage has to make.

    uv run python scripts/test_training_e2e.py        # ~2 min on the GPU

The model is deliberately small (d=128, 4 layers, 1.3 M parameters). This is a
test of the **path**, not of mesh quality; production is d=512, 12 layers,
40 M parameters.

| phase | what is asserted |
|---|---|
| 1 supervised | loss falls over a real run, validation is computed, the best-val checkpoint is written **and loads** |
| 2 resume | picks up at the right epoch with optimiser state, loss continues instead of restarting |
| 3 GRPO | real steps: KL >= 0, everything finite, and **no reward signal => no policy update** |
| 4 inference | that checkpoint, given a geometry, is carried through to a mesh or fails cleanly |

Result on an RTX 4060: 4 pass, 0 fail.

    1 supervised  loss 25.95 -> 10.89, val 16.54 -> 10.80, best-val ckpt loads
    2 resume      resumed at epoch 6, loss continued 10.89 -> 10.56 -> 9.41
    3 GRPO        reward flat at 0, update correctly 0, entropy stable
    4 inference   3 of 4 rollouts detokenized, none survived snapping

Phase 4 not reaching a mesh is the honest outcome for 1.3 M parameters: the
blockings are poor, the snap rejects them, and the pipeline **reports** that
rather than crashing. That is what is being checked.

## Three defects the test found

### 1. The GRPO entry point was not runnable

`meshtron/training/train_grpo.py` computed

```python
ROOT = os.path.dirname(os.path.abspath(__file__))   # meshtron/training/
sys.path.insert(0, os.path.join(ROOT, "scripts"))   # does not exist
```

left over from when the file sat in the repo root. Running it raised
`ModuleNotFoundError: No module named 'eval_family'`.

**Why no gate caught it.** `verify_pipeline.py` checked the module with
`from meshtron.training import train_grpo`, and `verify_pipeline.py` itself
lives in `scripts/` — so `scripts/` was already on `sys.path` and the import
resolved. The check passed *because of where the checker lives*. It now runs
the entry point as a subprocess instead. A check that imports a script does
not test the script.

### 2. GRPO could not use a token file without embedded conditioning

Token files of the family carry `surface_points`; the older ones do not, and
`train_hexarow_full.py` has `--src` to supply them by name. GRPO had no such
flag and died mid-training with `KeyError: 'vertices_polar'` from inside
`build_cloud`. It now takes the same `--src`, accepts either
`surface_points` or `vertices_polar` (both are what `surface_cloud` reads),
and when conditioning cannot be resolved at all it says so before the first
rollout instead of failing deep in a stack.

### 3. The KL anchor had the wrong sign — and manufactured gradient

```python
kl = ((logp_new - logp_ref) * mask).sum() / mask.sum()
```

This is the plain log-ratio mean. It is **not a divergence**: it goes negative
as the policy drifts, so `loss_pg + beta * kl` *rewards* leaving the reference.
Its gradient with respect to `logp_new` is a constant `beta`, which lowers the
probability of every sampled token uniformly — nothing to do with a KL.

The sharpest evidence needs no synthetic setup. A 1.3 M model earns **zero
reward on every rollout**, so every advantage is zero and the correct update is
exactly zero. Three GRPO steps, same seed, same data, only the estimator
differing:

| | mean_R | grad_norm | KL | entropy |
|---|---|---|---|---|
| naive | 0.0 | **0.242** | 0 → −0.0096 → −0.0194 | 5.120 → **5.146** |
| k3 | 0.0 | **0.0** | 0, 0, 0 | flat |

With no reward signal the naive term still produced a gradient norm of 0.24,
drove its own KL monotonically negative, and *raised* entropy — the signature
of probability mass being pushed off the sampled tokens. k3
(`exp(-r) + r - 1`, what GRPO specifies) gives exactly 0.

`--kl-estimator` now defaults to **k3**. `naive` is kept, because every
checkpoint in `data/` — `grpo_cart_step*.pt` included — was trained with it and
reproducing those runs requires it. Those checkpoints were trained with a term
that pushed the policy away from the reference whenever rewards were flat
within a group, which for a rarely-rewarding reward function is most of the
time. How much that cost is not measured here.

Phase 3 asserts `KL >= 0` and `flat reward => zero update`. Reverting the
default to `naive` makes it fail with
`KL went negative (-0.01846)`, which is the point of writing it.

## A fourth: the best-val checkpoint could not be loaded

`train_hexarow_full.py` wrote `--ckpt` with `model`, `opt`, `epoch`, `step`,
`best_val` and the bounds, but no `cfg`, `vocab` or `pad_id` — so
`load_model()` could not rebuild the model from it. The best checkpoint of an
interrupted run held the weights and not the shape needed to use them. It now
carries both, and phase 1 asserts the file loads rather than merely existing.

## --epochs is the learning-rate schedule, not a stopping point

A 600-epoch run on `hexarow_tokens_family_cart.pt` with the script's defaults
overfits hard. Best validation came at epoch 104 (3.162) and then rose without
recovering:

    epoch 104   train acc 0.325   val loss 3.162   <- best
    epoch 179   train acc 0.585   val loss 3.917
    epoch 249   train acc 0.745   val loss 5.076

That looked like it contradicted `hexarow_sft_cart_ep584.pt`, whose name says
its best validation was at epoch 584 on the same token file. The configs
explain it (`train_hexarow_full.py:364`):

```python
tot_steps = args.epochs * len(train_batches)
lr = args.lr * 0.5 * (1 + cos(pi * step / tot_steps))
```

`--epochs` sets the length of the cosine, so shortening a run does not stop it
early -- it compresses the whole decay into fewer steps:

| | peak lr | at a comparable epoch | effective lr |
|---|---|---|---|
| reference, `--epochs 3000`, stopped at 584 | 3e-4 | 584 = 19.5% of the schedule | **2.7e-4** |
| this run, `--epochs 600` | 1.3e-4 | 250 = 42% of the schedule | **8.1e-5** |

A factor of 3.4 apart.

**What this does and does not explain.** An earlier version of this note claimed
the reference avoided overfitting because a sustained high learning rate
regularises. That is not supported and the author of the reference run corrected
it: that run was *also* overfitting by epoch 584. It was aborted there because
584 was simply the last epoch that improved validation, and its checkpoint file
confirms the account -- `hexarow_sft_cart_ep584.pt` carries no `epoch` or
`best_val`, so it is not the best-val file the trainer writes but an `--out`
format file someone had to reconstruct from it, which is the same gap fix 3
above closes.

So both runs overfit. The difference is *when*: the reference kept improving on
validation until epoch 584, this run until epoch 104 -- 5.6x longer before the
turn, at a learning rate 3.4x higher. That is a measured difference in
configuration and in outcome. Calling the learning rate its cause was one
hypothesis among several, and a run with the reference values settled it:

    defaults   (lr 1.3e-4, dropout 0.17, wd 0.042)   best val 3.162 at epoch 104
    reference  (lr 3e-4,   dropout 0.1,  wd 0.01)    best val 3.320 at epoch  49

The reference recipe overfits **earlier** and to a worse optimum. The learning
rate schedule is not the lever, and `ep584` is explained without it: that run
used `--val-every 25`, so validation was sampled about 23 times across 584
epochs, and a "new best" that late is a noise minimum inside a rising trend
rather than real improvement.

What is left is the data. `hexarow_tokens_family_cart.pt` holds 683 training
samples but only **323 distinct geometries** -- an n2000 and an n8000 variant of
each machine. 323 geometries against 40 M parameters overfits by construction,
at epoch 50 to 105 whatever the schedule does.

**The script's defaults do not reproduce the production checkpoint.** They are
lr 1.3e-4, dropout 0.17, wd 0.042; `hexarow_sft_cart_ep584.pt` was trained with
lr 3e-4, dropout 0.1, wd 0.01 and `--epochs 3000`. Anyone training a model here
should copy the reference values, not the defaults, and treat `--epochs` as the
schedule length -- to train for fewer epochs, keep the schedule long and stop
the run, which the best-val checkpoint makes safe.

## Which data

`hexarow_tokens_family_cart.pt`: the coarse blockings, 683 train / 78 val,
11-25 blocks per sample (median 21), median 371 tokens. NOT
`hexarow_tokens_h05_family_cart.pt`, which is the fine variant at 36-188 blocks
and median 1776 tokens. Both reference checkpoints used the coarse file, so the
comparisons here are against models trained on the same data.

## Not addressed

- **Inverted cells** are parked by explicit decision; current mapping results
  are good enough for now. See the mapping decision log.
- **2D** remains two MISSING entries in `verify_pipeline.py`: the dataset at
  `/home/t1dde/Duty/projects/meshtron/quad_domain_data` is not wired in.
- **What the naive KL cost the existing checkpoints** is unmeasured. Answering
  it means a controlled rerun of the same GRPO schedule under both estimators.
