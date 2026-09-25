# Refactor proposal

Written 2026-09-24 while building the showcase. It is a proposal, not a change:
moving 59 top-level modules is wide and hard to undo, and which of several
parallel generations survives is a call for whoever owns the work, not for an
unattended session. `scripts/repo_inventory.py` regenerates every number here.

## What the repo actually looks like

"Cluttered" turns out to be the wrong diagnosis. Of 59 top-level modules, 52
are reachable from some entry point:

```
111 modules total (59 top level, 45 in scripts/, 6 in showcase/)
67 carry a __main__ guard -- most of scripts/ is one-shot tooling
genuinely orphaned: 7 modules, 1688 lines
```

The orphans:

```
tokenizer          460   superseded by tokenizer_v2
validation         325
blade_inject       317
testing             95
half_edge_v1        94   superseded by half_edge
test                33
face_assignment    364   committed 2026-09-24 as an experiment, deliberately
                         not wired into the production path
```

So the problem is not dead code. It is that several generations of the same
idea sit next to each other under names that do not say which is current:

```
tokenizer            460  orphan          tokenizer_v2          774  3 importers
half_edge_v1          94  orphan          half_edge             138  1 importer
domain_extractor     211  0 importers     domain_extractor_3d   222  4 importers
augment_subdivide    371  0 importers     augment_subdivide_3d  596  0 importers

polytron_tokenizer   677  9 importers     hexa_row_tokenizer    755  22 importers
polytron_chain       391                  polytron_geom_model   344
polytron_vertex_model 407                 polytron_pointer_model 267

train                 96  0 importers     train_smoke           162  0 importers
train_pointer        206  1 importer      train_hexarow_smoke   155  0 importers
trainer              404  3 importers     train_hexarow_full    485  7 importers
train_grpo           263  0 importers
```

`train_hexarow_full.py` is the live trainer, and it also *defines the model* --
`PointEncoder`, `Block` and `GPTCond` live inside the training script, which is
why `generate.py`, `scripts/eval_family.py` and four more modules import from
it. That is the single worst coupling in the repo: you cannot load a checkpoint
without importing the trainer.

## Proposed layout

```
meshtron/
  model/          GPTCond, PointEncoder, Block, attention, embeddings
                  (lifted out of train_hexarow_full.py -- everything that
                   loads a checkpoint stops depending on the trainer)
  data/           tokenizers, conditioning, dataset, augmentation
  geometry/       geometry_features, curve_model, seam_graph, edge_curves,
                  patch_paths, curved_bridge, block_mapping, tfi_bridge
  training/       train_hexarow_full, train_grpo, rewards, objectives, policy
  legacy/         the polytron generation, tokenizer, half_edge_v1, the older
                  trainers -- kept, not deleted, clearly labelled
  scripts/        unchanged, one-shot tooling
  showcase/       unchanged, the walkthrough
```

## Order, if it is done

1. **Lift the model out of `train_hexarow_full.py` into `model/`.** Highest
   value, smallest blast radius: seven modules import the trainer only to get
   `GPTCond`. Leave a re-export in the old place so nothing breaks in one step.
2. **Run the five gates.** `test_slot_parity`, `smoke_mesh_validation`,
   `test_conditioning_parity`, `test_rewards_hexarow`, `test_block_mapping`.
   They have to stay exit 0 after every move, which is what makes an
   incremental refactor safe here.
3. **Move the geometry group.** It is self-contained and the conform pipeline
   covers it, so a regression shows up immediately in
   `scripts/run_conform_batch.py`.
4. **Move the data group.** Watch `test_slot_parity` and
   `test_conditioning_parity` -- they exist precisely to catch tokenizer and
   conditioning drift.
5. **Decide the legacy set last**, with someone who knows which generation is
   worth keeping. `polytron_*` still has importers; `train`, `train_smoke`,
   `train_pointer` have none.

## Things not to touch

`trainer.py`, `scripts/eval_stop_rate.py` and `train_hexarow_smoke.py` are
off-limits by standing instruction. `trainer.py` has three importers, so the
legacy group cannot be moved wholesale without first checking what still needs
it.

## Already done

- `showcase.py` removed: five lines of shell notes with a `.py` extension,
  sitting at the root where it collided with the `showcase/` package. The two
  commands now live in `showcase/README.md`.
- `scripts/repo_inventory.py` added, so the numbers above can be checked rather
  than believed.
