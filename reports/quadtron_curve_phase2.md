# Quadtron curve stage — the edge stage on the 3D block generator

Date: 2026-10-01
Branch: `polytron-main`
Supersedes the boundary-shell reading in the first version of this file; see
"Correction" below.

## Correction: what the 3D "Quadtron" is

The 3D block generator is **`GPTCond` + `HexaRowTokenizer`**, not the
`Tokenizer2D` path:

- `HexaRowTokenizer` (`meshtron/data/hexa_row_tokenizer.py`) emits **full hex
  blocks** (8 corners per head block, row-planes over shared exit faces, axial
  pairing). It wraps `PolytronTokenizer(dim=3, corners_per_block=8)`, whose
  vocab is `(512+2·256) + 2048 + 6 = 3078` — the number in every
  `quadtron_*_train.log`.
- `Tokenizer2D` (`tokenizer_v2.py`) is the *other*, 2D-capable quad-face
  tokenizer; `domain_extractor_3d` feeds it the boundary quad shell. The earlier
  Phase 2 here ("boundary-shell curve head",
  `scripts/build_quadtron_curve_dataset.py`) was built on that mistaken mapping
  and is superseded — the edge stage targets the **full hex block edge set**.

## The chain

`scripts/infer_gptcond_curve.py`:

```
GPTCond + HexaRowTokenizer  --generate-->  blocks [B,8]   (linear edges)
        --CurveModel (Polytron edge head)-->  cubic per undirected block edge
        --curved TFI (polytron_tfi.mesh_candidate)-->  filled hex mesh
```

Three fills per rollout, to isolate the curve stage on identical generated
blocks: `straight` (no curves), `model` (predicted curves), `gt` (ground-truth
`edge_ctrl`, on GT blocks — the ceiling).

## Results

Generator `data/grpo_cart_step300.pt` (d=512, 12 layers, cart), curve head
`data/polytron_clean/polytron_curve_best.pt`, val split, `target_h 0.08`,
k=3, 20 geometries:

| variant | watertight | inverted share (median) | uncovered share (median) |
|---|---|---|---|
| `straight` | 100 % | 0.488 % | **43.5 %** |
| `model` | 100 % | 0.485 % | **23.3 %** |
| `gt` | 100 % | 0.043 % | 0.0 % |

Curve error against the GT cubic (Phase 1, GT blocks, 46 val, median
chord-relative): 1.55 %.

## Reading

- The edge stage **halves the uncovered surface** on the same generated blocks
  (43.5 % → 23.3 %) and does not hurt validity (watertight stays 100 %,
  inverted share is flat). That is the value the curve stage adds to the
  generator.
- The inverted share is not moved by the curves (0.49 % both). It is set by the
  block placement; the curve stage does not introduce folds.
- The remaining 23.3 % uncovered is dominated by the **generated blocks not
  wrapping the surface**, not by the curves: GT blocks + GT curves reach 0 %.
  Curving cannot repair a block that sits in the wrong place.

## Next: corrected Phase 2

Train a Quadtron-specific curve head on the full hex block-structure set and on
the generator's own output distribution, then re-run the chain above:

1. Dataset from the full hex blocks (`polytron_data_3d_full_aug.pt` / all
   `sample.npz`), spec fitted globally, GT cubic per hex edge
   (`polytron_blocks.encode_sample`), not the 453-family-clean subset.
2. Train `CurveModel` (cloud-conditioned); optionally a variant whose input
   vertices are quantised/jittered like the generator's.
3. Fine-tuning variant on **generated** blocks: run GPTCond, and for edges whose
   endpoints match a GT edge, supervise with the GT curve; skip the rest. This
   is the only way to match the generated distribution.
4. Re-run `infer_gptcond_curve.py` and compare `model` uncovered/inverted
   against the transfer head; no-cloud ablation for what the cloud contributes.
