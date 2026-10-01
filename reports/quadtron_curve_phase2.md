# Quadtron curve stage — learned edge stage vs back-mapping

Date: 2026-10-01
Branch: `polytron-main` (pushed)
Supersedes the boundary-shell reading of the earlier version of this file.

## Correction: what the 3D "Quadtron" is

The 3D block generator is **`GPTCond` + `HexaRowTokenizer`** (hex blocks, 8
corners; vocab 3078 = `PolytronTokenizer` dim=3), not the `Tokenizer2D`
quad-face path. The edge stage therefore targets the **full hex block edges**.

## Chain

`scripts/infer_gptcond_curve.py`:

```
GPTCond + HexaRowTokenizer --generate--> blocks [B,8] (linear edges)
   --CurveModel (learned edge head)--> cubic per undirected block edge
   --curved TFI--> filled hex mesh
```

Fills per generated rollout, on identical blocks:
`straight` (no curves) / `model` (learned curves) / `backmap` (non-learned:
snap corners, seam-route edges, `refill_curved`) / `gt` (ground-truth
`edge_ctrl` on GT blocks — ceiling).

## Curve head: transfer vs trained on the full hex set

78 val geometries (`hexa_curve_blocks.pt`), max chord-relative error vs GT:

| head | mean bin | exact | median | mean | p90 |
|---|---|---|---|---|---|
| transfer (`polytron_clean/polytron_curve_best`) | 16.61 | 12.8 % | 2.60 % | 5.03 % | 13.0 % |
| **trained on full hex set (`hexa_curve_best`)** | **8.52** | **15.7 %** | **1.05 %** | **2.77 %** | **6.1 %** |

Training the head on the full hex block structures and the generator's own split
(`scripts/build_hexa_curve_dataset.py`, 683/78) halves the median error.

## End-to-end: 20 val geometries, k=3, `grpo_cart_step300`, h=0.08

Projection on (default):

| variant | watertight | inverted share | uncovered share |
|---|---|---|---|
| straight | 100 % | 0.488 % | 43.5 % |
| **model** | 100 % | **0.293 %** | **25.6 %** |
| backmap | 100 % | 0.619 % | 43.4 % |
| gt (ceiling) | 100 % | 0.043 % | 0.0 % |

Projection off (control, isolates the curves):

| variant | watertight | inverted share | uncovered share |
|---|---|---|---|
| straight | 100 % | 0.071 % | 56.1 % |
| **model** | 100 % | **0.059 %** | **29.1 %** |
| gt (ceiling) | 100 % | 0.0 % | 4.2 % |

## Reading

- **The learned edge stage beats the back-mapping.** Uncovered 25.6 % vs 43.4 %
  with projection, and — the clean control — 29.1 % vs 43.4 % without
  projection. The curve head roughly halves the uncovered surface; the
  back-mapping leaves it at the straight-edge level.
- **It is the curves, not the projection.** With projection disabled the head
  still takes uncovered from 56.1 % to 29.1 %. Projection only shifts both
  down a little (and raises inverted cells).
- Back-mapping is not just unhelpful here, it is slightly worse than straight
  on inversion (0.62 % vs 0.49 %): snapping corners onto nearby seams and
  routing edges does not make a blocking span surface it does not span.
- Watertight stays 100 % everywhere; GT blocks + GT curves reach 0 % uncovered
  (projected) / 4.2 % (unprojected), so the residual ~26 % is the **generated
  block placement**, which no edge stage can repair.

## Reproduce

```bash
uv run python scripts/build_hexa_curve_dataset.py
uv run python -m meshtron.training.train_polytron --stage curve \
    --data data/hexa_curve_blocks.pt --out data/hexa_curve \
    --epochs 300 --bs 16 --lr 3e-4 --d 256 --heads 8 --layers 6 \
    --dropout 0.1 --n-points 2048 --eval-every 10 --weight-label 5

export PYTHONPATH=…/domain_partition_3D
uv run python scripts/infer_gptcond_curve.py --split val --n 20 --k 3 --backmap \
    --curve-ckpt data/hexa_curve_best.pt --out-dir data/infer_gptcond_curve_final
uv run python scripts/infer_gptcond_curve.py --split val --n 20 --k 3 --no-project \
    --curve-ckpt data/hexa_curve_best.pt --out-dir data/infer_gptcond_curve_noproj
```

## Next

- **Block placement is the limiter.** Rerank/beam the generator rollouts by
  coverage (the generator emits straight blocks; pick the one that spans the
  surface) before curving.
- Fine-tune the curve head on generated blocks (self-training): generate,
  supervise edges whose endpoint pair matches a GT edge, skip the rest.
- No-cloud ablation for the curve head, to quantify the conditioning.
