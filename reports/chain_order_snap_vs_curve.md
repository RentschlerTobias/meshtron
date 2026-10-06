# Chain order: snap vs learned edge curves (2026-10-05)

Question: in GPTCond -> (snap) -> curve head -> TFI, does snapping the
generated corners before the learned curves help, and which snap?

Setup: meshtron 30cee23, generator `data/grpo_cart_step300.pt`, curve head
retrained here with the report's recipe (`hexa_curve_best.pt`, best val 2.849
at epoch 30; dataset 683/78 from `scripts/build_hexa_curve_dataset.py`).
`infer_chain_order.py`, 20 val geometries, k=3 rollouts, h=0.08, projection
on; every variant on the SAME rollouts, best rollout per variant by `rank_key`.

| variant | watertight | inverted median | inverted mean | uncovered median | uncovered mean |
|---|---|---|---|---|---|
| straight | 100 % | 0.488 % | 0.727 % | 43.5 % | 43.0 % |
| model (curve head, no snap) | 100 % | 0.527 % | 0.640 % | 22.6 % | 20.5 % |
| **snap all corners -> model** | 100 % | 0.510 % | 0.647 % | **18.4 %** | **19.9 %** |
| snap boundary corners -> model | identical to "snap all" (every corner is on the boundary) | | | | |
| snap feature corners only -> model | 100 % | 0.469 % | 0.583 % | 24.7 % | 23.5 % |
| backmap (snap + seam routing, no head) | 100 % | 0.619 % | 0.801 % | 43.4 % | 43.5 % |
| GT blocks + GT curves | 100 % | 0.043 % | 0.095 % | 0.0 % | 0.2 % |

Paired against "model": snap all -> model is better on uncovered surface in
17/20 geometries (median -2.7 points), inversion +0.05 points; feature-only
snap better in 14/20 (median -0.9 points), inversion -0.04 points; backmap
worse in 20/20. Baseline rows reproduce reports/quadtron_curve_phase2.md
(straight 43.5, backmap 43.4, gt 0.0; model 22.6 here vs 25.6 there).

Reading:
- Order 1 generate -> 2 snap -> 3 learned curves -> TFI is the best tested.
  The head is trained on GT corners, which lie on the geometry; snapping first
  moves its input towards that distribution.
- "Boundary only" is a no-op on these structures: the passage is one block
  thick, every corner touches a boundary face.
- Snapping only onto geometry edges (curve endpoints / seams) moves ~26 of 64
  corners, gives less coverage but slightly fewer inverted cells: a trade-off,
  not a win.
- The residual ~18 % uncovered is block placement by the generator (GT blocks
  reach 0 %); no edge stage can repair it.
