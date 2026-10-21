# Blade insertion into the block structure (2026-10-20 .. 2026-10-21)

Goal: close the AlgoHex core block structure (12 blocks, tistos) to the full CFD domain BEFORE the TFI, with the
dtOO volume discarded and only its geometry (boundary + region interfaces from `case/mesh.msh`) used.

## Scripts (prototypes, paths point to /work on mars)
- `msh_walls.py` -- boundary / region-interface geometry of the dtOO msh (`load_regions`: core, O-grid, hub/shroud
  layer regions; walls per region; core|region cuts; layer|O-grid interface), smooth and seam projectors.
- `blocking.py` -- full blocking at block level (topology as dtOO: layers under the core only, O-ring from hub to
  shroud), graded radial edges, one joint TFI (`tfi_param.py`: index-parametrised edges, fixed layer counts).
  Options: `--min-gap` (relocate.widen_gaps), `--opt N --opt-n 17` (opt_blocking: surrogate optimisation).
- `surfmap.py` -- harmonic map of a layer cut onto its wall (keeps relative positions of block edges).
- `opt_blocking.py` -- corners + edge points optimised with curve_opt.TFISurrogate incl. projected faces.
- `plot_hub.py` -- unrolled plot of block edges on hub / shroud with crossing detection.
- `close_layers.py`, `blade_insert.py`, `block_edges_vtk.py`, `relocate.py`, `diag.py` -- earlier variants / tools.

## Status (tistos_s1_0131, GT blocks with post-processed curves)
Best: `blocking.py --min-gap 0.5 --opt 1000 --opt-n 17` -> 48 blocks, 26 inverted of 179k cells (all O-ring
blocks), core and both layers clean. The user still sees two crossing block edges after the projection onto the
hub (the core block ring around the O-ring vs. the O-ring footprint) -> approach paused.

## Next: keep the boundary layer in the AlgoHex domain
domain_partition_3D branch `feat/keep-bl` (tet_prep_v5 `--keep-prisms --remesh-all 0`): only the O-grid is cut,
hub and shroud are real walls, one O-grid cut from hub to shroud; all surfaces + seams remeshed (tri quality p1
0.79, no tets < 0.05 on 0131). AlgoHex on this input not run yet (needs OK, ~16 min .. 2.5 h per geometry).
Then: beam collapse / TFI / export -> insert the O-ring only (one cut) -> OpenFOAM comparison.
