# Plan: blade O-grid insertion at block level, BEFORE the TFI (2026-10-12)

## Goal
The flow model predicts the 12 core blocks (corners + cubic Bezier edges). The blade and its
O-grid layer were cut out before AlgoHex; the only trace in the data is the O-grid cut surface
(label 7, `ogrid_interface`). Today the chain fills only the core, so every mesh has a hole around
the blade. Target: insert the blade O-grid as additional BLOCKS (corners + edge curves) into the
predicted block structure and run ONE joint curved TFI over core + O-grid. No fine-mesh welding.

## What exists (read-only reuse, nothing in /opt/stack is modified)
- `domain_partition_3D/experimentell/hex3d_algohex/ogrid_extrude.py`: `OgridMap` (wall-normal
  structure of the dtOO O-grid: layers, parents, `column(Q)` = O-grid grid line from an interface
  point to the blade wall) and `extrude_ogrid`, which extrudes every FINE interface quad of the
  AlgoHex core along those lines (one block per core block face, conforming by construction).
  The block-level version below is the same idea applied to the 4 coarse block faces.
- `reattach.ogrid_blocks(elements, nodes)` + `dp3d.extraction.parse_msh`: dtOO O-grid hexes from
  `case/mesh.msh`.
- `meshtron.geometry.polytron_tfi.refill_polytron(V, blocks, curves, target_h, out_vtk, surface=...)`:
  takes arbitrary dense polylines per edge `{(a, b): [k, 3]}`, projects boundary corners/edges/faces
  onto `surface` patches, Gordon-Hall fill. New O-grid edges can be passed as exact O-grid lines.
- Data: `/opt/stack/papaya_datagen/runs/<name>/case/mesh.msh` (~21 MB, dtOO O-grid incl. blade wall)
  for the new tistos samples. The old `machine_*` samples have no msh (open point 3).
- Topology facts (data/flow_canon_papaya.pt, majority labels): exactly 4 core block faces lie on
  label 7 (blocks 4, 5, 6, 7), 8 interface corners, forming a ring around the blade
  -> the O-grid adds 4 blocks, 16 blocks in total.

## Method
1. **Interface faces from the topology, not from the prediction.** The 4 faces (blocks 4-7) are
   fixed per topology (computed once from the majority vertex/edge labels; stage A later supplies
   them per topology). Per-sample labels are noisy (318/416 samples show all 4), so they are not used.
2. **Columns per interface corner** (8): `OgridMap.column(V[c])` -> polyline from the interface to
   the blade wall. Wall corner = last point; the column itself is the radial edge curve of the new
   block. As in `extrude_ogrid`, the first point is set to the (snapped) core corner; the offset to
   the O-grid outer layer is reported (residual statistics).
3. **Wall edges**: each interface edge of the 4 faces (Bezier polyline, 64 points) is mapped point
   by point through `column(...)[-1]` onto the blade wall -> wall polyline. Interface edges
   shared by two faces are mapped once -> shared wall edge -> conforming.
4. **New blocks**: per interface face one hex = interface quad (4 core corners) + wall quad
   (4 wall corners), VTK corner order with positive volume (flip as in `extrude_ogrid`). 8 new
   corners, 4 new blocks, new edges: 8 radial (columns) + 8 wall edges (+ the existing 8 interface
   edges, shared with the core).
5. **Projection surface**: blade wall triangles from the msh (inner O-grid tube, the `OgridMap`
   wall) appended to the sample surface as label 8, so `refill_polytron` projects the wall faces
   of the new blocks onto the blade. Hub/shroud sides of the O-grid blocks lie on the boundary-layer
   cut extension, which is not in the sample surface -> left unprojected for now (they get the
   hub/shroud BL blocks later, same procedure).
6. **Joint TFI**: V' = core corners + 8 wall corners, B' = 12 + 4 blocks, curves = core Bezier
   polylines + radial + wall polylines -> one `refill_polytron` call -> one conforming hex mesh.
   Wall-normal resolution: first version uniform `target_h`; see open point 1.

## Implementation (prototype in /work/meshtron_v2)
- `blade_insert.py`: `load_ogrid(msh) -> OgridMap + wall triangles`, `interface_faces(topo)`,
  `insert_blade(V, B, curves, omap, faces) -> V', B', curves', part labels`; CLI for one sample.
  Imports `ogrid_extrude`, `reattach`, `dp3d` read-only (datagen re-reads domain_partition_3D from
  NFS, it must not change).
- `infer_chain_flow.py`: variants `flow_blade` and `gt_blade` (insertion + joint TFI), metrics
  split into core / O-grid cells.
- `export_vtk_val.py --blade`: `pred_mesh_blade.vtk`, `pred_blocks_blade.vtk` (cell data `part`:
  0 core, 1 O-grid), blade wall surface.
- Data: copy `case/mesh.msh` of the val_new (28) and then train items from papaya NFS to
  `/work/datagen_papaya/runs/<name>/mesh.msh` (read-only on NFS, ~21 MB each).

## Verification
1. GT core blocks + insertion on 3 val_new items first: watertight, 16 blocks, conforming
   interface (shared nodes by construction; check zero duplicate points after merge), no gap
   between O-grid wall face and blade wall (distance of wall-face nodes to the blade surface).
2. Quality: inverted cells and scaled Jacobian (min, p1) separately for core and O-grid cells;
   reference = fine-level `ogrid_extrude` on the GT AlgoHex mesh (its log numbers) and the dtOO
   O-grid itself (min sj 0.52, none inverted).
3. Then predicted blocks (flow model, seed 0) on all 28 val_new items, same metrics, plus blade-wall
   coverage (uncovered share of the blade wall) next to the existing core coverage.
4. VTK export of a few items for visual inspection (trailing edge, hub/shroud corners).

## Open points (decide before / during implementation)
1. **Wall-normal clustering**: uniform TFI resolution vs the dtOO O-grid's wall clustering
   (`extrude_ogrid` samples the O-grid layer positions). `refill_polytron` takes one `target_h`;
   per-block divisions / grading need a check of the TFI code (`blocks.divisions.json` exists in
   datagen) and may need a small extension.
2. **Trailing edge**: the O-grid parametrisation handles the sharp TE on the fine level; with only
   4 blocks the TE sits inside one block face. Check the scaled Jacobian there first; if bad, split
   the ring at the TE (needs a topology with more interface faces).
3. **Old machine_* geometries** have no O-grid msh: regenerate it with dtOO (`gen_mesh.py`, ~110 s
   each) if their parameters are available, or evaluate on tistos only.
4. Hub/shroud boundary layer (labels 5/6) with the same block-level procedure afterwards.
