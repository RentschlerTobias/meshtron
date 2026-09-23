# Decision log — Blocking geometry mapping (GT block structure onto npz geometry)

Session goal: transfer the transformer-generated / GT hexa block structure onto the real
machine geometry so that transfinite interpolation (TFI) yields a usable CFD mesh.
Repo: meshtron, branch `blocking-geometry-mapping`, tip `f43711c`.

## Design tree (live)

```
Block structure -> real geometry (CFD-grade hexa mesh)
├── x D1  Target surface        = npz hull (labels 1..7); O-grid + BL inserted later
├── ?  D2  Face conformity      Coons interiors drift up to 0.095 on labels 5/6/7
├── ?  D3  Cell validity        25743 inverted cells at h=0.01, min scaled J = -0.70
├── ?  D4  poly 77              last true chord edge, 0.074 off surface
├── ?  D5  Acceptance target    which h, which numeric gate counts as "done"
└── ?  D6  Generalisation       machine_0034 only, or batch over all machines
```

## D1 — Target surface for the projection

**Decision:** project onto the npz triangulated surface (`surface_points`,
`surface_tris`, `surface_tri_label`, labels 1..7) as it stands. The blade-side patch
(label 7) is the O-grid / boundary-layer interface hull, not the true blade wall; the
O-grid and boundary layer are inserted into the blade block in a later, separate step.

**Facts established before the decision:**
- The npz does contain a surface definition, not only feature curves:
  12539 surface points, 19516 labelled triangles, 7 patches. The seam/feature curves are
  *derived* from label boundaries.
- `data/hex3d_algohex/batch/machine_0034/machine_0034_tet.vtk` has exactly the same
  12539 points -> the npz surface is the tet-mesh boundary, nothing was decimated and
  nothing finer is available from that source.
- Triangle edge length: p50 0.053, max 0.122 -> faceting sag on the r=0.6 hub and
  r=1.8 shroud is roughly 5e-4. That is the accuracy ceiling of this target.
- The analytic source (30 `cV_ru_*` parameters,
  `../domain_partition_3D/data/dataset/sobol/machine_0034/params.json`) cannot be
  evaluated locally: no dtoo code in the checkout, the only .msh present is T1_9.
- Label 7 is an open 2-manifold sheet (3584 tris, 224 boundary edges, phi span
  0.769 rad) -> consistent with "npz = DTOO minus O-grid minus boundary layer".

**Options considered:**
1. npz hull as-is (chosen) — no new data needed, works for every machine.
2. Obtain the true blade wall (dtoo rerun / per-machine msh export) — blocks the plan
   until the data exists.
3. Measure the offset between npz label 7 and the true blade on T1_9 first, then decide.

**Why option 1:** the task is a topology transfer. The blade wall is not part of this
mesh level at all — it enters with the O-grid later — so the hull is the correct target
for the coarse block structure, and it is the only target available for all machines.

**Consequences:** surface conformity can never be better than the ~5e-4 faceting of the
npz triangulation; any tolerance tighter than that is meaningless. Reversible: switching
to a finer or analytic target later only changes the projection target object, not the
pipeline structure.

## D2 — Face conformity: project the Coons face interiors

**Decision:** option A. In `curved_bridge.py:_block_curved_mesh`, after `orient_face`,
project the *interior* points of a domain-boundary face onto its npz patch via
`surface_nearest`, then run Gordon-Hall. Edges and corners stay untouched, so the mesh
stays watertight. Ships behind a flag, default off, because `curved_bridge.py` is shared
with the production path `scripts/map_generated_blocks.py`.

**Facts established before the decision** (218081 boundary points of the h=0.01 run,
`data/features_debug/refined_h001_sp/..._refill_vgl.vtk`, attributed to npz patches):

```
label                       p50       p90       max      #>0.01
1 inlet  (plane z=0)      2.8e-17   1.2e-16   0.0089        0
2 outlet (plane z=2.5)    8.9e-16   1.3e-15   0.0095        0
3 periodic                1.1e-04   3.5e-04   0.0133       52
4 periodic                1.7e-04   6.2e-04   0.0010        0
5 hub    (cyl r=0.6)      3.4e-04   1.2e-03   0.0731      339
6 shroud (cyl r=1.8)      3.4e-04   2.5e-02   0.0947     7035
7 blade hull              5.9e-03   2.8e-02   0.0949    12201
```

Planar patches are already exact; only curved patches drift. This is the Coons chord sag
`r*(1-cos(dphi/2))`, which for r=1.8 and a face spanning ~0.65 rad is ~0.09 — exactly the
measured maximum. The defect is the missing "faces" level in the
corners -> edges -> faces -> volume hierarchy, not a bug in the edge routing.

**Options considered:** (A) project Coons interiors, ~20 lines; (B) A plus tangential
smoothing on the patch, ~60 lines; (C) per-patch UV unwrap and TFI in parameter space,
~300 lines; (D) no fix, hand the coarse blocks to ICEM/Pointwise.

**Why A:** it closes exactly the measured defect at exactly one place, and the achievable
floor (~5e-4 npz faceting, see D1) is reached by simple nearest-point projection. B stays
available as a reaction to measured cell quality; C only if A visibly fails on the blade
hull.

**Consequences:** residual error can never go below the ~5e-4 faceting floor, so the
existing 1e-3 tripwire is the right tolerance. Nearest-point projection can compress the
face parametrisation near seams and near the blade trailing edge — watch scaled Jacobian
there and escalate to B if needed. Reversible via the flag.

## D3 — Inverted cells: gate now, untangling later

**Decision:** option A now, option B as a separate later plan. The mapping pipeline reports
cell validity (inverted count per block, min scaled Jacobian) and gates on it; it does not
repair broken blockings. An untangling smoother (boundary constrained to the geometry,
interior optimised) becomes its own work item.

**Facts established before the decision** (machine_0034_n2000, h=0.05, 49464 cells):

```
variant                             inverted   min SJ   blocks (count)
GT corners raw, straight edges            9    -0.028   4 (4), 10 (5)
snapped corners, straight edges           9    -0.030   4 (4), 10 (5)
snapped corners, curved edges           303    -0.446   4 (123), 5 (8), 10 (172)
```

Two distinct causes:
1. 9 cells fold with straight edges already — a defect of the GT corner placement.
   Snapping changes nothing (9 before, 9 after).
2. The other 294 come from curving. The affected blocks are the thin ones
   (shortest edge 0.08-0.15 vs 0.43 for the thickest block, which stays at min SJ 0.46).
   When the edge bulge (~0.07) reaches the order of the block thickness, the cell folds.
9 of 12 blocks are clean (min SJ >= 0.36). The inverted fraction is h-independent
(0.61% at h=0.05, 0.41% at h=0.01), so this is geometry, not a refinement artifact.

**Dataset-wide screening** (all 748 `sample.npz` in `data/hex3d_algohex/batch`, trilinear
scaled Jacobian of the 8-corner hull):

```
samples with >=1 inverted coarse block : 650 / 748  (86.9%)
min SJ over samples                    : p05 -0.774  p25 -0.385  p50 -0.200
samples with min SJ >= 0.2             : 2
inverted block fraction per sample     : median 8.3%, p90 16.7%, max 25%
```

Caveat: the metric evaluates the corner hull trilinearly while GT blocks are mildly
curved. Curvature is small (arc/chord p95: median 1.083, max 1.217; planarity p95 median
0.012) and correlates only weakly with min SJ (-0.27 / -0.31), so the folds are largely
genuine. machine_0034_n2000 (min SJ -0.028) is milder than the dataset median.

**Consequence:** the training corpus itself carries folded blocks, so the untangling step
(B) will be needed for the generated blockings regardless of this plan. Keeping it out of
the mapping plan preserves the simplicity mandate; the gate makes the defect visible
instead of silent.

### D3 amendment — the existing "clean dataset" does not filter folds

`scripts/clean_base_npz.py` exists and is the clean-dataset step, but its gates are
(a) `|signed volume| <= 1e-9` (degenerate blocks) and (b) `blocks > 30`. Negatively
oriented blocks are *flipped*, not dropped. Folded blocks — positive total volume, negative
scaled Jacobian at some corners — pass untouched. Measured over the 748 samples:

```
drop by clean_base_npz degeneracy gate   12
drop by >30 blocks                       56
samples with negative sub-tet volumes   710
samples with min scaled Jacobian <= 0   650
```

`scripts/filter_refill_dataset.py` does carry a sub-tet gate (`--rel-neg`), but it applies
to the refill-derived (h-refined) dataset, not to the base blockings.

**Control experiment** (machine_0034_n2000, h=0.05, refill driven by the GT's *own* 84
edge polylines from `edge_polyline`, all 84 matched): 9 inverted cells, min SJ -0.186 —
worse than with straight edges (-0.030). The folds are therefore genuine, not an artifact
of evaluating curved GT blocks as straight-sided hexes.

**Consequence for D3 option A:** discarding folded blockings would remove ~87% of the
corpus, so "gate" must mean *measure and report per sample*, not *drop from the dataset*.
Repairing the corpus is part of B.

## D4 — Edge routing: geodesic on an associated patch

**Decision:** option B. Associate each block edge with its patch first, then route the edge
as a shortest path on that patch's triangulation (Dijkstra on the triangle-edge graph plus
smoothing). This replaces the ~180-line guard machinery of `_surface_path_fn`
(`walkable_runs`, `seam_between`, `snap_window`, `_walk`, `seam_neighbors`) with one rule.
Seam edges keep their exact seam-curve routing.

**Facts established before the decision** (poly 77, the last true chord edge):

```
endpoints        r ~ 0.597, phi -0.258 and +0.247, z 1.300 and 1.429  (hub/blade seam)
chord length     0.3254
max pull chord->surface      0.0746   (limit max_pull 0.15 -> passes)
label sequence of 33 samples 5 7 5 5 5 7 7 7 7 7 7 5 7 7 7 7 7 7 7 7 7 7 5 5 5 5 5 5 5 5 5 5 5
max jump between consecutive projections 0.1461  vs allowed 0.0153  -> DECLINED
projected arc / chord        1.447
```

The edge runs along the blade root, where hub (5) and blade hull (7) meet at a sharp angle.
Nearest-point projection is ill-conditioned there — the closest point flips between the two
patches, which produces the shredded label sequence and the 10x jumps. The guards fire
correctly; the edge stays a chord that cuts 0.063 into the hub. Block 10, whose edge this
is, is the worst folded block (172 of 1632 cells at h=0.05).

**Why B:** a path constrained to one associated patch cannot flip patches, so all six
rejection reasons disappear by construction rather than by tuning. It is also the answer to
the simplicity mandate (handoff section 5.3) and mirrors what ICEM/Pointwise do: associate
the entity, then project onto it. Risk: a Dijkstra path along triangle edges is jagged and
needs smoothing, otherwise cell quality suffers.

## D5 — Acceptance criterion

**Decision:** option A — gate on conformity only, report validity. Acceptance runs at
h=0.05.

Gate (exit non-zero on violation):
- `max_boundary_surface_dist <= 1e-3`
- chord edges (`edges_degenerate_chord`) == 0

Reported, never gated:
- inverted cells total and per block, `min_scaled_jacobian`

**Why:** D3 separated conformity from validity into two responsibilities, so a GT dataset
defect must not be able to fail a successful mapping. The 1e-3 tolerance sits a factor 2
above the npz faceting floor (~5e-4, see D1) and a factor 95 below today's error (0.0949).

**Why h=0.05:** all three effects are already fully visible there (0.0905 boundary error,
303 inverted cells, the same 5 chord edges as at h=0.01), while the run takes 4 s and
produces 5.6 MB instead of 2-3 min and 814 MB — the artifact can actually be inspected in
ParaView. A single h=0.01 run confirms at the end.

Measured runtimes on this machine: coarse (target_h=10) 1 s, h=0.05 4 s.

## D6 — Scope of validation

**Decision:** option B without the outlier classes. Blockings with more than 30 blocks are
excluded entirely (same rule as `scripts/clean_base_npz.py`, MAX_BLOCKS=30), so the corpus
is 692 of 748 samples. Validation is staged:

1. visual sign-off on `machine_0034_n2000` after the edge-routing step (D4),
2. visual sign-off on `machine_0034_n2000` after the face-projection step (D2),
3. numeric table over a stratified 25-sample set (below) — no VTK inspection,
4. full batch over the 692 samples with <= 30 blocks, summary JSON only; VTK written
   only where a gate fails.

Block-count distribution of the 692: 12 (138), 16 (112), 21 (96), 22 (314), plus
11 (2), 15 (11), 19 (14), 23 (1), 25 (4). machine_0034_n2000 has 12 blocks — the simplest
class — so a sign-off there alone would not generalise. 86.3% of the 692 carry at least one
folded block, median min SJ -0.197.

Stratified set (5 per major class by min scaled Jacobian quantile, 1 per minor class,
plus machine_0034_n2000 as the reference):

```
machine_0050_n8000 12  machine_0093_n8000 16  machine_0252_n8000 21  machine_0111_n8000 22
machine_0406_n8000 12  machine_0057_n2000 16  machine_0067_n8000 21  machine_0196_n8000 22
machine_0117_n8000 12  machine_0052_n8000 16  machine_0019_n2000 21  machine_0071_n8000 22
machine_0237_n8000 12  machine_0011_n8000 16  machine_0387_n8000 21  machine_0023_n2000 22
machine_0201_n2000 12  machine_0140_n8000 16  machine_0279_n8000 21  machine_0155_n8000 22
machine_0119_n2000 11  machine_0106_n2000 15  machine_0177_n2000 19  machine_0112_n8000 23
machine_0291_n8000 25  machine_0034_n2000 12 (reference)
```

Runtime budget: 4 s per sample at h=0.05 -> stratified set ~2 min, full batch ~45 min
serial. Keeping only summary JSON for the batch avoids ~4 GB of VTK.
