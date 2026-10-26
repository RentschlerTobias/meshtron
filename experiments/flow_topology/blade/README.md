# Blade insertion into the block structure (2026-10-20 .. 2026-10-26)

Goal: close the AlgoHex core block structure (tistos) to the full CFD domain, with the dtOO volume discarded and
only its geometry (boundary + region interfaces from `case/mesh.msh`) used.

## Current pipeline (boundary layer kept in the AlgoHex domain)
1. domain_partition_3D branch `feat/keep-bl`, `datagen/run_sample_keepbl.sh <run_dir>`:
   `tet_prep_v5 --keep-prisms --remesh-all 0` (only the O-grid is cut; hub/shroud are walls; all surfaces + seams
   remeshed), AlgoHex n=2000, `beam_collapse --width 32 --depth 20 --hausdorff-tol 0.1` (with the default 0.05 the
   collapse stalls at 30 blocks; 0.1 gives 12 blocks isomorphic to the old topology on tistos_s1_0131),
   tfi, export, `curve_postprocess` (feature arcs; corners re-ordered along their seam where nearest-point
   positions contradict the block order -- fixes overlapping arcs on the closed O-grid outline).
2. `oring.py`: core TFI (12 blocks, projected onto the physical patches + the O-grid cut), then the O-ring as ONE
   structured tube: outer side = the core nodes on the cut (conforming by construction), partner on the blade via
   two tube parametrisations (`tubeparam.py`: u = span, s = around), graded layers. `--opt N` optimises the core
   corners + curves on the TFI surrogate first (`opt_blocking.py`).
   Result tistos_s1_0131: core 426 inverted (73 with --opt), ring 356 (866 with --opt; 36 folded blade targets) --
   open: untangle folded targets in (u, s); ring cells near the tip at LE / TE.

## Scripts
- `msh_walls.py` -- boundary / region-interface geometry of the dtOO msh (`load_regions`), smooth / seam projectors.
- `tubeparam.py` -- harmonic (u) + arc-length (s) parametrisation of a tube-like patch (blade, O-grid cut).
- `oring.py` -- current approach (above).
- `opt_blocking.py` -- corners + edge points optimised with curve_opt.TFISurrogate incl. projected faces.
- `blocking.py` -- block-level variant (ring / layer blocks extruded from the core faces; `--extrapolate`,
  `--min-gap`, `--opt`); superseded for the ring by `oring.py`.
- `surfmap.py` (harmonic cut->wall map), `relocate.py` (`widen_gaps`), `close_layers.py`, `blade_insert.py`,
  `tfi_param.py` (index-parametrised edges, fixed layer counts) -- earlier variants / building blocks.
- `plot_wall.py`, `plot_blade.py`, `plot_hub.py`, `make_vtm.py`, `block_edges_vtk.py`, `diag.py` -- diagnostics.

## Paths
The scripts still use the absolute paths of the former working directory on mars (`/work/meshtron_v2`,
`/work/datagen_full/datagen/runs`, `/work/curves_post`, `/work/dp3d_feature_curves`); data copies on the NFS under
`/opt/stack/blade_blocking_20261021/` (tistos_s1_0131 inputs + results).
