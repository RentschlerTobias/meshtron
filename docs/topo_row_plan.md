# Topology-canonical HexaRow row plan

Replaces the geometric
row plan of `meshtron/data/hexa_row_tokenizer.py::build_row_plan` with one
that depends only on the labelled block complex; token grammar, emission
contract and `detokenize` are unchanged (`tokenize(..., emit_override=...)`).

- `meshtron/data/topo_row_plan.py` — frames (24 proper hex rotations) transported across
  faces, canonical anchor = min (row count, BFS code) over all
  (block, rotation) candidates, rows = dual chords along local x.
  `load_npz_sample()` reads a sample.npz with boundary-face labels.
- `scripts/build_topo_tokens.py` — sample.npz -> token file (build_hexarow_tokens
  format + per-item `topo` audit info), with a detokenize round trip per sample.
- `scripts/label_diff.py` — debug aid: boundary-face labels that differ between two samples.

## Why

`row_plan_diag2.py` (domain_partition_3D, experimentell/hex3d_algohex/analysis/): on the machine_0004 perturbation family
(identical 12-block topology after beam_collapse) the old plan gives 5-8 rows
(211-250 cart tokens). All variation comes from the walk's geometric
thresholds (z-layer gap, z-jump, azimuth of the next block); the
emission-level break never fired.

## Result (2026-10-04, 17 samples: 13 beam relabels + 4 fresh Sobol machines)

    python scripts/build_topo_tokens.py --merge-cut-labels --out tok.pt \
        <beam relabels>/*/sample_12.npz <fresh Sobol runs>/*/sample.npz

| variant | 12-block samples | identical code | ties (automorphisms) | round trip |
|---|---|---|---|---|
| old `build_row_plan` | 16 | — (5-8 rows) | — | ok |
| topo, raw labels 1-7 | 16 | 11/16 | 1 | 17/17 |
| **topo, `--merge-cut-labels`** | 16 | **16/16** | **1** | **17/17** |
| topo, no labels | 16 | 16/16 | 8 (geometric tie-break) | 17/17 |

All 12-block samples: rows [3, 2, 3, 3, 1], 211 cart tokens; the first
emitted block is the same physical block in every sample.

Labels: block faces straddle the boundary between the O-grid cut (7) and the
BL cuts at hub/shroud (5/6), so the majority label flips between samples.
Folding 5/6/7 into one "cut interface" label removes the flips; inlet,
outlet and the two periodic sides alone leave no automorphism.

## Open

- Used by the block generator v2 (scripts/blockgen_v2/build_dataset.py,
  reports/blockgen_v2_*.md). Not yet wired into `build_row_plan(..., order="topo")`
  or the family token builder (they drop surface labels).
- machine_0003 (16 blocks) is canonical too but a different topology.
