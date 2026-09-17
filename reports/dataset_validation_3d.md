# 3D Hex-Block Mesh Dataset — Validation & Integrity Report

**Scope:** `data/hex3d_algohex/batch/` (the training corpus for the meshtron transformer).
**Script:** `scripts/validate_3d_dataset.py` (numpy-only, rerunnable; full scan ≈1.6 s).
**Machine-readable output:** `data/hex3d_algohex/batch/validation_summary.json`.

---

## 1. Readiness verdict (TL;DR)

**Usable distinct geometries: 228** — 227 Sobol-sampled machines + 1 reference geometry (T1_9).
**Usable samples: 433 of 468** `sample.npz` files (35 excluded, see §8).

The data is **largely clean and structurally sound**, but it is **not yet training-ready without two decisions**:

1. **The direction-class count is *not* 11.** `dir_class` has **9–14 classes depending on the sample** (modal count 12, only ~26 % have 11). Any pipeline hard-coded to 11 classes will break or silently mis-group faces.
2. **The train/val split unit must be the machine (geometry), not the sample.** `machine_XXXX_n2000` and `machine_XXXX_n8000` are the *same underlying geometry* (byte-identical `params`), so splitting them across train/val leaks the held-out geometry.

There are **228 geometry identities, not ~500** — the "~500" figure counts *runs*, of which every geometry is represented twice (two resolutions) for 199 machines. For a transformer this is a **modest** corpus; treat it as a starting set, and split at geometry granularity.

---

## 2. Inventory & reconciliation — the 757 / 496 / 249 discrepancy

| Quantity | Value | Meaning |
|---|---|---|
| Top-level entries in `batch/` | **757** | 249 bare machine dirs + 496 machine-resolution dirs + 10 T1_9 dirs + 2 files (`manifest.txt`, `sample_history.tsv`) |
| `manifest.txt` lines | **496** | one line per AlgoHex *run* = 248 machines × 2 resolutions (n2000 + n8000) |
| Bare `machine_XXXX` dirs | **249** | the distinct geometries (`machine_0001`…`machine_0256`, minus 7 ids: **74, 85, 115, 129, 158, 167, 210**) |
| `sample_history.tsv` lines / unique runs | **383 / 373** | a *partial* job-status log |
| `sample.npz` on disk | **468** | 459 machine-resolution + 9 T1_9 |

**Why the three numbers differ:**

- **249 bare dirs** = number of distinct geometries with a prepared input tet mesh. `machine_0006` is one of them but **failed tet-prep** (its `tet_prep.log` ends in `IndexError: too many indices` — "0 boundary triangles"), so it never reached AlgoHex.
- **496 manifest lines** = 248 machines (249 minus `machine_0006`) × 2 resolutions. Each line names the `n2000`/`n8000` output directory.
- **468 sample.npz** = 496 manifest runs − 37 runs that produced no output. Those 37 are exactly the runs whose final history status is **32 `prune`** (killed at the 2400 s AlgoHex wall-clock, exit code 143) + **5 `fail`** (exit code 1). No `prune`/`fail` run left an npz; every `ok` run did.
- **373 unique history runs ≠ 496 manifest runs**: `sample_history.tsv` is *incomplete* — 123 manifest runs completed successfully (they have an npz) but were never logged. The history is a job-monitoring artifact, not an authoritative inventory; `manifest.txt` is authoritative for "what was queued".

**Resolution coverage per machine:** 225 machines have both `n2000` and `n8000`; 9 have exactly one; 14 have neither (both runs pruned/failed); `machine_0006` has neither (tet-prep failure). `T1_9` is a separate reference geometry meshed at 10 resolutions (n500…n8000), 9 of which have an npz (`T1_9_n6000` has an empty `clean_blocks.log` and no npz — it died at the clean_blocks stage).

---

## 3. Schema — actual vs. claimed

The task's stated schema is **incomplete by two keys**. The real `sample.npz` (verified against the `smoke2000` reference and all 468 batch files) contains:

| Key | Shape | dtype | Notes |
|---|---|---|---|
| `vertices` | [N, 3] | float64 | |
| `blocks` | [F, 8] | int64 | corner indices, VTK_HEXAHEDRON ordering |
| `quad_faces` | [F′, 4] | int64 | boundary shell faces |
| `edges` | [E, 2] | int64 | **directed** half-edges (each undirected edge × 2) |
| `edge_ctrl` | [E, 2, 3] | float64 | two control points per edge |
| `edge_polyline` | [P, 3] | float64 | CSR-encoded edge geometry |
| `edge_polyline_offset` | [E+1] | int64 | CSR offsets (monotonic, last == P) |
| `dir_class` | [E] | int64 | per-edge direction class |
| `params` | scalar | str | **JSON string**, the Sobol geometry parameters |
| `quality` | scalar | str | JSON string (fit residuals, degenerate/inflection edges, …) |
| `provenance` | scalar | str | JSON string (source vtk, n, git sha, exporter) |
| `dir_class_count` | [K] | int64 | **per-class *division* counts** (= `blocks.divisions.json` `counts`), K = max(`dir_class`)+1 |
| `surface_points` | [Np, 3] | float64 | full triangulated surface |
| `surface_tris` | [Nt, 3] | int64 | |
| `surface_tri_label` | [Nt] | int64 | |

Two corrections worth making explicit:

- **`dir_class_count` is not a histogram of `dir_class`.** It equals the `counts` field of the sibling `blocks.divisions.json` (number of subdivisions along each direction class) and sums to ~150, whereas `dir_class` has ~200+ entries. The edge histogram must be computed from `dir_class` itself.
- **`dir_class` is not fixed at 11 classes.** Observed class counts across samples: 9 classes (1 sample), 10 (88), 11 (113), 12 (203), 13 (10), 14 (18). The modal value is **12**, not 11. `dir_class_count` length always equals `max(dir_class)+1`, so the array is internally consistent; only the "11 classes" assumption is wrong.

---

## 4. Validation results — what passed, what failed

**468 files loaded, 0 load failures, 0 crashes.** Every check below is enforced per-sample and recorded in the JSON.

**Passing for all 468 samples** (no exceptions found):
- Key presence vs. the 15-key schema above; array ndim/shape consistency.
- Finiteness (no NaN/Inf in any float array).
- `edge_polyline_offset` monotonic, first == 0, last == len(`edge_polyline`).
- `dir_class` non-negative; `dir_class_count` length == max(`dir_class`)+1.
- `len(edge_ctrl) == len(dir_class) == len(edges)`.
- Index ranges of `blocks`/`quad_faces`/`edges`/`surface_tris` valid.
- **`edges` array is exactly 2× the undirected block-edge set** (every undirected edge appears once per direction).
- **`quad_faces` equals the boundary face set of the block complex** — every block face is shared by exactly 1 (boundary) or 2 (interior) blocks, and the set of valence-1 faces matches `quad_faces` exactly. Zero unmatched interior faces, zero face shared by >2 blocks (no non-manifold faces).
- **Zero duplicate vertices** in any sample.
- `blocks_without_lattice == 0` and `class_disagreements == 0` everywhere.

**Failing — 35 samples** (the exclusion list, §8), for exactly two root causes:
1. **Degenerate block (inverted or collapsed): 35 samples** — 28 with one *inverted* block (negative 6-tet volume) and 7 with one *collapsed* block (|volume| ≤ 1e-9, a numerical sliver).
2. **Non-manifold boundary edge: 5 samples** — one boundary edge shared by **4** boundary faces instead of 2 (a pinch at a collapsed block). These 5 are a subset of the collapsed-block cases.

---

## 5. Geometry integrity details

**Block volume** (6-tetrahedron decomposition around body diagonal 0–6; signed sum is positive for any valid, even concave, hex):

- Per-sample median block volume: 0.016–0.53 (median 0.14).
- Smallest *healthy* block volume ≈ 2×10⁻⁴ (thin trailing-edge blocks); the degenerate blocks drop to ~10⁻¹⁸…10⁻² (see §8). The 1e-9 collapsed threshold sits ~5 orders of magnitude below any legitimately thin block.

**Edge valence** (blocks incident per undirected block edge), aggregated over the 433 OK samples:

| Valence | Count | Interpretation |
|---|---|---|
| 1 | 18,507 | wedge/folded edge — a single block whose two boundary faces meet at a crease (blade trailing edge) |
| 2 | 33,036 | regular boundary edge |
| 3 | 125 | boundary singular edge |
| 4 | 4,594 | regular interior edge |
| 5 | 2,086 | interior singular edge (valence-5 singularity) |

Valence-1 edges are present in **every** sample (up to 61 per sample). This is characteristic of AlgoHex polycube output on thin trailing edges — not a defect — but it means these meshes are *not* structured hex meshes; the transformer sees irregular valence.

**Exporter's own quality flags** (from `quality` JSON): `fit_residual_p95` median 0.012 (max 0.027); 48 OK samples have `degenerate_edges > 0` (edge-level, not block-level, so not excluded). One sample (`machine_0165_n8000`, status OK) carries `n_invalid_param_tets = 309` and `valid_volume = 0.99985` — its blocks are geometrically fine but its parametrization was borderline; flagged as low-confidence rather than excluded.

---

## 6. Duplicate detection & train/val leakage

- **Vertex-coordinate hash (order-invariant): 0 duplicate groups.** No two samples share an identical vertex multiset.
- **`params`-payload hash: 235 distinct values, 226 groups with >1 member.** The 226 groups are exactly the 225 two-resolution machines plus `T1_9` (9 samples sharing the empty `{}` params). **All 225 machine pairs have byte-identical `params`** — `machine_XXXX_n2000` and `_n8000` are confirmed to be the same underlying geometry.
- `smoke2000/sample.npz` (outside `batch/`) is the **same geometry as `batch/T1_9_n2000`** (identical `params = {}`, same n=2000) but produced by a different run (different `git_sha`, slightly different vertex positions). They are near-duplicates, not exact duplicates — keep them together in any split.

**Leakage implication:** the split unit must be the `params` hash (equivalently, the machine id). A random sample-level split would place `n2000` and `n8000` of the same geometry on opposite sides of the fold.

---

## 7. Dataset-level summary

| Metric | Value |
|---|---|
| Usable samples | 433 (426 machine + 7 T1_9) |
| Distinct geometry identities | **228** (227 machines + T1_9) |
| Machines: 2 usable resolutions / 1 / 0 | 199 / 28 / 21 (14 no-npz + 7 fully-excluded) |
| Total blocks / vertices / directed edges | 9,480 / 26,437 / 116,696 |
| Blocks per sample | 11–75 (median 22) |
| Vertices per sample | 38–144 (median 64) |
| Edges per sample | 162–704 (median 280) |
| Surface points / triangles per sample | 11,239–13,207 / 17,334–20,702 |

**Direction-class histogram** (edge `dir_class`, 14 classes over the OK set):
`{0:14378, 1:14232, 2:13032, 3:9948, 4:8618, 5:8262, 6:9684, 7:8670, 8:8544, 9:8284, 10:6666, 11:4460, 12:1110, 13:808}` — classes 0–2 dominate (the three principal structured directions), with a long tail into classes 11–13 (only present in higher-resolution samples).

**Division histogram** (per-class subdivisions, = `dir_class_count`/`divisions.json`): similar shape — `{0:7654 … 10:3656, 11:2628, 12:264, 13:176}`.

---

## 8. Exclusion list

Three categories of unusable input, all enumerated in `validation_summary.json`:

**(a) 35 samples with a degenerate block / non-manifold boundary — exclude from training:**

*Inverted block (28):* `T1_9_n1000`, `machine_0005_n2000`, `machine_0005_n8000`, `machine_0019_n2000`, `machine_0025_n2000`, `machine_0033_n2000`, `machine_0035_n8000`, `machine_0053_n2000`, `machine_0053_n8000`, `machine_0060_n2000`, `machine_0067_n8000`, `machine_0079_n2000`, `machine_0079_n8000`, `machine_0087_n2000`, `machine_0119_n2000`, `machine_0120_n2000`, `machine_0128_n2000`, `machine_0134_n2000`, `machine_0134_n8000`, `machine_0156_n2000`, `machine_0171_n8000`, `machine_0183_n2000`, `machine_0224_n2000`, `machine_0229_n2000`, `machine_0229_n8000`, `machine_0234_n2000`, `machine_0239_n8000`, `machine_0242_n2000`.

*Collapsed block (7):* `T1_9_n4000`, `machine_0016_n8000`, `machine_0067_n2000`, `machine_0112_n2000`, `machine_0113_n8000`, `machine_0119_n8000`, `machine_0177_n2000`.

*Non-manifold boundary edge (5, all collapsed-block samples):* `machine_0067_n2000`, `machine_0112_n2000`, `machine_0113_n8000`, `machine_0119_n8000`, `machine_0177_n2000`.

Note: 7 machines (`machine_0005`, `machine_0053`, `machine_0067`, `machine_0079`, `machine_0119`, `machine_0134`, `machine_0229`) lose *both* resolutions to this defect — their geometry is intrinsically hard at the trailing edge.

**(b) 37 runs with no `sample.npz`** (pruned or failed; see §2): `machine_0039_n2000`, `machine_0063_n2000`, `machine_0069_n2000`, `machine_0069_n8000`, `machine_0096_n2000`, `machine_0096_n8000`, `machine_0108_n2000`, `machine_0108_n8000`, `machine_0123_n2000`, `machine_0123_n8000`, `machine_0132_n8000`, `machine_0143_n2000`, `machine_0143_n8000`, `machine_0146_n2000`, `machine_0146_n8000`, `machine_0162_n2000`, `machine_0164_n8000`, `machine_0165_n2000`, `machine_0168_n2000`, `machine_0168_n8000`, `machine_0191_n2000`, `machine_0191_n8000`, `machine_0194_n2000`, `machine_0194_n8000`, `machine_0200_n8000`, `machine_0203_n2000`, `machine_0203_n8000`, `machine_0204_n2000`, `machine_0204_n8000`, `machine_0215_n2000`, `machine_0226_n2000`, `machine_0226_n8000`, `machine_0228_n2000`, `machine_0228_n8000`, `machine_0245_n8000`, `machine_0250_n2000`, `machine_0250_n8000`, `T1_9_n6000`.

**(c) 1 machine never meshed:** `machine_0006` (tet-prep crashed, no tet mesh → not in manifest).

---

## 9. Recommended next steps

1. **Fix the direction-class handling.** Confirm `domain_extractor_3d.py` / `tokenizer_v2._order_quads_by_dir_class` tolerate 9–14 classes (they read `dir_class` values directly, so they likely do) — but any hard-coded "11" (e.g., a `num_classes` embedding dimension) must become data-driven (`max(dir_class)+1`).
2. **Split at geometry granularity.** Train/val split over the 228 `params` hashes (machines), never over samples. Keep `n2000` + `n8000` of a machine, and `T1_9` + `smoke2000`, on the same side of the fold.
3. **Exclude the 35 samples in §8(a)** (or, if the pipeline can drop single blocks, drop just the inverted/collapsed block — but do not train on a mesh containing a negative-volume element).
4. **Decide whether 228 geometries is enough.** It is a reasonable *starting* corpus but small for a transformer; the two resolutions are same-geometry (limited diversity), so consider whether more Sobol machines (or more than 2 resolutions) are worth the mesh time.
5. **Re-run the validator after any regeneration** (`python3 scripts/validate_3d_dataset.py`) — it is deterministic and takes ~2 s; the JSON is the single source of truth for the numbers above.

---

### Method note / assumptions

- "Degenerate block" = 6-tet signed volume ≤ 0 **or** |volume| ≤ 1e-9 (collapsed). The corner-Jacobian was deliberately *not* used as a hard criterion: these are concave AlgoHex blocks, where per-corner Jacobian signs alternate even for valid elements; the 6-tet volume is the reliable signal. The 1e-9 threshold sits ~5 orders below the smallest healthy block (≈2×10⁻⁴).
- Block/face/edge topology assumes VTK_HEXAHEDRON corner ordering, which was validated empirically (face and edge incidence both matched `quad_faces`/`edges` exactly on every sample).
- `params` hashes are canonicalised (JSON `sort_keys`) before hashing, so the geometry identity is robust to formatting. All numbers above are drawn from the script's own JSON output, not from any external source.
