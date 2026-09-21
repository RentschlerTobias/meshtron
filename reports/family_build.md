# Familien-Build (P1-1): Auswahl, Tokens, TFI-h=0.5

## Auswahl (D1, `scripts/select_family.py`)

- Quelle: `data/dedup_inventory.json`, `keep=true` = 832 Runs (ein Run pro `(geom_id,grid_id)`).
- Cap: `n_blocks <= 25`. Begruendung: kept-Histogramm dicht 11..25, naechster Wert
  36 (Gap 25->36); p95 ueber alle kept = 50.7.
- Zusatzfilter Gueltigkeit: keine Hexa-Face mit <4 eindeutigen Vert-Ids
  (weld-fusionierter Sliver; `HexaRowTokenizer` wirft sonst `DegenerateBlockError`).
  Betrifft 11 Runs.
- Ergebnis: **selected 761**, **dropped 71** (Cap 60, degeneriert 11).
  Geometrien **378**, davon **19** komplett raus (`non_ideal_all_modes`).
- Cap-verworfen (60): n_blocks 36..87. Degeneriert (11):
  `batch/machine_0016_n8000`, `0067_n2000`, `0112_n2000`, `0113_n8000`,
  `0119_n8000`, `0177_n2000`, `0273_n2000`, `0273_n8000`, `0318_n8000`,
  `0358_n8000`, `0361_n2000` (nur Geometrie von 0273 verliert dadurch alle Runs).
- Histogramm kept (`n_blocks`): 11:2, 12:152, 15:12, 16:125, 19:13, 21:109,
  22:354, 23:1, 25:4, 36:3, 38:1, 42:8, 45:1, 48:5, 54:2, 57:3, 65:2, 66:1,
  75:32, 78:1, 87:1.
- Output: `data/family_selection.json` (deterministisch, `sort_keys`, indent 2).

## Familie (D3, `scripts/build_family_tokens.py`)

- 761 Runs -> `data/hexarow_tokens_family_polar.pt` und `..._cart.pt`.
- Split geom-disjunkt (`conditioning.split_by_geometry`, 90/10, seed 0):
  **683 train / 78 val**, **323 / 36 Geometrien**.
- Tokens: polar Median 491 (min 279, max 609); cart Median 371 (min 211, max 461).
  Verhaeltnis cart/polar-Median = 0.756 (~0.75 Vertex-Anteil erwartet). Blocks-Median 21.
- Bounds dataset-weit (`bounds_from`): r `[0.5503, 1.8674]`, z `[-0.05, 2.55]`.
- `is_blade`: aus `sample.npz` `surface_points` + `surface_tris` +
  `surface_tri_label`, Label **5** (`conditioning.BLADE_LABEL`).
  Labelsemantik-Hinweis: `tet_prep.py` setzt `SURF_BLADE=5`; im v5-Datensatz ist
  5 = `bl_interface_hub` (blade-seitige Schnittflaeche, da die Blade-Wand im
  reduzierten Gebiet fehlt). Mittlerer Blade-Anteil der Oberflaechenpunkte 10.5 %.
- Erzeugte Sample-Keys (nur konsumierte): `vertices_polar`, `vertices_cartesian`,
  `faces[8,B]`, `edge_index[2,E]`, `surface_points`, `is_blade`. Bewusst
  weggelassen: `center`, `quad_faces`, `tri_coordinates`, `edge_ctrl`
  (vom Tokenizer/Conditioning nicht gelesen).

## TFI h=0.5 (D4)

- User-Floor h>=0.5. Refill ist **reines Python** (`tfi.py`:
  `load_blocks -> lattices -> direction_classes -> solve_block_divisions ->
  refill_complex`; numpy/scipy/meshio, kein AlgoHex, kein Cluster).
- Input: `<run>/blocks.vtk` (in allen 761 selektierten Dirs vorhanden);
  `machine_XXXX_tet.vtk` nur fuer die Label-Ausgabe, nicht fuer den Refill.
- Modus: `scripts/build_family_tokens.py --tfi-h 0.5`. Kein Schreibzugriff auf
  `data/hex3d_algohex/` (Refill in-memory, kanonischer Hash in tmp).
- Ergebnis: **761/761 Refill fehlerfrei**; Zellen min 36, Median 76, p95 117,
  max 135; davon `<= Cap 25`: **0**.
- **Blocker (Cap-Konflikt):** h=0.5 ist die coarseste erlaubte Aufloesung und
  liefert 36..135 Zellen. Da h<0.5 verboten ist, kann kein Augment-Sample den
  Cap 25 erfuellen -> 0 Samples. `data/hexarow_tokens_family_aug_h05_polar.pt`
  daher **nicht erzeugt** (kein Hack wie Cap-Anhebung).
- Optionen: (1) eigener Cap fuer den Augment-Arm (z.B. <= 135) und erneut
  `--tfi-h 0.5` laufen lassen; (2) h groeber als 0.5 (verletzt User-Floor);
  (3) Augmentation verwerfen. Laufzeit 761 Refills ~13 s (24 Worker) - kein
  Zeitblocker, keine Nondeterminismus-Beobachtung.
- Reproduktion: `uv run --with numpy --with torch --with scipy --with meshio
  python scripts/build_family_tokens.py --tfi-h 0.5`.

## Gates

- `scripts/test_conditioning_parity.py`: exit 0 - Paritaet bit-identisch, Split
  disjunkt, Weighting multiplizitaets-treu.
- `scripts/test_family_tokens.py`: exit 0 - 761 je Arm, Split disjunkt,
  roundtrip ok + 0 ungueltige VTK-Relabels (maxerr polar 0.0056, cart 0.0075).
- `scripts/test_slot_parity.py`: exit 0.
- `scripts/smoke_mesh_validation.py`: PASS.
