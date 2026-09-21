# Phase-0/P0-Diagnose: Overfit-Hexbloecke GT vs. generiert

Datum: 2026-09-21. Repo: `stack/meshtron`. Pfad: HexaRow 3D
(`train_hexarow_full.py` -> `generate.py` -> `mesh_validation.py`).
Analyse: `scripts/diagnose_overfit_blocks.py` (deterministisch, read-only).

## Verdikt

**(c) Emitter konsistent (gefixt)** — `build_row_plan` uebernimmt den Entry-Ring jedes Folgeblocks exakt aus dem Exit-Ring des Vorgaengers und bildet den Exit-Ring per axialem Pairing; damit ist die Row-Grammatik per Konstruktion round-trip-konsistent: 0 ungueltige Relabelings, 0 neu invertierte Zellen je Arm.

Die Block-*Zerlegung* ist korrekt (12/12 Koordinatenmengen je Arm innerhalb tol 0.02, max. Koordinatenfehler < 0.007 = Tokenisierungs-Quantisierung). Der Row-Emitter ist per Konstruktion round-trip-konsistent: **0/12 ungueltige Relabelings, 0/12 neu invertierte Zellen je Arm**. Verbleibende `min det(J) < 0` Zellen (gen 5->GT 4, gen 11->GT 7) sind GT-treue Grobblock-Artefakte: die GT-Bloecke [4, 7] sind selbst gefaltet und `min det(J)` ist invariant ueber alle 48 gueltigen Hex-Relabelings.

## Quell-Konvention der GT-Bloecke

Die GT-Bloecke (`data/polytron_data_3d_smoke.pt`, Sample 3,
`faces` [8,F] -> GT-Bloecke = `faces.T`) sind in **Standard-VTK_HEXAHEDRON-
Ordnung mit positiver Orientierung**. Belege in der erzeugenden Pipeline
`stack/domain_partition_3D/experimentell/hex3d_algohex/`:

- `export_sample.py:34-40` — `CORNERS = [(0,0,0),(1,0,0),(1,1,0),(0,1,0),
  (0,0,1),(1,0,1),(1,1,1),(0,1,1)]` = VTK-Ordnung; `FACES` = VTK-Face-Table.
- `tfi.py:43-46` — `CORNER` identisch; `tfi.py:148-160` `_fix_handedness`
  flippt einen Block bei negativem `ovm_io._hex_volume`.
- `ovm_io.py:130-160` `hex_cell_vertices` liefert explizit
  "VTK_HEXAHEDRON order" und flippt auf positives Volumen;
  `ovm_io.py:190` schreibt `CELL_TYPES` = 12.
- `ovm_io.py:107-127` korrigierte `_hex_volume` (Standard-6-Tet).

Damit ist die korrekte Abbildung "Block-Knotenordnung -> VTK-Hex-Ordnung"
die **Identitaet** (keine Permutation noetig). Die Ad-hoc-Probe-Permutation
`(0, 1, 2, 4, 7, 3, 6, 5)` ist ungerade: sie invertiert die Orientierung *jeder* Zelle
und ist daher kein Fix, sondern nur die entgegengesetzte Vorzeichen-
Konvention (siehe Abschnitt "Abgleich mit der Ad-hoc-Probe").

## GT-Referenz

12 Bloecke, 40 Vertices. Signierte Volumina (VTK):
Summe 5.2532, Minimum +0.03557. **Die GT-Bloecke selbst
sind nicht alle ideal:** min det(J) < 0 bei GT-Block/-Bloecken [4, 7] —
das sind grobe Block-Eck-Hexaeder (nicht die feinen, validen Hex-Zellen), die
durch die Block-Komplex-Vergroeberung leicht invertiert sein koennen. Das ist
Referenz-Kontext, kein Generierungsfehler.

## Methode

Je Arm: Tokens (`hexarow_overfit_1sample{,_cart}.pt`) + Sequenz
(`seq_overfit_{polar,cart}.pt`) -> `HexaRowTokenizer(r_bounds=..., z_bounds=...)`
-> `generate.detokenize_safe(seq, tok, tok.core.stop_token, coords=...)`;
polar -> kartesisch via `(r cos th, r sin th, z)`. Pro Blockpaar (Zentroid-NN
+ bipartiter Koordinatenmengen-Check):

- `vol(VTK)` = signed volume, Standard-VTK-6-Tet-Zerlegung (Diagonale 0-6).
- `vol(Probe-Perm)` = signed volume nach `(0, 1, 2, 4, 7, 3, 6, 5)`.
- `min det(J)` = Minimum der Jacobi-Determinante der trilinearen Hex-Abbildung
  auf 9^3 Stuetzstellen; `< 0` => Zelle faltet sich (invertiert).
- `max Planaritaet` = groesster Abstand eines Face-Knotens von der
  Newell-Ebene; gross gegenueber den GT-Faces => die 4 Knoten bilden kein
  Face.
- `Faces outward` = Anzahl der 6 VTK-Faces mit outward Newell-Normale.
- `Relabeling gueltig` = die per Koordinaten-NN bestimmte Slot-Abbildung
  bildet die 6 generierten VTK-Quads exakt auf die 6 GT-Quads ab.

## Arm `polar` (coords=`polar`)

12 generierte Bloecke, 40 Vertices; Koordinatenmengen alle innerhalb tol 0.02: **ja**; distinkte GT->gen-Slot-Permutationen **8/12**; alle Relabelings VTK-gueltig: **ja**; ungueltige Relabelings **0/12**, davon neu invertiert **0/12**.

| gen | gt | Zentroid-d | Koord-err | vol(VTK) | vol(Probe-Perm) | min det(J) | max Planaritaet | Faces outward | Kante<=1.25*bbox | Relabeling gueltig |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.0009 | 0.0033 | +0.64799 | -0.19789 | +0.23478 | 0.1255 | 6/6 | ja | ja |
| 1 | 6 | 0.0013 | 0.0048 | +0.93404 | -0.22963 | +0.43669 | 0.0867 | 6/6 | ja | ja |
| 2 | 1 | 0.0016 | 0.0048 | +0.43300 | -0.07934 | +0.18130 | 0.0909 | 6/6 | ja | ja |
| 3 | 10 | 0.0010 | 0.0060 | +0.28871 | -0.04828 | +0.12815 | 0.0727 | 6/6 | ja | ja |
| 4 | 5 | 0.0008 | 0.0055 | +0.04730 | -0.02268 | +0.02244 | 0.0727 | 6/6 | ja | ja |
| 5 | 4 | 0.0003 | 0.0055 | +0.11414 | -0.07373 | -0.00278 | 0.1351 | 6/6 | ja | ja |
| 6 | 8 | 0.0013 | 0.0052 | +0.08549 | -0.04967 | +0.03221 | 0.0825 | 6/6 | ja | ja |
| 7 | 11 | 0.0006 | 0.0055 | +0.22481 | -0.02246 | +0.09619 | 0.0739 | 6/6 | ja | ja |
| 8 | 2 | 0.0004 | 0.0060 | +0.53482 | -0.17771 | +0.22344 | 0.1579 | 6/6 | ja | ja |
| 9 | 9 | 0.0011 | 0.0037 | +1.07356 | -0.25893 | +0.50172 | 0.1080 | 6/6 | ja | ja |
| 10 | 3 | 0.0010 | 0.0055 | +0.62049 | -0.08536 | +0.21424 | 0.1807 | 6/6 | ja | ja |
| 11 | 7 | 0.0007 | 0.0050 | +0.05084 | -0.01369 | -0.03017 | 0.1311 | 5/6 | ja | ja |

Permutationssuche ueber alle 8! = 40320 (GT+generiert, 24 Bloecke): **3900** Permutationen mit durchweg positivem Volume. Identity enthalten: **True**; Probe-Permutation `(0, 1, 2, 4, 7, 3, 6, 5)` enthalten: **False**.

### Emitter-Konsistenz (`build_row_plan`)

```
  row=[0, 6, 1] links=['-', 'ok', 'ok'] emit-gueltig=[True, True, True]
  row=[10, 5] links=['-', 'ok'] emit-gueltig=[True, True]
  row=[4, 8] links=['-', 'ok'] emit-gueltig=[True, True]
  row=[11] links=['-'] emit-gueltig=[True]
  row=[2, 9, 3] links=['-', 'ok', 'ok'] emit-gueltig=[True, True, True]
  row=[7] links=['-'] emit-gueltig=[True]
```


## Arm `cart` (coords=`cart`)

12 generierte Bloecke, 40 Vertices; Koordinatenmengen alle innerhalb tol 0.02: **ja**; distinkte GT->gen-Slot-Permutationen **8/12**; alle Relabelings VTK-gueltig: **ja**; ungueltige Relabelings **0/12**, davon neu invertiert **0/12**.

| gen | gt | Zentroid-d | Koord-err | vol(VTK) | vol(Probe-Perm) | min det(J) | max Planaritaet | Faces outward | Kante<=1.25*bbox | Relabeling gueltig |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.0010 | 0.0037 | +0.64762 | -0.20013 | +0.23407 | 0.1277 | 6/6 | ja | ja |
| 1 | 6 | 0.0007 | 0.0048 | +0.93247 | -0.22806 | +0.43233 | 0.0853 | 6/6 | ja | ja |
| 2 | 1 | 0.0007 | 0.0048 | +0.43441 | -0.07448 | +0.18652 | 0.0985 | 6/6 | ja | ja |
| 3 | 10 | 0.0007 | 0.0041 | +0.28821 | -0.04765 | +0.12724 | 0.0777 | 6/6 | ja | ja |
| 4 | 5 | 0.0009 | 0.0041 | +0.04725 | -0.02290 | +0.02046 | 0.0777 | 6/6 | ja | ja |
| 5 | 4 | 0.0005 | 0.0049 | +0.11504 | -0.07437 | -0.00125 | 0.1338 | 6/6 | ja | ja |
| 6 | 8 | 0.0012 | 0.0049 | +0.08561 | -0.04916 | +0.03391 | 0.0846 | 6/6 | ja | ja |
| 7 | 11 | 0.0005 | 0.0048 | +0.22486 | -0.01965 | +0.10000 | 0.0748 | 6/6 | ja | ja |
| 8 | 2 | 0.0004 | 0.0041 | +0.53564 | -0.17823 | +0.22188 | 0.1542 | 6/6 | ja | ja |
| 9 | 9 | 0.0013 | 0.0041 | +1.07597 | -0.25645 | +0.50816 | 0.1089 | 6/6 | ja | ja |
| 10 | 3 | 0.0013 | 0.0041 | +0.61763 | -0.08266 | +0.21861 | 0.1842 | 6/6 | ja | ja |
| 11 | 7 | 0.0005 | 0.0041 | +0.05064 | -0.01356 | -0.03097 | 0.1314 | 5/6 | ja | ja |

Permutationssuche ueber alle 8! = 40320 (GT+generiert, 24 Bloecke): **3864** Permutationen mit durchweg positivem Volume. Identity enthalten: **True**; Probe-Permutation `(0, 1, 2, 4, 7, 3, 6, 5)` enthalten: **False**.

### Emitter-Konsistenz (`build_row_plan`)

```
  row=[0, 6, 1] links=['-', 'ok', 'ok'] emit-gueltig=[True, True, True]
  row=[10, 5] links=['-', 'ok'] emit-gueltig=[True, True]
  row=[4, 8] links=['-', 'ok'] emit-gueltig=[True, True]
  row=[11] links=['-'] emit-gueltig=[True]
  row=[2, 9, 3] links=['-', 'ok', 'ok'] emit-gueltig=[True, True, True]
  row=[7] links=['-'] emit-gueltig=[True]
```


## Abgleich mit der Ad-hoc-Probe

Die Probe behauptete negative Vorzeichen-Volumina unter Standard-VTK fuer GT
UND generierte Bloecke sowie eine Permutation `(0,1,2,4,7,3,6,5)`, die alle
12 GT-Bloecke positiv mache. **Das wird hier widerlegt:**

1. Unter der Standard-VTK-6-Tet-Zerlegung sind die signierten Volumina von
   **GT und generierten Bloecken bereits im identischen (positiven) Vorzeichen**
   — dieselbe Formel, die `mesh_validation._block_volumes` benutzt und die
   `mesh_validation` als "valid" durchlaufen laesst. Es gibt keinen
   Vorzeichenwechsel zwischen den Armen.
2. Die Probe-Permutation `(0, 1, 2, 4, 7, 3, 6, 5)` ist ungerade (4-Zyklus
   `3->4->7->5->3`, Paritaet -1) und negiert damit das Vorzeichen *aller*
   Zellen gleichzeitig. Sie "repariert" nichts — sie kodiert nur die
   entgegengesetzte Vorzeichen-Konvention. Ihre 24/24-Positivaussage ist ein
   Artefakt dieser Konvention; die Suche bestaetigt `probe_ok = False`.
3. **Keine** einzelne globale 8!-Permutation behebt die tatsaechliche
   Restdifferenz: die generierte Slot-Ordnung variiert pro Block (viele
   distinkte GT->gen-Permutationen), und die drei ungueltigen Relabelings
   brauchen je eine andere Korrektur (im Extremfall die blockweise
   Rueckordnung auf die GT-Reihenfolge). Ein konstantes Remap gibt es nicht.

## Root Cause (historisch, Task A)

Die Row-Grammatik dedupliziert: der Kopf einer Row emittiert 8 Vertices
(Entry-Ring + Exit-Ring), jeder Folgeblock nur seinen Exit-Ring (4 Vertices);
`detokenize` setzt den Entry-Ring eines Folgeblocks gleich dem Exit-Ring des
Vorgaengers (`hexa_row_tokenizer.py:409-415` Tokenize,
`hexa_row_tokenizer.py:486-498` Detokenize). `build_row_plan` berechnet aber
den Entry-Ring jedes Blocks **frisch** (`_rotate_orient_ring` mit lex-min-
Start, `hexa_row_tokenizer.py:248-281`). Dadurch gilt fuer manche
Folgebloecke `emit[b][0:4] != emit[prev][4:8]` — der Ring ist rotiert oder
gespiegelt — und `detokenize` reassembliert eine *andere* Konnektivitaet als
`emit[b]` beschreibt. Der Emitter-Abschnitt jeder Arm-Tabelle listet genau
diese Faelle (`reversed`/`rotk`) und die emit-gueltigen/ungueltigen Bloecke.

Konsequenz: das exportierte VTK (`generate.py:write_vtk`,
`scripts/compare_viz.py`) schreibt Zell-Typ 12 in Grammatik-Knotenordnung.
Fuer die betroffenen Bloecke ist das keine gueltige Hexaeder-Konnektivitaet;
ParaView rendert dort gefaltete Zellen. `mesh_validation` uebersieht das, weil
ihm ein Orientierungs-/Topologie-Check fehlt.

## Fix (Phase-0 Task B)

`build_row_plan` uebernimmt den Entry-Ring jedes Folgeblocks exakt aus dem
Exit-Ring des Vorgaengers (Rotation ODER Reversal; sonst Row-Break) und bildet
den Exit-Ring als axiales Pairing-Follow des Entry-Rings. Damit ist `emit[b]`
per Konstruktion eine gueltige VTK-Relabelung und die Row-Grammatik
round-trip-konsistent. `mesh_validation.validate_generated_mesh` prueft
zusaetzlich die 6 VTK-Face-Knotenmengen (Hex-Inzidenz) und die trilineare
Jacobi-Determinante auf 3x3x3 Stuetzstellen. `scripts/test_slot_parity.py`
bleibt gruen; `scripts/smoke_mesh_validation.py` deckt Winding-Flip und
gefaltete Zelle ab.

## Repro

```bash
uv run python scripts/diagnose_overfit_blocks.py     # schreibt diesen Report
uv run python scripts/test_slot_parity.py            # muss GREEN bleiben
```
