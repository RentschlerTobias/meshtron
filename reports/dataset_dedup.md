# Dataset-Deduplizierung AlgoHex n-Sweep

Grid-Identitaet: blake2b-16 ueber den kanonikalisierten Zellkomplex (Knoten-Relabeling nach quantisierten Koordinaten, Block-Sortierung, Quantisierung 1/1e+04). Geometrie-Identitaet aus `data/geom_ids.json` (read-only).

Keep-Policy: keep one run per (geom_id,grid_id), prefer n8000 over n2000; sweep n4000 over n1000; tie -> lexicographically first root.

## Runs pro Scan-Root

| Root | Runs | Unique Grids | Behalten | Verworfen |
|---|---:|---:|---:|---:|
| batch | 739 | 739 | 739 | 0 |
| batch_t19_sweep | 94 | 94 | 93 | 1 |

Variante-Verzeichnisse gescannt: 3668; gelesen: 833; fehlgeschlagen (npz fehlt/korrupt): 2835.

## Verteilung n_unique_grids pro Geometrie

`{1: 16, 2: 314, 3: 4, 4: 44}`

## Duplikate

- Runs insgesamt: 833
- Unique Grids: 832
- Geometrien: 378
- Geometrien mit >= 2 Runs: 362
- Als Grid-Duplikat verworfen: 1 (0.1 %)

## Antwort: n-Sweep

0/362 Geometrien mit >= 2 Runs liefern genau ein Grid (0.0 %) - ueberwiegend verschiedene Grids.

## Sonderfaelle

- machine_0006 hat geom_id null; Runs ohne Geometrie: 0.

## Implikation P1

Trainingsset nach Dedup: 832 Runs (ein Run pro (geom_id, grid_id)); 832 eindeutige Grids statt der rohen Run-Zahl. Ohne Dedup wuerden identische Grids mehrfach gewichtet.
