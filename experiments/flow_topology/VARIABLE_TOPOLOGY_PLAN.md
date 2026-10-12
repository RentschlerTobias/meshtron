# Plan: variable Blocktopologie mit Flow → Pointer → Flow (2026-10-12, freigegeben)

## Ausgangslage

- `flow_geo.py` erzeugt Ecken und Bezier-Offsets für **eine** feste Topologie:
  - `Topology` liest `M`=40 und `E`=84 aus dem ersten Trainings-Sample.
  - Jeder Token trägt Template-Position, Laplace-Eigenvektoren des einen Graphen und eine Normierung pro Token.
  - Die Konnektivität `conn` kommt bei der Inferenz von außen.
  - Die Kurven entstehen als 6 sehnenrelative Offsets pro **bekannter** Kante (`bezier_pts`).
- Stärken des Flow-Modells gegenüber AR-Polytron (`FLOW_REPORT.md`, val, GT-Topologie):
  - Ecken-Fehler 50,8 statt 86,9.
  - Unbedeckte Fläche 1,5 % statt 20,2 %.
  - Invertierte Zellen 0,05 % statt 1,1 %.
  - Ursache: kontinuierlich, Ecken und Kurven gemeinsam, keine Reihenfolge.

  Es erzeugt aber keine diskrete Struktur.
- Polytron Stufe 2 (`train.py` Task `conn`, `infer_chain_v2.decode_conn`) erzeugt `conn` als Pointer-Folge:
  8 Zeiger auf Vertex-Indizes pro Block, dann Stopp. Auf val 35/35 exakt. Das ist fast trivial, weil es nur eine Topologie gibt.
- Die Einheits-Topologie entsteht in `beam_collapse`:
  - AlgoHex liefert 16 bis 107 Blöcke.
  - Die Beam-Suche kollabiert auf 12 Blöcke (146/148 Samples `topo_code 49a0142abb`) und durchläuft unterwegs gültige Zwischenzustände (z. B. 16 Blöcke).
  - Roh-Samples mit 10/16/21/36/42 Blöcken werden bisher aussortiert.
- Schon topologie-unabhängig sind: `continuity.py`, `curve_opt.py` (Seam-Bögen, TFI-Optimierung), `refill_polytron` (TFI),
  der Tokenizer-Prototyp `/work/meshtron_topo` (kanonische Blockreihenfolge).

## Entscheidungen (2026-10-12)

- **Ziel pro Geometrie ist genau eine Topologie: die minimal notwendige Blockzahl.** Zwischenzustände der Beam-Suche sind keine Trainingsziele.
- **Die Anzahl ist bei neuen Geometrien unbekannt und muss vorhergesagt werden.** Stufe 1 sagt die Anzahl der Ecken voraus, die Blockzahl folgt aus der Pointer-Folge.
- **Verschiedene Topologien im Datensatz entstehen nur über verschiedene Geometrieklassen**, also Familien von Geometrien mit verschiedener minimaler Blockstruktur.

## Ziel

Für eine Geometrie (Punktwolke mit Patch-Labels) eine Blockstruktur **beliebiger Größe** erzeugen: Ecken, Blöcke und gekrümmte
Kanten. Das Modell soll auf Topologien übertragbar sein, die im Training nicht vorkommen.

## Architektur: drei Stufen

```
Punktwolke ─► [1] Flow: Ecken  ─► [2] Pointer: Blöcke (conn)  ─► [3] Flow: Kurven (+ Ecken-Nachjustierung)  ─► Seam-Bögen + TFI-Opt ─► Netz
               N Punkte, Menge      8 Zeiger pro Block, AR          6 Offsets pro Kante, aus conn abgeleitet
```

Jede Stufe macht, worin sie stark ist: Flow die kontinuierliche Geometrie, der Pointer die diskrete Struktur.

### Stufe 1: Flow-Ecken (`flow_vertex.py`)

- **Tokens:** N Vertex-Tokens mit je 3 Koordinaten, global normiert (Bounding-Box bzw. `r_bounds`/`z_bounds`). Kein Template, keine Identität pro Token.
- **Ablauf:** Rauschen → Ecken. Konditionierung über Cross-Attention auf die PointPerceiver-Latents (wie heute).
- **Anzahl N:**
  - Ein Kopf auf den Punktwolken-Latents sagt N vorher, als Klassifikation über N = 8…N_max.
  - Bei der Inferenz werden die Top-k-Werte für N gesampelt und am Ende über die Netzqualität gerankt.
  - Alternativ wird N vorgegeben, wie bei PolyFlow.
- **Training:** Rectified Flow auf der ungeordneten Menge. Rauschen und GT werden pro Batch per Hungarian-Algorithmus gekoppelt (OT-Kopplung wie MeshFlow), sonst lernt das Modell Mittelwerte.
- **Token-Merkmale:** Es gibt keine Topologie-Merkmale, denn die ist hier noch unbekannt. Optional ein vorhergesagtes Patch-Label pro Ecke als Hilfsziel (Multi-Hot), das Stufe 2 nutzt.
- **Padding und Masken** für variables N im Batch.

### Stufe 2: Pointer-Blöcke (`pointer_conn.py`, Basis `train.py` Task `conn`)

- **Eingabe:** die erzeugte Eckenmenge (ungeordnet, Set-Encoder ohne Positionskodierung) plus Punktwolken-Latents.
- **Ausgabe:** autoregressive Folge von Zeigern in die Eckenmenge, 8 pro Block (in VTK-Reihenfolge), dann Stopp.
- **Kanonische Zielreihenfolge** der Blöcke: Tokenizer `/work/meshtron_topo` (Rahmentransport, kanonischer Anker,
  `--merge-cut-labels`). Die Ecken selbst bleiben ungeordnet. Im Training werden sie gemischt, und die Ziele sind Indizes in die gemischte Menge.
- **Gültigkeit beim Decodieren erzwingen** (Masken auf die Zeiger):
  - keine doppelte Ecke im Block;
  - Würfel-Kantenstruktur: Zeiger 2…8 nur auf Ecken, die mit den bisherigen Ecken des Blocks eine konsistente Hex-Orientierung ergeben (positive Volumina der Eck-Tetraeder);
  - jede Fläche höchstens zwei Blöcken zugeordnet;
  - Stopp erst, wenn alle Flächen mit nur einem Block auf dem Rand liegen (Plausibilität über die Punktwolke) oder N_max Blöcke erreicht sind.
- **Robust gegen Fehler aus Stufe 1:**
  - Im Training gestörte GT-Ecken (Rauschen in der Größe des Stufe-1-Fehlers, ≈ 50e-3) sowie echte Stufe-1-Samples.
  - Überzählige Ecken dürfen unbenutzt bleiben, sie werden danach verworfen.
- **Mehrere Kandidaten** per Beam-Suche oder Sampling. Gerankt wird nach Gültigkeit und Netzqualität (wie `rank_key` in `polytron_tfi.py`).

### Stufe 3: Flow-Kurven und Ecken-Nachjustierung (`flow_geo_var.py`)

- **Topologie bekannt:** Aus `conn` folgen die Kanten. Merkmale **pro Sample** aus dem eigenen Graphen:
  - Random-Walk-PE (vorzeichen-invariant);
  - Valenz, Rand oder innen;
  - Rolle aus `continuity.detect` (Ecke, Feature, regulär, singulär);
  - Kantenklasse (Feature, 2D, 3D, innen);
  - Patch-Multi-Hot der Randflächen.
- **Graph-Bias in der Attention:** Hop-Abstand und Inzidenz Vertex↔Kante, im Graphormer-Stil.
- **Tokens:** Vertex-Tokens (3) und Kanten-Tokens (6 Offsets), wie heute.
  - Die Ecken starten nicht aus reinem Rauschen, sondern aus den Ecken von Stufe 1 plus Rauschen bei t₀ (SDEdit-artig). So passen sie sich der gewählten Topologie an, ohne neu erfunden zu werden.
  - Die Kanten starten aus Rauschen.
- **Training** auf GT-Ecken und GT-`conn`, mit gestörten Ecken als Start. Ziele sind die optimierten Kurven aus
  `curve_opt.py`: Seam-Bögen als G1-Kubik-Fit, TFI-optimierte Kanten.

### Nachbearbeitung (vorhanden)

`continuity.detect`, dann Seam-Bögen für Feature-Kanten, dann TFI-Optimierung der übrigen Kanten (`curve_opt.py`,
Hinge-Variante, Garanzha war schlechter), dann TFI und Netz.

## Phase 1: Meilensteine auf den vorhandenen 12-Block-Daten

Auf den vorhandenen Daten lässt sich die ganze Kette ohne neue Datengenerierung prüfen:

| Meilenstein | Kriterium |
|---|---|
| M1 Stufe 3 allein (GT-`conn`, Ecken aus `flowp`-Modell) | so gut wie das heutige `flowp_flow_all` (Chain ≤ 1 % unbedeckt auf val und val_new) |
| M2 Stufe 1 allein | Ecken-Fehler ≤ heutiger Flow-Fehler (≈ 50e-3) **ohne** Template; N korrekt (40) in ≥ 95 % |
| M3 Stufe 2 auf Stufe-1-Ecken | `conn` exakt gleich der GT-Topologie in ≥ 95 % (nicht auf GT-Ecken, sondern auf erzeugten) |
| M4 Kette 1→2→3 + Nachbearbeitung | Chain-Werte wie M1, alle Netze watertight |

Fällt M2 wegen des fehlenden Templates deutlich ab (Risiko R2), wird Stufe 1 auf eine grobe Startgeometrie konditioniert.

## Phase 2: Daten mit mehreren Topologien

Das ist der eigentliche Engpass. Ohne verschiedene Topologien lernt der Pointer nur die eine Tabelle.

1. **Minimalität sicherstellen:** Die Beam-Suche liefert heute die kleinste gültige Blockzahl, die sie findet. Pro Geometrie
   wird nur dieses Minimum gespeichert (wie bisher). Prüfen, ob `beam_collapse` mit größerer Breite/Tiefe für einzelne
   Geometrien noch weiter reduziert, damit das Ziel wirklich minimal und konsistent ist. Das betrifft `domain_partition_3D`
   und erfolgt **erst nach dem laufenden Datagen-Lauf, Zieldatei vorher abstimmen.**
2. **Bisher aussortierte Roh-Samples** (10/16/21/36/42 Blöcke) prüfen: Ist das jeweils das Minimum für diese Geometrie (dann aufnehmen) oder ein unvollständiger Kollaps (dann verwerfen)?
3. **Neue Geometrieklassen** über dtOO. Das ist die eigentliche Quelle verschiedener Topologien: andere Schaufelfamilien, Francis mit Splitter, Pumpen, Kanäle. Später HexMe/ABC.
4. **Format:** pro Item eigenes `conn`/`edges`/`verts` plus `topo_code` aus `audit.csv`, damit nach Topologie gesplittet werden kann.
   Neuer Builder `build_flow_dataset_var.py`.
5. **Splits:** `val`, `val_new` und neu `val_topo` mit Topologien, die im Training nicht vorkommen.

## Phase 3: Training und Bewertung mit mehreren Topologien

- Die Stufen getrennt trainieren (Teacher Forcing auf GT), dann die Kette auf `val_topo`.
- Metriken:
  - Stufe 1: Ecken-Fehler (Hungarian-Zuordnung zur GT), Trefferquote von N.
  - Stufe 2: Anteil gültiger Topologien, Anteil mit exakt der GT-Topologie, GT unter den Top-k.
  - Kette: unbedeckte Fläche, invertierte Zellen, sj_p1, Non-Ortho, Ring- und Feature-Knicke (`summarize_curve_opt.py`).
- Später Qualitäts-Reward pro Block bzw. Kante (Masked-DPO-Idee aus Mesh-RFT) für Stufe 2 und Stufe 3.

## Risiken

| # | Risiko | Gegenmaßnahme |
|---|---|---|
| R1 | Zu wenige verschiedene Topologien | neue Geometrieklassen in der Datengenerierung (Phase 2.3) |
| R2 | Ohne Template sinkt die Genauigkeit der Ecken | OT-Kopplung, Startgeometrie, Stufe 3 justiert die Ecken nach |
| R3 | Fehler aus Stufe 1 (fehlende oder überzählige Ecken) blockieren Stufe 2 | N-Kopf mit Top-k, Pointer darf Ecken ignorieren, Training auf gestörten bzw. echten Stufe-1-Ecken |
| R4 | Pointer erzeugt ungültige Blöcke | Masken beim Decodieren, Gültigkeitsprüfung nach Gao 2017, Beam/Sampling plus Ranking |
| R5 | Symmetrische Ecken (Automorphismen) | Labels der Randflächen reduzieren auf ≈1 Automorphismus (meshtron_topo: ties=1) |
| R6 | Lange Pointer-Folgen bei großen Topologien (100 Blöcke = 800 Zeiger) | unkritisch für AR auf der 3090; KV-Cache |
| R7 | Kollision mit dem laufenden Datagen-Lauf | Änderungen an `domain_partition_3D` erst danach, Zieldateien vorher abstimmen |

## Reihenfolge

1. Stufe 3 (`flow_geo_var.py`) auf 12-Block-Daten → M1.
2. Stufe 1 (`flow_vertex.py`) → M2.
3. Stufe 2 (Pointer mit kanonischer Reihenfolge aus meshtron_topo, Masken, Training auf gestörten Ecken) → M3, dann die Kette → M4.
4. Nach dem Datagen-Lauf: Minimalität prüfen, neue Geometrieklassen erzeugen (Phase 2), mit mehreren Topologien trainieren (Phase 3).

## Offene Frage an dich

- Welche Geometrieklassen kommen als nächstes in die Datengenerierung?
