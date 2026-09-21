# Cartesian vs. Polar — Single-Mesh-Overfit-Beweis

Datum: 2026-09-21. Repo: `stack/meshtron`, Pfad: HexaRow 3D (`train_hexarow_full.py` → `generate.py` → `mesh_validation.validate_generated_mesh`).

## These und Ergebnis

Früherer Zustand: Train-Loss sinkt, freie Generierung liefert nie eine gültige Blockstruktur.
Diagnose: **Slot-Embedding-Konventionsfehler** — `batchify` (Training) fütterte den Slot des *vorherigen* Tokens, `generate.py` (Inference) den Slot des *Target*-Tokens (`cnt % npt`). Frei laufend sah das Modell damit nie trainingskonsistente Slot-Embeddings; `slot_mask()` war unabhängig davon korrekt, deshalb stimmte Token-*Klasse*, aber nicht Koordinaten-*Identität*.

Fix: eigene Slot-Konvention auf beiden Seiten (Training unshifted `slot[i,:len] = _slot_ids(tk)`, Inference-Feed `0 if cnt==0 else (cnt-1)%npt`), `npt` parametrisiert (4=polar, 3=cartesisch). Regressionstest: `scripts/test_slot_parity.py` (rot → grün).

**Beweis: Beide Tokenisierungen sind mit unveränderter Architektur lernbar.** Nach dem Fix reproduziert greedy Free-Running (`--top-k 1`) das Trainings-Mesh **Token für Token exakt** und besteht das Validierungs-Gate.

## Experiment-Setup (identisch für beide Arme)

- Mesh: `data/polytron_data_3d_smoke.pt`, Sample `sample3` (Index 3): 12 Blöcke, 40 Vertices
- Tokens: `scripts/build_hexarow_tokens.py --only 3 [--coords cart]` (train==val Duplizierung, deterministische Punktwolke `rng(0)`)
- Modell: `GPTCond` d=256, 6 Layern, 8 Heads → **5.8M Parameter**
- Optimizer: AdamW lr=3e-4, warmup 50, cosine, 3000 Epochen, **dropout=0, wd=0**, 1 Batch/Epoche
- Validierung: `generate.py --top-k 1 --idx 3 … ` → Exit 0 + `mesh validation: valid`

## Ergebnisse

| Metrik | Polar (4 tok/Vertex) | Cart (3 tok/Vertex) |
|---|---|---|
| Stream-Länge (GT) | 296 Tokens | **224 Tokens (−24.3 %)** |
| Final train loss / tok-acc | 0.000000 / 1.000000 | 0.000 / 1.000 |
| EpochGate (loss<0.05, acc>0.99) | ≤ 3000 (Endlauf) | **72** |
| Epoch perfekt (loss→0, acc=1) | ≤ 3000 (Endlauf) | **87** (VAL perfekt: 99) |
| Trainingszeit 3000 Ep. | ~6 min | 351 s |
| Greedy-Seq vs. GT | **295/295 identisch**, dann am `--max-tokens`-Cap gestoppt (STOP = GT-Token 296) | **223/223 identisch**, analog (STOP = GT-Token 224) |
| Detokenisierung | verts=40 blocks=12, kein Trim | verts=40 blocks=12, kein Trim |
| `mesh validation:` | **valid** (Exit 0) | **valid** (Exit 0) |

Geometrie: greedy rekonstruiert die exakten GT-Tokens; die verbleibende Koordinatenabweichung ist rein die Tokenisierungs-Quantisierung (Tokenizer-Roundtrip-Max-Fehler ≈ 0.0074 bei `center=0`, `hexa_row_tokenizer.py`). Beide Arme sind geometrisch damit gleichwertig; Cart spart 24 % Sequenzlänge (dichte-Packing-Vorteil) und konvergiert tendenziell schneller (kürzerer Stream, weniger Zuordnungsaufwand pro Vertex).

Artefakte (gitignoret): `data/hexarow_overfit_1sample{,_cart}.pt`, `data/hexarow_overfit_{polar,cart}.pt(+_ckpt.pt)`, `data/gen_overfit_{polar,cart}.{vtk,png}`, `data/seq_overfit_{polar,cart}.pt`, `data/compare_overfit_{polar,cart}.vtk` (Part 1=true, 2=generated, 3=Conditioning-Punktwolke via `scripts/compare_viz.py`), Plots zur visuellen Kontrolle (blau generiert ≈ rot GT).

## Zusatzbefund: Conditioning-Punktwolke (Behob, mit Re-Validierung)

Die erste Versuchsreihe nutzte als Conditioning-Wolke `sample_points(vertices_polar, …)` — bei `sample3` lediglich **40 Eckpunkte**, 1000-fach mit Ersetzung gezogen. Das ist eine entartete Punktmenge (nur Cluster an Hub/Shroud-Kanten), welche die Volumen-Geometrie nicht beschreibt; die für die Laufrad-Geometrie entscheidende **Blatt-Windung über den Radius** fehlt vollständig. Der Overfit-Beweis war davon nicht betroffen (Einzel-Mesh ist trivial identifizierbar), für Generalisierung ist die Ecke-Wolke jedoch vergiftend.

Fix (symmetrisch auf beiden Seiten, Parity-Regel aus dem Slot-Bug beachtet): Conditioning jetzt auf **`surface_points` [11840×3]** — komplette Oberflächenbepunktung des Volumens. Training: `train_hexarow_full.surface_cloud()` in `prep()` (Fallback auf `vertices_polar`, wenn Feld fehlt); Inference: `generate.load_sample(…, with_surface=True)`. Polar-Rahmen (`center=0`), Normalisierung (r/z auf Checkpoint-Bounds, theta als sin/cos) und `PointEncoder` bleiben unverändert; Surface-r-Bereich [0.591, 1.809] liegt in den Bounds [0.568, 1.833]. Smoke: Trainings- und Inferenz-Wolke sind bei gleicher rng bit-identisch.

Re-Lauf beider Arme unter neuer Conditioning (diese Zahlen ersetzen die Epoch-Werte der Tabelle oben, welche Corner-Conditioning zeigen):

| | Polar | Cart |
|---|---|---|
| Erster Gate-Epoch (loss<0.05, acc>0.99) | **83** (0.036/0.993) | **80** (0.021/1.000) |
| Erster perfekter Epoch | **111** | **95** (VAL perfekt: 99) |
| Final train/val loss & acc | 0.000/1.000 beide | 0.000/1.000 beide |
| Greedy-Seq vs. GT | 295/295 identisch | 223/223 identisch |
| `mesh validation:` | **valid**, verts=40 blocks=12 | **valid**, verts=40 blocks=12 |
| Trainingszeit 3000 Ep. | 323 s | 296 s |

Befund: Surface-Conditioning beschleunigt sogar die **Memorisierung** erheblich (polar: perfect bei Ep. 111 statt erst im 3000er-Endlauf) — die Ecke-Wolke war nicht nur ein Generalisierungs-, sondern auch ein Konvergenz-Handicap. Für Full-Data-Läufe ist die Wolke jetzt ein aussagekräftiges Formsignal.

Nachtrag (Visualisierungs-Bug, behoben): Die Rücktransformation der normalisierten Wolke (r01, sinθ, cosθ, z01) → xyz war in `generate.py` und `scripts/compare_viz.py` mit falscher Klammerung implementiert (`rb[0] + r01·Δrb·cos(θ)` statt `(rb[0] + r01·Δrb)·cos(θ)`). Die Punktwolke erschien dadurch gegenüber der Geometrie diagonal um ≈(0.57, 0.57) verschoben und verzerrt — kein Blade war erkennbar. Die Conditioning des Modells war nie betroffen (das Modell konsumiert das normalisierte 4D-Tensor, vorwärts korrekt). Nach Klammer-Fix: rekonstruierte Punkte liegen auf exakt der Quell-Oberfläche (max. NN-Distanz 4.5e-16), BBox-Deckung mit GT-Hexen; alle VTKs/PNGs neu erzeugt, Overfit-Gates bleiben gültig (valid, blocks=12).

## Repro-Kommandos



```bash
uv run python scripts/build_hexarow_tokens.py --src data/polytron_data_3d_smoke.pt \
    --out data/hexarow_overfit_1sample.pt --only 3            # bzw. --coords cart
uv run python train_hexarow_full.py --tokens data/hexarow_overfit_1sample.pt \
    --src data/polytron_data_3d_smoke.pt --d 256 --layers 6 --heads 8 \
    --dropout 0 --wd 0 --lr 3e-4 --warmup 50 --epochs 3000 --val-every 100 \
    --out data/hexarow_overfit_polar.pt --ckpt data/hexarow_overfit_polar_ckpt.pt
uv run python generate.py --idx 3 --src data/polytron_data_3d_smoke.pt \
    --ckpt data/hexarow_overfit_polar.pt --top-k 1 \
    --out data/gen_overfit_polar.vtk --plot data/gen_overfit_polar.png \
    --dump-seq data/seq_overfit_polar.pt
```

Kosmetik-Befund (beide Arme): `generate.py` kappt `--max-tokens` auf die Pos-Embedding-Länge (GT-Länge −1), wodurch das allerletzte STOP wegfällt; `detokenize_safe` ergänzt STOP, Mesh bleibt valide. Kein Funktionsfehler.

## Schlussfolgerung / Next steps

1. Architektur-Kapazität war nie der Engpass — die Inference-Ungültigkeit war vollständig der Slot-Parity-Bug. Bewiesen.
2. **Cart und polar sind beide lernbar**; cart ist sequenzkürzer (−24 %) und konvergierte im Overfit schneller — Kandidat für den nächsten Full-Data-Lauf.
3. Alle vor dem Fix trainierten Checkpoints (`hexarow_slot_30ep_lt50.pt`, `hexarow_full_model_3090*`) sind unter der neuen Konvention für Free-Running **ungültig** (lagerte Slots); Full-Data- Retraining ist der nächste Schritt.
4. Memorization eines einzelnen Meshes ≠ Generalisierung; die Overfit-Probe isoliert nur die Plumbing-/Konventionsfrage.
5. Conditioning-Herkunft ist ein eigener Bug-Typ: datenbankseitig vorhandene Vollgeometrie (`surface_points`) wurde ignoriert. Seit dem Fix lernen beide Arme deutlich schneller (perfekt Ep. 111/95 statt Polar-Erstperfekt erst im 3000er-Endlauf) — Voraussetzung für den Full-Data-Lauf. `scripts/eval_stop_rate.py` nutzt weiter die alte Ecke-Wolke (bewusst nicht angefasst); bei Bedarf analog umstellen.
