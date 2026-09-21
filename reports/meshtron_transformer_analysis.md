# Technische Untersuchung: Meshtron-Transformer

**Stand:** 2026-09-20  
**Scope:** Quadtron/Meshtron und der daran angrenzende HexaRow-3D-Trainingspfad. Polytron und die Plan-B-Untersuchung sind ausdrücklich nicht Bestandteil dieses Reports.

## 1. Kurzfazit

Die vorhandenen Befunde belegen zwei getrennte Probleme:

1. **Das Modell lernt Tokenstatistik, aber die gezeigten Metriken belegen kein gültiges Mesh-Lernen.** Das aktive Training optimiert lokale nächste-Token-Cross-Entropy unter Teacher Forcing. Weder `TeacherForcingObjective` noch die Trainingsmetriken enthalten ein Signal für Face-/Row-Grammatik, Vertex-Eindeutigkeit, Adjazenz, Volumen oder Mannigfaltigkeit.
2. **Die Generationskette kann syntaktisch oder geometrisch ungültige Ergebnisse akzeptieren beziehungsweise verbergen.** Freies Koordinaten-Sampling, Quantisierungskollisionen, Exposure Bias und fehlende End-to-End-Validierung liegen zwischen sinkendem Loss und einem nutzbaren Mesh.

Das historische MeshtronDomain-Experiment zeigt echtes Overfitting mit ungültiger Ausgabe. Die neueren Quadtron-/HexaRow-Logs zeigen dagegen zunächst, dass die Token-Losses sinken und die Modelle Tokenmuster lernen. Sie beweisen weder, dass das aktuelle Modell nicht lernt, noch dass es valide Geometrie erzeugt. Die wichtigste Lücke ist deshalb nicht ein einzelner fehlender Hyperparameter, sondern ein fehlender messbarer End-to-End-Gültigkeitsbegriff.

## 2. Pfade und Abgrenzung

Im Repository existieren zwei relevante, aber getrennte Trainingspfade:

| Pfad | Einstieg | Modell | Ausgabeformat | Bewertung |
|---|---|---|---|---|
| Quadtron | `train.py` → `trainer.py` | `Quadtron` in `quadtron.py` | flache Quad-Koordinaten-Tokens | Teacher-Forcing-Loss, Tokenmetriken |
| HexaRow/Pfad 1 | `train_hexarow_full.py` | `GPTCond` | 3D-HexaRow-Tokens | gewichtete CE, Token-Accuracy |

`Polytron` und die alten `MeshtronDomain`-Komponenten sind architektonisch nicht mit dem aktiven Quadtron-Pfad gleichzusetzen. MeshtronDomain wird hier nur als historischer empirischer Vergleich verwendet.

## 3. Was das Training tatsächlich optimiert

### Quadtron

`trainer.py` baut Token- und Punktwolken-Loader, trainiert mit AdamW und Warmup/Cosine-Schedule und verwendet standardmäßig `TeacherForcingObjective`. In `objectives.py` wird die Cross-Entropy mit `reduction='sum'` über Nicht-PAD-Tokens berechnet und durch die Tokenanzahl normalisiert. Das entspricht der Optimierung von

> P(nächstes Token | bisherige Ground-Truth-Tokens, Punktwolke, Face Count).

Es ist kein Loss-Term vorhanden, der eine vollständige freie Sequenz detokenisiert oder eine Mesh-Eigenschaft bewertet. `Policy.sample()` wird für autoregressives Sampling beziehungsweise einen späteren RL-Pfad bereitgestellt, ist aber nicht Teil des normalen Teacher-Forcing-Laufs.

### Daten und Konditionierung

`dataset.py` erzeugt die Zielsequenz einmal, sampelt die Punktwolke aber dynamisch in `__getitem__`. Damit kann dieselbe Zielsequenz bei jedem Zugriff eine andere Konditionierung erhalten. Das ist als Augmentation vertretbar, erschwert aber die Diagnose eines reinen Overfit-Experiments. Die Sequenz wird außerdem auf die konfigurierte Länge gepadded beziehungsweise abgeschnitten; die Padding-Maske wird im Dataset angelegt, aber die Teacher-Forcing-Auswertung nutzt primär die Tokenmaske aus dem Objective.

`tokenizer_v2.py` quantisiert Koordinaten mit den aktuellen Bounds des Tokenizers. Diese Bounds werden pro Tokenisierung überschrieben und nicht zusammen mit jedem `MeshData`-Sample gespeichert. Für eine verlässliche Detokenisierung müssen deshalb die zum Sample gehörenden Bounds eindeutig bis zur Inferenz durchgereicht werden.

## 4. Evidenz für Overfitting und Lernfortschritt

### 4.1 Historischer MeshtronDomain-Baseline

`docs/ho_quad_transformer/01_current_model_and_diagnosis.md` dokumentiert den alten Lauf `runs_domain/e23bf276`:

| Kennzahl | Training | Validation |
|---|---:|---:|
| Bits/Token | 1.47 | 4.15 |
| Perplexity | 2.8 | 17.8 |
| Daten | ca. 80/20 Meshes | ca. 20 Meshes |

Die Validation war ab ungefähr Epoche 5 nahezu flach; der beste gespeicherte Lauf lag erst bei Epoche 99. Die Inferenzbilder waren leer, weil die generierte Sequenz nicht in eine gültige Blockstruktur rekonstruiert werden konnte. `deprecated/inference_domain.py` fängt den Rekonstruktionsfehler breit ab und setzt `gen_nodes`/`gen_faces` auf `None`. Damit wird ein Fehler nicht als strukturierte Metrik sichtbar.

**Bewertung:** Overfitting und ungültige Ausgabe sind für diesen historischen Pfad gut belegt.

### 4.2 Quadtron-/HexaRow-Logs

Die Logs `quadtron_4layer_train.log` und `quadtron_8layer_train.log` zeigen jeweils 672 Trainings- und 97 Validierungssamples mit Vokabular 3078:

| Lauf | Train-Loss | Val-Loss | Train-Acc. | Val-Acc. |
|---|---:|---:|---:|---:|
| 4 Layer, Epoche 0 → 49 | 65.45 → 2.38 | 21.29 → 2.51 | 0.002 → 0.429 | 0.014 → 0.411 |
| 8 Layer, Epoche 0 → 49 | 57.41 → 2.10 | 17.88 → 2.44 | 0.005 → 0.471 | 0.034 → 0.433 |

Die Kurven sinken und plateauieren. Am Ende existiert ein Train/Val-Abstand, aber kein Beleg für das historische 3x-Overfitting-Niveau. Die Logs stammen aus dem HexaRow-ähnlichen 3D-Pfad und dürfen nicht direkt als Messung des `train.py`-Quadtron-Pfads interpretiert werden.

Für den 3090-HexaRow-Lauf dokumentiert `HANDOFF_PFAD1.md` 1298 Train- und 68 Val-Samples, Validation-BPT 5.799 und Token-Accuracy 0.022. Der Generationsbefund lautet: die ersten zwei Rows sind korrekt, danach driftet die Row-Länge; nach dem Trim bleiben nur zwei Blocks. Das ist ein direkter Hinweis auf fehlendes Langzeitlernen der Row-Grammatik, nicht nur auf eine schlechte Loss-Kurve.

### 4.3 Was die Kurven nicht sagen

Token-Accuracy zählt Koordinaten- und Spezialtokens gleich. Eine steigende Accuracy kann daher häufige Quantisierungsbins wiedergeben, ohne dass die Modellsequenz eine konsistente Topologie bildet. Es gibt aktuell keine veröffentlichte Kurve für:

- Anteil vollständig detokenisierbarer Sequenzen,
- erwartete und tatsächliche Face-/Blockanzahl,
- eindeutige Vertex-Indizes pro Face/Block,
- Indexbereich und degenerierte Elemente,
- Kanten-/Face-Valenz und Mannigfaltigkeit,
- positive beziehungsweise konsistente Elementvolumina.

## 5. Warum die Ausgabe ungültig werden kann

### 5.1 Quadtron: freie Koordinaten statt Struktur

`tokenizer_v2.py` kodiert Koordinaten in einer flachen Sequenz. Bei der 2D-Row-Kompression werden implizite gemeinsame Kanten aus vorherigen Faces rekonstruiert; bei 3D werden analoge 12- beziehungsweise 6-Token-Gruppen verwendet. Das Format reduziert zwar die Sequenz, garantiert aber nicht, dass ein frei erzeugtes Token an der richtigen Zeilen-, Face- oder Kantenposition erscheint.

`detokenize()` schneidet unvollständige Koordinatengruppen ab und führt quantisierte Vertices über `unique_vertices_hash()` zusammen. Das macht die Ausgabe technisch decodierbar, ist aber keine Topologieprüfung. Derselbe Quantisierungswert kann unterschiedliche Originalvertices zusammenführen; anschließend können Faces wiederholte Indizes oder Nullfläche haben.

### 5.2 HexaRow: lokale Constraint-Decoding-Maske reicht nicht

`generate.py` beschränkt Tokenbereiche mit `slot_mask()` auf r/sin/cos/z-Slots und erlaubt Separator/Stop nur an vermuteten Row-Grenzen. Das verhindert viele lokale Syntaxfehler, erzwingt aber keine global korrekte Zahl und Länge der Rows. Wenn ein Special-Token oder ein Separator unerwartet kommt, kann der Slotzähler desynchronisieren. `detokenize_safe()` trimmt anschließend unvollständige oder zu kurze Rows. Diese Fehlerbehandlung ist nützlich für Diagnose, kann aber eine teilweise gültige Sequenz in eine scheinbar leere oder verkürzte Mesh-Ausgabe verwandeln.

#### Conditioning, Slot-Embedding und Special-Tokens

Im aktiven HexaRow-Training wird die konditionierende skalare Anzahl in `train_hexarow_full.py` intern noch `fc` beziehungsweise `face_count` genannt. Der tatsächlich eingesetzte Wert ist jedoch `it["blocks"]` und damit die **Anzahl der Hexa-Blöcke**, nicht die Anzahl der Faces eines einzelnen Blocks. Dieser Wert wird über `fc_emb` eingebettet und zusammen mit der Punktwolken-Konditionierung in den FiLM-Vektor des GPTCond-Modells aufgenommen. In `generate.py` wird derselbe Wert aus `faces.shape[1]` gelesen. Die Benennung ist daher irreführend, die Semantik ist bereits Blockcount-Conditioning.

Dieses Conditioning beeinflusst nur die Logits. Es erzwingt nicht automatisch, dass genau diese Blockanzahl generiert wird. Dafür müsste der Decoder zusätzlich `expected_blocks`, bereits erzeugte Blöcke und verbleibende Blöcke als strukturellen Zustand führen und `STOP` ausschließlich bei exakter Erfüllung zulassen.

Der Tokenizer definiert eigene Token-IDs für `START`, `END`, `SEP`, `SEP2`, `STOP` und `PAD` (`polytron_tokenizer.py`). `SEP` beendet eine Row, `STOP` beendet den Stream. Beide besitzen ein eigenes normales Token-Embedding über `self.tok`; sie sind daher nicht mit einem Radius-Token identisch.

Das separate `slot`-Embedding beschreibt nur die Koordinatenposition innerhalb eines Vertex:

```text
0 = r
1 = sin(theta)
2 = cos(theta)
3 = z
```

In `_slot_ids()` erhalten Special-Tokens derzeit zusätzlich Slot `0`. Ein Special-Token wird dadurch aber nicht zum Radius-Token, weil sein normales Token-Embedding eine andere Token-ID besitzt. Die aktuelle Darstellung ist dennoch semantisch unsauber: Slot `0` bedeutet gleichzeitig „Radiusposition“ und „Special-Token“. Ein eigener Slot `4 = special` wäre klarer, löst aber allein nicht den Row-Drift.

Die bestehende Maske erlaubt `SEP` und `STOP` tatsächlich nur nach vollständigen 4er-Koordinatengruppen und nach einer Mindestlänge der Row. Sie erzwingt aber keine exakte Row-Länge, keinen vollständigen Row-Plan und keine exakte Gesamtzahl von Blöcken. Nach der Mindestlänge bleiben weitere Gruppen erlaubt; deshalb ist die Maske eine lokale Slot-/Syntax-Constraint, aber noch keine globale HexaRow-Grammatik.

Die vorhandenen Befunde nennen zusätzlich ein hartes Positionslimit: In `scripts/eval_stop_rate.py` erreichten Testsequenzen die maximale Position ohne STOP und endeten mitten in einer Row. Ein solcher Abbruch kann nicht durch bessere VTK-Ausgabe repariert werden.

### 5.3 Rekonstruktion kann Fehler verstärken

Im deprecated 2D-Pfad führt `reconstruct_domain.py` Polar-Places in kartesische Punkte über, merged Punkte mit einem Schwellwert von `1e-3`, dedupliziert Kanten über Endpunkt und Kurvenmittelpunkt und erzeugt anschließend Coons-/TFI-Flächen. Dieser Ablauf kann nahe Punkte kollabieren und neue ungültige Faces erzeugen, bevor überhaupt ein globaler Mesh-Check stattfindet. `reconstruct_domain()` gibt das Ergebnis des Interpolators direkt zurück.

## 6. Fehlende Validierung als primärer Diagnosefehler

`generate.py:write_vtk()` schreibt die detokenisierten Vertices und Blocks direkt als Legacy-VTK. Es wird vor dem Schreiben nicht geprüft, ob

- alle Indizes im Vertexbereich liegen,
- jeder Block vier beziehungsweise acht verschiedene Eckpunkte besitzt,
- Elementvolumen degeneriert oder invertiert ist,
- Faces mehrfach beziehungsweise nicht-mannigfaltig vorkommen,
- die erwartete Face-/Blockanzahl erreicht wurde.

Ein Teil dieser Prüfungen existiert bereits in `scripts/validate_3d_dataset.py`, wird aber nicht auf generierte Meshes angewendet. Daher kann eine erfolgreiche Datei-Erzeugung fälschlich als erfolgreiche Inferenz erscheinen.

Zusätzliche Datenbefunde zeigen, dass die Repräsentation selbst empfindlich ist: `reports/dataset_validation_3d.md` nennt 28 invertierte und 7 kollabierte Blocks in 35 von 468 geprüften Samples. Diese Datenprobleme sind nicht automatisch der Grund für jeden Generationsfehler, müssen aber vor einem Lernurteil vom Modellfehler getrennt werden.

## 7. Priorisierte Ursachenbewertung

| Priorität | Befund | Status | Begründung |
|---|---|---|---|
| P0 | Kein End-to-End-Mesh-Validierungssignal | bestätigt | Training und Inferenz können nicht unterscheiden, ob sinkender Token-Loss ein gültiges Mesh erzeugt. |
| P0 | Freie Sequenz kann globale Row-/Face-Grammatik verletzen | bestätigt | dokumentierter Row-Drift; Constraints sind lokal und nachgelagerter Trim ist verlustbehaftet. |
| P1 | Historischer Daten-/Parameter-Mismatch | bestätigt für MeshtronDomain | ca. 80 Trainingsmeshes gegenüber ca. 5M Parametern, großer Train/Val-Abstand. Für aktuelle Pfade nicht 1:1 übertragen. |
| P1 | Quantisierung und Bounds-Verwaltung | plausibel, teilweise belegt | Kollisionsbefunde existieren; Bounds gehören aktuell nicht explizit zum Dataset-Sample. |
| P1 | Exposure Bias | plausibel | Training sieht Ground-Truth-Präfixe, freie Inferenz verarbeitet eigene Fehler weiter. |
| P2 | Conditioning-Kompression | plausibel | Punktwolke wird auf wenige Latents beziehungsweise einen Konditionierungsvektor reduziert; kein isolierter Kausalbeweis. |
| P2 | Datensplit und Datenqualität | bestätigt als Risiko | kleine bzw. geometrisch verwandte Splits und bereits degenerierte Samples erschweren Generalisierung. |

Es wäre nicht korrekt, aus den Logs allein einen einzelnen Architekturfehler als Root Cause zu behaupten. Sicher ist die Kombination aus lokalem Token-Loss, freier Struktur und fehlendem Validierungs-Gate.

## 8. Empfohlene Reihenfolge für die nächste Untersuchung

### Stufe 1: Messbarkeit herstellen

1. Einen gemeinsamen `validate_generated_mesh()`-Schritt nach Detokenisierung und vor VTK/Plotting einführen.
2. Pro Sample mindestens `decode_ok`, `expected_count_ok`, `index_range_ok`, `unique_vertices_ok`, `nondegenerate_ok`, `manifold_ok` und `orientation/volume_ok` loggen.
3. Diese Metriken auf drei Referenzklassen ausführen: Ground-Truth-Roundtrip, Teacher-Forced-Argmax und freie Generierung.

Damit wird getrennt, ob der Fehler in Tokenizer/Quantisierung, im Decoder oder in der freien Modellsequenz liegt.

### Stufe 2: Überfit-Experiment sauber isolieren

1. Einen sehr kleinen, bereinigten Datensatz verwenden und prüfen, ob **Roundtrip**, **Teacher-Forced-Argmax** und **freie Generierung** jeweils vollständig valide werden.
2. Punktwolken für den Test deterministisch machen und Bounds pro Sample speichern.
3. Validation nicht mit `drop_last=True` betreiben: Ist der Validation-Split kleiner als die Batchgröße, wird aktuell der gesamte Validation-Loader verworfen.
4. Erst danach Modellgröße, Dropout und Quantisierungsauflösung vergleichen.

### Stufe 3: Struktur robust machen

Erst wenn die Stufen 1 und 2 reproduzierbar messen, sind constrained decoding, ein strukturbezogener Loss oder RL-Rewards aussagekräftig. Eine weitere Loss-Optimierung ohne Mesh-Metrik würde sonst nur die bestehende Beobachtung präziser machen, dass Tokenwahrscheinlichkeiten sinken.

## 9. Validierungsstatus des Repositorys

Die vorhandenen generischen Dateien `validation.py`, `test.py` und `testing.py` sind laut `README.md` syntaktisch beziehungsweise durch einen Plan-B-Merge beschädigt. Sie sind daher keine belastbare Validierungsoberfläche für diesen Report. Die zentrale Analyse wurde statisch anhand von Trainings-/Inferenzpfad, Dokumentation, Logs und vorhandenen Dataset-Checks durchgeführt. Die neueren Änderungen betreffen ausschließlich diese Report-Datei; bestehender untracked Datenbestand und `graphify-out/` wurden nicht als Report-Artefakte benötigt.

## 10. Schlussfolgerung

Die Aussage „der Transformer lernt nicht“ ist zu grob. Für den historischen MeshtronDomain-Pfad stimmt sie praktisch im Sinne von Generalisierung und gültiger Ausgabe: dort ist Overfitting klar gemessen und die Rekonstruktion schlägt fehl. Für den aktuellen Quadtron-/HexaRow-Pfad zeigen die Kurven durchaus Tokenlernen. Das eigentliche Produktziel, ein gültiges Mesh aus freier Generierung, wird aber weder optimiert noch gemessen und ist durch mehrere nachgelagerte Mechanismen nicht garantiert.

Der nächste sinnvolle Schritt ist deshalb ein reproduzierbares Validierungs-Gate, nicht zunächst ein größerer Transformer oder ein weiterer Loss-Sweep.
