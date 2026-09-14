# Handoff: Meshtron Domain-Partition Extension

> **Branch:** `domain_partition` (gepusht auf origin)  
> **Base:** `origin/tokenizer-sorting`  
> **Repo:** `/root/repos/meshtron` (github.tik.uni-stuttgart.de:trentschler/meshtron)  
> **Last Update:** 2026-07-01  
> **Status:** Pipeline vollständig implementiert, bereit für GPU-Training

---

## Projekt-Ziel
Erweitere Meshtron (autoregressiver Transformer für Quad-Mesh-Generierung) um **Domain-Partition-Prediction mit gekrümmten Kanten**.  
Anstatt feine Quad-Meshes (~2000-8000 Faces, gerade Kanten) vorherzusagen, soll das Modell **grobe Domain-Partitionen** (~6-20 Faces, gekrümmte Kanten via Hermite-Splines) generieren. Die feine Quad-Mesh-Rekonstruktion erfolgt durch **Transfinite Interpolation** (Gmsh) als Post-Processing.

---

## Architektur-Übersicht

```
Raw domain_partition mesh (.pt)
    ├─ blocking_nodes (12, 2)
    ├─ blocking_faces (4, 6)
    ├─ edge_to_streamline (dict mit 100-Punkt-Kurven)
    └─ tri_coordinates (point cloud)

↓ domain_extractor.py

Polar coords + Tangenten + edge_index tensor
    └─ domain_data.pt

↓ tokenizer_domain.py

Token sequence (~150-700 tokens, 8 tokens pro Vertex-Platz)
    [r, θsin, θcos, t_norm, α_in sin, α_in cos, α_out sin, α_out cos]

↓ MeshtronDomain (meshtron_domain.py)

Logits → autoregressive sampling

↓ inference_domain.py / Policy.sample()

Generated tokens → detokenize

↓ reconstruct_domain.py

Polar → Cartesian → merge duplicates → Hermite splines
→ blocked_mesh Data → Transfinite_Interpolation → quad mesh
```

---

## Was funktioniert ✅

| Komponente | Datei | Status | Test |
|-----------|-------|--------|------|
| Daten-Extraction | `domain_extractor.py` | ✅ | 100 meshes → `domain_data.pt` |
| Tokenizer | `tokenizer_domain.py` | ✅ | 3 Strategies, 3 Embedding-Modes, sincos-encoding |
| Dataset | `dataset_domain.py` | ✅ | Modern MeshData-API, point-cloud sampling |
| Embedding | `domain_embedding.py` | ✅ | 3 Modi (split/shared/separate) |
| Modell | `meshtron_domain.py` | ✅ | Forward pass OK, kompatibel mit HourglassTransformer |
| Trainer | `domain_trainer.py` | ✅ | Modern pattern (bf16, warmup+cosine, JSONL logging, checkpoints) |
| Inference | `inference_domain.py` | ✅ | Autoregressive sampling + Rekonstruktion |
| Rekonstruktion | `reconstruct_domain.py` | ✅ | Hermite-Splines + Transfinite Interpolation |
| Visualisierung | `plot_domain_pipeline.py` | ✅ | 6-Phase Pipeline-Plots |
| Config | `config.py` | ✅ | `DomainTrainingConfig` hinzugefügt |

### Test-Ergebnisse (CPU, 10 Epochen, kleines Modell)
- **Loss:** 3.59 → 0.95 (Train), 2.72 → 1.37 (Val)
- **Generierung:** ~207 Tokens (GT=208) — Sequenzlänge wird gelernt
- **Rekonstruktion:** Round-trip funktioniert, aber Geometrie noch ungenau (nur 10 Epochen)

---

## Was als Nächstes gemacht werden muss 🎯

### 1. GPU-Training (HIGH PRIORITY)
Das ist der Haupt-Blocker. Alles wurde auf CPU getestet.

**Empfohlene Config für erstes Training:**
```python
from config import DomainTrainingConfig
from domain_trainer import DomainTrainer

cfg = DomainTrainingConfig(
    data_path='/root/repos/meshtron/domain_data.pt',
    quantization_r=256,        # höhere Auflösung als Smoke-Test
    quantization_a=128,
    sorting_strategy=0,        # 0=no compression (baseline)
    embedding_mode=0,          # 0=split vocab ranges (baseline)
    d_model=512,
    n_heads=8,
    stage_layers=(4, 8, 12, 16, 20),
    n_latents=512,
    batch_size=16,             # an VRAM anpassen
    num_epochs=100,
    learning_rate=1e-4,
    warmup_steps=1000,
    precision='bf16',
    log_dir='runs_domain',
    save_best=True,
)

trainer = DomainTrainer(cfg)
result = trainer.run()
```

**Start-Befehl:**
```bash
cd /root/repos/meshtron
python -c "
from domain_trainer import DomainTrainer
from config import DomainTrainingConfig
cfg = DomainTrainingConfig(...)  # siehe oben
trainer = DomainTrainer(cfg)
trainer.run()
"
```

### 2. Optuna-Sweep (HIGH PRIORITY)
Vergleiche die 3 Sorting-Strategien und 3 Embedding-Modes.

**Parameter-Raum:**
- `sorting_strategy`: [0, 1, 2]
- `embedding_mode`: [0, 1, 2]
- `quantization_r`: [64, 128, 256, 512]
- `quantization_a`: [32, 64, 128, 256]
- `d_model`: [256, 512]
- `stage_layers`: verschiedene Konfigurationen

**Wichtig:** `sweep.py` und `objectives.py` existieren bereits für das Original. Du kannst entweder:
- a) `sweep_domain.py` erstellen (Domain-Partition-Variante), oder
- b) `objectives.py` erweitern um einen Domain-Partition-Objective

### 3. Geometrische Validierung (MEDIUM PRIORITY)
Momentan wird nur **token-level cross-entropy** optimiert. Es gibt keine geometrische Loss-Funktion.

**Mögliche Metriken:**
- Chamfer-Distanz zwischen GT und generiertem Quad-Mesh
- Edge-Curvature-Error (Vergleich der Hermite-Splines)
- Face-Count-Accuracy
- Vertex-Position-Error (nach duplicate-merge)

**Implementation:** `metrics.py` erweitern oder `reconstruct_domain.py` um Vergleichsfunktionen ergänzen.

### 4. Bessere Hermite-Spline-Qualität (MEDIUM PRIORITY)
Momentan werden Tangenten aus K=3 Streamline-Punkten gefittet. Das ist robust aber möglicherweise zu grob.

**Alternativen:**
- K=5 oder K=10 für glattere Tangenten
- Tangenten direkt aus dem Frame-Field berechnen (statt aus Streamlines)
- Tangenten-Norm nicht als Skalar tokenisieren, sondern als Ratio zur Kantenlänge

### 5. Strategy C vervollständigen (LOW PRIORITY)
`tokenizer_domain.py` Strategy 2 (vertex-first + explicit faces) ist nur als Stub implementiert (`_build_sequence_vertex_first` ruft `_build_sequence_faces` auf). Für einen echten Vertex-first-Mode müsste man:
- Alle Vertices einmal emitieren (nur r, θ)
- Dann `[SEP]`
- Dann Face-Indices (4 Indices pro Face)
- Dann `[SEP]`
- Dann Edge-Tangenten

Das würde die Sequenzlänge reduzieren (keine redundanten Vertex-Duplikate), erfordert aber einen komplexeren Detokenizer.

---

## Wichtige Dateien und ihre Rollen

| Datei | Rolle | Wichtig für nächste Schritte |
|-------|-------|------------------------------|
| `domain_extractor.py` | Preprocessing: raw → polar + tangents | Nur bei Datensatz-Änderungen |
| `tokenizer_domain.py` | Tokenisierung: polar + sincos + 3 strategies | **Sweep-Parameter** |
| `dataset_domain.py` | Dataset: point-cloud sampling + padding | Unverändert lassen |
| `domain_embedding.py` | Embedding-Layer: 3 modes | **Sweep-Parameter** |
| `meshtron_domain.py` | Modell-Definition | Unverändert lassen |
| `domain_trainer.py` | Training-Loop (modern) | **GPU-Training starten** |
| `inference_domain.py` | Generierung + Rekonstruktion | **Nach Training testen** |
| `reconstruct_domain.py` | Hermite-Splines + Transfinite | **Geometrische Metriken hinzufügen** |
| `config.py` | `DomainTrainingConfig` | **Config anpassen für Sweeps** |
| `plot_domain_pipeline.py` | Visualisierung | Optional |
| `train_domain.py` | Einfaches Training (alt) | **Durch `domain_trainer.py` ersetzt** |

---

## Daten-Locations

- **Raw:** `/root/repos/meshtron/checkpoint_mesh_100.pt` (100 Meshes, domain_partition Format)
- **Preprocessed:** `/root/repos/meshtron/domain_data.pt` (Polar + Tangenten)
- **Training-Logs:** `runs_domain/<config-hash>/metrics.jsonl`
- **Checkpoints:** `runs_domain/<config-hash>/best.pt`, `last.pt`

---

## Bekannte Probleme / Bugs

1. **CPU-only getestet:** Alles wurde auf CPU entwickelt. `bf16` funktioniert nur auf GPU. Auf CPU nutze `precision='fp32'`.

2. **Einige Meshes haben >6 Faces:** Das Dataset enthält ~10% Outlier mit 16-20 Faces (statt 6). Das führt zu längeren Sequenzen (~500-700 Tokens). Max_seq_length ist automatisch angepasst.

3. **Gmsh-Reinitialisierung:** `Transfinite_Interpolation.__init__` ruft `gmsh.initialize()` auf, aber wenn man es mehrfach aufruft, kann es zu Problemen kommen. `reconstruct_domain.py` nutzt das Ergebnis aus `__init__` direkt (vermeidet doppelte `generate()`).

4. **Strategy 2 unvollständig:** `_build_sequence_vertex_first` ist ein Stub (ruft Strategy 0 auf).

5. **OpenMesh fehlt:** `half_edge.py` erfordert `openmesh`, das in `/root/venv/` nicht installiert ist. Domain-Partition-Code nutzt pure-Python-Alternative (`lexsort` + centroid grouping).

6. **Precision-Warning:** `torch.cuda.amp.GradScaler` ist deprecated → `torch.amp.GradScaler('cuda', ...)` nutzen (in `domain_trainer.py` Zeile 92).

---

## Design-Entscheidungen (festgelegt)

| Thema | Entscheidung | Begründung |
|-------|-------------|------------|
| Koordinaten | Polar (r, θ) | Rotation-invariant, kompakt |
| Winkel-Encoding | Sincos-Paar (2 Tokens) | Kein Wrap-around-Problem |
| Tangenten-Quelle | K=3 Streamline-Fit | Robust, einfach |
| Embedding-Modi | 3 Modi getestet | A=split ranges (baseline), B=shared, C=separate |
| Validation | Token-Level nur | Geometrisch erst bei Inference |
| Face-Indices | Explicit (Strategy 0/1) | Zuverlässiger als implizit |
| Transfinite divisions | External/fixed (z.B. 5-10) | Nicht Teil der Transformer-Sequenz |

---

## Schnellstart für nächste AI

### 1. Branch auschecken
```bash
git fetch origin
git checkout -b domain_partition origin/domain_partition
```

### 2. Training starten (GPU)
```bash
cd /root/repos/meshtron
python -c "
from domain_trainer import DomainTrainer
from config import DomainTrainingConfig

cfg = DomainTrainingConfig(
    quantization_r=256,
    quantization_a=128,
    sorting_strategy=0,
    embedding_mode=0,
    d_model=512,
    n_heads=8,
    stage_layers=(4, 8, 12, 16, 20),
    n_latents=512,
    batch_size=16,
    num_epochs=100,
    learning_rate=1e-4,
    warmup_steps=1000,
    precision='bf16',
    log_dir='runs_domain',
    save_best=True,
)

trainer = DomainTrainer(cfg)
trainer.run()
"
```

### 3. Inference + Rekonstruktion testen
```bash
python inference_domain.py --run-dir runs_domain/<config-hash> --ckpt best.pt --temperature 1.0
```

### 4. Sweeps starten
Siehe `sweep.py` für das Original-Pattern. Erstelle `sweep_domain.py` analog.

---

## Kontext aus vorheriger Session

- Vorheriger Branch: `main` mit rudimentärem `train_domain.py`
- Neuer Branch: `domain_partition` auf Basis von `origin/tokenizer-sorting`
- Alle Domain-Partition-Dateien wurden modernisiert:
  - `dataset_domain.py` → `DomainMeshData` (API-kompatibel mit `MeshData`)
  - `domain_trainer.py` → moderner Trainer (bf16, warmup+cosine, JSONL logging)
  - `inference_domain.py` → Inference mit Rekonstruktion
- Smoke-Test erfolgreich (2 Epochen CPU, loss sinkt)

---

*End of Handoff*
