# Studie: Transformer für 2D higher-order Quad-Block-Strukturen aus Punktwolken

Ausführliche Studie + Pläne + Rating für einen Transformer, der aus einer **Punktwolke** (Rand-Knoten
**oder** trianguliertes Mesh) eine **higher-order Quad-Block-Struktur** erzeugt, die anschließend per
**Transfinite/Coons-Interpolation** zum CFD-Netz um ein Blade wird.

## Wichtigster Befund vorab

Es existiert **bereits** ein passendes Modell (`MeshtronDomain`) und es **wurde trainiert**
(`runs_domain/e23bf276`). Ergebnis: **starkes Overfitting** (train ppl 2.8 vs. val ppl 17.8) und
**invalide Generierung** (Inferenz-Panels leer, Sequenz nicht rekonstruierbar). Der Engpass ist
**nicht die Architektur, sondern Datenmenge + fehlende Gültigkeitsgarantie.** Details:
[01_current_model_and_diagnosis.md](01_current_model_and_diagnosis.md).

## Dokumente

| Datei | Inhalt |
|---|---|
| [01_current_model_and_diagnosis.md](01_current_model_and_diagnosis.md) | Wie `MeshtronDomain` funktioniert + reale Trainingsergebnisse + Diagnose |
| [02_literature_survey.md](02_literature_survey.md) | Survey (Meshtron, PolyGen, QuadGPT, QuadLink, Point2Quad, TopGen, BrepGen, MGNet, DRL-MeshGen, Airfoil-Multiblock) mit Quellen |
| [03_plans_and_feasibility.md](03_plans_and_feasibility.md) | 6 Pläne (A–F) auf unser Problem gemappt + Durchführbarkeit |
| [04_rating_and_roadmap.md](04_rating_and_roadmap.md) | Rating-Matrix, Ranking, Roadmap, Nächstschritte |
| [05_face_block_generator.md](05_face_block_generator.md) | Fokus-Konzept Plan B: **Quad-Faces per Pointer-Modell** aus Punktwolke + FaceCount, aufgesetzt auf realen Code |
| [06_edge_geometry_study.md](06_edge_geometry_study.md) | **Kantengeometrie (Stufe 3):** Vergleich `hermite` / `bezier` k=2 / `cubic_bezier`, Daten-Befund (0 Wendepunkte über 330k Kanten), Empfehlung `cubic_bezier` (4 Tok, median 0.7 %) |
| [07_polygen_walkthrough.md](07_polygen_walkthrough.md) | **PolyGen Schritt-für-Schritt:** Mesh→Tokens→Modell→Tokens→Mesh an einem durchgerechneten Mini-Beispiel (ASCII/Mermaid), inkl. Literaturliste |

## Empfehlung (Kurzfassung)

**Kombiniert, zweiphasig:**

1. **Phase 1 — Plan A (Baseline härten):** Daten ×20–×50 (klassischer Multiblock-Generator +
   Augmentation) + Constrained Decoding + Blade-Prior. Behebt die zwei realen Defekte, geringes Risiko.
2. **Phase 2 — Plan B (PolyGen-artige Zwei-Stufen):** Vertex-Modell → **Pointer-Face-Modell**
   (Gültigkeit per Konstruktion) → **Kontrollpunkt-Kopf** (echtes higher-order). Bestes
   gewichtetes Rating (**4.35**), nutzt vorhandenen Stack.

Optionale Forschungsspur: **Plan D (Layout/Cross-Field, TopGen)** für Generalisierung auf neue
Blocktopologien.

## Rating auf einen Blick

| Plan | Gewichtet | Kern |
|---|---|---|
| **B** PolyGen 2-Stufen + HO | **4.35** | Gültigkeit + higher-order, Repo-Fit |
| **A** Baseline härten | **3.65** | schnell, risikoarm, Datenfundament |
| **D** Layout/Cross-Field | **3.55** | beste Generalisierung, hohes Risiko |
| **E** Unsupervised (MGNet) | 3.35 | umgeht Datenmangel |
| **C** Link-basiert (QuadLink) | 3.30 | kohärenter Edge-Flow |
| **F** VQ-Codebook (MeshGPT) | 2.80 | zu datenhungrig für jetzt |

Kriterien/Gewichte und Herleitung: [04_rating_and_roadmap.md](04_rating_and_roadmap.md).
