# 05 — Face-/Block-Generator (Quad-Topologie aus Punktwolke + FaceCount)

> Fokus-Konzept zu **Plan B** aus [03](03_plans_and_feasibility.md)/[04](04_rating_and_roadmap.md):
> Wie erzeugt ein Transformer **explizite quadrilaterale Blockstrukturen (Faces)**, konditioniert
> auf eine **Punktwolke** (Rand oder trianguliert) und optional **FaceCount**?
> Aufgesetzt auf den **realen Repo-Stand** (`meshtron_domain.py`, `tokenizer_domain.py`,
> `point_encoder.py`, `faceCount_encoder.py`).

## 0. Ausgangslage im Code (was schon da ist)

Die Konditionierung existiert bereits und ist gesund:

- `PerceiverPointEncoder` (`point_encoder.py`): Punktwolke `[B,N,2]` → Fourier-Features →
  `n_latents` Latents `[B,n_latents,d]` via Cross-/Self-Attention.
- `FaceCountEncoder` (`faceCount_encoder.py`): sinusoidale Kodierung der Blockzahl → `[B,d]`.
- `MeshtronDomain.forward` (`meshtron_domain.py:85`) konkateniert beide zu `latent_condition`
  und speist sie per Cross-Attention in den `HourglassTransformer`.

**Der Engpass ist nicht die Konditionierung, sondern die Face-Repräsentation.**

## 1. Kernproblem: Faces sind aktuell *implizit*

Im `DomainTokenizer` ist ein Quad = **4 aufeinanderfolgende Vertex-Plätze** (je 8 Tokens).
`detokenize` (`tokenizer_domain.py:578`) rekonstruiert Faces rein positionell:

```python
n_faces = n_places // 4
faces_as_places.append([fi*4 + i for i in range(4)])
```

Folgen daraus:

1. **Geteilte Blockecken werden dupliziert** neu emittiert (jedes Face bringt seine 4 Ecken
   selbst mit). Ein innerer Gitterknoten mit Valenz 4 taucht bis zu 4× auf — jedes Mal mit
   eigenem Quantisierungsfehler → Ecken „driften auseinander", Blöcke passen nicht bündig.
2. **Keine Konnektivität als lernbares Signal.** Das Modell muss geteilte Kanten *implizit*
   durch Zahlengleichheit treffen. Genau das schlägt bei Generierung fehl → invalide/leere
   Rekonstruktion (die beobachteten leeren Panels).
3. **Sequenz ist unnötig lang** (Ecken ×Valenz), was den O(seq²)-Attention-Druck verschärft.

Row-Kompression (Strategy 1, `_arrange_face`) mildert das nur heuristisch entlang einer Zeile,
löst aber die Grundfrage „welche Places sind *derselbe* Knoten?" nicht.

## 2. Zielarchitektur: Zwei-Stufen (Topologie ⟂ Geometrie)

Trenne **wo** die Blockecken liegen von **welche** Ecken ein Quad bilden. Beide Stufen teilen
sich die Konditionierung (Punktwolke + FaceCount).

```
                 ┌─────────────────────────────────────────────┐
 Punktwolke ──▶  │ PerceiverPointEncoder ─┐                     │
 FaceCount  ──▶  │ FaceCountEncoder ──────┴─▶ latent_condition  │  (reuse, unverändert)
                 └─────────────────────────────────────────────┘
                            │                         │
                            ▼                         ▼
            ┌───────────────────────┐   ┌──────────────────────────────┐
   Stufe 1  │  Vertex-Modell        │   │  Stufe 2: Face-Pointer-Modell │
            │  AR über *eindeutige*  │──▶│  AR über Quads = 4 *Zeiger*   │
            │  Blockecken (r,θ)      │ V │  auf Stufe-1-Vertices         │
            └───────────────────────┘   └──────────────────────────────┘
                            │                         │
                            └──────────┬──────────────┘
                                       ▼
                         Stufe 3: Kanten-Geometrie-Kopf
                    (k Kontrollpunkte / HO-Tangenten pro *gerichteter* Blockkante)
                                       ▼
                       reconstruct_domain_coons  +  blade_inject
```

### Stufe 1 — Vertex-Modell (eindeutige Blockecken)

- Autoregressiv über die **deduplizierte** Menge der Blocking-Knoten, sortiert (lexikografisch
  nach `(θ, r)` oder `(y, x)` — `lexsort` ist schon in `_order_faces` in Verwendung).
- Kodierung wie gehabt: polar `(r, θ_sin, θ_cos)` quantisiert (`quantization_r/a`). Die
  HO-Tangenten wandern in Stufe 3 (sie gehören zu *Kanten*, nicht zu *Ecken*).
- Konditioniert auf `latent_condition`. Ende via `end_token`.
- Ergebnis: eine **Vertexliste** `V = [v_0 … v_{M-1}]` mit stabilen Indizes.

### Stufe 2 — Face-Pointer-Modell (die eigentliche Blockstruktur)

- PolyGen-artiges **Pointer-Network**: pro Quad werden **4 Indizes** in `V` erzeugt, indem der
  Decoder-Zustand gegen die **Vertex-Embeddings aus Stufe 1** dottet (Pointer-Softmax über
  `M+1` Ziele; `+1` = `stop`).
- **Gültigkeit per Konstruktion:** ein Zeiger kann nur auf einen **existierenden** Vertex
  zeigen. Kein „driftender Duplikat-Knoten", keine ungültigen Indizes mehr → der Baseline-Fehler
  ist strukturell weg.
- **FaceCount** steuert hier direkt die Sequenzlänge (Anzahl Quads) — starkes, sauberes
  Konditionierungssignal für Stufe 2. Kann als hartes Stop-Kriterium oder als Soft-Prior dienen.
- Kanonische Eckreihenfolge (CCW, kleinster Index zuerst) macht das Ziel eindeutig — dieselbe
  Logik wie `ccw_ring` in `_arrange_face`.

### Stufe 3 — Higher-order-Kantenkopf (**Half-Edge / Multigraph-korrekt**)

> **Korrektur gegenüber der ersten Fassung.** Der frühere Satz „eine physische Kante existiert
> einmal → einmal emittieren" ist **falsch** für Blade-/Loch-Geometrie. Das Mesh ist ein
> **gerichteter Multigraph**: zwischen zwei Knoten kann es **mehrere verschiedene Kurven** geben.
> An den Daten belegt (`domain_data.pt`): pro Mesh existiert (mind.) ein Paar `(u→v)`/`(v→u)`, dessen
> Streamlines **keine Umkehrung voneinander** sind (max. Kurvenabstand ~0.08 ≙ ~12 % Chord) — genau
> Druck-/Saugseite am Blade, gleiche Endpunkte, **unterschiedliche Splines**. Vgl. `[[meshtron-known-bugs]]`
> Bug 1 (Parallelkanten am NACA-Loch) und Bug 3 (gegenläufige Half-Edge-Tangenten).

- **Geometrie hängt an der gerichteten Half-Edge, nicht am ungerichteten Paar `{u,v}`.** Jedes Face
  ist ein gerichteter Ring `v0→v1→v2→v3→v0`; die 4 Kanten sind Half-Edges. Die HO-Parameter
  (`t_norm, α_in, α_out` aus `edge_tangents[(u,v)]`, **oder** `k` Bézier-Kontrollpunkte, k=2 → S-Kurven,
  behebt Wendepunkt-Limit) werden **pro Half-Edge** kodiert, so wie das Face sie durchläuft.
- **Kein Dedup nach `{u,v}`.** Genau dieses Mergen hat in der alten Pipeline die zwei Blade-Seiten
  zusammengeworfen (Bug 1) bzw. gegenläufige Tangenten gemittelt und Kanten begradigt (Bug 3).
- **Twin-Beziehung explizit statt implizit:**
  - *Innere geteilte Kante:* `(u→v)` und `(v→u)` **sind** Umkehrungen (Twin). Für Bündigkeit die
    Geometrie **einmal** lernen und für den Nachbarblock **umdrehen** (nicht neu/mitteln) — bzw.
    beide Half-Edges emittieren + Konsistenz-Loss.
  - *Blade-/Randkante:* `(u→v)` und `(v→u)` sind **kein** Twin → als **getrennte** Half-Edges mit
    eigener Geometrie behandeln. Die Richtung disambiguiert die beiden Seiten von selbst.
  - Twin-Test wie in `reconstruct_domain` (geometrisch: Endpunkt-Paar + Kurvenmittelpunkt, Chord-
    Toleranz `MERGE_TOL_FRAC`), **nicht** rein topologisch.
- Blade-Kanten werden nach der Rekonstruktion per `blade_inject.py` exakt überschrieben (Blade
  ist bekannt, siehe `[[meshtron-blade-injection]]`).

## 3. Warum das die beobachteten Defekte behebt

| Defekt heute | Ursache | Fix durch Stufen-Modell |
|---|---|---|
| Leere/invalide Gen-Panels | Faces implizit, Indizes müssen „zufällig" matchen | Pointer → nur gültige Indizes |
| Blöcke nicht bündig | Ecken dupliziert, jede mit eigenem Quant-Fehler | Ecke existiert genau 1× |
| Kanten-Lücken (innere Kante) | geteilte Kante 2× unabhängig kodiert | Twin: einmal + umdrehen (Bug 3) |
| Blade-Seiten verschmolzen | Dedup nach `{u,v}` mergt Druck/Saug | Half-Edge nach Richtung, kein `{u,v}`-Dedup (Bug 1) |
| Lange Sequenzen / OOM | Ecken × Valenz | Vertices dedupliziert → kürzer |
| Keine echten S-Kurven | 1 quadr. Bézier | k=2 Kontrollpunkte |

## 4. Reuse aus dem Repo (was bleibt, was neu ist)

**Bleibt unverändert:** `PerceiverPointEncoder`, `FaceCountEncoder`, `HourglassTransformer`,
`reconstruct_domain_coons`, `blade_inject.py`, die Quantisierungs-Helfer im Tokenizer.

**Neu / zu ändern:**
- Tokenizer: eine `tokenize_twostage`-Variante, die (a) Vertices dedupliziert + sortiert,
  (b) Faces als **Index-Tripel/Quadrupel** statt als wiederholte Koordinaten ablegt. Die
  Extraktion (`domain_extractor.py`) liefert `faces [4,n_faces]` schon als Indizes — die
  Deduplikation ist also fast geschenkt.
- Modellkopf: Pointer-Attention-Kopf (Query = Decoder-State, Keys = Stufe-1-Vertex-Embeddings).
- Sampler (`inference_domain.py`/`policy.py`): zweiphasig (erst Vertices bis `end`, dann Faces
  bis `stop`), mit hartem Constrained-Decoding (Pointer-Maske auf gültige Indizes).

## 5. Varianten / offene Design-Entscheidungen

1. **Ein Transformer, zwei Phasen** (Vertices dann Faces in *einer* Sequenz mit `sep_token` —
   `sep_token` existiert bereits) **vs. zwei getrennte Modelle** (PolyGen-Original). Ein-Modell
   spart Parameter und teilt die Konditionierung natürlicher; zwei Modelle sind einfacher zu
   debuggen/trainieren. → **Empfehlung: ein Modell, zwei Phasen** (kleiner Repo-Eingriff).
2. **HO-Repräsentation:** Hermite (bestehend, round-trip erprobt) vs. k=2 Bézier (mehr Ausdruck).
   → Start mit Hermite, Bézier als Upgrade wenn S-Kurven gebraucht werden.
3. **Vertexkoordinaten:** polar (gut für O-/C-Grid ums Zentrum, bestehend) beibehalten.
4. **FaceCount als hartes Stop** vs. Soft-Prior in Stufe 2.

### Stufe 3 — validiert im Prototyp (2026-07-27)

`prototype_twostage.py` implementiert Stufe 3 als Half-Edge-Kantenkopf, multigraph-treu:

- Sequenz erweitert um `sep2` + `geom*4F`. Pro **gerichteter** Face-Seite `p0→p1` (Traversal-
  Reihenfolge) **6 Tokens**: `αstart(sin,cos), tn_start, αend(sin,cos), tn_end`. Jede gerichtete
  Kante erscheint mesh-weit **genau einmal** (Twin liegt im Nachbarblock in Gegenrichtung) →
  **kein `{u,v}`-Dedup**, Blade Druck/Saug bleiben getrennt. Fehlende Richtungen im Datensatz: **0 %**.
- Hermite-Tangenten: `T0 = tn_start·[cosα,sinα]` am Start, `T1 = tn_end·[cosα,sinα]` am Ende
  (Layout `edge_tangents[e]=[α_s, tn_s, α_e, tn_e]` an den Daten verifiziert).
- **Round-trip (300 Meshes)**: Topologie 300/300 exakt; Stufe-3-Kanten vs. `edge_to_streamline`
  **median 1.3 %, mean 2.2 %** rel-Fehler/Chord. Ausreißer bis ~58 % sind **inhärent** (ein
  kubischer Hermite auf einer entarteten Mini-Kante, chord≈0.016; roh sogar 62 %) → Bug-2-Limit,
  nicht Stufe-3. Seq-Len mean 261 (min 208, max 654), vocab 1096.
- Viz: `viz_twostage.py` Frame `09_curved_faces.png` (links gerade Quads ↔ rechts rekonstruierte
  gekrümmte Half-Edges auf GT-Streamline, rote Blade-Parallelkante), `10_edge_tokenize.png`
  (eine Kante → 6 Tokens explizit), `11_compare_methods.png` (Hermite vs Bézier).

**Umschaltbare Repräsentation (`repr_mode`, 300-Mesh-Vergleich):**

| Modus | Tok/Kante | Seq mean | Geom-Fehler median / mean | Kurventyp |
|---|---|---|---|---|
| `hermite` (kubisch, aus Extractor-Tangenten) | 6 | 261 | 1.3 % / 2.2 % | 1 Wendepunkt möglich |
| `bezier` k=2 (quadr., nur α, Kontrollpkt=Tangentenschnitt) | 4 | 199 (−24 %) | 6.0 % / 8.5 % | 0 Wendepunkte |
| **`cubic_bezier` (kubisch, best-fit)** | **4** | **199 (−24 %)** | **0.7 % / 1.2 %** | 1 Wendepunkt möglich |

**`cubic_bezier` gewinnt BEIDE Achsen: 4 Tok UND genauer als hermite.** Grund: 2 freie
Kontrollpunkte (Chord-lokal `s1,h1,s2,h2`, per Least-Squares an die Streamline gefittet) statt
an die Extractor-Tangenten gebunden. `bezier` k=2 ist schwach nicht wegen fehlender S-Kurve,
sondern weil sein Kontrollpunkt fix am Tangentenschnitt sitzt (0 Freiheitsgrade) → schlecht auf
asymmetrischen Bögen.

**Wichtiger Daten-Befund:** über **alle 330.164 Kanten** des 10k-Satzes hat **0** eine
Wendepunkt-Kurve (Streamlines sind reine Einzelbögen). Das Wendepunkt-Limit der Quadratik ist
also **theoretisch und wird von den Daten nie getroffen** — kein Argument gegen `bezier`. Der
worst-case (~58 %) ist bei allen Kubik-Modi identisch = eine entartete Mini-Kante (chord≈0.016),
inhärent für *ein* Polynom pro Kante ([[meshtron-known-bugs]] Bug 2). Für noch tiefer:
Mehr-Segment/Kante. **Empfehlung: `cubic_bezier` als Default** (kürzeste Sequenz + genaueste).

## 6. Rating & Einordnung

Unverändert zu [04](04_rating_and_roadmap.md): **Plan B = 4.35 (Rang 1)**. Mit 10.014 Meshes ist
der Datenbedarf der zweiten Stufe (kleines Topologie-Vokabular, 6–~50 Faces) unkritisch. Empfohlene
Reihenfolge bleibt: **Phase 0** (Baseline auf 10k neu trainieren = ehrliche Messlatte) → **Phase 1**
(Constrained Decoding als Sofort-Validitätsfix) → **Phase 2** (dieses Stufen-Modell als Zielarchitektur).

## 7. Konkrete nächste Schritte (im Repo)

1. `domain_extractor.py`-Output prüfen: liegen `faces` als globale Vertex-Indizes vor (ja) →
   Deduplikations-Mapping `place→unique_vertex` bauen.
2. `tokenizer_domain.py`: `tokenize_twostage(mesh_data)` (Vertices dedupliziert + `sep` +
   Face-Index-Quadrupel) und passendes `detokenize_twostage`. Round-trip-Test wie
   `testing()`.
3. `meshtron_domain.py`: Pointer-Kopf + zweiphasiger `generate`.
4. `policy.py`: Pointer-Maske (nur gültige Vertex-Indizes) für Constrained Decoding.
5. Metriken: **Gen-Rekonstruktionsrate** (Ziel >90 %), Bündigkeit geteilter Kanten (Sagitta-Diff),
   HO-Kantenfehler.



