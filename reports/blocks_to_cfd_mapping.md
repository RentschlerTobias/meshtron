# Generated Blocks → CFD: Geometrie-Mapping + TFI-Bruecke

Stand 2026-09-21. Neue Dateien: `block_mapping.py`, `tfi_bridge.py`,
`scripts/map_generated_blocks.py`, `scripts/test_block_mapping.py`.
Kein bestehender Code geaendert; TFI-Repo nur importiert.

## Pipeline

```
tokens-Item -> conditioning.build_cloud (blade_weight=3.0, Paritaet)
  -> k Rollouts generate.generate()  (k=1 greedy top_k=1, k>1 top_k=0)
  -> detokenize_safe -> (vpt, blk)
  -> Ecken C = vpt[blk] auf sample.npz-Feature-Modell snappen
  -> Score (mean snap dist, min det J) -> argmin unter den gueltigen
  -> Vergleichs-VTK / Roh-VTK / TFI-Refill-VTK / summary.json
```

Item-Aufloesung: `train+val` konkateniert, `item['dir']` -> 
`data/hex3d_algohex/<dir>/sample.npz` (autoritatives Feature-Modell:
`vertices`, `blocks`, `edge_polyline`+offset, `surface_tris`+`surface_tri_label`).

## Design: Dimensionsprioritaet statt Naechster-Punkt

Nutzer-Vorgabe: Ecken duerfen NICHT blind auf die naechste Flaeche projizieren.
`FeatureModel.snap_corners` prueft in dieser Reihenfolge, jeder Treffer beendet:

1. **Feature-Punkt** — naechster GT-Blockeckpunkt, `dist <= tol_v` (Default 0.06):
   LE/TE-Spitzen, Wand-Junction-Vertices.
2. **Feature-Kurve** — naechstes Segment der `edge_polyline`, Segmentprojektion,
   `dist <= tol_e` (Default 0.04): Blade-Profilkanten und sonstige Feature-Kanten.
3. **Flaeche** — naechstes `surface_tris`-Dreieck (`clean_blocks._closest_point_on_tris`,
   Import, exakt dieselbe Metrik wie die Boundary-Behandlung).

Der Knackpunkt ist, dass es kein argmin ueber die drei Distanzen ist: ein Punkt
0.03 neben einer Blade-Kante bleibt auf der Kante, auch wenn das naechste
Flaeschendreieck naeher liegt. Genau das schuetzt LE/TE vor dem Glattbuegeln
durch reine Flaechenprojektion. `feature_id` = vertex- | edge- | Dreiecksindex.
Auf den GT-Ecken ist das Snappen idempotent (dist 0, tier `vertex`).

## v5-Label-Befund (Paritaetsfalle, NICHT gefixt)

`conditioning.py:18-20` kommentiert `BLADE_LABEL = 5` als "5=blade". Die reale
`sample.npz`-Semantik (v5, verifiziert) ist:

| label | Bedeutung |
|---|---|
| 1 | inlet |
| 2 | outlet |
| 3, 4 | periodic |
| **5** | **bl_interface_hub** |
| **6** | **bl_interface_shroud** |
| 7 | ogrid_interface |

Label 5 ist also die Hub-BL-Schnittflaeche, nicht die Blade-Wand. Die
`blade_weight=3.0`-Oversampling in `conditioning.build_cloud` gewichtet damit
tatsaechlich Hub-Interface-Punkte. Das ist ein **bekannter Paritaets-Trap**:
Train und Inferenz nutzen dieselbe (falsch benannte) Konstante, die
Train/Inferenz-Paritaet bleibt also erhalten und `conditioning.py` wird
bewusst NICHT geaendert. Fuer das Snapping ist das irrelevant, weil es direkt
auf `edge_polyline`/`vertices` arbeitet, nicht ueber das Label.

## TFI-Bruecke (`tfi_bridge.refill_cfd`)

Import-only via sys.path auf das externe `hex3d_algohex`-Repo, Aufrufkette wie
dessen `tests/test_refill.py:27-52`:

```
weld(C) -> build_topology -> lat (1x1x1-Lattice, tfi.CORNER-Permutation)
  -> direction_classes -> solve_block_divisions(target_h)
  -> refill_complex -> check_watertight -> export_vtk.write_vtk
```

Modi im Report:
- `conforming` — `direction_classes` koppelt Bloecke ueber gemeinsame Flaechen.
- `independent_fallback` — Kopplung/MILP schlug fehl (nicht-face-sharing
  generierte Struktur): jede Blockachse eigene Klasse, keine Konformitaet
  erzwungen, weld+refill laeuft trotzdem. `fallback_reason` nennt den Grund.

## Chord-Modus-Caveat + Ausbaupfad

v1 traegt nur die 8 Blockecken im Lattice. Die sechs Randflaechen werden als
**gerade Sehnen** resampled, der Innenraum per Gordon-Hall gefuellt
("chord mode"). Krumme Feature-Kanten werden also durch Sehnen approximiert —
akzeptabel fuer v1, dokumentiert.

Konsequenz: die bekannten gefalteten Grobbloecke (GT `blocks` enthalten
`min det J < 0`; fuer idx 687 zwei Stueck) erzeugen nach dem Refill invertierte
Feinzellen. `check_watertight` ist trotzdem `True` (Topologie), die
`scaled_jacobian`-Qualitaet ist es nicht.

Ausbaupfad (curved/coons): `sample.npz` traegt `edge_ctrl [E,2,3]`
(Bezier-Kontrollpunkte) und `edge_polyline` (diskretisierte Kurven). Statt
1x1x1-Kantenlattice die Randkanten mit der Polyline-Diskretisierung seeden
(Lattice mit Zwischenpunkten) bzw. die sechs Randflaechen als Coons-Patches aus
den gekruemmten Randkurven bauen; `refill_block` resampled dann die echten
Kurven statt der Sehnen.

## Nutzung

```bash
uv run python scripts/map_generated_blocks.py \
  --tokens data/hexarow_tokens_family_cart.pt \
  --ckpt data/grpo_cart_step300.pt --idx 687 --k 8
```

Optionen: `--k` (1=greedy), `--temperature 0.7`, `--seed 0`, `--tol-v 0.06`,
`--tol-e 0.04`, `--target-h 0.08`, `--out-dir data/map_<name>/`.
Ausgaben im out-dir: `compare.vtk` (part 1=GT, 2=gesnappt, 3=Punktwolke),
`compare_raw.vtk` (part 2=roh/ungesnappt), `cfd_refill.vtk` (TFI),
`summary.json`. Exit 2 nur, wenn kein Kandidat strukturell gueltig ist.

## E2E-Ergebnisse

Beide Laeufe: `grpo_cart_step300.pt`, k=8 stochastisch, temp 0.7, seed 0,
tol_v 0.06, tol_e 0.04, target_h 0.08. TFI-Modus beide **conforming**,
`check_watertight True`, kein Fallback.

Positivitaets-Gate (`minJ > 0`): in beiden Laeufen 0 Kandidaten. Ursache ist
das Datensatz-Artefakt (gefaltete Grobbloecke im GT selbst). Der CLI faellt
dann auf `structural_valid` zurueck und markiert `positivity_fallback: true`.

### idx 687 — batch__machine_0034_n2000 (12 GT-Bloecke, eval-validity 1.0)

Feature-Modell: 40 GT-Ecken, 168 Feature-Kanten, 19516 Flaeschentris.

| Rollout | Bloecke | tier vertex/edge/surface | mean snap dist | min det J | strukturell |
|---|---|---|---|---|---|
| 0 | 12 | 25 / 50 / 21 | 0.019139 | -0.012819 | ja |
| 1,3,4 | 12 | 45 / 29 / 22 | 0.029781 | -0.009061 | nein (1 Dup-Ecke) |
| 2,5,6,7 | 12 | 55 / 26 / 15 | 0.026324 | -0.000480 | nein (1 Dup-Ecke) |

Gewaehlt: Rollout 0 (mean_d 0.019139, tiers 25/50/21, 12 Bloecke).
TFI: 12656 Zellen, 3918 Randflaechen, watertight True; min scaled Jacobian
-0.5284, 9 Zellen <= 0 (Chord-Modus, s.o.).

### idx 684 — batch__machine_0005_n8000 (21 GT-Bloecke, eval-validity 0.25)

Feature-Modell: 62 GT-Ecken, 274 Feature-Kanten, 19106 Flaeschentris.

| Rollout | Bloecke | tier vertex/edge/surface | mean snap dist | min det J | strukturell |
|---|---|---|---|---|---|
| 0,1,2,3,6 | 22 | 63 / 70 / 43 | 0.025760 | -0.024220 | nein (1 Dup-Ecke) |
| 4 | 22 | 85 / 65 / 26 | 0.027750 | -0.030800 | nein (1 Dup-Ecke) |
| 5 | 22 | 55 / 78 / 43 | 0.020715 | -0.020455 | ja |
| 7 | 22 | 40 / 86 / 50 | 0.023030 | -0.001450 | nein (1 Dup-Ecke) |

Gewaehlt: Rollout 5 (mean_d 0.020715, tiers 55/78/43, 22 Bloecke; GT 21 —
Blockzahl-Mismatch ist erlaubt, das ist der multimodale Punkt).
TFI: 12885 Zellen, 3938 Randflaechen, watertight True; min scaled Jacobian
-0.3179, 50 Zellen <= 0.

### Beobachtung: Snapping kann Ecken verschmelzen

7/8 bzw. 7/8 Rollouts werden nach dem Snappen strukturell ungueltig durch
"1 duplicate vertex coordinate" — zwei generierte Ecken landen auf demselben
Feature-Punkt. Das ist ein echtes Ergebnis (nicht CFD-fertig) und wird
verworfen; nur der strukturell gueltige Kandidat wird weiterverarbeitet.

## Tests

`scripts/test_block_mapping.py` (keine GPU, npz idx 687):
- Idempotenz auf GT-Ecken: dist < 1e-12, tier `vertex`.
- GT-Ecke + 0.03 entlang Flaechennormale -> tier `vertex` (Feature-Punkt-
  Prioritaet, Nutzer-Vorgabe).
- Dieselbe Ecke mit tol_v=0.01 -> tier `edge`; +0.05 -> tier `surface`.
- Blade-Kanteninneres (weit weg von GT-Ecken): +0.03 -> `edge`, +0.05 -> `surface`.
- Score-Ordnung: Versatz strikt schlechter.

Bestehende Gates gruen: `test_slot_parity`, `smoke_mesh_validation`,
`test_rewards_hexarow`, `test_conditioning_parity`.
