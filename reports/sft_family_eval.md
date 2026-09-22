# SFT Family Eval — Cart-Arm, Checkpoint Ep 584

Datum: 2026-09-21. Checkpoint: `data/hexarow_sft_cart_ep584.pt` (aus dem bei Ep 584/3000
gestoppten Nightly konvertiert; train loss 0.015-0.025, tok-acc 0.992-0.996, VAL-Plateau
loss ~9.7-10.5 / tok-acc ~0.32). Eval: `scripts/eval_family.py --k 8` über die 78
Val-Items (36 Geometrien, geometrie-disjunkter Split), je Item 8 stochastische Rollouts
(T=0.7, Multinomial) + 1 Greedy. Runtime 610 s.

## Aggregate (624 Rollouts)

| Metrik | Wert | Bedeutung |
|---|---|---|
| **validity_rate_at_k** | **0.6715** | 67 % aller Rollouts enden sauber (STOP) UND validieren als Mesh → **Gate ≥0.6 BESTANDEN** |
| stop_rate | 1.0 | Generierung terminiert immer (slot_mask + STOP-nach-SEP-Fix) |
| mesh_valid_rate | 0.6715 | Alle gestoppten Sequenzen sind valide Meshes |
| greedy_valid_rate | 0.6795 | Greedy ≈ Stochastik-Baseline |
| mode_coverage | **0.0** | Kein einziger Rollout reproduziert die exakte AlgoHex-Zerlegung des Items |
| mean_blocks_gen / gt | 19.47 / 19.6 | Blockzahl trifft im Mittel fast exakt |
| mean_min_detJ | 0.107 | Generierte Zellen durchweg positiv, gut konditioniert |
| mean_edit_proxy | 0.89 | Tokenstruktur weicht stark von GT ab (~89 % Differenz) |
| tokens_per_s | 403 | |

## Interpretation

1. **Architektur-Beweis auf Familienebene:** Das Modell erzeugt aus der Conditioning-Punktwolke
   unbekannter Geometrien frei laufend valide Hex-Blockstrukturen mit passender Blockzahl —
   ohne eine einzige Grammatikverletzung (stop_rate 1.0, keine ungültigen Relabelings).
2. **Kein GT-Mimicry:** Das Modell lernt NICHT die spezifischen AlgoHex-Zerlegungen nachzubauen
   (mode_coverage 0, edit 0.89), sondern eigene valide Dekompositionen. Konsistent mit dem
   VAL-Plateau: Teacher-Forcing-Acc 0.32 auf Val spiegelt die Multimodalität (1-4 Zielstrukturen
   je Geometrie) plus begrenzte Epoche/Kapazität wider.
3. **Für die Optimierungsschleife u.U. genau richtig:** Der Anwendungsfall braucht valide
   Blockstrukturen passender Dichte um ein variables Blade — nicht zwangsläufig AlgoHex-Identität.
   Ob "eigene valide Zerlegung" akzeptabel ist, entscheidet der Anwendungs-/Qualitätsbedarf
   (Maschenqualität mit min detJ 0.107 gut; Hinweis: mean_min_detJ mittelt nur über die 67 %
   validen Rollouts, die 33 % invaliden fehlen darin).

## Offene Punkte für P2 (GRPO)

- Belohnungssignal ist da: 67 % valide Rollouts → GRPO hat Gruppen-Signal (nicht 0/1-degenerate).
- Zielkonflikt klären: r_conform (Kongruenz zur GT-Zerlegung) würde mode_coverage drücken;
  r_valid/r_quality alleine optimiert "irgendeine valide Struktur". Empfehlung: r_valid +
  r_quality + r_conform (blade-gewichtet) wie geplant, aber r_conform niedrig gewichten —
  sonst bestraft GRPO genau die Zerlegungsvielfalt, die die 0.67 validiert.
- Validity 0.67 → 0.8+ via GRPO (200-500 Steps, G=8, KL β=0.04) ist das nächste messbare Ziel.
- Failure-Mode-Aufschlüsselung der 205 invaliden Rollouts: siehe nächste Sektion.

## Failure-Mode-Analyse (Per-Item-Breakdown aus `data/eval_sft_cart.json`)

Klassifikation aller 624 stochastischen Rollouts:

| Befund | Zahl | Konsequenz |
|---|---|---|
| Nicht gestoppt (Cap/Länge) | **0 / 624 (0 %)** | Kein Cap-Problem: kein Val-Item erreicht cap=461 (max. GT-Länge 435) |
| Gestoppt, aber Mesh invalide | **205 / 624 (32,9 %)** | Sämtliche Fehler sind Mesh-Qualität, NICHT Stream-Struktur |
| Items perfekt / teilweise / nie | 22 / 51 / 5 | 78 Val-Items, greedy valid auf 53/78 |

Was die Fehler NICHT sind:
- **Keine Längen-/Cap-Ausreißer:** Zero-Validity-Items haben median GT-Länge 371 (298–432) —
  kürzer als die positiven (median 383). Stop-Timing und Positionslimit sind sauber.
- **Keine Blockzahl-Abweichung:** `block_within1` fast überall erfüllt (gen~21 vs gt 21–22).

Was die Fehler sind:
- **Geometrie-geclustert:** Nur **3 von 36 Val-Geometrien** haben Mean-Validity < 0.25
  (0.12 / 0.16 / 0.19); die `machine_0005`-Familie scheitert über ALLE 4 Grids hinweg.
  5 Geometrien mischen 0- und >0-validity Grids (ein Grid schwer, eines leicht).
- **Qualitätsart, nicht global:** Auch invalide Rollouts zeigen mean detJ ≈ +0.0997 —
  die Invalidität rührt vermutlich aus vereinzelten gefalteten Zellen oder Duplikat-/Nicht-Mannifold-Artefakten,
  nicht aus global deformierter Geometrie. (Rollout-Records enthalten nur Booleans; eine
  Reason-Capture-Wiederaufnahme ist optionales Follow-up.)

Schlussfolgerung für P2: Genau das Fehlerprofil, das ein per-Zell-Qualitäts-Reward
(`r_quality` mit min-detJ-Strafe) direkt adressiert — der Greifpunkt liegt in
Geometrie-Regionen, nicht in der Grammatik. GRPO kann diese 33 % gezielt bestrafen,
ohne die 67 % validen Strukturen zu gefährden.

## Repro

```bash
uv run python scripts/eval_family.py --tokens data/hexarow_tokens_family_cart.pt \
  --ckpt data/hexarow_sft_cart_ep584.pt --k 8 --out-json data/eval_sft_cart.json
```
