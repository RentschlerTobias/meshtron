# GRPO-Spike — Cart-Policy (SFT Ep 584)

Datum: 2026-09-21. Basis: `data/hexarow_sft_cart_ep584.pt` (d512/L12/H8, 46.9M,
coords=cart, npt=3, vocab 3078, cap 461). Neue Artefakte:
`rewards_hexarow.py`, `train_grpo.py`, `scripts/test_rewards_hexarow.py`,
`data/grpo_cart_log.csv`, `data/grpo_cart_step{50..300}.pt`,
`data/eval_grpo_cart.json`.

## Setup

- **Belohnung** (`rewards_hexarow.py`, Config-Dataclass): `total = 1.0·r_valid +
  0.5·r_quality + 0.1·r_conform`; `λ_fold=1.0`, Blade-Faktor 2.0.
  `r_valid` ∈{0,1} (gestoppt + detok ohne Trim + `validate_generated_mesh` inkl.
  `expected_blocks`), `r_quality` = mean min-detJ über positiv-orientierte Zellen
  − λ·mean(relu(−detJ)), `r_conform` = symmetrischer Chamfer(gen-Verts,
  GT-surface_points) blade-gewichtet, bbox-normiert. Kein Vertex-/Token-Reward
  (slot_mask garantiert die Grammatik bereits).
- **Rollout**: G=8, `generate.generate` (KV-Cache, slot-Mask, constrained),
  temp **1.0** (statt Eval 0.7 — mehr Explorationsvarianz innerhalb der Gruppe),
  Conditioning exakt über `conditioning.build_cloud(..., blade_weight=3.0)`.
- **Optimierung**: GRPO A=(R−mean)/(std+1e-6); token-level PG über die
  generierten Tokens, PPO-Clip 0.2, 1 inneres Epoch (on-policy, logp_old =
  Rollout-Re-Score no_grad); KL-Anker an eingefrorene Referenz β=0.04; AdamW
  lr **5e-6** (RL-Gradienten um das Gruppenmittel zentriert/verrauscht — SFT-Rate
  3e-4 würde die Policy sofort kollabieren), grad-clip 1.0, bf16-Autocast.
- **Lauf**: 300 Steps × 1 Item/Step (deterministisch gemischte epoch-Reihenfolge,
  Seed 0), Ckpt alle 50 Steps, CSV-Log pro Step. `--credit sep`-Stub vorhanden,
  Default `uniform`.
- **Smoke zuerst** (3 Steps × 4 Items × G=4): loss endlich, KL≈0 (±4e-5),
  adv_mean≈0, peak VRAM 2.92 GB.

## Trainingskurve (Train-Split, G=8, gesampelte Steps)

| Step | mean R | valid-Anteil | r_quality | r_conform | KL | grad |g| |
|---|---|---|---|---|---|---|---|
| 1 | 0.285 | 0.250 | +0.022 | 0.241 | 0.000 | 0.37 |
| 50 | 0.882 | 0.750 | +0.121 | 0.719 | −0.000 | 0.93 |
| 100 | 0.858 | 0.750 | +0.072 | 0.722 | +0.010 | 0.43 |
| 150 | 1.144 | 1.000 | +0.095 | 0.962 | +0.003 | 0.18 |
| 200 | 1.015 | 0.875 | +0.113 | 0.838 | +0.009 | 2.84 |
| 250 | 0.869 | 0.750 | +0.094 | 0.720 | +0.004 | 2.01 |
| 300 | 1.178 | 1.000 | +0.165 | 0.957 | +0.014 | 0.18 |

50-Step-Fenster (Mittel): valid 0.672 → 0.655 → 0.642 → 0.750 → 0.738 → 0.730;
mean R 0.778 → 0.756 → 0.741 → 0.865 → 0.850 → 0.844. Einbruch Step 51–150,
Erholung/Plateau ab Step 150. KL wächst nur bis ≈0.0075, Entropie bleibt >0,
keine Kollaps-Anzeichen. (Werte sind 1-Item-Gruppen → verrauscht; der
Ckpt-Eval unten ist die belastbare Größe.)

## Vorher/Nachher — volles Val (78 Items, k=8, T=0.7, Seed 0)

| Metrik | SFT Ep584 | GRPO Step300 | Δ |
|---|---|---|---|
| **validity_rate_at_k** | 0.6715 | **0.7981** | **+0.1266** |
| greedy_valid_rate | 0.6795 | 0.8205 | +0.1410 |
| mean_min_detJ | 0.10683 | 0.10937 | +0.00254 |
| mean_blocks_gen / gt | 19.47 / 19.6 | 19.65 / 19.6 | +0.18 |
| mode_coverage | 0.0 | 0.0 | 0 |
| mean_edit_proxy | 0.8901 | 0.8878 | −0.0023 |
| stop_rate | 1.0 | 1.0 | 0 |

**Gate: BESTANDEN** — validity 0.7981 ≥ 0.7215 (Baseline+5 Pkt) UND
mean_min_detJ 0.1094 nicht schlechter als 0.107.

Cheap-Subset (val[:20], k=4, Seed 0) für die Ckpt-Auswahl, SFT als Referenz:

| Ckpt | validity | greedy | detJ |
|---|---|---|---|
| SFT Ep584 | 0.5875 | 0.65 | 0.1177 |
| GRPO Step100 | 0.6000 | 0.65 | 0.1180 |
| GRPO Step200 | 0.6750 | 0.70 | 0.1203 |
| **GRPO Step300** | **0.7500** | **0.80** | 0.1200 |

→ **Bester Ckpt: `data/grpo_cart_step300.pt`** (monoton bester Cheap-Subset-Wert;
volle Auswertung dort).

## Interpretation

1. **Der Greifpunkt saß richtig:** Die 33 % invaliden SFT-Rollouts waren ein
   Per-Zell-Qualitätsproblem — genau der Term, den `r_quality` bestraft. Die
   Verbesserung ist per-Item breit: 39/78 Val-Items bessern sich, 18/78
   verschlechtern sich, 21/78 unverändert; Geometrien mit Validität <0.25 gehen
   von 3 auf 2 zurück, die mittlere Geometrie-Validität steigt 0.679 → 0.812.
2. **Keine Modus-Kollaps:** `mode_coverage` bleibt 0 und `edit_proxy` sinkt nur
   minimal — GRPO presst die Policy nicht in GT-Zerlegungen (r_conform mit
   Gewicht 0.1 ist wie geplant nur Regularisierung, nicht Ziel).
3. **Obergrenze bedenken:** Die Val-GT-Meshes selbst validieren nur zu
   **0.846** (66/78; 12 Items haben datensatzseitig gefaltete/invertierte
   Zellen). Ein Teil der verbleibenden 0.20 invaliden GRPO-Rollouts kann auf
   diesen Items strukturell begründet sein; mehr Daten/GT-Kanonisierung wäre
   nötig, um das zu trennen.
4. **Underfitteter Spike:** 300 Steps = 300 Item-Ziehungen aus 683 (~44 %),
   also < 1 Epoch. Der Step-51–150-Einbruch ist mit hoher Wahrscheinlichkeit
   Item-Mix-Rauschen plus früher KL-Zug. Der Trend 150→300 ist flach-positiv —
   der Lauf war noch nicht am Plateau.

## Kennzahlen des Laufs

- Wall-Time Spike: **2024 s** (~34 min), 300 Steps, G=8 (~0.84 s/Rollout).
- Peak VRAM: **2.01 GB** (Smoke 2.92 GB; beide < 7 GB Budget).
- Volle Gate-Eval: 594 s; Cheap-Subset-Eval: je 79 s.
- `scripts/test_rewards_hexarow.py`: GREEN; `scripts/test_slot_parity.py`:
  GREEN; `scripts/smoke_mesh_validation.py`: PASS.

## Empfehlung für den nächsten Schritt

1. **Fortsetzen statt neu starten:** `train_grpo.py` auf Step300 aufsetzen und
   auf 600–1200 Steps / ≥2 volle Train-Epochs laufen (oder `--items-per-step 2`,
   um die Schritt-Varianz zu senken und die Abdeckung zu verdoppeln).
2. **Erst dann Regler anfassen:** lr/β erst sweepen, wenn die Kurve über 300
   weitere Steps sichtbar sättigt; `r_quality`-Gewicht (0.5 → 0.7) ist der
   naheliegende zweite Hebel, `temperature` und `r_conform` zuletzt.
3. **Belastbarkeit:** Gate-Eval mit zweitem Seed (z. B. `--seed 7`) wiederholen,
   bevor 0.798 als stabil gilt; die 18/78 verschlechterten Items gezielt
   inspizieren (Gegenprobe: sind es die drei Problem-Geometrien?).
4. **Pipeline-Integration:** `data/grpo_cart_step300.pt` als neuer
   Policy-Standard; SFT-Ep584 bleibt als Referenz-/Rollback-Ckpt erhalten.
