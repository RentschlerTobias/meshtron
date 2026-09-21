#!/usr/bin/env python3
"""test_conditioning_parity.py — Regressionslock fuer conditioning.py.

(a) build_cloud (weights=None) == train_hexarow_full.sample_points ueber
    surface_cloud, gleicher rng/Input, sample 3 von polytron_data_3d_smoke.pt
    -> bit-identisch (Paritaet, die ein spaeterer Train/Infer-Swap nicht brechen
    darf).
(b) split_by_geometry: identische geom_id nie auf beiden Seiten.
(c) Blade-Weighting aendert nur Multiplizitaeten: gezogene Punkte bleiben eine
    Teilmenge der Quellpunkte, Blade-Anteil steigt.

Plain asserts, exit 0/1 wie scripts/test_slot_parity.py.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np
import torch

from build_hexarow_tokens import bounds_from
from conditioning import _normalize, build_cloud, split_by_geometry, surface_cloud
from train_hexarow_full import sample_points


def _expect(failures: list[str], cond: bool, msg: str) -> None:
    try:
        assert cond, msg
    except AssertionError as e:
        failures.append(str(e))


def _unique_rows(a: np.ndarray) -> set:
    return set(map(tuple, np.asarray(a, dtype=np.float64).tolist()))


def main() -> int:
    failures: list[str] = []
    smoke = torch.load(os.path.join(ROOT, "data", "polytron_data_3d_smoke.pt"),
                       weights_only=False)
    raw = smoke[3]
    rb, zb = bounds_from(smoke)
    n = 500

    # --- (a) ungewichtete Paritaet, bit-identisch ---------------------------
    ref = sample_points(surface_cloud(raw), n, rb, zb, np.random.default_rng(7))
    got, mask = build_cloud(raw, n, rb, zb, np.random.default_rng(7))
    _expect(failures, ref.shape == (n, 4),
            f"sample_points Form {ref.shape} != ({n}, 4)")
    _expect(failures, got.shape == (n, 4),
            f"build_cloud Form {got.shape} != ({n}, 4)")
    _expect(failures, mask is None,
            "build_cloud(unweighted) muss is_blade=None liefern")
    _expect(failures, np.array_equal(ref, got),
            "build_cloud(unweighted) != sample_points(surface_cloud(...)): "
            f"maxdiff={np.abs(ref - got).max():.3e}")

    # --- (b) Geometrie-disjunkter Split -------------------------------------
    items = [{"geom_id": f"g{i % 7}", "name": f"s{i}"} for i in range(30)]
    tr, va = split_by_geometry(items, val_frac=0.1, seed=0)
    tg = {it["geom_id"] for it in tr}
    vg = {it["geom_id"] for it in va}
    _expect(failures, bool(tr) and bool(va), "Split liefert leere Seite")
    _expect(failures, not (tg & vg),
            f"geom_id auf beiden Seiten: {sorted(tg & vg)}")
    _expect(failures, len(tr) + len(va) == len(items),
            "Split verliert/dupliziert Items")

    # --- (c) Blade-Weighting: nur Multiplizitaeten --------------------------
    rng = np.random.default_rng(11)
    pts = rng.normal(size=(400, 3))
    pts[:, 0] = np.abs(pts[:, 0]) + 0.1  # r>0
    is_blade = np.zeros(400, dtype=bool)
    is_blade[:40] = True
    sample = {"surface_points": torch.tensor(pts, dtype=torch.float64),
              "is_blade": is_blade}
    rb2, zb2 = (-2.0, 2.0), (-2.0, 2.0)
    w_got, w_mask = build_cloud(sample, 2000, rb2, zb2,
                                np.random.default_rng(3), blade_weight=3.0)
    _expect(failures, w_mask is not None and w_mask.dtype == bool,
            "Weighting liefert keine is_blade-Maske")
    src = _unique_rows(_normalize(surface_cloud(sample), rb2, zb2))
    _expect(failures, _unique_rows(w_got) <= src,
            "Weighting erzeugt Punkte ausserhalb der Quellmenge "
            "(Multiplizitaet statt Auswahl verletzt)")
    blade_frac = float(np.asarray(w_mask, dtype=float).mean())
    _expect(failures, blade_frac > 0.1,
            f"Blade-Oversample wirkungslos: Anteil {blade_frac:.3f} <= Basis 0.1")

    # --- (a) gelabeltes Sample: Parity + identischer RNG-Verbrauch ----------
    # Dokumentierter Parity-Pfad ist blade_weight=None: build_cloud(sample,
    # ..., None) == sample_points(surface_cloud(sample), ...) bit-identisch und
    # mit identischem RNG-Verbrauch. blade_weight=1.0 laeuft ueber den
    # gewichteten Zweig (rng.choice(..., replace=True, p=...)) und ist daher
    # NICHT bit-identisch; fuer 1.0 wird Neutralitaet (uniforme Auswahl) geprueft.
    syn_rng = np.random.default_rng(21)
    syn_pts = np.abs(syn_rng.normal(size=(300, 3))) + 0.05
    syn_blade = np.zeros(300, dtype=bool)
    syn_blade[:30] = True
    syn = {"surface_points": torch.tensor(syn_pts, dtype=torch.float64),
           "is_blade": syn_blade}
    rb_s, zb_s = (-1.0, 3.0), (-1.0, 1.0)
    n_s = 1000
    rng_ref, rng_cmp = np.random.default_rng(7), np.random.default_rng(7)
    ref_syn = sample_points(surface_cloud(syn), n_s, rb_s, zb_s, rng_ref)
    got_none, mask_none = build_cloud(syn, n_s, rb_s, zb_s, rng_cmp,
                                      blade_weight=None)
    _expect(failures, np.array_equal(ref_syn, got_none),
            "build_cloud(None) != sample_points(surface_cloud(...)) auf gelabeltem "
            f"Sample: maxdiff={np.abs(ref_syn - got_none).max():.3e}")
    _expect(failures, mask_none is None,
            "build_cloud(None) muss is_blade=None liefern")
    _expect(failures, rng_ref.bit_generator.state == rng_cmp.bit_generator.state,
            "build_cloud(None) verbraucht den RNG anders als sample_points")
    src_syn = _unique_rows(_normalize(surface_cloud(syn), rb_s, zb_s))
    got_w1, mask_w1 = build_cloud(syn, n_s, rb_s, zb_s, np.random.default_rng(7),
                                  blade_weight=1.0)
    _expect(failures, _unique_rows(got_w1) <= src_syn,
            "blade_weight=1.0 erzeugt Punkte ausserhalb der Quellmenge")
    _expect(failures, mask_w1 is not None and mask_w1.dtype == bool,
            "blade_weight=1.0 muss die is_blade-Maske der Ziehung liefern")
    f1_syn = float(np.asarray(mask_w1, dtype=float).mean())
    base_syn = float(syn_blade.mean())
    _expect(failures, abs(f1_syn - base_syn) < 0.03,
            f"blade_weight=1.0 nicht uniform: Anteil {f1_syn:.4f} vs Basis {base_syn:.4f}")

    # --- (b) blade_weight=3.0: ~3x haeufiger in der Auswahl als unter 1.0 ----
    n_src = 4000
    src_blade = np.zeros(n_src, dtype=bool)
    src_blade[:40] = True                      # Basis-Anteil 1%
    src_pts = np.random.default_rng(23).random((n_src, 3))
    samp_b = {"surface_points": torch.tensor(src_pts, dtype=torch.float64),
              "is_blade": src_blade}
    rb_b, zb_b = (0.0, 1.0), (0.0, 1.0)
    n_draw = 40000
    _, m1 = build_cloud(samp_b, n_draw, rb_b, zb_b, np.random.default_rng(101),
                        blade_weight=1.0)
    _, m3 = build_cloud(samp_b, n_draw, rb_b, zb_b, np.random.default_rng(101),
                        blade_weight=3.0)
    fb1 = float(np.asarray(m1, dtype=float).mean())
    fb3 = float(np.asarray(m3, dtype=float).mean())
    base_b = 0.01
    exp3 = 3 * base_b / (1 + 2 * base_b)
    _expect(failures, abs(fb1 - base_b) < 0.005,
            f"Blade-Anteil unter 1.0 {fb1:.4f} != Basis {base_b:.4f}")
    _expect(failures, abs(fb3 - exp3) < 0.01,
            f"Blade-Anteil unter 3.0 {fb3:.4f} != Erwartung {exp3:.4f} (3b/(1+2b))")
    ratio = fb3 / max(1e-9, fb1)
    _expect(failures, 2.6 <= ratio <= 3.4,
            f"Blade-Oversample-Ratio {ratio:.2f} nicht ~3x")

    if failures:
        print("RED — conditioning-Paritaet verletzt:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("GREEN — conditioning: Paritaet bit-identisch, Split disjunkt, "
          "Weighting multiplizitaets-treu.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
