#!/usr/bin/env python3
"""test_rewards_hexarow.py — Unit-Tests fuer rewards_hexarow (P2-1).

Prueft auf echten Familien-Items (cart):
  (a) Monotonie: reward(GT) > reward(GT mit vertauschten Ecken eines Blocks) >
      reward(Garbage) — strikt, letzteres exakt 0.
  (b) Chamfer GT-vs-GT: self-Chamfer exakt 0, r_conform(GT) hoch.
  (c) Endlichkeit aller Terme auf 20 zufaelligen Items.

Plain asserts, exit 0/1 wie scripts/test_slot_parity.py.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch

from hexa_row_tokenizer import HexaRowTokenizer
from rewards_hexarow import (HexaRowRewardConfig, chamfer_symmetric,
                             make_hexarow_reward)

CKPT = os.path.join(ROOT, "data", "hexarow_sft_cart_ep584.pt")
TOKENS = os.path.join(ROOT, "data", "hexarow_tokens_family_cart.pt")


def _expect(failures: list[str], cond: bool, msg: str) -> None:
    try:
        assert cond, msg
    except AssertionError as e:
        failures.append(str(e))


def _shuffle_one_block(tokens: list[int], npt: int, sep: int) -> list[int]:
    """Vertauscht die ersten beiden Ecken (npt-Token-Gruppen) des ersten
    Fortsetzungsblocks nach dem ersten SEP -> valide Struktur, andere Zelle."""
    t = list(tokens)
    seps = [k for k, x in enumerate(t) if x == sep]
    if not seps:
        return t
    s = seps[0] + 1
    end = s + 8 * npt
    if end > len(t):
        return t
    groups = [t[s + k * npt:s + (k + 1) * npt] for k in range(8)]
    groups[0], groups[1] = groups[1], groups[0]
    return t[:s] + [x for g in groups for x in g] + t[end:]


def main() -> int:
    failures: list[str] = []
    ck = torch.load(CKPT, weights_only=False)
    rb = tuple(float(v) for v in ck["r_bounds"])
    zb = tuple(float(v) for v in ck["z_bounds"])
    tok = HexaRowTokenizer(r_bounds=rb, z_bounds=zb)
    core = tok.core
    npt = int(ck.get("npt", 3))
    ds = torch.load(TOKENS, weights_only=False)
    reward = make_hexarow_reward(tok, HexaRowRewardConfig(), coords="cart")

    # --- (a) strikte Monotonie auf 3 echten Items ---------------------------
    picked = 0
    for it in ds["train"][:80]:
        gt_tokens = it["tokens"].tolist()
        r_gt = reward(gt_tokens, it)
        if not r_gt.valid:
            continue
        r_bad = reward(_shuffle_one_block(gt_tokens, npt, core.sep_token), it)
        if not r_bad.valid:
            continue
        rng = np.random.default_rng(1234)
        garbage = ([core.start_token]
                   + [int(x) for x in rng.integers(0, core.Qr, 300)]
                   + [core.stop_token])
        r_gb = reward(garbage, it)
        _expect(failures, r_gt.total > r_bad.total > r_gb.total,
                f"Monotonie verletzt fuer {it['name']}: "
                f"gt={r_gt.total:.4f} bad={r_bad.total:.4f} garbage={r_gb.total:.4f}")
        _expect(failures, r_gb.total == 0.0 and not r_gb.valid,
                f"Garbage muss invalid/0 sein, war {r_gb}")
        picked += 1
        if picked == 3:
            break
    _expect(failures, picked == 3, f"nur {picked}/3 valide GT-Items gefunden")

    # --- (b) Chamfer GT-vs-GT ----------------------------------------------
    it0 = ds["train"][0]
    gt = it0["surface_points"].numpy()
    _expect(failures, chamfer_symmetric(gt, gt) == 0.0,
            "self-Chamfer der GT-Wolke muss exakt 0 sein")
    r0 = reward(it0["tokens"].tolist(), it0)
    _expect(failures, r0.valid and r0.r_conform > 0.9,
            f"r_conform(GT) zu niedrig: {r0}")

    # --- (c) Endlichkeit auf 20 zufaelligen Items ---------------------------
    rng20 = np.random.default_rng(7)
    idxs = rng20.choice(len(ds["train"]), size=20, replace=False)
    for i in idxs:
        it = ds["train"][int(i)]
        r = reward(it["tokens"].tolist(), it)
        _expect(failures, all(np.isfinite([r.r_valid, r.r_quality, r.r_conform,
                                           r.total])),
                f"nicht-finite Belohnung fuer {it['name']}: {r}")

    if failures:
        print("RED — rewards_hexarow verletzt:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"GREEN — rewards_hexarow: Monotonie ({picked} Items), GT-Chamfer 0, "
          "20 Items endlich.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
