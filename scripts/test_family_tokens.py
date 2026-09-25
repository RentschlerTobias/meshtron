#!/usr/bin/env python3
"""test_family_tokens.py — Gate fuer data/hexarow_tokens_family_{polar,cart}.pt.

(a) Sample-Zahl je Arm == D1-selected (family_selection.json).
(b) bounds vorhanden, coords/npt/family gesetzt.
(c) Train/Val geom-disjunkt.
(d) 3 zufaellige Samples je Arm: Retokenize == gespeicherter Stream,
    emit_consistency -> 0 ungueltige VTK-Relabels, roundtrip block/coord ok.

Plain asserts, exit 0/1 wie scripts/test_slot_parity.py.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np
import torch

from build_family_tokens import load_sample
from diagnose_overfit_blocks import emit_consistency
from meshtron.data.hexa_row_tokenizer import HexaRowTokenizer, roundtrip


def _expect(failures: list[str], cond: bool, msg: str) -> None:
    try:
        assert cond, msg
    except AssertionError as e:
        failures.append(str(e))


def check_arm(arm: str, path: str, n_sel: int, failures: list[str]) -> None:
    ds = torch.load(path, weights_only=False)
    items = ds["train"] + ds["val"]
    _expect(failures, len(items) == n_sel,
            f"[{arm}] {len(items)} Samples != D1-selected {n_sel}")
    _expect(failures, "r_bounds" in ds and "z_bounds" in ds
            and len(ds["r_bounds"]) == 2 and len(ds["z_bounds"]) == 2,
            f"[{arm}] bounds fehlen/fehlerhaft")
    _expect(failures, ds["coords"] == arm and ds["npt"] == (3 if arm == "cart" else 4),
            f"[{arm}] coords/npt falsch: {ds.get('coords')}/{ds.get('npt')}")
    _expect(failures, bool(ds.get("family")), f"[{arm}] family-Key fehlt")

    tg = {it["geom_id"] for it in ds["train"]}
    vg = {it["geom_id"] for it in ds["val"]}
    _expect(failures, not (tg & vg),
            f"[{arm}] geom_id in train UND val: {sorted(tg & vg)[:5]}")
    _expect(failures, bool(ds["train"]) and bool(ds["val"]),
            f"[{arm}] leerer Split")

    tok = HexaRowTokenizer(r_bounds=tuple(ds["r_bounds"]),
                           z_bounds=tuple(ds["z_bounds"]))
    rng = np.random.default_rng(42)
    pick = rng.choice(len(items), size=3, replace=False)
    samples = []
    for idx in pick:
        it = items[int(idx)]
        sample, _, _ = load_sample(it["dir"])
        samples.append(sample)
        ids = tok.tokenize(sample, coords=arm)
        _expect(failures, ids == it["tokens"].tolist(),
                f"[{arm}] Retokenize != gespeicherter Stream ({it['dir']})")
        rows = emit_consistency(sample["faces"].T.tolist(),
                                sample["vertices_cartesian"].numpy(),
                                sample["edge_index"])
        n_invalid = sum(1 for r in rows for v in r.block_valid if not v)
        _expect(failures, n_invalid == 0,
                f"[{arm}] {n_invalid} ungueltige VTK-Relabels in {it['dir']}")

    ok, maxerr, msgs = roundtrip(samples, tok, granularity="row", coords=arm)
    _expect(failures, ok,
            f"[{arm}] roundtrip rot: maxerr={maxerr:.4f} {msgs[:2]}")
    print(f"[{arm}] {len(items)} samples, {len(pick)} roundtrip ok, "
          f"maxerr={maxerr:.4f}, geom train/val {len(tg)}/{len(vg)}")


def main() -> int:
    failures: list[str] = []
    sel = json.loads(open(os.path.join(ROOT, "data", "family_selection.json")).read())
    n_sel = int(sel["_meta"]["n_selected"])
    for arm in ("polar", "cart"):
        path = os.path.join(ROOT, "data", f"hexarow_tokens_family_{arm}.pt")
        check_arm(arm, path, n_sel, failures)
    if failures:
        print("RED — family tokens Gate verletzt:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"GREEN — family tokens: {n_sel} je Arm, Split disjunkt, "
          "roundtrip + 0 ungueltige Relabels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
