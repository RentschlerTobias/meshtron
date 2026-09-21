#!/usr/bin/env python3
"""Familien-Auswahl nach Blockzahl-Cap + Gueltigkeitsfilter.

Liest data/dedup_inventory.json (runs mit keep=true = ein Run pro
(geom_id,grid_id)) und setzt einen Cap auf n_blocks am natuerlichen
Verteilungs-Gap: Werte 11..25 dicht, naechster Wert 36 -> Cap 25.
Zusaetzlich fliegen Runs mit degenerierter Block-Konnektivitaet raus
(Hexa-Face mit <4 eindeutigen Vert-Ids = geweldeter Sliver; der
HexaRowTokenizer wuerde sie mit DegenerateBlockError ablehnen).
Geometrien, deren ALLE Runs so entfallen, verlassen die Familie komplett
(non_ideal_all_modes=true) - "Baselines mit viel zu vielen Bloecken
aussortieren".

Output data/family_selection.json, deterministisch (sort_keys, indent 2,
trailing NL).
"""
from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "hex3d_algohex"
IN_JSON = ROOT / "data" / "dedup_inventory.json"
OUT_JSON = ROOT / "data" / "family_selection.json"
CAP = 25  # natuerlicher Gap: dicht bis 25, naechster Wert 36
CAP_RULE = "n_blocks <= 25 (Gap 25->36 im kept-Histogramm)"
_HEX_FACE_Q = ([0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
               [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7])


def load_kept() -> list[dict]:
    inv = json.loads(IN_JSON.read_text())
    return [r for r in inv["runs"] if r["keep"]]


def is_degenerate(rel: str) -> bool:
    """True wenn ein Block eine Hexa-Face mit <4 eindeutigen Vert-Ids hat
    (weld-fusionierter Sliver) oder die npz nicht lesbar ist."""
    p = DATA / rel / "sample.npz"
    try:
        with np.load(p) as d:
            blocks = np.asarray(d["blocks"], dtype=np.int64)
    except Exception:
        return True
    for b in blocks:
        ids = [int(x) for x in b]
        for q in _HEX_FACE_Q:
            if len({ids[i] for i in q}) < 4:
                return True
    return False


def _proc(rel: str):
    return rel, is_degenerate(rel)


def build_geometries(runs: list[dict]) -> list[dict]:
    by_geom: dict[str, list[dict]] = {}
    for r in runs:
        by_geom.setdefault(r["geom_id"] or "", []).append(r)
    geoms = []
    for geom, rs in by_geom.items():
        any_sel = any(r["select"] for r in rs)
        geoms.append({
            "geom_id": geom,
            "machines": sorted({r["machine"] for r in rs}),
            "any_selected": any_sel,
            "non_ideal_all_modes": not any_sel,
        })
    geoms.sort(key=lambda g: g["geom_id"])
    return geoms


def hist(values: list[int]) -> dict:
    u, c = np.unique(np.asarray(values, dtype=np.int64), return_counts=True)
    return {int(k): int(v) for k, v in zip(u, c)}


def main() -> None:
    kept = load_kept()
    degen: dict[str, bool] = {}
    with mp.get_context("fork").Pool(min(24, mp.cpu_count())) as pool:
        for rel, bad in pool.imap(_proc, [r["dir"] for r in kept]):
            degen[rel] = bad
    runs = []
    for r in kept:
        nb = int(r["n_blocks"])
        cap_ok = nb <= CAP
        bad = degen[r["dir"]]
        runs.append({
            "dir": r["dir"], "machine": r["machine"], "geom_id": r["geom_id"],
            "grid_id": r["grid_id"], "n_blocks": nb,
            "degenerate": bad, "select": bool(cap_ok and not bad),
        })
    runs.sort(key=lambda r: r["dir"])
    geoms = build_geometries(runs)

    selected = [r for r in runs if r["select"]]
    cap_dropped = [r for r in runs if r["n_blocks"] > CAP]
    degenerate = [r for r in runs if r["degenerate"]]
    dropped = [r for r in runs if not r["select"]]
    nb = np.asarray([r["n_blocks"] for r in runs], dtype=np.int64)
    meta = {
        "source": "data/dedup_inventory.json",
        "cap_n_blocks": CAP,
        "cap_rule": CAP_RULE,
        "validity_rule": "keine Hexa-Face mit <4 eindeutigen Vert-Ids (kein "
                         "weld-fusionierter Sliver; Tokenizer wirft sonst "
                         "DegenerateBlockError)",
        "n_kept_input": len(runs),
        "n_selected": len(selected),
        "n_dropped": len(dropped),
        "n_cap_dropped": len(cap_dropped),
        "n_degenerate": len(degenerate),
        "n_geoms_total": len(geoms),
        "n_geoms_dropped_all_modes": sum(g["non_ideal_all_modes"] for g in geoms),
        "n_blocks_hist_kept": hist([r["n_blocks"] for r in runs]),
        "p95_n_blocks_kept": round(float(np.percentile(nb, 95)), 4),
        "dropped_dirs": [r["dir"] for r in dropped],
        "degenerate_dirs": [r["dir"] for r in degenerate],
    }
    payload = {"_meta": meta, "runs": runs, "geometries": geoms}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")

    print(f"kept input:        {len(runs)}")
    print(f"selected (<= {CAP} & valid): {len(selected)}")
    print(f"dropped:           {len(dropped)} "
          f"(cap {len(cap_dropped)}, degeneriert {len(degenerate)})")
    print(f"geometries:        {len(geoms)} (all-mode dropped: "
          f"{meta['n_geoms_dropped_all_modes']})")
    print(f"p95 n_blocks:      {meta['p95_n_blocks_kept']}")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
