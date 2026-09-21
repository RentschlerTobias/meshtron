#!/usr/bin/env python3
"""Baue HexaRow-Tokens fuer die selektierte Geometrie-Familie.

Quelle: data/family_selection.json (select=true Runs, Cap n_blocks<=25).
Pro Run sample.npz -> Sample-Dict (vertices_polar/cartesian, faces[8,B],
edge_index[2,E]) -> HexaRowTokenizer polar + cart.

Split GEOM-disjunkt via conditioning.split_by_geometry (90/10, seed 0),
nicht positional wie im alten Script. Items tragen name, geom_id, grid_id,
tokens, blocks plus surface_points + is_blade (Blade-Label 5) fuer die
gewichtete Konditionierung.

  uv run python scripts/build_family_tokens.py
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import tempfile
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np
import torch

from build_hexarow_tokens import bounds_from
from conditioning import point_is_blade, polar_from_xyz, split_by_geometry
from hexa_row_tokenizer import HexaRowTokenizer

DATA = os.path.join(ROOT, "data", "hex3d_algohex")
SEL_JSON = os.path.join(ROOT, "data", "family_selection.json")
AUG_CAP = 25  # gleicher Cap wie D1 (n_blocks <= 25)
TFI_REPO = os.path.join(os.path.dirname(ROOT), "domain_partition_3D",
                        "experimentell", "hex3d_algohex")
_HEX_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7))
_TOK = None
_RUNS = None
_TFI = None
_TFI_H = None


def load_sample(rel):
    """sample.npz -> (sample-dict, surface_points [P,3] float32, is_blade [P])."""
    p = os.path.join(DATA, rel, "sample.npz")
    with np.load(p) as d:
        V = np.asarray(d["vertices"], dtype=np.float64)
        B = np.asarray(d["blocks"], dtype=np.int64)
        sp = np.asarray(d["surface_points"], dtype=np.float32)
        is_blade = point_is_blade(sp.shape[0], d["surface_tris"],
                                  d["surface_tri_label"])
        edges = (np.asarray(d["edges"], dtype=np.int64)
                 if "edges" in d.files else np.zeros((0, 2), dtype=np.int64))
    vp = polar_from_xyz(V)
    sample = {
        "vertices_polar": torch.tensor(vp, dtype=torch.float32),
        "vertices_cartesian": torch.tensor(V, dtype=torch.float32),
        "faces": torch.tensor(B.T, dtype=torch.long),
        "edge_index": torch.tensor(edges.T, dtype=torch.long),
    }
    return sample, sp, is_blade


def _bounds_one(i):
    with np.load(os.path.join(DATA, _RUNS[i]["dir"], "sample.npz")) as d:
        V = np.asarray(d["vertices"], dtype=np.float64)
    vp = polar_from_xyz(V).astype(np.float32)
    return i, vp, ""


def _proc_one(i):
    r = _RUNS[i]
    try:
        sample, sp, is_blade = load_sample(r["dir"])
        ids_p = _TOK.tokenize(sample, coords="polar")
        ids_c = _TOK.tokenize(sample, coords="cart")
    except Exception as e:
        return i, None, f"SKIP {r['dir']}: {type(e).__name__}: {e}"
    return i, {
        "name": r["dir"].replace("/", "__"),
        "dir": r["dir"], "geom_id": r["geom_id"], "grid_id": r["grid_id"],
        "blocks": int(sample["faces"].shape[1]),
        "tokens_polar": torch.tensor(ids_p, dtype=torch.long),
        "tokens_cart": torch.tensor(ids_c, dtype=torch.long),
        "surface_points": torch.tensor(sp, dtype=torch.float32),
        "is_blade": torch.tensor(is_blade, dtype=torch.bool),
    }, ""


def run_pool(fn, jobs_n, jobs):
    results = [None] * jobs_n
    with mp.get_context("fork").Pool(jobs) as pool:
        for n, (i, val, msg) in enumerate(pool.imap(fn, range(jobs_n)), 1):
            results[i] = val
            if msg:
                print(msg)
            if n % 100 == 0 or n == jobs_n:
                print(f"  [{n}/{jobs_n}]", flush=True)
    return results


def save_arm(path, items, rb, zb, coords, vocab, extra=None):
    npt = 3 if coords == "cart" else 4
    payload = {"train": items[0], "val": items[1], "vocab": vocab,
               "r_bounds": rb, "z_bounds": zb, "coords": coords, "npt": npt,
               "family": f"{coords}-v1"}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"wrote {path}: {len(items[0])} train / {len(items[1])} val "
          f"npt={npt} family={payload['family']}")


def _hex_edges(blocks):
    e = set()
    for b in blocks:
        for a, c in _HEX_EDGES:
            u, v = int(b[a]), int(b[c])
            e.add((u, v) if u < v else (v, u))
    return np.asarray(sorted(e), dtype=np.int64)


def _canon(vertices, blocks, tmp):
    """Kanonischer grid_id via dedupe_grid-Hashfunktion (DATA umgebogen,
    damit kein Schreibzugriff auf data/hex3d_algohex noetig ist)."""
    import dedupe_grid as dg
    sub = os.path.join(tmp, "g")
    os.makedirs(sub, exist_ok=True)
    np.savez(os.path.join(sub, "sample.npz"),
             vertices=np.asarray(vertices, dtype=np.float64),
             blocks=np.asarray(blocks, dtype=np.int64))
    old = dg.DATA
    dg.DATA = Path(tmp)
    try:
        return dg.grid_id("g")[0]
    finally:
        dg.DATA = old


def _refill_one(i):
    r = _RUNS[i]
    bvtk = os.path.join(DATA, r["dir"], "blocks.vtk")
    try:
        P, H, B, f2h = _TFI.load_blocks(bvtk)
        lat, missing = _TFI.lattices(P, H, B, f2h, verbose=False)
        if missing:
            return i, None, f"AUG-SKIP {r['dir']}: {len(missing)} non-lattice"
        classes = _TFI.direction_classes(lat, f2h, H, B, verbose=False)
        counts = _TFI.solve_block_divisions(lat, classes, P, _TFI_H,
                                            frozen_counts={}, bound="max",
                                            verbose=False)
        Pn, Hn, Bn, _ = _TFI.refill_complex(P, H, B, f2h, lat, classes, counts,
                                            verbose=False)
    except Exception as e:
        return i, None, f"AUG-SKIP {r['dir']}: {type(e).__name__}: {e}"
    if int(Hn.shape[0]) > AUG_CAP:
        return i, None, ""  # ueber Cap -> verworfen
    return i, {"dir": r["dir"], "geom_id": r["geom_id"],
               "grid_id": f"tfi-h05:{r['grid_id']}",
               "cells": int(Hn.shape[0]),
               "V": Pn.astype(np.float32), "B": Hn.astype(np.int64)}, ""


def augment(args, raws, rb, zb, vocab):
    """--tfi-h: pro selektiertem Run tfi-Refill, Cap-Recheck, Dedup, polar-Arm."""
    sys.path.insert(0, args.tfi_repo)
    global _TFI, _TFI_H
    import tfi
    _TFI, _TFI_H = tfi, args.tfi_h
    res = [r for r in run_pool(_refill_one, len(_RUNS), args.jobs)
           if isinstance(r, dict)]
    print(f"tfi h={args.tfi_h}: {len(res)}/{len(_RUNS)} <= cap {AUG_CAP}")
    tmp = tempfile.mkdtemp(prefix="tfi_aug_")
    seen, uniq = set(), []
    for a in res:
        key = (a["geom_id"], _canon(a["V"], a["B"], tmp))
        if key not in seen:
            seen.add(key)
            uniq.append(a)
    print(f"dedupe: {len(uniq)} unique grids")
    if not uniq:
        print("keine Augment-Samples ueber dem Cap -> kein Artifact (Blocker)")
        return
    base = [{"geom_id": r["geom_id"]} for r in raws]
    tr, va = split_by_geometry(base, val_frac=0.1, seed=0)
    side = {it["geom_id"]: "val" for it in va}
    surf = {r["dir"]: (r["surface_points"], r["is_blade"]) for r in raws}
    train, val = [], []
    for a in uniq:
        V, B = a["V"].astype(np.float64), a["B"]
        sample = {"vertices_cartesian": torch.tensor(V, dtype=torch.float32),
                  "vertices_polar": torch.tensor(polar_from_xyz(V),
                                                 dtype=torch.float32),
                  "faces": torch.tensor(B.T, dtype=torch.long),
                  "edge_index": torch.tensor(_hex_edges(B).T, dtype=torch.long)}
        sp, bl = surf[a["dir"]]
        item = {"tokens": torch.tensor(_TOK.tokenize(sample, coords="polar"),
                                       dtype=torch.long),
                "name": a["dir"].replace("/", "__") + "#tfi-h05",
                "dir": a["dir"], "geom_id": a["geom_id"],
                "grid_id": a["grid_id"], "blocks": a["cells"],
                "surface_points": sp, "is_blade": bl, "coords": "polar"}
        (val if side.get(a["geom_id"]) == "val" else train).append(item)
    out = args.aug_out or os.path.join(args.out_dir,
                                       "hexarow_tokens_family_aug_h05_polar.pt")
    save_arm(out, (train, val), rb, zb, "polar", vocab,
             extra={"family": "polar-v1-aug-h05", "aug_h": args.tfi_h})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=min(24, mp.cpu_count()))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--tfi-h", type=float, default=None,
                    help="TFI-Refill-Augmentation bei diesem Ziel-h "
                         "(>=0.5, User-Floor); braucht --with scipy --with meshio")
    ap.add_argument("--tfi-repo", default=TFI_REPO,
                    help="Pfad zu domain_partition_3D/.../hex3d_algohex (tfi.py)")
    ap.add_argument("--aug-out", default=None)
    args = ap.parse_args()
    if args.tfi_h is not None and args.tfi_h < 0.5:
        raise SystemExit("--tfi-h < 0.5 verboten (User-Floor)")

    global _TOK, _RUNS
    sel = json.loads(open(SEL_JSON).read())
    _RUNS = [r for r in sel["runs"] if r["select"]]
    _RUNS.sort(key=lambda r: r["dir"])
    print(f"selected runs: {len(_RUNS)}")

    vps = [r for r in run_pool(_bounds_one, len(_RUNS), args.jobs)
           if isinstance(r, np.ndarray)]
    light = [{"vertices_polar": v} for v in vps]
    if len(light) != len(_RUNS):
        raise SystemExit(f"bounds: {len(light)}/{len(_RUNS)} runs lesbar")
    rb, zb = bounds_from(light)
    print(f"r_bounds={tuple(round(v,4) for v in rb)} "
          f"z_bounds={tuple(round(v,4) for v in zb)}")
    _TOK = HexaRowTokenizer(r_bounds=rb, z_bounds=zb)
    vocab = _TOK.core.vocab_size

    raws = [r for r in run_pool(_proc_one, len(_RUNS), args.jobs)
            if isinstance(r, dict)]
    raws.sort(key=lambda r: r["dir"])
    print(f"tokenized: {len(raws)}/{len(_RUNS)}")

    for coords, tag in (("polar", "polar"), ("cart", "cart")):
        items = [{
            "tokens": r[f"tokens_{coords}"], "name": r["name"],
            "dir": r["dir"],
            "geom_id": r["geom_id"], "grid_id": r["grid_id"],
            "blocks": r["blocks"], "surface_points": r["surface_points"],
            "is_blade": r["is_blade"], "coords": coords,
        } for r in raws]
        train, val = split_by_geometry(items, val_frac=0.1, seed=0)
        lens = np.array([len(it["tokens"]) for it in train + val])
        print(f"[{coords}] train {len(train)} / val {len(val)} geoms "
              f"{len({it['geom_id'] for it in train})}/{len({it['geom_id'] for it in val})} "
              f"tok min {lens.min()} median {int(np.median(lens))} max {lens.max()}")
        save_arm(os.path.join(args.out_dir, f"hexarow_tokens_family_{tag}.pt"),
                 (train, val), rb, zb, coords, vocab)

    if args.tfi_h is not None:
        augment(args, raws, rb, zb, vocab)


if __name__ == "__main__":
    main()
