#!/usr/bin/env python3
"""Build the Polytron dataset from the selected family runs.

Same selection (data/family_selection.json, n_blocks <= 25) and the same
geometry-disjoint split (conditioning.split_by_geometry, 90/10, seed 0) as the
Quadtron family tokens, so the two families are compared on identical
held-out geometries.

Every item is round-tripped on the way in: decode(encode(sample)) must give
back the block topology exactly and the corners to within half a quantisation
bin. An item that fails is reported and dropped, never silently kept.

  uv run python scripts/build_polytron_dataset.py [--limit 40]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from meshtron.data.conditioning import split_by_geometry  # noqa: E402
from meshtron.data.polytron_blocks import (decode_seq, fit_spec, item_seq,  # noqa: E402
                                           load_npz, make_item, selected_runs)

DATA = os.path.join(ROOT, "data", "hex3d_algohex")
_RUNS: list = []
_SPEC = None


def _verts(i):
    return load_npz(os.path.join(DATA, _RUNS[i]["dir"], "sample.npz"))["vertices"]


def roundtrip_error(raw: dict, it: dict, spec) -> str:
    """'' if the item decodes back to the sample, else the reason."""
    V, B, curves = decode_seq(item_seq(it), spec)
    V0 = raw["vertices"]
    nn = it["src"]
    step = (np.asarray(spec.hi) - np.asarray(spec.lo)) / (spec.q_vert - 1)
    err = np.abs(V - V0[nn]).max()
    if (np.abs(V - V0[nn]) > 0.5 * step + 1e-9).any():
        return f"corner error {err:.2e}"
    if len(set(map(tuple, it["vq"].tolist()))) != len(V0):
        # legal (pointers keep them apart) but the vertex model cannot emit
        # the same cell twice under a strict order mask -- counted, kept
        pass
    key = lambda blocks: {frozenset(b) for b in blocks}  # noqa: E731
    if key(nn[B]) != key(raw["blocks"]):
        return "block topology changed"
    if len(curves) * 2 != len(raw["edges"]):
        return f"edge count {len(curves)} vs {len(raw['edges']) // 2}"
    return ""


def _item(i):
    r = _RUNS[i]
    raw = load_npz(os.path.join(DATA, r["dir"], "sample.npz"))
    meta = {"name": r["dir"].replace("/", "__"), "dir": r["dir"],
            "geom_id": r["geom_id"], "grid_id": r["grid_id"]}
    try:
        it = make_item(raw, _SPEC, meta)
    except Exception as e:  # noqa: BLE001
        return i, None, f"SKIP {r['dir']}: {type(e).__name__}: {e}"
    err = roundtrip_error(raw, it, _SPEC)
    if err:
        return i, None, f"SKIP {r['dir']}: {err}"
    return i, it, ""


def clean_runs(runs: list) -> list:
    """The two kinds of training target the Polytron cannot learn from.

    T-junctions: two blocks touching across part of a side, which the
    four-corner face cannot express -- the structure leaves a slit (scanned
    by detect_block_tjunctions.py into data/polytron_tjunctions_<batch>.json).

    Contradictions: the same geometry at the same block count with a
    DIFFERENT blocking (n2000 vs n8000 AlgoHex runs). The model sees identical
    conditioning and two targets; it can reproduce at most one. The n2000 run
    is kept when present, otherwise the first in directory order."""
    bad = set()
    for b in ("batch", "batch_t19_sweep"):
        p = os.path.join(ROOT, "data", f"polytron_tjunctions_{b}.json")
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}: run detect_block_tjunctions.py "
                             f"--all --batch data/hex3d_algohex/{b} --out {p}")
        bad |= {f"{b}/{r['name']}" for r in json.load(open(p))["rows"]
                if r["tjunctions"]}
    kept = [r for r in runs if r["dir"] not in bad]
    print(f"clean: {len(runs) - len(kept)} runs with T-junctions dropped")
    by = {}
    for r in sorted(kept, key=lambda r: (not r["dir"].endswith("_n2000"), r["dir"])):
        by.setdefault((r["geom_id"], r["n_blocks"]), r)
    out = sorted(by.values(), key=lambda r: r["dir"])
    print(f"clean: {len(kept) - len(out)} contradicting targets dropped, "
          f"{len(out)} runs left")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="default data/polytron_blocks_clean.pt with --clean, "
                         "else data/polytron_blocks_family.pt")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=min(16, mp.cpu_count()))
    ap.add_argument("--q-vert", type=int, default=512)
    ap.add_argument("--clean", action="store_true",
                    help="drop samples with block T-junctions and keep ONE "
                         "target per (geometry, block count)")
    args = ap.parse_args()

    if args.out is None:
        args.out = os.path.join(ROOT, "data", "polytron_blocks_clean.pt" if args.clean
                                else "polytron_blocks_family.pt")
    global _RUNS, _SPEC
    _RUNS = selected_runs(ROOT)
    if args.clean:
        _RUNS = clean_runs(_RUNS)
    if args.limit:
        _RUNS = _RUNS[:args.limit]
    with mp.get_context("fork").Pool(args.jobs) as pool:
        vs = pool.map(_verts, range(len(_RUNS)))
    _SPEC = fit_spec(vs, q_vert=args.q_vert)
    print(f"runs {len(_RUNS)}  bounds lo={np.round(_SPEC.lo, 4)} "
          f"hi={np.round(_SPEC.hi, 4)}  q_vert={_SPEC.q_vert}")
    with mp.get_context("fork").Pool(args.jobs) as pool:
        res = pool.map(_item, range(len(_RUNS)))
    items = []
    for _i, it, msg in res:
        if msg:
            print(msg)
        if it is not None:
            items.append(it)
    train, val = split_by_geometry(items, val_frac=0.1, seed=0)
    nv = np.array([len(it["vq"]) for it in items])
    nb = np.array([it["n_blocks"] for it in items])
    ne = np.array([len(it["edges"]) for it in items])
    print(f"items {len(items)}/{len(_RUNS)}  train {len(train)} val {len(val)}  "
          f"geoms {len({i['geom_id'] for i in train})}/"
          f"{len({i['geom_id'] for i in val})}")
    print(f"verts {nv.min()}..{nv.max()}  blocks {nb.min()}..{nb.max()}  "
          f"edges {ne.min()}..{ne.max()}")
    torch.save({"spec": _SPEC.to_json(), "train": train, "val": val,
                "family": "polytron-v1"}, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
