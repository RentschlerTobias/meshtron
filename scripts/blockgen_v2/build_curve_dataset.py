"""Curve-head dataset from the beam-relabelled samples, same split as the v2
generator set (so the curve head's val geometries are the generator's).

Reuses meshtron's scripts/build_hexa_curve_dataset.build_item (GT edge_ctrl ->
chord offsets -> quantised); only the sample source and the split differ.

    python scripts/blockgen_v2/build_curve_dataset.py --split-from <v2 set>.pt --out <curve set>.pt --canonical

--canonical is REQUIRED for the inference chain: without it the curve head learns the raw
sample vertex order, while the generator emits the canonical order (11 % surface uncovered
on exact GT corners, reports/blockgen_v2_canonical_best_case.md).
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths  # noqa: E402  (puts the repo root on sys.path)
sys.path.insert(0, os.path.join(_paths.ROOT, "scripts"))
from build_hexa_curve_dataset import build_item, load_sample  # noqa: E402
from meshtron.data.polytron_blocks import fit_spec  # noqa: E402


def to_canonical(raw, it):
    """Renumber a raw sample into the canonical order of the v2 item (exact coordinate match)."""
    from scipy.spatial import cKDTree
    Vc = it["verts"].numpy().astype(np.float64)
    d, perm = cKDTree(raw["vertices"]).query(Vc)                 # canonical id -> raw id
    assert d.max() < 1e-9 and len(set(perm.tolist())) == len(Vc) == len(raw["vertices"])
    raw2c = np.empty(len(perm), np.int64); raw2c[perm] = np.arange(len(perm))
    out = dict(raw)
    out["vertices"] = raw["vertices"][perm]
    out["blocks"] = it["conn"].numpy().astype(np.int64).reshape(-1, 8)
    out["edges"] = raw2c[np.asarray(raw["edges"], np.int64)]       # direction (and its ctrl) kept
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-from", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--q-vert", type=int, default=512)
    ap.add_argument("--canonical", action="store_true",
                    help="renumber vertices/blocks into the generator's canonical order (it['verts'], it['conn'])")
    a = ap.parse_args()
    ds = torch.load(a.split_from, weights_only=False)
    split_of = {it["name"]: s for s in ("train", "val") for it in ds[s]}
    names = sorted(split_of)
    raws = {n: load_sample(os.path.join(_paths.need_samples(), n, "sample.npz")) for n in names}
    if a.canonical:
        canon = {it["name"]: it for s in ("train", "val") for it in ds[s]}
        for n in names:
            raws[n] = to_canonical(raws[n], canon[n])
    spec = fit_spec([raws[n]["vertices"] for n in names], q_vert=a.q_vert)
    items = {"train": [], "val": []}
    miss = 0
    for n in names:
        it = build_item(raws[n], spec, {"name": n, "dir": n, "geom_id": n.rsplit("_n", 1)[0], "grid_id": None})
        miss += it["n_missing_edges"]
        items[split_of[n]].append(it)
    torch.save({"spec": spec.to_json(), **items, "family": "hexa-curve-v2-relabel"}, a.out)
    print(f"wrote {a.out}: train {len(items['train'])} val {len(items['val'])}, edges without GT record {miss}")


if __name__ == "__main__":
    main()
