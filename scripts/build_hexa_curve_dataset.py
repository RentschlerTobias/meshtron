#!/usr/bin/env python3
"""build_hexa_curve_dataset.py -- Phase 2 (corrected) data: edge curves for the
full hex block structures, on the 3D block generator's own split.

The 3D block generator is GPTCond + HexaRowTokenizer. Its token file carries the
exact train/val assignment (`dir` -> split), so this builder reuses it: the
curve head's val geometries are the generator's val geometries, and the chain
evaluation is apples-to-apples.

Per sample (from sample.npz):
  vq      quantised vertices (spec fitted globally over all samples)
  blocks  canonical hex blocks [F,8] (24 rotations, lex smallest)
  edges   unique undirected hex-block edges (a < b), sorted
  cq      GT cubic-Bezier offsets per edge (edge_ctrl -> chord offsets ->
          mu-law companded -> quantised)
  + surface_points / point_labels / n_blocks, what train_polytron's curve
  stage reads.

    uv run python scripts/build_hexa_curve_dataset.py
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from meshtron.data.polytron_blocks import (PolytronSpec, ctrl_to_offsets,  # noqa: E402
                                           fit_spec, point_labels)

DATA = os.path.join(ROOT, "data", "hex3d_algohex")


def load_sample(path: str) -> dict:
    with np.load(path, allow_pickle=True) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def hex_edges(blocks: np.ndarray) -> np.ndarray:
    """Unique undirected edges of hex blocks [F,8], a < b, sorted."""
    ed = set()
    for b in np.asarray(blocks):
        for i, j in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                     (0, 4), (1, 5), (2, 6), (3, 7)):
            a, c = int(b[i]), int(b[j])
            ed.add((a, c) if a < c else (c, a))
    return np.asarray(sorted(ed), dtype=np.int64).reshape(-1, 2)


def build_item(raw: dict, spec: PolytronSpec, meta: dict) -> dict:
    V = np.asarray(raw["vertices"], float)
    B = np.asarray(raw["blocks"], np.int64)                 # [F,8]
    edges = hex_edges(B)
    ctrl = {(int(a), int(b)): (c1, c2)
            for (a, b), (c1, c2) in zip(np.asarray(raw["edges"]),
                                        np.asarray(raw["edge_ctrl"]))}
    off = np.zeros((len(edges), 6))
    n_missing = 0
    for k, (a, b) in enumerate(edges):
        c = ctrl.get((int(a), int(b))) or ctrl.get((int(b), int(a)))
        if c is None:
            n_missing += 1
            continue
        off[k] = ctrl_to_offsets(V[a], V[b], c[0], c[1])
    lab = point_labels(len(raw["surface_points"]), raw["surface_tris"],
                       raw["surface_tri_label"])
    return dict(meta, vq=spec.quant_xyz(V), blocks=B, edges=edges,
                cq=spec.quant_curve(off), src=np.arange(len(V)),
                n_blocks=int(B.shape[0]), n_missing_edges=int(n_missing),
                surface_points=raw["surface_points"].astype(np.float32),
                point_labels=lab)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tokens", default=os.path.join(ROOT, "data",
                                                     "hexarow_tokens_family_cart.pt"),
                    help="source of the generator's train/val split")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "hexa_curve_blocks.pt"))
    ap.add_argument("--q-vert", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    ds = torch.load(args.tokens, weights_only=False)
    split_of = {it["dir"]: "train" for it in ds["train"]}
    split_of.update({it["dir"]: "val" for it in ds["val"]})
    meta_of = {it["dir"]: {"name": it["name"], "dir": it["dir"],
                           "geom_id": it.get("geom_id"), "grid_id": it.get("grid_id")}
               for it in ds["train"] + ds["val"]}
    dirs = sorted(split_of)
    if args.limit:
        dirs = dirs[:args.limit]

    verts = [load_sample(os.path.join(DATA, d, "sample.npz"))["vertices"] for d in dirs]
    spec = fit_spec(verts, q_vert=args.q_vert)
    print(f"spec lo={np.round(spec.lo,4)} hi={np.round(spec.hi,4)} q_vert={spec.q_vert}")

    items = {"train": [], "val": []}
    n_missing = 0
    for d in dirs:
        raw = load_sample(os.path.join(DATA, d, "sample.npz"))
        it = build_item(raw, spec, meta_of[d])
        n_missing += it["n_missing_edges"]
        items[split_of[d]].append(it)

    nv = [len(it["vq"]) for s in items.values() for it in s]
    ne = [len(it["edges"]) for s in items.values() for it in s]
    print(f"items train {len(items['train'])} val {len(items['val'])}")
    print(f"verts {min(nv)}..{max(nv)}  edges {min(ne)}..{max(ne)}  "
          f"edges without GT record {n_missing}")
    torch.save({"spec": spec.to_json(), **items, "family": "hexa-curve-v1"}, args.out)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
