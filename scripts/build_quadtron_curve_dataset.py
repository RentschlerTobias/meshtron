#!/usr/bin/env python3
"""build_quadtron_curve_dataset.py -- Phase 2 data: edge curves for the
structures Quadtron actually emits.

Quadtron 3D is trained on cartesian vertices + the BOUNDARY quad faces of the
block structure (`domain_extractor_3d.build_quadtron_sample`: `faces =
quad_faces.T`, "boundary quad faces -- Quadtron/tokenizer_v2"). Its edge stage
therefore has to curve the boundary-shell edges, not the full hex-block edge
set the Polytron head is trained on.

Per sample:
  vq      quantised vertices, spec fitted from the sample vertices
  edges   unique undirected edges of the boundary quad faces
  cq      GT Bezier control-point offsets for those edges (edge_ctrl -> chord
          offsets -> mu-law companded -> quantised), straight if no record
  + surface_points / point_labels / n_blocks, exactly what train_polytron's
  curve stage reads.

The geometry-disjoint train/val split is copied from the Polytron dataset by
`dir`, so Phase 1 and Phase 2 are compared on identical held-out geometries.

    uv run python scripts/build_quadtron_curve_dataset.py \
        --out data/quadtron_curve_blocks.pt
"""
from __future__ import annotations

import argparse
import json
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
    """All npz arrays; `polytron_blocks.load_npz` drops quad_faces."""
    with np.load(path, allow_pickle=True) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def face_edges(F: np.ndarray) -> np.ndarray:
    """[Q,4] quad faces -> unique undirected edges [E,2], a < b, sorted."""
    e = set()
    F = np.asarray(F)
    for f in F:
        for k in range(len(f)):
            a, b = int(f[k]), int(f[(k + 1) % len(f)])
            e.add((a, b) if a < b else (b, a))
    return np.asarray(sorted(e), dtype=np.int64).reshape(-1, 2)


def build_item(raw: dict, spec: PolytronSpec, meta: dict) -> dict:
    V = np.asarray(raw["vertices"], float)
    F = np.asarray(raw["quad_faces"], np.int64)          # [Q,4]
    edges = face_edges(F)
    # directed GT control points keyed on the raw vertex ids
    ctrl = {}
    for (a, b), (c1, c2) in zip(np.asarray(raw["edges"]), np.asarray(raw["edge_ctrl"])):
        ctrl[(int(a), int(b))] = (c1, c2)
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
    return dict(meta, vq=spec.quant_xyz(V), edges=edges, cq=spec.quant_curve(off),
                src=np.arange(len(V)), n_blocks=int(np.asarray(raw["blocks"]).shape[0]),
                n_missing_edges=int(n_missing),
                surface_points=raw["surface_points"].astype(np.float32),
                point_labels=lab)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--polytron", default=os.path.join(ROOT, "data",
                                                       "polytron_blocks_clean.pt"),
                    help="source of the geometry-disjoint split")
    ap.add_argument("--out", default=os.path.join(ROOT, "data",
                                                  "quadtron_curve_blocks.pt"))
    ap.add_argument("--q-vert", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    ref = torch.load(args.polytron, weights_only=False)
    run_of = {}
    for split in ("train", "val"):
        for it in ref[split]:
            run_of[it["dir"]] = split
    dirs = sorted(run_of)
    if args.limit:
        dirs = dirs[:args.limit]

    verts = [load_sample(os.path.join(DATA, d, "sample.npz"))["vertices"] for d in dirs]
    spec = fit_spec(verts, q_vert=args.q_vert)
    print(f"spec lo={np.round(spec.lo,4)} hi={np.round(spec.hi,4)} q_vert={spec.q_vert}")

    items = {"train": [], "val": []}
    skipped = 0
    n_missing = 0
    for d in dirs:
        raw = load_sample(os.path.join(DATA, d, "sample.npz"))
        if "quad_faces" not in raw:
            skipped += 1
            continue
        meta = {"name": d.replace("/", "__"), "dir": d}
        it = build_item(raw, spec, meta)
        n_missing += it["n_missing_edges"]
        items[run_of[d]].append(it)

    nv = [len(it["vq"]) for s in items.values() for it in s]
    ne = [len(it["edges"]) for s in items.values() for it in s]
    print(f"items train {len(items['train'])} val {len(items['val'])} skipped {skipped}")
    print(f"verts {min(nv)}..{max(nv)}  edges {min(ne)}..{max(ne)}  "
          f"edges without GT record {n_missing}")
    torch.save({"spec": spec.to_json(), **items, "family": "quadtron-curve-v1"},
               args.out)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
