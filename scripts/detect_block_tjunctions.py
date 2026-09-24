"""detect_block_tjunctions.py -- exact test for non-conforming block interfaces.

A block face in the npz format is defined by four corners. When two blocks
touch across only PART of a side -- a T-junction -- that interface cannot be
expressed, so both sides are recorded as domain boundary and the Coons
reconstruction leaves a void between them (measured 0.046 wide on
machine_0387_n8000, about one cell at h=0.05).

The AlgoHex mesh itself is fine: `blocks.vtk` is a conforming hex mesh. The
defect enters with the block abstraction (tfi.lattices -> 8 corners per block),
so regenerating the data with AlgoHex would reproduce it.

The test needs no geometry and no refill: load blocks.vtk, find block pairs
that share fine facets, and check whether they also share a block face.

  blocks that share fine facets but no block face  ->  T-junction

Corpus result (689 samples): 118 affected (17.1%). Block-count classes 11, 15,
19 and 21 carry them; 12, 16, 22 and 25 have exactly none. Cross-check against
the geometric void detector (scripts/detect_block_voids.py): samples with a
T-junction have >= 6 void facet pairs (median 64), samples without have <= 7
(median 2, detector noise) -- a threshold of 11 separates them perfectly.

Usage:
  uv run python scripts/detect_block_tjunctions.py --all --out data/tjunctions.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEX3D = ("/home/t1dde/hydrostack_pipeline/stack/domain_partition_3D/"
         "experimentell/hex3d_algohex")
if HEX3D not in sys.path:
    sys.path.insert(0, HEX3D)

import export_sample as ex  # noqa: E402  (extern, read-only)
import tfi  # noqa: E402

HEX_FACES = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5),
             (2, 3, 7, 6), (3, 0, 4, 7))


def scan(blocks_vtk: str) -> dict:
    P, H, B, f2h = tfi.load_blocks(blocks_vtk)
    lat, missing = tfi.lattices(P, H, B, f2h, verbose=False)
    corners = ex.block_corners(lat)
    owner = defaultdict(set)
    mult = Counter()
    for ci, c in enumerate(H):
        for f in HEX_FACES:
            k = tuple(sorted(int(c[x]) for x in f))
            mult[k] += 1
            owner[k].add(int(B[ci]))
    block_faces = {r: {frozenset(int(list(corners[r])[i]) for i in f)
                       for f in HEX_FACES} for r in corners}
    shared = Counter()
    for k, v in owner.items():
        if len(v) == 2:
            shared[tuple(sorted(v))] += 1
    tj = [(a, b, n) for (a, b), n in shared.items()
          if a in block_faces and b in block_faces
          and not (block_faces[a] & block_faces[b])]
    return {
        "blocks": len(lat), "hexes": int(len(H)),
        "blocks_without_lattice": len(missing),
        "fine_facet_multiplicity": {str(k): int(v)
                                    for k, v in Counter(mult.values()).items()},
        "block_pairs_sharing_facets": len(shared),
        "tjunctions": len(tj),
        "tjunction_facets": int(sum(n for _, _, n in tj)),
        "tjunction_pairs": [[int(a), int(b), int(n)] for a, b, n in tj],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="find block-level T-junctions")
    ap.add_argument("--samples", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--batch", default=os.path.join(ROOT, "data",
                                                    "hex3d_algohex", "batch"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data",
                                                  "tjunctions.json"))
    args = ap.parse_args()

    if args.all:
        paths = sorted(glob.glob(os.path.join(args.batch, "*", "blocks.vtk")))
    else:
        paths = [os.path.join(args.batch, s, "blocks.vtk")
                 for s in args.samples.split(",") if s]

    rows, failed = [], []
    for k, p in enumerate(paths, 1):
        name = os.path.basename(os.path.dirname(p))
        try:
            row = scan(p)
        except Exception as exc:  # noqa: BLE001
            failed.append({"name": name,
                           "error": f"{type(exc).__name__}: {exc}"})
            continue
        row["name"] = name
        rows.append(row)
        flag = "T-JUNCTION" if row["tjunctions"] else "ok"
        print(f"[{k}/{len(paths)}] {name:24s} blocks={row['blocks']:3d} "
              f"tjunctions={row['tjunctions']:2d} "
              f"({row['tjunction_facets']:4d} facets)  {flag}", flush=True)

    bad = [r for r in rows if r["tjunctions"]]
    by_blocks = defaultdict(lambda: [0, 0])
    for r in rows:
        e = by_blocks[r["blocks"]]
        e[0] += 1
        e[1] += 1 if r["tjunctions"] else 0
    summary = {"n": len(rows), "with_tjunctions": len(bad), "failed": failed,
               "by_block_count": {str(k): {"n": v[0], "affected": v[1]}
                                  for k, v in sorted(by_blocks.items())},
               "rows": rows}
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nsaved {args.out}")
    print(f"{len(bad)}/{len(rows)} samples with block-level T-junctions, "
          f"{len(failed)} errors")
    for k, v in sorted(by_blocks.items()):
        print(f"   {k:3d} blocks: {v[0]:4d} samples, {v[1]:4d} affected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
