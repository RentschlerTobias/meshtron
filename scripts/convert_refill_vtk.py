"""Convert tfi.py-refilled blocks VTKs (deliverable *_h*.vtk) into meshtron
hexarow sample format (.pt list of dicts), ready for build_hexarow_tokens.py.

Input  : legacy VTK UNSTRUCTURED_GRID, hexa cells (type 12) + block_id cell data.
Output : {samples: [...], meta: {...}} — each sample:
         vertices_cartesian f64, vertices_polar f32 (via domain_extractor_3d.to_cylindrical),
         faces [8,F] (hexas transposed), name, level (target-h string).

Usage:
    uv run python scripts/convert_refill_vtk.py \
        --src data/hex3d_algohex/deliverable --out data/hexarow_refill_ladder.pt
"""

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import numpy as np
import torch

from domain_extractor_3d import to_cylindrical  # noqa: E402  (repo root on sys.path)


def load_legacy_hexa(path):
    """Parse legacy ASCII VTK: return (pts, hexas[F,8], block_id[F])."""
    pts, hexas, blk = [], [], []
    with open(path) as f:
        lines = f.read().splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("POINTS"):
            n = int(ln.split()[1])
            j = i + 1
            pts = []
            while len(pts) < n:
                pts.append([float(v) for v in lines[j].split()])
                j += 1
            pts = np.array(pts, dtype=np.float64)
            i = j
            continue
        if ln.startswith("CELLS"):
            n, m = int(ln.split()[1]), int(ln.split()[2])
            j = i + 1
            while j < len(lines) and len(hexas) < n:
                parts = lines[j].split()
                if int(parts[0]) == 8:
                    hexas.append([int(x) for x in parts[1:9]])
                j += 1
            i = j
            continue
        if ln.startswith("CELL_DATA"):
            # first scalar array after this header is block_id (color)
            j = i + 1
            while j < len(lines) and not lines[j].startswith(("SCALARS", "LOOKUP")):
                j += 1
            j += 2 if lines[j].startswith("SCALARS") else 0
            while j < len(lines):
                try:
                    blk.append(int(float(lines[j])))
                    j += 1
                except ValueError:
                    break
            i = j
            continue
        i += 1
    return (np.array(pts, dtype=np.float64), np.array(hexas, dtype=np.int64),
            np.array(blk, dtype=np.int64) if blk else None)


def _proc_one(i):
    pts, hexas, blk = load_legacy_hexa(_PATHS[i].as_posix())
    p = _PATHS[i]
    m = re.search(r"_h([0-9.]+)\.vtk$", p.name)
    level = m.group(1) if m else "?"
    pv = to_cylindrical(pts)
    s = {
        "vertices_cartesian": torch.tensor(pts, dtype=torch.float64),
        "vertices_polar": torch.tensor(pv, dtype=torch.float32),
        "faces": torch.tensor(hexas.T, dtype=torch.long),  # [8, F]
        "block_ids": None if blk is None else torch.tensor(blk),
        "name": p.stem,
        "level": level,
        "blocks": int(hexas.shape[0]),
    }
    print(f"[{i+1}/{len(_PATHS)}] {p.name}: {hexas.shape[0]} hexes, {len(pts)} pts",
          flush=True)
    return s


_PATHS = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/hex3d_algohex/deliverable")
    ap.add_argument("--pattern", default=r"T1_9_blocks_.*_h[0-9.]+\.vtk$")
    ap.add_argument("--out", default="data/hexarow_refill_ladder.pt")
    ap.add_argument("--jobs", type=int, default=min(24, os.cpu_count() or 1))
    a = ap.parse_args()

    global _PATHS
    _PATHS = sorted(p for p in Path(a.src).glob("*.vtk")
                    if re.match(a.pattern, p.name))
    print(f"{len(_PATHS)} files match")
    if not _PATHS:
        torch.save({"samples": [], "meta": {"source": "tfi.py refill sweep",
                                            "n": 0}}, a.out)
        return
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    with ctx.Pool(a.jobs) as pool:
        samples = pool.map(_proc_one, range(len(_PATHS)), chunksize=1)
    torch.save({"samples": samples,
                "meta": {"source": "tfi.py refill sweep", "n": len(samples)}},
               a.out)
    print(f"wrote {a.out} ({len(samples)} samples)")


if __name__ == "__main__":
    main()
