"""Convert raw sample.npz (corner-hex block structures) to training samples.

Reads <src>/{batch,batch_t19_sweep}/*/sample.npz, converts to the polytron
sample format (vertices_cartesian f64, vertices_polar f32, faces [8,B]),
applies the quality gates and writes {'samples': [...], 'meta': {...}}.

Gates (drop sample, log reason):
  - blocks > --max-blocks (default 30; user rule)
  - any negative sub-tet volume (hex_tet_volumes from validate_3d_dataset)

Usage:
    uv run python scripts/clean_base_npz.py \
        --src data/hex3d_algohex_compressed/hex3d_algohex \
        --out data/polytron_batch_clean.pt
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meshtron.data.domain_extractor_3d import to_cylindrical  # noqa: E402


def _proc_one(path_str):
    path = Path(path_str)
    name = path.parent.name
    try:
        s = dict(np.load(path, allow_pickle=True))
        verts = np.asarray(s["vertices"], dtype=np.float64)
        blocks = np.asarray(s["blocks"], dtype=np.int64)  # [F, 8]
        for bi in range(blocks.shape[0]):
            if _signed_vol(blocks[bi], verts) < 0:
                blocks[bi] = blocks[bi][::-1]
        eps = 1e-9
        bad = [bi for bi in range(blocks.shape[0])
               if abs(_signed_vol(blocks[bi], verts)) <= eps]
        if bad:
            return (name, None, f"drop: {len(bad)} degenerate blocks")
        if blocks.shape[0] > MAX_BLOCKS[0]:
            return (name, None, f"drop: {blocks.shape[0]} blocks > {MAX_BLOCKS[0]}")
        sample = {
            "name": name,
            "machine": name.rsplit("_n", 1)[0],
            "blocks": blocks.shape[0],
            "vertices_cartesian": torch.tensor(verts, dtype=torch.float64),
            "vertices_polar": torch.tensor(
                to_cylindrical(verts), dtype=torch.float32),
            "faces": torch.tensor(blocks.T, dtype=torch.long),
        }
        return (name, sample, "ok")
    except Exception as exc:  # noqa: BLE001
        return (name, None, f"drop: {type(exc).__name__}: {exc}")


MAX_BLOCKS = [30]

# Hexa-Ecken in VTK-Ordnung: diese 6 Sub-Tets müssen ein positives
# Gesamtvolumen liefern (Metrik wie hex_tet_volumes)
_TETS = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6),
         (0, 4, 5, 6), (0, 5, 1, 6)]


def _signed_vol(blk, verts):
    tot = 0.0
    for t in _TETS:
        a, b, c, d = (verts[blk[k]] for k in t)
        tot += np.dot(np.cross(b - a, c - a), d - a) / 6.0
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="dir containing batch/ and/or batch_t19_sweep/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-blocks", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=min(24, os.cpu_count() or 1))
    a = ap.parse_args()
    MAX_BLOCKS[0] = a.max_blocks

    src = Path(a.src)
    dirs = [src / d for d in ("batch", "batch_t19_sweep") if (src / d).is_dir()]
    paths = sorted(str(p) for d in dirs for p in d.glob("*/sample.npz"))
    print(f"found {len(paths)} sample.npz under {src}")

    import multiprocessing as mp

    with mp.get_context("fork").Pool(processes=a.jobs) as pool:
        results = pool.map(_proc_one, paths, chunksize=8)

    samples, dropped = [], []
    for name, sample, msg in results:
        if sample is None:
            dropped.append(f"{name}: {msg}")
        else:
            samples.append(sample)
    machines = sorted({s["machine"] for s in samples})
    out = {
        "samples": samples,
        "meta": {
            "src": str(src),
            "max_blocks": a.max_blocks,
            "n_samples": len(samples),
            "n_machines": len(machines),
            "dropped": dropped,
        },
    }
    torch.save(out, a.out)
    print(f"wrote {a.out}: {len(samples)} samples, {len(machines)} machines "
          f"({len(dropped)} dropped)")
    for m in dropped[:10]:
        print(" ", m)


if __name__ == "__main__":
    main()
