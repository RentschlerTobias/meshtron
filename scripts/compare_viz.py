"""Vergleichs-VTK: True-Mesh + generierte Hexe + Conditioning-Punktwolke.

Ein UNSTRUCTURED_GRID mit CELL_DATA 'part': 1=true, 2=generated, 3=punktwolke.
ParaView: Threshold(part) + Coloring je Teil.

Beispiel:
  uv run python scripts/compare_viz.py \
    --src data/polytron_data_3d_smoke.pt --idx 3 \
    --seq data/seq_overfit_polar.pt --tokens data/hexarow_overfit_1sample.pt \
    --out data/compare_overfit_polar.vtk
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hexa_row_tokenizer import HexaRowTokenizer  # noqa: E402
from generate import detokenize_safe, load_sample, mesh_to_polar, sample_points  # noqa: E402


def _hexes_to_vtk_blocks(faces_t: np.ndarray) -> list[list[int]]:
    return [[int(i) for i in row] for row in faces_t]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/polytron_data_3d_smoke.pt")
    ap.add_argument("--idx", type=int, default=3)
    ap.add_argument("--seq", default="data/seq_overfit_polar.pt")
    ap.add_argument("--tokens", default="data/hexarow_overfit_1sample.pt")
    ap.add_argument("--out", default="data/compare_overfit_polar.vtk")
    ap.add_argument("--n-points", type=int, default=1000)
    args = ap.parse_args()

    tk_pt = torch.load(args.tokens, map_location="cpu", weights_only=False)
    rb, zb = tuple(tk_pt["r_bounds"]), tuple(tk_pt["z_bounds"])
    coords = tk_pt.get("coords", "polar")

    src = torch.load(args.src, weights_only=False)
    obj = src["samples"] if isinstance(src, dict) and "samples" in src else src
    xyz, _nblocks, name, faces_t, surf = load_sample(obj, args.idx, with_surface=True)
    assert faces_t is not None, "sample ohne faces"
    gt_blocks = _hexes_to_vtk_blocks(np.asarray(faces_t, dtype=np.int64))

    seq = torch.load(args.seq, map_location="cpu", weights_only=False)
    seq = seq.tolist() if hasattr(seq, "tolist") else list(seq)
    tok = HexaRowTokenizer(r_bounds=rb, z_bounds=zb)
    res, trim = detokenize_safe(seq, tok, tok.core.stop_token, coords=coords)
    assert res is not None, f"detokenize fehlgeschlagen: {trim}"
    vpt, blk = res
    vnp = vpt.numpy()
    gen_cart = (vnp if coords == "cart"
                else np.stack([vnp[:, 0] * np.cos(vnp[:, 1]),
                               vnp[:, 0] * np.sin(vnp[:, 1]),
                               vnp[:, 2]], axis=-1))
    gen_blocks = [[int(i) for i in b] for b in blk.numpy()]

    rng = np.random.default_rng(0)
    cloud_xyz = surf if surf is not None else xyz
    pts = sample_points(mesh_to_polar(cloud_xyz), args.n_points, rb, zb, rng)
    r01, s01, c01, z01 = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    th = np.arctan2(s01, c01)
    r_real = rb[0] + r01 * (rb[1] - rb[0])
    pc = np.stack([r_real * np.cos(th),
                   r_real * np.sin(th),
                   zb[0] + z01 * (zb[1] - zb[0])], axis=-1)

    parts = [(np.asarray(xyz, dtype=np.float64), gt_blocks, 1, 12),
             (gen_cart, gen_blocks, 2, 12),
             (pc, [[k] for k in range(len(pc))], 3, 1)]
    m = sum(len(p[0]) for p in parts)
    V = np.concatenate([p[0] for p in parts], axis=0)

    cells: list[str] = []
    types: list[str] = []
    scalars: list[str] = []
    base = 0
    for _, blocks, part, ctype in parts:
        for b in blocks:
            cells.append(f"{len(b)} " + " ".join(str(i + base) for i in b))
            types.append(str(ctype))
            scalars.append(str(part))
        base += len(next(p for p in parts if p[2] == part)[0])

    assert len(cells) == sum(len(p[1]) for p in parts), "zell-payment fehlerhaft"

    with open(args.out, "w") as fh:
        fh.write("# vtk DataFile Version 2.0\nmeshtron compare (true|generated|pointcloud)\nASCII\n")
        fh.write("DATASET UNSTRUCTURED_GRID\n")
        fh.write(f"POINTS {m} double\n")
        for p in V:
            fh.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        total = sum(len(c.split()) for c in cells)
        fh.write(f"CELLS {len(cells)} {total}\n")
        fh.write("\n".join(cells) + "\n")
        fh.write(f"CELL_TYPES {len(types)}\n" + "\n".join(types) + "\n")
        fh.write(f"CELL_DATA {len(scalars)}\n")
        fh.write("SCALARS part int 1\nLOOKUP_TABLE default\n")
        fh.write("\n".join(scalars) + "\n")

    print(f"mesh={name} coords={coords} | true blocks={len(gt_blocks)} "
          f"generated blocks={len(gen_blocks)} punktwolke={len(pc)}")
    if trim:
        print(f"note: seq wurde beim detokenize getrimmt: {trim}")
    print(f"saved {args.out}  (part: 1=true, 2=generated, 3=punktwolke)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
