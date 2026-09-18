"""Spot-check VTK export of TFI-augmented fine dataset samples.

Usage:
    uv run python scripts/export_fine_vtk.py \
        --src data/fine/polytron_data_3d_fine.pt \
        --idx 0 461 900 --out-dir data/fine/vtk
Writes one .vtk per selected index (hexa cells only, cartesian coords).
"""

import argparse
from pathlib import Path

import numpy as np
import torch


def write_vtk(path, verts, faces):
    """faces: [8, F] integer tensor -> legacy VTK UNSTRUCTURED_GRID hexes."""
    faces = np.asarray(faces if isinstance(faces, np.ndarray) else faces.numpy(), dtype=np.int64)
    F = faces.shape[1]
    pts = np.asarray(verts, dtype=np.float64)
    cells = np.concatenate(
        [np.tile([8], (F, 1)), (faces.T)], axis=1).ravel()
    cell_types = np.full(F, 12, dtype=np.uint8)
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\nfine hexa\nASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {len(pts)} double\n")
        for p in pts:
            f.write(f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")
        f.write(f"CELLS {F} {len(cells)}\n")
        f.write("\n".join(" ".join(map(str, c)) for c in cells.reshape(F, 9)))
        f.write("\n")
        f.write(f"CELL_TYPES {F}\n")
        f.write("\n".join(map(str, cell_types)))
        f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/fine/polytron_data_3d_fine.pt")
    ap.add_argument("--idx", type=int, nargs="+", default=[0, 15, 461, 476, 900])
    ap.add_argument("--out-dir", default="data/fine/vtk")
    a = ap.parse_args()

    data = torch.load(a.src, map_location="cpu", weights_only=False)
    samples = data if isinstance(data, list) else data.get("samples", data["train"])
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i in a.idx:
        s = samples[i]
        verts = s["vertices_cartesian"].numpy()
        faces = s["faces"] if s["faces"].shape[0] == 8 else s["faces"].T
        p = out / f"fine_{i}.vtk"
        write_vtk(p, verts, faces)
        print(f"{i}: blocks={faces.shape[1]} verts={len(verts)} -> {p}")


if __name__ == "__main__":
    main()
