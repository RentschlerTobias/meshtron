#!/usr/bin/env python3
"""debug_inverted_cells.py — mark inverted TFI cells in a legacy VTK for ParaView.

Reads a curved/chord CFD volume written by curved_bridge/tfi_bridge, recomputes
the scaled jacobian per hex cell (same clean_blocks function the refill report
uses) and writes <input>_invdebug.vtk with two CELL_DATA arrays:
'scaled_jacobian' (float) and 'inverted' (int 0/1). ParaView: Threshold
'inverted' == 1 to see WHERE the mesh folds.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def read_vtk(path):
    """Minimal legacy ASCII VTK reader -> (points (n,3), cells, ctypes, part)."""
    with open(path) as fh:
        tok = fh.read().split()
    i = 0
    while tok[i] != "POINTS":
        i += 1
    npts = int(tok[i + 1])
    pts = np.array([float(x) for x in tok[i + 3:i + 3 + 3 * npts]]).reshape(npts, 3)
    j = i + 3 + 3 * npts
    assert tok[j] == "CELLS"
    ncells = int(tok[j + 1])
    cells, k = [], j + 3
    for _ in range(ncells):
        m = int(tok[k])
        cells.append([int(x) for x in tok[k + 1:k + 1 + m]])
        k += 1 + m
    assert tok[k] == "CELL_TYPES"
    ctypes = np.array([int(x) for x in tok[k + 2:k + 2 + ncells]])
    part = None
    for c in range(k + 2 + ncells, len(tok) - 4):
        if tok[c] == "SCALARS" and tok[c + 1] == "part":
            part = np.array([int(x) for x in tok[c + 6:c + 6 + ncells]])
            break
    return pts, cells, ctypes, part


def write_vtk(path, pts, cells, ctypes, arrays: dict):
    with open(path, "w") as fh:
        fh.write("# vtk DataFile Version 3.0\nmeshtron inverted-cell debug\nASCII\n"
                 "DATASET UNSTRUCTURED_GRID\n")
        fh.write(f"POINTS {len(pts)} double\n")
        for p in pts:
            fh.write(f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")
        csz = sum(len(c) + 1 for c in cells)
        fh.write(f"CELLS {len(cells)} {csz}\n")
        for c in cells:
            fh.write(f"{len(c)} " + " ".join(str(x) for x in c) + "\n")
        fh.write(f"CELL_TYPES {len(cells)}\n")
        for t in ctypes:
            fh.write(f"{int(t)}\n")
        fh.write(f"CELL_DATA {len(cells)}\n")
        for name, (vals, fmt) in arrays.items():
            scalar = "float" if fmt == "%.6f" else "int"
            fh.write(f"SCALARS {name} {scalar} 1\nLOOKUP_TABLE default\n")
            for v in vals:
                fh.write((fmt % v) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="mark inverted hex cells in a VTK")
    ap.add_argument("--vtk", default=os.path.join(ROOT, "data", "map_batch__machine_0034_n2000", "cfd_curved.vtk"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sys.path.insert(0, os.environ.get("HEX3D_REPO", "/home/t1dde/hydrostack_pipeline/"
                                      "stack/domain_partition_3D/experimentell/hex3d_algohex"))
    import clean_blocks as cb

    pts, cells, ctypes, _ = read_vtk(args.vtk)
    hex_idx = np.nonzero(ctypes == 12)[0]
    H = np.array([cells[i] for i in hex_idx], dtype=int)
    sj = cb.scaled_jacobians(pts, H)
    inv = np.zeros(len(cells), int)
    inv[hex_idx] = (sj <= 0.0).astype(int)
    sj_full = np.zeros(len(cells))
    sj_full[hex_idx] = sj

    out = args.out or os.path.splitext(args.vtk)[0] + "_invdebug.vtk"
    write_vtk(out, pts, cells, ctypes, {"scaled_jacobian": (sj_full, "%.6f"),
                                        "inverted": (inv, "%d")})
    bad = hex_idx[sj <= 0.0]
    P = np.array([np.mean([pts[c[k]] for k in range(8)], axis=0) for c in cells])
    print(f"[invdebug] cells={len(cells)} hexes={len(hex_idx)} inverted={len(bad)} "
          f"min_sj={sj.min():.4f}")
    if len(bad):
        cen = P[bad]
        r = np.hypot(cen[:, 0], cen[:, 1])
        th = np.degrees(np.arctan2(cen[:, 1], cen[:, 0])) % 360
        print(f"  centroid bbox: r {r.min():.3f}..{r.max():.3f}  "
              f"z {cen[:, 2].min():.3f}..{cen[:, 2].max():.3f}")
        hist, _ = np.histogram(th, bins=12, range=(0, 360))
        print("  theta sectors (30deg bins):", [int(h) for h in hist])
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
