#!/usr/bin/env python3
"""One VTK with the generated mesh AND the real geometry, for ParaView.

    uv run python scripts/polytron_compare_vtk.py --mesh X_mesh.vtk \
        --npz data/hex3d_algohex/batch/machine_0001_n2000/sample.npz --out X_vs_geom.vtk

Cells:
  source 0  labelled npz surface triangles        (patch_label 1..7)
  source 1  generated hex mesh                    (block_id, scaled_jacobian, inverted)
Point data `dist_to_surface`: exact distance of every mesh point to the npz
surface (0 on the geometry points themselves), so "sits on" and "covers" can
both be read off one picture: colour the mesh boundary by it, and look for
surface triangles with no mesh in front of them.
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from meshtron.data.polytron_blocks import load_npz  # noqa: E402
from meshtron.geometry.polytron_tfi import _point_to_tris, read_hex_vtk  # noqa: E402


def write_compare(mesh_vtk: str, npz: str, out: str) -> dict:
    import meshio
    from meshtron.geometry.curved_bridge import _load
    _t, _e, _b, cb = _load()
    raw = load_npz(npz)
    SP, ST, SL = raw["surface_points"], raw["surface_tris"], raw["surface_tri_label"]
    P, H = read_hex_vtk(mesh_vtk)
    m = meshio.read(mesh_vtk)
    bid = np.concatenate([np.asarray(d).ravel() for d in m.cell_data["block_id"]])
    sj = cb.scaled_jacobians(P, H)
    dist = _point_to_tris(P, SP[ST[:, 0]], SP[ST[:, 1]], SP[ST[:, 2]])
    n = len(SP)
    pts = np.vstack([SP, P])
    with open(out, "w") as fh:
        fh.write("# vtk DataFile Version 2.0\n"
                 "polytron mesh (source 1) vs npz geometry (source 0)\nASCII\n"
                 "DATASET UNSTRUCTURED_GRID\n")
        fh.write(f"POINTS {len(pts)} double\n")
        fh.write("".join(f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f}\n" for q in pts))
        nc = len(ST) + len(H)
        fh.write(f"CELLS {nc} {4 * len(ST) + 9 * len(H)}\n")
        fh.write("".join(f"3 {t[0]} {t[1]} {t[2]}\n" for t in ST))
        fh.write("".join("8 " + " ".join(str(int(x) + n) for x in h) + "\n" for h in H))
        fh.write(f"CELL_TYPES {nc}\n" + "5\n" * len(ST) + "12\n" * len(H))
        fh.write(f"CELL_DATA {nc}\n")
        for name, a, b in (("source", np.zeros(len(ST)), np.ones(len(H))),
                           ("patch_label", SL, -np.ones(len(H))),
                           ("block_id", -np.ones(len(ST)), bid),
                           ("inverted", np.zeros(len(ST)), sj <= 0)):
            fh.write(f"SCALARS {name} int 1\nLOOKUP_TABLE default\n")
            fh.write("".join(f"{int(v)}\n" for v in np.concatenate([a, b])))
        fh.write("SCALARS scaled_jacobian double 1\nLOOKUP_TABLE default\n")
        fh.write("".join(f"{v:.6f}\n" for v in np.concatenate([np.ones(len(ST)), sj])))
        fh.write(f"POINT_DATA {len(pts)}\nSCALARS dist_to_surface double 1\n"
                 "LOOKUP_TABLE default\n")
        fh.write("".join(f"{v:.3e}\n" for v in np.concatenate([np.zeros(n), dist])))
    return {"cells": int(len(H)), "inverted": int((sj <= 0).sum()),
            "mesh_points_max_dist": float(dist.max())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(write_compare(a.mesh, a.npz, a.out), "->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
