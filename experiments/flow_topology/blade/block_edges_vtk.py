#!/usr/bin/env python3
"""Block-level view of the closed domain, no TFI: core block edges, their images on the walls (O-ring on the blade,
hub / shroud layers), radial lines at the block corners, and the wall surfaces from case/mesh.msh.
cell data `set`: 0 core block edges, 1/3/5 edge images on blade / hub / shroud (cut 1/2/3), 2/4/6 radial lines at
block corners of cut 1/2/3; triangles 10 blade, 11 hub, 12 shroud (wall surfaces, geometry only).

    python block_edges_vtk.py --name tistos_s1_0131 --dir out/layers_0131_dbg
"""
import argparse
import os
import sys

sys.path.insert(0, "/opt/stack/meshtron")
sys.path.insert(0, "/work/meshtron_v2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import meshio  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

import blade_insert as BI  # noqa: E402
import msh_walls as MW  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", default="tistos_s1_0131")
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    V, B, curves = BI.gt_post_structure(a.name)
    core = meshio.read(f"{a.dir}/core.vtk").points
    z = np.load(f"{a.dir}/columns.npz")
    P = z["P"]; col = {}
    for r, c, cc in zip(z["roots"], z["cut"], z["cols"]):
        col.setdefault(int(r), {})[int(c)] = cc
    tree = cKDTree(core)
    diag = float(np.linalg.norm(np.ptp(core, 0)))
    pts, lines, sets = [], [], []

    def add(Q, s):
        base = sum(len(p) for p in pts); pts.append(np.asarray(Q)); n = len(Q)
        lines.append(np.stack([base + np.arange(n - 1), base + np.arange(1, n)], 1)); sets.append(np.full(n - 1, s))

    def images(ids, level=0):
        """edge node chain -> its wall image chain(s), recursively through later cuts."""
        if len(ids) < 2 or not all(i in col for i in ids):
            return
        for c in sorted(set.intersection(*(set(col[i]) for i in ids))):
            tips = [int(col[i][c][-1]) for i in ids]
            add(P[tips], 2 * c - 1)
            images(tips, level + 1)

    def radial(i):
        for c, cc in col.get(i, {}).items():
            add(P[cc], 2 * c)
            radial(int(cc[-1]))
    for (p, q), Q in curves.items():
        add(Q, 0)
        # core mesh nodes on this edge, ordered along it
        dense = BI._arclength(np.asarray(Q, float), 400) if hasattr(BI, "_arclength") else Q
        d, j = tree.query(dense)
        ids = []
        for jj, dd in zip(j, d):
            if dd < 2e-3 * diag and (not ids or ids[-1] != jj):
                ids.append(int(jj))
        ids = list(dict.fromkeys(ids))
        images(ids)
    for c in range(len(V)):
        dd, j = tree.query(V[c])
        radial(int(j))
    Pf, T, L, info = MW.load_boundary(f"{BI.RUNS}/{a.name}/case/mesh.msh")
    raw = dict(np.load(f"{BI.RUNS}/{a.name}/sample.npz", allow_pickle=True))
    from meshtron.geometry.polytron_tfi import SurfaceProjector
    cls = BI.classify_core_patches(raw, SurfaceProjector(Pf, T, L), 2e-3 * float(np.linalg.norm(np.ptp(raw["surface_points"], 0))))
    order = [int(o) for o in z["order"]]
    walls = [cls[o][1] for o in order]
    tri_cells, tri_sets = [], []
    base = sum(len(p) for p in pts); pts.append(Pf)
    for k, w in enumerate(walls):
        tri_cells.append(T[L == w] + base); tri_sets.append(np.full(int((L == w).sum()), 10 + k))
    X = np.concatenate(pts)
    out = f"{a.dir}/block_edges_blade.vtk"
    meshio.Mesh(X, [("line", np.concatenate(lines)), ("triangle", np.concatenate(tri_cells))],
                cell_data={"set": [np.concatenate(sets).astype(np.int32), np.concatenate(tri_sets).astype(np.int32)]}
                ).write(out, binary=True)
    print("wrote", out, {s: int((np.concatenate(sets) == s).sum()) for s in range(7)}, "walls (cut order):", walls)


if __name__ == "__main__":
    main()
