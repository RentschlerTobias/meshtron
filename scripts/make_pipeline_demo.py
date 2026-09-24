"""make_pipeline_demo.py -- one VTK per pipeline stage, for walking an audience
through the whole chain on a single machine.

Stages, in the order they run:

  01 geometry      the parametric runner geometry as the labelled npz surface
  02 tet mesh      the gmsh volume mesh the hex mesher starts from
  03 point cloud   what the transformer actually sees as conditioning
  04 algohex       AlgoHex's fine hex mesh, coloured by block
  05 block complex the coarse block structure the export keeps (GT)
  06 transformer   the block structure the transformer generates
  07 post GT       GT blocks conformed to the geometry and refilled (this repo)
  08 post generated  the same for the generated blocks

Everything is copied or rebuilt into one directory with numbered names and a
README, so the files can be dropped into ParaView in order.

Usage:
  uv run python scripts/make_pipeline_demo.py --machine machine_0034_n2000
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from block_mapping import SnapConfigV2, snap_corners_v2  # noqa: E402
from curved_bridge import refill_curved  # noqa: E402
from geometry_features import FeatureModelV2  # noqa: E402
from patch_paths import (PatchPaths, make_boundary_face_test,  # noqa: E402
                         make_face_projector, snap_seam_path)
from scripts.conform_gt_blocks import _boundary_edge_pred  # noqa: E402
from scripts.map_generated_blocks import _seam_path_fn  # noqa: E402

BATCH = os.path.join(ROOT, "data", "hex3d_algohex", "batch")
LABELS = {1: "inlet", 2: "outlet", 3: "periodic", 4: "periodic",
          5: "hub", 6: "shroud", 7: "blade hull"}


def _write_tris(path, P, T, lab, title):
    with open(path, "w") as fh:
        fh.write(f"# vtk DataFile Version 2.0\n{title}\nASCII\n")
        fh.write("DATASET UNSTRUCTURED_GRID\n")
        fh.write(f"POINTS {len(P)} double\n")
        for p in P:
            fh.write("%.9f %.9f %.9f\n" % tuple(p))
        fh.write(f"CELLS {len(T)} {4 * len(T)}\n")
        for t in T:
            fh.write("3 %d %d %d\n" % tuple(int(x) for x in t))
        fh.write(f"CELL_TYPES {len(T)}\n")
        for _ in T:
            fh.write("5\n")
        fh.write(f"CELL_DATA {len(T)}\nSCALARS patch_label int 1\n")
        fh.write("LOOKUP_TABLE default\n")
        for v in lab:
            fh.write("%d\n" % int(v))


def _write_hexes(path, P, H, arrays, title):
    with open(path, "w") as fh:
        fh.write(f"# vtk DataFile Version 2.0\n{title}\nASCII\n")
        fh.write("DATASET UNSTRUCTURED_GRID\n")
        fh.write(f"POINTS {len(P)} double\n")
        for p in P:
            fh.write("%.9f %.9f %.9f\n" % tuple(p))
        fh.write(f"CELLS {len(H)} {9 * len(H)}\n")
        for c in H:
            fh.write("8 " + " ".join(str(int(x)) for x in c) + "\n")
        fh.write(f"CELL_TYPES {len(H)}\n")
        for _ in H:
            fh.write("12\n")
        fh.write(f"CELL_DATA {len(H)}\n")
        for name, vals in arrays.items():
            isint = np.issubdtype(np.asarray(vals).dtype, np.integer)
            fh.write(f"SCALARS {name} {'int' if isint else 'double'} 1\n")
            fh.write("LOOKUP_TABLE default\n")
            for v in vals:
                fh.write(("%d\n" % v) if isint else ("%.9e\n" % v))


def _read_parts(path):
    """(points, cells, part) from a compare VTK written by compare_viz."""
    with open(path) as fh:
        L = fh.read().split("\n")
    i = next(k for k, l in enumerate(L) if l.startswith("POINTS"))
    n = int(L[i].split()[1])
    P = np.array([[float(x) for x in L[i + 1 + k].split()] for k in range(n)])
    j = next(k for k, l in enumerate(L) if l.startswith("CELLS"))
    m = int(L[j].split()[1])
    cells = [[int(x) for x in L[j + 1 + k].split()][1:] for k in range(m)]
    s = next(k for k, l in enumerate(L) if l.startswith("SCALARS"))
    part = np.array([int(float(L[s + 2 + k])) for k in range(m)])
    return P, cells, part


def main() -> int:
    ap = argparse.ArgumentParser(description="pipeline stage artifacts")
    ap.add_argument("--machine", default="machine_0034_n2000")
    ap.add_argument("--target-h", type=float, default=0.05)
    ap.add_argument("--out", default=os.path.join(ROOT, "data",
                                                  "pipeline_demo"))
    ap.add_argument("--map-dir", default="")
    args = ap.parse_args()
    name = args.machine
    base = name.rsplit("_n", 1)[0]
    out = args.out
    os.makedirs(out, exist_ok=True)
    npz = os.path.join(BATCH, name, "sample.npz")
    z = np.load(npz, allow_pickle=True)
    made = []

    # 01 geometry
    p = os.path.join(out, "01_geometry_surface.vtk")
    _write_tris(p, np.asarray(z["surface_points"], float),
                np.asarray(z["surface_tris"], np.int64),
                np.asarray(z["surface_tri_label"], np.int64),
                f"{name} stage 01 geometry: npz surface, scalar patch_label "
                f"(1 inlet, 2 outlet, 3/4 periodic, 5 hub, 6 shroud, "
                f"7 blade hull)")
    made.append(("01_geometry_surface.vtk",
                 "runner geometry as the labelled npz surface"))

    # 02 tet mesh
    tet = os.path.join(BATCH, base, f"{base}_tet.vtk")
    if os.path.exists(tet):
        shutil.copyfile(tet, os.path.join(out, "02_tet_mesh.vtk"))
        made.append(("02_tet_mesh.vtk", "gmsh tetrahedral volume mesh"))

    # 03 conditioning point cloud
    md = args.map_dir or os.path.join(ROOT, "data", f"map_batch__{name}")
    for cand in ("cloud_full_band5.00.vtk", "cloud_full_band0.00.vtk",
                 "cloud_band5.00.vtk"):
        src = os.path.join(md, cand)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(out, "03_point_cloud.vtk"))
            made.append(("03_point_cloud.vtk",
                         "conditioning cloud the transformer is given"))
            break

    # 04 AlgoHex fine hex mesh
    bv = os.path.join(BATCH, name, "blocks.vtk")
    if os.path.exists(bv):
        shutil.copyfile(bv, os.path.join(out, "04_algohex_hexmesh.vtk"))
        made.append(("04_algohex_hexmesh.vtk",
                     "AlgoHex hex mesh, scalar block_id"))

    # 05 GT block complex
    V = np.asarray(z["vertices"], float)
    B = np.asarray(z["blocks"], np.int64)
    _write_hexes(os.path.join(out, "05_block_complex_gt.vtk"), V, B,
                 {"block_id": np.arange(len(B))},
                 f"{name} stage 05 block complex (GT): {len(B)} coarse blocks")
    made.append(("05_block_complex_gt.vtk",
                 f"coarse block structure kept by the export ({len(B)} blocks)"))

    # 06 transformer blocks
    cmp_path = os.path.join(md, "compare.vtk")
    if os.path.exists(cmp_path):
        P, cells, part = _read_parts(cmp_path)
        gen = [c for c, pt in zip(cells, part) if pt == 2 and len(c) == 8]
        if gen:
            used = sorted({int(v) for c in gen for v in c})
            rm = {v: i for i, v in enumerate(used)}
            _write_hexes(os.path.join(out, "06_transformer_blocks.vtk"),
                         P[used], [[rm[int(v)] for v in c] for c in gen],
                         {"block_id": np.arange(len(gen))},
                         f"{name} stage 06 transformer-generated blocks "
                         f"({len(gen)} blocks)")
            made.append(("06_transformer_blocks.vtk",
                         f"block structure the transformer generated "
                         f"({len(gen)} blocks)"))

    # 07 GT blocks conformed and refilled
    fm = FeatureModelV2(npz, cache_dir=os.path.join(ROOT, "data", "features"))
    target = types.SimpleNamespace(curves=fm.seam_curves,
                                   surface_nearest=fm.surface_nearest)
    C = fm.vertices[fm.blocks].astype(np.float64)
    C_snap, records = snap_corners_v2(target, C, SnapConfigV2())
    stats = {"routes": 0}
    raw = _seam_path_fn(fm.seam_curves, records, stats, tol=1e-9)

    def seam(p0, p1, n):
        r = raw(p0, p1, n)
        return None if r is None else (snap_seam_path(fm.seam_curves, fm,
                                                      r[0]), r[1])

    geo = PatchPaths(fm, records=records, stats=stats,
                     is_boundary=_boundary_edge_pred(fm.blocks, C_snap))

    def path_fn(p0, p1, n):
        r = seam(p0, p1, n)
        return r if r is not None else geo(p0, p1, n)

    p7 = os.path.join(out, "07_postproc_gt_conformed.vtk")
    rep = refill_curved(C_snap, args.target_h, p7, fm=target,
                        path_fn=path_fn, write_edges=False,
                        face_project_fn=make_face_projector(geo, stats),
                        is_boundary_face=make_boundary_face_test(fm))
    made.append(("07_postproc_gt_conformed.vtk",
                 f"GT blocks conformed to the geometry and refilled "
                 f"({rep['cells_after']} cells, h={args.target_h})"))

    # 08 generated blocks conformed
    for cand in ("cfd_curved.vtk", "cfd_refill.vtk"):
        src = os.path.join(md, cand)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(
                out, "08_postproc_generated_conformed.vtk"))
            made.append(("08_postproc_generated_conformed.vtk",
                         "generated blocks conformed and refilled"))
            break

    with open(os.path.join(out, "README.md"), "w") as fh:
        fh.write(f"# Pipeline stages, {name}\n\n")
        fh.write("Load in numeric order; each file is one stage of the "
                 "chain.\n\n")
        for f, d in made:
            fh.write(f"- `{f}` -- {d}\n")
        fh.write("\nColouring hints: stage 01 by `patch_label`, 04 and 05 and "
                 "06 by `block_id`, 07 by `block_id`.\n")
    print(f"\nwrote {len(made)} stages to {out}/")
    for f, d in made:
        sz = os.path.getsize(os.path.join(out, f)) / 1e6
        print(f"   {f:38s} {sz:7.2f} MB   {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
