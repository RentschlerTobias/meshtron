"""compare_dtoo_npz.py — visual diff of the DTOO ground-truth walls vs the
per-machine reduced npz surface the transformer conditions on.

Purpose: answer "what exactly was cut away?" by rendering, side by side in one
VTK file:

  part 1 (DTOO) : outer skin of the DTOO .msh, one part per physical surface
  part 100     : the reduced npz surface patches (one part per npz label)

Both surfaces are also exported separately so they can be shown/hidden in
ParaView independently. Everything is triangles (VTK cell type 5) with a `part`
scalar; use Threshold (or Extract Surface) on `part` to isolate one region.

Additionally an ASCII legend (`.json`) lists, per npz label, the particle count
and the radial/z extent, plus the DTOO label scheme, so the label mismatch is
visible at a glance.

Usage:
  uv run python scripts/compare_dtoo_npz.py
  uv run python scripts/compare_dtoo_npz.py --npz data/hex3d_algohex/batch/<name>/sample.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DP3D_REPO = Path("/home/t1dde/hydrostack_pipeline/stack/domain_partition_3D")
HEX3D_REPO = DP3D_REPO / "experimentell" / "hex3d_algohex"
DEFAULT_MSH = DP3D_REPO / "data" / "T1_9" / "T1_9_ru_gridGmsh.msh"
DEFAULT_NPZ = ROOT / "data" / "hex3d_algohex" / "batch" / "machine_0034_n2000" / "sample.npz"

NPZ_SURF_NAMES = {1: "inlet", 2: "outlet", 3: "periodic_A", 4: "periodic_B",
                  5: "bl_iface_hub", 6: "bl_iface_shroud", 7: "ogrid_iface"}

DTOO_PART_BASE = 1     # 1..N DTOO surfaces
NPZ_PART_BASE = 100    # 100.. npz patches


def _write_parts_vtk(path: Path, parts, title: str) -> None:
    """Legacy ASCII VTK, one UNSTRUCTURED_GRID with POINTS + CELLS + part scalar.

    parts: list of (pts [N,3] float, cells [list[list[int]]], part_id int,
                    cell_type int). cell_type 5=tri, 4=polyline, 1=vertex.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pts_all, cells_flat, ctype, part_of = [], [], [], []
    off = 0
    for pts, cells, pid, ctype_i in parts:
        pts = np.asarray(pts, float)
        pts_all.append(pts)
        for c in cells:
            cells_flat.append([off + int(i) for i in c])
            ctype.append(ctype_i)
            part_of.append(int(pid))
        off += len(pts)
    P = np.vstack(pts_all) if pts_all else np.zeros((0, 3))

    with path.open("w") as fh:
        fh.write("# vtk DataFile Version 3.0\n")
        fh.write(f"{title}\n")
        fh.write("ASCII\nDATASET UNSTRUCTURED_GRID\n")
        fh.write(f"POINTS {len(P)} double\n")
        for p in P:
            fh.write(f"{p[0]:.9g} {p[1]:.9g} {p[2]:.9g}\n")
        fh.write(f"CELLS {len(cells_flat)} {sum(len(c) + 1 for c in cells_flat)}\n")
        for c in cells_flat:
            fh.write(f"{len(c)} " + " ".join(str(i) for i in c) + "\n")
        fh.write(f"CELL_TYPES {len(cells_flat)}\n")
        for t in ctype:
            fh.write(f"{t}\n")
        fh.write(f"CELL_DATA {len(cells_flat)}\n")
        fh.write("SCALARS part int 1\n")
        fh.write("LOOKUP_TABLE default\n")
        for v in part_of:
            fh.write(f"{v}\n")


def _tris_by_label(pts, tris, labels):
    """label -> (sub-points, sub-tris reindexed) with only used points kept."""
    out = {}
    for L in sorted({int(v) for v in np.unique(labels)}):
        T = tris[labels == L]
        if len(T) == 0:
            continue
        used = np.unique(T)
        remap = -np.ones(int(used.max()) + 1, np.int64)
        remap[used] = np.arange(len(used))
        out[L] = (pts[used], remap[T])
    return out


def surfaces_from_npz(path: Path):
    z = np.load(path)
    pts = np.asarray(z["surface_points"], float)
    tris = np.asarray(z["surface_tris"], np.int64)
    labels = np.asarray(z["surface_tri_label"])
    groups = _tris_by_label(pts, tris, labels)
    report = {}
    for L, (P, T) in groups.items():
        r = np.hypot(P[:, 0], P[:, 1])
        report[f"npz_{L}_{NPZ_SURF_NAMES.get(L, 'L' + str(L))}"] = {
            "part": NPZ_PART_BASE + L, "n_points": int(len(P)),
            "n_tris": int(len(T)),
            "r": [round(float(r.min()), 4), round(float(r.max()), 4)],
            "z": [round(float(P[:, 2].min()), 4), round(float(P[:, 2].max()), 4)],
        }
    return groups, report


def outer_skin_from_msh(path: Path):
    sys.path.insert(0, str(HEX3D_REPO))
    import tet_prep_v2 as t2  # noqa: E402 (external repo, import-only)

    nodes, elements = t2.parse_msh(path)
    tris, gid = t2.outer_boundary(nodes, elements)
    ids = t2.to_physical(nodes, tris, gid)
    used = sorted({v for tri in tris for v in tri})
    tag2i = {v: i for i, v in enumerate(used)}
    pts = np.array([nodes[v] for v in used], float)
    tris_i = np.array([[tag2i[v] for v in tri] for tri in tris], np.int64)
    return pts, tris_i, np.asarray(ids), dict(t2.tp.SURF_NAMES)


def build_compare(msh_path: Path, npz_path: Path, out: Path):
    """Kept for programmatic use: returns (report, dtoo_parts, npz_parts)."""
    n_pts, _ = surfaces_from_npz(npz_path)
    d_pts, d_tris, d_ids, dto_names = outer_skin_from_msh(msh_path)

    parts, report = [], {}
    report["scheme"] = {
        "npz_labels": {str(k): v for k, v in NPZ_SURF_NAMES.items()},
        "dtoo_labels": {str(k): v for k, v in dto_names.items()},
    }

    # --- DTOO surfaces (parts 1..N) ---
    for L, (P, T) in _tris_by_label(d_pts, d_tris, d_ids).items():
        pid = DTOO_PART_BASE + len(parts)
        parts.append((P, T.tolist(), pid, 5))
        r = np.hypot(P[:, 0], P[:, 1])
        report[f"dtoo_{L}_{dto_names.get(L, 'L' + str(L))}"] = {
            "part": pid, "n_points": int(len(P)), "n_tris": int(len(T)),
            "r": [round(float(r.min()), 4), round(float(r.max()), 4)],
            "z": [round(float(P[:, 2].min()), 4), round(float(P[:, 2].max()), 4)],
        }

    # --- npz reduced patches (parts 100..) ---
    for L, (P, T) in n_pts.items():
        pid = NPZ_PART_BASE + L
        parts.append((P, T.tolist(), pid, 5))

    _write_parts_vtk(out, parts, f"DTOO walls (parts 1..{len(dto_names)}) vs "
                                 f"reduced npz patches (parts 100..)")

    # separate files for clean side-by-side isolation
    dto_parts = [p for p in parts if p[2] < NPZ_PART_BASE]
    npz_parts = [p for p in parts if p[2] >= NPZ_PART_BASE]
    _write_parts_vtk(out.with_name(out.stem + "_dtoo.vtk"), dto_parts, "DTOO outer skin")
    _write_parts_vtk(out.with_name(out.stem + "_npz.vtk"), npz_parts, "reduced npz surface")
    return report, sorted({int(p[2]) for p in dto_parts}), sorted({int(p[2]) for p in npz_parts})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--msh", type=Path, default=DEFAULT_MSH)
    ap.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out = args.out or ROOT / "data" / "features_debug" / "dtoo_vs_npz_surfaces.vtk"
    n_pts, _ = surfaces_from_npz(args.npz)
    d_pts, d_tris, d_ids, dto_names = outer_skin_from_msh(args.msh)  # slow (~15 min)
    parts, report = [], {}
    report["scheme"] = {"npz_labels": {str(k): v for k, v in NPZ_SURF_NAMES.items()},
                        "dtoo_labels": {str(k): v for k, v in dto_names.items()}}
    for L, (P, T) in sorted(_tris_by_label(d_pts, d_tris, d_ids).items()):
        pid = DTOO_PART_BASE + L
        parts.append((P, T.tolist(), pid, 5))
        r = np.hypot(P[:, 0], P[:, 1])
        report[f"dtoo_{L}_{dto_names.get(L, 'L' + str(L))}"] = {
            "part": pid, "n_points": int(len(P)), "n_tris": int(len(T)),
            "r": [round(float(r.min()), 4), round(float(r.max()), 4)],
            "z": [round(float(P[:, 2].min()), 4), round(float(P[:, 2].max()), 4)]}
    for L, (P, T) in sorted(n_pts.items()):
        parts.append((P, T.tolist(), NPZ_PART_BASE + L, 5))
    _write_parts_vtk(out, parts, "DTOO walls vs reduced npz patches")
    _write_parts_vtk(out.with_name(out.stem + "_dtoo.vtk"),
                     [p for p in parts if p[2] < NPZ_PART_BASE], "DTOO outer skin")
    _write_parts_vtk(out.with_name(out.stem + "_npz.vtk"),
                     [p for p in parts if p[2] >= NPZ_PART_BASE], "reduced npz surface")
    with out.with_suffix(".json").open("w") as fh:
        json.dump(report, fh, indent=2)
    print(f"[compare] wrote {out} (+ _dtoo.vtk / _npz.vtk / .json)")
    for k, v in sorted(report.items(), key=lambda kv: kv[0]):
        if k == "scheme":
            continue
        print(f"  {k:32s} {json.dumps(v)}")
    print("scheme:", json.dumps(report["scheme"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
