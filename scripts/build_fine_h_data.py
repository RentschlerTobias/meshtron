"""build_fine_h_data.py -- adaptive (target_h) curved-TFI hex dataset.

For every ``data/hex3d_algohex/batch/*/sample.npz`` machine, rebuild the
ground-truth block structure as a *curved* TFI hex mesh whose per-block
divisions are solved ADAPTIVELY for a target edge length ``h`` (default 0.5).
This is the non-uniform counterpart of ``augment_subdivide_3d.subdivide_core``:
the lever is ``tfi.solve_block_divisions`` inside
``curved_bridge._classes_counts`` (large blocks split, small blocks stay), not
a fixed n*n*n subdivision.

The pipeline is a verbatim replication of ``curved_bridge.refill_curved`` --
same ``FeatureModelV2``, same first weld, same ``_classes_counts``/
``build_structures``/``_block_curved_mesh`` calls, same FINAL weld -- minus the
VTK export and the heavy watermark/Jacobian report. The final ``tfi.weld`` is
kept (the demo's 210 pre-weld grid points weld to 122 vertices and the cells
must reference the welded points), and every index array (hex cells and
edge_index endpoints) is remapped through that final weld.

Emitted per machine:
  * a polytron-family item (plain dict) -- the primary deliverable, and
  * a matching quadtron twin (torch_geometric ``Data``).

Run:
    uv run python scripts/build_fine_h_data.py --target-h 0.5 --jobs 8 \\
        --out-polytron data/fine/polytron_data_3d_h05_from_batch.pt \\
        --out-quadtron data/fine/quadtron_data_3d_h05_from_batch.pt
"""
from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scipy.spatial import cKDTree  # noqa: E402

import curved_bridge  # noqa: E402
import edge_curves  # noqa: E402
from conditioning import point_is_band, point_is_blade  # noqa: E402
from domain_extractor_3d import (per_face_dir_class, subsample_points,  # noqa: E402
                                 to_cylindrical)
from geometry_features import FeatureModelV2  # noqa: E402

DEFAULT_OUT_POLY = ROOT / "data" / "fine" / "polytron_data_3d_h05_from_batch.pt"
DEFAULT_OUT_QUAD = ROOT / "data" / "fine" / "quadtron_data_3d_h05_from_batch.pt"
DEFAULT_BATCH = ROOT / "data" / "hex3d_algohex" / "batch"

_SELFTEST_NAME = "machine_0034_n2000"
_SELFTEST_CELLS = 48
_SELFTEST_VERTS = 122
_SELFTEST_SURF = 12539


def _build_edges(h: np.ndarray) -> np.ndarray:
    """Hex boundary surface faces [Q,4] (faces belonging to exactly one cell)."""
    from hexa_row_tokenizer import _HEX_FACE_Q
    local = np.asarray(_HEX_FACE_Q, dtype=np.int64)          # [6,4]
    allf = h[:, local].reshape(-1, 4)                        # [6N,4]
    key = np.sort(allf, axis=1)
    _uniq, inv, cnt = np.unique(key, axis=0, return_inverse=True,
                                 return_counts=True)
    return allf[cnt[inv] == 1]


def _edge_arrays(st: edge_curves.CurvedStructure, P: np.ndarray,
                 pts: np.ndarray, remap2: np.ndarray):
    """Edge index/ctrl/streamline from ``st.edge_pts``, endpoints welded to P2.

    ``st.edge_pts`` keys are first-weld (P) corner ids; their coordinates appear
    bit-identically in the pre-weld grid ``pts``. We locate each corner in
    ``pts`` and apply the FINAL weld ``remap2`` so every endpoint references the
    deduplicated vertex array.
    """
    tree_pts = cKDTree(pts)
    _, j = tree_pts.query(P)                  # P-corner -> grid index in pts
    p_to_p2 = remap2[j]                       # -> final welded vertex id
    keys = np.asarray(sorted(st.edge_pts), dtype=np.int64)   # [E,2], sorted pair
    eids = p_to_p2[keys]                      # [E,2] welded endpoints
    E = len(keys)
    ctrl = np.zeros((E, 2, 3), np.float32)
    stream = {}
    for e, k in enumerate(map(tuple, keys.tolist())):
        poly = np.asarray(st.edge_pts[k], np.float64)
        ctrl[e, 0] = poly[0]
        ctrl[e, 1] = poly[-1]
        a, b = int(eids[e, 0]), int(eids[e, 1])
        key2 = (a, b) if a < b else (b, a)
        stream[key2] = np.asarray(
            poly if a < b else poly[::-1], np.float32)
    return eids, ctrl, stream


def _boundary_quads(h: np.ndarray, P2: np.ndarray, fm, raw: dict):
    """Adaptive boundary quads + an inherited per-quad dir_class.

    Each boundary face is matched to its nearest ORIGINAL coarse surface quad
    (KDTree over the centroids of ``fm.vertices[fm.quad_faces]``). Its
    dir_class is taken from that coarse quad: the raw npz ``dir_class`` is
    per-edge, so we use the documented coarse ``quad_faces``-aligned class
    (``domain_extractor_3d.per_face_dir_class``, i.e. the class of each quad's
    first edge). If the npz ever ships ``quad_dir_class``, that is used
    verbatim instead.
    """
    quads = _build_edges(h)                                  # [Q,4]
    fq = np.asarray(raw["quad_faces"], np.int64)             # [Fc,4] coarse shell
    verts = np.asarray(raw["vertices"], float)
    if "quad_dir_class" in raw:
        coarse_dir = np.asarray(raw["quad_dir_class"], np.int64)
    else:
        lut = {(int(u), int(v)): e for e, (u, v) in enumerate(fm.edges)}
        coarse_dir = per_face_dir_class(fq, fm.edges, raw["dir_class"], lut)
    tree = cKDTree(verts[fq].mean(axis=1))
    cent = np.asarray(P2, float)[quads].mean(axis=1)
    _, nearest = tree.query(cent)
    return quads, np.asarray(coarse_dir, np.int64)[nearest]


def _build_one(npz_path: str, target_h: float, cache_dir: str | None,
               max_tri_points: int) -> dict:
    """One machine -> (polytron_item, quadtron_item). Raises on any failure."""
    import torch

    fm = FeatureModelV2(npz_path, cache_dir=cache_dir)
    name = Path(npz_path).parent.name
    raw = dict(np.load(npz_path, allow_pickle=True))

    C = np.asarray(fm.vertices[fm.blocks], np.float64)       # (nb,8,3)
    if C.ndim != 3 or C.shape[1:] != (8, 3):
        raise ValueError(f"corners shape {C.shape} != (nb,8,3)")

    tfi, _ev, bc, _cb = curved_bridge._load()
    nb = C.shape[0]
    P, remap = tfi.weld(C.reshape(-1, 3))
    H = remap.reshape(nb, 8)
    B = np.arange(nb, dtype=np.int64)
    lat = {r: (np.ones(3, int), curved_bridge._lattice_vert(H[r], tfi.CORNER))
           for r in range(nb)}
    _f2h, classes, counts, _mode, _reason = curved_bridge._classes_counts(
        tfi, bc, lat, H, B, P, target_h)
    cof = tfi.class_of_axis(classes)
    st = edge_curves.build_structures(fm, H, C, fm.blocks, counts, cof,
                                      path_fn=None)

    chunks, cells = [], []
    pts_of = {int(H[r, c]): C[r, c] for r in range(nb) for c in range(8)}
    for r in range(nb):
        Xi = curved_bridge._block_curved_mesh(tfi, r, st.dims[r], H, C, st,
                                              pts_of)
        base = sum(len(c) for c in chunks)
        ids = np.arange(base, base + Xi[..., 0].size).reshape(Xi.shape[:3])
        chunks.append(Xi.reshape(-1, 3))
        cells.append(tfi.block_cells(ids))
    pts = np.vstack(chunks)
    Hn = np.vstack(cells)
    P2, remap2 = tfi.weld(pts)
    Hn = remap2[Hn]

    eids, ctrl, stream = _edge_arrays(st, P, pts, remap2)
    quads, quad_dir = _boundary_quads(Hn, P2, fm, raw)

    surf = np.asarray(fm.surface_points, np.float64)         # full, no subsample
    tris = np.asarray(fm.surface_tris, np.int64)
    tlabel = np.asarray(fm.surface_tri_label, np.int64)
    is_blade = point_is_blade(len(surf), tris, tlabel)
    is_band = point_is_band(len(surf), tris, tlabel)

    rng = np.random.default_rng(
        int(hashlib.md5(name.encode()).hexdigest()[:8], 16))
    tri = subsample_points(surf, max_tri_points, rng)

    vp = to_cylindrical(np.asarray(P2, np.float64)).astype(np.float32)
    poly_item = {
        "name": name,
        "vertices_polar": torch.tensor(vp, dtype=torch.float32),
        "vertices_cartesian": torch.tensor(np.asarray(P2, np.float64),
                                           dtype=torch.float64),
        "faces": torch.tensor(Hn.T, dtype=torch.long),
        "edge_index": torch.tensor(eids.T, dtype=torch.long),
        "edge_ctrl": torch.tensor(ctrl, dtype=torch.float32),
        "edge_to_streamline": {
            key: torch.tensor(val, dtype=torch.float32)
            for key, val in stream.items()},
        "center": torch.tensor([0.0, 0.0, 0.0]),
        "quad_faces": torch.tensor(quads.T, dtype=torch.long),
        "surface_points": torch.tensor(surf, dtype=torch.float32),
        "is_blade": torch.tensor(is_blade, dtype=torch.bool),
        "is_band": torch.tensor(is_band, dtype=torch.bool),
        "tri_coordinates": torch.tensor(tri, dtype=torch.float32),
        "subdiv_n": None,
    }
    from torch_geometric.data import Data
    quad_item = Data(
        x=torch.tensor(np.asarray(P2, np.float32), dtype=torch.float32),
        faces=torch.tensor(quads.T, dtype=torch.long),
        tri_coordinates=torch.tensor(surf.astype(np.float32),
                                     dtype=torch.float32),
        dir_class=torch.tensor(quad_dir, dtype=torch.long),
    )
    quad_item.name = name
    return {"poly": poly_item, "quad": quad_item, "n_cells": int(len(Hn)),
            "n_verts": int(len(P2)), "n_surf": int(len(surf))}


def _worker(npz_path: str, target_h: float, cache_dir: str | None,
            max_tri_points: int):
    name = Path(npz_path).parent.name
    try:
        out = _build_one(npz_path, target_h, cache_dir, max_tri_points)
        return name, out, ""
    except Exception as exc:  # noqa: BLE001 - per-machine skip, never abort
        return name, None, f"SKIP {name}: {type(exc).__name__}: {exc}"


def _selftest(args) -> int:
    target = DEFAULT_BATCH / _SELFTEST_NAME / "sample.npz"
    print(f"[selftest] {target}", flush=True)
    name, out, msg = _worker(str(target), args.target_h, args.feature_cache,
                             args.max_tri_points)
    if out is None:
        print(f"[selftest] FAIL: {msg}")
        return 1
    n_cells, n_verts, n_surf = out["n_cells"], out["n_verts"], out["n_surf"]
    print(f"[selftest] {name}: cells={n_cells} verts={n_verts} surf={n_surf}")
    ok = True
    if n_cells != _SELFTEST_CELLS:
        print(f"[selftest] FAIL cells {n_cells} != {_SELFTEST_CELLS}")
        ok = False
    if n_verts != _SELFTEST_VERTS:
        print(f"[selftest] FAIL verts {n_verts} != {_SELFTEST_VERTS}")
        ok = False
    if n_surf != _SELFTEST_SURF:
        print(f"[selftest] FAIL surface_points {n_surf} != {_SELFTEST_SURF}")
        ok = False
    print("[selftest] PASS" if ok else "[selftest] FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-h", type=float, default=0.5)
    ap.add_argument("--jobs", type=int, default=min(mp.cpu_count(), 8))
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--feature-cache", default=str(ROOT / "data" / "features"))
    ap.add_argument("--batch", default=str(DEFAULT_BATCH))
    ap.add_argument("--out-polytron", default=str(DEFAULT_OUT_POLY))
    ap.add_argument("--out-quadtron", default=str(DEFAULT_OUT_QUAD))
    ap.add_argument("--max-tri-points", type=int, default=768)
    ap.add_argument("--selftest", action="store_true",
                    help="run only machine_0034_n2000 and assert 48/122")
    args = ap.parse_args()

    if args.selftest:
        return _selftest(args)

    srcs = sorted(str(p) for p in Path(args.batch).glob("*/sample.npz"))
    if args.limit:
        srcs = srcs[: args.limit]
    print(f"{len(srcs)} samples (target_h={args.target_h}, jobs={args.jobs})",
          flush=True)

    polytron, quadtron, failures = [], [], []
    cells_by_name = {}
    ctx = mp.get_context("fork")
    with ctx.Pool(args.jobs) as pool:
        asyncs = [pool.apply_async(_worker, (s, args.target_h,
                                             args.feature_cache,
                                             args.max_tri_points))
                  for s in srcs]
        for n, a in enumerate(asyncs, 1):
            name, out, msg = a.get()
            if msg:
                print(msg, flush=True)
                failures.append(msg)
                continue
            polytron.append(out["poly"])
            quadtron.append(out["quad"])
            cells_by_name[name] = out["n_cells"]
            if args.limit or n % 25 == 0 or n == len(asyncs):
                print(f"  [{n}/{len(srcs)}] {name}: {out['n_cells']} cells, "
                      f"{out['n_verts']} verts", flush=True)

    os.makedirs(os.path.dirname(args.out_polytron), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_quadtron), exist_ok=True)
    import torch
    torch.save(polytron, args.out_polytron)
    torch.save(quadtron, args.out_quadtron)
    print(f"\nwrote {args.out_polytron} ({len(polytron)} items)")
    print(f"wrote {args.out_quadtron} ({len(quadtron)} items)")
    print(f"ok={len(polytron)} fail={len(failures)}")
    if args.limit:
        for name, nc in cells_by_name.items():
            print(f"  {name}: {nc} cells")
    return 0 if polytron else 1


if __name__ == "__main__":
    raise SystemExit(main())
