"""detect_block_voids.py -- find unmeshed voids between blocks of a GT blocking.

A valid block structure partitions the domain: every face is either shared by
two blocks or lies on the domain boundary.  machine_0252_n8000 violates this --
blocks 10, 11 and 12 each own one unshared face around the same six corners,
the three rectangular sides of a triangular prism that no hexahedron can fill.
The result is a slit of about 0.010 (a fifth of a cell) running radially
through the passage, and a mesh volume deficit of 0.45%.

Detection is geometric, not topological, because the topological signature
("face owned by one block") is exactly the assumption the rest of the pipeline
makes about the domain boundary: refill the blocking, take the facets that
bound it, and look for PAIRS of facets from DIFFERENT blocks that lie on top of
each other with anti-parallel normals and whose midpoint is INSIDE the domain
(ray cast against the closed npz surface).  Two walls of a slit inside the
domain is a void; two walls around the blade is not, because that midpoint
falls outside.

Usage:
  uv run python scripts/detect_block_voids.py --all --out data/void_scan.json
  uv run python scripts/detect_block_voids.py --samples a,b --target-h 0.1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from meshtron.geometry.curved_bridge import refill_curved  # noqa: E402
from meshtron.geometry.geometry_features import FeatureModelV2  # noqa: E402

BATCH = os.path.join(ROOT, "data", "hex3d_algohex", "batch")
HEX_FACES = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5),
             (2, 3, 7, 6), (3, 0, 4, 7))


def inside_mask(P, T, Q, seed=0, chunk=256):
    """Ray casting against the closed npz surface."""
    rng = np.random.default_rng(seed)
    d = rng.normal(size=3)
    d /= np.linalg.norm(d)
    v0, v1, v2 = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    e1, e2 = v1 - v0, v2 - v0
    h = np.cross(d, e2)
    a = np.einsum('ij,ij->i', e1, h)
    ok = np.abs(a) > 1e-12
    inv = np.zeros_like(a)
    inv[ok] = 1.0 / a[ok]
    out = np.zeros(len(Q), bool)
    for s in range(0, len(Q), chunk):
        q = Q[s:s + chunk]
        sv = q[:, None, :] - v0[None]
        u = np.einsum('cij,ij->ci', sv, h) * inv[None]
        qv = np.cross(sv, e1[None])
        v = (qv @ d) * inv[None]
        t = np.einsum('cij,ij->ci', qv, e2) * inv[None]
        hit = (ok[None] & (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1)
               & (t > 1e-9))
        out[s:s + chunk] = (hit.sum(axis=1) % 2) == 1
    return out


def _resample(Q, n):
    s = np.concatenate([[0.0],
                        np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))])
    if s[-1] <= 1e-12:
        return np.repeat(Q[:1], n, axis=0)
    q = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(q, s, Q[:, k]) for k in range(3)], axis=1)


def _read_mesh(path):
    with open(path) as fh:
        L = fh.read().split("\n")
    i = next(k for k, l in enumerate(L) if l.startswith("POINTS"))
    n = int(L[i].split()[1])
    P = np.array([[float(x) for x in L[i + 1 + k].split()] for k in range(n)])
    j = next(k for k, l in enumerate(L) if l.startswith("CELLS"))
    m = int(L[j].split()[1])
    H = np.array([[int(x) for x in L[j + 1 + k].split()][1:] for k in range(m)])
    s = next(k for k, l in enumerate(L) if l.startswith("SCALARS block_id"))
    bid = np.array([int(float(L[s + 2 + k])) for k in range(m)])
    return P, H, bid


def surface_volume(P, T):
    v0, v1, v2 = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    return abs(np.einsum('ij,ij->i', v0, np.cross(v1, v2)).sum() / 6.0)


TETS = ((0, 1, 3, 4), (1, 2, 3, 6), (1, 3, 4, 6), (1, 4, 5, 6), (3, 4, 6, 7))


def hex_volume(P, H):
    v = 0.0
    for a, b, c, d in TETS:
        p0, p1, p2, p3 = P[H[:, a]], P[H[:, b]], P[H[:, c]], P[H[:, d]]
        m = np.stack([p1 - p0, p2 - p0, p3 - p0], axis=-1)
        v += np.abs(np.linalg.det(m) / 6.0).sum()
    return float(v)


def scan(name: str, target_h: float, tmp: str) -> dict:
    t0 = time.time()
    npz = os.path.join(BATCH, name, "sample.npz")
    z = np.load(npz, allow_pickle=True)
    SP = np.asarray(z["surface_points"], float)
    ST = np.asarray(z["surface_tris"], np.int64)
    V = np.asarray(z["vertices"], float)
    B = np.asarray(z["blocks"], np.int64)
    E = z["edges"]
    EP = z["edge_polyline"]
    OFF = z["edge_polyline_offset"]

    poly = {}
    for k, (a, b) in enumerate(E):
        Q = EP[OFF[k]:OFF[k + 1]]
        poly[(int(a), int(b))] = Q
        poly[(int(b), int(a))] = Q[::-1]
    km = {}
    for r in range(B.shape[0]):
        for c in range(8):
            km[np.round(V[B[r, c]], 9).tobytes()] = int(B[r, c])

    def gt_path(p0, p1, n):
        a = km.get(np.round(np.asarray(p0, float), 9).tobytes())
        b = km.get(np.round(np.asarray(p1, float), 9).tobytes())
        if a is None or b is None or (a, b) not in poly:
            return None
        R = _resample(np.asarray(poly[(a, b)], float), n)
        R[0], R[-1] = p0, p1
        return R, 700000

    shim = types.SimpleNamespace(curves=None, surface_nearest=None)
    refill_curved(V[B].astype(float), target_h, tmp, fm=shim,
                  path_fn=gt_path, write_edges=False)
    P, H, bid = _read_mesh(tmp)
    os.remove(tmp)

    cnt, ordr, own = {}, {}, {}
    for ci, c in enumerate(H):
        for f in HEX_FACES:
            ids = [int(c[x]) for x in f]
            k = tuple(sorted(ids))
            cnt[k] = cnt.get(k, 0) + 1
            ordr[k] = ids
            own[k] = int(bid[ci])
    bq = [k for k, v in cnt.items() if v == 1]
    q = np.array([ordr[k] for k in bq])
    ob = np.array([own[k] for k in bq])
    cen = P[q].mean(axis=1)
    hh = float(np.median(np.linalg.norm(P[q[:, 1]] - P[q[:, 0]], axis=1)))
    nr = np.cross(P[q[:, 2]] - P[q[:, 0]], P[q[:, 3]] - P[q[:, 1]])
    nr /= np.maximum(np.linalg.norm(nr, axis=1, keepdims=True), 1e-15)
    tree = cKDTree(cen)
    hits = [(a, b) for a, b in tree.query_pairs(0.3 * hh)
            if ob[a] != ob[b] and abs(float(np.dot(nr[a], nr[b]))) > 0.8]
    n_void = 0
    gap = []
    blocks = set()
    if hits:
        mid = np.array([(cen[a] + cen[b]) / 2 for a, b in hits])
        ins = inside_mask(SP, ST, mid)
        for k, (a, b) in enumerate(hits):
            if not ins[k]:
                continue
            n_void += 1
            gap.append(float(np.linalg.norm(cen[a] - cen[b])))
            blocks.add(int(ob[a]))
            blocks.add(int(ob[b]))
    vol_dom = surface_volume(SP, ST)
    vol_mesh = hex_volume(P, H)
    return {
        "name": name, "blocks": int(B.shape[0]), "cells": int(len(H)),
        "cell_size": hh,
        "void_facet_pairs": n_void,
        "void_blocks": sorted(blocks),
        "gap_p50": float(np.percentile(gap, 50)) if gap else 0.0,
        "gap_max": float(max(gap)) if gap else 0.0,
        "volume_domain": float(vol_dom), "volume_mesh": vol_mesh,
        "volume_ratio": float(vol_mesh / vol_dom),
        "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="scan blockings for voids")
    ap.add_argument("--samples", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--max-blocks", type=int, default=30)
    ap.add_argument("--target-h", type=float, default=0.1)
    ap.add_argument("--out", default=os.path.join(ROOT, "data",
                                                  "void_scan.json"))
    args = ap.parse_args()

    if args.all:
        names = []
        for d in sorted(os.listdir(BATCH)):
            p = os.path.join(BATCH, d, "sample.npz")
            if not os.path.exists(p):
                continue
            try:
                nb = int(np.load(p, allow_pickle=True)["blocks"].shape[0])
            except Exception:
                continue
            if nb <= args.max_blocks:
                names.append(d)
    else:
        names = [s for s in args.samples.split(",") if s]

    tmp = os.path.join(os.path.dirname(args.out), "_void_tmp.vtk")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows, failed = [], []
    for k, name in enumerate(names, 1):
        try:
            row = scan(name, args.target_h, tmp)
        except Exception as exc:  # noqa: BLE001
            failed.append({"name": name,
                           "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{k}/{len(names)}] {name}: ERROR {type(exc).__name__}")
            continue
        rows.append(row)
        print(f"[{k}/{len(names)}] {name:24s} blocks={row['blocks']:3d} "
              f"void_pairs={row['void_facet_pairs']:5d} "
              f"gap={row['gap_p50']:.4f} vol_ratio={row['volume_ratio']:.4f}")
    with_void = [r for r in rows if r["void_facet_pairs"] > 0]
    summary = {"n": len(rows), "with_void": len(with_void),
               "failed": failed, "target_h": args.target_h, "rows": rows}
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nsaved {args.out}")
    print(f"{len(with_void)}/{len(rows)} samples with voids, "
          f"{len(failed)} errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
