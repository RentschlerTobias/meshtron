"""Dataset for the flow-matching geometry model (flow_geo.py): continuous block
corners + continuous Bezier edge control points, in the canonical vertex order of
the v2 generator set, plus the topology information the model is conditioned on.

Per item (same train/val split as --split-from):
  verts      [M,3]  block corners, dataset units (canonical order, = v2 it['verts'])
  conn       [B,8]  canonical blocks
  edges      [E,2]  unique undirected hex edges (a < b), canonical ids
  eoff       [E,6]  chord-relative Bezier offsets (polytron_blocks.ctrl_to_offsets),
                    continuous GT, not the quantised curve-head targets
  vlab, elab [M,7], [E,7]  multi-hot: labelled surfaces (1..7, blade = 7) the
                    vertex / the edge midpoint lies on (distance < --tol * bbox diag)
  points     [P,14] xyz normalised to [-1,1] with the dataset bounds, the v2
                    point features (r, sin, cos, z normalised) and 7 label channels
  topo       topology key (hash of conn + edges)

    python build_flow_dataset.py --split-from data/v2_canon.pt --out data/flow_canon.pt
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/opt/stack/meshtron")
sys.path.insert(0, "/opt/stack/meshtron/scripts")
from build_curve_dataset import OUT, to_canonical  # noqa: E402
from build_hexa_curve_dataset import hex_edges, load_sample  # noqa: E402
from meshtron.data.polytron_blocks import ctrl_to_offsets  # noqa: E402


def closest_dist(P, A, B, C):
    """Distance of points P [N,3] to the triangle set (A,B,C) [T,3] each -> [N].
    Closest point on triangle (Ericson, Real-Time Collision Detection 5.1.5)."""
    out = np.full(len(P), np.inf)
    for i0 in range(0, len(P), 16):
        p = P[i0:i0 + 16, None]                                     # [n,1,3]
        ab, ac, ap = B - A, C - A, p - A
        d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
        bp = p - B; d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
        cp = p - C; d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
        va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
        den = np.where(np.abs(va + vb + vc) < 1e-30, 1e-30, va + vb + vc)
        v, w = vb / den, vc / den
        q = A + ab * v[..., None] + ac * w[..., None]               # interior
        # regions (later assignments win, ordered like Ericson's early returns, reversed)
        e_bc = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)
        t = np.clip((d4 - d3) / np.where((d4 - d3) + (d5 - d6) == 0, 1e-30, (d4 - d3) + (d5 - d6)), 0, 1)
        q = np.where(e_bc[..., None], B + (C - B) * t[..., None], q)
        e_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        t = np.clip(d2 / np.where(d2 - d6 == 0, 1e-30, d2 - d6), 0, 1)
        q = np.where(e_ac[..., None], A + ac * t[..., None], q)
        q = np.where(((d6 >= 0) & (d5 <= d6))[..., None], C, q)
        e_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        t = np.clip(d1 / np.where(d1 - d3 == 0, 1e-30, d1 - d3), 0, 1)
        q = np.where(e_ab[..., None], A + ab * t[..., None], q)
        q = np.where(((d3 >= 0) & (d4 <= d3))[..., None], B, q)
        q = np.where(((d1 <= 0) & (d2 <= 0))[..., None], A, q)
        out[i0:i0 + 16] = np.linalg.norm(p - q, axis=-1).min(1)
    return out


def surface_multihot(X, raw, tol):
    Pts, tris, tl = (np.asarray(raw[k]) for k in ("surface_points", "surface_tris", "surface_tri_label"))
    m = np.zeros((len(X), 7), np.float32)
    for lab in range(1, 8):
        sel = tris[tl == lab]
        if len(sel):
            m[:, lab - 1] = closest_dist(X, Pts[sel[:, 0]], Pts[sel[:, 1]], Pts[sel[:, 2]]) < tol
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-from", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--src-root", default=OUT, help="directory with <name>/sample.npz (default: the relabelled set)")
    ap.add_argument("--tol", type=float, default=6e-3, help="on-surface tolerance, fraction of the bbox diagonal (distance histogram gap at ~10^-2.2: vertices lie on the exact geometry, not on the coarse triangulation)")
    a = ap.parse_args()
    ds = torch.load(a.split_from, weights_only=False)
    rmax = float(ds["r_bounds"][1]); z0, z1 = (float(v) for v in ds["z_bounds"])
    lo, hi = np.array([-rmax, -rmax, z0]), np.array([rmax, rmax, z1])
    out = {"train": [], "val": [], "r_bounds": ds["r_bounds"], "z_bounds": ds["z_bounds"]}
    stats = {"missing_ctrl": 0, "vert_no_label": 0, "edge_no_label": 0, "n_v": 0, "n_e": 0}
    for split in ("train", "val"):
        for it in ds[split]:
            raw = to_canonical(load_sample(os.path.join(a.src_root, it["name"], "sample.npz")), it)
            V = raw["vertices"]; conn = raw["blocks"]
            edges = hex_edges(conn)
            ctrl = {(int(p), int(q)): c for (p, q), c in zip(raw["edges"], raw["edge_ctrl"])}
            eoff = np.zeros((len(edges), 6))
            for k, (p, q) in enumerate(edges):
                if (p, q) in ctrl:
                    c = ctrl[(p, q)]; eoff[k] = ctrl_to_offsets(V[p], V[q], c[0], c[1])
                elif (q, p) in ctrl:
                    c = ctrl[(q, p)]; eoff[k] = ctrl_to_offsets(V[p], V[q], c[1], c[0])
                else:
                    stats["missing_ctrl"] += 1
            S = np.asarray(raw["surface_points"])
            tol = a.tol * np.linalg.norm(np.ptp(S, 0))
            # edge midpoint on the cubic Bezier (t = 0.5)
            B1 = (2 * V[edges[:, 0]] + V[edges[:, 1]]) / 3 + eoff[:, :3] * np.linalg.norm(V[edges[:, 1]] - V[edges[:, 0]], axis=1, keepdims=True)
            B2 = (V[edges[:, 0]] + 2 * V[edges[:, 1]]) / 3 + eoff[:, 3:] * np.linalg.norm(V[edges[:, 1]] - V[edges[:, 0]], axis=1, keepdims=True)
            mid = (V[edges[:, 0]] + 3 * B1 + 3 * B2 + V[edges[:, 1]]) / 8
            vlab, elab = surface_multihot(V, raw, tol), surface_multihot(mid, raw, tol)
            stats["vert_no_label"] += int((vlab.sum(1) == 0).sum()); stats["edge_no_label"] += int((elab.sum(1) == 0).sum())
            stats["n_v"] += len(V); stats["n_e"] += len(edges)
            # the v2 point features are built from the same raw surface points (same order)
            pf = it["points"].float().numpy()
            assert len(pf) == len(S)
            xyz = (S - lo) / (hi - lo) * 2 - 1
            topo = hashlib.sha1(conn.astype(np.int64).tobytes() + edges.astype(np.int64).tobytes()).hexdigest()[:10]
            out[split].append({"name": it["name"], "machine": it["machine"], "topo": topo,
                               "verts": torch.tensor(V, dtype=torch.float32),
                               "conn": torch.tensor(conn, dtype=torch.int16),
                               "edges": torch.tensor(edges, dtype=torch.int16),
                               "eoff": torch.tensor(eoff, dtype=torch.float32),
                               "vlab": torch.tensor(vlab), "elab": torch.tensor(elab),
                               "points": torch.tensor(np.hstack([xyz, pf]), dtype=torch.float16)})
    torch.save(out, a.out)
    topos = {}
    for it in out["train"] + out["val"]:
        topos[it["topo"]] = topos.get(it["topo"], 0) + 1
    print(f"wrote {a.out}: train {len(out['train'])} val {len(out['val'])} | topologies {topos} | {stats}")


if __name__ == "__main__":
    main()
