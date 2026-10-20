#!/usr/bin/env python3
"""Discrete frame field from the raw AlgoHex hex mesh (hex_hex_x.ovm) and edge-alignment diagnostics.

The fine AlgoHex hex mesh realises the frame field AlgoHex computed (its grid lines are integral curves of the
field). Per hex cell the three axis directions are the means of its four parallel edges, orthonormalised
(polar decomposition); the field at any point is the frame of the nearest cell centre. Octahedral symmetry is
handled by the alignment measure itself: angle(t) = arccos(max_i |t . a_i|) for a unit tangent t, so no axis
matching between cells is needed.

    python frame_field.py --name tistos_s1_0131 --viz viz/tistos_s1_0131 viz/tistos_s1_0131_perm
"""
import argparse
import json
import os
import sys

import meshio
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/opt/stack/domain_partition_3D/experimentell/hex3d_algohex")
import ovm_io  # noqa: E402

RUNS = "/work/datagen_full/datagen/runs"
# VTK hex corner order: parallel edge groups per local axis
AXIS_EDGES = (((0, 1), (3, 2), (4, 5), (7, 6)),
              ((0, 3), (1, 2), (4, 7), (5, 6)),
              ((0, 4), (1, 5), (2, 6), (3, 7)))


class FrameField:
    def __init__(self, ovm_path):
        pts, edges, faces, polys = ovm_io.read_ovm(ovm_path)
        H, skipped = ovm_io.ovm_to_cells(pts, edges, faces, polys)
        self.P, self.H, self.skipped = np.asarray(pts, float), H, skipped
        X = self.P[H]                                               # [C,8,3]
        A = np.stack([np.mean([X[:, j] - X[:, i] for i, j in grp], 0) for grp in AXIS_EDGES], 2)   # [C,3,3] cols
        U, _, Vt = np.linalg.svd(A)
        self.R = U @ Vt                                             # nearest orthonormal frame (columns = axes)
        self.size = np.linalg.norm(A, axis=1).mean(1)               # mean cell edge length
        self.c = X.mean(1)
        self.tree = cKDTree(self.c)

    def frames(self, q):
        return self.R[self.tree.query(np.asarray(q, float).reshape(-1, 3))[1]]

    def misalign(self, P):
        """Per-segment misalignment angle (deg) of a polyline P [n,3] against the field at segment midpoints."""
        t = np.diff(P, axis=0); t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-30)
        R = self.frames((P[1:] + P[:-1]) / 2)
        c = np.abs(np.einsum("ni,nij->nj", t, R)).max(1)
        return np.degrees(np.arccos(np.clip(c, -1, 1)))

    def write_vtk(self, path, every=1):
        sel = np.arange(0, len(self.c), every)
        pd = {f"axis{i}": self.R[sel][:, :, i] * self.size[sel, None] * 0.5 for i in range(3)}
        meshio.Mesh(self.c[sel], [("vertex", np.arange(len(sel))[:, None])], point_data=pd).write(path, binary=True)


def lines(path):
    m = meshio.read(path)
    L = np.concatenate([c.data for c in m.cells if c.type == "line"])
    cd = {k: np.concatenate([d for d, c in zip(m.cell_data[k], m.cells) if c.type == "line"]) for k in m.cell_data}
    out, cur, meta = [], [L[0, 0], L[0, 1]], []
    for i in range(1, len(L)):
        if L[i, 0] == cur[-1]:
            cur.append(L[i, 1])
        else:
            out.append(m.points[cur]); meta.append({k: int(cd[k][i - 1]) for k in cd}); cur = [L[i, 0], L[i, 1]]
    out.append(m.points[cur]); meta.append({k: int(cd[k][-1]) for k in cd})
    return out, meta


def summary(ff, curves, label):
    per = [ff.misalign(P) for P in curves]
    mean = np.array([a.mean() for a in per]); mx = np.array([a.max() for a in per])
    return {"set": label, "n_edges": len(per), "mean_deg_median": float(np.median(mean)),
            "mean_deg_p90": float(np.percentile(mean, 90)), "max_deg_median": float(np.median(mx)),
            "max_deg_p90": float(np.percentile(mx, 90)), "per_edge_mean": mean.tolist()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--viz", nargs="*", default=[], help="viz_pipeline output dirs to evaluate")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    R = f"{RUNS}/{a.name}"
    ff = FrameField(f"{R}/hex_hex_x.ovm")
    s = np.load(f"{R}/sample.npz")
    d, _ = cKDTree(ff.P).query(s["vertices"])
    print(f"[ff] {a.name}: {len(ff.H)} hex cells ({ff.skipped} skipped), cell size median {np.median(ff.size):.4f}; "
          f"block corners to nearest hex node: median {np.median(d):.2e}, max {d.max():.2e}")
    out = a.out or (a.viz[0] if a.viz else ".")
    os.makedirs(out, exist_ok=True)
    ff.write_vtk(f"{out}/12_framefield.vtk")
    res = []
    # ground truth: AlgoHex polylines (grid lines, ~ field lines) and the exported cubics
    E, ctrl, poly, off = s["edges"], s["edge_ctrl"], s["edge_polyline"], s["edge_polyline_offset"]
    V = s["vertices"]; t = np.linspace(0, 1, 33)[:, None]
    keep = [k for k, (p, q) in enumerate(E) if p < q]
    res.append(summary(ff, [poly[off[k]:off[k + 1]] for k in keep], "GT AlgoHex polylines"))
    res.append(summary(ff, [(1 - t) ** 3 * V[E[k, 0]] + 3 * (1 - t) ** 2 * t * ctrl[k][0] + 3 * (1 - t) * t ** 2 * ctrl[k][1]
                            + t ** 3 * V[E[k, 1]] for k in keep], "GT export cubics"))
    for vd in a.viz:
        for f in ("05_curves_stage3.vtk", "06_curves_arcs.vtk", "08_curves_opt.vtk"):
            if os.path.exists(f"{vd}/{f}"):
                ls, _ = lines(f"{vd}/{f}")
                res.append(summary(ff, ls, f"{os.path.basename(vd)}/{f[:-4]}"))
    print(f"{'curve set':42s} {'edges':>5s} | mean misalignment: median  p90 | max per edge: median  p90")
    for r in res:
        print(f"{r['set']:42s} {r['n_edges']:5d} | {r['mean_deg_median']:24.1f} {r['mean_deg_p90']:5.1f} | "
              f"{r['max_deg_median']:19.1f} {r['max_deg_p90']:5.1f}")
    # the hub edge of interest (nearest to the strongly curved predicted edge)
    mid = np.array([0.574, 0.172, 0.839])
    for vd in a.viz:
        ls, _ = lines(f"{vd}/05_curves_stage3.vtk")
        P = min(ls, key=lambda Q: np.linalg.norm(Q[len(Q) // 2] - mid))
        m = ff.misalign(P)
        print(f"hub edge in {os.path.basename(vd)}/05: mean {m.mean():.1f} deg, max {m.max():.1f} deg")
    k = min(keep, key=lambda k: np.linalg.norm(poly[(off[k] + off[k + 1]) // 2] - mid))
    m = ff.misalign(poly[off[k]:off[k + 1]])
    print(f"hub edge GT AlgoHex polyline (edge {k}): mean {m.mean():.1f} deg, max {m.max():.1f} deg")
    json.dump(res, open(f"{out}/12_framefield_alignment.json", "w"), indent=1)


if __name__ == "__main__":
    main()
