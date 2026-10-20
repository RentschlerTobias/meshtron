#!/usr/bin/env python3
"""Block edges traced as field lines of the discrete AlgoHex frame field (frame_field.py), as VTK.

Per block edge a -> b: start along the frame axis closest to the chord direction and integrate the field
(direction at a point = inverse-distance mean of the k nearest cell frames, each contributing its axis best
aligned with the current direction, so the octahedral symmetry needs no matching); a pull towards the target
that grows with the progress (weight s^2) makes the line arrive. Traced from both ends and blended
(u * forward + (1 - u) * reversed backward), so the result does not depend on the edge direction.

Written for the GT block structure (sample.npz) and for the curves of a viz_pipeline run, next to the
curves they are compared with.

  13_fieldlines.vtk   cell data 'set': 0 GT AlgoHex polyline, 1 GT field line, 2 predicted curve (06 arcs),
                      3 predicted field line; 'edge': edge index within its set; 'dev': max distance of the
                      curve to its field line / chord (on the curve cells, sets 0 and 2)

    python field_lines.py --name tistos_s1_0131 --viz viz/tistos_s1_0131_perm
"""
import argparse
import json
import os

import meshio
import numpy as np
from scipy.spatial import cKDTree

from frame_field import RUNS, FrameField, lines


def field_dir(ff, p, d, k=6):
    dist, idx = ff.tree.query(p, k=k)
    w = 1.0 / np.maximum(dist, 1e-9)
    acc = np.zeros(3)
    for wi, R in zip(w, ff.R[idx]):
        c = R.T @ d                                       # cos to the three axes
        j = int(np.argmax(np.abs(c)))
        acc += wi * np.sign(c[j]) * R[:, j]
    n = np.linalg.norm(acc)
    return acc / n if n > 1e-12 else d


def trace(ff, a, b, n_steps=48):
    L = np.linalg.norm(b - a)
    h = L / n_steps
    R0 = ff.frames(a)[0]
    c = R0.T @ ((b - a) / L); j = int(np.argmax(np.abs(c)))
    d = np.sign(c[j]) * R0[:, j]
    P = [a.copy()]; p = a.copy(); s = 0.0
    for _ in range(4 * n_steps):
        to_b = b - p; r = np.linalg.norm(to_b)
        if r < 1.5 * h:
            break
        d = field_dir(ff, p, d)
        w = min(1.0, (s / L) ** 2)
        d = (1 - w) * d + w * to_b / r
        d /= np.linalg.norm(d)
        p = p + h * d; s += h; P.append(p.copy())
    P.append(b.copy())
    return np.asarray(P)


def resample(P, n):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    q = np.linspace(0, s[-1], n)
    return np.stack([np.interp(q, s, P[:, k]) for k in range(3)], 1)


def field_edge(ff, a, b, n=65):
    F = resample(trace(ff, a, b), n)
    B = resample(trace(ff, b, a), n)[::-1]
    u = np.linspace(0, 1, n)[:, None]
    return (1 - u) * F + u * B


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--viz", required=True, help="viz_pipeline output dir (06_curves_arcs.vtk)")
    ap.add_argument("--field", choices=("hex", "tet"), default="hex",
                    help="hex: field of the AlgoHex hex mesh (frame_field.py); tet: own field on tet.vtk (frame_field_tet.py)")
    a = ap.parse_args()
    if a.field == "tet":
        from frame_field_tet import TetFrameField
        ff = TetFrameField(f"{RUNS}/{a.name}/tet.vtk", verbose=False)
    else:
        ff = FrameField(f"{RUNS}/{a.name}/hex_hex_x.ovm")
    s = np.load(f"{RUNS}/{a.name}/sample.npz")
    V, E, poly, off = s["vertices"], s["edges"], s["edge_polyline"], s["edge_polyline_offset"]
    sets = []                                              # (set id, edge id, polyline, dev)
    gt_dev = []
    for k, (p, q) in enumerate(E):
        if p >= q:
            continue
        Pl = poly[off[k]:off[k + 1]]
        Fl = field_edge(ff, V[p], V[q])
        dv = cKDTree(Fl).query(Pl)[0].max() / np.linalg.norm(V[q] - V[p])
        sets += [(0, k, Pl, dv), (1, k, Fl, 0.0)]; gt_dev.append(dv)
    ls, meta = lines(f"{a.viz}/06_curves_arcs.vtk")
    pr_dev = []
    for i, (P, m) in enumerate(zip(ls, meta)):
        Fl = field_edge(ff, P[0], P[-1])
        dv = cKDTree(Fl).query(P)[0].max() / np.linalg.norm(P[-1] - P[0])
        sets += [(2, i, P, dv), (3, i, Fl, 0.0)]; pr_dev.append((dv, i, m.get("kind", -1), P[len(P) // 2]))
    pts, cells, sid, eid, dev = [], [], [], [], []
    base = 0
    for st, k, P, dv in sets:
        pts.append(P); idx = base + np.arange(len(P)); base += len(P)
        seg = np.stack([idx[:-1], idx[1:]], 1); cells.append(seg)
        sid += [st] * len(seg); eid += [k] * len(seg); dev += [dv] * len(seg)
    out = f"{a.viz}/13_fieldlines.vtk" if a.field == "hex" else f"{a.viz}/15_fieldlines_tet.vtk"
    meshio.Mesh(np.concatenate(pts), [("line", np.concatenate(cells))],
                cell_data={"set": [np.array(sid)], "edge": [np.array(eid)], "dev": [np.array(dev)]}).write(out, binary=True)
    if a.field == "hex":
        ff.write_vtk(f"{a.viz}/12_framefield.vtk")
    gd = np.array(gt_dev); pd = np.array([x[0] for x in pr_dev])
    print(f"[fieldlines] {out}")
    print(f"GT AlgoHex polylines vs field lines: dev/chord median {np.median(gd):.3f} p90 {np.percentile(gd, 90):.3f} max {gd.max():.3f}")
    print(f"predicted curves      vs field lines: dev/chord median {np.median(pd):.3f} p90 {np.percentile(pd, 90):.3f} max {pd.max():.3f}")
    mid = np.array([0.574, 0.172, 0.839])
    hub = min(pr_dev, key=lambda x: np.linalg.norm(x[3] - mid))
    print(f"hub edge (predicted edge {hub[1]}, kind {hub[2]}): dev/chord {hub[0]:.3f}")
    top = sorted(pr_dev, key=lambda x: -x[0])[:6]
    print("largest deviations (predicted edge, kind 0 cubic / 2 seam arc, dev/chord, midpoint):")
    for dv, i, kd, m in top:
        print(f"  edge {i:3d} kind {kd} dev {dv:.3f} mid {np.round(m, 3)}")
    json.dump({"gt_dev": gd.tolist(), "pred_dev": pd.tolist()}, open(out.replace(".vtk", ".json"), "w"))


if __name__ == "__main__":
    main()
