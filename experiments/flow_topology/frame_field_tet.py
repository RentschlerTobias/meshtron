#!/usr/bin/env python3
"""Own octahedral frame field on the AlgoHex input tet mesh (tet.vtk): boundary- and feature-aligned, smooth.

Needs only the tet mesh with its labelled boundary triangles and feature lines (the AlgoHex input), so it works
for new geometries without AlgoHex. One orthonormal frame per tet vertex (columns = the three axes):

  constraints   boundary vertex: one axis = the area-weighted surface normal (rotation about it free);
                feature-line vertex (feature segments / feature vertices of tet.vtk): one axis = the line tangent,
                a second axis = the normal component orthogonal to it
  smoothing     Jacobi relaxation: every vertex becomes the polar projection of (own frame + sum of neighbour
                frames), each neighbour first matched to it by the best of the 24 proper cube symmetries
                (octahedral symmetry, as in field-aligned meshing); constraints re-imposed after each sweep
  lookup        frame at a point = frame of the nearest vertex; the alignment / tracing code (frame_field.py,
                field_lines.py) matches axes per neighbour, so the symmetry needs no global consistency

    python frame_field_tet.py --name tistos_s1_0131 --out viz/tistos_s1_0131_perm
"""
import argparse
import itertools
import os

import meshio
import numpy as np
from scipy.spatial import cKDTree

RUNS = "/opt/stack/meshtron/experiments/flow_topology/datagen/runs"


def cube_group():
    """The 24 proper signed permutation matrices."""
    G = []
    for p in itertools.permutations(range(3)):
        for s in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for i in range(3):
                M[p[i], i] = s[i]
            if np.linalg.det(M) > 0:
                G.append(M)
    return np.asarray(G)


def polar(S):
    U, _, Vt = np.linalg.svd(S)
    R = U @ Vt
    neg = np.linalg.det(R) < 0
    if neg.any():
        U[neg, :, -1] *= -1
        R[neg] = U[neg] @ Vt[neg]
    return R


def align_axis(R, d, mask):
    """For the frames R[mask]: set the axis best aligned with d[mask] (unit) to +-d, re-orthonormalise the rest."""
    if not mask.any():
        return R
    Rm, dm = R[mask], d[mask]
    c = np.einsum("nij,ni->nj", Rm, dm)
    j = np.abs(c).argmax(1)
    out = Rm.copy()
    rows = np.arange(len(Rm))
    a0 = dm * np.sign(c[rows, j])[:, None]
    k1, k2 = (j + 1) % 3, (j + 2) % 3
    b = Rm[rows, :, k1] - (np.einsum("ni,ni->n", Rm[rows, :, k1], a0))[:, None] * a0
    b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    cvec = np.cross(a0, b)
    out[rows, :, j] = a0; out[rows, :, k1] = b; out[rows, :, k2] = cvec
    R = R.copy(); R[mask] = out
    return R


def align_two(R, t, n, mask):
    """Feature vertices: one axis = t, a second = component of n orthogonal to t."""
    R = align_axis(R, t, mask)
    n2 = n - (np.einsum("ni,ni->n", n, t))[:, None] * t
    ok = mask & (np.linalg.norm(n2, axis=1) > 1e-6)
    n2[ok] /= np.linalg.norm(n2[ok], axis=1, keepdims=True)
    Rm, tm, nm = R[ok], t[ok], n2[ok]
    rows = np.arange(len(Rm))
    ct = np.abs(np.einsum("nij,ni->nj", Rm, tm)).argmax(1)
    cn = np.einsum("nij,ni->nj", Rm, nm); cn[rows, ct] = 0
    j = np.abs(cn).argmax(1)
    a1 = nm * np.sign(cn[rows, j])[:, None]
    out = Rm.copy()
    out[rows, :, j] = a1
    k = 3 - ct - j
    third = np.cross(out[rows, :, ct], a1)
    third *= np.sign(np.einsum("ni,ni->n", third, Rm[rows, :, k]) + 1e-12)[:, None]
    out[rows, :, k] = third
    out = polar(out)
    R = R.copy(); R[ok] = out
    return R


class TetFrameField:
    def __init__(self, tet_vtk, iters=400, verbose=True):
        m = meshio.read(tet_vtk)
        P = np.asarray(m.points, float)
        cells = {c.type: c.data for c in m.cells}
        T = cells["tetra"]
        tri = cells.get("triangle", np.zeros((0, 3), int))
        seg = cells.get("line", np.zeros((0, 2), int))
        self.P = P
        # boundary normals per vertex AND patch (area weighted): averaging across a crease gives a direction that
        # belongs to neither surface, so creases get their own constraint below
        lab = None
        for k, v in m.cell_data.items():
            for c, d in zip(m.cells, v):
                if c.type == "triangle":
                    lab = np.asarray(d).ravel()
        fn = np.cross(P[tri[:, 1]] - P[tri[:, 0]], P[tri[:, 2]] - P[tri[:, 0]])
        nrm = np.zeros_like(P)
        for k in range(3):
            np.add.at(nrm, tri[:, k], fn)
        bnd = np.linalg.norm(nrm, axis=1) > 0
        nrm[bnd] /= np.linalg.norm(nrm[bnd], axis=1, keepdims=True)
        patch_n = {}                                         # (vertex, label) -> summed normal
        for t_, (a_, b_, c_) in enumerate(tri):
            for v in (a_, b_, c_):
                key = (int(v), int(lab[t_]) if lab is not None else 0)
                patch_n[key] = patch_n.get(key, 0) + fn[t_]
        per_v = {}
        for (v, l), n in patch_n.items():
            per_v.setdefault(v, []).append(n / max(np.linalg.norm(n), 1e-30))
        crease = np.zeros(len(P), bool); corner = np.zeros(len(P), bool)
        n1 = nrm.copy(); n2 = nrm.copy()
        for v, ns in per_v.items():
            if len(ns) < 2:
                continue
            # pairwise most different patch normals
            c_, i, j = min(((abs(ns[i] @ ns[j]), i, j) for i in range(len(ns)) for j in range(i + 1, len(ns))))
            if c_ < np.cos(np.radians(30)):                 # a real crease between two patches
                n1[v], n2[v] = ns[i], ns[j]
                crease[v] = True
                if len(ns) >= 3:
                    corner[v] = True
        self.crease, self.corner = crease, corner
        tan = np.zeros_like(P)
        sd = P[seg[:, 1]] - P[seg[:, 0]]
        sd /= np.maximum(np.linalg.norm(sd, axis=1, keepdims=True), 1e-12)
        for a_, b_ in ((0, 1),):
            pass
        # accumulate tangents sign-consistently: flip to agree with the first one seen per vertex
        first = np.zeros_like(P)
        for (i, j), d in zip(seg, sd):
            for v in (i, j):
                if not first[v].any():
                    first[v] = d
                tan[v] += d if d @ first[v] >= 0 else -d
        feat = np.linalg.norm(tan, axis=1) > 0
        tan[feat] /= np.linalg.norm(tan[feat], axis=1, keepdims=True)
        # creases: the crease tangent n1 x n2 replaces / complements the tagged feature-line tangent
        ct = np.cross(n1, n2); ok = crease & (np.linalg.norm(ct, axis=1) > 1e-6)
        ct[ok] /= np.linalg.norm(ct[ok], axis=1, keepdims=True)
        tan[ok & ~feat] = ct[ok & ~feat]
        feat = feat | ok
        nrm[crease] = n1[crease]
        self.bnd, self.feat = bnd, feat
        # unique tet edges
        E = np.concatenate([T[:, [i, j]] for i, j in itertools.combinations(range(4), 2)])
        E = np.unique(np.sort(E, 1), axis=0)
        self.E = E
        G = cube_group()
        R = np.repeat(np.eye(3)[None], len(P), 0)
        # corners (>= 3 patches): frame fixed from two patch normals
        def fix_corners(R):
            if corner.any():
                a0 = n1[corner]; b0 = n2[corner] - (np.einsum("ni,ni->n", n2[corner], a0))[:, None] * a0
                b0 /= np.maximum(np.linalg.norm(b0, axis=1, keepdims=True), 1e-12)
                R = R.copy(); R[corner] = np.stack([a0, b0, np.cross(a0, b0)], 2)
            return R
        R = align_axis(R, nrm, bnd & ~feat)
        R = align_two(R, tan, nrm, feat)
        R = fix_corners(R)
        for it in range(iters):
            Mij = np.einsum("eki,ekj->eij", R[E[:, 0]], R[E[:, 1]])         # R_i^T R_j
            score = np.einsum("gab,eab->eg", G, Mij)                       # trace(G^T M)
            g = score.argmax(1)
            Rj_to_i = np.einsum("eab,ebc->eac", R[E[:, 1]], np.transpose(G[g], (0, 2, 1)))
            Ri_to_j = np.einsum("eab,ebc->eac", R[E[:, 0]], G[g])
            S = R.copy()
            np.add.at(S, E[:, 0], Rj_to_i)
            np.add.at(S, E[:, 1], Ri_to_j)
            R = polar(S)
            R = align_axis(R, nrm, bnd & ~feat)
            R = align_two(R, tan, nrm, feat)
            R = fix_corners(R)
            if verbose and (it % 100 == 0 or it == iters - 1):
                energy = float((3 - score.max(1)).mean())
                print(f"[tetff] iter {it:4d} mean edge misfit {energy:.4f}", flush=True)
        self.R = R
        self.tree = cKDTree(P)
        self.size = np.full(len(P), float(np.median(np.linalg.norm(P[E[:, 1]] - P[E[:, 0]], axis=1))))
        self.c = P
        # singular edges: frame mismatch around tet edges is not tracked; report the edge misfit instead
        Mij = np.einsum("eki,ekj->eij", R[E[:, 0]], R[E[:, 1]])
        self.edge_misfit = 3 - np.einsum("gab,eab->eg", G, Mij).max(1)
        vm = np.zeros(len(P)); np.maximum.at(vm, E[:, 0], self.edge_misfit); np.maximum.at(vm, E[:, 1], self.edge_misfit)
        self.vertex_misfit_deg = np.degrees(np.arccos(np.clip((3 - vm - 1) / 2, -1, 1)))   # largest neighbour angle

    def frames(self, q):
        return self.R[self.tree.query(np.asarray(q, float).reshape(-1, 3))[1]]

    def misalign(self, Pl):
        t = np.diff(Pl, axis=0); t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-30)
        R = self.frames((Pl[1:] + Pl[:-1]) / 2)
        c = np.abs(np.einsum("ni,nij->nj", t, R)).max(1)
        return np.degrees(np.arccos(np.clip(c, -1, 1)))

    def write_vtk(self, path, every=1):
        sel = np.arange(0, len(self.P), every)
        pd = {f"axis{i}": self.R[sel][:, :, i] * self.size[sel, None] * 0.5 for i in range(3)}
        pd["constraint"] = (self.bnd[sel].astype(int) + self.feat[sel].astype(int) + self.corner[sel].astype(int))
        pd["misfit_deg"] = self.vertex_misfit_deg[sel]
        meshio.Mesh(self.P[sel], [("vertex", np.arange(len(sel))[:, None])], point_data=pd).write(path, binary=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=400)
    a = ap.parse_args()
    import time
    t0 = time.time()
    ff = TetFrameField(f"{RUNS}/{a.name}/tet.vtk", a.iters)
    print(f"[tetff] {len(ff.P)} vertices ({int(ff.bnd.sum())} boundary, {int(ff.feat.sum())} on feature lines / creases, "
          f"{int(ff.corner.sum())} corners), "
          f"{len(ff.E)} edges, {time.time() - t0:.0f}s; edges with misfit > 0.5: {int((ff.edge_misfit > 0.5).sum())}")
    os.makedirs(a.out, exist_ok=True)
    ff.write_vtk(f"{a.out}/14_tet_framefield.vtk")
    np.savez_compressed(f"{a.out}/14_tet_framefield.npz", P=ff.P, R=ff.R, bnd=ff.bnd, feat=ff.feat)
    # comparison with the field realised by the AlgoHex hex mesh and alignment of the AlgoHex grid lines
    from frame_field import FrameField
    hf = FrameField(f"{RUNS}/{a.name}/hex_hex_x.ovm")
    Rh = hf.R; Rt = ff.frames(hf.c)
    G = cube_group()
    tr = np.einsum("gab,nab->ng", G, np.einsum("nki,nkj->nij", Rt, Rh)).max(1)       # best trace
    ang = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))
    print(f"[tetff] frame angle to the AlgoHex hex-mesh frames: median {np.median(ang):.1f} deg, p90 {np.percentile(ang, 90):.1f}")
    s = np.load(f"{RUNS}/{a.name}/sample.npz")
    E, poly, off = s["edges"], s["edge_polyline"], s["edge_polyline_offset"]
    keep = [k for k, (p, q) in enumerate(E) if p < q]
    for name, f in (("tet field", ff), ("AlgoHex hex field", hf)):
        m = np.array([f.misalign(poly[off[k]:off[k + 1]]).mean() for k in keep])
        print(f"[tetff] GT AlgoHex polylines vs {name:18s}: mean misalignment median {np.median(m):.1f} deg, p90 {np.percentile(m, 90):.1f}")


if __name__ == "__main__":
    main()
