#!/usr/bin/env python3
"""Block-level curve optimisation through the TFI (step 2 of HANDOFF_2026-10-12).

Per block structure (corners fixed, after snapping):
  1. feature edges (continuity.detect) become the exact seam arc of the input geometry between their
     two corners: dense polylines handed to the TFI as they are, G1 wherever the seam is smooth
     (a single cubic cannot follow the LE/TE hairpins the O-ring corners sit on);
  2. the remaining edges keep their cubic and their control points are optimised through a
     differentiable edge-based Gordon-Hall TFI (mesh nodes are linear in the edge points):
       quality   hinge on the corner scaled Jacobian of the surrogate cells
       surface   boundary edges on their patch (targets re-projected every few steps)
       bending   1 - cos(kink) for every 2d/3d continuity pair
       data      distance of the control points to the input (prediction or GT)
The result is evaluated with the real fill (refill_polytron + surface projection) and the same mesh
metrics as infer_chain_flow.py, plus the kink at every continuity pair measured on the curves that
go into the fill.

    PYTHONPATH=<dp3d hex3d_algohex> python curve_opt.py --split val_new --src-root /work/datagen_papaya/runs \
        --ckpt runs/flowp_flow_all_s0/last.pt --out-dir chain/curve_opt_valnew
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, "/opt/stack/meshtron")
sys.path.insert(0, "/work/meshtron_v2")
sys.path.insert(0, "/work/rl_reward")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from infer_chain_v2 import snap_all  # noqa: E402  (before meshtron.geometry, like infer_chain_flow.py)
import continuity as C  # noqa: E402
from meshtron.data.polytron_blocks import CORNERS, HEX_EDGES, bezier, offsets_to_ctrl  # noqa: E402
from meshtron.geometry.geometry_features import FeatureModelV2  # noqa: E402
from meshtron.geometry.polytron_tfi import SurfaceProjector, read_hex_vtk, refill_polytron, surface_fit  # noqa: E402
from mesh_quality import from_vtk, r_mesh  # noqa: E402
from rewards_v2 import _CORNER_NB  # noqa: E402

VARIANTS = ("pred", "pred_seam", "pred_opt", "gt", "gt_seam", "gt_opt")
# also available: pred_gar, gt_gar (seam arcs + Garanzha energy instead of the hinge)


# ------------------------------------------------------------------ feature arcs
class SmoothSeams:
    """Seam curves of the input triangulation as interpolating cubic splines (C2, through every
    seam node), sampled `refine` times denser than the nodes. The raw seam polyline turns by up
    to ~40 deg per node at tight LE/TE hairpins; the geometry it samples is smooth."""

    def __init__(self, cs, refine=16):
        from scipy.interpolate import splev, splprep
        from scipy.spatial import cKDTree
        self.pts, self.arclen, self.closed = [], [], []
        for c in range(cs.n_curves):
            P = cs.segment(c)
            closed = bool(cs.closed[c])
            if closed and np.linalg.norm(P[0] - P[-1]) < 1e-12:
                P = P[:-1]
            keep = np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-12]
            P = P[keep]
            if len(P) >= 4:
                u = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
                if closed:
                    u = np.r_[u, u[-1] + np.linalg.norm(P[0] - P[-1])]
                    P = np.vstack([P, P[:1]])
                tck, _ = splprep(P.T, u=u, s=0, per=int(closed), k=3)
                Q = np.stack(splev(np.linspace(0, u[-1], refine * (len(P) - 1) + 1), tck), 1)
            else:
                Q = P
            self.pts.append(Q); self.closed.append(closed)
            self.arclen.append(np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))])
        self.trees = [cKDTree(Q) for Q in self.pts]

    def on_curve(self, c, q):
        """Distance, arclength and point of the nearest sample of curve c."""
        d, j = self.trees[c].query(np.asarray(q, float).reshape(-1, 3))
        return d, self.arclen[c][j], self.pts[c][j]

    def edge_curve(self, pa, pb, hint, tol):
        """The curve both endpoints lie on (distance <= tol); among several (at junctions) the one
        nearest the edge midpoint `hint`. The midpoint is not required to be near: an input cubic
        may cut across a hairpin of the seam. None if no curve carries both endpoints."""
        best = None
        for c in range(len(self.pts)):
            d = self.on_curve(c, np.stack([pa, pb, hint]))[0]
            if max(d[0], d[1]) <= tol and (best is None or d[2] < best[0]):
                best = (float(d[2]), c)
        return None if best is None else best[1]


def seam_arc(cs, c, pa, pb, hint, n):
    """Curve c of cs (SmoothSeams) from pa to pb, n points by arclength; on a closed curve the arc
    whose midpoint is nearer `hint`."""
    s, P0 = cs.arclen[c], cs.pts[c]
    _, t, _ = cs.on_curve(c, np.stack([pa, pb]))
    L = float(s[-1]); ta, tb = float(t[0]), float(t[1])
    arcs = [np.linspace(ta, tb, n)]
    if cs.closed[c]:
        arcs.append(np.linspace(ta, tb + L if tb < ta else tb - L, n) % L)
    best = None
    for q in arcs:
        P = np.stack([np.interp(q, s, P0[:, k]) for k in range(3)], 1)
        dm = np.linalg.norm(P[n // 2] - hint)
        if best is None or dm < best[0]:
            best = (dm, P)
    P = best[1]
    P[0], P[-1] = pa, pb
    return P


def feature_arcs(V, E, c1, c2, seams, det, n=257, tol_frac=0.2):
    """Feature edges -> exact (smoothed) seam arcs. Two passes: assign each feature edge to the seam curve
    carrying both endpoints, move vertices inside a feature line (not corners) onto their curve, assign
    again (an input vertex may sit a few percent of a short edge off the seam). Corners stay.
    Returns (V', {edge: arc})."""
    V = V.copy()
    cur_of = {}
    for _ in range(2):
        for k in sorted(det["feat"]):
            a, b = E[k]
            L = np.linalg.norm(V[b] - V[a])
            c = seams.edge_curve(V[a], V[b], bezier(V[a], V[b], c1[k], c2[k], 3)[1], tol_frac * L)
            if c is not None:
                cur_of[k] = c
        for v in {int(x) for k in cur_of for x in E[k]} - det["corner"]:
            cs = {cur_of[k] for k in cur_of if v in E[k]}
            if len(cs) == 1:
                V[v] = seams.on_curve(cs.pop(), V[v])[2][0]
    out = {}
    for k, c in cur_of.items():
        a, b = E[k]
        out[k] = seam_arc(seams, c, V[a], V[b], bezier(V[a], V[b], c1[k], c2[k], 3)[1], n)
    return V, out


def to_curves(V, E, c1, c2, arcs, n=257):
    """{(a,b) a<b: polyline}: feature arcs where given, cubic otherwise."""
    cur = {}
    for k, (a, b) in enumerate(E):
        P = arcs[k] if k in arcs else bezier(V[a], V[b], c1[k], c2[k], n)
        cur[(int(a), int(b)) if a < b else (int(b), int(a))] = P if a < b else P[::-1]
    return cur


def poly_kink(cur, v, ea, eb, frac=0.03):
    """Kink (deg) at v between two polylines (edge vertex pairs ea, eb), tangents over frac of length."""
    def tan_out(e):
        key = tuple(sorted(int(x) for x in e))
        P = cur[key] if key[0] == v else cur[key][::-1]
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        j = max(1, int(np.searchsorted(s, frac * s[-1])))
        t = P[j] - P[0]
        return t / max(np.linalg.norm(t), 1e-30)
    u, w = -tan_out(ea), tan_out(eb)
    return float(np.degrees(np.arccos(np.clip(u @ w, -1, 1))))


# ------------------------------------------------------------------ differentiable TFI
def resample_t(P, n):
    """[E,m,3] -> [E,n,3] uniform in arclength (differentiable in the points)."""
    d = (P[:, 1:] - P[:, :-1]).norm(dim=-1)
    s = torch.cat([torch.zeros_like(d[:, :1]), d.cumsum(1)], 1)
    s = s / s[:, -1:].clamp(min=1e-12)
    q = torch.linspace(0, 1, n, device=P.device, dtype=P.dtype).expand(len(P), n).contiguous()
    i = torch.searchsorted(s.detach().contiguous(), q).clamp(1, P.shape[1] - 1)
    s0, s1 = s.gather(1, i - 1), s.gather(1, i)
    w = ((q - s0) / (s1 - s0).clamp(min=1e-12)).clamp(0, 1)[..., None]
    g = lambda j: P.gather(1, j[..., None].expand(-1, -1, 3))  # noqa: E731
    return g(i - 1) * (1 - w) + g(i) * w


def bezier_t(V, E, c1, c2, m):
    t = torch.linspace(0, 1, m, device=V.device, dtype=V.dtype)[None, :, None]
    P0, P1 = V[E[:, 0]][:, None], V[E[:, 1]][:, None]
    return (1 - t) ** 3 * P0 + 3 * (1 - t) ** 2 * t * c1[:, None] + 3 * (1 - t) * t ** 2 * c2[:, None] + t ** 3 * P1


class TFISurrogate:
    """Edge-based Gordon-Hall per block: X = sum_edges blend * E(s) - 2 sum_corners trilinear * C."""

    def __init__(self, B, E, n):
        eid = {(int(a), int(b)): (k, False) for k, (a, b) in enumerate(E)}
        eid.update({(int(b), int(a)): (k, True) for k, (a, b) in enumerate(E)})
        self.n = n
        self.e_idx, self.e_rev, self.e_axis, self.e_pos = [], [], [], []
        for b in B:
            row = []
            for i, j in HEX_EDGES:
                ax = [d for d in range(3) if CORNERS[i][d] != CORNERS[j][d]][0]
                if CORNERS[i][ax] > CORNERS[j][ax]:                    # orient along +axis
                    i, j = j, i
                k, rev = eid[(int(b[i]), int(b[j]))]
                row.append((k, rev, ax, CORNERS[i]))
            self.e_idx.append([r[0] for r in row]); self.e_rev.append([r[1] for r in row])
            self.e_axis.append([r[2] for r in row]); self.e_pos.append([r[3] for r in row])
        self.B = torch.as_tensor(np.asarray(B))
        u = torch.linspace(0, 1, n)
        self.U = torch.stack(torch.meshgrid(u, u, u, indexing="ij"), -1)  # [n,n,n,3]

    def __call__(self, V, EP):
        """V [M,3], EP [E,n,3] edge points from E[k,0] to E[k,1] -> X [nb,n,n,n,3]."""
        U = self.U.to(V)
        out = []
        for bi in range(len(self.B)):
            X = torch.zeros(*U.shape[:3], 3, dtype=V.dtype, device=V.device)
            for (k, rev, ax, pos) in zip(self.e_idx[bi], self.e_rev[bi], self.e_axis[bi], self.e_pos[bi]):
                P = EP[k].flip(0) if rev else EP[k]                    # [n,3] along axis ax
                w = torch.ones_like(U[..., 0])
                for d in range(3):
                    if d != ax:
                        w = w * (U[..., d] if pos[d] else 1 - U[..., d])
                shape = [1, 1, 1]; shape[ax] = self.n
                X = X + w[..., None] * P.reshape(*shape, 3)
            for c, cc in enumerate(CORNERS):
                w = torch.ones_like(U[..., 0])
                for d in range(3):
                    w = w * (U[..., d] if cc[d] else 1 - U[..., d])
                X = X - 2 * w[..., None] * V[self.B[bi, c]]
            out.append(X)
        return torch.stack(out)


def cell_sj(X):
    """[nb,n,n,n,3] -> [cells,8] corner scaled Jacobians of the surrogate cells."""
    cs = [X[:, i:X.shape[1] - 1 + i, j:X.shape[2] - 1 + j, k:X.shape[3] - 1 + k] for i, j, k in
          ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))]
    cells = torch.stack(cs, -2).reshape(-1, 8, 3)
    nb = torch.as_tensor(_CORNER_NB, device=X.device)
    e = cells[:, nb, :] - cells[:, :, None, :]
    e = e / e.norm(dim=-1, keepdim=True).clamp(min=1e-30)
    return torch.linalg.det(e)


def cell_corner_jac(X):
    """[nb,n,n,n,3] -> ([cells,8,3,3] corner Jacobians = edge triads (columns), [cells] mean edge length)."""
    cs = [X[:, i:X.shape[1] - 1 + i, j:X.shape[2] - 1 + j, k:X.shape[3] - 1 + k] for i, j, k in
          ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))]
    cells = torch.stack(cs, -2).reshape(-1, 8, 3)
    nb = torch.as_tensor(_CORNER_NB, device=X.device)
    e = cells[:, nb, :] - cells[:, :, None, :]                     # [N,8,3(edge),3(xyz)]
    return e.transpose(-1, -2), e.norm(dim=-1).mean((1, 2))


def chi(D, eps):
    """Garanzha et al. 2021 regularisation of the determinant: positive for all D, -> D for eps -> 0."""
    return 0.5 * (D + torch.sqrt(eps ** 2 + D ** 2))


def garanzha_energy(J, eps, theta):
    """Regularised Winslow (shape) + volume energy per corner (Garanzha et al. 2021, 'Foldover-free maps
    in 50 lines of code'), d = 3: f = tr(J^T J) / chi^(2/3), g = (det^2 + 1) / chi; E = (1-theta) f + theta g.
    J must be normalised by the target size. Returns (mean energy, min det)."""
    D = torch.linalg.det(J)
    c = chi(D, eps)
    f = (J ** 2).sum((-1, -2)) / c ** (2.0 / 3.0)
    g = (D ** 2 + 1) / c
    return ((1 - theta) * f + theta * g).mean(), D.min()


# ------------------------------------------------------------------ optimisation
class CurveProblem:
    """Shared setup of the edge-curve optimisation: free control points of the non-feature edges,
    fixed feature arcs, differentiable TFI surrogate, surface / bending / data terms."""

    def __init__(self, V, B, E, c1, c2, arcs, det, proj, epat, n=9, m=33):
        dt = torch.float64
        self.E, self.proj, self.epat, self.n, self.m = E, proj, epat, n, m
        self.Vt = torch.as_tensor(V, dtype=dt); self.Et = torch.as_tensor(E)
        self.L = (self.Vt[self.Et[:, 1]] - self.Vt[self.Et[:, 0]]).norm(dim=1)
        self.free = torch.tensor([k not in arcs for k in range(len(E))])
        self.c10, self.c20 = torch.as_tensor(c1, dtype=dt), torch.as_tensor(c2, dtype=dt)
        self.x1 = self.c10[self.free].clone().requires_grad_(True)
        self.x2 = self.c20[self.free].clone().requires_grad_(True)
        self.fixedEP = torch.zeros(len(E), n, 3, dtype=dt)
        for k, P in arcs.items():
            self.fixedEP[k] = resample_t(torch.as_tensor(P, dtype=dt)[None], n)[0]
        self.tfi = TFISurrogate(B, E, n)
        self.bnd = [k for k in range(len(E)) if self.free[k] and len(epat[k]) == 1]   # free edges on one patch
        self.bend = [(v, e, f) for (v, e, f), kd in det["pairs"].items() if kd != "feature"]
        self.targets = None

    def ctrl(self):
        c1t = self.c10.clone(); c2t = self.c20.clone()
        c1t[self.free] = self.x1; c2t[self.free] = self.x2
        return c1t, c2t

    def mesh(self):
        """(surrogate nodes X, dense edge points, c1, c2) for the current variables."""
        c1t, c2t = self.ctrl()
        dense = bezier_t(self.Vt, self.Et, c1t, c2t, self.m)
        EP = torch.where(self.free[:, None, None], resample_t(dense, self.n), self.fixedEP)
        return self.tfi(self.Vt, EP), dense, c1t, c2t

    def reproject(self):
        """Surface targets for the interior points of the free boundary edges (current position)."""
        if not self.bnd:
            return
        with torch.no_grad():
            _, dense, _, _ = self.mesh()
        Q = dense[self.bnd][:, 1:-1]
        self.targets = torch.as_tensor(np.stack([self.proj.onto(Q[j].numpy(), list(self.epat[k]))
                                                 for j, k in enumerate(self.bnd)]), dtype=torch.float64)

    def side_terms(self, dense, c1t, c2t):
        """(surface, bending, data) losses."""
        L, E = self.L, self.E
        ls = torch.zeros((), dtype=torch.float64)
        if self.bnd:
            Q = dense[self.bnd][:, 1:-1]
            ls = (((Q - self.targets) / L[self.bnd][:, None, None]) ** 2).sum(-1).mean()
        lb = torch.zeros((), dtype=torch.float64)
        for v, e, f in self.bend:
            def tan_out(k):
                a, b = int(E[k, 0]), int(E[k, 1])
                t = c1t[k] - self.Vt[a] if a == v else c2t[k] - self.Vt[b]
                return t / t.norm().clamp(min=1e-30)
            lb = lb + (1 + (tan_out(e) * tan_out(f)).sum())             # 1 - cos(kink)
        lb = lb / max(1, len(self.bend))
        ld = ((((c1t - self.c10) ** 2).sum(1) + ((c2t - self.c20) ** 2).sum(1)) / L ** 2)[self.free].mean()
        return ls, lb, ld

    def result(self):
        c1t, c2t = self.ctrl()
        return c1t.detach().numpy(), c2t.detach().numpy()


def optimise(V, B, E, c1, c2, arcs, det, proj, epat, n=9, m=33, iters=300, lr=2e-3, w=None, reproj=25):
    """Optimise the control points of the non-feature edges; returns (c1, c2, log).
    w["method"] = "hinge" (Adam on a scaled-Jacobian hinge) or "garanzha" (regularised Winslow + volume
    energy with the epsilon continuation of Garanzha et al. 2021, L-BFGS per epsilon step)."""
    w = {"method": "hinge", "quality": 1.0, "surface": 10.0, "bending": 1.0, "data": 0.1, "tau": 0.6,
         "theta": 0.1, "outer": 15, "inner": 30, **(w or {})}
    pr = CurveProblem(V, B, E, c1, c2, arcs, det, proj, epat, n, m)
    if w["method"] == "garanzha":
        return _optimise_garanzha(pr, w)
    opt = torch.optim.Adam([pr.x1, pr.x2], lr=lr * float(pr.L.mean()))
    log = []
    for it in range(iters + 1):
        if it % reproj == 0:
            pr.reproject()
        X, dense, c1t, c2t = pr.mesh()
        sj = cell_sj(X).min(1).values
        lq = (torch.relu(w["tau"] - sj) ** 2).mean() + 10 * (torch.relu(-sj) ** 2).mean()
        ls, lb, ld = pr.side_terms(dense, c1t, c2t)
        loss = w["quality"] * lq + w["surface"] * ls + w["bending"] * lb + w["data"] * ld
        if it % 50 == 0 or it == iters:
            log.append({"it": it, **{k: float(x.detach()) for k, x in (("loss", loss), ("q", lq), ("s", ls), ("b", lb), ("d", ld))},
                        "sj_min": float(sj.min().detach()), "sj_p1": float(torch.quantile(sj.detach(), 0.01))})
        if it == iters:
            break
        opt.zero_grad(); loss.backward(); opt.step()
    return (*pr.result(), log)


def _optimise_garanzha(pr, w):
    """Outer loop over epsilon (Garanzha et al. 2021, sec. 'untangling'): minimise E_eps with L-BFGS, then
    sigma = max(1 - E_new/E_old, 0.1), mu = (1 - sigma) chi(D_min, eps), eps <- 2 sqrt(mu (mu - D_min)) while
    D_min < mu, else eps -> ~0. Target per corner (TMOP-style W): the start geometry's edge lengths along the
    three corner edges with right angles, J W^-1 scales each column by 1/|e_i0|; so the energy keeps the
    aspect ratio of the cells and penalises skew / folding / size change. Surface targets are re-projected
    once per outer step."""
    with torch.no_grad():
        X0, _, _, _ = pr.mesh()
        J0, _ = cell_corner_jac(X0)
    winv = 1.0 / J0.norm(dim=-2, keepdim=True).clamp(min=1e-12)       # [N,8,1,3] column scales
    eps = torch.tensor(1.0, dtype=torch.float64)
    log = []

    def total():
        X, dense, c1t, c2t = pr.mesh()
        J, _ = cell_corner_jac(X)
        eq, dmin = garanzha_energy(J * winv, eps, w["theta"])
        ls, lb, ld = pr.side_terms(dense, c1t, c2t)
        return w["quality"] * eq + w["surface"] * ls + w["bending"] * lb + w["data"] * ld, eq, dmin, X, (ls, lb, ld)

    pr.reproject()
    for k in range(w["outer"]):
        with torch.no_grad():
            e_old = float(total()[0])
        opt = torch.optim.LBFGS([pr.x1, pr.x2], lr=1.0, max_iter=w["inner"], line_search_fn="strong_wolfe",
                                history_size=20)

        def closure():
            opt.zero_grad()
            loss = total()[0]
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            loss, eq, dmin, X, (ls, lb, ld) = total()
            sj = cell_sj(X).min(1).values
        e_new = float(loss)
        sigma = max(1.0 - e_new / max(e_old, 1e-30), 0.1)
        mu = (1 - sigma) * float(chi(dmin, eps))
        log.append({"it": k, "eps": float(eps), "loss": e_new, "q": float(eq), "s": float(ls), "b": float(lb), "d": float(ld),
                    "det_min": float(dmin), "sj_min": float(sj.min()), "sj_p1": float(torch.quantile(sj, 0.01))})
        if float(dmin) < mu:
            eps = torch.tensor(2 * np.sqrt(max(mu * (mu - float(dmin)), 0.0)), dtype=torch.float64)
        else:
            eps = torch.tensor(1e-9, dtype=torch.float64)
        if float(dmin) > 0 and 1 - e_new / max(e_old, 1e-30) < 1e-3:
            break
        pr.reproject()
    return (*pr.result(), log)


# ------------------------------------------------------------------ evaluation
def mesh_eval(V, B, cur, raw, target_h, vtk):
    try:
        rep = refill_polytron(V, B, cur, target_h, vtk, write_edges=False,
                              surface=(raw["surface_points"], raw["surface_tris"], raw["surface_tri_label"]))
        P, _ = read_hex_vtk(vtk)
        sf = surface_fit(P, rep["boundary_point_ids"], rep["boundary_quads"], raw["surface_points"], raw["surface_tris"])
        q = from_vtk(vtk)
        return {"watertight": bool(rep["watertight"]),
                "inv_share": rep["inverted_curved"] / max(1, rep["cells_after"]),
                "uncovered": float(sf["uncovered_share"]), "sj_p1": q["sj_p1"], "sj_min": q["sj_min"],
                "nonortho_max": q["nonortho_max"], "skew_max": q["skew_max"], "R_mesh": r_mesh(q)}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def closed_chain_edges(pairs):
    """Edges of chains that close on themselves (every edge paired at both ends)."""
    deg = {}
    for (v, e, f) in pairs:
        deg[e] = deg.get(e, 0) + 1; deg[f] = deg.get(f, 0) + 1
    out = set()
    for ch in C.chains(pairs, np.zeros((max(deg) + 1, 2), int) if deg else np.zeros((0, 2), int)):
        if all(deg.get(e, 0) == 2 for e in ch):
            out |= set(ch)
    return out


def pair_kinks(cur, E, det, frac=0.01):
    """Kink per continuity pair on the curves that go into the fill (tangents over frac of the
    edge length: a smooth curve gives ~curvature * frac, a kink keeps its angle)."""
    out = {}
    closed = closed_chain_edges(det["pairs"])
    for (v, e, f), kd in det["pairs"].items():
        if kd == "feature":
            kd = "feature_closed" if e in closed and f in closed else "feature_open"
        out.setdefault(kd, []).append(poly_kink(cur, v, E[e], E[f], frac))
    for kd, lst in det["ambiguous"].items():
        if kd == "3d":
            out["ambiguous"] = [poly_kink(cur, v, E[e], E[f], frac) for v, cand in lst for e, f in cand]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="/work/meshtron_v2/data/flow_canon_papaya.pt")
    ap.add_argument("--split", default="val_new", choices=("train", "val", "val_new"))
    ap.add_argument("--src-root", default=C.RELABEL)
    ap.add_argument("--ckpt", default="/work/meshtron_v2/runs/flowp_flow_all_s0/last.pt")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--target-h", type=float, default=0.08)
    ap.add_argument("--weights", default="{}", help="JSON overrides for the loss weights")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--feature-cache", default="/work/curve/features_v2")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    torch.set_num_threads(2)
    variants = a.variants.split(",")
    data = torch.load(a.data, weights_only=False)
    model = None
    if any(v.startswith("pred") for v in variants):
        from flow_geo import sample
        from train import batch_points
        model, tp, fa = C.load_flow(a.ckpt, data)
        gen = torch.Generator(device="cuda"); gen.manual_seed(1234)
    wts = json.loads(a.weights)

    rows = []
    for it in data[a.split][a.start:a.start + a.n]:
        name = it["name"]
        npz = os.path.join(a.src_root, name, "sample.npz")
        with np.load(npz, allow_pickle=True) as zf:
            raw = {k: np.asarray(zf[k]) for k in zf.files}
        fm = FeatureModelV2(npz, cache_dir=a.feature_cache)
        seams = SmoothSeams(fm.seam_curves)
        proj = SurfaceProjector(raw["surface_points"], raw["surface_tris"], raw["surface_tri_label"])
        V = it["verts"].numpy().astype(np.float64)
        B = it["conn"].numpy().astype(np.int64)
        E = it["edges"].numpy().astype(np.int64)
        c1, c2 = offsets_to_ctrl(V[E[:, 0]], V[E[:, 1]], it["eoff"].numpy().astype(np.float64))
        vpatch = [set((np.flatnonzero(l.numpy() > 0.5) + 1).tolist()) for l in it["vlab"]]
        det = C.detect(V, B, E, (c1, c2), fm, vpatch)
        cnt, per = C.block_faces(B)
        uf = list({tuple(sorted(f)): f for fs in per for f in fs}.values())
        bq = [f for f in uf if cnt[tuple(sorted(f))] == 1]
        qlab = C.quad_patch(bq, V, E, (c1, c2), fm, vpatch)
        share2 = C.edge_face_sets(E, bq)
        epat = [set(int(qlab[q]) for q in share2[k]) for k in range(len(E))]

        geo = {"gt": (V, c1, c2)}
        if model is not None:
            pts, pmask = batch_points([it], "cuda")
            with torch.no_grad():
                z = model.enc(pts, pmask)
                v, o = tp.decode(*sample(model, z, fa["mode"], fa["steps"], gen))
            Vp = snap_all(v[0].cpu().numpy().astype(np.float64), B, fm)
            p1, p2 = offsets_to_ctrl(Vp[E[:, 0]], Vp[E[:, 1]], o[0].cpu().numpy().astype(np.float64))
            geo["pred"] = (Vp, p1, p2)
        row = {"name": name}
        for var in variants:
            src, _, mode = var.partition("_")
            Vg, g1, g2 = geo[src]
            arcs = {}
            if mode in ("seam", "opt", "gar"):
                Vg, arcs = feature_arcs(Vg, E, g1, g2, seams, det)
                row.setdefault("moved", {})[var] = float(np.linalg.norm(Vg - geo[src][0], axis=1).max())
            log = None
            if mode in ("opt", "gar"):
                wv = {**wts, "method": "garanzha", "quality": wts.get("quality_gar", 0.01), "outer": wts.get("outer", 25)} \
                    if mode == "gar" else wts
                t0 = time.perf_counter()
                g1, g2, log = optimise(Vg, B, E, g1, g2, arcs, det, proj, epat, iters=a.iters, w=wv)
                row.setdefault("opt_seconds", {})[var] = time.perf_counter() - t0
            cur = to_curves(Vg, E, g1, g2, arcs)
            vtk = os.path.join(a.out_dir, f"{name}_{var}.vtk")
            r = mesh_eval(Vg, B, cur, raw, a.target_h, vtk)
            r["kinks"] = pair_kinks(cur, E, det)
            r["n_arcs"] = len(arcs)
            if log:
                r["opt_log"] = log
            row[var] = r
        rows.append(row)
        print(f"{name:20s} " + " | ".join(
            f"{v} " + ("ERR" if "error" in row[v] else
                       f"unc {row[v]['uncovered']:.3f} inv {100 * row[v]['inv_share']:.2f}% sj1 {row[v]['sj_p1']:.2f} "
                       f"ring {np.median(row[v]['kinks'].get('feature_closed', [0])):.0f}") for v in variants), flush=True)
        with open(os.path.join(a.out_dir, "rows.json"), "w") as fh:
            json.dump(rows, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
