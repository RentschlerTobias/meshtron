"""Joint optimisation of a full blocking (corners + edge curves) with the differentiable Gordon-Hall surrogate
(curve_opt.TFISurrogate), before the real TFI.

Variables: every corner not pinned (pinned = on >= 3 boundary patches) and the interior points of every edge
(n samples per edge). Constraints by geometry: a corner / edge on 1 boundary patch slides on it, on 2 patches on
their seam (targets re-projected every `reproj` iterations, quadratic penalty). Loss: scaled-Jacobian hinge of
the surrogate cells + geometry penalty + edge smoothness (second differences) + weak data term. Edge points are
sampled from the dense curves by index for parametrised edges (grading / point correspondence) and by arc length
otherwise; the result deforms the dense curves by the optimised displacement, interpolated along the same
parameter, so grading and correspondence survive."""
from collections import defaultdict

import numpy as np
import torch

import curve_opt as CO
from meshtron.geometry.edge_curves import face_cycle
from meshtron.geometry.polytron_tfi import _arclength

HEX_EDGES = CO.HEX_EDGES


def sample_param(Q, n, by_index):
    Q = np.asarray(Q, float)
    if by_index:
        u = np.linspace(0, 1, len(Q))
    else:
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]; u = s / s[-1]
    return u


def boundary_faces(B, with_pos=False):
    own = defaultdict(list)
    for bi, b in enumerate(B):
        for axis in (0, 1, 2):
            for side in (0, 1):
                ids = [int(b[c]) for c in face_cycle(axis, side)]
                own[frozenset(ids)].append((bi, axis, side, ids))
    one = [v[0] for v in own.values() if len(v) == 1]
    return one if with_pos else [f[3] for f in one]


def optimise_blocking(V, B, curves, param, geo, phys_labels, n=9, iters=400, lr=2e-3, reproj=25,
                      w=None, verbose=True):
    w = {"quality": 1.0, "geo": 100.0, "smooth": 1.0, "data": 0.01, "tau": 0.5, **(w or {})}
    dt = torch.float64
    V = np.asarray(V, float); B = np.asarray(B, np.int64)
    E = sorted({(min(int(b[i]), int(b[j])), max(int(b[i]), int(b[j]))) for b in B for i, j in HEX_EDGES})
    eidx = {e: k for k, e in enumerate(E)}
    # boundary labels of faces -> corner / edge constraint sets
    bf = boundary_faces(B, with_pos=True)
    lab_ids = list(phys_labels)
    pl_index = [geo.near.labels.index(l) for l in lab_ids]
    vlab, elab = defaultdict(set), defaultdict(set)
    flab = []
    for bi_, ax_, sd_, ids in bf:
        probe = [V[i] for i in ids]
        for a, b in zip(ids, ids[1:] + ids[:1]):
            Q = curves.get((min(a, b), max(a, b)))
            probe.append(Q[len(Q) // 2] if Q is not None else 0.5 * (V[a] + V[b]))
        d = geo.near.dists(np.array(probe))[:, pl_index].mean(0)
        lab = lab_ids[int(np.argmin(d))]
        flab.append((bi_, ax_, sd_, lab))
        for a, b in zip(ids, ids[1:] + ids[:1]):
            vlab[a].add(lab); elab[(min(a, b), max(a, b))].add(lab)
    pinned = np.array([len(vlab[v]) >= 3 for v in range(len(V))])
    # base edge samples
    P0, U = [], []
    for e in E:
        Q = curves.get(e)
        Q = np.linspace(V[e[0]], V[e[1]], 33) if Q is None else np.asarray(Q, float)
        u = sample_param(Q, n, e in param)
        t = np.linspace(0, 1, n)
        P0.append(np.stack([np.interp(t, u, Q[:, d]) for d in range(3)], 1)); U.append(u)
    P0 = torch.tensor(np.array(P0), dtype=dt)
    L = (P0[:, -1] - P0[:, 0]).norm(dim=1).clamp(min=1e-9)
    Vt0 = torch.tensor(V, dtype=dt)
    dV = torch.zeros_like(Vt0, requires_grad=True)
    dI = torch.zeros(len(E), n - 2, 3, dtype=dt, requires_grad=True)
    Et = torch.tensor(E)
    tlin = torch.linspace(0, 1, n, dtype=dt)[None, :, None]
    tfi = CO.TFISurrogate(B, np.array(E), n)
    pin = torch.tensor(pinned)[:, None]
    vcons = [v for v in range(len(V)) if 0 < len(vlab[v]) < 3]
    econs = [k for k, e in enumerate(E) if elab.get(e)]
    tV = None; tE = None; fdel = None
    u1 = torch.linspace(0, 1, n, dtype=dt)

    def face_slice(X, bi, ax, sd):
        idx = [slice(None)] * 3; idx[ax] = (n - 1) if sd else 0
        return X[(bi, *idx)]                                  # [n,n,3] over the other two axes (in order)

    def face_offsets(X):
        """projected - Coons interior per boundary face (zero on the face boundary)."""
        out = []
        for bi, ax, sd, lab in flab:
            F = face_slice(X, bi, ax, sd).numpy()
            G = F.copy()
            G[1:-1, 1:-1] = geo.onto(F[1:-1, 1:-1].reshape(-1, 3), [lab]).reshape(n - 2, n - 2, 3)
            out.append(torch.tensor(G - F, dtype=dt))
        return out

    def add_faces(X):
        if fdel is None:
            return X
        X = X.clone()
        for (bi, ax, sd, lab), D in zip(flab, fdel):
            wv = u1 if sd else 1 - u1
            shp = [1, 1, 1]; shp[ax] = n
            Dx = D.unsqueeze(ax)                                  # broadcast along the face normal axis
            X[bi] = X[bi] + wv.reshape(*shp, 1) * Dx
        return X

    def state():
        Vt = Vt0 + torch.where(pin, torch.zeros_like(dV), dV)
        EP = P0 + (1 - tlin) * (Vt[Et[:, 0]] - Vt0[Et[:, 0]])[:, None] + tlin * (Vt[Et[:, 1]] - Vt0[Et[:, 1]])[:, None]
        EP = torch.cat([EP[:, :1], EP[:, 1:-1] + dI, EP[:, -1:]], 1)
        return Vt, EP

    def targets():
        with torch.no_grad():
            Vt, EP = state()
        tv = {v: torch.tensor(geo.onto(Vt[v].numpy()[None], sorted(vlab[v]))[0], dtype=dt) for v in vcons}
        te = {k: torch.tensor(geo.onto(EP[k, 1:-1].numpy(), sorted(elab[E[k]])), dtype=dt) for k in econs}
        return tv, te
    opt = torch.optim.Adam([dV, dI], lr=lr * float(L.mean()))
    log = []
    for it in range(iters + 1):
        if it % reproj == 0:
            tV, tE = targets()
            with torch.no_grad():
                Vt_, EP_ = state(); fdel = face_offsets(tfi(Vt_, EP_))
        Vt, EP = state()
        X = add_faces(tfi(Vt, EP))
        sj = CO.cell_sj(X).min(1).values
        lq = (torch.relu(w["tau"] - sj) ** 2).mean() + 10 * (torch.relu(-sj) ** 2).mean()
        lg = torch.zeros((), dtype=dt)
        if vcons:
            lg = lg + torch.stack([((Vt[v] - tV[v]) ** 2).sum() for v in vcons]).mean() / L.mean() ** 2
        if econs:
            lg = lg + torch.stack([(((EP[k, 1:-1] - tE[k]) / L[k]) ** 2).sum(-1).mean() for k in econs]).mean()
        d2 = EP[:, 2:] - 2 * EP[:, 1:-1] + EP[:, :-2]
        d20 = P0[:, 2:] - 2 * P0[:, 1:-1] + P0[:, :-2]
        ls = (((d2 - d20) / L[:, None, None]) ** 2).sum(-1).mean()          # keep the base shape, kill zigzag
        ld = ((dV / L.mean()) ** 2).sum(1).mean() + ((dI / L[:, None, None]) ** 2).sum(-1).mean()
        loss = w["quality"] * lq + w["geo"] * lg + w["smooth"] * ls + w["data"] * ld
        if it % 50 == 0 or it == iters:
            rec = {"it": it, "loss": float(loss), "q": float(lq), "g": float(lg), "s": float(ls),
                   "sj_min": float(sj.min()), "sj_p1": float(torch.quantile(sj.detach(), 0.01)),
                   "neg": int((sj <= 0).sum())}
            log.append(rec)
            if verbose:
                print(f"[opt] it {it:4d} loss {rec['loss']:.4e} q {rec['q']:.3e} g {rec['g']:.2e} "
                      f"sj_min {rec['sj_min']:.3f} p1 {rec['sj_p1']:.3f} neg-cells {rec['neg']}", flush=True)
        if it == iters:
            break
        opt.zero_grad(); loss.backward(); opt.step()
    # write back: corners onto their geometry, dense curves deformed along their own parameter
    with torch.no_grad():
        Vt, EP = state()
    Vn = Vt.numpy().copy()
    for v in vcons:
        Vn[v] = geo.onto(Vn[v][None], sorted(vlab[v]))[0]
    out = dict(curves)
    t = np.linspace(0, 1, n)
    for k, e in enumerate(E):
        Q = curves.get(e)
        Q = np.linspace(V[e[0]], V[e[1]], 33) if Q is None else np.asarray(Q, float)
        disp = EP[k].numpy() - P0[k].numpy()
        dq = np.stack([np.interp(U[k], t, disp[:, d]) for d in range(3)], 1)
        Qn = Q + dq
        Qn[0], Qn[-1] = Vn[e[0]], Vn[e[1]]
        if elab.get(e) and len(Qn) > 2:
            Qn[1:-1] = geo.onto(Qn[1:-1], sorted(elab[e]))
        out[e] = Qn
    return Vn, out, log, {"pinned": int(pinned.sum()), "constrained_corners": len(vcons), "constrained_edges": len(econs)}
