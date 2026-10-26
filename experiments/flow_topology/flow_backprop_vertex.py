#!/usr/bin/env python3
"""Reward backprop for stage 1 (flow_vertex.FlowVertex), DRaFT-style: the model's OWN deterministic ODE trajectory
(same Euler sampler as at inference) is computed WITH gradient and a differentiable loss on the end points is
back-propagated through all (or the last --grad-steps) steps.

End-point loss per corner set (distances relative to the geometry diagonal):
    match  : Hungarian-matched squared distance to the GT corners (assignment without gradient) -> every GT corner
             is hit exactly once; a doubled corner leaves another GT corner unmatched and costs there
  + w_rep * sum over pairs relu(rep_thr - d_ij)^2 / rep_thr^2  -> pushes exactly the colliding points apart
Regulariser: beta * mean ||v_theta - v_ref||^2 along the generated trajectory (stay close to the pretrained flow).
Point encoder and count head stay frozen. Hard-example mining as in flow_grpo_vertex.py.

    python flow_backprop_vertex.py --out runs/vtx_bp_s0 --steps 300
"""
import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, "/opt/stack/meshtron/experiments/flow_topology")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

import flow_vertex as FVX  # noqa: E402
from train import batch_points  # noqa: E402


def diag_of(it):
    return float(np.linalg.norm(np.ptp(it["points"][:, :3].double().numpy(), 0)))


def min_pair(X, diag):
    D = np.linalg.norm(X[:, None] - X[None], axis=-1); np.fill_diagonal(D, np.inf)
    return float(D.min()) / diag


def dup_threshold(items, frac=0.5, q=1.0):
    """Same data-derived duplicate threshold as infer_chain_m4.dup_threshold."""
    return frac * float(np.percentile([min_pair(it["verts"].double().numpy(), diag_of(it)) for it in items], q))


def mine_hard(model, nrm, items, thr, dev, reps=4, steps=32, bs=16):
    gen = torch.Generator(device=dev); gen.manual_seed(99)
    hard = []
    with torch.no_grad():
        for i in range(0, len(items), bs):
            chunk = items[i:i + bs]
            pts, pmask = batch_points(chunk, dev)
            z = model.enc(pts, pmask)
            n = [len(it["verts"]) for it in chunk]
            for _ in range(reps):
                S = FVX.sample(model, z, n, steps, gen)
                for j, it in enumerate(chunk):
                    if min_pair(nrm.dec_v(S[j]).cpu().double().numpy(), diag_of(it)) < thr:
                        hard.append(i + j)
    return sorted(set(hard))


def trajectory(model, ref, x, z, vmask, n_steps, grad_steps):
    """Euler ODE as in flow_vertex.sample; gradient through the last grad_steps steps. Returns end points and the
    mean squared velocity deviation from the reference model along the way."""
    dt = 1.0 / n_steps
    dev_sq, m = [], vmask[..., None].float()
    for k in range(n_steps):
        t = torch.full((x.shape[0],), k * dt, device=x.device)
        if k < n_steps - grad_steps:
            with torch.no_grad():
                x = x + model(x, t, z, vmask) * dt
            continue
        v = model(x, t, z, vmask)
        with torch.no_grad():
            vr = ref(x, t, z, vmask)
        dev_sq.append((((v - vr) ** 2) * m).sum() / (m.sum() * 3))
        x = x + v * dt
    return x, torch.stack(dev_sq).mean()


def endpoint_loss(Xn, it, nrm, rep_thr, w_rep):
    """Xn [N,3] normalised end points of one trajectory -> (loss, stats)."""
    diag = diag_of(it)
    X = nrm.dec_v(Xn)
    G = it["verts"].to(X.device).float()
    D = torch.cdist(X, G) / diag
    r, c = linear_sum_assignment(D.detach().cpu().numpy())
    match = (D[r, c] ** 2).mean()
    P = torch.cdist(X, X) / diag
    iu = torch.triu_indices(len(X), len(X), 1, device=X.device)
    pd = P[iu[0], iu[1]]
    rep = (torch.relu(rep_thr - pd) ** 2).sum() / rep_thr ** 2
    return match + w_rep * rep, {"match": 1000 * float(D[r, c].mean()), "min_pair": float(pd.min())}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="/opt/stack/meshtron/experiments/flow_topology/data/flow_canon_full.pt")
    ap.add_argument("--vtx-ckpt", default="/opt/stack/meshtron/experiments/flow_topology/runs/vtx_full_s0/best.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=300, help="updates")
    ap.add_argument("--batch", type=int, default=8, help="geometries per update")
    ap.add_argument("--group", type=int, default=4, help="noise draws per geometry")
    ap.add_argument("--ode-steps", type=int, default=32)
    ap.add_argument("--grad-steps", type=int, default=32, help="steps with gradient (from the end)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=1.0, help="weight of the velocity deviation from the reference")
    ap.add_argument("--w-match", type=float, default=100.0, help="scales the matched squared distance (~1e-4)")
    ap.add_argument("--w-rep", type=float, default=1.0)
    ap.add_argument("--rep-factor", type=float, default=1.5, help="repulsion range = factor x duplicate threshold")
    ap.add_argument("--hard-frac", type=float, default=0.5)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--stop-worse", type=float, default=3.0)
    ap.add_argument("--print-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = "cuda"
    os.makedirs(a.out, exist_ok=True)
    data = torch.load(a.data, weights_only=False)
    tr = data["train"]
    model, nrm, vargs = FVX.load(a.vtx_ckpt, data)
    ref = copy.deepcopy(model).eval()
    for p in list(ref.parameters()) + list(model.enc.parameters()) + list(model.count.parameters()):
        p.requires_grad_(False)
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    thr = dup_threshold(tr); rep_thr = a.rep_factor * thr
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
    t0 = time.time()
    hard = mine_hard(ref, nrm, tr, thr, dev)
    print(f"[bp] dup threshold {thr:.2e}, repulsion range {rep_thr:.2e} | hard train items {len(hard)}/{len(tr)} "
          f"({time.time()-t0:.0f}s)", flush=True)
    log = []

    def evaluate(tag):
        ev = {s: FVX.evaluate(model, nrm, data[s], a.ode_steps, dev, n_samples=8, dup_thr=thr) for s in ("val", "val_full")}
        rec = {"step": tag, **{f"{s}_{k}": ev[s][k] for s in ev for k in ("corner", "corner_best", "dup_rate")}}
        print(f"[bp] eval {tag} | " + " ".join(f"{k} {v:.3f}" for k, v in rec.items() if k != "step"), flush=True)
        log.append(rec)
        return rec

    ev0 = evaluate(0)
    for step in range(1, a.steps + 1):
        idx = [int(rng.choice(hard)) if hard and rng.random() < a.hard_frac else int(rng.integers(len(tr)))
               for _ in range(a.batch)]
        opt.zero_grad()
        st = {"loss": [], "match": [], "dup": [], "vdev": []}
        for i in idx:
            it = tr[i]
            pts, pmask = batch_points([it], dev)
            with torch.no_grad():
                z = model.enc(pts, pmask).repeat(a.group, 1, 1)
            vmask = torch.ones(a.group, len(it["verts"]), dtype=torch.bool, device=dev)
            x0 = torch.randn(a.group, len(it["verts"]), 3, device=dev, generator=gen)
            X, vdev = trajectory(model, ref, x0, z, vmask, a.ode_steps, a.grad_steps)
            le = []
            for g in range(a.group):
                l_, s_ = endpoint_loss(X[g], it, nrm, rep_thr, a.w_rep / a.w_match)
                le.append(l_)
                st["match"].append(s_["match"]); st["dup"].append(float(s_["min_pair"] < thr))
            loss = (a.w_match * torch.stack(le).mean() + a.beta * vdev) / a.batch
            loss.backward()
            st["loss"].append(float(loss.detach()) * a.batch); st["vdev"].append(float(vdev.detach()))
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        rec = {"step": step, **{k: float(np.mean(v)) for k, v in st.items()}, "t": time.time() - t0}
        log.append(rec)
        if step % a.print_every == 0 or step == 1:
            print(f"[bp] step {step:4d} | loss {rec['loss']:.4f} match {rec['match']:.1f}e-3 dup {rec['dup']:.3f} "
                  f"vdev {rec['vdev']:.2e} ({rec['t']:.0f}s)", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            ev = evaluate(step)
            torch.save({"model": model.state_dict(), "args": vargs, "bp": vars(a), "step": step}, f"{a.out}/step{step}.pt")
            if all(ev[f"{s_}_corner"] > ev0[f"{s_}_corner"] + a.stop_worse for s_ in ("val", "val_full")):
                print("[bp] stop: corners worse than at the start on val and val_full", flush=True)
                break
        json.dump(log, open(f"{a.out}/log.json", "w"), indent=1)
    json.dump(log, open(f"{a.out}/log.json", "w"), indent=1)


if __name__ == "__main__":
    main()
