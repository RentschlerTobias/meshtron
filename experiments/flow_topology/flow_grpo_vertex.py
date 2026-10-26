#!/usr/bin/env python3
"""Flow-GRPO for stage 1 (flow_vertex.FlowVertex): RL fine-tuning on the model's OWN trajectories.

Sampling as an SDE with the marginals of the learned flow (Flow-GRPO, Liu et al. 2025), in our convention
x_t = (1-t) x0 + t x1, x0 ~ N(0, I), velocity v = model(x_t, t):
    x0_hat = x_t - t v,  score = -x0_hat / (1-t)
    x_{k+1} = x_k + [v - sigma_k^2/2 * x0_hat/(1-t)] dt + sigma_k sqrt(dt) eps,   sigma_k = a sqrt(1-t_k)
so every Euler step is a Gaussian policy N(mu_theta(x_k), sigma_k^2 dt I) with a closed-form log-probability and a
closed-form KL to the reference model (same sigma). The last step is deterministic (no noise left in the result).
The point encoder and the count head stay frozen; only the velocity network is trained.

Reward per rollout (one corner set), from the downstream chain (infer_chain_m4: unique snap, pointer greedy +
--ptr-k samples, curve_postprocess):
    w_chain * [some pointer candidate is positively oriented, uses all corners and has all feature edges]  (no GT)
  - w_dup   * [two raw corners closer than the data-derived duplicate threshold]                           (no GT)
  + w_iso   * [top-ranked candidate isomorphic to the GT block complex]                                    (GT)
  - w_err   * Hungarian-matched corner error (1e-3 x diagonal)                                             (GT)
Group-relative advantages over G rollouts of the same geometry; on-policy single update (ratio 1).
Hard-example mining: train geometries where ODE samples of the reference model contain duplicates are drawn with
probability --hard-frac.

    PYTHONPATH=<dp3d hex3d_algohex> python flow_grpo_vertex.py --out runs/vtx_grpo_s0 --steps 200
"""
import argparse
import copy
import json
import os
import sys
import time
from multiprocessing import get_context

sys.path.insert(0, "/opt/stack/meshtron")
sys.path.insert(0, "/opt/stack/meshtron/experiments/flow_topology")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

import infer_chain_m4 as M4  # noqa: E402  (import-order guard, curve_postprocess, snap_unique, ...)
from grpo_pointer import reward_worker  # noqa: E402

P2, FV, FVX = M4.P2, M4.FV, M4.FVX


# ----------------------------------------------------------------------- SDE policy
def step_mean(model, x, t, z, vmask, a):
    """Mean of the Gaussian Euler step and its noise std (per unit sqrt(dt))."""
    tt = torch.full((x.shape[0],), t, device=x.device)
    v = model(x, tt, z, vmask).float()
    sig = a * (1 - t) ** 0.5
    x0h = x - t * v
    return v - 0.5 * sig ** 2 * x0h / max(1 - t, 1e-4), sig


@torch.no_grad()
def rollout(model, z, vmask, n_steps, a, gen):
    """G stochastic trajectories (z, vmask already repeated G times). Returns states [n_steps+1, G, N, 3]."""
    dt = 1.0 / n_steps
    x = torch.randn(vmask.shape[0], vmask.shape[1], 3, device=z.device, generator=gen)
    xs = [x]
    for k in range(n_steps):
        mu, sig = step_mean(model, x, k * dt, z, vmask, a)
        x = x + mu * dt
        if k < n_steps - 1:
            x = x + sig * dt ** 0.5 * torch.randn(x.shape, device=x.device, generator=gen)
        xs.append(x)
    return torch.stack(xs)


def logp_and_kl(model, ref, xs, z, vmask, n_steps, a, ks):
    """Sum over the stochastic steps ks of log pi(x_{k+1}|x_k) per trajectory [G], and the mean KL to ref [G]."""
    dt = 1.0 / n_steps
    m = vmask[..., None].float()
    lp, kl = 0.0, 0.0
    for k in ks:
        mu, sig = step_mean(model, xs[k], k * dt, z, vmask, a)
        with torch.no_grad():
            mur, _ = step_mean(ref, xs[k], k * dt, z, vmask, a)
        var = sig ** 2 * dt
        res = xs[k + 1] - xs[k] - mu * dt
        lp = lp - ((res ** 2) * m).sum((1, 2)) / (2 * var)
        kl = kl + ((((mu - mur) * dt) ** 2) * m).sum((1, 2)) / (2 * var)
    return lp, kl / len(ks)


# ----------------------------------------------------------------------- reward
def matched_rel(P, G, diag):
    D = np.linalg.norm(P[:, None] - G[None], axis=-1)
    r, c = linear_sum_assignment(D)
    return 1000 * float(D[r, c].mean()) / diag


class Chain:
    """Downstream chain pieces for the reward: unique snap -> pointer candidates (GPU, main process)."""

    def __init__(self, data, ptr_ckpt, dev, ptr_k, thr):
        ck = torch.load(ptr_ckpt, weights_only=False); pa = ck["args"]
        self.ptr = P2.SetPointer2(pa["d"], pa["layers"], pa["heads"], ck["max_len"]).to(dev)
        self.ptr.load_state_dict(ck["model"]); self.ptr.eval()
        self.pnrm = FV.Norm(data, data["train"]).to(dev)
        self.dev, self.k, self.thr, self.fms = dev, ptr_k, thr, {}

    def fm(self, name):
        if name not in self.fms:
            self.fms[name] = M4.FeatureModelV2(P2.npz_of(name), cache_dir=P2.FEAT_CACHE)
        return self.fms[name]

    @torch.no_grad()
    def candidates(self, it, P, pts, pmask, gen, fixed_seed=True):
        diag = M4.diag_of(it)
        Ps = M4.snap_unique(P, self.fm(it["name"]), self.thr * diag)
        x = self.pnrm.enc_v(torch.tensor(Ps, dtype=torch.float32, device=self.dev))[None]
        vmask = torch.ones(1, len(Ps), dtype=torch.bool, device=self.dev)
        e, z = self.ptr.memory(x, vmask, pts, pmask, torch.zeros(1, len(Ps), 7, device=self.dev))
        if fixed_seed:                      # same corner set -> same candidates -> same reward
            gen = torch.Generator(device=self.dev); gen.manual_seed(1234 + sum(map(ord, it["name"])))
        cands = [P2.decode(self.ptr, e, z, vmask)] + [P2.decode(self.ptr, e, z, vmask, 0.7, gen) for _ in range(self.k)]
        return Ps, [c for c in cands if c[0]]


def mine_hard(model, nrm, items, thr, dev, reps=4, steps=32, bs=16):
    """Indices of train items where ODE samples of the reference model contain a duplicate corner pair."""
    gen = torch.Generator(device=dev); gen.manual_seed(99)
    hard = []
    with torch.no_grad():
        for i in range(0, len(items), bs):
            chunk = items[i:i + bs]
            pts, pmask = M4.batch_points(chunk, dev)
            z = model.enc(pts, pmask)
            n = [len(it["verts"]) for it in chunk]
            for _ in range(reps):
                S = FVX.sample(model, z, n, steps, gen)
                for j, it in enumerate(chunk):
                    if M4.min_pair(nrm.dec_v(S[j]).cpu().double().numpy(), M4.diag_of(it)) < thr:
                        hard.append(i + j)
    return sorted(set(hard))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="/opt/stack/meshtron/experiments/flow_topology/data/flow_canon_full.pt")
    ap.add_argument("--vtx-ckpt", default="/opt/stack/meshtron/experiments/flow_topology/runs/vtx_full_s0/best.pt")
    ap.add_argument("--ptr-ckpt", default="/opt/stack/meshtron/experiments/flow_topology/runs/ptr2_nolab/ep50.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=200, help="GRPO updates")
    ap.add_argument("--batch", type=int, default=4, help="geometries per update")
    ap.add_argument("--group", type=int, default=8, help="rollouts per geometry")
    ap.add_argument("--ode-steps", type=int, default=32)
    ap.add_argument("--a", type=float, default=0.3, help="SDE noise level (0.7 degrades the corners: matched error 22 -> 37e-3)")
    ap.add_argument("--train-frac", type=float, default=0.5, help="share of the stochastic steps used per update")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.01, help="KL weight")
    ap.add_argument("--ptr-k", type=int, default=2, help="pointer samples per rollout besides greedy")
    ap.add_argument("--w-chain", type=float, default=1.0)
    ap.add_argument("--w-dup", type=float, default=1.0)
    ap.add_argument("--w-iso", type=float, default=1.0)
    ap.add_argument("--w-err", type=float, default=0.02)
    ap.add_argument("--no-gt", action="store_true", help="reward without GT terms (iso, corner error)")
    ap.add_argument("--skip-chain", action="store_true", help="no pointer / postprocess (reward = dup + corner error only)")
    ap.add_argument("--hard-frac", type=float, default=0.5)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--print-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--free-ptr-seed", action="store_true", help="random pointer samples (noisy reward)")
    ap.add_argument("--stop-worse", type=float, default=3.0,
                    help="stop when the val AND val_full corner errors exceed their start values by this much")
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = "cuda"
    os.makedirs(a.out, exist_ok=True)
    data = torch.load(a.data, weights_only=False)
    tr, va = data["train"], data["val"]
    model, nrm, vargs = FVX.load(a.vtx_ckpt, data)
    ref = copy.deepcopy(model).eval()
    for p in list(ref.parameters()) + list(model.enc.parameters()) + list(model.count.parameters()):
        p.requires_grad_(False)
    model.eval()                                                   # no dropout; frozen parts stay frozen
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    thr = M4.dup_threshold(tr)
    chain = Chain(data, a.ptr_ckpt, dev, a.ptr_k, thr)
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
    w_iso, w_err = (0.0, 0.0) if a.no_gt else (a.w_iso, a.w_err)

    t0 = time.time()
    hard = mine_hard(ref, nrm, tr, thr, dev)
    print(f"[fgrpo] dup threshold {thr:.2e} | hard train items {len(hard)}/{len(tr)} ({time.time()-t0:.0f}s)", flush=True)
    pool = get_context("spawn").Pool(a.workers)
    log = []

    def evaluate(tag):
        ev = {s: FVX.evaluate(model, nrm, data[s], a.ode_steps, dev, n_samples=4) for s in ("val", "val_full")}
        rec = {"step": tag, **{f"{s}_{k}": ev[s][k] for s in ev for k in ("corner", "corner_best", "dup_rate")}}
        print(f"[fgrpo] eval {tag} | " + " ".join(f"{k} {v:.3f}" for k, v in rec.items() if k != "step"), flush=True)
        log.append(rec)
        return rec

    ev0 = evaluate(0)
    for step in range(1, a.steps + 1):
        idx = [int(rng.choice(hard)) if hard and rng.random() < a.hard_frac else int(rng.integers(len(tr)))
               for _ in range(a.batch)]
        groups, jobs = [], []
        for i in idx:
            it = tr[i]
            pts, pmask = M4.batch_points([it], dev)
            n = len(it["verts"])
            with torch.no_grad():
                z = model.enc(pts, pmask).repeat(a.group, 1, 1)
            vmask = torch.ones(a.group, n, dtype=torch.bool, device=dev)
            xs = rollout(model, z, vmask, a.ode_steps, a.a, gen)
            P = nrm.dec_v(xs[-1]).cpu().double().numpy()
            G = it["verts"].double().numpy(); diag = M4.diag_of(it)
            npz = P2.npz_of(it["name"]); gt = it["conn"].long().numpy()
            meta = []
            for g in range(a.group):
                Ps, cands = (None, []) if a.skip_chain else chain.candidates(it, P[g], pts, pmask, gen, not a.free_ptr_seed)
                meta.append({"dup": M4.min_pair(P[g], diag) < thr, "err": matched_rel(P[g], G, diag),
                             "ll": [c[1] for c in cands], "j0": len(jobs)})
                for blocks, _ in cands:
                    jobs.append((Ps, blocks, gt, npz, {"orient": 0, "feat": 0, "iso": 0, "used": 0}))
                meta[-1]["j1"] = len(jobs)
            groups.append((z, vmask, xs, meta))
        res = pool.map(reward_worker, jobs)
        loss_terms, stats, n_inf = [], {"r": [], "chain": [], "dup": [], "iso": [], "err": []}, 0
        opt.zero_grad()
        for z, vmask, xs, meta in groups:
            R = []
            for mt in meta:
                rr = res[mt["j0"]:mt["j1"]]
                ok = [c["orient"] * c["feat"] * c["used"] for c in rr]
                chain_ok = float(max(ok)) if ok else 0.0
                iso = 0.0
                if rr:                                             # top-ranked as in infer_chain_m4
                    best = max(range(len(rr)), key=lambda q: (rr[q]["orient"], rr[q]["feat"], rr[q]["used"], mt["ll"][q]))
                    iso = rr[best]["iso"]
                r = a.w_chain * chain_ok - a.w_dup * mt["dup"] + w_iso * iso - w_err * mt["err"]
                R.append(r)
                for k_, v_ in (("r", r), ("chain", chain_ok), ("dup", float(mt["dup"])), ("iso", iso), ("err", mt["err"])):
                    stats[k_].append(v_)
            R = np.array(R)
            if R.std() < 1e-6:
                continue
            n_inf += 1
            A = torch.tensor((R - R.mean()) / (R.std() + 1e-6), dtype=torch.float32, device=dev)
            ks = sorted(rng.choice(a.ode_steps - 1, max(1, int(a.train_frac * (a.ode_steps - 1))), replace=False).tolist())
            lp, kl = logp_and_kl(model, ref, xs, z, vmask, a.ode_steps, a.a, ks)
            loss = (-(A * lp / len(ks)).mean() + a.beta * kl.mean()) / a.batch
            loss.backward()
            loss_terms.append((float(loss.detach()), float(kl.mean().detach())))
        if loss_terms:
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        rec = {"step": step, **{k: float(np.mean(v)) for k, v in stats.items()}, "informative": n_inf,
               "kl": float(np.mean([k for _, k in loss_terms])) if loss_terms else 0.0, "t": time.time() - t0}
        log.append(rec)
        if step % a.print_every == 0 or step == 1:
            print(f"[fgrpo] step {step:4d} | R {rec['r']:.3f} chain {rec['chain']:.2f} dup {rec['dup']:.2f} "
                  f"iso {rec['iso']:.2f} err {rec['err']:.1f} | inf {n_inf}/{len(groups)} kl {rec['kl']:.3f} "
                  f"({rec['t']:.0f}s)", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            ev = evaluate(step)
            torch.save({"model": model.state_dict(), "args": vargs, "grpo": vars(a), "step": step}, f"{a.out}/step{step}.pt")
            if all(ev[f"{s_}_corner"] > ev0[f"{s_}_corner"] + a.stop_worse for s_ in ("val", "val_full")):
                print(f"[fgrpo] stop: corners worse than at the start on val and val_full", flush=True)
                json.dump(log, open(f"{a.out}/log.json", "w"), indent=1)
                break
        json.dump(log, open(f"{a.out}/log.json", "w"), indent=1)
    pool.close()


if __name__ == "__main__":
    main()
