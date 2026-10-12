"""Stage B of the flow plan: block corners + Bezier edge curves of a GIVEN topology
from the labelled point cloud, by flow matching over a set of continuous tokens.

Tokens (one topology, data/flow_*.pt from build_flow_dataset.py):
  vertex  3 values: corner minus the topology's mean train corner, / global std
  edge    6 values: chord-relative Bezier offsets minus the mean, / global std
Every token also gets topology features that do not depend on the GT geometry:
token type, the topology's majority surface labels (multi-hot, blade = 7),
Laplacian eigenvectors of the block-edge graph, and the Fourier-encoded mean-
template position. No sequence order: a set transformer (DiT blocks with adaLN-
Zero time conditioning, self-attention over all tokens, cross-attention to the
PointPerceiver latents of the point cloud).

Modes:
  flow     rectified flow: x_t = (1-t) noise + t data, target velocity data - noise,
           logit-normal t; Euler sampling with --steps steps
  regress  same network, input state 0 and t = 0, direct MSE on the data (control)

Evaluation on val (and --eval-train train items), dataset units * 1000:
  corner     mean corner distance per item, median over items
  curve      mean distance of the predicted to the GT Bezier at t = 1/4, 1/2, 3/4
  curve_gtc  same with GT corners + predicted offsets (curve shape only)
  for flow: one sample with a fixed seed, the mean of --n-samples samples, the
  best-of-N corner error (oracle) and the sample spread; baselines: mean template.

    python flow_geo.py --data data/flow_canon.pt --mode flow --parts all --out runs/flow_s0
"""
import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Attn, PointPerceiver, mlp  # noqa: E402
from train import batch_points  # noqa: E402

N_LAP = 8
N_FREQ = 6


# ----------------------------------------------------------------------- topology / normalisation
class Topology:
    """Everything the model may know about one topology (computed on train only)."""

    def __init__(self, items, lo, hi):
        it0 = items[0]
        self.edges = it0["edges"].long()
        self.M, self.E = len(it0["verts"]), len(self.edges)
        V = torch.stack([it["verts"] for it in items]); O = torch.stack([it["eoff"] for it in items])
        self.vmean, self.omean = V.mean(0), O.mean(0)
        self.vstd = (V - self.vmean).reshape(-1, 3).std(0).clamp(min=1e-6)
        self.ostd = (O - self.omean).reshape(-1, 6).std(0).clamp(min=1e-6)
        vl = torch.stack([it["vlab"] for it in items]); el = torch.stack([it["elab"] for it in items])
        self.vlab = (vl.mean(0) > 0.5).float(); self.elab = (el.mean(0) > 0.5).float()
        # Laplacian eigenvectors of the vertex graph (fixed sign convention: largest |entry| positive)
        A = torch.zeros(self.M, self.M)
        A[self.edges[:, 0], self.edges[:, 1]] = 1; A[self.edges[:, 1], self.edges[:, 0]] = 1
        deg = A.sum(1)
        L = torch.eye(self.M) - A / torch.sqrt(deg[:, None] * deg[None])
        _, vec = torch.linalg.eigh(L.double())
        vec = vec[:, 1:N_LAP + 1].float()
        sign = torch.sign(vec.gather(0, vec.abs().argmax(0, keepdim=True)))
        self.lap = vec * sign
        self.vtpl = (self.vmean - lo) / (hi - lo) * 2 - 1                         # template in [-1,1]
        a, b = self.edges[:, 0], self.edges[:, 1]
        self.etpl = (self.vtpl[a] + self.vtpl[b]) / 2
        self.vfeat = torch.cat([self.vlab, self.lap], 1)                          # [M,7+L]
        self.efeat = torch.cat([self.elab, self.lap[a] + self.lap[b], (self.lap[a] - self.lap[b]).abs()], 1)

    def encode(self, items):
        dev = self.vmean.device
        V = torch.stack([it["verts"] for it in items]).to(dev); O = torch.stack([it["eoff"] for it in items]).to(dev)
        return (V - self.vmean) / self.vstd, (O - self.omean) / self.ostd

    def decode(self, xv, xo):
        return xv * self.vstd + self.vmean, xo * self.ostd + self.omean

    def to(self, dev):
        for k, v in vars(self).items():
            if torch.is_tensor(v):
                setattr(self, k, v.to(dev))
        return self


def fourier(x):
    f = (2.0 ** torch.arange(N_FREQ, device=x.device)) * math.pi
    a = x[..., None] * f
    return torch.cat([x, a.sin().flatten(-2), a.cos().flatten(-2)], -1)


# ----------------------------------------------------------------------- model
def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class DiTBlock(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(d)
        self.ln3 = nn.LayerNorm(d, elementwise_affine=False)
        self.sa, self.ca, self.ff = Attn(d, heads), Attn(d, heads), mlp(d)
        self.ada = nn.Linear(d, 6 * d)
        nn.init.zeros_(self.ada.weight); nn.init.zeros_(self.ada.bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, z, c):
        s1, b1, g1, s2, b2, g2 = self.ada(F.silu(c))[:, None].chunk(6, -1)
        x = x + g1 * self.drop(self.sa(modulate(self.ln1(x), b1, s1)))
        x = x + self.drop(self.ca(self.ln2(x), z))
        return x + g2 * self.drop(self.ff(modulate(self.ln3(x), b2, s2)))


class FlowGeo(nn.Module):
    def __init__(self, topo, d, layers, heads, n_latent, dropout=0.0, curves=True, point_dim=14):
        super().__init__()
        self.topo, self.curves = topo, curves
        nf = 3 * (1 + 2 * N_FREQ)
        self.enc = PointPerceiver(point_dim, d, heads, n_latent, 2)
        self.vin = nn.Linear(3, d)
        self.vtop = nn.Sequential(nn.Linear(topo.vfeat.shape[1] + nf, d), nn.GELU(), nn.Linear(d, d))
        self.ein = nn.Linear(6, d)
        self.etop = nn.Sequential(nn.Linear(topo.efeat.shape[1] + nf, d), nn.GELU(), nn.Linear(d, d))
        self.tmlp = nn.Sequential(nn.Linear(64, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList(DiTBlock(d, heads, dropout) for _ in range(layers))
        self.lnf = nn.LayerNorm(d, elementwise_affine=False)
        self.adaf = nn.Linear(d, 2 * d)
        self.vout, self.eout = nn.Linear(d, 3), nn.Linear(d, 6)
        for m in (self.adaf, self.vout, self.eout):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def temb(self, t):
        f = torch.exp(-math.log(1000.0) * torch.arange(32, device=t.device) / 32)
        a = t[:, None] * 1000.0 * f
        return self.tmlp(torch.cat([a.sin(), a.cos()], -1))

    def forward(self, xv, xo, t, z):
        tp = self.topo; B = xv.shape[0]
        hv = self.vin(xv) + self.vtop(torch.cat([tp.vfeat, fourier(tp.vtpl)], -1))[None]
        h = hv
        if self.curves:
            he = self.ein(xo) + self.etop(torch.cat([tp.efeat, fourier(tp.etpl)], -1))[None]
            h = torch.cat([hv, he], 1)
        c = self.temb(t) + z.mean(1)
        for b in self.blocks:
            h = b(h, z, c)
        sh, sc = self.adaf(F.silu(c))[:, None].chunk(2, -1)
        h = modulate(self.lnf(h), sh, sc)
        M = tp.M
        vo = self.vout(h[:, :M])
        eo = self.eout(h[:, M:]) if self.curves else torch.zeros(B, tp.E, 6, device=h.device)
        return vo, eo


# ----------------------------------------------------------------------- sampling / metrics
@torch.no_grad()
def sample(model, z, mode, steps, gen=None):
    tp = model.topo; B = z.shape[0]; dev = z.device
    if mode == "regress":
        return model(torch.zeros(B, tp.M, 3, device=dev), torch.zeros(B, tp.E, 6, device=dev),
                     torch.zeros(B, device=dev), z)
    xv = torch.randn(B, tp.M, 3, device=dev, generator=gen)
    xo = torch.randn(B, tp.E, 6, device=dev, generator=gen)
    for i in range(steps):
        t = torch.full((B,), i / steps, device=dev)
        vv, vo = model(xv, xo, t, z)
        xv, xo = xv + vv / steps, xo + vo / steps
    if not model.curves:                     # verts-only model: mean-template curves
        xo = torch.zeros_like(xo)
    return xv, xo


def bezier_pts(V, O, edges, ts=(0.25, 0.5, 0.75)):
    """V [B,M,3], O [B,E,6] -> [B,E,len(ts),3]."""
    P0, P1 = V[:, edges[:, 0]], V[:, edges[:, 1]]
    L = (P1 - P0).norm(dim=-1, keepdim=True)
    B1 = (2 * P0 + P1) / 3 + O[..., :3] * L
    B2 = (P0 + 2 * P1) / 3 + O[..., 3:] * L
    t = torch.tensor(ts, device=V.device)[:, None]
    return ((1 - t) ** 3 * P0[:, :, None] + 3 * (1 - t) ** 2 * t * B1[:, :, None]
            + 3 * (1 - t) * t ** 2 * B2[:, :, None] + t ** 3 * P1[:, :, None])


def errors(tp, Vp, Op, Vg, Og):
    """per-item corner / curve / curve_gtc errors (dataset units * 1000)."""
    e = tp.edges
    corner = (Vp - Vg).norm(dim=-1).mean(1) * 1000
    curve = (bezier_pts(Vp, Op, e) - bezier_pts(Vg, Og, e)).norm(dim=-1).mean((1, 2)) * 1000
    gtc = (bezier_pts(Vg, Op, e) - bezier_pts(Vg, Og, e)).norm(dim=-1).mean((1, 2)) * 1000
    return corner, curve, gtc


@torch.no_grad()
def evaluate(model, items, mode, steps, n_samples, dev, bs=16, seed=1234):
    model.eval(); tp = model.topo
    acc = {k: [] for k in ("corner", "curve", "curve_gtc", "corner_ensmean", "curve_ensmean",
                           "corner_bestofn", "spread", "corner_tpl", "curve_tpl")}
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        pts, pmask = batch_points(chunk, dev)
        z = model.enc(pts, pmask)
        xv_g, xo_g = (x.to(dev) for x in tp.encode(chunk))
        Vg, Og = tp.decode(xv_g, xo_g)
        runs = [sample(model, z, mode, steps, gen) for _ in range(1 if mode == "regress" else n_samples)]
        dec = [tp.decode(*r) for r in runs]
        c, cu, g = errors(tp, *dec[0], Vg, Og)
        acc["corner"] += c.tolist(); acc["curve"] += cu.tolist(); acc["curve_gtc"] += g.tolist()
        Vm = torch.stack([d[0] for d in dec]).mean(0); Om = torch.stack([d[1] for d in dec]).mean(0)
        c, cu, _ = errors(tp, Vm, Om, Vg, Og)
        acc["corner_ensmean"] += c.tolist(); acc["curve_ensmean"] += cu.tolist()
        allc = torch.stack([errors(tp, *d, Vg, Og)[0] for d in dec])
        acc["corner_bestofn"] += allc.min(0).values.tolist()
        acc["spread"] += (torch.stack([d[0] for d in dec]) - Vm).norm(dim=-1).mean((0, 2)).mul(1000).tolist()
        Vt = tp.vmean.expand_as(Vg); Ot = tp.omean.expand_as(Og)
        c, cu, _ = errors(tp, Vt, Ot, Vg, Og)
        acc["corner_tpl"] += c.tolist(); acc["curve_tpl"] += cu.tolist()
    return {k: float(np.median(v)) for k, v in acc.items()} | {"n": len(items)}


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("flow", "regress"), default="flow")
    ap.add_argument("--parts", choices=("verts", "all"), default="all")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--n-latent", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--steps", type=int, default=32, help="Euler steps at sampling")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--curve-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=1.0)
    ap.add_argument("--eval-train", type=int, default=34)
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = "cuda"
    data = torch.load(a.data, weights_only=False)
    tr, va = data["train"], data["val"]
    if a.train_frac < 1.0:
        tr = [tr[i] for i in sorted(rng.choice(len(tr), max(1, round(a.train_frac * len(tr))), replace=False))]
    assert len({it["topo"] for it in tr + va + data.get("val_new", [])}) == 1, "one topology per run (stage A not implemented yet)"
    rmax = float(data["r_bounds"][1]); z0, z1 = (float(v) for v in data["z_bounds"])
    lo, hi = torch.tensor([-rmax, -rmax, z0]), torch.tensor([rmax, rmax, z1])
    tp = Topology(tr, lo, hi).to(dev)
    model = FlowGeo(tp, a.d, a.layers, a.heads, a.n_latent, a.dropout, curves=a.parts == "all").to(dev)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    os.makedirs(a.out, exist_ok=True)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[{a.mode}/{a.parts}] params {n_par/1e6:.2f} M | train {len(tr)} val {len(va)} | M {tp.M} E {tp.E}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * math.ceil(len(tr) / a.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))

    def loss_on(chunk):
        pts, pmask = batch_points(chunk, dev)
        x1v, x1o = (x.to(dev) for x in tp.encode(chunk))
        B = len(chunk)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model.enc(pts, pmask)
            if a.mode == "flow":
                t = torch.sigmoid(torch.randn(B, device=dev))
                x0v, x0o = torch.randn_like(x1v), torch.randn_like(x1o)
                tv = t[:, None, None]
                pv, po = model((1 - tv) * x0v + tv * x1v, (1 - tv) * x0o + tv * x1o, t, z)
                tgt_v, tgt_o = x1v - x0v, x1o - x0o
            else:
                pv, po = model(torch.zeros_like(x1v), torch.zeros_like(x1o), torch.zeros(B, device=dev), z)
                tgt_v, tgt_o = x1v, x1o
        lv = F.mse_loss(pv.float(), tgt_v)
        lo_ = F.mse_loss(po.float(), tgt_o) if a.parts == "all" else torch.zeros((), device=dev)
        return lv, lo_

    hist, t0 = [], time.time()
    for ep in range(1, a.epochs + 1):
        model.train(); tot = np.zeros(2); order = rng.permutation(len(tr))
        for i in range(0, len(tr), a.bs):
            lv, lo_ = loss_on([tr[j] for j in order[i:i + a.bs]])
            loss = lv + a.curve_weight * lo_
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - a.ema)
            tot += [float(lv.detach()), float(lo_.detach())]
        if ep % a.eval_every == 0 or ep == a.epochs:
            nb = math.ceil(len(tr) / a.bs)
            ev = evaluate(ema, va, a.mode, a.steps, 2, dev)
            hist.append({"epoch": ep, "loss_v": tot[0] / nb, "loss_curve": tot[1] / nb, **ev})
            print(f"[{a.mode}/{a.parts}] ep {ep:5d} loss v {tot[0]/nb:.4f} c {tot[1]/nb:.4f} | val corner "
                  f"{ev['corner']:.1f} curve {ev['curve']:.1f} gtc {ev['curve_gtc']:.1f} (tpl {ev['corner_tpl']:.1f}/"
                  f"{ev['curve_tpl']:.1f}) ({time.time()-t0:.0f}s)", flush=True)
    torch.save({"model": ema.state_dict(), "args": vars(a)}, f"{a.out}/last.pt")
    summ = {"val": evaluate(ema, va, a.mode, a.steps, a.n_samples, dev)}
    if data.get("val_new"):                  # second, held-out set of new geometries
        summ["val_new"] = evaluate(ema, data["val_new"], a.mode, a.steps, a.n_samples, dev)
    if a.eval_train:
        summ["train"] = evaluate(ema, tr[:a.eval_train], a.mode, a.steps, a.n_samples, dev)
    summ.update(mode=a.mode, parts=a.parts, params_M=n_par / 1e6, n_train=len(tr), seed=a.seed,
                epochs=a.epochs, d=a.d, layers=a.layers, train_s=time.time() - t0)
    print(f"[{a.mode}/{a.parts}] RESULT {json.dumps(summ)}", flush=True)
    json.dump({"summary": summ, "history": hist}, open(f"{a.out}/result.json", "w"), indent=1)


if __name__ == "__main__":
    main()
