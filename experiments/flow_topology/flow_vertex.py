#!/usr/bin/env python3
"""Stage 1 of VARIABLE_TOPOLOGY_PLAN.md: block corners as an unordered SET of unknown size, by flow matching.

  - count head: the number of corners N is predicted from the point cloud (classification over 1..n_max);
    at inference the argmax (or the top-k for ranking downstream) fixes the number of tokens;
  - corner tokens carry no identity (no template, no topology features): N tokens start as Gaussian noise in the
    normalised dataset box, a set transformer (self-attention among the tokens, cross-attention to the
    PointPerceiver latents, adaLN time conditioning) moves them to the corners;
  - training: rectified flow with optimal-transport coupling per item (Hungarian matching of the noise points to
    the GT corners, as in MeshFlow), otherwise the target of a token is ambiguous and the model learns means;
  - evaluation: corner error after Hungarian matching of prediction and GT (dataset units * 1000), count accuracy.

    python flow_vertex.py --data data/flow_canon_papaya.pt --out runs/vtx_s0
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
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flow_geo import DiTBlock, modulate  # noqa: E402
from flow_geo_var import Norm  # noqa: E402
from model import PointPerceiver  # noqa: E402
from train import batch_points, pad  # noqa: E402


class SetAttnBlock(DiTBlock):
    """DiT block whose self-attention ignores padded tokens."""

    def forward(self, x, z, c, kmask=None):
        if kmask is None:
            return super().forward(x, z, c)
        s1, b1, g1, s2, b2, g2 = self.ada(F.silu(c))[:, None].chunk(6, -1)
        h = modulate(self.ln1(x), b1, s1)
        x = x + g1 * self.drop(self.sa(h, h, kmask))
        x = x + self.drop(self.ca(self.ln2(x), z))
        return x + g2 * self.drop(self.ff(modulate(self.ln3(x), b2, s2)))


class FlowVertex(nn.Module):
    def __init__(self, d=128, layers=6, heads=4, n_latent=32, n_max=256, dropout=0.0, point_dim=14):
        super().__init__()
        self.n_max = n_max
        self.enc = PointPerceiver(point_dim, d, heads, n_latent, 2)
        self.count = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, n_max + 1))
        self.vin = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        self.tmlp = nn.Sequential(nn.Linear(64, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList(SetAttnBlock(d, heads, dropout) for _ in range(layers))
        self.lnf = nn.LayerNorm(d, elementwise_affine=False)
        self.adaf = nn.Linear(d, 2 * d)
        self.vout = nn.Linear(d, 3)
        for m in (self.adaf, self.vout):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def count_logits(self, z):
        return self.count(torch.cat([z.mean(1), z.max(1).values], -1))

    def temb(self, t):
        f = torch.exp(-math.log(1000.0) * torch.arange(32, device=t.device) / 32)
        a = t[:, None] * 1000.0 * f
        return self.tmlp(torch.cat([a.sin(), a.cos()], -1))

    def forward(self, xv, t, z, vmask):
        h = self.vin(xv)
        c = self.temb(t) + z.mean(1)
        for blk in self.blocks:
            h = blk(h, z, c, vmask)
        sh, sc = self.adaf(F.silu(c))[:, None].chunk(2, -1)
        return self.vout(modulate(self.lnf(h), sh, sc))


@torch.no_grad()
def sample(model, z, n, steps, gen=None):
    """n: list of corner counts per item -> list of [n_i,3] normalised corners."""
    dev = z.device
    vmask = pad([torch.ones(k, dtype=torch.bool) for k in n], False, torch.bool).to(dev)
    xv = torch.randn(len(n), vmask.shape[1], 3, device=dev, generator=gen)
    for i in range(steps):
        t = torch.full((len(n),), i / steps, device=dev)
        xv = xv + model(xv, t, z, vmask) / steps
    return [xv[b, :k] for b, k in enumerate(n)]


def ot_couple(x0, x1, mask):
    """Permute x1 per item so that it is matched to x0 by minimal total squared distance (padding untouched)."""
    out = x1.clone()
    for b in range(len(x1)):
        k = int(mask[b].sum())
        cost = torch.cdist(x0[b, :k].float(), x1[b, :k].float()).pow(2).cpu().numpy()
        r, c = linear_sum_assignment(cost)
        out[b, r] = x1[b, c]
    return out


def matched_error(P, G):
    """Mean corner distance after Hungarian matching (P, G [n,3] / [m,3] dataset units), unmatched GT corners
    count with their distance to the nearest prediction."""
    D = torch.cdist(P.double(), G.double()).cpu().numpy()
    r, c = linear_sum_assignment(D)
    d = list(D[r, c])
    miss = sorted(set(range(D.shape[1])) - set(c))
    d += [float(D[:, j].min()) for j in miss]
    return 1000 * float(np.mean(d))


@torch.no_grad()
def evaluate(model, nrm, items, steps, dev, bs=16, seed=1234, n_samples=1):
    model.eval()
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    acc = {"corner": [], "corner_gtn": [], "count_ok": [], "corner_best": []}
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        pts, pmask = batch_points(chunk, dev)
        z = model.enc(pts, pmask)
        n_pred = model.count_logits(z).argmax(-1).tolist()
        n_gt = [len(it["verts"]) for it in chunk]
        samp = sample(model, z, [max(1, k) for k in n_pred], steps, gen)
        samp_gt = [sample(model, z, n_gt, steps, gen) for _ in range(n_samples)]
        for j, it in enumerate(chunk):
            G = it["verts"].to(dev)
            acc["corner"].append(matched_error(nrm.dec_v(samp[j]), G))
            errs = [matched_error(nrm.dec_v(s[j]), G) for s in samp_gt]
            acc["corner_gtn"].append(errs[0]); acc["corner_best"].append(min(errs))
            acc["count_ok"].append(float(n_pred[j] == n_gt[j]))
    return {"corner": float(np.median(acc["corner"])), "corner_gtn": float(np.median(acc["corner_gtn"])),
            "corner_best": float(np.median(acc["corner_best"])), "count_acc": float(np.mean(acc["count_ok"])),
            "n": len(items)}


def load(ckpt, data, dev="cuda"):
    ck = torch.load(ckpt, weights_only=False); a = ck["args"]
    nrm = Norm(data, data["train"]).to(dev)
    model = FlowVertex(a["d"], a["layers"], a["heads"], a["n_latent"], a["n_max"]).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    return model, nrm, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--n-latent", type=int, default=32)
    ap.add_argument("--n-max", type=int, default=256)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--count-weight", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = "cuda"
    data = torch.load(a.data, weights_only=False)
    tr, va = data["train"], data["val"]
    nrm = Norm(data, tr).to(dev)
    model = FlowVertex(a.d, a.layers, a.heads, a.n_latent, a.n_max).to(dev)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    os.makedirs(a.out, exist_ok=True)
    n_par = sum(p.numel() for p in model.parameters())
    counts = sorted({len(it["verts"]) for it in tr})
    print(f"[vtx] params {n_par/1e6:.2f} M | train {len(tr)} val {len(va)} | corner counts in train {counts}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * math.ceil(len(tr) / a.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))

    def loss_on(chunk):
        pts, pmask = batch_points(chunk, dev)
        V = pad([it["verts"].float() for it in chunk], 0.0, torch.float32).to(dev)
        vmask = pad([torch.ones(len(it["verts"]), dtype=torch.bool) for it in chunk], False, torch.bool).to(dev)
        n_gt = torch.tensor([len(it["verts"]) for it in chunk], device=dev)
        x1 = nrm.enc_v(V)
        x0 = torch.randn_like(x1)
        x1 = ot_couple(x0, x1, vmask)
        B = len(chunk)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model.enc(pts, pmask)
            t = torch.sigmoid(torch.randn(B, device=dev))
            tv = t[:, None, None]
            pv = model((1 - tv) * x0 + tv * x1, t, z, vmask)
            logits = model.count_logits(z)
        m = vmask[..., None].float()
        lv = ((pv.float() - (x1 - x0)) ** 2 * m).sum() / (m.sum() * 3)
        lc = F.cross_entropy(logits.float(), n_gt)
        return lv, lc

    hist, t0, best = [], time.time(), None
    for ep in range(1, a.epochs + 1):
        model.train(); tot = np.zeros(2); order = rng.permutation(len(tr))
        for i in range(0, len(tr), a.bs):
            lv, lc = loss_on([tr[j] for j in order[i:i + a.bs]])
            loss = lv + a.count_weight * lc
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - a.ema)
            tot += [float(lv.detach()), float(lc.detach())]
        if ep % a.eval_every == 0 or ep == a.epochs:
            nb = math.ceil(len(tr) / a.bs)
            ev = evaluate(ema, nrm, va, a.steps, dev)
            hist.append({"epoch": ep, "loss_v": tot[0] / nb, "loss_count": tot[1] / nb, **ev})
            if best is None or ev["corner_gtn"] < best[0]:
                best = (ev["corner_gtn"], ep)
                torch.save({"model": ema.state_dict(), "args": vars(a), "epoch": ep}, f"{a.out}/best.pt")
            print(f"[vtx] ep {ep:5d} loss v {tot[0]/nb:.4f} n {tot[1]/nb:.4f} | val corner {ev['corner']:.1f} "
                  f"(gt n {ev['corner_gtn']:.1f}) count acc {ev['count_acc']:.2f} ({time.time()-t0:.0f}s)", flush=True)
    torch.save({"model": ema.state_dict(), "args": vars(a), "epoch": a.epochs}, f"{a.out}/last.pt")
    ck = torch.load(f"{a.out}/best.pt", weights_only=False); ema.load_state_dict(ck["model"])
    summ = {"best_epoch": best[1], "val": evaluate(ema, nrm, va, a.steps, dev, n_samples=8)}
    for sp in [k for k in data if k.startswith("val_") and data[k]]:      # val_new, val_full, ...
        summ[sp] = evaluate(ema, nrm, data[sp], a.steps, dev, n_samples=8)
    summ["train"] = evaluate(ema, nrm, tr[:34], a.steps, dev)
    summ.update(params_M=n_par / 1e6, n_train=len(tr), seed=a.seed, epochs=a.epochs, train_s=time.time() - t0)
    print(f"[vtx] RESULT {json.dumps(summ)}", flush=True)
    json.dump({"summary": summ, "history": hist}, open(f"{a.out}/result.json", "w"), indent=1)


if __name__ == "__main__":
    main()
