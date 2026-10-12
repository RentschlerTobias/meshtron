"""Polytron stage 1 with continuous / ordinal-aware coordinate heads (prototype).

Compares output heads for the vertex task of train.py (each block corner once,
canonical order) on the same data, encoder and decoder:

  hlgauss   train.py token model (3 tokens/vertex, 512 bins), but cross-entropy
            against Gaussian-smoothed targets centred on the TRUE continuous
            coordinate (HL-Gauss, Farebrother et al. 2024). Only the loss changes.
  bins      one decoder step per vertex; input = Fourier-encoded previous vertex,
            output = 3 independent categorical heads over --bins bins.
            --bins 512 = per-vertex counterpart of the token model.
  binoff    as bins, plus a regressed offset inside the bin (per-bin offsets,
            L1 loss on the offset of the GT bin, LATO.2-style); use --bins 64.
  gmm       one step per vertex, diagonal Gaussian mixture over (x, y, z)
            (GIVT / Q-FAT); decoding = mean of the most probable component.
  reg       one step per vertex, plain smooth-L1 regression (deterministic).

All per-vertex variants add a stop head (BCE). Coordinates are normalised to
[-1, 1] with the tokenizer bounds (x, y in [-R_MAX, R_MAX], z in [Z_MIN, Z_MAX]),
so bin b of B corresponds to the same interval as in the token model when B = 512.
Evaluation reports the same corner metric as train.py (dataset units * 1000).

    python train_cont.py --variant gmm --data data/v2_canon.pt --out runs/cont_gmm_s0 \
        --d 128 --layers 2 --heads 4 --n-latent 32 --epochs 400 --select last --eval-train 34
"""
import argparse
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
sys.path.insert(0, "/opt/stack/meshtron")
from model import BlockGen, DecBlock, PointPerceiver  # noqa: E402
from train import batch_points, evaluate_generation, token_batch  # noqa: E402

Q_TOKEN = 512   # bins of the token model (PolytronTokenizer.Qr)


# ----------------------------------------------------------------------- normalisation
def bounds(data):
    rmax = float(data["r_bounds"][1]); z0, z1 = (float(v) for v in data["z_bounds"])
    return torch.tensor([-rmax, -rmax, z0]), torch.tensor([rmax, rmax, z1])


def to_unit(v, lo, hi):        # dataset units -> [-1, 1]
    return (v - lo) / (hi - lo) * 2 - 1


def from_unit(c, lo, hi):
    return (c + 1) / 2 * (hi - lo) + lo


# ----------------------------------------------------------------------- per-vertex model
class FourierIn(nn.Module):
    """[...,3] in [-1,1] -> [...,d]: raw coords + sin/cos at n_freq octaves, then MLP."""

    def __init__(self, d, n_freq=8):
        super().__init__()
        self.register_buffer("freq", (2.0 ** torch.arange(n_freq)) * math.pi)
        self.mlp = nn.Sequential(nn.Linear(3 + 6 * n_freq, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, c):
        a = c[..., None] * self.freq                                   # [...,3,F]
        f = torch.cat([c, a.sin().flatten(-2), a.cos().flatten(-2)], -1)
        return self.mlp(f)


class VertGen(nn.Module):
    """Decoder over vertices: step i reads vertex i-1 (start embedding at i=0)
    and predicts vertex i, or stop at i = M."""

    def __init__(self, variant, d, layers, heads, max_len, dropout=0.1, n_latent=64,
                 point_dim=11, bins=512, k=8):
        super().__init__()
        self.variant, self.bins, self.k = variant, bins, k
        self.enc = PointPerceiver(point_dim, d, heads, n_latent, 2)
        self.vin = FourierIn(d)
        self.start = nn.Parameter(torch.randn(d) * 0.02)
        self.pos = nn.Embedding(max_len, d)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.blocks = nn.ModuleList(DecBlock(d, heads, dropout) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
        n_out = {"bins": 3 * bins, "binoff": 6 * bins, "gmm": k + 6 * k, "reg": 3}[variant]
        self.head = nn.Linear(d, n_out)
        self.stop = nn.Linear(d, 1)

    def encode(self, pts, mask):
        return self.enc(pts, mask)

    def decode(self, c_in, z):
        """c_in [B,L,3] normalised previous vertices (row 0 ignored -> start)."""
        B, L, _ = c_in.shape
        h = self.vin(c_in)
        h = torch.cat([self.start[None, None].expand(B, 1, -1), h[:, 1:]], 1)
        h = self.drop(h + self.pos(torch.arange(L, device=c_in.device))[None])
        for b in self.blocks:
            h = b(h, z)
        h = self.ln(h)
        return self.head(h).float(), self.stop(h).squeeze(-1).float()

    # ---- head-specific loss / point estimate
    def split_gmm(self, out):
        K = self.k
        logit = out[..., :K]
        mu = out[..., K:4 * K].unflatten(-1, (K, 3))
        log_s = out[..., 4 * K:].unflatten(-1, (K, 3)).clamp(-9.0, 2.0)
        return logit, mu, log_s

    def coord_loss(self, out, c):
        """out [N,n_out], c [N,3] targets in [-1,1]."""
        B = self.bins
        if self.variant == "reg":
            return F.smooth_l1_loss(out, c, beta=0.01)
        if self.variant == "gmm":
            logit, mu, log_s = self.split_gmm(out)
            lp = (-0.5 * ((c[:, None] - mu) / log_s.exp()) ** 2 - log_s - 0.5 * math.log(2 * math.pi)).sum(-1)
            return -(torch.logsumexp(F.log_softmax(logit, -1) + lp, -1)).mean() / 3
        u = (c + 1) / 2 * (B - 1)                                   # continuous bin coordinate
        tgt = u.round().clamp(0, B - 1).long()
        lg = out[..., :3 * B].unflatten(-1, (3, B))
        loss = F.cross_entropy(lg.reshape(-1, B), tgt.reshape(-1))
        if self.variant == "binoff":
            off = torch.tanh(out[..., 3 * B:].unflatten(-1, (3, B))) * 0.5
            pred = torch.gather(off, -1, tgt[..., None]).squeeze(-1)
            loss = loss + F.l1_loss(pred, u - tgt) * 4.0          # offset in bins, |.| <= 0.5
        return loss

    def point(self, out):
        """out [...,n_out] -> deterministic point estimate in [-1,1]."""
        B = self.bins
        if self.variant == "reg":
            return out.clamp(-1, 1)
        if self.variant == "gmm":
            logit, mu, _ = self.split_gmm(out)
            j = logit.argmax(-1)
            return torch.gather(mu, -2, j[..., None, None].expand(*j.shape, 1, 3)).squeeze(-2).clamp(-1, 1)
        lg = out[..., :3 * B].unflatten(-1, (3, B))
        b = lg.argmax(-1)
        u = b.float()
        if self.variant == "binoff":
            off = torch.tanh(out[..., 3 * B:].unflatten(-1, (3, B))) * 0.5
            u = u + torch.gather(off, -1, b[..., None]).squeeze(-1)
        return u / (B - 1) * 2 - 1


def vert_batch(items, lo, hi, dev):
    """inputs [B,L,3] (row 0 = start placeholder), targets [B,L,3], tmask [B,L]
    (coordinate target present), stop target [B,L], smask [B,L]."""
    cs = [to_unit(it["verts"].float(), lo, hi) for it in items]
    L = max(len(c) for c in cs) + 1
    cin = torch.zeros(len(items), L, 3); tgt = torch.zeros(len(items), L, 3)
    tm = torch.zeros(len(items), L, dtype=torch.bool); st = torch.zeros(len(items), L)
    sm = torch.zeros(len(items), L, dtype=torch.bool)
    for i, c in enumerate(cs):
        M = len(c)
        cin[i, 1:M + 1] = c; tgt[i, :M] = c; tm[i, :M] = True; st[i, M] = 1.0; sm[i, :M + 1] = True
    return cin.to(dev), tgt.to(dev), tm.to(dev), st.to(dev), sm.to(dev)


@torch.no_grad()
def evaluate_vert(model, items, lo, hi, dev, bs=8):
    model.eval(); res = []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        pts, pmask = batch_points(chunk, dev)
        z = model.encode(pts, pmask)
        max_len = max(len(it["verts"]) for it in chunk) + 20
        c = torch.zeros(len(chunk), 1, 3, device=dev)
        n_v = torch.full((len(chunk),), -1)
        for step in range(max_len):
            out, sl = model.decode(c, z)
            stop = (sl[:, -1] > 0).cpu()
            for b in range(len(chunk)):
                if n_v[b] < 0 and stop[b]:
                    n_v[b] = step
            if (n_v >= 0).all():
                break
            c = torch.cat([c, model.point(out[:, -1])[:, None]], 1)
        for b, it in enumerate(chunk):
            r = {"name": it["name"], "valid": bool(n_v[b] >= 0)}
            r["struct_ok"] = r["valid"] and int(n_v[b]) == len(it["verts"])
            if r["struct_ok"]:
                V = from_unit(c[b, 1:int(n_v[b]) + 1].cpu(), lo, hi).numpy()
                d = np.linalg.norm(V - it["verts"].numpy(), axis=1) * 1000
                r["corner_mean"], r["corner_max"] = float(d.mean()), float(d.max())
            res.append(r)
    ok = [r for r in res if r["struct_ok"]]
    return {"n": len(res), "valid": sum(r["valid"] for r in res), "struct_ok": len(ok),
            "corner_mean_median": float(np.median([r["corner_mean"] for r in ok])) if ok else None,
            "corner_max_median": float(np.median([r["corner_max"] for r in ok])) if ok else None}, res


# ----------------------------------------------------------------------- HL-Gauss token targets
def hlgauss_targets(chunk, y, lo, hi, sigma, vocab, dev):
    """Soft targets [B,L,vocab] for the token model: coordinate positions get a
    Gaussian (in bins) around the true continuous coordinate, specials stay one-hot."""
    Bsz, L = y.shape
    p = torch.zeros(Bsz, L, vocab, device=dev)
    valid = y != -100
    p[valid] = F.one_hot(y[valid], vocab).float()
    edges = torch.arange(Q_TOKEN + 1, device=dev, dtype=torch.float32) - 0.5
    for i, it in enumerate(chunk):
        u = ((to_unit(it["verts"].float(), lo, hi) + 1) / 2 * (Q_TOKEN - 1)).reshape(-1).to(dev)   # [3M]
        cdf = torch.special.ndtr((edges[None] - u[:, None]) / sigma)
        g = cdf[:, 1:] - cdf[:, :-1]
        g = g / g.sum(-1, keepdim=True)
        n = len(u)                                    # positions 0..n-1 of y are the coordinates
        p[i, :n] = 0.0
        p[i, :n, :Q_TOKEN] = g
    return p, valid


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("hlgauss", "bins", "binoff", "gmm", "reg"), required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--n-latent", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--select", choices=("best", "last"), default="last")
    ap.add_argument("--train-frac", type=float, default=1.0)
    ap.add_argument("--eval-train", type=int, default=34)
    ap.add_argument("--bins", type=int, default=512, help="bins/binoff: bins per coordinate")
    ap.add_argument("--k", type=int, default=8, help="gmm: mixture components")
    ap.add_argument("--sigma", type=float, default=3.0, help="hlgauss: target std in bins")
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = "cuda"
    data = torch.load(a.data, weights_only=False)
    tr, va = data["train"], data["val"]
    if a.train_frac < 1.0:
        tr = [tr[i] for i in sorted(rng.choice(len(tr), max(1, round(a.train_frac * len(tr))), replace=False))]
    lo, hi = bounds(data)
    sp = data["special"]; specials = set(sp.values())
    os.makedirs(a.out, exist_ok=True)
    tok_model = a.variant == "hlgauss"
    if tok_model:
        max_len = max(len(it["vtok"]) for it in tr + va) + 64
        model = BlockGen(data["vocab"], a.d, a.layers, a.heads, max_len, sp["pad"], a.dropout, a.n_latent).to(dev)
    else:
        max_len = max(len(it["verts"]) for it in tr + va) + 32
        model = VertGen(a.variant, a.d, a.layers, a.heads, max_len, a.dropout, a.n_latent,
                        bins=a.bins, k=a.k).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[{a.variant}] params {n_par/1e6:.2f} M | train {len(tr)} val {len(va)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * math.ceil(len(tr) / a.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))

    def loss_on(items, train):
        model.train(train)
        tot, n = 0.0, 0
        order = rng.permutation(len(items)) if train else np.arange(len(items))
        for i in range(0, len(items), a.bs):
            chunk = [items[j] for j in order[i:i + a.bs]]
            pts, pmask = batch_points(chunk, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16), torch.set_grad_enabled(train):
                if tok_model:
                    x, y, sl = token_batch(chunk, "vtok", sp["pad"], specials, dev)
                    lg = model(x, sl, pts, pmask).float()
                    p, m = hlgauss_targets(chunk, y, lo, hi, a.sigma, lg.shape[-1], dev)
                    loss = -(p * F.log_softmax(lg, -1)).sum(-1)[m].mean()
                else:
                    cin, tgt, tm, st, sm = vert_batch(chunk, lo, hi, dev)
                    out, sl = model.decode(cin, model.encode(pts, pmask))
                    loss = model.coord_loss(out[tm], tgt[tm]) + \
                        F.binary_cross_entropy_with_logits(sl[sm], st[sm])
            if train:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tot += float(loss) * len(chunk); n += len(chunk)
        return tot / n

    def evaluate(items):
        if tok_model:
            return evaluate_generation("vertex", model, data, items, dev)
        return evaluate_vert(model, items, lo, hi, dev)

    best, hist, t0 = float("inf"), [], time.time()
    for ep in range(1, a.epochs + 1):
        trl = loss_on(tr, True)
        if ep % a.eval_every == 0 or ep == a.epochs:
            vl = loss_on(va, False)
            hist.append({"epoch": ep, "train_loss": trl, "val_loss": vl})
            flag = ""
            if vl < best - 1e-4:
                best, flag = vl, " *"
                torch.save({"model": model.state_dict(), "args": vars(a), "epoch": ep, "val_loss": vl}, f"{a.out}/best.pt")
            print(f"[{a.variant}] ep {ep:4d} train {trl:.4f} val {vl:.4f} ({time.time()-t0:.0f}s){flag}", flush=True)
    torch.save({"model": model.state_dict(), "args": vars(a), "epoch": ep, "val_loss": vl}, f"{a.out}/last.pt")
    ck = torch.load(f"{a.out}/{a.select}.pt", weights_only=False); model.load_state_dict(ck["model"]); model.eval()
    summ, rows = evaluate(va)
    if a.eval_train:
        summ["train_eval"], _ = evaluate(tr[:a.eval_train])
    summ.update(variant=a.variant, select=a.select, n_train=len(tr), params_M=n_par / 1e6, best_epoch=ck["epoch"],
                best_val_loss=ck["val_loss"], seed=a.seed, bins=a.bins, k=a.k, sigma=a.sigma, train_s=time.time() - t0)
    print(f"[{a.variant}] RESULT {json.dumps(summ)}", flush=True)
    json.dump({"summary": summ, "rows": rows, "history": hist}, open(f"{a.out}/result.json", "w"), indent=1)


if __name__ == "__main__":
    main()
