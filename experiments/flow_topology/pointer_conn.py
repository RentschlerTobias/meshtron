#!/usr/bin/env python3
"""Stage 2 of VARIABLE_TOPOLOGY_PLAN.md: blocks as pointers into an UNORDERED corner set (M3).

Polytron's PointerNet (train.py, task conn) with the vertex-index position embedding removed: the corners are a
set (shuffled per batch), so the model cannot memorise "vertex 21". Per block 8 pointers in VTK corner order,
blocks in the canonical order of the data (meshtron_topo row plan), then a stop pointer.

  - inputs: corners (normalised, + Gaussian noise in training: stage-1 corners deviate from GT) and the
    PointPerceiver latents of the labelled point cloud;
  - decoding with validity masks: no corner twice in a block, stop only at a block boundary, a block whose
    face would get a third block is not allowed (checked when its 4th / 8th corner is chosen);
  - evaluation on GT corners, GT + noise and stage-1 corners (flow_vertex.py samples): predicted corners are
    matched to GT by Hungarian assignment, a topology is exact when the set of blocks (as corner sets) equals GT.

    python pointer_conn.py --data data/flow_canon_full.pt --vtx-ckpt runs/vtx_full_s0/best.pt --out runs/ptr_s0
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
from flow_geo_var import Norm  # noqa: E402
from model import DecBlock, PointPerceiver  # noqa: E402
from train import batch_points, pad  # noqa: E402

FACES = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))


def fourier(x, n=6):
    f = (2.0 ** torch.arange(n, device=x.device)) * math.pi
    a = x[..., None] * f
    return torch.cat([x, a.sin().flatten(-2), a.cos().flatten(-2)], -1)


class SetPointer(nn.Module):
    def __init__(self, d=192, layers=6, heads=6, max_len=1024, n_latent=32, dropout=0.1, point_dim=14):
        super().__init__()
        self.venc = nn.Sequential(nn.Linear(3 * 13, d), nn.GELU(), nn.Linear(d, d))
        self.enc = PointPerceiver(point_dim, d, heads, n_latent, 2)
        self.vblocks = nn.ModuleList(nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True,
                                                                norm_first=True) for _ in range(3))
        self.vcross = nn.ModuleList(DecBlock(d, heads, dropout) for _ in range(1))
        self.stop = nn.Parameter(torch.randn(d) * 0.02)
        self.start = nn.Parameter(torch.randn(d) * 0.02)
        self.pos = nn.Embedding(max_len, d)                  # position in the pointer SEQUENCE (not vertex id)
        self.corner = nn.Embedding(8, d)
        self.dec = nn.ModuleList(DecBlock(d, heads, dropout) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)

    def memory(self, V, vmask, pts, pmask):
        z = self.enc(pts, pmask)
        e = self.venc(fourier(V))
        for blk in self.vblocks:
            e = blk(e, src_key_padding_mask=~vmask)
        return e, z

    def logits(self, e, z, vmask, ptr_in):
        B = e.shape[0]
        mem = torch.cat([e, z], 1)
        E = torch.cat([e, self.stop[None, None].expand(B, 1, -1)], 1)
        L = ptr_in.shape[1]
        g = torch.where(ptr_in < 0, torch.zeros_like(ptr_in), ptr_in)
        h = torch.gather(E, 1, g[..., None].expand(-1, -1, E.shape[-1])).clone()
        h[:, 0] = self.start
        ar = torch.arange(L, device=e.device)
        h = h + self.pos(ar)[None] + self.corner((ar - 1).clamp(min=0) % 8)[None]
        for blk in self.dec:
            h = blk(h, mem)
        lg = torch.einsum("bld,bmd->blm", self.q(self.ln(h)), E) / math.sqrt(E.shape[-1])
        valid = torch.cat([vmask, torch.ones(B, 1, dtype=torch.bool, device=e.device)], 1)
        return lg.masked_fill(~valid[:, None], -1e9)


def make_batch(items, nrm, dev, sigma=0.0, rng=None):
    """Shuffled corners (+ noise), pointer input / target sequences in the shuffled ids (stop = index M_max)."""
    Vs, tgs = [], []
    for it in items:
        M = len(it["verts"])
        perm = rng.permutation(M) if rng is not None else np.arange(M)    # new position -> old id
        inv = np.empty(M, np.int64); inv[perm] = np.arange(M)                 # old id -> new position
        V = it["verts"][torch.as_tensor(perm)]
        Vs.append(V); tgs.append(torch.as_tensor(inv)[it["conn"].long().reshape(-1)])
    V = pad(Vs, 0.0, torch.float32).to(dev)
    vmask = pad([torch.ones(len(v), dtype=torch.bool) for v in Vs], False, torch.bool).to(dev)
    x = nrm.enc_v(V)
    if sigma > 0:
        x = x + sigma * torch.randn_like(x)
    Mx = V.shape[1]
    t = pad([torch.cat([g, torch.tensor([Mx])]) for g in tgs], -100).to(dev)
    ptr_in = torch.cat([torch.full((len(items), 1), -1, device=dev), t[:, :-1]], 1)
    ptr_in[ptr_in == -100] = 0
    return x, vmask, ptr_in, t


@torch.no_grad()
def decode(model, e, z, vmask, max_blocks=256):
    """Greedy pointer decoding of ONE item with validity masks -> list of blocks (8 local ids each)."""
    M = int(vmask[0].sum()); dev = e.device
    max_blocks = min(max_blocks, (model.pos.num_embeddings - 2) // 8)
    seq, blocks, face_count = [-1], [], {}
    for _ in range(8 * max_blocks + 1):
        lg = model.logits(e, z, vmask, torch.tensor([seq], device=dev))[0, -1].clone()
        pos = (len(seq) - 1) % 8
        cur = seq[len(seq) - pos:] if pos else []
        if pos:                                            # inside a block: no stop, no repeated corner
            lg[M] = -1e9
            for c in cur:
                lg[c] = -1e9
        nxt = int(lg[:M + 1].argmax())
        if nxt == M:
            break
        seq.append(nxt)
        if (len(seq) - 1) % 8 == 0:
            b = seq[-8:]
            keys = [tuple(sorted(b[i] for i in f)) for f in FACES]
            if any(face_count.get(k, 0) >= 2 for k in keys):  # third block on a face: drop the block, stop
                seq = seq[:-8]
                break
            for k in keys:
                face_count[k] = face_count.get(k, 0) + 1
            blocks.append(b)
    return blocks


def topo_exact(blocks, gt_conn, match):
    """match: predicted corner id -> GT corner id. Exact when the block corner sets coincide."""
    pred = {frozenset(match.get(c, -1) for c in b) for b in blocks}
    gt = {frozenset(int(c) for c in b) for b in gt_conn}
    return pred == gt and len(blocks) == len(gt_conn)


@torch.no_grad()
def evaluate(model, nrm, items, dev, sigma_eval=0.0, vtx=None, seed=1234):
    """Exact-topology rate with corners = GT (sigma 0), GT + noise, or stage-1 samples (vtx = (model, nrm))."""
    model.eval()
    rng = np.random.default_rng(seed); gen = torch.Generator(device=dev); gen.manual_seed(seed)
    n_exact, n_blocks = 0, []
    for it in items:
        G = it["verts"].double().numpy()
        pts, pmask = batch_points([it], dev)
        if vtx is None:
            perm = rng.permutation(len(G))
            P = G[perm] + (sigma_eval * float(nrm.s.mean()) * rng.standard_normal(G.shape) if sigma_eval else 0)
        else:
            vm, vn = vtx
            import flow_vertex as FVX
            zv = vm.enc(pts, pmask)
            k = int(vm.count_logits(zv).argmax(-1)[0])
            P = vn.dec_v(FVX.sample(vm, zv, [k], 32, gen)[0]).cpu().double().numpy()
        D = np.linalg.norm(P[:, None] - G[None], axis=2)
        r, c = linear_sum_assignment(D)
        match = {int(i): int(j) for i, j in zip(r, c)}
        x = nrm.enc_v(torch.tensor(P, dtype=torch.float32, device=dev))[None]
        vmask = torch.ones(1, len(P), dtype=torch.bool, device=dev)
        e, z = model.memory(x, vmask, pts, pmask)
        blocks = decode(model, e, z, vmask)
        n_exact += topo_exact(blocks, it["conn"].long().numpy(), match)
        n_blocks.append(len(blocks))
    return {"exact": n_exact / len(items), "n": len(items), "blocks_median": float(np.median(n_blocks))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vtx-ckpt", default=None, help="stage-1 model for the evaluation on generated corners")
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--sigma", type=float, default=0.04, help="corner noise in training (dataset units)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = "cuda"
    data = torch.load(a.data, weights_only=False)
    tr, va = data["train"], data["val"]
    nrm = Norm(data, tr).to(dev)
    sig = a.sigma / float(nrm.s.mean())
    max_len = max(it["conn"].numel() for it in tr) * 2 + 2
    model = SetPointer(a.d, a.layers, a.heads, max_len).to(dev)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    os.makedirs(a.out, exist_ok=True)
    print(f"[ptr] params {sum(p.numel() for p in model.parameters())/1e6:.2f} M | train {len(tr)} | sigma {a.sigma}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * math.ceil(len(tr) / a.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 300) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
    vtx = None
    if a.vtx_ckpt:
        import flow_vertex as FVX
        vm, vn, _ = FVX.load(a.vtx_ckpt, data)
        vtx = (vm, vn)
    t0, hist = time.time(), []
    for ep in range(1, a.epochs + 1):
        model.train(); tot = 0.0; order = rng.permutation(len(tr))
        for i in range(0, len(tr), a.bs):
            chunk = [tr[j] for j in order[i:i + a.bs]]
            x, vmask, ptr_in, t = make_batch(chunk, nrm, dev, sig, rng)
            pts, pmask = batch_points(chunk, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                e, z = model.memory(x, vmask, pts, pmask)
                lg = model.logits(e, z, vmask, ptr_in)
            loss = F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), t.reshape(-1), ignore_index=-100)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - 0.999)
            tot += float(loss.detach())
        if ep % a.eval_every == 0 or ep == a.epochs:
            ev = {"gt": evaluate(ema, nrm, va, dev)["exact"], "noise": evaluate(ema, nrm, va, dev, a.sigma)["exact"]}
            if vtx:
                ev["stage1"] = evaluate(ema, nrm, va, dev, vtx=vtx)["exact"]
            hist.append({"epoch": ep, "loss": tot / math.ceil(len(tr) / a.bs), **ev})
            print(f"[ptr] ep {ep:4d} loss {hist[-1]['loss']:.4f} | val exact {ev} ({time.time()-t0:.0f}s)", flush=True)
    torch.save({"model": ema.state_dict(), "args": vars(a), "max_len": max_len}, f"{a.out}/last.pt")
    summ = {}
    for sp in [k for k in data if k.startswith("val") and isinstance(data[k], list) and data[k]]:
        summ[sp] = {"gt": evaluate(ema, nrm, data[sp], dev), "noise": evaluate(ema, nrm, data[sp], dev, a.sigma)}
        if vtx:
            summ[sp]["stage1"] = evaluate(ema, nrm, data[sp], dev, vtx=vtx)
    summ["train_s"] = time.time() - t0
    print(f"[ptr] RESULT {json.dumps(summ)}", flush=True)
    json.dump({"summary": summ, "history": hist}, open(f"{a.out}/result.json", "w"), indent=1)


if __name__ == "__main__":
    main()
