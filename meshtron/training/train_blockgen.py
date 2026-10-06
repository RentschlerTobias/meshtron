"""Train + evaluate the block generators v2 on a dataset of scripts/blockgen_v2/build_dataset.py.

Tasks:
  hexarow   HexaRow cart tokens (topo order), every corner emitted per block row
  vertex    Polytron stage 1: each block corner once (canonical order), 3 tokens/vertex
  conn      Polytron stage 2: 8 pointers per block into the GT vertex list
            (teacher-forced vertices; at evaluation also on generated ones)

Both token tasks use meshtron.model.blockgen.BlockGen (cross-attention to a Perceiver encoding of
ALL labelled surface points, no block-count input). The pointer task uses
PointerNet below. Early stopping on validation loss; the best checkpoint is
evaluated by generation on the validation geometries:
  valid      the sequence decodes (hexarow: detokenizes; vertex: whole vertices + stop)
  struct_ok  same number of vertices / rows as GT (structure identical)
  corner_mm  mean / max distance of generated corners to the GT corners in
             canonical order (only where struct_ok), in dataset units * 1000

    python -m meshtron.training.train_blockgen --task vertex --data <v2 set>.pt --d 256 --layers 4 \
        --out runs/blockgen_v2/vertex_m

Reports: reports/blockgen_v2_night_grid.md, reports/blockgen_v2_canonical_best_case.md.
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

from meshtron.model.blockgen import BlockGen, slots  # noqa: E402
from meshtron.data.hexa_row_tokenizer import HexaRowTokenizer  # noqa: E402


# ----------------------------------------------------------------------- pointer
class PointerNet(nn.Module):
    """Vertices [B,M,3] (+ cloud latents) -> autoregressive pointers into them."""

    def __init__(self, d, layers, heads, max_len, n_latent=32, dropout=0.1):
        super().__init__()
        from meshtron.model.blockgen import DecBlock, PointPerceiver
        self.venc = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        self.vpos = nn.Embedding(256, d)
        self.enc = PointPerceiver(11, d, heads, n_latent, 1)
        self.vblocks = nn.ModuleList(nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True,
                                                                norm_first=True) for _ in range(2))
        self.stop = nn.Parameter(torch.randn(d) * 0.02)
        self.start = nn.Parameter(torch.randn(d) * 0.02)
        self.pos = nn.Embedding(max_len, d)
        self.corner = nn.Embedding(8, d)
        self.dec = nn.ModuleList(DecBlock(d, heads, dropout) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)

    def forward(self, V, vmask, ptr_in, pts, pmask):
        B, M, _ = V.shape
        z = self.enc(pts, pmask)
        e = self.venc(V) + self.vpos(torch.arange(M, device=V.device))[None]
        for blk in self.vblocks:
            e = blk(e, src_key_padding_mask=~vmask)
        mem = torch.cat([e, z], 1)
        E = torch.cat([e, self.stop[None, None].expand(B, 1, -1)], 1)          # targets: M vertices + stop
        L = ptr_in.shape[1]
        gather = torch.where(ptr_in < 0, torch.zeros_like(ptr_in), ptr_in)
        h = torch.gather(E, 1, gather[..., None].expand(-1, -1, E.shape[-1]))
        h[:, 0] = self.start
        h = h + self.pos(torch.arange(L, device=V.device))[None] + \
            self.corner((torch.arange(L, device=V.device) - 1).clamp(min=0) % 8)[None]
        for blk in self.dec:
            h = blk(h, mem)
        logits = torch.einsum("bld,bmd->blm", self.q(self.ln(h)), E) / math.sqrt(E.shape[-1])
        valid = torch.cat([vmask, torch.ones(B, 1, dtype=torch.bool, device=V.device)], 1)
        return logits.masked_fill(~valid[:, None], -1e9)


# ----------------------------------------------------------------------- data
def pad(seqs, val, dtype=torch.long):
    n = max(len(s) for s in seqs)
    out = torch.full((len(seqs), n) + tuple(seqs[0].shape[1:]), val, dtype=dtype)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.as_tensor(s).to(dtype)
    return out


def batch_points(items, dev):
    pts = pad([it["points"].float() for it in items], 0.0, torch.float32)
    mask = pad([torch.ones(len(it["points"]), dtype=torch.bool) for it in items], False, torch.bool)
    return pts.to(dev), mask.to(dev)


def token_batch(items, key, pad_id, specials, dev):
    seq = pad([it[key].long() for it in items], pad_id)
    x, y = seq[:, :-1], seq[:, 1:].clone()
    y[y == pad_id] = -100
    return x.to(dev), y.to(dev), slots(x, specials).to(dev)


def conn_batch(items, dev):
    V = pad([it["verts"] for it in items], 0.0, torch.float32)
    vmask = pad([torch.ones(len(it["verts"]), dtype=torch.bool) for it in items], False, torch.bool)
    tg = []
    for it in items:
        M = len(it["verts"])
        tg.append(torch.cat([it["conn"].long().reshape(-1), torch.tensor([M])]))   # M = stop index
    t = pad(tg, -100)
    # the stop target is index M of each sample; padded batch: map to column Vmax
    for i, it in enumerate(items):
        t[i, len(tg[i]) - 1] = V.shape[1]
    ptr_in = torch.cat([torch.full((len(items), 1), -1), t[:, :-1]], 1)
    ptr_in[ptr_in == -100] = 0
    return V.to(dev), vmask.to(dev), ptr_in.to(dev), t.to(dev)


# ----------------------------------------------------------------------- generation
@torch.no_grad()
def generate(model, pts, pmask, start, stop, specials, max_len):
    z = model.encode(pts, pmask)
    x = torch.full((pts.shape[0], 1), start, dtype=torch.long, device=pts.device)
    done = torch.zeros(pts.shape[0], dtype=torch.bool, device=pts.device)
    for _ in range(max_len - 1):
        lg = model.decode(x, slots(x, specials), z)[:, -1]
        nxt = lg.argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, stop), nxt)
        x = torch.cat([x, nxt[:, None]], 1)
        done |= nxt == stop
        if done.all():
            break
    return x.cpu()


def dequant_vertices(tok, ids):
    c = tok.core; rmax = float(c.R_MAX)
    return np.array([[c._dq_scalar(ids[i] - c.off_r, -rmax, rmax), c._dq_scalar(ids[i + 1] - c.off_r, -rmax, rmax),
                      c._dq_scalar(ids[i + 2] - c.off_r, c.Z_MIN, c.Z_MAX)] for i in range(0, len(ids) - 2, 3)])


def evaluate_generation(task, model, data, items, dev, bs=8):
    sp = data["special"]; specials = set(sp.values())
    tok = HexaRowTokenizer(r_bounds=tuple(data["r_bounds"]), z_bounds=tuple(data["z_bounds"]))
    res = []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        pts, pmask = batch_points(chunk, dev)
        key = "tokens" if task == "hexarow" else "vtok"
        out = generate(model, pts, pmask, sp["start"], sp["stop"], specials,
                       max(len(it[key]) for it in chunk) + 40)
        for it, seq in zip(chunk, out.tolist()):
            r = {"name": it["name"], "valid": False, "struct_ok": False}
            if sp["stop"] in seq[1:]:
                seq = seq[:seq.index(sp["stop"], 1) + 1]
            gt = it[key].long().tolist()
            try:
                if task == "hexarow":
                    vpt, blk = tok.detokenize(seq, coords="cart")
                    V = vpt.numpy(); r["valid"] = len(blk) > 0
                    r["struct_ok"] = (len(blk) == it["blocks"] and len(V) == len(it["verts"])
                                      and [t in specials for t in seq] == [t in specials for t in gt])
                else:
                    body = [t for t in seq[1:] if t != sp["stop"]]
                    r["valid"] = seq[-1] == sp["stop"] and len(body) % 3 == 0 and all(t not in specials for t in body)
                    V = dequant_vertices(tok, body) if r["valid"] else None
                    r["struct_ok"] = r["valid"] and len(V) == len(it["verts"])
                if r["struct_ok"]:
                    d = np.linalg.norm(V - it["verts"].numpy(), axis=1) * 1000
                    r["corner_mean"], r["corner_max"] = float(d.mean()), float(d.max())
            except Exception as e:  # noqa: BLE001
                r["error"] = f"{type(e).__name__}"
            res.append(r)
    ok = [r for r in res if r["struct_ok"]]
    return {"n": len(res), "valid": sum(r["valid"] for r in res), "struct_ok": len(ok),
            "corner_mean_median": float(np.median([r["corner_mean"] for r in ok])) if ok else None,
            "corner_max_median": float(np.median([r["corner_max"] for r in ok])) if ok else None}, res


@torch.no_grad()
def evaluate_conn(model, items, dev):
    model.eval(); exact = 0; res = []
    for it in items:
        V, vmask, ptr_in, t = conn_batch([it], dev)
        pts, pmask = batch_points([it], dev)
        # greedy pointer decoding on GT vertices
        seq = [-1]
        for _ in range(t.shape[1]):
            lg = model(V, vmask, torch.tensor([seq], device=dev).clamp(min=-1), pts, pmask)
            nxt = int(lg[0, -1].argmax()); seq.append(nxt)
            if nxt == V.shape[1]:
                break
        pred = seq[1:]; gt = t[0].tolist()
        exact += pred == gt
        res.append({"name": it["name"], "exact": pred == gt})
    return {"n": len(items), "exact": exact}, res


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("hexarow", "vertex", "conn"), required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--n-latent", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=12, help="evaluations without val improvement")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--select", choices=("best", "last"), default="best",
                    help="checkpoint to evaluate: best val loss, or the last epoch (overfit / best case)")
    ap.add_argument("--train-frac", type=float, default=1.0, help="use a random subset of train (learning curve)")
    ap.add_argument("--eval-train", type=int, default=0, help="also evaluate generation on the first N train items")
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = "cuda"
    data = torch.load(a.data, weights_only=False)
    tr, va = data["train"], data["val"]
    if a.train_frac < 1.0:
        tr = [tr[i] for i in sorted(rng.choice(len(tr), max(1, round(a.train_frac * len(tr))), replace=False))]
    sp = data["special"]; specials = set(sp.values())
    os.makedirs(a.out, exist_ok=True)
    key = "tokens" if a.task == "hexarow" else "vtok"
    if a.task == "conn":
        max_len = max(it["conn"].numel() for it in tr + va) + 2
        model = PointerNet(a.d, a.layers, a.heads, max_len, dropout=a.dropout).to(dev)
    else:
        max_len = max(len(it[key]) for it in tr + va) + 64
        model = BlockGen(data["vocab"], a.d, a.layers, a.heads, max_len, sp["pad"], a.dropout, a.n_latent).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[{a.task}] params {n_par/1e6:.2f} M | train {len(tr)} val {len(va)} | max_len {max_len}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * math.ceil(len(tr) / a.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))

    def loss_on(items, train):
        model.train(train)
        tot, n, corr, cnt = 0.0, 0, 0, 0
        order = rng.permutation(len(items)) if train else np.arange(len(items))
        for i in range(0, len(items), a.bs):
            chunk = [items[j] for j in order[i:i + a.bs]]
            pts, pmask = batch_points(chunk, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16), torch.set_grad_enabled(train):
                if a.task == "conn":
                    V, vmask, ptr_in, y = conn_batch(chunk, dev)
                    lg = model(V, vmask, ptr_in, pts, pmask)
                else:
                    x, y, sl = token_batch(chunk, key, sp["pad"], specials, dev)
                    lg = model(x, sl, pts, pmask)
                loss = F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), y.reshape(-1), ignore_index=-100)
            if train:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            m = y != -100
            corr += int((lg.argmax(-1)[m] == y[m]).sum()); cnt += int(m.sum())
            tot += float(loss) * len(chunk); n += len(chunk)
        return tot / n, corr / cnt

    best, bad, hist, t0 = float("inf"), 0, [], time.time()
    for ep in range(1, a.epochs + 1):
        trl, tra = loss_on(tr, True)
        if ep % a.eval_every == 0 or ep == a.epochs:
            vl, vac = loss_on(va, False)
            hist.append({"epoch": ep, "train_loss": trl, "train_acc": tra, "val_loss": vl, "val_acc": vac})
            flag = ""
            if vl < best - 1e-4:
                best, bad, flag = vl, 0, " *"
                torch.save({"model": model.state_dict(), "args": vars(a), "epoch": ep, "val_loss": vl}, f"{a.out}/best.pt")
            else:
                bad += 1
            print(f"[{a.task}] ep {ep:4d} train {trl:.4f}/{tra:.3f} val {vl:.4f}/{vac:.3f} ({time.time()-t0:.0f}s){flag}", flush=True)
            if bad >= a.patience:
                print(f"[{a.task}] early stop at epoch {ep}", flush=True); break
    torch.save({"model": model.state_dict(), "args": vars(a), "epoch": ep, "val_loss": vl}, f"{a.out}/last.pt")
    ck = torch.load(f"{a.out}/{a.select}.pt", weights_only=False); model.load_state_dict(ck["model"]); model.eval()
    if a.task == "conn":
        summ, rows = evaluate_conn(model, va, dev)
    else:
        summ, rows = evaluate_generation(a.task, model, data, va, dev)
    if a.eval_train:
        sub = tr[:a.eval_train]
        st, _ = evaluate_conn(model, sub, dev) if a.task == "conn" else evaluate_generation(a.task, model, data, sub, dev)
        summ["train_eval"] = st
    summ.update(select=a.select, n_train=len(tr))
    summ.update(task=a.task, params_M=n_par / 1e6, best_epoch=ck["epoch"], best_val_loss=ck["val_loss"],
                d=a.d, layers=a.layers, train_s=time.time() - t0)
    print(f"[{a.task}] RESULT {json.dumps(summ)}", flush=True)
    json.dump({"summary": summ, "rows": rows, "history": hist}, open(f"{a.out}/result.json", "w"), indent=1)


if __name__ == "__main__":
    main()
