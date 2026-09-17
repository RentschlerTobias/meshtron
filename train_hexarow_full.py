"""train_hexarow_full.py

Pfad 1: HexaRowToken-Training auf vollem augmentierten 3D-Datensatz mit
Conditioning auf Punktwolke (n verteilte vertices_polar-Punkte) und
face_count (Blockzahl).

Tokens werden aus scripts/build_hexarow_tokens.py geladen; die Punktwolken
kommen direkt aus dem Source-.pt (verknuepft ueber 'name' der Samples).

  uv run python train_hexarow_full.py \
      --tokens data/hexarow_tokens_3d_full_aug.pt \
      --src data/polytron_data_3d_full_aug.pt
"""
import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from hexa_row_tokenizer import HexaRowTokenizer


def sample_points(vp, n, rb, zb, rng):
    """Punktwolke aus vertices_polar (r,theta,z), normalisiert:
    r -> [0,1] ueber rb, theta -> (sin,cos), z -> [0,1] ueber zb."""
    idx = rng.choice(len(vp), size=n, replace=len(vp) < n)
    p = vp[idx]
    r = (p[:, 0] - rb[0]) / max(1e-9, rb[1] - rb[0])
    z = (p[:, 2] - zb[0]) / max(1e-9, zb[1] - zb[0])
    return np.stack([r, np.sin(p[:, 1]), np.cos(p[:, 1]), z], axis=-1)


class PointEncoder(nn.Module):
    """MLP je Punkt + Attention-Pooling auf n_latent Queries -> Vektor d."""

    def __init__(self, d, n_latent=16, dim=4):
        super().__init__()
        h = d // 2
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, d))
        self.q = nn.Parameter(torch.randn(n_latent, d) * 0.02)
        self.proj = nn.Linear(d, d)

    def forward(self, pts):
        # pts [B, P, 4] -> [B, d]
        f = self.mlp(pts)
        q = self.q[None].expand(f.shape[0], -1, -1)
        a = F.scaled_dot_product_attention(q, f, f)
        return self.proj(a).mean(1)


class Block(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, L, D = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(B, L, 3, self.heads, D // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).reshape(B, L, D)
        x = x + self.drop(self.proj(a))
        x = x + self.mlp(self.drop(self.ln2(x)))
        return x


class GPTCond(nn.Module):
    """Decoder-only GPT mit Conditioning: Punktwolke + face_count werden zu
    je einem Vektor codiert und per FiLM (scale/shift) auf die Token-Embeddings
    gegeben."""

    def __init__(self, vocab, d, layers, heads, max_len, pad_id, dropout,
                 n_points, n_latent):
        super().__init__()
        self.pad_id = pad_id
        self.tok = nn.Embedding(vocab, d, padding_idx=pad_id)
        self.pos = nn.Embedding(max_len, d)
        self.drop = nn.Dropout(dropout)
        self.penc = PointEncoder(d, n_latent, dim=4)
        self.fc_emb = nn.Sequential(nn.Linear(1, d),
                                    nn.GELU(),
                                    nn.Linear(d, d))
        self.blocks = nn.ModuleList(Block(d, heads, dropout) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight

    def condition(self, pc, fc):
        hv = self.penc(pc) + self.fc_emb(fc[:, None])
        return hv

    def forward(self, x, pc, fc):
        B, L = x.shape
        pos = torch.arange(L, device=x.device)
        h = self.tok(x) + self.pos(pos)[None]
        c = self.condition(pc, fc)
        h = self.drop(h) * (1 + torch.tanh(c)[:, None]) + c[:, None]
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.ln(h))


def batchify(items, pad_id, device):
    """items: list of {tokens, points, blocks} -> tensors x, pc, fc."""
    L = max(len(it["tokens"]) for it in items)
    x = torch.full((len(items), L), pad_id, dtype=torch.long)
    for i, it in enumerate(items):
        x[i, : len(it["tokens"])] = torch.as_tensor(it["tokens"], dtype=torch.long)
    pc = torch.as_tensor(np.stack([it["points"] for it in items]), dtype=torch.float32)
    fc = torch.as_tensor([it["blocks"] for it in items], dtype=torch.float32)
    return x.to(device), pc.to(device), fc.to(device)


def make_batches(items, budget=16384, cap=32):
    """Laengensortierte Token-Budget-Batches: Summe aus Batchgroesse *
    max-Seqlen im Batch <= budget (Padding-homogen, kein OOM durch Mix
    aus 300- und 22k-Token-Sequenzen). Einzelne Samples groesser als das
    Budget laufen als Einzelbatch."""
    order = sorted(range(len(items)), key=lambda i: len(items[i]["tokens"]))
    batches, cur = [], []
    for i in order:
        li = len(items[i]["tokens"])
        if cur:
            ml = max(li, len(items[cur[-1]]["tokens"]))
            if (len(cur) + 1) * ml > budget or len(cur) >= cap:
                batches.append(cur)
                cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    return batches


@torch.no_grad()
def evaluate(model, batches, items, pad_id, vocab, dev, lossf, dtype):
    model.eval()
    vl, va, wn = [], [], []
    for idxs in batches:
        its = [items[i] for i in idxs]
        x, pc, fc = batchify(its, pad_id, dev)
        with torch.autocast(dev, dtype=dtype, enabled=dev == "cuda"):
            logits = model(x[:, :-1], pc, fc)
        vl.append(lossf(logits.float().reshape(-1, vocab),
                        x[:, 1:].reshape(-1)).item())
        pred = logits.argmax(-1)
        tgt = x[:, 1:]
        m = tgt != pad_id
        va.append(((pred[m] == tgt[m]).float().sum().item(), m.sum().item()))
    model.train()
    ntok = sum(n for _, n in va)
    return (float(np.average([l for l in vl], weights=[len(idxs) for idxs in batches])),
            float(sum(c for c, _ in va) / max(1, ntok)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="data/hexarow_tokens_3d_full_aug.pt")
    ap.add_argument("--src", default="data/polytron_data_3d_full_aug.pt")
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--n-points", type=int, default=1000)
    ap.add_argument("--n-latent", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.17)
    ap.add_argument("--token-budget", type=int, default=16384,
                    help="max batchgroesse * max-seqlen pro Batch (Padding)")
    ap.add_argument("--batch-cap", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--maxblocks", type=int, default=0)  # 0 = alle
    ap.add_argument("--lr", type=float, default=1.3e-4)
    ap.add_argument("--wd", type=float, default=0.042)
    ap.add_argument("--warmup", type=int, default=600)
    ap.add_argument("--val-every", type=int, default=1)
    ap.add_argument("--out", default="data/hexarow_full_model.pt")
    ap.add_argument("--ckpt", default="data/hexarow_full_model_ckpt.pt",
                    help="per-Epoch-Checkpoint (model+opt) fuer Resume")
    ap.add_argument("--resume", default="", help="Checkpoint-File zum Weiterlaufen")
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    probe = HexaRowTokenizer()
    vocab, pad_id = probe.core.vocab_size, probe.core.pad_token
    print(f"vocab={vocab} pad={pad_id} dev={dev} dtype={dtype}")

    ds = torch.load(args.tokens, weights_only=False)
    src = torch.load(args.src, weights_only=False)
    name2sample = {}
    for i, s in enumerate(src):
        name2sample[s.get("name", f"sample{i}")] = s
    rb, zb = ds["r_bounds"], ds["z_bounds"]
    rb = tuple(float(v) for v in rb)
    zb = tuple(float(v) for v in zb)
    print(f"bounds r={rb} z={zb}")

    rng = np.random.default_rng(0)

    def prep(lst):
        items = []
        for s in lst:
            if args.maxblocks and s["blocks"] > args.maxblocks:
                continue
            raw = name2sample.get(s["name"])
            if raw is None:
                continue
            pts = sample_points(np.array(raw["vertices_polar"], dtype=np.float64),
                                args.n_points, rb, zb, rng)
            items.append({"tokens": s["tokens"].tolist(), "points": pts,
                          "blocks": s["blocks"], "name": s["name"]})
        return items

    train_items, val_items = prep(ds["train"]), prep(ds["val"])
    lens = np.array([len(it["tokens"]) for it in train_items + val_items])
    print(f"{len(train_items)} train / {len(val_items)} val | seq-len min {lens.min()} "
          f"mean {int(lens.mean())} max {lens.max()}")
    max_len = int(lens.max())

    train_batches = make_batches(train_items, args.token_budget, args.batch_cap)
    val_batches = make_batches(val_items, args.token_budget, args.batch_cap)
    bsizes = np.array([len(b) for b in train_batches])
    print(f"{len(train_batches)} train batches (median {int(np.median(bsizes))}, "
          f"max {bsizes.max()} pro Batch) | {len(val_batches)} val batches")

    model = GPTCond(vocab, args.d, args.layers, args.heads, max_len + 1,
                    pad_id, args.dropout, args.n_points, args.n_latent).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"Modell: d={args.d} layers={args.layers} heads={args.heads} "
          f"params={nparam / 1e6:.1f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lossf = nn.CrossEntropyLoss(ignore_index=pad_id)
    tot_steps = args.epochs * len(train_batches)

    start_ep = 0
    start_step = 0
    if args.resume:
        ck = torch.load(args.resume, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_ep = ck["epoch"] + 1
        start_step = ck["step"]
        step = start_step
        print(f"resume: epoch {start_ep}, step {step}")

    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    step = start_step
    model.train()
    for ep in range(start_ep, args.epochs):
        border = np.random.default_rng(2 + ep).permutation(len(train_batches))
        ep_loss, ep_corr, ep_tok, nb = 0.0, 0, 0, 0
        pbar = tqdm(border, desc=f"epoch {ep:2d}", unit="bt",
                    dynamic_ncols=True, leave=False)
        for bi in pbar:
            items = [train_items[j] for j in train_batches[bi]]
            x, pc, fc = batchify(items, pad_id, dev)
            with torch.autocast(dev, dtype=dtype, enabled=dev == "cuda"):
                logits = model(x[:, :-1], pc, fc)
                loss = lossf(logits.float().reshape(-1, vocab),
                             x[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # LR: linearer Warmup, danach Cosine-Decay ueber Gesamtsteps
            lr = (args.lr * step / args.warmup if step < args.warmup else
                  args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, step / max(1, tot_steps)))))
            for g in opt.param_groups:
                g["lr"] = lr
            opt.step()
            step += 1
            with torch.no_grad():
                pred = logits.argmax(-1)
                tgt = x[:, 1:]
                m = tgt != pad_id
                ep_corr += (pred[m] == tgt[m]).sum().item()
                ep_tok += m.sum().item()
            ep_loss += loss.item()
            nb += 1
            pbar.set_postfix(loss=f"{ep_loss / nb:.3f}",
                             acc=f"{ep_corr / max(1, ep_tok):.3f}", lr=f"{lr:.1e}")
        pbar.close()
        vram = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0
        msg = (f"epoch {ep:2d}  loss {ep_loss / max(1, nb):.3f}  "
               f"tok-acc {ep_corr / max(1, ep_tok):.3f}  peakVRAM {vram:.2f} GB")
        if (ep + 1) % args.val_every == 0 or ep == args.epochs - 1:
            vl, va = evaluate(model, val_batches, val_items, pad_id, vocab,
                              dev, lossf, dtype)
            msg += f"  VAL loss {vl:.3f} tok-acc {va:.3f}"
        print(msg + f"  ({time.time() - t0:.0f}s)")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": ep, "step": step,
                    "r_bounds": rb, "z_bounds": zb}, args.ckpt)
        print(f"  checkpoint -> {args.ckpt}")

    torch.save({"model": model.state_dict(),
                "cfg": vars(args), "vocab": vocab, "pad_id": pad_id,
                "r_bounds": rb, "z_bounds": zb}, args.out)
    print(f"saved {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
