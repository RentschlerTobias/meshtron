"""train_hexarow_smoke.py

Smoke-Training fuer HexaRowTokenizer-Streams (decoder-only GPT, flaches Vocab).
Nutzt die vorberechneten Token-Sequenzen aus scripts/build_hexarow_tokens.py:
  uv run python scripts/build_hexarow_tokens.py --src data/polytron_data_3d_aug.pt
  uv run python train_hexarow_smoke.py --tokens data/hexarow_tokens_3d_aug.pt
"""
import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hexa_row_tokenizer import HexaRowTokenizer


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, L, D = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(B, L, 3, self.heads, D // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).reshape(B, L, D)
        x = x + self.proj(a)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab, d, layers, heads, max_len, pad_id):
        super().__init__()
        self.pad_id = pad_id
        self.tok = nn.Embedding(vocab, d, padding_idx=pad_id)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device)
        h = self.tok(x) + self.pos(pos)[None]
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.ln(h))


def make_batch(seqs, pad_id, device):
    L = max(len(s) for s in seqs)
    x = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.as_tensor(s, dtype=torch.long)
    return x.to(device)


def evaluate(model, seqs, pad_id, vocab, batch, dev, lossf):
    model.eval()
    vl, va = [], []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            x = make_batch(seqs[i : i + batch], pad_id, dev)
            logits = model(x[:, :-1])
            vl.append(lossf(logits.reshape(-1, vocab), x[:, 1:].reshape(-1)).item())
            pred = logits.argmax(-1)
            tgt = x[:, 1:]
            m = tgt != pad_id
            va.append((pred[m] == tgt[m]).float().mean().item())
    model.train()
    return float(np.mean(vl)), float(np.mean(va))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="data/hexarow_tokens_3d_aug.pt")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--maxblocks", type=int, default=128)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    probe = HexaRowTokenizer()  # vocab/pad ids stammen aus der Core-Vocab-Definition
    vocab = probe.core.vocab_size
    pad_id = probe.core.pad_token
    print(f"vocab={vocab} pad={pad_id}")

    print(f"lade {args.tokens} ...")
    ds = torch.load(args.tokens, weights_only=False)
    train_seqs = [s["tokens"].tolist() for s in ds["train"] if s["blocks"] <= args.maxblocks]
    val_seqs = [s["tokens"].tolist() for s in ds["val"] if s["blocks"] <= args.maxblocks]
    lens = np.array([len(s) for s in train_seqs + val_seqs])
    print(f"{len(train_seqs)} train / {len(val_seqs)} val | seq-len min {lens.min()} "
          f"mean {int(lens.mean())} max {lens.max()}")
    max_len = int(lens.max())

    model = GPT(vocab, args.d, args.layers, args.heads, max_len + 1, pad_id).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"Modell: d={args.d} layers={args.layers} heads={args.heads} "
          f"params={nparam/1e6:.1f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warm = max(1, int(0.05 * args.steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, args.steps - warm))))
    lossf = nn.CrossEntropyLoss(ignore_index=pad_id)

    rng = np.random.default_rng(0)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model.train()
    for step in range(args.steps):
        bi = rng.integers(0, len(train_seqs), min(args.batch, len(train_seqs)))
        x = make_batch([train_seqs[i] for i in bi], pad_id, dev)
        logits = model(x[:, :-1])
        loss = lossf(logits.reshape(-1, vocab), x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0 or step == args.steps - 1:
            with torch.no_grad():
                pred = logits.argmax(-1)
                tgt = x[:, 1:]
                m = tgt != pad_id
                acc = (pred[m] == tgt[m]).float().mean().item()
            vram = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0
            print(f"  step {step:4d}  loss {loss.item():.3f}  tok-acc {acc:.3f}  "
                  f"lr {sched.get_last_lr()[0]:.1e}  peakVRAM {vram:.2f} GB")

    vl, va = evaluate(model, val_seqs, pad_id, vocab, args.batch, dev, lossf)
    print(f"VAL loss {vl:.3f}  tok-acc {va:.3f}  ({time.time()-t0:.0f}s, {args.steps} steps)")


if __name__ == "__main__":
    main()
