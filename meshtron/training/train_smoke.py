"""
train_smoke.py

Smoke-Training: kleiner decoder-only GPT ueber die PolytronTokenizer-Sequenzen
(flaches Vocab, Pointer-Indizes als normale Tokens). Testet, ob der Transformer die
Mesh-Token-Struktur ueberhaupt lernt (Loss faellt, Next-Token-Accuracy steigt) und
misst den VRAM-Footprint. NICHT der finale Pointer-Kopf — reiner Funktions-Smoke.

  ~/Environments/meshtron/bin/python train_smoke.py \
      [--data domain_data_smoke.pt] [--d 256] [--layers 6] [--heads 8]
      [--batch 8] [--steps 400] [--maxfaces 96]
"""

import argparse
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from meshtron.data.polytron_tokenizer import PolytronTokenizer


class Block(nn.Module):
    """Pre-norm Decoder-Block. Attention direkt via F.scaled_dot_product_attention
    mit is_causal=True -> Flash/mem-efficient-Backend (O(seq) Speicher)."""

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
        q, k, v = qkv.permute(2, 0, 3, 1, 4)          # je [B, heads, L, hd]
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
        self.head.weight = self.tok.weight            # weight tying

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
        x[i, :len(s)] = torch.tensor(s)
    return x.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='domain_data_smoke.pt')
    ap.add_argument('--d', type=int, default=256)
    ap.add_argument('--layers', type=int, default=6)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--maxfaces', type=int, default=96)
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = PolytronTokenizer(repr_mode='cubic_bezier')
    pad_id = 3076
    vocab = 3078

    print(f"lade {args.data} ...")
    data = torch.load(args.data, weights_only=False)
    data = [d for d in data if d['faces'].shape[1] <= args.maxfaces]
    print(f"tokenisiere {len(data)} Meshes ...")
    seqs = []
    for d in data:
        toks, _ = tok.tokenize(d)
        seqs.append(toks)
    lens = np.array([len(s) for s in seqs])
    print(f"seq-len: min {lens.min()} mean {int(lens.mean())} max {lens.max()}  "
          f"| mesh-embed (int64): {lens.mean()*8/1024:.1f} KB avg, {lens.max()*8/1024:.1f} KB max")
    max_len = int(lens.max())

    n_val = max(1, len(seqs) // 10)
    rng = np.random.default_rng(0); perm = rng.permutation(len(seqs))
    val_ids = set(perm[:n_val].tolist())
    train = [seqs[i] for i in range(len(seqs)) if i not in val_ids]
    val = [seqs[i] for i in range(len(seqs)) if i in val_ids]

    model = GPT(vocab, args.d, args.layers, args.heads, max_len + 1, pad_id).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"Modell: d={args.d} layers={args.layers} heads={args.heads}  "
          f"params={nparam/1e6:.1f}M  (fp32+AdamW ~{nparam*16/1e6:.0f} MB Zustand)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warm = int(0.05 * args.steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(1, warm) if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, args.steps - warm))))
    lossf = nn.CrossEntropyLoss(ignore_index=pad_id)

    if dev == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model.train()
    for step in range(args.steps):
        bi = rng.integers(0, len(train), args.batch)
        x = make_batch([train[i] for i in bi], pad_id, dev)
        logits = model(x[:, :-1])
        loss = lossf(logits.reshape(-1, vocab), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 50 == 0 or step == args.steps - 1:
            with torch.no_grad():
                pred = logits.argmax(-1); tgt = x[:, 1:]
                m = tgt != pad_id
                acc = (pred[m] == tgt[m]).float().mean().item()
            vram = torch.cuda.max_memory_allocated() / 1e9 if dev == 'cuda' else 0
            print(f"  step {step:4d}  loss {loss.item():.3f}  tok-acc {acc:.3f}  "
                  f"lr {sched.get_last_lr()[0]:.1e}  peakVRAM {vram:.2f} GB")

    # val
    model.eval()
    with torch.no_grad():
        vl = []; va = []
        for i in range(0, len(val), args.batch):
            x = make_batch(val[i:i + args.batch], pad_id, dev)
            logits = model(x[:, :-1])
            vl.append(lossf(logits.reshape(-1, vocab), x[:, 1:].reshape(-1)).item())
            pred = logits.argmax(-1); tgt = x[:, 1:]; m = tgt != pad_id
            va.append((pred[m] == tgt[m]).float().mean().item())
    print(f"VAL loss {np.mean(vl):.3f}  tok-acc {np.mean(va):.3f}  "
          f"({time.time()-t0:.0f}s, {args.steps} steps)")


if __name__ == '__main__':
    main()
