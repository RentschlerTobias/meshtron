"""v2 block-structure generator: decoder-only transformer with cross-attention
to a Perceiver encoding of ALL labelled surface points. No block-count input.

Differences to meshtron's GPTCond (meshtron/model/gpt_cond.py):
  * GPTCond pools the cloud into ONE vector (mean of 16 latents) and applies it
    as FiLM to every token; here every decoder layer cross-attends to the
    n_latent geometry latents, so coordinates can be read off the geometry.
  * points carry 7 label channels (O-grid cut = blade = label 7) and all
    points are used (key-padding mask), not a 1000-point random subset.
  * no block-count conditioning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attn(nn.Module):
    def __init__(self, d, heads, causal=False):
        super().__init__()
        self.h, self.causal = heads, causal
        self.q, self.kv, self.o = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)

    def forward(self, x, mem=None, mem_mask=None):
        B, L, D = x.shape
        mem = x if mem is None else mem
        q = self.q(x).view(B, L, self.h, D // self.h).transpose(1, 2)
        k, v = self.kv(mem).view(B, mem.shape[1], 2, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        mask = None if mem_mask is None else mem_mask[:, None, None, :]   # True = attend
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=self.causal and mem_mask is None)
        return self.o(a.transpose(1, 2).reshape(B, L, D))


def mlp(d):
    return nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))


class PointPerceiver(nn.Module):
    """[B,P,in_dim] points (+ mask) -> [B,n_latent,d] latents."""

    def __init__(self, in_dim, d, heads, n_latent, depth=2):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(in_dim, d), nn.GELU(), nn.Linear(d, d))
        self.lat = nn.Parameter(torch.randn(n_latent, d) * 0.02)
        self.cross = nn.ModuleList(Attn(d, heads) for _ in range(depth))
        self.self_ = nn.ModuleList(Attn(d, heads) for _ in range(depth))
        self.ff = nn.ModuleList(mlp(d) for _ in range(depth))
        self.ln = nn.ModuleList(nn.ModuleList(nn.LayerNorm(d) for _ in range(4)) for _ in range(depth))

    def forward(self, pts, mask):
        f = self.inp(pts)
        z = self.lat[None].expand(pts.shape[0], -1, -1)
        for c, s, ff, ln in zip(self.cross, self.self_, self.ff, self.ln):
            z = z + c(ln[0](z), ln[1](f), mask)
            z = z + s(ln[2](z))
            z = z + ff(ln[3](z))
        return z


class DecBlock(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.sa, self.ca, self.ff = Attn(d, heads, causal=True), Attn(d, heads), mlp(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, z):
        x = x + self.drop(self.sa(self.ln1(x)))
        x = x + self.drop(self.ca(self.ln2(x), z))
        return x + self.drop(self.ff(self.ln3(x)))


class BlockGen(nn.Module):
    def __init__(self, vocab, d, layers, heads, max_len, pad_id, dropout=0.1,
                 n_latent=64, point_dim=11, n_slot=4, enc_depth=2):
        super().__init__()
        self.pad_id, self.max_len = pad_id, max_len
        self.tok = nn.Embedding(vocab, d, padding_idx=pad_id)
        self.pos = nn.Embedding(max_len, d)
        self.slot = nn.Embedding(n_slot, d)
        self.enc = PointPerceiver(point_dim, d, heads, n_latent, enc_depth)
        self.blocks = nn.ModuleList(DecBlock(d, heads, dropout) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight
        self.drop = nn.Dropout(dropout)
        # tied head: N(0,1) embeddings (nn.Embedding default) gave logits of
        # size ~sqrt(d) and a start loss of ~60 instead of ln(vocab) ~ 8
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        nn.init.normal_(self.slot.weight, std=0.02)

    def encode(self, pts, mask):
        return self.enc(pts, mask)

    def decode(self, x, slot, z):
        L = x.shape[1]
        h = self.tok(x) + self.pos(torch.arange(L, device=x.device))[None] + self.slot(slot)
        h = self.drop(h)
        for b in self.blocks:
            h = b(h, z)
        return self.head(self.ln(h))

    def forward(self, x, slot, pts, mask):
        return self.decode(x, slot, self.encode(pts, mask))


def slots(tokens, specials, npt=3):
    """Coordinate slot per position: 0..npt-1 for coordinate tokens (cycling
    within each row), npt for specials. tokens: LongTensor [..., L]."""
    out = torch.full_like(tokens, npt)
    flat, sflat = tokens.reshape(-1, tokens.shape[-1]), out.reshape(-1, tokens.shape[-1])
    for b in range(flat.shape[0]):
        c = 0
        for i, t in enumerate(flat[b].tolist()):
            if t in specials:
                c = 0
            else:
                sflat[b, i] = c % npt; c += 1
    return out
