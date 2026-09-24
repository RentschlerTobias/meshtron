"""gpt_cond.py -- the model, lifted out of the training script.

`GPTCond` is a decoder-only transformer over quantised block-structure tokens,
conditioned on a surface point cloud and the block count: both are encoded into
ONE vector per sample and applied to every token position as a FiLM scale and
shift.

It lived inside train_hexarow_full.py, which meant nothing could load a
checkpoint without importing the trainer -- generate.py, scripts/eval_family.py
and five more modules did exactly that. The definitions are unchanged; the
trainer now imports them back, so every existing `from train_hexarow_full
import GPTCond` keeps working.

  PointEncoder  per-point MLP, attention pooling onto n_latent learned queries
  Block         pre-norm attention + MLP, causal
  GPTCond       embeddings, FiLM conditioning, the stack, a tied head
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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
                 n_points, n_latent, npt=4):
        super().__init__()
        self.pad_id = pad_id
        self.tok = nn.Embedding(vocab, d, padding_idx=pad_id)
        self.pos = nn.Embedding(max_len, d)
        # PolyGen-Style coordinate-type embedding: Slot im Vertex (0=r,1=sin,2=cos,3=z)
        self.slot = nn.Embedding(npt, d)
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

    def forward(self, x, pc, fc, slot=None):
        B, L = x.shape
        pos = torch.arange(L, device=x.device)
        h = self.tok(x) + self.pos(pos)[None]
        if slot is not None:
            h = h + self.slot(slot)
        c = self.condition(pc, fc)
        h = self.drop(h) * (1 + torch.tanh(c)[:, None]) + c[:, None]
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.ln(h))
