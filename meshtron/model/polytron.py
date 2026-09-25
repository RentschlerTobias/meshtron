"""polytron.py -- PolyGen for curvilinear hex block structures.

Three models, each trained on its own with teacher forcing and chained at
inference (arXiv 2002.10880, sections 2.2 and 2.3, lifted from triangles to
hexahedra and extended by a curve stage):

  VertexModel  p(vertices | cloud, n_blocks)
               autoregressive over flattened quantised (z, y, x) triples plus a
               stop token; decoder with cross-attention onto the cloud latents.
  BlockModel   p(blocks | vertices, cloud, n_blocks)
               pointer network: a bidirectional encoder embeds every vertex,
               the causal decoder emits 8 pointers per block, the logits are
               dot products against the vertex embeddings. Exactly n_blocks
               blocks, so no stop token.
  CurveModel   p(edge curves | vertices, blocks, cloud)
               per undirected block edge, six categorical offsets of the two
               inner Bezier control points. Non-autoregressive: the edges
               attend to each other and to the cloud, and the edge shape is
               close to a function of its corners and the geometry.

The conditioning cloud (normalised xyz + patch multi-hot) is encoded once per
model by `CloudEncoder`, a Perceiver-style cross-attention onto learned
latents, so the decoders can look up WHERE on the geometry a corner or an edge
has to go -- a single pooled vector, as in GPTCond, cannot say that.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------

class Attn(nn.Module):
    def __init__(self, d, heads, cacheable=False):
        super().__init__()
        self.h = heads
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.o = nn.Linear(d, d)
        # Cross-attention onto a fixed memory (cloud, vertex encodings): at
        # sampling time the memory is the same every step, so its K/V are
        # projected once. Only without grad; `clear_cache` before each use.
        self.cacheable = cacheable
        self.cache_on = False
        self._cache = None

    def _kv(self, mem):
        base = mem[:1] if mem.shape[0] > 1 and mem.stride(0) == 0 else mem
        if not (self.cacheable and self.cache_on and not torch.is_grad_enabled()):
            return self.kv(base).expand(mem.shape[0], -1, -1)
        key = (base.data_ptr(), tuple(base.shape))
        if self._cache is None or self._cache[0] != key:
            self._cache = (key, self.kv(base))
        return self._cache[1].expand(mem.shape[0], -1, -1)

    def forward(self, x, mem, mask=None, causal=False):
        B, L, D = x.shape
        S = mem.shape[1]
        q = self.q(x).view(B, L, self.h, D // self.h).transpose(1, 2)
        k, v = self._kv(mem).reshape(B, S, 2, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        if causal:
            cm = torch.ones(L, S, dtype=torch.bool, device=x.device).tril()
            mask = cm[None, None] if mask is None else (mask & cm[None, None])
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(a.transpose(1, 2).reshape(B, L, D))


class Layer(nn.Module):
    """Pre-norm self-attention, optional cross-attention, MLP."""

    def __init__(self, d, heads, dropout, cross=True):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.sa = Attn(d, heads)
        self.cross = cross
        if cross:
            self.n2 = nn.LayerNorm(d)
            self.ca = Attn(d, heads, cacheable=True)
        self.n3 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, self_mask=None, causal=False, mem=None, mem_mask=None):
        h = self.n1(x)
        x = x + self.drop(self.sa(h, h, self_mask, causal))
        if self.cross and mem is not None:
            x = x + self.drop(self.ca(self.n2(x), mem, mem_mask))
        return x + self.drop(self.mlp(self.n3(x)))


def clear_cache(model, on=True):
    """Drop cached K/V and switch caching on (the samplers) or off. Only the
    samplers turn it on: the key is a data pointer, which a later batch may
    reuse, so a training-time validation pass must never see it."""
    for m in model.modules():
        if isinstance(m, Attn):
            m._cache = None
            m.cache_on = on


def key_mask(valid):
    """[B,S] bool (True = real) -> [B,1,1,S] attention mask."""
    return valid[:, None, None, :]


class Fourier(nn.Module):
    def __init__(self, n_freq=8):
        super().__init__()
        self.register_buffer("f", (2.0 ** torch.arange(n_freq)) * math.pi)

    def forward(self, x):                       # [.., 3] in [-1,1]
        a = x[..., None] * self.f               # [.., 3, F]
        return torch.cat([x, a.sin().flatten(-2), a.cos().flatten(-2)], -1)


class CloudEncoder(nn.Module):
    """[B,P,3+n_lab] -> [B,n_latent,d]."""

    def __init__(self, d, heads, n_latent=128, n_lab=7, layers=2, dropout=0.0,
                 n_freq=8, point_memory=False):
        super().__init__()
        # point_memory: return the per-point features next to the latents, so
        # a decoder can look a position up instead of reading it from a
        # pooled summary (the first corner has to be read to within a bin)
        self.point_memory = point_memory
        self.ff = Fourier(n_freq)
        self.inp = nn.Sequential(nn.Linear(3 + 6 * n_freq + n_lab, d), nn.GELU(),
                                 nn.Linear(d, d))
        self.lat = nn.Parameter(torch.randn(n_latent, d) * 0.02)
        self.pool = Layer(d, heads, dropout, cross=True)
        self.self_layers = nn.ModuleList(Layer(d, heads, dropout, cross=False)
                                         for _ in range(layers))
        self.n = nn.LayerNorm(d)

    def forward(self, pc):
        f = self.inp(torch.cat([self.ff(pc[..., :3]), pc[..., 3:]], -1))
        z = self.lat[None].expand(pc.shape[0], -1, -1)
        z = self.pool(z, mem=f)
        for l in self.self_layers:
            z = l(z)
        if self.point_memory:
            return self.n(torch.cat([z, f], 1))
        return self.n(z)


class VertexEmbed(nn.Module):
    """Quantised (x, y, z) -> d: one table per axis, summed (PolyGen 2.3)."""

    def __init__(self, d, q):
        super().__init__()
        self.e = nn.ModuleList(nn.Embedding(q, d) for _ in range(3))

    def forward(self, vq):                      # [B,M,3] long
        return sum(self.e[k](vq[..., k]) for k in range(3))


class VertexEncoder(nn.Module):
    """Bidirectional encoder over the vertex set, cross-attending the cloud."""

    def __init__(self, d, heads, layers, q, dropout, max_blocks=64):
        super().__init__()
        self.emb = VertexEmbed(d, q)
        self.nb = nn.Embedding(max_blocks, d)
        self.layers = nn.ModuleList(Layer(d, heads, dropout) for _ in range(layers))
        self.n = nn.LayerNorm(d)

    def forward(self, vq, vvalid, nb, lat):
        h = self.emb(vq) + self.nb(nb)[:, None]
        m = key_mask(vvalid)
        for l in self.layers:
            h = l(h, self_mask=m, mem=lat)
        return self.n(h)


# --------------------------------------------------------------------------
# stage 1: vertices
# --------------------------------------------------------------------------

class VertexModel(nn.Module):
    def __init__(self, q=512, d=256, heads=8, layers=8, dropout=0.1,
                 max_verts=128, max_blocks=64, n_latent=128, point_memory=False):
        super().__init__()
        self.q = q
        self.STOP, self.BOS = q, q + 1
        self.cloud = CloudEncoder(d, heads, n_latent, dropout=dropout,
                                  point_memory=point_memory)
        self.val = nn.Embedding(q + 2, d)
        self.coord = nn.Embedding(4, d)          # 0 x, 1 y, 2 z, 3 bos
        self.vidx = nn.Embedding(max_verts + 1, d)
        self.nb = nn.Embedding(max_blocks, d)
        self.layers = nn.ModuleList(Layer(d, heads, dropout) for _ in range(layers))
        self.n = nn.LayerNorm(d)
        self.head = nn.Linear(d, q + 1)          # values + STOP

    @staticmethod
    def flatten(vq):
        """[M,3] xyz -> token list z,y,x per vertex, then STOP."""
        return vq[:, [2, 1, 0]].reshape(-1)

    def positions(self, L, device):
        t = torch.arange(L, device=device)
        # input position 0 is BOS; token j (0-based) sits at input j+1
        coord = torch.where(t == 0, torch.full_like(t, 3), 2 - (t - 1) % 3)
        vidx = torch.where(t == 0, torch.zeros_like(t), (t - 1) // 3 + 1)
        return coord, vidx.clamp(max=self.vidx.num_embeddings - 1)

    def forward(self, inp, valid, pc, nb, lat=None):
        """inp [B,L] = BOS + tokens[:-1]; returns logits [B,L,q+1]."""
        if lat is None:
            lat = self.cloud(pc)
        coord, vidx = self.positions(inp.shape[1], inp.device)
        h = self.val(inp) + self.coord(coord)[None] + self.vidx(vidx)[None]
        h = h + self.nb(nb)[:, None]
        m = key_mask(valid)
        for l in self.layers:
            h = l(h, self_mask=m, causal=True, mem=lat)
        return self.head(self.n(h))


# --------------------------------------------------------------------------
# stage 2: blocks (pointer network)
# --------------------------------------------------------------------------

class BlockModel(nn.Module):
    def __init__(self, q=512, d=256, heads=8, enc_layers=6, dec_layers=6,
                 dropout=0.1, max_blocks=64, n_latent=128):
        super().__init__()
        self.cloud = CloudEncoder(d, heads, n_latent, dropout=dropout)
        self.enc = VertexEncoder(d, heads, enc_layers, q, dropout, max_blocks)
        self.bos = nn.Parameter(torch.randn(d) * 0.02)
        self.slot = nn.Embedding(8, d)
        self.bidx = nn.Embedding(max_blocks, d)
        self.nb = nn.Embedding(max_blocks, d)
        self.layers = nn.ModuleList(Layer(d, heads, dropout) for _ in range(dec_layers))
        self.n = nn.LayerNorm(d)
        self.qp = nn.Linear(d, d)
        self.kp = nn.Linear(d, d)

    def encode(self, vq, vvalid, pc, nb):
        lat = self.cloud(pc)
        return self.enc(vq, vvalid, nb, lat)

    def decode(self, H, vvalid, ptr_in, nb):
        """ptr_in [B,T] pointers already emitted (targets shifted right, the
        first slot ignored and replaced by BOS). Returns logits [B,T,M]."""
        B, T = ptr_in.shape
        g = torch.gather(H, 1, ptr_in.clamp(min=0)[..., None].expand(-1, -1, H.shape[-1]))
        g = torch.cat([self.bos.expand(B, 1, -1), g[:, 1:]], 1)
        t = torch.arange(T, device=H.device)
        h = g + self.slot(t % 8)[None] + self.bidx((t // 8).clamp(
            max=self.bidx.num_embeddings - 1))[None] + self.nb(nb)[:, None]
        mm = key_mask(vvalid)
        for l in self.layers:
            h = l(h, causal=True, mem=H, mem_mask=mm)
        q = self.qp(self.n(h))
        k = self.kp(H)
        logits = q @ k.transpose(1, 2) / math.sqrt(q.shape[-1])
        return logits.masked_fill(~vvalid[:, None, :], float("-inf"))

    def forward(self, vq, vvalid, pc, nb, ptr_in):
        H = self.encode(vq, vvalid, pc, nb)
        return self.decode(H, vvalid, ptr_in, nb)


# --------------------------------------------------------------------------
# stage 3: edge curves
# --------------------------------------------------------------------------

class CurveModel(nn.Module):
    def __init__(self, q=512, qc=256, d=256, heads=8, enc_layers=4,
                 edge_layers=6, dropout=0.1, max_blocks=64, n_latent=128):
        super().__init__()
        self.qc = qc
        self.cloud = CloudEncoder(d, heads, n_latent, dropout=dropout)
        self.enc = VertexEncoder(d, heads, enc_layers, q, dropout, max_blocks)
        self.edge_in = nn.Sequential(nn.Linear(3 * d + 4, d), nn.GELU(),
                                     nn.Linear(d, d))
        self.layers = nn.ModuleList(Layer(d, heads, dropout) for _ in range(edge_layers))
        self.n = nn.LayerNorm(d)
        self.head = nn.Linear(d, 6 * qc)

    def forward(self, vq, vvalid, pc, nb, edges, evalid, vnorm):
        """edges [B,E,2] ids (a<b), vnorm [B,M,3] dequantised xyz in [-1,1]
        (chord direction and length as explicit features). -> [B,E,6,qc]."""
        lat = self.cloud(pc)
        H = self.enc(vq, vvalid, nb, lat)
        d = H.shape[-1]
        ia = edges[..., 0].clamp(min=0)
        ib = edges[..., 1].clamp(min=0)
        ha = torch.gather(H, 1, ia[..., None].expand(-1, -1, d))
        hb = torch.gather(H, 1, ib[..., None].expand(-1, -1, d))
        pa = torch.gather(vnorm, 1, ia[..., None].expand(-1, -1, 3))
        pb = torch.gather(vnorm, 1, ib[..., None].expand(-1, -1, 3))
        ch = pb - pa
        L = ch.norm(dim=-1, keepdim=True)
        feat = torch.cat([ha, hb, ha * hb, ch / L.clamp(min=1e-6), L], -1)
        h = self.edge_in(feat)
        m = key_mask(evalid)
        for l in self.layers:
            h = l(h, self_mask=m, mem=lat)
        out = self.head(self.n(h))
        return out.view(*out.shape[:2], 6, self.qc)
