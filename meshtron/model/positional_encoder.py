"""
Rotary Positional Encoding (RoPE) for MeshTron
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class RoPEPositionalEncoding(nn.Module):
    def __init__(self,
                 dim: int,
                 max_seq_len: int = 10000,
                 base: int = 10000,
                 device=None):
        super().__init__()

        if dim % 2 != 0:
            raise ValueError(f"RoPE dimension must be even, got {dim}")

        self.dim = dim
        self.max_position_embeddings = max_seq_len
        self.base = base
        self.device = device

        inv_freq = self._compute_inv_freq()
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._precompute_rotation_cache(self.max_position_embeddings)

    def _compute_inv_freq(self) -> torch.Tensor:
        inv_freq = 1.0 / \
            (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        return inv_freq

    def _precompute_rotation_cache(self, max_len: int):
        positions = torch.arange(max_len, dtype=torch.float)
        freqs = torch.outer(positions, self.inv_freq)

        cos_cached = freqs.cos()
        sin_cached = freqs.sin()

        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def _apply_rotary_pos_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x_pairs = x.view(*x.shape[:-1], -1, 2)

        x_even = x_pairs[..., 0]
        x_odd = x_pairs[..., 1]

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        rotated_pairs = torch.stack([rotated_even, rotated_odd], dim=-1)
        rotated_x = rotated_pairs.view(*x.shape)

        return rotated_x

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                position_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:

        # 1. Shape Handling: Unterscheidung zwischen 3D (ohne Heads) und 4D (mit Heads)
        if query.dim() == 4:
            batch_size, n_heads, seq_len, d_k = query.shape
        elif query.dim() == 3:
            batch_size, seq_len, d_k = query.shape
            n_heads = 1
        else:
            raise ValueError(
                f"Query tensor has unexpected number of dimensions: {query.dim()}")

        # 2. Position IDs erstellen, falls nicht vorhanden
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=query.device).unsqueeze(
                0).expand(batch_size, -1)

        # 3. Rotationsmatrizen holen (Shape: [batch_size, seq_len, d_k // 2])
        cos, sin = self._get_rotation_matrices(position_ids)

        # 4. Broadcasting vorbereiten: Dimension für Heads einfügen
        # Wenn Input 4D ist [B, H, S, D], müssen cos/sin [B, 1, S, D] sein
        if query.dim() == 4:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        # 5. Rotation anwenden
        rotated_query = self._apply_rotary_pos_emb(query, cos, sin)
        rotated_key = self._apply_rotary_pos_emb(key, cos, sin)

        return rotated_query, rotated_key

    def _get_rotation_matrices(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        max_pos = position_ids.max().item()

        if max_pos < self.max_position_embeddings:
            cos = self.cos_cached[position_ids]
            sin = self.sin_cached[position_ids]
        else:
            freqs = torch.outer(position_ids.float().flatten(), self.inv_freq)
            cos = freqs.cos().view(*position_ids.shape, -1)
            sin = freqs.sin().view(*position_ids.shape, -1)

        return cos, sin
