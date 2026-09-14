"""
Flat Transformer Architecture (simplified from Hourglass)
All stages operate at full sequence length — no shortening or upsampling.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from attention import MultiHeadAttention


class HourglassTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.2,
                 max_position: Optional[int] = 1000, use_flash_attention: bool = False,
                 sliding_window_size: int = 0):
        super().__init__()
        self.attention = MultiHeadAttention(
            d_model, n_heads, dropout=dropout, max_position=max_position,
            use_flash_attention=use_flash_attention)
        self.sliding_window_size = sliding_window_size
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, is_casual: Optional[bool] = True, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if is_casual:
            _, seq_len, _ = x.shape
            mask = self._causal_mask(seq_len, x.device)
        else:
            mask = None

        attn_out = self.attention(
            x, x, x, mask=mask, position_ids=position_ids)
        x = self.norm1(x + attn_out)
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        return x

    def _causal_mask(self, seq_len, device):
        """Standard causal mask; additionally blocks positions more than
        `sliding_window_size` tokens in the past when set (>0). 0 (default) =
        full causal attention, unchanged from before this flag existed.

        Built as a boolean OR of two independent conditions, then converted to
        an additive float mask in one shot -- doing this via two sequential
        `masked_fill(mask <cond>, -inf)` calls on the same tensor is a trap:
        after the first call sets some entries to -inf, a second threshold
        check like `mask <= 0` re-matches those -inf entries and overwrites
        them back to 0.
        """
        blocked = torch.triu(torch.ones(
            (seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1)
        if self.sliding_window_size and self.sliding_window_size > 0:
            blocked = blocked | torch.tril(
                torch.ones((seq_len, seq_len), device=device, dtype=torch.bool),
                diagonal=-self.sliding_window_size)
        mask = torch.zeros((seq_len, seq_len), device=device)
        mask = mask.masked_fill(blocked, float('-inf'))
        return mask


class CrossAttentionCondition(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 max_position: Optional[int] = None, use_flash_attention: bool = False):
        super().__init__()
        self.attention = MultiHeadAttention(
            d_model, n_heads, dropout=dropout, is_cross_attention=True, max_position=max_position,
            use_flash_attention=use_flash_attention)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attention(
            query, key, value, mask=mask, position_ids=position_ids)
        x = self.norm1(query + attn_out)
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        return x


class HourglassStage(nn.Module):
    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 max_position: Optional[int] = None, use_flash_attention: bool = False,
                 sliding_window_size: int = 0):
        super().__init__()
        self.layers = nn.ModuleList([
            HourglassTransformerBlock(
                d_model, n_heads, d_ff, dropout, max_position,
                use_flash_attention=use_flash_attention,
                sliding_window_size=sliding_window_size)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, is_casual, position_ids) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, is_casual, position_ids)
        return x


class HourglassTransformer(nn.Module):
    """
    Flat Transformer — same interface as the Hourglass version.
    Each stage runs at full sequence length followed by cross-attention conditioning.
    No shortening or upsampling, so there is no information leak.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        stage_layers: Tuple[int, int, int, int, int] = (4, 8, 12, 16, 20),
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_position: Optional[int] = None,
        use_flash_attention: bool = False,
        sliding_window_size: int = 0,
    ):
        super().__init__()
        self.d_model = d_model

        self.stages = nn.ModuleList([
            HourglassStage(n, d_model, n_heads, d_ff, dropout, max_position,
                           use_flash_attention=use_flash_attention,
                           sliding_window_size=sliding_window_size)
            for n in stage_layers
        ])
        self.conditioners = nn.ModuleList([
            CrossAttentionCondition(
                d_model, n_heads, d_ff, dropout, max_position,
                use_flash_attention=use_flash_attention)
            for _ in stage_layers
        ])

    def forward(
        self,
        x: torch.Tensor,
        latent_condition: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        is_casual: Optional[bool] = True
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        if position_ids is None:
            position_ids = torch.arange(
                seq_len, dtype=torch.long, device=x.device
            ).unsqueeze(0).expand(batch_size, -1)

        for stage, conditioner in zip(self.stages, self.conditioners):
            x = stage(x, is_casual, position_ids)
            x = conditioner(x, latent_condition, latent_condition,
                            mask=None, position_ids=position_ids)

        return x
