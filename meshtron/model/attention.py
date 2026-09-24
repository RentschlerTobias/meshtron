import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from meshtron.model.positional_encoder import RoPEPositionalEncoding as RoPE
# from rotary_positional_embedding import RotaryPositionalEmbedding as RoPE
from typing import Optional, Tuple, Dict


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8, dropout=0.1, is_cross_attention: Optional[bool] = False,
                 max_position: Optional[int] = 10000, use_flash_attention: bool = False):
        super().__init__()

        assert d_model % n_heads == 0, "d_model muss durch n_heads teilbar sein"
        self.is_cross_attention = is_cross_attention
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension pro Head
        self.use_flash_attention = use_flash_attention
        self.dropout_p = dropout

        # Linear layers für Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.rope = RoPE(self.d_k, max_seq_len=max_position)

    def forward(self, q, k, v, mask=None, position_ids=None):
        """
        Args:
            q,k,v: [batch_size, seq_len, d_model] oder [seq_len, d_model]
            mask: Optional attention mask

        Returns:
            output: Same shape as input
        """
        # Handle single sequence vs batch
        if q.dim() == 2:
            q = q.unsqueeze(0)  # Add batch dimension
            k = k.unsqueeze(0)  # Add batch dimension
            v = v.unsqueeze(0)  # Add batch dimension
            single_sequence = True
        else:
            single_sequence = False

        batch_size, seq_len, d_model = q.shape
        # 1. Linear transformations for Q, K, V
        Q = self.w_q(q)  # [batch_size, seq_len, d_model]
        K = self.w_k(k)  # [batch_size, seq_len, d_model]
        V = self.w_v(v)  # [batch_size, seq_len, d_model]
        # 2. Reshape for multi-head attention (Splitten der Heads)

        # [batch_size, seq_len, d_model] -> [batch_size, n_heads, seq_len, d_k]
        # Q splitten
        b_q, len_q, d_model_q = Q.shape
        Q_split = Q.view(b_q, len_q, self.n_heads, self.d_k).transpose(1, 2)

        # K splitten (nutzt len_k statt len_q)
        b_k, len_k, d_model_k = K.shape
        K_split = K.view(b_k, len_k, self.n_heads, self.d_k).transpose(1, 2)

        # V splitten (nutzt len_v statt len_q)
        b_v, len_v, d_model_v = V.shape
        V_split = V.view(b_v, len_v, self.n_heads, self.d_k).transpose(1, 2)
        # 3. RoPE anwenden
        # Hinweis: Stellen Sie sicher, dass Ihre RoPE-Implementierung damit klarkommt,
        # dass Q und K unterschiedliche Längen haben, falls position_ids übergeben werden.

        if self.is_cross_attention:
            # Bei Cross-Attention: Kein RoPE, da Q und K unterschiedliche Längen haben
            rotated_Q = Q_split
            rotated_K = K_split
        else:
            # Bei Self-Attention: Beide rotieren
            rotated_Q, rotated_K = self.rope(
                Q_split, K_split, position_ids=position_ids)

        # 4. Attention
        # Output wird Länge von Q haben (len_q)
        attention_output = self.scaled_dot_product_attention(
            rotated_Q, rotated_K, V_split, mask)

        # 5. Concatenate
        # Hier nutzen wir wieder len_q, da der Output dem Query folgt
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            b_q, len_q, self.d_model
        )

        output = self.w_o(attention_output)

        if single_sequence:
            output = output.squeeze(0)

        return output

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Args:
            Q, K, V: [batch_size, n_heads, seq_len, d_k]
            mask: Optional additive mask (0 / -inf), same convention either backend

        Returns:
            output: [batch_size, n_heads, seq_len, d_k]
        """
        if self.use_flash_attention:
            # torch's native SDPA dispatches to Flash / memory-efficient / math
            # kernels depending on hardware+dtype; accepts the same additive
            # float mask convention `_causal_mask`/windowed mask already build,
            # so no mask-format conversion is needed here.
            return F.scaled_dot_product_attention(
                Q, K, V, attn_mask=mask,
                dropout_p=self.dropout_p if self.training else 0.0,
            )

        d_k = Q.size(-1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores + mask
            # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        output = torch.matmul(attention_weights, V)

        return output
