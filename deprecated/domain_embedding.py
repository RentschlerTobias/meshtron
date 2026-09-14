"""
domain_embedding.py

DEPRECATED: part of the MeshtronDomain family (failed experiment, superseded by
Polytron). Kept for reference only, not maintained or extended.

Embedding-Layer fuer Domain-Partition Tokenizer.
Unterstuetzt 3 Modi:
  0: Split vocab ranges (ein nn.Embedding)
  1: Shared vocab       (ein nn.Embedding)
  2: Separate embeddings (zwei nn.Embedding: scalar + angle)
"""


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn


class DomainEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        max_len: int = 5000,
        embedding_mode: int = 0,
        quantization_r: int = 512,
        quantization_a: int = 256,
        n_special: int = 5,
    ):
        super().__init__()
        self.embedding_mode = embedding_mode
        self.d_model = d_model
        self.quantization_r = quantization_r
        self.quantization_a = quantization_a
        self.n_special = n_special
        self.vocab_size = vocab_size

        if embedding_mode in (0, 1):
            # Ein einziges Embedding fuer alle Token
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.scalar_emb = None
            self.angle_emb = None
        elif embedding_mode == 2:
            # Separate Embeddings
            # Skalare: r, t_norm, special tokens
            # Winkel: sincos fuer theta, alpha_in, alpha_out
            self.scalar_vocab = quantization_r + n_special
            self.angle_vocab = quantization_a
            self.scalar_emb = nn.Embedding(self.scalar_vocab, d_model)
            self.angle_emb = nn.Embedding(self.angle_vocab, d_model)
            self.embedding = None
        else:
            raise ValueError(f"embedding_mode {embedding_mode} ungueltig")

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [batch_size, seq_len] oder [seq_len]
        Returns:
            embeddings: [batch_size, seq_len, d_model] oder [seq_len, d_model]
        """
        if self.embedding_mode in (0, 1):
            return self.embedding(tokens)

        # Modus 2: separate Embeddings
        # Token-Struktur:
        #   [0, Qr-1]              -> Skalare (r, t_norm)
        #   [Qr, Qr+Qa-1]          -> Winkel (sincos)
        #   [Qr+Qa, Qr+Qa+n_special-1] -> Special Tokens
        
        scalar_mask = tokens < self.quantization_r
        angle_mask = (tokens >= self.quantization_r) & (tokens < self.quantization_r + self.quantization_a)
        special_mask = tokens >= self.quantization_r + self.quantization_a

        # Initialisiere Output
        out_shape = tokens.shape + (self.d_model,)
        embeddings = torch.zeros(out_shape, device=tokens.device, dtype=torch.float32)

        # Skalare
        if scalar_mask.any():
            embeddings[scalar_mask] = self.scalar_emb(tokens[scalar_mask])

        # Winkel
        if angle_mask.any():
            angle_tokens = tokens[angle_mask] - self.quantization_r
            embeddings[angle_mask] = self.angle_emb(angle_tokens)

        # Special tokens -> scalar_emb mit offset
        if special_mask.any():
            special_tokens = tokens[special_mask] - self.quantization_a
            embeddings[special_mask] = self.scalar_emb(special_tokens)

        return embeddings
