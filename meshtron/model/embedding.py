import torch
import torch.nn as nn
import math

from meshtron.model.positional_encoder import RoPEPositionalEncoding


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(max_len*2) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model] oder [seq_len, d_model]
        if x.dim() == 3:  # Batch
            seq_len = x.size(1)
            return x + self.pe[:seq_len].unsqueeze(0)
        else:  # Einzelne Sequenz
            seq_len = x.size(0)
            return x + self.pe[:seq_len]


class Embedding(nn.Module):
    def __init__(self, vocab_size=259, d_model=512, max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # self.pos_encoding = PositionalEncoding(d_model, max_len)
        # self.pos_encoding = RoPEPositionalEncoding(d_model, max_len)

    def forward(self, tokens):
        embeddings = self.embedding(tokens)
        return embeddings
