import torch.nn as nn
import torch
import math


def fourier_features(x, n_freqs=8):
    """
    x: [B, N, input_dim]
    Encodes each coordinate with sin/cos at 2^0 ... 2^(n_freqs-1) frequencies.
    returns: [B, N, input_dim * (1 + 2*n_freqs)]
    """
    freqs = 2.0 ** torch.arange(n_freqs, device=x.device, dtype=x.dtype)  # [n_freqs]
    x_freq = x.unsqueeze(-1) * freqs * math.pi  # [B, N, input_dim, n_freqs]
    sin_feat = torch.sin(x_freq).flatten(-2)     # [B, N, input_dim*n_freqs]
    cos_feat = torch.cos(x_freq).flatten(-2)     # [B, N, input_dim*n_freqs]
    return torch.cat([x, sin_feat, cos_feat], dim=-1)  # [B, N, input_dim*(1+2*n_freqs)]


class PerceiverPointEncoder(nn.Module):

    def __init__(self, d_model=128, input_dim=2, n_latents=32, n_freqs=8,
                 n_point_labels=None):
        """n_point_labels: wenn gesetzt, erwartet forward Punkte [B,N,input_dim+1],
        wobei die LETZTE Spalte ein Kategorie-Label {0..n_point_labels-1} ist
        (0=Singularitaet/Ecke, 1=Rand, 2=Feld, 3=Pad). Das Label wird per Embedding
        addiert -> das Modell kann Ecken (=exakte Vertices) von Rand/Feld trennen.
        None -> altes Verhalten (nur x,y)."""
        super().__init__()

        self.n_freqs = n_freqs
        self.n_point_labels = n_point_labels
        # Must match fourier_features()'s actual output width exactly: x itself
        # (input_dim) + sin_feat (input_dim*n_freqs) + cos_feat (input_dim*n_freqs).
        # `input_dim + 4*n_freqs` looks like the same thing but only coincides
        # with this for input_dim=2 (the only value this ever ran with before
        # dim=3 support existed) -- e.g. input_dim=3, n_freqs=8: 3+32=35 here
        # vs the function's real 3*(1+16)=51, a silent shape-mismatch crash at
        # the first forward() call. Compute it the same way the function does.
        fourier_dim = input_dim * (1 + 2 * n_freqs)  # z.B. 2*(1+16) = 34

        self.latents = nn.Parameter(torch.randn(n_latents, d_model))

        self.point_proj = nn.Linear(fourier_dim, d_model)
        if n_point_labels is not None:
            self.label_emb = nn.Embedding(n_point_labels, d_model)

        # Cross-attention: latents attend to points
        self.cross_attention = nn.MultiheadAttention(
            d_model, num_heads=4, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff_cross = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

        # Self-attention between latents
        self.self_attention = nn.MultiheadAttention(
            d_model, num_heads=4, batch_first=True
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.ff_self = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, points):
        """
        points: [batch, n_points, 2]  (oder [.,.,3] mit Label in Spalte -1, falls
                n_point_labels gesetzt)
        output: [batch, n_latents, d_model]
        """
        batch_size = points.shape[0]

        if self.n_point_labels is not None:
            coords = points[..., :-1]                            # [B, N, 2]
            labels = points[..., -1].long().clamp(0, self.n_point_labels - 1)
            point_features = fourier_features(coords, self.n_freqs)
            point_features = self.point_proj(point_features)     # [B, N, d_model]
            point_features = point_features + self.label_emb(labels)
        else:
            point_features = fourier_features(points, self.n_freqs)  # [B, N, 2+4*nf]
            point_features = self.point_proj(point_features)         # [B, N, d_model]

        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)

        # Cross-attention block (Pre-Norm)
        normed = self.norm1(latents)
        attn_out, _ = self.cross_attention(
            query=normed, key=point_features, value=point_features
        )
        latents = latents + attn_out
        latents = latents + self.ff_cross(self.norm2(latents))

        # Self-attention block (Pre-Norm)
        normed = self.norm3(latents)
        attn2_out, _ = self.self_attention(normed, normed, normed)
        latents = latents + attn2_out
        latents = latents + self.ff_self(self.norm4(latents))

        return latents  # [batch, n_latents, d_model]
