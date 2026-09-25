import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FaceCountEncoder(nn.Module):

    """
    Encodiert Face Count zu Latent Embeddings
    Sinusoidal Encoding mit Offset für min_face_count
    Ähnlich wie Positional Encoding, aber für Face Counts
    """

    def __init__(self, d_model: int, min_face_count: int = 10, max_face_count: int = 100):
        super().__init__()

        self.min_face_count = min_face_count
        self.max_face_count = max_face_count
        self.d_model = d_model

        # Optional: Learnable scaling
        self.scale = nn.Parameter(torch.tensor(1.0))

        # Optional: MLP für weitere Verarbeitung
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, face_count):
        """
        Sinusoidal Encoding mit Offset
        """
        batch_size = face_count.shape[0]
        device = face_count.device

        # Offset by min_face_count
        # [batch, 1]
        position = (face_count - self.min_face_count).unsqueeze(1).float()

        # Erstelle div_term für verschiedene Frequenzen
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device) *
            -(math.log(10000.0) / self.d_model)
        )

        # Erstelle Encoding
        encoding = torch.zeros(batch_size, self.d_model, device=device)
        encoding[:, 0::2] = torch.sin(position * div_term * self.scale)
        encoding[:, 1::2] = torch.cos(position * div_term * self.scale)

        output = self.mlp(encoding)

        return output
