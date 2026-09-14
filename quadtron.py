
from embedding import Embedding

from typing import Optional, Tuple, Dict
from hourglass_transformer import HourglassTransformer
import torch
import torch.nn as nn
from point_encoder import PerceiverPointEncoder
from faceCount_encoder import FaceCountEncoder

import torch.nn.functional as F

from tqdm import tqdm


class Quadtron(nn.Module):
    """
    Quadtron: rohe Quad-Faces, row-encoded Tokenization (tokenizer_v2.Tokenizer2D).
    Punktwolke + Face-Count -> flache Quad-Token-Sequenz.
    """

    def __init__(self,
                 vocab_size: int,
                 d_model: int = 1024,
                 max_seq_length: int = 8000,
                 n_latents: int = 1024,
                 input_dim: int = 2,
                 min_face_count: int = 2000,
                 max_face_count: int = 8000,
                 n_heads: int = 8,
                 stage_layers: Tuple[int, int, int,
                                     int, int] = (4, 8, 12, 16, 20),
                 dropout: float = 0.1,
                 ffn_mult: int = 4,
                 use_flash_attention: bool = False,
                 sliding_window_size: int = 0,
                 verbose=True):
        super().__init__()

        if verbose == True:
            print('init Embedding')
        self.n_heads = n_heads
        self.embedder = Embedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_seq_length
        )

        if verbose == True:
            print('init PerceiverPointEncoder')

        self.point_encoder = PerceiverPointEncoder(
            d_model=d_model,
            input_dim=input_dim,
            n_latents=n_latents
        )

        if verbose == True:
            print('init FaceCountEncoder')

        self.face_count_encoder = FaceCountEncoder(
            d_model=d_model,
            min_face_count=min_face_count,
            max_face_count=max_face_count
        )

        if verbose == True:
            print('init HourglassTransformer')

        self.transformer = HourglassTransformer(
            d_model=d_model,
            n_heads=self.n_heads,
            stage_layers=stage_layers,
            d_ff=ffn_mult * d_model,
            dropout=dropout,
            max_position=max_seq_length,
            use_flash_attention=use_flash_attention,
            sliding_window_size=sliding_window_size,
        )

        # Output Head - projiziert zurück auf Vocab Size
        self.output_head = nn.Linear(d_model, vocab_size)

        # Speichere Konfiguration
        self.vocab_size = vocab_size
        self.d_model = d_model

    def forward(self, input_tokens, point_cloud, face_count, position_ids):
        """
        Forward Pass für Training

        input_tokens: [batch_size, seq_len] - Token IDs
        point_cloud: [batch_size, num_points, 2] - Point Cloud für Conditioning
        """
        # Embeddings mit Positional Encoding (bereits in deinem Embedder)
        embeddings = self.embedder(input_tokens)

        # Encode Point Cloud zu Latent Condition
        latent_points = self.point_encoder(point_cloud)
        latent_faces = self.face_count_encoder(face_count)
        latent_faces_seq = latent_faces.unsqueeze(1)

        latent_condition = torch.cat([latent_points, latent_faces_seq], dim=1)
        # Durch Hourglass Transformer (mit kausaler Maske!)
        output = self.transformer(
            x=embeddings,
            latent_condition=latent_condition,
            position_ids=position_ids,
            is_casual=True  # WICHTIG: Kausale Maske aktiviert fuer autoregressives training

        )

        # Projiziere auf Vocabulary
        logits = self.output_head(output)

        return logits

    @torch.no_grad()
    def generate(self, point_cloud, face_count, start_tokens, max_length=10000, temperature=1.0, device='cuda'):
        """
        AUTOREGRESSIVE GENERIERUNG - Schritt für Schritt

        Bei Generierung: Token für Token generieren

        Args:
            point_cloud: [1, num_points, 2] - Point Cloud Condition
            start_tokens: Optionale Start-Tokens (z.B. [BOS])
            max_length: Maximale Länge der generierten Sequenz
            temperature: Sampling-Temperatur (höher = mehr Variabilität)
        """
        self.eval()

        generated = torch.tensor(start_tokens, device=device).clone().detach()
        # generated = start_tokens.clone().detach().device(device)

        # Encode Point Cloud einmal
        latent_points = self.point_encoder(point_cloud.to(device))
        latent_faces = self.face_count_encoder(face_count.to(device))

        latent_condition = torch.cat(
            [latent_points, latent_faces.unsqueeze(1)], dim=1)

        #
        # EOS Token (angenommen als vocab_size - 2)
        eos_token = self.vocab_size - 2

        for i in tqdm(range(max_length), desc="num tokens"):
            # Embeddings für bisherige Sequenz
            embeddings = self.embedder(generated)

            # Forward durch Transformer
            output = self.transformer(
                x=embeddings,
                latent_condition=latent_condition,
                is_casual=True
            )

            # Nur das letzte Token für Vorhersage nutzen
            # [batch_size, vocab_size]
            logits = self.output_head(output[:, -1, :])

            # Sampling mit Temperatur
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Füge neues Token zur Sequenz hinzu
            generated = torch.cat([generated, next_token], dim=1)

            # Stoppe bei EOS Token
            if next_token.item() == eos_token:
                break

        return generated.squeeze(0)  # Entferne Batch-Dimension
