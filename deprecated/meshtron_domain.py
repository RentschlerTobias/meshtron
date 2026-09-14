"""
meshtron_domain.py

DEPRECATED: failed experiment (severe overfitting, invalid generated output —
see docs/ho_quad_transformer/01_current_model_and_diagnosis.md). Superseded by
Polytron (polytron_tokenizer.py, polytron_chain.py). Kept for reference only,
not maintained or extended.

Meshtron-Variante fuer Domain-Partition Training.

Unterschiede zum Original:
  - DomainEmbedding statt Embedding
  - Kuerzere Sequenzen (max ~1000 statt 8000)
  - Kleinere Face-Count Range (1-50 statt 2000-8000)
  - Kein Windowing noetig (Sequenzen sind kurz genug)
"""


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from typing import Tuple, Dict
from hourglass_transformer import HourglassTransformer
from point_encoder import PerceiverPointEncoder
from faceCount_encoder import FaceCountEncoder
from domain_embedding import DomainEmbedding


class MeshtronDomain(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        max_seq_length: int = 1000,
        n_latents: int = 512,
        input_dim: int = 2,
        n_point_labels: int = None,
        min_face_count: int = 1,
        max_face_count: int = 50,
        n_heads: int = 8,
        stage_layers: Tuple[int, int, int, int, int] = (2, 4, 6, 8, 10),
        embedding_mode: int = 0,
        quantization_r: int = 512,
        quantization_a: int = 256,
        verbose: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.n_heads = n_heads

        if verbose:
            print("init DomainEmbedding")
        self.embedder = DomainEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_seq_length,
            embedding_mode=embedding_mode,
            quantization_r=quantization_r,
            quantization_a=quantization_a,
        )

        if verbose:
            print("init PerceiverPointEncoder")
        self.point_encoder = PerceiverPointEncoder(
            d_model=d_model,
            input_dim=input_dim,
            n_latents=n_latents,
            n_point_labels=n_point_labels,
        )

        if verbose:
            print("init FaceCountEncoder")
        self.face_count_encoder = FaceCountEncoder(
            d_model=d_model,
            min_face_count=min_face_count,
            max_face_count=max_face_count,
        )

        if verbose:
            print("init HourglassTransformer")
        self.transformer = HourglassTransformer(
            d_model=d_model,
            n_heads=n_heads,
            stage_layers=stage_layers,
            d_ff=4 * d_model,
            dropout=0.1,
            max_position=max_seq_length,
        )

        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_tokens, point_cloud, face_count, position_ids=None):
        """
        Forward Pass fuer Training.

        Args:
            input_tokens: [batch_size, seq_len]
            point_cloud: [batch_size, num_points, 2] (oder 3 mit Label-Spalte, falls
                          n_point_labels gesetzt)
            face_count: [batch_size]
            position_ids: optional [batch_size, seq_len]
        """
        # Embeddings
        embeddings = self.embedder(input_tokens)

        # Point Cloud + Face Count Conditioning
        latent_points = self.point_encoder(point_cloud)
        latent_faces = self.face_count_encoder(face_count)
        latent_faces_seq = latent_faces.unsqueeze(1)

        latent_condition = torch.cat([latent_points, latent_faces_seq], dim=1)

        # Transformer
        output = self.transformer(
            x=embeddings,
            latent_condition=latent_condition,
            position_ids=position_ids,
        )

        # Output Head
        logits = self.output_head(output)
        return logits

    def generate(self, point_cloud, face_count, tokenizer, max_length=1000,
                 temperature=1.0, top_k=None, device='cuda'):
        """
        Autoregressive Generierung.
        
        Args:
            point_cloud: [1, num_points, 2]
            face_count: [1]
            tokenizer: DomainTokenizer
        """
        self.eval()
        with torch.no_grad():
            # Start tokens
            start_tokens = [tokenizer.start_token] * 8
            generated = torch.tensor([start_tokens], dtype=torch.long, device=device)

            for _ in range(max_length):
                position_ids = torch.arange(generated.size(1), device=device).unsqueeze(0)
                logits = self.forward(generated, point_cloud, face_count, position_ids)
                next_token_logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    v, _ = torch.topk(next_token_logits, top_k)
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float('inf')

                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                generated = torch.cat([generated, next_token], dim=1)

                if next_token.item() == tokenizer.end_token:
                    break

        return generated[0].cpu().tolist()
