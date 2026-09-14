from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    """Duenner Wrapper um ein Sequenzmodell (Quadtron/Polytron), der Trainings- und Sampling-Pfad trennt.

    Heute nutzt der Trainer nur `logits()` (Teacher Forcing).
    `sample()` ist fuer eine spaetere RL-Phase vorbereitet, in der ganze
    Sequenzen ausgerollt werden, um sequenzbasierte Rewards zu lernen.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def logits(
        self,
        input_tokens: torch.Tensor,
        point_cloud: torch.Tensor,
        face_count: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.model(input_tokens, point_cloud, face_count, position_ids)

    @torch.no_grad()
    def sample(
        self,
        point_cloud: torch.Tensor,
        face_count: torch.Tensor,
        start_tokens: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        eos_token: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressives Sampling fuer Inferenz / RL-Rollouts."""
        device = next(self.model.parameters()).device
        generated = start_tokens.to(device)
        if generated.dim() == 1:
            generated = generated.unsqueeze(0)

        for _ in range(max_length):
            logits = self.model(generated, point_cloud, face_count, None)
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            if eos_token is not None and (next_token == eos_token).all():
                break
        return generated
