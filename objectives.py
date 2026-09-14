from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from policy import Policy


@dataclass
class ObjectiveOutput:
    loss: torch.Tensor       # skaliert auf pro-Token-Magnitude (fuer backward)
    loss_sum: float          # Summe der NLL ueber valide Tokens (fuer Logging)
    n_tokens: int            # Anzahl valider Tokens im Batch


class Objective(ABC):
    @abstractmethod
    def compute(self, batch: dict, policy: Policy) -> ObjectiveOutput:
        ...


class TeacherForcingObjective(Objective):
    """Standard-Cross-Entropy mit Pad-Ignore.

    Der zurueckgegebene `loss` ist pro-Token gemittelt (sum / n_tokens),
    damit die Gradientenmagnitude unabhaengig von Batch-Size und Padding ist.
    Fuer Logging wird zusaetzlich die unnormierte `loss_sum` mitgegeben,
    damit der Akkumulator korrekt gewichtet mitteln kann.
    """

    def __init__(self, pad_token: int):
        self.pad_token = int(pad_token)

    def compute(self, batch: dict, policy: Policy) -> ObjectiveOutput:
        device = next(policy.parameters()).device
        input_tokens = batch["input_tokens"].to(device, non_blocking=True)
        target_tokens = batch["target_tokens"].to(device, non_blocking=True)
        point_cloud = batch["point_cloud"].to(device, non_blocking=True)
        face_count = batch["face_count"].to(device, non_blocking=True)

        logits = policy.logits(input_tokens, point_cloud, face_count, None)

        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_targets = target_tokens.reshape(-1)

        loss_sum_t = F.cross_entropy(
            flat_logits,
            flat_targets,
            ignore_index=self.pad_token,
            reduction="sum",
        )
        n_tokens = int((flat_targets != self.pad_token).sum().item())

        if n_tokens == 0:
            zero = loss_sum_t * 0.0
            return ObjectiveOutput(loss=zero, loss_sum=0.0, n_tokens=0)

        loss = loss_sum_t / n_tokens
        return ObjectiveOutput(
            loss=loss,
            loss_sum=float(loss_sum_t.detach().item()),
            n_tokens=n_tokens,
        )


# ------------------------------------------------------------------
# Part B: RL curriculum (see docs/decisions/2026-09-13-meshtron-refactor-plan.md).
# Pluggable advantage estimator: GRPO (group-relative, no critic, default) and
# PPO (critic-based) share the identical `RLObjective` skeleton and differ only
# in where the baseline comes from -- swapping one for the other is a config
# change (PipelineConfig.advantage_estimator), not a rewrite of the RL loop.
# ------------------------------------------------------------------
class AdvantageEstimator(ABC):
    @abstractmethod
    def compute(self, rewards: torch.Tensor) -> torch.Tensor:
        """rewards: [G] (one rollout group, same condition) -> advantages [G]."""
        ...


class GroupRelativeAdvantage(AdvantageEstimator):
    """GRPO: baseline = mean of the group's own rewards, no critic network.
    Default -- needs no second network to train/stabilize, which matters at
    the small model/dataset sizes this project trains at (see decision log)."""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def compute(self, rewards: torch.Tensor) -> torch.Tensor:
        if rewards.numel() <= 1:
            return torch.zeros_like(rewards)
        mean = rewards.mean()
        std = rewards.std(unbiased=False)
        return (rewards - mean) / (std + self.eps)


class CriticAdvantage(AdvantageEstimator):
    """PPO-style: baseline = a learned value head over caller-supplied
    condition features (e.g. face_count), not the group mean. Deliberately
    minimal (retrofittable behind the same interface if GRPO proves unstable
    in practice, per the decision log -- not the default, not gold-plated)."""

    def __init__(self, feature_dim: int = 1, hidden: int = 32, device=None):
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        if device is not None:
            self.critic = self.critic.to(device)

    def compute(self, rewards: torch.Tensor, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        if features is None:
            # No condition features supplied -> falls back to the group mean,
            # i.e. behaves like GroupRelativeAdvantage's un-normalized cousin.
            return rewards - rewards.mean()
        baseline = self.critic(features).squeeze(-1)
        return rewards - baseline

    def parameters(self):
        return self.critic.parameters()


class RLObjective(Objective):
    """Policy-gradient objective for the RL curriculum. Two-pass pattern,
    standard for autoregressive policy gradients:
      1. Roll out `rollouts_per_condition` sequences per batch item with the
         CURRENT policy (`Policy.sample()`, no_grad -- cheap generation).
      2. Re-score those exact sampled sequences through `Policy.logits()`
         (teacher-forcing on the model's OWN samples, WITH grad) to get a
         differentiable log-probability per sequence.
    Loss = -(advantage * sum_t log P(token_t)), advantage from a pluggable
    `AdvantageEstimator`. Returns the same `ObjectiveOutput(loss, loss_sum,
    n_tokens)` contract `TeacherForcingObjective` does, so `Trainer._epoch`
    needs no changes to consume either objective interchangeably.

    `reward_fn(token_ids: List[int], face_count: int) -> float` is supplied
    by the caller (see rewards.py for the concrete per-stage implementations,
    Part B2) -- kept as a plain callable here so this class has no knowledge
    of what "good" means for any particular curriculum stage.
    """

    def __init__(
        self,
        reward_fn: Callable[[list, int], float],
        advantage_estimator: AdvantageEstimator,
        pad_token: int,
        eos_token: int,
        start_prefix_len: int = 8,
        rollouts_per_condition: int = 8,
        temperature: float = 1.0,
        max_length: Optional[int] = None,
    ):
        self.reward_fn = reward_fn
        self.advantage_estimator = advantage_estimator
        self.pad_token = int(pad_token)
        self.eos_token = int(eos_token)
        self.start_prefix_len = start_prefix_len
        self.rollouts_per_condition = rollouts_per_condition
        self.temperature = temperature
        self.max_length = max_length

    def compute(self, batch: dict, policy: Policy) -> ObjectiveOutput:
        device = next(policy.parameters()).device
        point_cloud = batch["point_cloud"].to(device, non_blocking=True)
        face_count = batch["face_count"].to(device, non_blocking=True)
        input_tokens = batch["input_tokens"].to(device, non_blocking=True)
        start_tokens = input_tokens[:, : self.start_prefix_len]
        max_length = self.max_length or input_tokens.size(1)

        B = point_cloud.size(0)
        G = self.rollouts_per_condition
        per_item_losses = []
        n_reward_tokens = 0

        for b in range(B):
            pc = point_cloud[b : b + 1].expand(G, *point_cloud.shape[1:])
            fc = face_count[b : b + 1].expand(G)
            st = start_tokens[b : b + 1].expand(G, -1)

            with torch.no_grad():
                rollouts = policy.sample(
                    point_cloud=pc, face_count=fc, start_tokens=st,
                    max_length=max_length, temperature=self.temperature,
                    eos_token=self.eos_token,
                )  # [G, L]

            fc_val = int(face_count[b].item())
            rewards = torch.tensor(
                [self.reward_fn(rollouts[g].tolist(), fc_val) for g in range(G)],
                dtype=torch.float32, device=device,
            )
            advantages = self.advantage_estimator.compute(rewards)

            # Re-score WITH grad: teacher-forcing over the policy's own samples.
            gen_input = rollouts[:, :-1]
            gen_target = rollouts[:, 1:]
            logits = policy.logits(gen_input, pc, fc, None)          # [G, L-1, V]
            log_probs = F.log_softmax(logits, dim=-1)
            token_logp = log_probs.gather(-1, gen_target.unsqueeze(-1)).squeeze(-1)
            valid = (gen_target != self.pad_token).float()
            seq_logp = (token_logp * valid).sum(dim=1)                # [G]

            per_item_losses.append(-(advantages.detach() * seq_logp))
            n_reward_tokens += int(valid.sum().item())

        loss = torch.cat(per_item_losses).mean()
        return ObjectiveOutput(
            loss=loss,
            loss_sum=float(loss.detach().item()) * B,
            n_tokens=max(n_reward_tokens, 1),
        )
