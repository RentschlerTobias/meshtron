import math
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from config import PipelineConfig
from dataset import MeshData
from logger import JSONLLogger
from quadtron import Quadtron
from metrics import EpochMetrics, TokenLossAccumulator
from objectives import (
    CriticAdvantage,
    GroupRelativeAdvantage,
    RLObjective,
    TeacherForcingObjective,
)
from policy import Policy
from reproducibility import dataloader_generator, set_seed, worker_init_fn
from tokenizer_v2 import Tokenizer2D


class TrainingCancelled(Exception):
    """Raise from an `on_epoch` callback (Trainer.run) or a `stop_check`
    (polytron_chain.run_polytron) to abort a run cooperatively, between
    epochs/stages -- Trainer.run()'s `finally` block still writes the partial
    result and closes the logger before this propagates to the caller."""


@dataclass
class RunResult:
    best_val_bpt: float
    best_val_nll: float
    best_val_perplexity: float
    best_epoch: int
    final_train_bpt: float
    final_val_bpt: float
    epochs_run: int
    config_hash: str
    run_dir: str


_PRECISION_DTYPE = {
    "fp32": None,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


class Trainer:
    """Schlanker Trainer fuer Quadtron.

    - Token-gewichteter Loss (sum/n_tokens) -> vergleichbar ueber Batch-Size,
      Sequenzlaenge und Padding hinweg.
    - bf16/fp16 Autocast optional, GradScaler nur fuer fp16.
    - Linear-Warmup + Cosine-Annealing.
    - JSONL-Logging pro Run, Checkpointing nur wenn ausdruecklich verlangt.
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        set_seed(cfg.seed, cudnn_deterministic=cfg.cudnn_deterministic)

        self.tokenizer = Tokenizer2D(
            quantization_levels=cfg.quantization,
            verbose=False,
            sorting_strategy=cfg.sorting_strategy,
            dim=cfg.dim,
        )

        (
            self.train_loader,
            self.val_loader,
            self.max_length,
            self.max_face_count,
            self.min_face_count,
        ) = self._build_loaders()

        self.model = Quadtron(
            vocab_size=self.tokenizer.vocab_size,
            d_model=cfg.d_model,
            max_seq_length=self.max_length,
            n_latents=cfg.n_latents,
            input_dim=cfg.dim,
            min_face_count=self.min_face_count,
            max_face_count=self.max_face_count,
            n_heads=cfg.n_heads,
            stage_layers=tuple(cfg.stage_layers),
            dropout=cfg.dropout,
            ffn_mult=cfg.ffn_mult,
            use_flash_attention=cfg.use_flash_attention,
            sliding_window_size=cfg.sliding_window_size,
            verbose=False,
        ).to(self.device)

        if cfg.init_checkpoint:
            ckpt = torch.load(cfg.init_checkpoint, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])

        self.policy = Policy(self.model)
        self.objective = self._build_objective(cfg)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        steps_per_epoch = max(1, len(self.train_loader) // max(1, cfg.accumulation_steps))
        self.total_steps = steps_per_epoch * cfg.num_epochs
        self.scheduler = self._build_scheduler(self.total_steps, cfg.warmup_steps)

        self.amp_dtype = _PRECISION_DTYPE.get(cfg.precision)
        self.use_autocast = self.amp_dtype is not None and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=(cfg.precision == "fp16"))

        self.config_hash = cfg.hash()
        self.logger = JSONLLogger(cfg.log_dir, self.config_hash, cfg.to_dict())

        self.global_step = 0
        self.best_val_bpt = float("inf")
        self.best_val_nll = float("inf")
        self.best_val_ppl = float("inf")
        self.best_epoch = -1
        self.last_train: Optional[EpochMetrics] = None
        self.last_val: Optional[EpochMetrics] = None

    # ------------------------------------------------------------------ public

    def run(
        self,
        on_epoch: Optional[Callable[[int, EpochMetrics, Optional[EpochMetrics]], None]] = None,
    ) -> RunResult:
        """Train the model.

        Args:
            on_epoch: optional callback invoked after each epoch with
                (epoch_index, train_metrics, val_metrics_or_None). May raise to
                abort training early (e.g. `optuna.TrialPruned`). Partial logs
                are still flushed via the `finally` block.
        """
        cfg = self.cfg
        epochs_run = 0
        try:
            for epoch in tqdm(range(cfg.num_epochs), desc="epochs"):
                epochs_run = epoch + 1
                train_metrics = self._epoch(self.train_loader, train=True, desc=f"train e{epoch}")
                self.last_train = train_metrics

                if (epoch % max(1, cfg.val_every_n_epochs)) == 0:
                    val_metrics = self._epoch(
                        self._val_iter(),
                        train=False,
                        desc=f"val e{epoch}",
                    )
                    self.last_val = val_metrics

                    improved = val_metrics.bits_per_token < self.best_val_bpt
                    if improved:
                        self.best_val_bpt = val_metrics.bits_per_token
                        self.best_val_nll = val_metrics.nll_per_token
                        self.best_val_ppl = val_metrics.perplexity
                        self.best_epoch = epoch
                        if cfg.save_best:
                            self._save_checkpoint("best.pt", epoch)

                    self.logger.log(
                        epoch=epoch,
                        step=self.global_step,
                        lr=self._current_lr(),
                        train_nll=train_metrics.nll_per_token,
                        train_bpt=train_metrics.bits_per_token,
                        train_ppl=train_metrics.perplexity,
                        train_tokens=train_metrics.n_tokens,
                        val_nll=val_metrics.nll_per_token,
                        val_bpt=val_metrics.bits_per_token,
                        val_ppl=val_metrics.perplexity,
                        val_tokens=val_metrics.n_tokens,
                        improved=improved,
                    )

                    if on_epoch is not None:
                        on_epoch(epoch, train_metrics, val_metrics)

                    if epoch - self.best_epoch >= cfg.early_stopping_patience:
                        break
                else:
                    self.logger.log(
                        epoch=epoch,
                        step=self.global_step,
                        lr=self._current_lr(),
                        train_nll=train_metrics.nll_per_token,
                        train_bpt=train_metrics.bits_per_token,
                        train_ppl=train_metrics.perplexity,
                        train_tokens=train_metrics.n_tokens,
                    )
                    if on_epoch is not None:
                        on_epoch(epoch, train_metrics, None)

            if cfg.save_last:
                self._save_checkpoint("last.pt", epochs_run - 1)
        finally:
            result = RunResult(
                best_val_bpt=self.best_val_bpt,
                best_val_nll=self.best_val_nll,
                best_val_perplexity=self.best_val_ppl,
                best_epoch=self.best_epoch,
                final_train_bpt=self.last_train.bits_per_token if self.last_train else float("nan"),
                final_val_bpt=self.last_val.bits_per_token if self.last_val else float("nan"),
                epochs_run=epochs_run,
                config_hash=self.config_hash,
                run_dir=str(self.logger.run_dir),
            )
            self.logger.write_result(**result.__dict__)
            self.logger.close()
        return result

    # ----------------------------------------------------------------- private

    def _build_objective(self, cfg: PipelineConfig):
        """TeacherForcingObjective (Teil A) unless cfg.rl_enabled, in which
        case RLObjective (Teil B) with the configured AdvantageEstimator and
        a Quadtron reward function for cfg.rl_curriculum_stage. Swapping GRPO
        <-> critic-based advantage is a config change (cfg.advantage_estimator),
        not a code change -- see objectives.py / the decision log."""
        if not cfg.rl_enabled:
            return TeacherForcingObjective(pad_token=self.tokenizer.pad_token)

        from rewards import make_quadtron_reward

        if cfg.advantage_estimator == "group_relative":
            estimator = GroupRelativeAdvantage()
        elif cfg.advantage_estimator == "critic":
            estimator = CriticAdvantage(device=self.device)
        else:
            raise ValueError(
                f"Unknown advantage_estimator={cfg.advantage_estimator!r} "
                "(expected 'group_relative' or 'critic')")

        reward_fn = make_quadtron_reward(self.tokenizer, cfg.rl_curriculum_stage,
                                         cfg.reward_weights)
        return RLObjective(
            reward_fn=reward_fn,
            advantage_estimator=estimator,
            pad_token=self.tokenizer.pad_token,
            eos_token=self.tokenizer.end_token,
            rollouts_per_condition=cfg.rl_rollouts_per_condition,
            temperature=cfg.rl_temperature,
            max_length=cfg.rl_max_length or None,
        )

    def _epoch(self, loader, train: bool, desc: str) -> EpochMetrics:
        self.model.train(mode=train)
        accumulator = TokenLossAccumulator()

        if train:
            self.optimizer.zero_grad(set_to_none=True)

        accum_counter = 0
        progress = tqdm(loader, desc=desc, leave=False)
        for batch in progress:
            ctx = (
                torch.autocast(device_type="cuda", dtype=self.amp_dtype)
                if self.use_autocast
                else _NullCtx()
            )
            grad_ctx = torch.enable_grad() if train else torch.no_grad()
            with grad_ctx, ctx:
                out = self.objective.compute(batch, self.policy)
            loss = out.loss

            if train:
                scaled = loss / max(1, self.cfg.accumulation_steps)
                self.scaler.scale(scaled).backward() if self.scaler.is_enabled() else scaled.backward()
                accum_counter += 1

                if accum_counter >= self.cfg.accumulation_steps:
                    if self.scaler.is_enabled():
                        self.scaler.unscale_(self.optimizer)
                    if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    if self.scaler.is_enabled():
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
                    self.global_step += 1
                    accum_counter = 0

            accumulator.update(out.loss_sum, out.n_tokens)
            progress.set_postfix(bpt=f"{accumulator.compute().bits_per_token:.3f}")

        return accumulator.compute()

    def _val_iter(self):
        if self.cfg.max_val_batches and self.cfg.max_val_batches > 0:
            return islice(self.val_loader, self.cfg.max_val_batches)
        return self.val_loader

    def _build_loaders(self):
        meshes = torch.load(self.cfg.data_path, weights_only=False)

        max_faces = 0
        min_faces = float("inf")
        for mesh in meshes:
            n = mesh.faces.size(1)
            max_faces = max(max_faces, n)
            min_faces = min(min_faces, n)

        max_token_sequence = (
            max_faces * self.tokenizer.tokens_per_face
            + 2 * self.tokenizer.n_start_end_tokens_repeat
        )
        self.tokenizer.max_length_padding = max_token_sequence

        gen = dataloader_generator(self.cfg.seed)
        train_meshes, val_meshes = random_split(
            meshes,
            [self.cfg.train_val_ratio, 1.0 - self.cfg.train_val_ratio],
            generator=gen,
        )

        train_dataset = MeshData(
            train_meshes,
            self.tokenizer,
            n_sample_points=self.cfg.n_sample_points,
            verbose=False,
            dim=self.cfg.dim,
        )
        val_dataset = MeshData(
            val_meshes,
            self.tokenizer,
            n_sample_points=self.cfg.n_sample_points,
            verbose=False,
            dim=self.cfg.dim,
        )

        common = dict(
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory and torch.cuda.is_available(),
            worker_init_fn=worker_init_fn if self.cfg.num_workers > 0 else None,
        )

        train_loader = DataLoader(
            train_dataset, shuffle=True, generator=gen, drop_last=True, **common
        )
        val_loader = DataLoader(val_dataset, shuffle=False, **common)

        max_length = max(train_dataset.max_seq_length, val_dataset.max_seq_length)
        return train_loader, val_loader, max_length, max_faces, min_faces

    def _build_scheduler(self, total_steps: int, warmup_steps: int):
        warmup_steps = max(0, min(warmup_steps, max(1, total_steps - 1)))

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _save_checkpoint(self, name: str, epoch: int) -> None:
        path = Path(self.logger.run_dir) / name
        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_bpt": self.best_val_bpt,
                "config": self.cfg.to_dict(),
                "max_seq_length": int(self.max_length),
                "max_face_count": int(self.max_face_count),
                "min_face_count": int(self.min_face_count),
                "pad_token": int(self.tokenizer.pad_token),
                "start_token": int(self.tokenizer.start_token),
                "end_token": int(self.tokenizer.end_token),
            },
            path,
        )


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
