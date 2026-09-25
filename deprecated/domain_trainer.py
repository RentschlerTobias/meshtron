"""
domain_trainer.py

DEPRECATED: part of the MeshtronDomain family (failed experiment, superseded by
Polytron / polytron_chain.py). Kept for reference only, not maintained or
extended.

Moderner Trainer fuer Domain-Partition Meshtron.
API-kompatibel mit trainer.py (tokenizer-sorting Branch).
"""


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
from itertools import islice
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from meshtron.training.config import DomainTrainingConfig
from dataset_domain import DomainMeshData
from meshtron.training.logger import JSONLLogger
from meshtron_domain import MeshtronDomain
from meshtron.training.metrics import EpochMetrics, TokenLossAccumulator
from meshtron.training.objectives import TeacherForcingObjective
from meshtron.training.policy import Policy
from meshtron.training.reproducibility import dataloader_generator, set_seed, worker_init_fn
from tokenizer_domain import DomainTokenizer


class DomainTrainer:
    """Trainer fuer Domain-Partition Meshtron.

    API-kompatibel mit Trainer (trainer.py) aber fuer Domain-Partition-Daten.
    """

    def __init__(self, cfg: DomainTrainingConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        set_seed(cfg.seed, cudnn_deterministic=cfg.cudnn_deterministic)

        self.tokenizer = DomainTokenizer(
            quantization_r=cfg.quantization_r,
            quantization_a=cfg.quantization_a,
            sorting_strategy=cfg.sorting_strategy,
            embedding_mode=cfg.embedding_mode,
            verbose=False,
        )

        (
            self.train_loader,
            self.val_loader,
            self.max_length,
            self.max_face_count,
            self.min_face_count,
        ) = self._build_loaders()

        self.model = MeshtronDomain(
            vocab_size=self.tokenizer.vocab_size,
            d_model=cfg.d_model,
            max_seq_length=self.max_length,
            n_latents=cfg.n_latents,
            input_dim=2,
            n_point_labels=4 if getattr(cfg, 'point_cloud_labels', False) else None,
            min_face_count=self.min_face_count,
            max_face_count=self.max_face_count,
            n_heads=cfg.n_heads,
            stage_layers=tuple(cfg.stage_layers),
            embedding_mode=cfg.embedding_mode,
            quantization_r=cfg.quantization_r,
            quantization_a=cfg.quantization_a,
            verbose=False,
        ).to(self.device)

        self.policy = Policy(self.model)
        self.objective = TeacherForcingObjective(pad_token=self.tokenizer.pad_token)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        steps_per_epoch = max(1, len(self.train_loader) // max(1, cfg.accumulation_steps))
        self.total_steps = steps_per_epoch * cfg.num_epochs
        self.scheduler = self._build_scheduler(self.total_steps, cfg.warmup_steps)

        self.amp_dtype = {
            "fp32": None,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }.get(cfg.precision)
        self.use_autocast = self.amp_dtype is not None and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler('cuda', enabled=(cfg.precision == "fp16"))

        self.config_hash = cfg.hash()
        self.logger = JSONLLogger(cfg.log_dir, self.config_hash, cfg.to_dict())

        self.global_step = 0
        self.best_val_bpt = float("inf")
        self.best_val_nll = float("inf")
        self.best_val_ppl = float("inf")
        self.best_epoch = -1
        self.last_train: Optional[EpochMetrics] = None
        self.last_val: Optional[EpochMetrics] = None

    def run(
        self,
        on_epoch: Optional[Callable[[int, EpochMetrics, Optional[EpochMetrics]], None]] = None,
    ):
        """Train the model."""
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
            result = {
                "best_val_bpt": self.best_val_bpt,
                "best_val_nll": self.best_val_nll,
                "best_val_perplexity": self.best_val_ppl,
                "best_epoch": self.best_epoch,
                "final_train_bpt": self.last_train.bits_per_token if self.last_train else float("nan"),
                "final_val_bpt": self.last_val.bits_per_token if self.last_val else float("nan"),
                "epochs_run": epochs_run,
                "config_hash": self.config_hash,
                "run_dir": str(self.logger.run_dir),
            }
            self.logger.write_result(**result)
            self.logger.close()
        return result

    # ----------------------------------------------------------------- private

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
                if self.scaler.is_enabled():
                    self.scaler.scale(scaled).backward()
                else:
                    scaled.backward()
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
            n = mesh['faces'].shape[1]
            max_faces = max(max_faces, n)
            min_faces = min(min_faces, n)

        # Schätze max_seq_length
        n_places = sum(m['faces'].shape[1] * 4 for m in meshes) / len(meshes)
        tokens_per_mesh = int(n_places * 8 + 16)  # 8 tokens/place + start/end
        self.tokenizer.max_length_padding = max(tokens_per_mesh, 1000)

        gen = dataloader_generator(self.cfg.seed)
        train_meshes, val_meshes = random_split(
            meshes,
            [self.cfg.train_val_ratio, 1.0 - self.cfg.train_val_ratio],
            generator=gen,
        )

        wl = getattr(self.cfg, 'point_cloud_labels', False)
        train_dataset = DomainMeshData(
            list(train_meshes),
            self.tokenizer,
            n_sample_points=self.cfg.n_sample_points,
            with_labels=wl,
            verbose=False,
        )
        val_dataset = DomainMeshData(
            list(val_meshes),
            self.tokenizer,
            n_sample_points=self.cfg.n_sample_points,
            with_labels=wl,
            verbose=False,
        )

        common = dict(
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory and torch.cuda.is_available(),
            drop_last=True,
            worker_init_fn=worker_init_fn if self.cfg.num_workers > 0 else None,
        )

        train_loader = DataLoader(train_dataset, shuffle=True, generator=gen, **common)
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
        pass
