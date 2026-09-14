"""
train_domain.py

DEPRECATED: MeshtronDomain was a failed experiment, superseded by Polytron
(polytron_chain.py). Kept for reference only, not maintained or extended — the
import below is stale (DomainData/get_loaders don't exist in dataset_domain.py
any more, see DomainMeshData/get_domain_loaders) and was left as-is rather than
fixed, since this file is not on the active path.

Trainingsskript fuer MeshtronDomain.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import numpy as np

from meshtron_domain import MeshtronDomain
from dataset_domain import DomainData, get_loaders
from tokenizer_domain import DomainTokenizer


class TrainerDomain:
    def __init__(
        self,
        data_path='/root/repos/meshtron/domain_data.pt',
        checkpoint_dir='/root/repos/meshtron/checkpoints_domain',
        quantization_r=64,
        quantization_a=32,
        sorting_strategy=0,
        embedding_mode=0,
        d_model=512,
        n_latents=512,
        batch_size=8,
        num_epochs=100,
        learning_rate=1e-4,
        stage_layers=(2, 4, 6, 8, 10),
        n_heads=8,
        max_seq_length=None,
        verbose=True,
        device=None,
    ):
        self.verbose = verbose
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.quantization_r = quantization_r
        self.quantization_a = quantization_a
        self.sorting_strategy = sorting_strategy
        self.embedding_mode = embedding_mode
        self.d_model = d_model
        self.n_latents = n_latents
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.stage_layers = stage_layers
        self.n_heads = n_heads
        self.data_path = data_path
        self.max_seq_length = max_seq_length

        # Tokenizer und Data
        self.tokenizer = DomainTokenizer(
            quantization_r=quantization_r,
            quantization_a=quantization_a,
            sorting_strategy=sorting_strategy,
            embedding_mode=embedding_mode,
            verbose=verbose,
        )

        self.train_loader, self.val_loader, self.max_length, self.tokenizer = get_loaders(
            data_path=data_path,
            train_ratio=0.8,
            batch_size=batch_size,
            tokenizer=self.tokenizer,
        )

        if self.max_seq_length is not None:
            self.max_length = self.max_seq_length

        # Modell
        if self.verbose:
            print('\ninit MeshtronDomain')
        self.model = MeshtronDomain(
            vocab_size=self.tokenizer.vocab_size,
            d_model=d_model,
            max_seq_length=self.max_length,
            n_latents=n_latents,
            input_dim=2,
            min_face_count=1,
            max_face_count=50,
            n_heads=n_heads,
            stage_layers=stage_layers,
            embedding_mode=embedding_mode,
            quantization_r=quantization_r,
            quantization_a=quantization_a,
            verbose=verbose,
        ).to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=1e-6
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token)

        # Checkpointing
        self.notation = (
            f"qr{quantization_r}_qa{quantization_a}_"
            f"str{sorting_strategy}_emb{embedding_mode}_"
            f"d{d_model}_nl{n_latents}_bs{batch_size}_"
            f"heads{n_heads}_layers{'_'.join(str(s) for s in stage_layers)}"
        )
        self.checkpoint_dir = Path(checkpoint_dir) / self.notation
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float('inf')
        self.training_history = {'train_loss': [], 'val_loss': [], 'epoch': []}
        self.current_epoch = 0

        if self.verbose:
            print(f"\n=== Training Config ===")
            print(f"device: {self.device}")
            print(f"vocab_size: {self.tokenizer.vocab_size}")
            print(f"max_seq_length: {self.max_length}")
            print(f"d_model: {d_model}")
            print(f"batch_size: {batch_size}")
            print(f"epochs: {num_epochs}")
            print(f"lr: {learning_rate}")
            print(f"checkpoint_dir: {self.checkpoint_dir}")

    def run_epoch(self, loader, is_train=True):
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_samples = 0
        num_batches = 0

        pbar = tqdm(loader, desc='Train' if is_train else 'Val')
        for batch in pbar:
            input_tokens = batch['input_tokens'].to(self.device)
            target_tokens = batch['target_tokens'].to(self.device)
            point_cloud = batch['point_cloud'].to(self.device)
            face_count = batch['face_count'].to(self.device)
            padding_mask = batch['padding_mask'].to(self.device)

            if is_train:
                self.optimizer.zero_grad()

            logits = self.model(input_tokens, point_cloud, face_count)
            # logits: [batch, seq_len, vocab_size]
            # target: [batch, seq_len]
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                target_tokens.view(-1)
            )

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            total_loss += loss.item() * input_tokens.size(0)
            total_samples += input_tokens.size(0)
            num_batches += 1

            pbar.set_postfix({'loss': loss.item()})

        return total_loss / total_samples if total_samples > 0 else 0.0

    def train(self):
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            train_loss = self.run_epoch(self.train_loader, is_train=True)
            val_loss = self.run_epoch(self.val_loader, is_train=False)

            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['epoch'].append(epoch)

            self.scheduler.step()

            print(f"Epoch {epoch+1}/{self.num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Checkpoint speichern
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pt')
                print(f"  -> New best model saved (val_loss={val_loss:.4f})")

            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')

    def save_checkpoint(self, filename):
        path = self.checkpoint_dir / filename
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'config': {
                'quantization_r': self.quantization_r,
                'quantization_a': self.quantization_a,
                'sorting_strategy': self.sorting_strategy,
                'embedding_mode': self.embedding_mode,
                'd_model': self.d_model,
                'n_latents': self.n_latents,
                'batch_size': self.batch_size,
                'stage_layers': self.stage_layers,
                'n_heads': self.n_heads,
            }
        }, path)

    def load_checkpoint(self, filename):
        path = self.checkpoint_dir / filename
        if not path.exists():
            print(f"Checkpoint {path} not found")
            return
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.current_epoch = ckpt['epoch']
        self.best_val_loss = ckpt['best_val_loss']
        self.training_history = ckpt['training_history']
        print(f"Loaded checkpoint from epoch {self.current_epoch}")


if __name__ == '__main__':
    trainer = TrainerDomain(
        quantization_r=64,
        quantization_a=32,
        sorting_strategy=0,
        embedding_mode=0,
        d_model=256,
        n_latents=256,
        batch_size=8,
        num_epochs=50,
        learning_rate=1e-3,
        stage_layers=(2, 4, 6, 8, 10),
        n_heads=4,
    )
    trainer.train()
