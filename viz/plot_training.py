"""
plot_training.py

Visualisiert Training-Loss und andere Metriken.
"""


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch


def plot_loss_history(trainer, output_dir='./figures'):
    """Plot Train/Val Loss."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hist = trainer.training_history
    epochs = hist['epoch']
    train_loss = hist['train_loss']
    val_loss = hist['val_loss']

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_loss, 'b-', linewidth=2, label='Train Loss', marker='o', markersize=4)
    ax.plot(epochs, val_loss, 'r-', linewidth=2, label='Val Loss', marker='s', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax.set_title('Training History', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Best val loss marker
    best_idx = np.argmin(val_loss)
    ax.axvline(x=epochs[best_idx], color='green', linestyle='--', alpha=0.5, label=f'Best Val @ Epoch {epochs[best_idx]+1}')
    ax.scatter([epochs[best_idx]], [val_loss[best_idx]], s=100, c='green', zorder=5, edgecolors='black')

    plt.tight_layout()
    output_file = output_dir / 'domain_training_loss.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Gespeichert: {output_file}")


def plot_token_distribution(tokenizer, data, output_dir='./figures'):
    """Histogramm der Token-Werte im Datensatz."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_tokens = []
    for mesh in data:
        tokens = tokenizer.tokenize(mesh)
        all_tokens.extend(tokens)

    all_tokens = np.array(all_tokens)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogramm
    ax1 = axes[0]
    ax1.hist(all_tokens, bins=tokenizer.vocab_size, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(x=tokenizer.coord_vocab_size, color='red', linestyle='--', linewidth=2, label='Special Token Boundary')
    ax1.set_xlabel('Token ID', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Token Distribution in Dataset', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Token-Typen
    ax2 = axes[1]
    type_names = ['r', 'θsin', 'θcos', 't_norm', 'α_in sin', 'α_in cos', 'α_out sin', 'α_out cos', 'special']
    type_counts = []

    # Annahme: 8-Takt fuer Koordinaten
    coord_mask = all_tokens < tokenizer.coord_vocab_size
    coord_tokens = all_tokens[coord_mask]
    positions = np.arange(len(coord_tokens)) % 8

    for i in range(8):
        type_counts.append(np.sum(positions == i))
    type_counts.append(np.sum(~coord_mask))

    colors = plt.cm.Set2(np.linspace(0, 1, 9))
    bars = ax2.bar(type_names, type_counts, color=colors, edgecolor='black')
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Token Type Distribution', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_file = output_dir / 'token_distribution.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Gespeichert: {output_file}")


def plot_sequence_length_distribution(data, tokenizer, output_dir='./figures'):
    """Histogramm der Sequenzlängen."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lengths = []
    for mesh in data:
        tokens = tokenizer.tokenize(mesh)
        lengths.append(len(tokens))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(lengths, bins=30, color='teal', edgecolor='black', alpha=0.7)
    ax.axvline(x=np.mean(lengths), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(lengths):.1f}')
    ax.axvline(x=np.median(lengths), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(lengths):.1f}')
    ax.set_xlabel('Sequence Length (tokens)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Sequence Length Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / 'sequence_lengths.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Gespeichert: {output_file}")


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/root/repos/meshtron')

    from train_domain import TrainerDomain
    from tokenizer_domain import DomainTokenizer

    # Daten laden
    data = torch.load('/root/repos/meshtron/domain_data.pt', weights_only=False)

    # Tokenizer
    tok = DomainTokenizer(quantization_r=64, quantization_a=32,
                         sorting_strategy=0, embedding_mode=0, verbose=False)

    print("=== Plotting Token Distribution ===")
    plot_token_distribution(tok, data)

    print("\n=== Plotting Sequence Lengths ===")
    plot_sequence_length_distribution(data, tok)

    print("\n=== Plotting Training Loss (if checkpoint exists) ===")
    try:
        trainer = TrainerDomain(
            quantization_r=32, quantization_a=16,
            sorting_strategy=0, embedding_mode=0,
            d_model=128, n_latents=128,
            batch_size=4, num_epochs=10,
            learning_rate=1e-3,
            stage_layers=(2, 4, 6, 8, 10),
            n_heads=4,
            verbose=False,
        )
        trainer.load_checkpoint('best_model.pt')
        plot_loss_history(trainer)
    except Exception as e:
        print(f"Could not load checkpoint: {e}")

    print("\nAlle Training-Plots erstellt!")
