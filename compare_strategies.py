"""Compare tokenization strategies 0, 1, and 3.

Runs a grid of (seed x sorting_strategy) with all other hyper-parameters
fixed to the best configuration from Optuna sweep.
Metrics captured:
  - best_val_bpt (primary - compression quality)
  - bits_per_mesh = bpt * val_tokens / n_val_meshes
  - peak GPU memory and wall-clock time

Each run appends one row to summary.jsonl + summary.csv.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path

import torch

from config import PipelineConfig
from trainer import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-config", type=Path,
                   default=Path("best_config.json"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="Seeds to sweep (default: 0 1 2).")
    p.add_argument("--strategies", type=int, nargs="+", default=[0, 1, 2, 3],
                   choices=[0, 1, 2, 3],
                   help="Strategies to compare (default: 0 1 2 3).")
    p.add_argument("--log-dir", type=str,
                   default="runs/meshtron-strategy-compare",
                   help="Parent dir for run directories.")
    p.add_argument("--summary-path", type=Path,
                   default=None,
                   help="Override summary output path.")
    p.add_argument("--epochs", type=int, default=30,
                   help="Number of epochs per run.")
    p.add_argument("--early-stopping", type=int, default=15,
                   help="Early stopping patience.")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override batch_size from base config.")
    return p.parse_args()


def run_one(base: PipelineConfig, seed: int, strategy: int,
            log_dir: str, epochs: int, early_stopping: int,
            batch_size: int | None = None) -> dict:
    overrides: dict = {
        **base.to_dict(),
        "seed": seed,
        "sorting_strategy": strategy,
        "num_epochs": epochs,
        "early_stopping_patience": early_stopping,
        "log_dir": log_dir,
        "save_best": False,
        "save_last": False,
    }
    if batch_size is not None:
        overrides["batch_size"] = batch_size
    cfg = PipelineConfig.from_dict(overrides)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    trainer = Trainer(cfg)
    n_val = len(trainer.val_loader.dataset)
    n_train = len(trainer.train_loader.dataset)
    max_len = int(trainer.max_length)
    tokens_per_face = int(trainer.tokenizer.tokens_per_face)
    vocab_size = int(trainer.tokenizer.vocab_size)

    t0 = time.perf_counter()
    result = trainer.run()
    elapsed = time.perf_counter() - t0

    val_tokens = int(trainer.last_val.n_tokens) if trainer.last_val else 0
    train_tokens = int(
        trainer.last_train.n_tokens) if trainer.last_train else 0
    bits_per_mesh = (
        result.best_val_bpt * val_tokens / n_val if n_val > 0 else float("nan")
    )

    peak_mem_bytes = (
        torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    )

    return {
        "seed": seed,
        "sorting_strategy": strategy,
        "config_hash": result.config_hash,
        "run_dir": result.run_dir,
        "best_val_bpt": result.best_val_bpt,
        "best_val_nll": result.best_val_nll,
        "best_val_perplexity": result.best_val_perplexity,
        "best_epoch": result.best_epoch,
        "epochs_run": result.epochs_run,
        "final_train_bpt": result.final_train_bpt,
        "final_val_bpt": result.final_val_bpt,
        "train_tokens_per_epoch": train_tokens,
        "val_tokens_per_epoch": val_tokens,
        "n_train_meshes": n_train,
        "n_val_meshes": n_val,
        "bits_per_mesh": bits_per_mesh,
        "max_seq_length": max_len,
        "tokens_per_face": tokens_per_face,
        "vocab_size": vocab_size,
        "peak_gpu_mem_bytes": int(peak_mem_bytes),
        "peak_gpu_mem_mib": peak_mem_bytes / (1024 ** 2),
        "wall_time_s": elapsed,
    }


def _append_row(summary_path: Path, row: dict) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")
    csv_path = summary_path.with_suffix(".csv")
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def aggregate_results(summary_path: Path) -> dict:
    """Compute mean/std per strategy across seeds."""
    rows = []
    with open(summary_path) as f:
        for line in f:
            rows.append(json.loads(line))

    strategies = sorted(set(r["sorting_strategy"] for r in rows))
    agg = {}
    for s in strategies:
        s_rows = [r for r in rows if r["sorting_strategy"] == s]
        metrics = ["best_val_bpt", "bits_per_mesh", "peak_gpu_mem_mib",
                   "wall_time_s", "max_seq_length", "tokens_per_face"]
        agg[s] = {"n": len(s_rows)}
        for m in metrics:
            vals = [r[m] for r in s_rows]
            agg[s][f"{m}_mean"] = sum(vals) / len(vals)
            agg[s][f"{m}_std"] = (
                (sum((v - sum(vals)/len(vals))**2 for v in vals) / len(vals)) ** 0.5
                if len(vals) > 1 else 0.0
            )
    return agg


def plot_comparison(agg: dict, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    strategies = sorted(agg.keys())
    metrics = ["best_val_bpt", "bits_per_mesh",
               "peak_gpu_mem_mib", "wall_time_s"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        means = [agg[s][f"{metric}_mean"] for s in strategies]
        stds = [agg[s][f"{metric}_std"] for s in strategies]
        labels = [f"S{s}" for s in strategies]
        colors = ["#1f77b4", "#2ca02c", "#ff7f0e"][:len(strategies)]

        bars = ax.bar(labels, means, yerr=stds, capsize=5,
                      color=colors, edgecolor="black", alpha=0.8)
        ax.set_ylabel(metric)
        ax.set_title(metric.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.3)

        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stds)*0.05,
                    f"{mean:.4f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle(
        "Strategy Comparison: S0 (Lexicographic) vs S1 (Directed) vs S3 (Compressed+CW)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


def main() -> None:
    args = parse_args()
    base = PipelineConfig.from_dict(json.loads(args.base_config.read_text()))

    summary_path = args.summary_path or Path(args.log_dir) / "summary.jsonl"
    grid = [(s, k) for k in args.strategies for s in args.seeds]
    print(f"Running {len(grid)} cells: {grid}")

    for i, (seed, strategy) in enumerate(grid, start=1):
        print(f"\n=== [{i}/{len(grid)}] seed={seed} strategy={strategy} ===")
        row = run_one(
            base=base,
            seed=seed,
            strategy=strategy,
            log_dir=args.log_dir,
            epochs=args.epochs,
            early_stopping=args.early_stopping,
            batch_size=args.batch_size,
        )
        _append_row(summary_path, row)
        print(json.dumps(row, indent=2, default=str))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n=== Aggregation ===")
    agg = aggregate_results(summary_path)
    print(json.dumps(agg, indent=2, default=str))

    plot_path = Path(args.log_dir) / "comparison.png"
    plot_comparison(agg, plot_path)

    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
