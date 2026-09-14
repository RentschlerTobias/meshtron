"""Fair sorting-strategy comparison driver.

Runs a grid of (seed x sorting_strategy) with all other hyper-parameters
fixed to the best configuration from the meshtron-all sweep (trial 126).
For each run we capture metrics that the Optuna sweep does not surface:
  - val_bpt and val_nll (per-token, what the sweep optimised)
  - val_tokens / train_tokens (sequence-length cost)
  - bits_per_mesh = val_bpt * val_tokens / n_val_meshes
    (the fair per-mesh description length)
  - peak GPU memory and wall-clock time per run

Each run appends one row to a JSONL+CSV summary at --summary-path. By
default the script runs the full grid (seeds x strategies) in one
process, but a single cell can be selected via --seed / --strategy.

Examples:
    # full default grid (3 seeds x 2 strategies)
    python compare_sorting.py

    # custom grid
    python compare_sorting.py --seeds 0 1 2 3 4 --strategies 1 2

    # one cell only (used by SLURM array)
    python compare_sorting.py --seed 1 --strategy 2
"""

from __future__ import annotations


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
                   default=Path("compare_sorting_base.json"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="Seeds to sweep (default: 0 1 2). Ignored if --seed is set.")
    p.add_argument("--strategies", type=int, nargs="+", default=[1, 2],
                   choices=[1, 2],
                   help="Strategies to sweep (default: 1 2). Ignored if --strategy is set.")
    p.add_argument("--seed", type=int, default=None,
                   help="Run a single seed (overrides --seeds).")
    p.add_argument("--strategy", type=int, default=None, choices=[1, 2],
                   help="Run a single strategy (overrides --strategies).")
    p.add_argument("--trial-epochs", type=int, default=None,
                   help="Override num_epochs from base config.")
    p.add_argument("--log-dir", type=str, default=None,
                   help="Override log_dir from base config.")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override batch_size from base config (same value for "
                        "every cell in this invocation).")
    p.add_argument("--scenario", type=str, default="fixed_batch",
                   help="Tag stored on every row; used by analyze_sorting.py "
                        "to group results (e.g. 'fixed_batch', 'mem_matched').")
    p.add_argument("--summary-path", type=Path,
                   default=Path("runs/meshtron-sorting-fair/summary.jsonl"),
                   help="Append one JSON line per run here.")
    return p.parse_args()


def run_one(base: PipelineConfig, seed: int, strategy: int,
            trial_epochs: int | None, log_dir: str | None,
            batch_size: int | None, scenario: str) -> dict:
    overrides: dict = {
        **base.to_dict(),
        "seed": seed,
        "sorting_strategy": strategy,
    }
    if trial_epochs is not None:
        overrides["num_epochs"] = trial_epochs
    if log_dir is not None:
        overrides["log_dir"] = log_dir
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
    train_tokens = int(trainer.last_train.n_tokens) if trainer.last_train else 0
    bits_per_mesh = (
        result.best_val_bpt * val_tokens / n_val if n_val > 0 else float("nan")
    )

    peak_mem_bytes = (
        torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    )

    return {
        "scenario": scenario,
        "seed": seed,
        "sorting_strategy": strategy,
        "batch_size": int(cfg.batch_size),
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


def main() -> None:
    args = parse_args()
    base = PipelineConfig.from_dict(json.loads(args.base_config.read_text()))

    seeds = [args.seed] if args.seed is not None else list(args.seeds)
    strategies = [args.strategy] if args.strategy is not None else list(args.strategies)

    grid = [(s, k) for k in strategies for s in seeds]
    print(f"Running {len(grid)} cell(s): {grid}")

    for i, (seed, strategy) in enumerate(grid, start=1):
        print(f"\n=== [{i}/{len(grid)}] seed={seed} strategy={strategy} ===")
        row = run_one(
            base=base,
            seed=seed,
            strategy=strategy,
            trial_epochs=args.trial_epochs,
            log_dir=args.log_dir,
            batch_size=args.batch_size,
            scenario=args.scenario,
        )
        _append_row(args.summary_path, row)
        print(json.dumps(row, indent=2, default=str))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
