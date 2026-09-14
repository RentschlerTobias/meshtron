"""Optuna sweep driver for Meshtron training.

Two-stage workflow:
    stage1  ->  optimise lr, warmup, dropout, weight_decay, n_latents
    stage2  ->  optimise architecture (d_model, depth, n_heads)

Each trial builds a PipelineConfig (inheriting any fields not in the search
space from --base-config or the dataclass defaults), runs the Trainer, and
returns best_val_bpt. The objective is minimised.

Trials are pruned via Optuna pruners (Median by default) using per-epoch
validation bits-per-token reported through the Trainer's on_epoch callback.

Multiple processes can share a study by pointing at the same storage URL.
"""

import argparse
import json
from pathlib import Path
from typing import Callable

import optuna
from optuna.pruners import HyperbandPruner, MedianPruner, NopPruner
from optuna.samplers import TPESampler

from config import PipelineConfig
from metrics import EpochMetrics
from trainer import Trainer


# ---------------------------------------------------------------- objectives

def _make_on_epoch(trial: optuna.Trial) -> Callable:
    def on_epoch(epoch: int, _train: EpochMetrics, val: EpochMetrics | None) -> None:
        if val is None:
            return
        trial.report(val.bits_per_token, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return on_epoch


def _run_trial(trial: optuna.Trial, cfg: PipelineConfig) -> float:
    trainer = Trainer(cfg)
    trial.set_user_attr("config_hash", cfg.hash())
    trial.set_user_attr("run_dir", str(trainer.logger.run_dir))
    trial.set_user_attr("max_seq_length", int(trainer.max_length))
    trial.set_user_attr("vocab_size", int(trainer.tokenizer.vocab_size))
    result = trainer.run(on_epoch=_make_on_epoch(trial))
    return result.best_val_bpt


def stage1_objective(trial: optuna.Trial, base: PipelineConfig) -> float:
    """Regularisation + optimisation search space."""
    overrides = dict(
        learning_rate=trial.suggest_float(
            "learning_rate", 1e-5, 5e-4, log=True),
        warmup_steps=trial.suggest_int("warmup_steps", 0, 2000, step=100),
        dropout=trial.suggest_float("dropout", 0.0, 0.3),
        weight_decay=trial.suggest_float("weight_decay", 0.0, 0.1),
        n_latents=trial.suggest_categorical(
            "n_latents", [8, 16, 32, 64]),
    )
    cfg = PipelineConfig.from_dict({**base.to_dict(), **overrides})
    return _run_trial(trial, cfg)


_LAYER_OPTIONS = [2, 4, 6]


def stage2_objective(trial: optuna.Trial, base: PipelineConfig) -> float:
    """Architecture search. Number of stages and layers per stage are independent."""
    d_model = trial.suggest_categorical("d_model", [256, 384, 512, 768])
    n_heads = max(1, d_model // 64)

    n_stages = trial.suggest_int("n_stages", 2, 5)

    layers = [
        trial.suggest_categorical(f"layer_{i}", _LAYER_OPTIONS)
        for i in range(n_stages)
    ]

    overrides = dict(
        d_model=int(d_model),
        n_heads=int(n_heads),
        stage_layers=tuple(int(l) for l in layers),
    )
    cfg = PipelineConfig.from_dict({**base.to_dict(), **overrides})
    return _run_trial(trial, cfg)


def stage3_objective(trial: optuna.Trial, base: PipelineConfig) -> float:
    """Fine-tune: learning rate, warmup, batch size, regularization."""
    overrides = dict(
        learning_rate=trial.suggest_float(
            "learning_rate", 1e-4, 5e-4, log=True),
        warmup_steps=trial.suggest_int("warmup_steps", 0, 1000, step=100),
        weight_decay=trial.suggest_float(
            "weight_decay", 0.001, 0.05, log=True),
        dropout=trial.suggest_float("dropout", 0.0, 0.2),
        batch_size=trial.suggest_categorical("batch_size", [8, 16, 32]),
        accumulation_steps=trial.suggest_categorical(
            "accumulation_steps", [1, 2, 4]),
    )
    cfg = PipelineConfig.from_dict({**base.to_dict(), **overrides})
    return _run_trial(trial, cfg)


def sorting_objective(trial: optuna.Trial, base: PipelineConfig) -> float:
    """Compare full vs row-compressed encoding on otherwise identical configs."""
    overrides = dict(
        sorting_strategy=trial.suggest_categorical("sorting_strategy", [1, 2]),
    )
    cfg = PipelineConfig.from_dict({**base.to_dict(), **overrides})
    return _run_trial(trial, cfg)


def all_stages_objective(trial: optuna.Trial, base: PipelineConfig) -> float:
    """Optimize all parameters from stage1, stage2 and sorting simultaneously."""
    d_model = trial.suggest_categorical("d_model", [256, 384, 512])
    n_heads = max(1, d_model // 64)

    n_stages = trial.suggest_int("n_stages", 3, 5)
    layers = [
        trial.suggest_categorical(f"layer_{i}", _LAYER_OPTIONS)
        for i in range(n_stages)
    ]

    overrides = dict(
        d_model=int(d_model),
        n_heads=int(n_heads),
        stage_layers=tuple(int(l) for l in layers),
        learning_rate=trial.suggest_float(
            "learning_rate", 1e-5, 5e-4, log=True),
        warmup_steps=trial.suggest_int("warmup_steps", 0, 2000, step=100),
        dropout=trial.suggest_float("dropout", 0.0, 0.3),
        weight_decay=trial.suggest_float("weight_decay", 0.0, 0.1),
        n_latents=trial.suggest_categorical(
            "n_latents", [8, 16, 32, 64]),
        sorting_strategy=trial.suggest_categorical("sorting_strategy", [1, 2]),
    )
    cfg = PipelineConfig.from_dict({**base.to_dict(), **overrides})
    return _run_trial(trial, cfg)


OBJECTIVES = {"stage1": stage1_objective,
              "stage2": stage2_objective,
              "sorting": sorting_objective,
              "all": all_stages_objective}


# ------------------------------------------------------------------ driver

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna sweep over PipelineConfig.")
    p.add_argument("--stage", choices=list(OBJECTIVES), default="stage1",
                   help="stage1: lr/warmup/dropout/weight_decay/n_latents, "
                        "stage2: d_model/n_heads/n_stages/stage_layers, "
                        "sorting: sorting_strategy, "
                        "all: all parameters from stage1+stage2+sorting")
    p.add_argument("--storage", type=str, default="sqlite:///sweep.db",
                   help="Optuna storage URL (sqlite:///path.db or postgresql://...).")
    p.add_argument("--study-name", type=str, default=None,
                   help="Optuna study name. Default: 'meshtron-<stage>'.")
    p.add_argument("--n-trials", type=int, default=1,
                   help="Trials this process will run before exiting.")
    p.add_argument("--trial-epochs", type=int, default=15,
                   help="num_epochs per trial; overrides config.")
    p.add_argument("--early-stopping-patience", type=int, default=5,
                   help="Per-trial early stopping patience.")
    p.add_argument("--base-config", type=Path, default=None,
                   help="Optional JSON PipelineConfig used as the base; "
                        "fields outside the search space are inherited.")
    p.add_argument("--log-dir", type=str, default="runs",
                   help="Parent dir for per-trial Trainer run directories.")
    p.add_argument(
        "--pruner", choices=["median", "hyperband", "none"], default="median")
    p.add_argument("--sampler-seed", type=int, default=0,
                   help="Seed for Optuna's TPE sampler (sampling reproducibility).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    base = (
        PipelineConfig.from_dict(json.loads(args.base_config.read_text()))
        if args.base_config else PipelineConfig()
    )

    study_name = args.study_name or f"meshtron-{args.stage}"
    log_root = Path(args.log_dir) / study_name

    # Forced overrides shared by every trial
    base = PipelineConfig.from_dict({
        **base.to_dict(),
        "num_epochs": args.trial_epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "log_dir": str(log_root),
        "save_best": False,
        "save_last": False,
    })

    pruner = {
        "median": MedianPruner(n_warmup_steps=2),
        "hyperband": HyperbandPruner(min_resource=2, max_resource=args.trial_epochs),
        "none": NopPruner(),
    }[args.pruner]

    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        direction="minimize",
        sampler=TPESampler(seed=args.sampler_seed),
        pruner=pruner,
        load_if_exists=True,
    )

    # Persist the base config alongside the study so analysis tools find it.
    log_root.mkdir(parents=True, exist_ok=True)
    (log_root / "base_config.json").write_text(json.dumps(base.to_dict(), indent=2, default=str))

    objective = OBJECTIVES[args.stage]
    study.optimize(
        lambda t: objective(t, base),
        n_trials=args.n_trials,
        gc_after_trial=True,
        catch=(RuntimeError,),  # OOM etc. -> mark trial failed, keep sweeping
    )

    summary = {
        "study_name": study_name,
        "n_trials_total": len(study.trials),
        "best_value": study.best_value if study.best_trial else None,
        "best_params": study.best_params if study.best_trial else None,
        "best_trial_number": study.best_trial.number if study.best_trial else None,
    }
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
