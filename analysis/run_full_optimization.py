"""Full optimization pipeline for Meshtron sorting method comparison.

Workflow:
    1. For each sorting method (0, 1, 3):
       a. Stage A: Optimize architecture (d_model, n_heads, n_stages, stage_layers, n_latents)
       b. Stage B: Optimize hyperparameters (lr, warmup, dropout, weight_decay)
       c. Stage C: Optimize training dynamics (batch_size, accumulation_steps, lr)
    2. Final comparison: Train best configs with 3 seeds each
    3. Generate reports and plots

Usage:
    python run_full_optimization.py --sandbox /sandbox
"""


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from meshtron.training.config import PipelineConfig
from meshtron.training.metrics import EpochMetrics
from meshtron.training.trainer import Trainer


def convert_optuna_params_to_config(params: dict) -> dict:
    """Convert Optuna's stage_a params (n_stages, layer_{i}) to PipelineConfig stage_layers.

    Optuna returns n_stages and layer_0, layer_1, ... as separate parameters.
    PipelineConfig uses a single stage_layers tuple where len(tuple) = n_stages.
    This function converts from the Optuna format to the PipelineConfig format.
    """
    if "n_stages" not in params:
        return params

    n_stages = params.pop("n_stages")
    layers = []
    for i in range(n_stages):
        key = f"layer_{i}"
        if key in params:
            layers.append(int(params.pop(key)))

    params["stage_layers"] = tuple(layers)
    return params


def is_stage_complete(study_name: str, storage_path: Path, n_trials: int) -> tuple[bool, dict | None]:
    """Check if a study has completed all desired trials.

    Returns:
        (is_complete, best_params) - if complete, best_params contains the best trial params
    """
    db_path = storage_path / \
        f"sweep_{study_name.split('-')[1]}_{study_name.split('-')[-1]}.db"
    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=f"sqlite:///{db_path}"
        )
        completed = len([t for t in study.trials if t.state ==
                        optuna.trial.TrialState.COMPLETE])
        if completed >= n_trials and study.best_trial:
            return True, study.best_params
    except Exception as e:
        print(f"  Warning: Could not load study {study_name}: {e}")
    return False, None


def load_best_config_from_dbs(sorting_strategy: int, storage_path: Path) -> PipelineConfig | None:
    """Reconstruct best PipelineConfig from all three stage databases.

    This is useful when a run crashed and we need to recover the best config
    from the existing Optuna databases.

    Returns:
        PipelineConfig with params from all completed stages, or None if any stage is missing.
    """
    base = PipelineConfig()
    best_params_per_stage = {}

    for stage in STAGES:
        study_name = f"meshtron-s{sorting_strategy}-stage-{stage}"
        db_path = storage_path / f"sweep_s{sorting_strategy}_{stage}.db"
        try:
            study = optuna.load_study(
                study_name=study_name,
                storage=f"sqlite:///{db_path}"
            )
            if study.best_trial:
                best_params_per_stage[stage] = dict(study.best_params)
            else:
                return None
        except Exception:
            return None

    if len(best_params_per_stage) != 3:
        return None

    stage_a_params = convert_optuna_params_to_config(
        dict(best_params_per_stage["a"]))
    stage_b_params = dict(best_params_per_stage["b"])
    stage_c_params = dict(best_params_per_stage["c"])

    config_dict = {
        **base.to_dict(),
        **stage_a_params,
        **stage_b_params,
        **stage_c_params,
        "sorting_strategy": sorting_strategy,
    }

    return PipelineConfig.from_dict(config_dict)


SORTING_METHODS = [0, 1, 3]
STAGES = ["a", "b", "c"]

STAGE_CONFIG = {
    "a": {
        "trials": 25,
        "epochs": 15,
        "patience": 5,
        "description": "Architecture",
    },
    "b": {
        "trials": 25,
        "epochs": 15,
        "patience": 5,
        "description": "Hyperparameters",
    },
    "c": {
        "trials": 20,
        "epochs": 25,
        "patience": 7,
        "description": "Training Dynamics",
    },
}


def _make_on_epoch(trial: optuna.Trial):
    def on_epoch(epoch: int, _train: EpochMetrics, val: EpochMetrics | None) -> None:
        if val is None:
            return
        trial.report(val.bits_per_token, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return on_epoch


def _run_trial(trial: optuna.Trial, cfg: PipelineConfig, log_dir: Path) -> float:
    cfg_dict = {**cfg.to_dict(), "log_dir": str(log_dir),
                "save_best": False, "save_last": False}
    trial_cfg = PipelineConfig.from_dict(cfg_dict)
    trainer = Trainer(trial_cfg)
    trial.set_user_attr("config_hash", trial_cfg.hash())
    trial.set_user_attr("run_dir", str(trainer.logger.run_dir))
    trial.set_user_attr("max_seq_length", int(trainer.max_length))
    trial.set_user_attr("vocab_size", int(trainer.tokenizer.vocab_size))
    result = trainer.run(on_epoch=_make_on_epoch(trial))
    return result.best_val_bpt


def stage_a_objective(trial: optuna.Trial, base: PipelineConfig, log_dir: Path) -> float:
    d_model = trial.suggest_categorical("d_model", [128, 256, 384, 512])
    n_heads = max(1, d_model // 64)
    n_stages = trial.suggest_int("n_stages", 3, 5)
    layers = [trial.suggest_categorical(
        f"layer_{i}", [2, 4, 6]) for i in range(n_stages)]
    n_latents = trial.suggest_categorical("n_latents", [4, 8, 16, 32, 64])

    overrides = dict(
        d_model=int(d_model),
        n_heads=int(n_heads),
        stage_layers=tuple(int(l) for l in layers),
        n_latents=int(n_latents),
    )
    cfg = PipelineConfig.from_dict({**base.to_dict(), **overrides})
    return _run_trial(trial, cfg, log_dir)


def stage_b_objective(trial: optuna.Trial, base: PipelineConfig, log_dir: Path) -> float:
    overrides = dict(
        learning_rate=trial.suggest_float(
            "learning_rate", 1e-5, 1e-3, log=True),
        warmup_steps=trial.suggest_int("warmup_steps", 0, 2000, step=100),
        dropout=trial.suggest_float("dropout", 0.0, 0.3),
        weight_decay=trial.suggest_float("weight_decay", 0.0, 0.1),
    )
    cfg = PipelineConfig.from_dict({**base.to_dict(), **overrides})
    return _run_trial(trial, cfg, log_dir)


def stage_c_objective(trial: optuna.Trial, base: PipelineConfig, log_dir: Path) -> float:
    overrides = dict(
        batch_size=trial.suggest_categorical(
            "batch_size", [8, 16, 24]),
        accumulation_steps=trial.suggest_categorical(
            "accumulation_steps", [1, 2, 4]),
        learning_rate=trial.suggest_float(
            "learning_rate", 1e-5, 5e-4, log=True),
    )
    cfg = PipelineConfig.from_dict({**base.to_dict(), **overrides})
    return _run_trial(trial, cfg, log_dir)


STAGE_OBJECTIVES = {
    "a": stage_a_objective,
    "b": stage_b_objective,
    "c": stage_c_objective,
}


def run_stage_optimization(
    sorting_strategy: int,
    stage: str,
    base_config: PipelineConfig,
    storage_path: Path,
    study_name: str,
    n_trials: int,
    n_epochs: int,
    patience: int,
    parallel: int = 2,
) -> dict:
    log_dir = storage_path / f"logs_{study_name}"
    log_dir.mkdir(parents=True, exist_ok=True)

    base = PipelineConfig.from_dict({
        **base_config.to_dict(),
        "sorting_strategy": sorting_strategy,
        "num_epochs": n_epochs,
        "early_stopping_patience": patience,
        "log_dir": str(log_dir),
        "save_best": False,
        "save_last": False,
    })

    pruner = MedianPruner(n_warmup_steps=2)
    sampler = TPESampler(seed=0)

    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{storage_path /
                             f'sweep_s{sorting_strategy}_{stage}.db'}",
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    (log_dir / "base_config.json").write_text(
        json.dumps(base.to_dict(), indent=2, default=str)
    )

    def objective_fn(t): return STAGE_OBJECTIVES[stage](t, base, log_dir)

    print(f"\n{'='*60}")
    print(f"Stage {stage.upper()} | Sorting {
          sorting_strategy} | {n_trials} trials")
    print(f"Study: {study_name}")
    print(f"{'='*60}\n")

    start_time = time.time()

    study.optimize(
        objective_fn,
        n_trials=n_trials,
        gc_after_trial=True,
        catch=(RuntimeError,),
    )

    elapsed = time.time() - start_time

    summary = {
        "study_name": study_name,
        "sorting_strategy": sorting_strategy,
        "stage": stage,
        "n_trials_total": len(study.trials),
        "n_trials_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "best_value": study.best_value if study.best_trial else None,
        "best_params": study.best_params if study.best_trial else None,
        "best_trial_number": study.best_trial.number if study.best_trial else None,
        "elapsed_seconds": elapsed,
    }

    print(f"\nStage {stage.upper()} complete for sorting {sorting_strategy}")
    print(f"Best value: {summary['best_value']:.4f}")
    print(f"Elapsed: {elapsed:.1f}s")

    return summary


def train_with_seeds(
    config: PipelineConfig,
    seeds: list[int],
    output_dir: Path,
) -> list[dict]:
    results = []
    for seed in seeds:
        cfg_dict = {**config.to_dict(), "seed": seed,
                    "save_best": True, "save_last": True}
        cfg = PipelineConfig.from_dict(cfg_dict)

        run_dir = output_dir / f"seed_{seed}"
        trainer = Trainer(cfg)
        result = trainer.run()

        results.append({
            "seed": seed,
            "best_val_bpt": result.best_val_bpt,
            "best_val_perplexity": result.best_val_perplexity,
            "best_epoch": result.best_epoch,
            "epochs_run": result.epochs_run,
            "config_hash": cfg.hash(),
            "run_dir": str(trainer.logger.run_dir),
        })
        print(f"  Seed {seed}: val_bpt={result.best_val_bpt:.4f}")
    return results


def run_final_comparison(
    best_configs: dict[int, PipelineConfig],
    storage_path: Path,
    n_seeds: int = 3,
) -> dict:
    print(f"\n{'='*60}")
    print(f"Final Comparison | {n_seeds} seeds per sorting method")
    print(f"{'='*60}\n")

    seeds = list(range(n_seeds))
    all_results = {}

    for sorting_strategy, config in best_configs.items():
        print(f"\nSorting Strategy {sorting_strategy}:")
        output_dir = storage_path / "final_models" / f"s{sorting_strategy}"
        output_dir.mkdir(parents=True, exist_ok=True)

        results = train_with_seeds(config, seeds, output_dir)
        all_results[sorting_strategy] = results

        avg_bpt = sum(r["best_val_bpt"] for r in results) / len(results)
        print(f"  Average val_bpt: {avg_bpt:.4f}")

    comparison = {
        "n_seeds": n_seeds,
        "results": {},
    }

    for sorting_strategy, results in all_results.items():
        bpts = [r["best_val_bpt"] for r in results]
        perplexities = [r["best_val_perplexity"] for r in results]

        comparison["results"][sorting_strategy] = {
            "mean_val_bpt": sum(bpts) / len(bpts),
            "std_val_bpt": (sum((b - sum(bpts)/len(bpts))**2 for b in bpts) / len(bpts)) ** 0.5,
            "mean_val_perplexity": sum(perplexities) / len(perplexities),
            "individual_runs": results,
        }

    return comparison


def generate_reports(storage_path: Path, summaries: list[dict], final_comparison: dict):
    reports_dir = storage_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    stage_reports = {}
    for s in summaries:
        key = f"s{s['sorting_strategy']}_stage_{s['stage']}"
        stage_reports[key] = {
            "best_value": s["best_value"],
            "best_params": s["best_params"],
            "n_trials": s["n_trials_total"],
            "elapsed_seconds": s["elapsed_seconds"],
        }

    report = {
        "optimization_summaries": stage_reports,
        "final_comparison": final_comparison,
    }

    report_path = reports_dir / "full_optimization_report.json"
    with open(report_path, "w") as f:
        json.dump(report, indent=2, default=str, fp=f)

    print(f"\nReport saved to: {report_path}")

    print("\n" + "="*60)
    print("FINAL COMPARISON SUMMARY")
    print("="*60)
    for sorting_strategy, data in final_comparison["results"].items():
        print(f"\nStrategy {sorting_strategy}:")
        print(f"  Mean val_bpt: {data['mean_val_bpt']:.4f} ± {
              data['std_val_bpt']:.4f}")
        print(f"  Mean perplexity: {data['mean_val_perplexity']:.4f}")

    return report


def get_sandbox_dir() -> Path:
    sandbox = Path("/sandbox")
    try:
        (sandbox / ".write_test").touch()
        (sandbox / ".write_test").unlink()
        return sandbox
    except (OSError, PermissionError):
        return Path.cwd()


def main():
    parser = argparse.ArgumentParser(
        description="Full optimization pipeline for Meshtron")
    parser.add_argument("--sorting-methods", type=int, nargs="+", default=SORTING_METHODS,
                        help="Sorting methods to optimize")
    parser.add_argument("--parallel-trials", type=int, default=2,
                        help="Number of parallel trials")
    parser.add_argument("--n-seeds", type=int, default=3,
                        help="Number of seeds for final comparison")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume from specific sorting:stage, e.g. '0:c' or '1:b'")
    parser.add_argument("--recover-config", type=str, default=None,
                        help="Recover best config from DB for a sorting method, e.g. '0' or '1'")
    args = parser.parse_args()

    sandbox = get_sandbox_dir()
    sandbox.mkdir(parents=True, exist_ok=True)
    (sandbox / "best_configs").mkdir(parents=True, exist_ok=True)
    (sandbox / "reports").mkdir(parents=True, exist_ok=True)

    print(f"Sandbox: {sandbox}")
    print(f"Sorting methods: {args.sorting_methods}")
    print(f"Parallel trials: {args.parallel_trials}")

    base_config = PipelineConfig()
    best_configs = {}

    all_summaries = []

    if args.recover_config is not None:
        sorting = int(args.recover_config)
        print(f"\nRecovering best config from DBs for sorting {sorting}...")
        recovered = load_best_config_from_dbs(sorting, sandbox)
        if recovered:
            out_path = sandbox / "best_configs" / f"s{sorting}_best.json"
            out_path.write_text(json.dumps(
                recovered.to_dict(), indent=2, default=str))
            print(f"Recovered config saved to: {out_path}")
            print(f"Config: {recovered}")
        else:
            print("Failed to recover config - missing or incomplete stages in DB")
        return

    resume_sorting = None
    resume_stage = None
    if args.resume_from:
        parts = args.resume_from.split(":")
        if len(parts) == 2:
            resume_sorting = int(parts[0])
            resume_stage = parts[1]
            print(f"\nResuming from sorting {
                  resume_sorting}, stage {resume_stage}")

    for sorting_strategy in args.sorting_methods:
        if resume_sorting is not None and sorting_strategy < resume_sorting:
            print(f"\n\n{'#'*60}")
            print(f"# SKIPPING SORTING STRATEGY {
                  sorting_strategy} (before resume point)")
            print(f"{'#'*60}")
            continue

        if resume_sorting is not None and sorting_strategy == resume_sorting:
            recovered = load_best_config_from_dbs(sorting_strategy, sandbox)
            if recovered:
                print(f"\n  Recovered config from DB for sorting {
                      sorting_strategy}")
                best_configs[sorting_strategy] = recovered
            else:
                print(f"\n  Warning: Could not recover full config from DB for sorting {
                      sorting_strategy}")
                print(f"  Will try to load stage-by-stage from DB...")

        print(f"\n\n{'#'*60}")
        print(f"# OPTIMIZING SORTING STRATEGY {sorting_strategy}")
        print(f"{'#'*60}")

        stage_results = {}

        for stage in STAGES:
            if resume_stage is not None and sorting_strategy == resume_sorting:
                stage_order = {"a": 0, "b": 1, "c": 2}
                if stage_order.get(stage, 0) < stage_order.get(resume_stage, 0):
                    print(f"  Stage {stage.upper()
                                     } before resume point, skipping...")
                    continue

            cfg = STAGE_CONFIG[stage]

            study_name = f"meshtron-s{sorting_strategy}-stage-{stage}"

            is_complete, existing_best_params = is_stage_complete(
                study_name, sandbox, cfg["trials"]
            )

            if is_complete and existing_best_params:
                print(f"\n  Stage {stage.upper()
                                   } already complete, skipping...")
                best_params = existing_best_params
                summary = {
                    "study_name": study_name,
                    "sorting_strategy": sorting_strategy,
                    "stage": stage,
                    "n_trials_total": cfg["trials"],
                    "n_trials_completed": cfg["trials"],
                    "best_value": None,
                    "best_params": best_params,
                    "best_trial_number": None,
                    "elapsed_seconds": 0,
                    "skipped": True,
                }
            else:
                summary = run_stage_optimization(
                    sorting_strategy=sorting_strategy,
                    stage=stage,
                    base_config=base_config if stage == "a" else best_configs[sorting_strategy],
                    storage_path=sandbox,
                    study_name=study_name,
                    n_trials=cfg["trials"],
                    n_epochs=cfg["epochs"],
                    patience=cfg["patience"],
                    parallel=args.parallel_trials,
                )
                best_params = summary["best_params"]

            all_summaries.append(summary)
            stage_results[stage] = summary

            if best_params:
                converted_params = convert_optuna_params_to_config(
                    dict(best_params))
                if stage == "a":
                    best_configs[sorting_strategy] = PipelineConfig.from_dict({
                        **base_config.to_dict(),
                        **converted_params,
                        "sorting_strategy": sorting_strategy,
                    })
                else:
                    current = best_configs[sorting_strategy]
                    best_configs[sorting_strategy] = PipelineConfig.from_dict({
                        **current.to_dict(),
                        **converted_params,
                    })

            val_bpt = summary.get("best_value") or "N/A"
            print(f"  Stage {stage.upper()} best value: {val_bpt}")

        (sandbox / "best_configs" / f"s{sorting_strategy}_best.json").write_text(
            json.dumps(
                best_configs[sorting_strategy].to_dict(), indent=2, default=str)
        )

    print(f"\n\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")

    final_comparison = run_final_comparison(
        best_configs=best_configs,
        storage_path=sandbox,
        n_seeds=args.n_seeds,
    )

    report = generate_reports(sandbox, all_summaries, final_comparison)

    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Results and models saved in: {sandbox}")


if __name__ == "__main__":
    main()
