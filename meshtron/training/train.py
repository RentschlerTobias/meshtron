import argparse
import json
from pathlib import Path

from meshtron.training.config import PipelineConfig
from meshtron.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Meshtron training entry point.")
    p.add_argument("--config", type=Path, default=None,
                   help="Optional path to a JSON config; CLI flags override its fields.")
    # Pipeline-Auswahl
    p.add_argument("--model-family", type=str, choices=["quadtron", "polytron"],
                   help="Which architecture to train.")
    p.add_argument("--dim", type=int, choices=[2, 3])
    # Daten
    p.add_argument("--data-path", type=str)
    p.add_argument("--quantization", type=int)
    p.add_argument("--sorting-strategy", type=int)
    p.add_argument("--repr-mode", type=str, choices=["hermite", "bezier", "cubic_bezier"],
                   help="Polytron Stage-3 edge geometry representation.")
    p.add_argument("--corners-per-block", type=int,
                   help="Polytron pointer count per face: 4 (2D quad) or 8 (3D hex block).")
    # Modell
    p.add_argument("--d-model", type=int)
    p.add_argument("--n-heads", type=int)
    p.add_argument("--stage-layers", type=int, nargs="+")
    p.add_argument("--n-latents", type=int)
    p.add_argument("--dropout", type=float)
    p.add_argument("--ffn-mult", type=int)
    # Attention-Backend
    p.add_argument("--use-flash-attention", action="store_true")
    p.add_argument("--sliding-window-size", type=int,
                   help="0 = full attention (default), >0 = window size in tokens.")
    # Optimierung
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--weight-decay", type=float)
    p.add_argument("--warmup-steps", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--accumulation-steps", type=int)
    p.add_argument("--num-epochs", type=int)
    p.add_argument("--early-stopping-patience", type=int)
    # Laufzeit
    p.add_argument("--seed", type=int)
    p.add_argument("--precision", type=str, choices=["fp32", "bf16", "fp16"])
    p.add_argument("--log-dir", type=str)
    p.add_argument("--save-best", action="store_true")
    p.add_argument("--save-last", action="store_true")
    # Reinforcement Learning (Teil B)
    p.add_argument("--rl-enabled", action="store_true",
                   help="Swap teacher-forcing for the RL curriculum objective (Quadtron only "
                        "for now, see docs/decisions/2026-09-13-meshtron-refactor-plan.md).")
    p.add_argument("--advantage-estimator", type=str, choices=["group_relative", "critic"])
    p.add_argument("--rl-rollouts-per-condition", type=int)
    p.add_argument("--rl-curriculum-stage", type=str, choices=["vertex", "face", "row", "mesh"])
    p.add_argument("--rl-temperature", type=float)
    p.add_argument("--rl-max-length", type=int)
    p.add_argument("--init-checkpoint", type=str,
                   help="Pretrained checkpoint to fine-tune with RL from (required in practice "
                        "for --rl-enabled -- a random policy gets ~0 reward, see decision log).")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    base = (
        PipelineConfig.from_dict(json.loads(Path(args.config).read_text()))
        if args.config
        else PipelineConfig()
    )
    overrides = {
        k: v for k, v in vars(args).items()
        if k != "config" and v is not None and v is not False
    }
    if "stage_layers" in overrides:
        overrides["stage_layers"] = tuple(overrides["stage_layers"])
    merged = {**base.to_dict(), **overrides}
    return PipelineConfig.from_dict(merged)


def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    if cfg.model_family == "quadtron":
        result = Trainer(cfg).run()
        print(json.dumps(result.__dict__, indent=2, default=str))
    elif cfg.model_family == "polytron":
        from polytron_chain import run_polytron
        run_polytron(cfg)
    else:
        raise ValueError(f"Unknown model_family={cfg.model_family!r} (expected 'quadtron' or 'polytron')")


if __name__ == "__main__":
    main()
