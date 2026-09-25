"""
inference_domain.py

DEPRECATED: part of the MeshtronDomain family (failed experiment, superseded by
Polytron). Kept for reference only, not maintained or extended.

Inference / Generation für Domain-Partition Meshtron.
Autoregressive Generierung + Hermite-Spline Rekonstruktion + Transfinite Interpolation.
"""


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os
from pathlib import Path
from typing import Tuple

_HERE = Path(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

from meshtron.viz import plotting_tools
from meshtron.training.config import DomainTrainingConfig
from dataset_domain import DomainMeshData
from meshtron_domain import MeshtronDomain
from meshtron.training.policy import Policy
from meshtron.legacy.reconstruct_domain import reconstruct_domain
from tokenizer_domain import DomainTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Domain-Partition Meshtron inference.")
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Run directory, e.g. runs_domain/<config-hash>.")
    p.add_argument("--ckpt", type=str, default="best.pt",
                   help="Checkpoint filename inside the run dir.")
    p.add_argument("--data-path", type=Path,
                   default=_HERE / "domain_data.pt",
                   help="Path to domain_data.pt")
    p.add_argument("--n-sample-points", type=int, default=None,
                   help="Override n_sample_points")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-length", type=int, default=1000,
                   help="Max generation length")
    p.add_argument("--transfinite-divisions", type=int, default=5,
                   help="Transfinite interpolation divisions")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory for plots")
    p.add_argument("--device", type=str, default=None,
                   help="cuda or cpu (auto by default)")
    return p.parse_args()


def load_run(run_dir: Path, ckpt_name: str, device: torch.device) -> Tuple[MeshtronDomain, Policy, DomainTrainingConfig, dict, DomainTokenizer]:
    ckpt_path = run_dir / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = DomainTrainingConfig.from_dict(ckpt["config"])

    tokenizer = DomainTokenizer(
        quantization_r=cfg.quantization_r,
        quantization_a=cfg.quantization_a,
        sorting_strategy=cfg.sorting_strategy,
        embedding_mode=cfg.embedding_mode,
        verbose=False,
    )

    model = MeshtronDomain(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg.d_model,
        max_seq_length=ckpt["max_seq_length"],
        n_latents=cfg.n_latents,
        input_dim=2,
        min_face_count=ckpt["min_face_count"],
        max_face_count=ckpt["max_face_count"],
        n_heads=cfg.n_heads,
        stage_layers=tuple(cfg.stage_layers),
        embedding_mode=cfg.embedding_mode,
        quantization_r=cfg.quantization_r,
        quantization_a=cfg.quantization_a,
        verbose=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    policy = Policy(model)
    return model, policy, cfg, ckpt, tokenizer


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model, policy, cfg, ckpt, tokenizer = load_run(args.run_dir, args.ckpt, device)

    out_dir = args.out_dir or (args.run_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    n_sample_points = args.n_sample_points or cfg.n_sample_points
    meshes = torch.load(args.data_path, weights_only=False)

    tokenizer.max_length_padding = ckpt["max_seq_length"]
    test_data = DomainMeshData(
        meshes, tokenizer,
        n_sample_points=n_sample_points,
        with_labels=getattr(cfg, 'point_cloud_labels', False),
        verbose=False,
    )

    # Ground Truth für Vergleich
    original_meshes = torch.load(str(_HERE / "checkpoint_mesh_100.pt"), weights_only=False)

    summary = {
        "run_dir": str(args.run_dir),
        "ckpt": args.ckpt,
        "temperature": args.temperature,
        "n_meshes": len(meshes),
    }

    for idx in range(min(len(test_data), 20)):  # max 20 samples
        batch = test_data[idx]
        point_cloud = batch["point_cloud"].unsqueeze(0).to(device)
        face_count = torch.tensor([batch["face_count"]], dtype=torch.long, device=device)

        # GT Tokens
        gt_tokens = test_data.data[idx]

        # Generierung
        start = torch.full((1, 8), tokenizer.start_token, dtype=torch.long, device=device)
        with torch.no_grad():
            generated = policy.sample(
                point_cloud=point_cloud,
                face_count=face_count,
                start_tokens=start,
                max_length=args.max_length,
                temperature=args.temperature,
                eos_token=tokenizer.end_token,
            )
        gen_tokens = generated[0].cpu().tolist()

        # Rekonstruktion
        mesh_data = meshes[idx]
        center = mesh_data['center'].numpy()

        try:
            output_gt = tokenizer.detokenize(gt_tokens)
            quad_gt = reconstruct_domain(output_gt, center, transfinite_divisions=args.transfinite_divisions)
            gt_nodes = quad_gt.x.numpy()
            gt_faces = quad_gt.faces.numpy()
        except Exception as e:
            print(f"  [{idx}] GT reconstruction failed: {e}")
            gt_nodes, gt_faces = None, None

        try:
            output_gen = tokenizer.detokenize(gen_tokens)
            quad_gen = reconstruct_domain(output_gen, center, transfinite_divisions=args.transfinite_divisions)
            gen_nodes = quad_gen.x.numpy()
            gen_faces = quad_gen.faces.numpy()
        except Exception as e:
            print(f"  [{idx}] Gen reconstruction failed: {e}")
            gen_nodes, gen_faces = None, None

        # Plot
        fig, axes = plotting_tools.plt.subplots(1, 2, figsize=(12, 5))

        if gt_nodes is not None and gt_faces is not None:
            ax = axes[0]
            for fi in range(gt_faces.shape[1]):
                face = gt_faces[:, fi]
                coords = gt_nodes[face]
                ax.fill(coords[:, 0], coords[:, 1], color='lightblue', alpha=0.5, edgecolor='blue', linewidth=0.3)
            ax.set_title("Ground Truth", fontweight='bold')
            ax.set_aspect('equal')

        if gen_nodes is not None and gen_faces is not None:
            ax = axes[1]
            for fi in range(gen_faces.shape[1]):
                face = gen_faces[:, fi]
                coords = gen_nodes[face]
                ax.fill(coords[:, 0], coords[:, 1], color='lightcoral', alpha=0.5, edgecolor='red', linewidth=0.3)
            ax.set_title(f"Generated (T={args.temperature})", fontweight='bold')
            ax.set_aspect('equal')

        fig.suptitle(f"Mesh {idx} | {batch['face_count']} Faces | Gen: {len(gen_tokens)} tokens", fontweight='bold')
        plotting_tools.plt.tight_layout()
        plotting_tools.plt.savefig(out_dir / f"mesh_{idx:03d}.png", dpi=200, bbox_inches='tight')
        plotting_tools.plt.close()

        summary[f"mesh_{idx}"] = {
            "n_tokens_gt": len(gt_tokens),
            "n_tokens_gen": len(gen_tokens),
            "gen_success": gen_nodes is not None,
        }

    # Summary speichern
    (out_dir / "inference_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nInference complete. Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
