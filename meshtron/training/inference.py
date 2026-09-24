"""Standalone inference / generation for a trained Quadtron run.

Usage:
    python inference.py --run-dir runs/<config-hash>
    python inference.py --run-dir runs/<config-hash> --ckpt last.pt --temperature 0.9
    python inference.py --run-dir runs/<config-hash> --meta-mesh ./meta_mesh.pt

The script rebuilds the model from the checkpoint metadata (no need to
re-tokenize the training dataset), samples sequences for each mesh in the
meta-mesh file, and plots ground-truth vs. generated quads.
"""

import argparse
import json
from pathlib import Path
from typing import Tuple

import torch
from torch_geometric.data import Data

from meshtron.viz import plotting_tools
from meshtron.training.config import PipelineConfig
from meshtron.data.dataset import MeshData
from meshtron.model.quadtron import Quadtron
from meshtron.training.policy import Policy
from meshtron.data.tokenizer_v2 import Tokenizer2D


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quadtron inference / generation.")
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Run directory, e.g. runs/<config-hash>.")
    p.add_argument("--ckpt", type=str, default="best.pt",
                   help="Checkpoint filename inside the run dir.")
    p.add_argument("--meta-mesh", type=Path, default=Path("./meta_mesh.pt"),
                   help="Path to a meta-mesh .pt file with nodes_T3/T4 + faces_T3/T4.")
    p.add_argument("--n-sample-points", type=int, default=None,
                   help="Override n_sample_points; default = config value.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--prefix-tokens", type=int, default=8,
                   help="Number of ground-truth tokens used to seed generation.")
    p.add_argument("--max-extra-tokens", type=int, default=32,
                   help="Extra tokens beyond `face_count * tokens_per_face`.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory for plots. Default: <run-dir>/figures/")
    p.add_argument("--device", type=str, default=None, help="cuda or cpu (auto by default).")
    return p.parse_args()


def load_run(run_dir: Path, ckpt_name: str, device: torch.device) -> Tuple[Quadtron, Policy, PipelineConfig, dict, Tokenizer2D]:
    ckpt_path = run_dir / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = PipelineConfig.from_dict(ckpt["config"])

    tokenizer = Tokenizer2D(
        quantization_levels=cfg.quantization,
        verbose=False,
        sorting_strategy=cfg.sorting_strategy,
        dim=cfg.dim,
    )

    model = Quadtron(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg.d_model,
        max_seq_length=ckpt["max_seq_length"],
        n_latents=cfg.n_latents,
        input_dim=cfg.dim,
        min_face_count=ckpt["min_face_count"],
        max_face_count=ckpt["max_face_count"],
        n_heads=cfg.n_heads,
        stage_layers=tuple(cfg.stage_layers),
        dropout=cfg.dropout,
        ffn_mult=cfg.ffn_mult,
        use_flash_attention=cfg.use_flash_attention,
        sliding_window_size=cfg.sliding_window_size,
        verbose=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    policy = Policy(model).to(device)
    return model, policy, cfg, ckpt, tokenizer


def load_meta_meshes(path: Path) -> list[Data]:
    meta = torch.load(path, weights_only=False)
    return [
        Data(x=meta.nodes_T3.clone(), faces=meta.faces_T3.clone(),
             tri_coordinates=meta.tri_coordinates.clone()),
        Data(x=meta.nodes_T4.clone(), faces=meta.faces_T4.clone(),
             tri_coordinates=meta.tri_coordinates.clone()),
    ]


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model, policy, cfg, ckpt, tokenizer = load_run(args.run_dir, args.ckpt, device)

    out_dir = args.out_dir or (args.run_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    n_sample_points = args.n_sample_points or cfg.n_sample_points
    meshes = load_meta_meshes(args.meta_mesh)

    # max_length_padding is needed by MeshData / tokenizer; we use the value
    # baked into the checkpoint so token indices stay aligned with training.
    tokenizer.max_length_padding = ckpt["max_seq_length"]
    test_data = MeshData(
        meshes, tokenizer,
        n_sample_points=n_sample_points,
        verbose=False,
        dim=cfg.dim,
    )

    summary = {
        "run_dir": str(args.run_dir),
        "ckpt": args.ckpt,
        "best_val_bpt": float(ckpt.get("best_val_bpt", float("nan"))),
        "epoch": int(ckpt.get("epoch", -1)),
        "device": str(device),
        "results": [],
    }

    for i, mesh in enumerate(meshes):
        face_count = test_data.face_count[i]
        point_cloud = test_data.get_point_cloud(mesh, n_sample_points).unsqueeze(0).to(device)
        face_count_tensor = torch.tensor([face_count], device=device)

        print(f"\n=== Mesh {i}: {face_count} faces ===")

        true_tokens = tokenizer.tokenize(mesh.x[:, 0:cfg.dim], mesh.faces, getattr(mesh, 'dir_class', None))
        vertices_true, quads_true = tokenizer.detokenize(true_tokens)
        true_path = out_dir / f"true_mesh_{i}_faces{face_count}.png"
        plotting_tools.plt_mesh(vertices_true, quads_true, output_file=str(true_path))

        start_tokens = torch.tensor(true_tokens[: args.prefix_tokens],
                                    dtype=torch.long, device=device).unsqueeze(0)
        max_length = face_count * tokenizer.tokens_per_face + args.max_extra_tokens

        generated = policy.sample(
            point_cloud=point_cloud,
            face_count=face_count_tensor,
            start_tokens=start_tokens,
            max_length=max_length,
            temperature=args.temperature,
            eos_token=tokenizer.end_token,
        ).squeeze(0).tolist()

        vertices_gen, quads_gen = tokenizer.detokenize(generated)
        n_gen_faces = (
            quads_gen.shape[1] if quads_gen.dim() > 1 and quads_gen.shape[1] > 0 else 0
        )
        gen_path = out_dir / f"generated_mesh_{i}_faces{face_count}.png"
        point_cloud_viz = (point_cloud[0].cpu() + 1) / 2  # [-1,1] -> [0,1]
        plotting_tools.plt_mesh(
            vertices_gen, quads_gen,
            point_cloud=point_cloud_viz,
            output_file=str(gen_path),
        )

        print(f"  Generated: {n_gen_faces} faces (target {face_count})")
        print(f"  Wrote: {true_path}")
        print(f"  Wrote: {gen_path}")

        summary["results"].append({
            "mesh": i,
            "target_faces": int(face_count),
            "generated_faces": int(n_gen_faces),
            "true_plot": str(true_path),
            "generated_plot": str(gen_path),
        })

    (out_dir / "inference_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print(f"\nSummary: {out_dir / 'inference_summary.json'}")


if __name__ == "__main__":
    main()
