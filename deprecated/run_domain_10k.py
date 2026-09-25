"""
run_domain_10k.py

DEPRECATED: part of the MeshtronDomain family (failed experiment, superseded by
Polytron). Kept for reference only, not maintained or extended.

Launcher fuer die Phase-0-Baseline: MeshtronDomain auf dem vollen 10k-Datensatz
(domain_data_10k.pt) neu trainieren. Nutzt DomainTrainer + DomainTrainingConfig.

Beispiel (RTX Pro Blackwell 48 GB, kein Queueing):
    python run_domain_10k.py --batch-size 48 --num-epochs 60 --d-model 512

Cluster (kurze Jobs): kleiner --num-epochs + spaeter resume ueber Checkpoint.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from dataclasses import replace

from meshtron.training.config import DomainTrainingConfig
from domain_trainer import DomainTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="./domain_data_10k.pt")
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--num-epochs", type=int, default=60)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-latents", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n-sample-points", type=int, default=768)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--stage-layers", type=int, nargs="+", default=[2, 4, 6, 8, 10])
    ap.add_argument("--sorting-strategy", type=int, default=0,
                    help="0=row-structured (keine Kompr.), 1=row-compressed, 2=vertex-first")
    ap.add_argument("--no-point-labels", action="store_true",
                    help="Punktwolken-Label-Konditionierung AUS (alter Zustand, zum Vergleich)")
    ap.add_argument("--log-dir", default="runs_domain")
    args = ap.parse_args()

    cfg = DomainTrainingConfig(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        d_model=args.d_model,
        n_latents=args.n_latents,
        n_heads=args.n_heads,
        stage_layers=tuple(args.stage_layers),
        learning_rate=args.lr,
        n_sample_points=args.n_sample_points,
        sorting_strategy=args.sorting_strategy,
        point_cloud_labels=not args.no_point_labels,
        log_dir=args.log_dir,
        save_best=True,
    )
    print(f"sorting_strategy={cfg.sorting_strategy}  point_cloud_labels={cfg.point_cloud_labels}")
    print("Config-Hash:", cfg.hash())
    trainer = DomainTrainer(cfg)
    print("vocab:", trainer.tokenizer.vocab_size,
          "max_seq_len:", trainer.max_length,
          "faces:", trainer.min_face_count, "..", trainer.max_face_count)
    result = trainer.run()
    print("DONE:", result)


if __name__ == "__main__":
    main()
