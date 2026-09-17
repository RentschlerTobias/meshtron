#!/usr/bin/env bash
# 3D pipeline smoke test for meshtron.
# Run from the repo root:
#   bash scripts/3d_pipeline_smoke.sh
#
# What it does:
#   1. Ensures the uv environment is present (uv sync).
#   2. Converts the single smoke sample from data/hex3d_algohex/smoke2000/
#      into the two transformer dataset formats in a temporary directory.
#   3. Repeats the single smoke sample 6x so Polytron's hardcoded 10% val
#      split yields a non-empty training set (5 train / 1 val).
#   4. Runs a tiny Quadtron 3D training smoke.
#   5. Runs a tiny Polytron 3D training smoke end-to-end (S1/S2/S3 + chain).
#
# Requirements: uv, NVIDIA driver, ~8 GB GPU VRAM (the smoke uses <<1 GB).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

TMPDIR="/tmp/opencode/meshtron-smoke"
mkdir -p "$TMPDIR"

echo "== 1. uv sync =="
uv sync

echo ""
echo "== 2. Convert smoke sample =="
# domain_extractor_3d.py expects a parent directory and globs */sample.npz.
# The smoke sample lives at data/hex3d_algohex/smoke2000/sample.npz.
uv run python domain_extractor_3d.py \
    --src data/hex3d_algohex/ \
    --out-quadtron "$TMPDIR/quadtron_data_3d.pt" \
    --out-polytron "$TMPDIR/polytron_data_3d.pt" \
    --max-tri-points 768

echo ""
echo "== 3. Build 6x repeated smoke set for non-empty Polytron train split =="
uv run python - <<PY
import torch
q = torch.load("$TMPDIR/quadtron_data_3d.pt", weights_only=False)
p = torch.load("$TMPDIR/polytron_data_3d.pt", weights_only=False)
n = 6
q6 = q * n
p6 = p * n
torch.save(q6, "$TMPDIR/quadtron_data_3d_6x.pt")
torch.save(p6, "$TMPDIR/polytron_data_3d_6x.pt")
print(f"6x sets: quadtron={len(q6)}, polytron={len(p6)}")
PY

echo ""
echo "== 4. Quadtron 3D smoke (d_model=64, 1 epoch, bf16) =="
uv run python - <<PY
import sys, time, torch
from train import main as train_main

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
sys.argv = [
    "train.py",
    "--model-family", "quadtron",
    "--dim", "3",
    "--data-path", "$TMPDIR/quadtron_data_3d_6x.pt",
    "--d-model", "64",
    "--n-heads", "2",
    "--stage-layers", "2", "2", "2",
    "--n-latents", "16",
    "--batch-size", "1",
    "--num-epochs", "1",
    "--precision", "bf16",
    "--quantization", "256",
    "--log-dir", "$TMPDIR/runs",
    "--save-last",
]
train_main()
elapsed = time.time() - t0
peak = torch.cuda.max_memory_allocated() / 1024 / 1024
print(f"\n[QUADTRON_SMOKE] elapsed={elapsed:.2f}s peakVRAM={peak:.1f} MiB")
PY

echo ""
echo "== 5. Polytron 3D smoke (d_model=64, 1 epoch, bf16, hex blocks) =="
uv run python - <<PY
import sys, time, torch
from train import main as train_main

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
sys.argv = [
    "train.py",
    "--model-family", "polytron",
    "--dim", "3",
    "--data-path", "$TMPDIR/polytron_data_3d_6x.pt",
    "--d-model", "64",
    "--batch-size", "1",
    "--num-epochs", "1",
    "--precision", "bf16",
    "--corners-per-block", "8",
    "--repr-mode", "cubic_bezier",
    "--log-dir", "$TMPDIR/runs",
]
train_main()
elapsed = time.time() - t0
peak = torch.cuda.max_memory_allocated() / 1024 / 1024
print(f"\n[POLYTRON_SMOKE] elapsed={elapsed:.2f}s peakVRAM={peak:.1f} MiB")
PY

echo ""
echo "== Smoke complete =="
echo "Temporary artifacts are in $TMPDIR"
