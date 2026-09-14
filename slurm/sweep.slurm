#!/bin/bash
#SBATCH --job-name=meshtron-sweep
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --array=0-29%4
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# ----------------------------------------------------------------------------
# Heterogeneous cluster note:
#   The job runs on any GPU node by default. To pin trials to specific GPUs
#   use either of these (cluster-dependent; pick what your site supports):
#
#   #SBATCH --constraint="rtx3090|rtx_pro5000"
#   #SBATCH --partition=gpu                  # or two separate submits per part.
#
# You can also submit two arrays in parallel against the same Optuna storage,
# one per GPU type, e.g.:
#   STAGE=stage1 sbatch --constraint=rtx3090       sweep.slurm
#   STAGE=stage1 sbatch --constraint=rtx_pro5000   sweep.slurm
# ----------------------------------------------------------------------------

set -euo pipefail

mkdir -p slurm_logs

# --- Environment (adjust to your site) --------------------------------------
source /mnt/scratch/trentschler/environments/python/env_courier/bin/activate
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# --- Sweep parameters (override via `--export=ALL,STAGE=stage2,...` or env) -
STAGE=${STAGE:-stage1}
STORAGE=${STORAGE:-"sqlite:///${SLURM_SUBMIT_DIR:-$PWD}/sweep_${STAGE}.db?timeout=60"}
STUDY=${STUDY:-"meshtron-${STAGE}"}
TRIALS_PER_TASK=${TRIALS_PER_TASK:-1}
TRIAL_EPOCHS=${TRIAL_EPOCHS:-15}
PRUNER=${PRUNER:-median}
PATIENCE=${PATIENCE:-5}
LOG_DIR=${LOG_DIR:-"${SLURM_SUBMIT_DIR:-$PWD}/runs"}
BASE_CONFIG=${BASE_CONFIG:-}

# Optuna sampler seed: derive from array index so each task explores differently
# while remaining reproducible.
SAMPLER_SEED=${SAMPLER_SEED:-$SLURM_ARRAY_TASK_ID}

echo "[$(date)] host=$(hostname) node=$SLURMD_NODENAME array=$SLURM_ARRAY_TASK_ID gpu=${CUDA_VISIBLE_DEVICES:-?}"
echo "         stage=$STAGE study=$STUDY storage=$STORAGE trials=$TRIALS_PER_TASK"

CMD=(python sweep.py
  --stage "$STAGE"
  --storage "$STORAGE"
  --study-name "$STUDY"
  --n-trials "$TRIALS_PER_TASK"
  --trial-epochs "$TRIAL_EPOCHS"
  --early-stopping-patience "$PATIENCE"
  --pruner "$PRUNER"
  --log-dir "$LOG_DIR"
  --sampler-seed "$SAMPLER_SEED")

if [[ -n "$BASE_CONFIG" ]]; then
  CMD+=(--base-config "$BASE_CONFIG")
fi

srun "${CMD[@]}"
