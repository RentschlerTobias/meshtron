#!/bin/bash
# train_s1.slurm  --  bwUniCluster 3.0, gpu_a100_short (A100, 30 min max)
# ---------------------------------------------------------------------------
# NUR Stufe 1 (Vertices) auf dem vollen augmentierten Datensatz. Im Gegensatz
# zu chain_e2e.py speichert vertex_head_prototype.py best-val JEDE Epoche
# (--save) -> ein 30-min-Kill der dev-Queue rettet trotzdem das beste Modell.
# Damit laesst sich das eigentliche Ziel (bessere High-Facecount-Genauigkeit)
# auch in der Kurzjob-Queue verfolgen.
#
# Hinweis: KEIN Resume. Ein Job = so viele Epochen wie in 30 min passen.
#   -> nach 1. Lauf Epochenzeit im Log ansehen und EPOCHS so setzen, dass der
#      Kosinus-LR sauber ausannealt (sonst best-val zwar gespeichert, aber LR
#      nicht ganz runter). Danach ggf. auf gpu_a100_il (48h) fuer den Vollrun.
#
# Setup (einmalig, LOGIN-Node -- hat Internet):
#   cd $(ws_find meshtron)/meshtron
#   export UV_CACHE_DIR=$PWD/.uv-cache && uv sync
#
# Aufruf:
#   sbatch train_s1.slurm
#   EPOCHS=20 BATCH=128 sbatch train_s1.slurm
#   sbatch --partition=gpu_a100_il --time=06:00:00 train_s1.slurm   # Vollrun
#
# Logs -> slurm_logs/s1_<jobid>.{out,err}
# Ergebnis -> ${SAVE} (bestes S1, vertex_head-Format; von chain_e2e --load-s1 nutzbar)
# ---------------------------------------------------------------------------

#SBATCH --job-name=meshtron-s1
#SBATCH --partition=gpu_a100_short
#SBATCH --output=slurm_logs/s1_%j.out
#SBATCH --error=slurm_logs/s1_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=127500mb

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm_logs

# --- Environment: uv-managed .venv (siehe train_planb.slurm) -----------------
ENV_PATH=${ENV_PATH:-$PWD/.venv}
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
if [[ ! -f "${ENV_PATH}/bin/activate" ]]; then
  echo "kein venv unter ${ENV_PATH} -> versuche 'uv sync' ..."
  command -v uv >/dev/null 2>&1 || { echo "FEHLER: 'uv' fehlt (siehe train_planb.slurm)"; exit 1; }
  uv sync
fi
source "${ENV_PATH}/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Hyperparameter (per env ueberschreibbar) -------------------------------
DATA=${DATA:-domain_data_aug.pt}
DMODEL=${DMODEL:-256}
BATCH=${BATCH:-256}         # A100 40GB: batch96 -> ~11GB peak; 256 ~28GB, Luft bleibt
EPOCHS=${EPOCHS:-30}
LR=${LR:-8e-4}             # ~sqrt-Skalierung von 5e-4@96 auf batch 256
SEED=${SEED:-0}
INIT=${INIT:-}             # Warmstart: weiter von vorherigem S1-ckpt (Gewichte). Bei
                           # Fortsetzung LR kleiner sinnvoll, z.B. LR=3e-4 EPOCHS=10.
SAVE=${SAVE:-runs_planb/s1_aug_${SLURM_JOB_ID:-local}.pt}   # bestehendes s1_var.pt NICHT ueberschreiben

echo "[$(date -Is)] host=$(hostname) node=${SLURMD_NODENAME:-?} part=${SLURM_JOB_PARTITION:-?} gpu=${CUDA_VISIBLE_DEVICES:-?}"
echo "  data=$DATA d=$DMODEL batch=$BATCH epochs=$EPOCHS lr=$LR seed=$SEED save=$SAVE init=${INIT:-<none>}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

CMD=(python -u vertex_head_prototype.py
  --data "$DATA" --d-model "$DMODEL" --batch "$BATCH"
  --epochs "$EPOCHS" --lr "$LR" --seed "$SEED" --save "$SAVE")
if [[ -n "$INIT" ]]; then
  CMD+=(--init "$INIT")       # Warmstart -> weitertrainieren
fi
echo "+ ${CMD[*]}"
srun "${CMD[@]}"

echo "[$(date -Is)] fertig -> $SAVE"
