#!/bin/bash
# train_planb.slurm  --  bwUniCluster 3.0, dev_gpu_a100_il (A100, 30 min max)
# ---------------------------------------------------------------------------
# Trainiert die 3-Stufen Plan-B Pipeline (S1 vertices -> S2 pointer faces ->
# S3 cubic_bezier HO-geometry) auf dem augmentierten Datensatz
# (domain_data_aug.pt, variable Facecounts 6/24/54/96 = 6n^2). chain_e2e.py
# trainiert die Koepfe, speichert nach --save-dir und rendert eine E2E-Galerie.
#
# bwUniCluster 3.0 GPU-Queues -- Namen/Limits mit `sinfo` verifizieren:
#   sinfo -o "%P %l %G %c %m" | grep -i a100
#   gpu_a100_short   Kurzjob/dev (getestet)  <- diese
#   gpu_a100_il      48 h max   4x A100   (fuer volles S1)
#   (Wiki nennt dev-Queue teils dev_gpu_a100_il -- Live-sinfo entscheidet)
#
# 30-MIN-LIMIT: volles S1 von Grund auf (30k Meshes) passt hier NICHT rein.
#   (A) EMPFOHLEN - S1 wiederverwenden, nur S2+S3 (passt bequem):
#         LOAD_S1=$(ws_find meshtron)/runs_planb/s1_var.pt sbatch train_planb.slurm
#   (B) voll, Epochen runter (best-val Checkpoints retten Teilfortschritt):
#         EP1=8 EP2=8 EP3=8 sbatch train_planb.slurm
#   Fuer VOLLES S1-Training: Partition wechseln ->
#         sbatch --partition=gpu_a100_il --time=08:00:00 train_planb.slurm
#
# Setup (einmalig, auf der LOGIN-Node -- hat Internet):
#   ws_allocate meshtron 60           # falls noch kein Workspace
#   cd $(ws_find meshtron)/meshtron   # Repo im Workspace (grosse Daten dorthin!)
#   export UV_CACHE_DIR=$PWD/.uv-cache  # Cache auf /work (Hardlinks, kein $HOME)
#   uv sync                           # baut .venv aus pyproject.toml + uv.lock
#
# Aufruf:
#   sbatch train_planb.slurm
#   LOAD_S1=runs_planb/s1_var.pt BATCH=128 sbatch train_planb.slurm
#   ENV_PATH=/eigenes/venv sbatch train_planb.slurm   # anderes venv erzwingen
#
# Logs -> slurm_logs/planb_<jobid>.{out,err}
# Ergebnis -> ${SAVE_DIR}/{s1,s2,s3}.pt  +  ${OUT} (Galerie-PNG)
# ---------------------------------------------------------------------------

#SBATCH --job-name=meshtron-planb
#SBATCH --partition=gpu_a100_short
#SBATCH --output=slurm_logs/planb_%j.out
#SBATCH --error=slurm_logs/planb_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=127500mb

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm_logs

# --- Environment: uv-managed .venv (aus pyproject.toml + uv.lock) ------------
# torch cu128 wheels bringen CUDA-Runtime mit -> nur NVIDIA-Treiber noetig (auf
# GPU-Knoten vorhanden), kein cuda-Modul. uv holt auch Python 3.12 selbst.
# EMPFEHLUNG: `uv sync` EINMAL auf der LOGIN-Node laufen lassen (hat Internet;
# Compute-Node evtl. nicht). Der Block unten ist nur Fallback.
ENV_PATH=${ENV_PATH:-$PWD/.venv}
# uv-Cache auf dasselbe FS wie .venv (Workspace/work) -> Hardlinks statt Copy,
# schnell + schont $HOME-Quota. Behebt "Failed to hardlink"-Warnung.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
if [[ ! -f "${ENV_PATH}/bin/activate" ]]; then
  echo "kein venv unter ${ENV_PATH} -> versuche 'uv sync' ..."
  if ! command -v uv >/dev/null 2>&1; then
    echo "FEHLER: 'uv' nicht im PATH. Einmalig installieren:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh    # dann Shell neu"
    echo "Danach auf der LOGIN-Node (Internet!):  cd '$PWD' && uv sync"
    exit 1
  fi
  uv sync                       # erstellt $PWD/.venv aus uv.lock
fi
source "${ENV_PATH}/bin/activate"

# variable Sequenzlaengen + mehrere Modelle -> Fragmentierung vermeiden
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Hyperparameter (per env ueberschreibbar) -------------------------------
DATA=${DATA:-domain_data_aug.pt}
DMODEL=${DMODEL:-256}
BATCH=${BATCH:-96}          # A100 (bis 510GB Knoten-RAM/40GB VRAM) vertraegt viel
EP1=${EP1:-30}              # S1 vertices  (nur relevant ohne LOAD_S1)
EP2=${EP2:-25}              # S2 pointer faces
EP3=${EP3:-25}              # S3 HO-geometry
LR=${LR:-5e-4}              # LR fuer alle 3 Koepfe
SEED=${SEED:-0}
LIMIT=${LIMIT:-}            # nur erste K Meshes -> schnellerer Beispiel-Bau (ex_g/bezier
                           # ist teuer). Fuer 30-min-Test z.B. LIMIT=8000. Leer=alle 31952.
SAVE_DIR=${SAVE_DIR:-runs_planb/e2e_${SLURM_JOB_ID:-local}}
OUT=${OUT:-figures/e2e/e2e_gallery_${SLURM_JOB_ID:-local}.png}
LOAD_S1=${LOAD_S1:-}        # optional: vortrainiertes S1 (vertex_head-Format) -> nur S2+S3

echo "[$(date -Is)] host=$(hostname) node=${SLURMD_NODENAME:-?} part=${SLURM_JOB_PARTITION:-?} gpu=${CUDA_VISIBLE_DEVICES:-?}"
echo "  data=$DATA d=$DMODEL batch=$BATCH ep=[$EP1,$EP2,$EP3] lr=$LR seed=$SEED limit=${LIMIT:-<all>}"
echo "  save-dir=$SAVE_DIR  out=$OUT  load-s1=${LOAD_S1:-<none>}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

CMD=(python -u chain_e2e.py
  --data "$DATA"
  --d-model "$DMODEL"
  --batch "$BATCH"
  --ep1 "$EP1" --ep2 "$EP2" --ep3 "$EP3"
  --lr "$LR"
  --seed "$SEED"
  --save-dir "$SAVE_DIR"
  --out "$OUT")

if [[ -n "$LIMIT" ]]; then
  CMD+=(--limit "$LIMIT")     # Beispiel-Bau auf Subset beschraenken
fi
if [[ -n "$LOAD_S1" ]]; then
  CMD+=(--load-s1 "$LOAD_S1")   # S1-Training ueberspringen, nur S2+S3
fi

echo "+ ${CMD[*]}"
srun "${CMD[@]}"
echo "[$(date -Is)] fertig."
