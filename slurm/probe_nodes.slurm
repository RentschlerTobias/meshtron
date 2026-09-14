#!/bin/bash
# Probe-Job: submitted ein Array, jede Task loggt auf welchem Node sie
# gelandet ist, welche GPU sie zugewiesen bekommen hat, und ein paar
#
# Aufruf-Beispiele:
#   sbatch probe_nodes.slurm
#   sbatch --constraint=rtx_pro5000 probe_nodes.slurm
#   ARRAY_SIZE=16 CONCURRENCY=8 sbatch probe_nodes.slurm
#   sbatch --partition=gpu probe_nodes.slurm
#
# Output landet in slurm_logs/probe_<jobid>_<task>.{out,err}.
# Nach Lauf:  cat slurm_logs/probe_*.out  oder  ./probe_nodes.slurm summary

#SBATCH --job-name=gpu-node-test
#SBATCH --output=slurm_logs/probe_%A_%a.out
#SBATCH --error=slurm_logs/probe_%A_%a.err
#SBATCH --array=0-7%8
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

# --- Optional: lokale "summary" Subkommando ohne sbatch -----------------
if [[ "${1:-}" == "summary" ]]; then
  echo "=== Cluster overview =================================="
  sinfo -o "%P %.6D %.10c %.8m %N %f %G" || true
  echo
  echo "=== Recent probe results (slurm_logs/probe_*.out) ====="
  for f in slurm_logs/probe_*.out; do
    [[ -e "$f" ]] || {
      echo "(keine Probe-Logs gefunden)"
      break
    }
    echo "--- $f"
    grep -E '^(HOST|GPU|CPU|MEM|FEATURE|PARTITION)' "$f" || true
  done
  exit 0
fi

set -euo pipefail
mkdir -p slurm_logs

# Optional override fuer Array-Groesse / Concurrency via Resubmit:
#   ARRAY_SIZE=16 CONCURRENCY=8 sbatch probe_nodes.slurm
# (SLURM erlaubt --array nicht aus dem Skript heraus zu aendern, daher per CLI:
#   sbatch --array=0-$((ARRAY_SIZE-1))%$CONCURRENCY probe_nodes.slurm )

echo "============================================================"
echo "HOST       : $(hostname)"
echo "NODENAME   : ${SLURMD_NODENAME:-?}"
echo "JOBID      : ${SLURM_JOB_ID:-?}  ARRAY=${SLURM_ARRAY_JOB_ID:-?}_${SLURM_ARRAY_TASK_ID:-?}"
echo "PARTITION  : ${SLURM_JOB_PARTITION:-?}"
echo "GPU IDS    : ${CUDA_VISIBLE_DEVICES:-?}  (SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-?})"
echo "DATE       : $(date -Is)"
echo "------------------------------------------------------------"

echo "FEATURES   :"
scontrol show node "${SLURMD_NODENAME:-$(hostname)}" |
  grep -E 'AvailableFeatures|ActiveFeatures|Gres=|CPUTot|RealMemory|CfgTRES' ||
  true

echo "------------------------------------------------------------"
echo "CPU info   :"
echo "  cores avail to job : $(nproc)"
echo "  model              : $(awk -F: '/model name/ {print $2; exit}' /proc/cpuinfo | sed 's/^ //')"

echo "MEM info   :"
awk '/MemTotal|MemAvailable/ {printf "  %-14s %s %s\n", $1, $2, $3}' /proc/meminfo

echo "------------------------------------------------------------"
echo "GPU (nvidia-smi):"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version,compute_cap \
    --format=csv,noheader
  echo
  echo "GPU pid snapshot:"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
else
  echo "  nvidia-smi nicht im PATH"
fi

echo "------------------------------------------------------------"
echo "PyTorch CUDA check:"
python - <<'PY' 2>&1 || echo "  (kein python / kein torch verfuegbar)"
import os
try:
    import torch
    print(f"  torch              : {torch.__version__}")
    print(f"  cuda available     : {torch.cuda.is_available()}")
    print(f"  device count       : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  cuda:{i}  {p.name}  cc={p.major}.{p.minor}  "
              f"mem={p.total_memory/1e9:.1f} GB  sms={p.multi_processor_count}")
    print(f"  CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
except Exception as e:
    print(f"  torch import failed: {e}")
PY

echo "============================================================"
