#!/bin/bash
# Best-case study on one topology (2026-10-06, reports/blockgen_v2_canonical_best_case.md):
# overfit vertex (1500 epochs) + conn (last checkpoint, also evaluated on train), learning
# curve of the vertex task, curve head in the generator's canonical order, chain on train + val.
#   BLOCKGEN_SAMPLES=<relabel>/out BLOCKGEN_WORK=<dir> PYTHONPATH=<dp3d hex3d_algohex> \
#     best_case.sh <v2 canon set>.pt
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); PY=${PY:-python}; CORES=${CORES:-0-7}
W=${BLOCKGEN_WORK:-$ROOT/runs/blockgen_v2}; DATA=$1; mkdir -p "$W"
export PYTHONDONTWRITEBYTECODE=1
log() { echo "[$(date +%F' '%T)] $*"; }
train() {  # name task [extra]
  local n=$1 t=$2; shift 2
  log "train $n"
  (cd "$ROOT" && OMP_NUM_THREADS=4 taskset -c $CORES $PY -m meshtron.training.train_blockgen --task $t \
     --data "$DATA" --out "$W/$n" --d 128 --layers 2 --heads 4 --bs 16 "$@") > "$W/$n.log" 2>&1
  grep RESULT "$W/$n.log" || log "FAILED $n"
}
train of_vertex_S vertex --n-latent 32 --epochs 1500 --eval-every 50 --patience 1000 --select last --eval-train 34
train of_conn_S   conn   --epochs 200 --eval-every 10 --patience 1000 --select last --eval-train 34
for f in 0.25 0.5 1.0; do
  train lc_vertex_S_$f vertex --n-latent 32 --epochs 400 --eval-every 5 --patience 12 --train-frac $f
done
log "curve dataset (canonical order) + curve head"
$PY "$ROOT/scripts/blockgen_v2/build_curve_dataset.py" --split-from "$DATA" --out "$W/curve_canon_order.pt" --canonical
(cd "$ROOT" && OMP_NUM_THREADS=4 taskset -c $CORES $PY -m meshtron.training.train_polytron --stage curve \
   --data "$W/curve_canon_order.pt" --out "$W/curve_order" --epochs 300 --bs 16 --lr 3e-4 --d 256 --heads 8 \
   --layers 6 --dropout 0.1 --n-points 2048 --eval-every 10 --weight-label 7) > "$W/curve_order.log" 2>&1
args="--data $DATA --vertex $W/of_vertex_S/last.pt --conn $W/of_conn_S/last.pt --curve $W/curve_order_last.pt"
for s in train val; do
  log "chain $s"
  OMP_NUM_THREADS=2 taskset -c $CORES $PY "$ROOT/scripts/blockgen_v2/infer_chain_v2.py" --split $s --n 34 $args \
     --out-dir "$W/chain_$s" > "$W/chain_$s.log" 2>&1
done
log "best case done"
