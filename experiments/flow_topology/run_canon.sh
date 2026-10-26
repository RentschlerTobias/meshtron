#!/bin/bash
# Canonical-topology runs (2026-10-06): overfit "best case" models (last epoch, also
# evaluated on train) + learning curve on the vertex task. Marker CANON_DONE.
set -u
M=/opt/stack/meshtron/experiments/flow_topology; PY=/opt/venv-sci/bin/python
export PYTHONDONTWRITEBYTECODE=1
log() { echo "[$(date +%F' '%T)] $*"; }
run() {  # name task d layers [extra]
  local n=$1 t=$2 d=$3 l=$4; shift 4
  log "train $n"
  (cd $M && OMP_NUM_THREADS=4 taskset -c 0-7 $PY train.py --task $t --data data/v2_canon.pt --out runs/$n \
     --d $d --layers $l --eval-every 10 --bs 16 "$@") > $M/runs_$n.log 2>&1
  grep RESULT $M/runs_$n.log || log "FAILED $n (see runs_$n.log)"
}
rm -f $M/CANON_DONE
# overfit / best case: full 400 epochs, last checkpoint, eval on val + 34 train items
run of_vertex_S  vertex  128 2 --heads 4 --n-latent 32 --epochs 400 --patience 1000 --select last --eval-train 34
run of_conn_S    conn    128 2 --heads 4 --epochs 200 --patience 1000 --select last --eval-train 34
run of_hexarow_S hexarow 128 2 --heads 4 --n-latent 32 --epochs 400 --patience 1000 --select last --eval-train 34
# learning curve: best-val checkpoint, val error vs train size
for f in 0.25 0.5 1.0; do
  run lc_vertex_S_$f vertex 128 2 --heads 4 --n-latent 32 --epochs 400 --patience 12 --eval-every 5 --train-frac $f
done
log "CANON DONE"; touch $M/CANON_DONE
