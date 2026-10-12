#!/bin/bash
# Curve head on the relabelled data, after run_all2.sh has built data/v2_cart.pt
# (signalled by its "train vertex_M" log line). Cores 8-11 (relabel lanes done by then).
set -u
M=/work/meshtron_v2; PY=/opt/venv-sci/bin/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/opt/stack/domain_partition_3D/experimentell/hex3d_algohex
log() { echo "[$(date +%F' '%T)] $*"; }
log "waiting for the v2 dataset"
until grep -q "train vertex_M" $M/run_all.log 2>/dev/null; do sleep 60; done
log "curve dataset"
(cd $M && taskset -c 8-11 $PY build_curve_dataset.py --split-from data/v2_cart.pt --out data/curve_v2.pt) 2>&1 | tail -1
log "train curve head (weight-label 7)"
(cd /opt/stack/meshtron && OMP_NUM_THREADS=4 taskset -c 8-11 $PY -m meshtron.training.train_polytron --stage curve \
   --data $M/data/curve_v2.pt --out $M/runs/curve_v2 --epochs 300 --bs 16 --lr 3e-4 --d 256 --heads 8 \
   --layers 6 --dropout 0.1 --n-points 2048 --eval-every 10 --weight-label 7) > $M/runs_curve_v2.log 2>&1
tail -1 $M/runs_curve_v2.log
log "CURVE DONE"; touch $M/CURVE_DONE
