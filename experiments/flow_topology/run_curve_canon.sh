#!/bin/bash
# Curve head on the canonical-topology set, full 300 epochs (overfit allowed; use _last.pt).
set -u
M=/work/meshtron_v2; PY=/opt/venv-sci/bin/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/opt/stack/domain_partition_3D/experimentell/hex3d_algohex
(cd /opt/stack/meshtron && OMP_NUM_THREADS=4 taskset -c 8-11 $PY -m meshtron.training.train_polytron --stage curve \
   --data $M/data/curve_canon.pt --out $M/runs/curve_canon --epochs 300 --bs 16 --lr 3e-4 --d 256 --heads 8 \
   --layers 6 --dropout 0.1 --n-points 2048 --eval-every 10 --weight-label 7) > $M/runs_curve_canon.log 2>&1
tail -1 $M/runs_curve_canon.log; touch $M/CURVE_CANON_DONE
