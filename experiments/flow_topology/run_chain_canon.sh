#!/bin/bash
# Chain evaluation of the overfit canonical models on train (best case) and val.
# Waits on files (never on process patterns). Marker CHAIN_DONE.
set -u
M=/work/meshtron_v2; PY=/opt/venv-sci/bin/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/opt/stack/domain_partition_3D/experimentell/hex3d_algohex OMP_NUM_THREADS=2
until [ -f $M/CURVE_CANON_DONE ] && grep -q RESULT $M/runs_of_conn_S.log 2>/dev/null; do sleep 30; done
args="--vertex $M/runs/of_vertex_S/last.pt --conn $M/runs/of_conn_S/last.pt --curve $M/runs/curve_canon_last.pt"
taskset -c 12,13 $PY $M/infer_chain_v2.py --split train --n 34 $args --out-dir $M/chain/of_train > $M/chain_of_train.log 2>&1 &
p1=$!
taskset -c 14,15 $PY $M/infer_chain_v2.py --split val --n 34 $args --out-dir $M/chain/of_val > $M/chain_of_val.log 2>&1 &
p2=$!
wait $p1 $p2
touch $M/CHAIN_DONE
