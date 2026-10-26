#!/bin/bash
# Best case for stage 1: vertex model trained to convergence on train, then the chain on train.
set -u
M=/opt/stack/meshtron/experiments/flow_topology; PY=/opt/venv-sci/bin/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/opt/stack/domain_partition_3D/experimentell/hex3d_algohex
(cd $M && OMP_NUM_THREADS=4 taskset -c 0-7 $PY train.py --task vertex --data data/v2_canon.pt --out runs/of_vertex_S_long \
   --d 128 --layers 2 --heads 4 --n-latent 32 --epochs 1500 --eval-every 50 --patience 1000 --bs 16 \
   --select last --eval-train 34) > $M/runs_of_vertex_S_long.log 2>&1
grep RESULT $M/runs_of_vertex_S_long.log
args="--vertex $M/runs/of_vertex_S_long/last.pt --conn $M/runs/of_conn_S/last.pt --curve $M/runs/curve_order_last.pt"
export OMP_NUM_THREADS=2
taskset -c 0-3 $PY $M/infer_chain_v2.py --split train --n 34 $args --out-dir $M/chain/long_train > $M/chain_long_train.log 2>&1 &
p1=$!
taskset -c 4-7 $PY $M/infer_chain_v2.py --split val --n 34 $args --out-dir $M/chain/long_val > $M/chain_long_val.log 2>&1 &
p2=$!
wait $p1 $p2
touch $M/LONG_DONE
