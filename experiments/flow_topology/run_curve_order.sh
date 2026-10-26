#!/bin/bash
# Curve head in the generator's canonical vertex/block order (fix for the order mismatch
# found 2026-10-06), then the chain on train (best case) and val. Marker ORDER_DONE.
set -u
M=/opt/stack/meshtron/experiments/flow_topology; PY=/opt/venv-sci/bin/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/opt/stack/domain_partition_3D/experimentell/hex3d_algohex
(cd /opt/stack/meshtron && OMP_NUM_THREADS=4 taskset -c 0-7 $PY -m meshtron.training.train_polytron --stage curve \
   --data $M/data/curve_canon_order.pt --out $M/runs/curve_order --epochs 300 --bs 16 --lr 3e-4 --d 256 --heads 8 \
   --layers 6 --dropout 0.1 --n-points 2048 --eval-every 10 --weight-label 7) > $M/runs_curve_order.log 2>&1
args="--vertex $M/runs/of_vertex_S/last.pt --conn $M/runs/of_conn_S/last.pt --curve $M/runs/curve_order_last.pt"
export OMP_NUM_THREADS=2
taskset -c 0-3 $PY $M/infer_chain_v2.py --split train --n 34 $args --out-dir $M/chain/order_train > $M/chain_order_train.log 2>&1 &
p1=$!
taskset -c 4-7 $PY $M/infer_chain_v2.py --split val --n 34 $args --out-dir $M/chain/order_val > $M/chain_order_val.log 2>&1 &
p2=$!
wait $p1 $p2
touch $M/ORDER_DONE
