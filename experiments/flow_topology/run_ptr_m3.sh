#!/bin/bash
# M3 of VARIABLE_TOPOLOGY_PLAN.md: set pointer (pointer_conn.py) on the full data, evaluated on GT, GT+noise
# and stage-1 corners (runs/vtx_full_s0/best.pt).
M=/work/meshtron_v2; PY=/opt/venv-sci/bin/python; cd $M
export OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
taskset -c 0-3 $PY pointer_conn.py --data data/flow_canon_full.pt --vtx-ckpt runs/vtx_full_s0/best.pt --out runs/ptr_s0 \
  --epochs 150 --eval-every 25 --seed 0 > ptr_s0.log 2>&1
touch $M/PTR_M3_DONE
