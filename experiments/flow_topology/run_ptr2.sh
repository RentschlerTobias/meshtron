#!/bin/bash
# M3 round 2: pointer_conn2.py (sliding augmentation, patch features, snap before the pointer, K=8 selection).
M=/opt/stack/meshtron/experiments/flow_topology; PY=/opt/venv-sci/bin/python; cd $M
export OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/opt/stack/domain_partition_3D/experimentell/hex3d_algohex
taskset -c 0-3 $PY pointer_conn2.py --data data/flow_canon_full.pt --vtx-ckpt runs/vtx_full_s0/best.pt --out runs/ptr2_s0 \
  --epochs 150 --eval-every 50 --k 8 --seed 0 > ptr2_s0.log 2>&1
touch $M/PTR2_DONE
