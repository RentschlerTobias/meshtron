#!/bin/bash
# Pointer round 2b: A = sliding without corner labels, B = sliding + labels with 15 % corruption. 75 epochs.
M=/opt/stack/meshtron/experiments/flow_topology; PY=/opt/venv-sci/bin/python; cd $M
export OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/opt/stack/domain_partition_3D/experimentell/hex3d_algohex
taskset -c 0-3 $PY pointer_conn2.py --data data/flow_canon_full.pt --vtx-ckpt runs/vtx_full_s0/best.pt --out runs/ptr2_nolab \
  --epochs 75 --eval-every 25 --k 4 --labels none > ptr2_nolab.log 2>&1 &
taskset -c 4-7 $PY pointer_conn2.py --data data/flow_canon_full.pt --vtx-ckpt runs/vtx_full_s0/best.pt --out runs/ptr2_labnoise \
  --epochs 75 --eval-every 25 --k 4 --label-noise 0.15 > ptr2_labnoise.log 2>&1 &
wait
touch $M/PTR2B_DONE
