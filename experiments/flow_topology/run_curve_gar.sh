#!/bin/bash
# Hinge vs. Garanzha energy (curve_opt.py, two-pass seam assignment), val_new + val, 2 lanes x 4 cores.
M=/opt/stack/meshtron/experiments/flow_topology; PY=/opt/venv-sci/bin/python; cd $M
export OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/opt/stack/domain_partition_3D/experimentell/hex3d_algohex
V=pred_seam,pred_opt,pred_gar,gt_seam,gt_opt,gt_gar

taskset -c 0-3 $PY curve_opt.py --split val_new --src-root /opt/stack/meshtron/experiments/flow_topology/datagen_papaya/runs --variants $V --out-dir chain/curve_gar_valnew > curve_gar_valnew.log 2>&1 &
p1=$!
taskset -c 4-7 $PY curve_opt.py --split val --src-root /opt/stack/meshtron/experiments/flow_topology/relabel/out --variants $V --out-dir chain/curve_gar_val > curve_gar_val.log 2>&1 &
p2=$!
wait $p1 $p2
touch $M/CURVE_GAR_DONE
