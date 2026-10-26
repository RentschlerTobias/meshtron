#!/bin/bash
# Step 2 (HANDOFF_2026-10-12): seam arcs + TFI curve optimisation on val_new and val, 2 lanes x 4 cores.
M=/opt/stack/meshtron/experiments/flow_topology; PY=/opt/venv-sci/bin/python; cd $M
export OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/opt/stack/domain_partition_3D/experimentell/hex3d_algohex
taskset -c 0-3 $PY curve_opt.py --split val_new --src-root /opt/stack/meshtron/experiments/flow_topology/datagen_papaya/runs --out-dir chain/curve_opt_valnew > curve_opt_valnew.log 2>&1 &
p1=$!
taskset -c 4-7 $PY curve_opt.py --split val --src-root /opt/stack/meshtron/experiments/flow_topology/relabel/out --out-dir chain/curve_opt_val > curve_opt_val.log 2>&1 &
p2=$!
wait $p1 $p2
touch $M/CURVE_OPT_DONE
