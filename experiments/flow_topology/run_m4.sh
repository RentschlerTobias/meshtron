#!/bin/bash
# M4: whole chain (stage 1 -> snap -> pointer + feature-complete ranking -> stage 3 model B with known features
# -> TFI -> TFI curve optimisation) on val, val_new, val_full.
M=/opt/stack/meshtron/experiments/flow_topology; PY=/opt/venv-sci/bin/python; cd $M
export OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/opt/stack/domain_partition_3D/experimentell/hex3d_algohex
taskset -c 0-3 $PY infer_chain_m4.py --split val --out-dir chain/m4_val > chain_m4_val.log 2>&1 &
taskset -c 4-7 $PY infer_chain_m4.py --split val_new --out-dir chain/m4_val_new > chain_m4_val_new.log 2>&1 &
taskset -c 8-11 $PY infer_chain_m4.py --split val_full --out-dir chain/m4_val_full > chain_m4_val_full.log 2>&1 &
wait
touch $M/M4_DONE
