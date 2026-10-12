#!/bin/bash
# M1 retry: stage-3 model with 600 epochs (val corner best near ep 500 in runs/var_s0), then the M1 chain.
M=/work/meshtron_v2; PY=/opt/venv-sci/bin/python; cd $M
export OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
taskset -c 0-3 $PY flow_geo_var.py --data data/flow_canon_papaya.pt --out runs/var_e600_s0 --epochs 600 --eval-every 100 --sigma 0.035 --seed 0 > var_e600_s0.log 2>&1
export PYTHONPATH=/opt/stack/domain_partition_3D/experimentell/hex3d_algohex
taskset -c 0-3 $PY infer_chain_var.py --var-ckpt runs/var_e600_s0/last.pt --split val_new --src-root /work/datagen_papaya/runs --out-dir chain/var_m1b_valnew > chain_var_m1b_valnew.log 2>&1 &
taskset -c 4-7 $PY infer_chain_var.py --var-ckpt runs/var_e600_s0/last.pt --split val --src-root /work/relabel/out --out-dir chain/var_m1b_val > chain_var_m1b_val.log 2>&1 &
wait
touch $M/VAR_M1B_DONE
