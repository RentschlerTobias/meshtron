#!/bin/bash
# Re-run of the two val_new chains of run_flow_papaya.sh (raw samples of the new data live in
# /work/datagen_papaya/runs, not in /work/relabel/out).
M=/work/meshtron_v2; PY=/opt/venv-sci/bin/python; cd $M
export OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/opt/stack/domain_partition_3D/experimentell/hex3d_algohex
SR=/work/datagen_papaya/runs
taskset -c 0-3 $PY infer_chain_flow.py --data data/flow_canon_papaya.pt --ckpt runs/flowp_flow_all_s0/last.pt \
  --split val_new --src-root $SR --out-dir chain/flowp_flow_all_valnew > chain_flowp_flow_all_valnew.log 2>&1 &
p1=$!
taskset -c 4-7 $PY infer_chain_flow.py --data data/flow_canon.pt --eval-data data/flow_canon_papaya.pt --src-root $SR \
  --ckpt runs/flow_flow_all_s0/last.pt --split val_new --out-dir chain/flow_flow_all_valnew > chain_flow_flow_all_valnew.log 2>&1 &
p2=$!
wait $p1 $p2
touch $M/FLOWP_VALNEW_DONE
