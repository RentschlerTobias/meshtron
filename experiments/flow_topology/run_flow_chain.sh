#!/bin/bash
# After run_flow.sh: end-to-end chain (snap + curved TFI) for the seed-0 checkpoint of
# every flow_geo configuration, val split, 4 jobs x 2 cores on cores 0-7.
set -u
M=/work/meshtron_v2; PY=/opt/venv-sci/bin/python
cd $M
while [ ! -f FLOW_DONE ]; do sleep 60; done
export OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/opt/stack/domain_partition_3D/experimentell/hex3d_algohex
cpus=("0-1" "2-3" "4-5" "6-7"); pids=(); k=0
for cfg in flow_all flow_verts regress_all regress_verts; do
  taskset -c ${cpus[$k]} $PY infer_chain_flow.py --ckpt runs/flow_${cfg}_s0/last.pt --split val --n 34 --n-best 4 \
    --out-dir chain/flow_${cfg}_val > chain_flow_${cfg}_val.log 2>&1 &
  pids+=($!); k=$((k+1))
done
wait "${pids[@]}"
touch $M/FLOW_CHAIN_DONE
