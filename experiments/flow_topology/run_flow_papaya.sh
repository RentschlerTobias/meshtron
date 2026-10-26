#!/bin/bash
# flow_geo (flow, verts + curves) on old + new tistos data (data/flow_canon_papaya.pt: train 416,
# val 34 old, val_new 28 new), 3 seeds in parallel; then the chain for seed 0 on both val sets and
# the old-data seed-0 model on val_new (cores 0-7).
set -u
M=/opt/stack/meshtron/experiments/flow_topology; PY=/opt/venv-sci/bin/python
cd $M
export OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
cpus=("0-1" "2-4" "5-7"); pids=()
for seed in 0 1 2; do
  out=runs/flowp_flow_all_s$seed
  taskset -c ${cpus[$seed]} $PY flow_geo.py --data data/flow_canon_papaya.pt --mode flow --parts all --seed $seed \
    --out $out --epochs 1500 --eval-every 250 > $out.log 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
touch $M/FLOWP_DONE
export PYTHONPATH=/opt/stack/domain_partition_3D/experimentell/hex3d_algohex
taskset -c 0-2 $PY infer_chain_flow.py --data data/flow_canon_papaya.pt --ckpt runs/flowp_flow_all_s0/last.pt \
  --split val --out-dir chain/flowp_flow_all_val > chain_flowp_flow_all_val.log 2>&1 &
p1=$!
taskset -c 3-5 $PY infer_chain_flow.py --data data/flow_canon_papaya.pt --ckpt runs/flowp_flow_all_s0/last.pt \
  --split val_new --out-dir chain/flowp_flow_all_valnew > chain_flowp_flow_all_valnew.log 2>&1 &
p2=$!
taskset -c 6-7 $PY infer_chain_flow.py --data data/flow_canon.pt --eval-data data/flow_canon_papaya.pt \
  --ckpt runs/flow_flow_all_s0/last.pt --split val_new --out-dir chain/flow_flow_all_valnew > chain_flow_flow_all_valnew.log 2>&1 &
p3=$!
wait $p1 $p2 $p3
touch $M/FLOWP_CHAIN_DONE
