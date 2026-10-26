#!/bin/bash
# Flow-matching geometry model (flow_geo.py): flow vs regress x verts vs all, 3 seeds,
# 303 train / 34 val (data/flow_canon.pt). 3 runs at a time on the GPU, CPU cores 0-7.
set -u
M=/opt/stack/meshtron/experiments/flow_topology; PY=/opt/venv-sci/bin/python
cd $M
export OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
jobs_list=()
for seed in 0 1 2; do for mode in flow regress; do for parts in verts all; do
  jobs_list+=("$mode $parts $seed")
done; done; done
cpus=("0-1" "2-4" "5-7")
i=0
while [ $i -lt ${#jobs_list[@]} ]; do
  pids=()
  for slot in 0 1 2; do
    [ $i -ge ${#jobs_list[@]} ] && break
    set -- ${jobs_list[$i]}
    out=runs/flow_${1}_${2}_s${3}
    taskset -c ${cpus[$slot]} $PY flow_geo.py --data data/flow_canon.pt --mode $1 --parts $2 --seed $3 \
      --out $out --epochs 1500 --eval-every 250 > $out.log 2>&1 &
    pids+=($!); i=$((i+1))
  done
  wait "${pids[@]}"
done
touch $M/FLOW_DONE
