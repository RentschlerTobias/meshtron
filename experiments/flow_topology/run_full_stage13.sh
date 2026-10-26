#!/bin/bash
# Stage 1 (flow_vertex.py) and stage 3 (flow_geo_var.py) on data/flow_canon_full.pt (train 995), in parallel on the GPU.
# Epochs scaled to ~2.4x more steps per epoch: stage 1 600 ep (~38k steps, M2 run had 26k and was still improving),
# stage 3 300 ep (~19k steps, M1 best near 13-16k).
M=/opt/stack/meshtron/experiments/flow_topology; PY=/opt/venv-sci/bin/python; cd $M
export OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
taskset -c 0-3 $PY flow_vertex.py --data data/flow_canon_full.pt --out runs/vtx_full_s0 --epochs 600 --eval-every 50 --seed 0 > vtx_full_s0.log 2>&1 &
p1=$!
taskset -c 4-7 $PY flow_geo_var.py --data data/flow_canon_full.pt --out runs/var_full_s0 --epochs 300 --eval-every 50 --sigma 0.035 --seed 0 > var_full_s0.log 2>&1 &
p2=$!
wait $p1 $p2
touch $M/FULL_STAGE13_DONE
