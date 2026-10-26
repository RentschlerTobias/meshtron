#!/bin/bash
# M2 of VARIABLE_TOPOLOGY_PLAN.md: stage-1 corner set model (flow_vertex.py) on the papaya data, seed 0.
M=/opt/stack/meshtron/experiments/flow_topology; PY=/opt/venv-sci/bin/python; cd $M
export OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
taskset -c 0-3 $PY flow_vertex.py --data data/flow_canon_papaya.pt --out runs/vtx_s0 --epochs 1000 --seed 0 > vtx_s0.log 2>&1
touch $M/VTX_M2_DONE
