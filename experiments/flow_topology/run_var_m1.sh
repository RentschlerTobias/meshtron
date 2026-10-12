#!/bin/bash
# M1 of VARIABLE_TOPOLOGY_PLAN.md: stage-3 model (flow_geo_var.py) on the 12-block papaya data, seed 0.
M=/work/meshtron_v2; PY=/opt/venv-sci/bin/python; cd $M
export OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
taskset -c 0-3 $PY flow_geo_var.py --data data/flow_canon_papaya.pt --out runs/var_s0 --sigma 0.035 --seed 0 > var_s0.log 2>&1
touch $M/VAR_M1_DONE
