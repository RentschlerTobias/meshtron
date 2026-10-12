#!/bin/bash
# Stage 3 on the post-processed curve targets (data/flow_canon_full_post.pt), same settings as runs/var_full_s0
# (raw targets) for a direct comparison; starts when that run has written its result.
M=/work/meshtron_v2; PY=/opt/venv-sci/bin/python; cd $M
until [ -f runs/var_full_s0/result.json ]; do sleep 60; done
export OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
taskset -c 4-7 $PY flow_geo_var.py --data data/flow_canon_full_post.pt --out runs/var_post_s0 --epochs 300 --eval-every 50 --sigma 0.035 --seed 0 > var_post_s0.log 2>&1
touch $M/VAR_POST_DONE
