#!/bin/bash
# Stage 3 with permutation augmentation (A) and with permutation augmentation + masked known tokens (B) on the
# post-processed targets, same settings as runs/var_post_s0; plus the known x permutation evaluation of that
# baseline (known tokens only fixed at sampling = inpainting by replacement).
M=/opt/stack/meshtron/experiments/flow_topology; R=/work/meshtron_v2/runs; D=/work/meshtron_v2/data/flow_canon_full_post.pt
PY=/opt/venv-sci/bin/python; cd $M
export OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
taskset -c 0-3 $PY flow_geo_var.py --data $D --out $R/varm_perm_s0 --epochs 300 --eval-every 50 --sigma 0.035 --perm-aug > $R/varm_perm_s0.log 2>&1 &
p1=$!
taskset -c 4-7 $PY flow_geo_var.py --data $D --out $R/varm_mask_s0 --epochs 300 --eval-every 50 --sigma 0.035 --perm-aug --mask-policy recipe > $R/varm_mask_s0.log 2>&1 &
p2=$!
taskset -c 8-9 $PY flow_geo_var.py --data $D --out $R/varm_base_eval --eval-ckpt $R/var_post_s0/last.pt > $R/varm_base_eval.log 2>&1
wait $p1 $p2
touch $R/VARM_DONE
