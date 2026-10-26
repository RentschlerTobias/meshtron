#!/bin/bash
# Chain (flowp corners -> stage 3 -> snap -> TFI) for stage 3 trained on the full data with raw targets
# (runs/var_full_s0, data/flow_canon_full.pt) and with post-processed targets (runs/var_post_s0,
# data/flow_canon_full_post.pt), on val, val_new, val_full.
M=/opt/stack/meshtron/experiments/flow_topology; PY=/opt/venv-sci/bin/python; cd $M
export OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/opt/stack/domain_partition_3D/experimentell/hex3d_algohex
run() {  # tag data ckpt split cores
  taskset -c $5 $PY infer_chain_var.py --data data/$2 --var-ckpt runs/$3/last.pt --split $4 --src-root auto \
    --out-dir chain/$1_$4 > chain_$1_$4.log 2>&1
}
for sp in val val_new val_full; do
  run varfull flow_canon_full.pt var_full_s0 $sp 0-3 &
  run varpost flow_canon_full_post.pt var_post_s0 $sp 4-7 &
  wait
done
touch $M/CHAIN_FULL_DONE
