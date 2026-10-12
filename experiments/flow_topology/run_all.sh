#!/bin/bash
# Overnight chain (2026-10-05): relabel done -> TFI backfill -> audit -> dataset -> training grid -> summary.
# Waits on the lane done-files of /work/relabel (never on a process pattern).
set -u
R=/work/relabel; M=/work/meshtron_v2; PY=/opt/venv-sci/bin/python
export PYTHONDONTWRITEBYTECODE=1
log() { echo "[$(date +%F' '%T)] $*"; }

log "waiting for relabel lanes"
until [ -f $R/lane0.done ] && [ -f $R/lane1.done ] && [ -f $R/lane2.done ] && [ -f $R/lane3.done ]; do sleep 60; done
log "relabel done: $(ls $R/out/*/sample.npz | wc -l) samples; failures: $(cat $R/lane*.log | grep -vc '^OK')"

log "TFI backfill (samples exported without refill)"
ls $R/out | xargs -P 8 -I{} bash -c 'k=$(( $(echo {} | cksum | cut -d" " -f1) % 8 )); '$R'/retfi.sh {} $k'

log "audit"
(cd $R && OMP_NUM_THREADS=1 taskset -c 0-7 $PY audit.py) 2>&1 | grep -v -i "warn\|filesize"

log "dataset"
(cd $M && OMP_NUM_THREADS=4 taskset -c 0-7 $PY build_dataset.py --out $M/data/v2_cart.pt) 2>&1 | grep -v -i "warn\|filesize" | tail -3

run() {  # name task d layers [extra]
  local n=$1 t=$2 d=$3 l=$4; shift 4
  log "train $n"
  (cd $M && OMP_NUM_THREADS=4 taskset -c 0-7 $PY train.py --task $t --data data/v2_cart.pt --out runs/$n \
     --d $d --layers $l --epochs 400 --eval-every 5 --patience 12 --bs 16 "$@") > $M/runs_$n.log 2>&1
  grep RESULT $M/runs_$n.log || log "FAILED $n (see runs_$n.log)"
}
mkdir -p $M/runs
run vertex_M  vertex  256 4
run hexarow_M hexarow 256 4
run conn_M    conn    256 4
run vertex_S  vertex  128 2 --heads 4 --n-latent 32
run hexarow_S hexarow 128 2 --heads 4 --n-latent 32
run conn_S    conn    128 2 --heads 4
run vertex_L  vertex  512 12
run hexarow_L hexarow 512 12

log "summary"
$PY - <<'EOF'
import glob, json
rows = []
for f in sorted(glob.glob('/work/meshtron_v2/runs/*/result.json')):
    s = json.load(open(f))["summary"]; s["run"] = f.split('/')[-2]; rows.append(s)
with open('/work/meshtron_v2/SUMMARY.md', 'w') as fh:
    fh.write("# v2 training grid (topo order, all labelled points, no block count)\n\n")
    fh.write("| run | params M | best epoch | val loss | val n | valid | struct ok | corner mean (median) | corner max (median) | exact conn |\n|---|---|---|---|---|---|---|---|---|---|\n")
    for s in rows:
        fh.write(f"| {s['run']} | {s['params_M']:.2f} | {s['best_epoch']} | {s['best_val_loss']:.3f} | {s['n']} | "
                 f"{s.get('valid','')} | {s.get('struct_ok','')} | {s.get('corner_mean_median') or ''} | "
                 f"{s.get('corner_max_median') or ''} | {s.get('exact','')} |\n")
    fh.write("\ncorner values: distance generated vs GT corner in canonical order, dataset units x 1000.\n")
print(open('/work/meshtron_v2/SUMMARY.md').read())
EOF
log "ALL DONE"
touch $M/ALL_DONE
