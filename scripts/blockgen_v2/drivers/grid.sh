#!/bin/bash
# Training grid of the block generators v2 (night run 2026-10-05, reports/blockgen_v2_night_grid.md):
# vertex / hexarow / conn at sizes S, M, L on one dataset, best-val checkpoint, summary table.
#   BLOCKGEN_WORK=<dir> grid.sh <dataset.pt>     (cores: CORES, default 0-7)
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); PY=${PY:-python}; CORES=${CORES:-0-7}
W=${BLOCKGEN_WORK:-$ROOT/runs/blockgen_v2}; DATA=$1; mkdir -p "$W"
export PYTHONDONTWRITEBYTECODE=1
log() { echo "[$(date +%F' '%T)] $*"; }
run() {  # name task d layers [extra]
  local n=$1 t=$2 d=$3 l=$4; shift 4
  log "train $n"
  (cd "$ROOT" && OMP_NUM_THREADS=4 taskset -c $CORES $PY -m meshtron.training.train_blockgen --task $t \
     --data "$DATA" --out "$W/$n" --d $d --layers $l --epochs 400 --eval-every 5 --patience 12 --bs 16 "$@") \
     > "$W/$n.log" 2>&1
  grep RESULT "$W/$n.log" || log "FAILED $n (see $W/$n.log)"
}
run vertex_M  vertex  256 4
run hexarow_M hexarow 256 4
run conn_M    conn    256 4
run vertex_S  vertex  128 2 --heads 4 --n-latent 32
run hexarow_S hexarow 128 2 --heads 4 --n-latent 32
run conn_S    conn    128 2 --heads 4
run vertex_L  vertex  512 12
run hexarow_L hexarow 512 12
$PY - "$W" <<'PY'
import glob, json, sys
w = sys.argv[1]; rows = []
for f in sorted(glob.glob(f"{w}/*/result.json")):
    s = json.load(open(f))["summary"]; s["run"] = f.split('/')[-2]; rows.append(s)
with open(f"{w}/SUMMARY.md", "w") as fh:
    fh.write("| run | params M | best epoch | val loss | val n | valid | struct ok | corner mean (median) | corner max (median) | exact conn |\n|---|---|---|---|---|---|---|---|---|---|\n")
    for s in rows:
        fh.write(f"| {s['run']} | {s['params_M']:.2f} | {s['best_epoch']} | {s['best_val_loss']:.3f} | {s['n']} | "
                 f"{s.get('valid','')} | {s.get('struct_ok','')} | {s.get('corner_mean_median') or ''} | "
                 f"{s.get('corner_max_median') or ''} | {s.get('exact','')} |\n")
    fh.write("\ncorner values: distance generated vs GT corner in canonical order, dataset units x 1000.\n")
print(open(f"{w}/SUMMARY.md").read())
PY
log "grid done"
