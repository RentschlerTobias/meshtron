#!/bin/bash
# All-machine fine refill sweep (mars):
#   batch + batch_t19_sweep */blocks.vtk  ×  h {0.05,0.04,0.03,0.02}
# - 24 parallel workers, idempotent (skips existing outputs)
# - needs: domain_partition_3D repo with data/T1_9/T1_9_tet_v5.vtk
# - outputs: <OUT>/T1_9_blocks_machine_<name>_h<h>.vtk + .log
set -u

MESH_BASE="/mnt/fs2/home/trentschler/ws_meshtron/meshtron/data"
TET_HOME="/mnt/fs2/home/trentschler/ws_domain_partition/domain_partition_3D"
OUTDIR="$MESH_BASE/fine_refill"
LOGDIR="$MESH_BASE/fine_refill_logs"
HS="0.05 0.04 0.03 0.02"
JOBS=24

mkdir -p "$OUTDIR" "$LOGDIR"
cd "$TET_HOME" || exit 1

one() {
  local vtk="$1" h="$2"
  local m name out log
  name=$(basename "$(dirname "$vtk")")
  out="$OUTDIR/${name}_h${h}.vtk"
  log="$LOGDIR/${name}_h${h}.log"
  [ -f "$out" ] && { echo "skip $name h$h"; return 0; }
  cd "$TET_HOME" || return 1
  if uv run --with numpy --with scipy --with meshio python \
      experimentell/hex3d_algohex/tfi.py "$vtk" \
      --target-h "$h" --solve-divisions --apply-divisions \
      --out "$out" > "$log" 2>&1; then
    echo "OK   $name h$h"
  else
    echo "FAIL $name h$h  (siehe $log)"
  fi
}
export -f one
export TET_HOME LOGDIR

for SRC in "$MESH_BASE/batch" "$MESH_BASE/batch_t19_sweep"; do
  ls "$SRC"/*/blocks.vtk 2>/dev/null
done | while read -r f; do
  for h in $HS; do printf '%s %s\n' "$f" "$h"; done
done | xargs -P "$JOBS" -n 2 bash -c 'one "$1" "$2"' _

echo "refill_all_machines done"
