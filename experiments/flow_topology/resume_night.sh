#!/bin/bash
# Resume the overnight run in a fresh container (e.g. started from a tmux session on
# the host), after the original Claude session -- and with it all its jobs -- ended.
#
#   1. relabel: drop half-finished samples (no sample.npz) and their claims, restart
#      6 lanes (cores 0-7 forward, 8-11 backward); finished samples are skipped
#   2. run_all2.sh: TFI backfill -> audit -> v2 dataset -> training grid -> SUMMARY.md
#   3. run_curve.sh: curve head (weight-label 7) on the same split
# The script stays in the foreground until everything has finished, so the
# container lives exactly as long as the work. Do NOT run it while the original
# session's jobs are still alive.
set -u
R=/work/relabel; M=/work/meshtron_v2
log() { echo "[$(date +%F' '%T)] $*"; }

cd $R
n_drop=0
for d in out/*; do
  [ -f "$d/sample.npz" ] && continue
  rm -rf "$d" "locks/$(basename "$d")"; n_drop=$((n_drop + 1))
done
rm -f lane*.done lane_rev*.done
log "relabel: $(ls out/*/sample.npz 2>/dev/null | wc -l)/369 finished, $n_drop half-finished dropped"

pids=()
for k in 0 1 2 3; do ./run_lane.sh $k >> lane$k.log 2>&1 & pids+=($!); done
./run_lane_rev.sh 0 8,9 >> lane_rev0.log 2>&1 & pids+=($!)
./run_lane_rev.sh 1 10,11 >> lane_rev1.log 2>&1 & pids+=($!)

cd $M
rm -f ALL_DONE CURVE_DONE
./run_all2.sh > run_all.log 2>&1 & pids+=($!)
./run_curve.sh > run_curve.log 2>&1 & pids+=($!)
log "started ${#pids[@]} jobs; logs: $R/lane*.log, $M/run_all.log, $M/run_curve.log"
wait "${pids[@]}"
log "all finished: $(ls $M/ALL_DONE $M/CURVE_DONE 2>/dev/null | tr '\n' ' ')"
