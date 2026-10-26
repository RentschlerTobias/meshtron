#!/bin/bash
# Wait for all datagen lanes, then pack /work/datagen into one xz archive and copy it to NFS.
set -u
cd /work/datagen
while kill -0 $(awk "{print \$2}" start_lanes.pid) 2>/dev/null; do sleep 60; done
echo "[$(date +%T)] lanes finished: npz $(ls runs/*/sample.npz | wc -l), FAIL $(grep -c FAIL lanes.log)"
A=/work/datagen_tistos_s1_papaya_2026-10-12.tar.xz
tar -C /work -cf - --exclude=datagen/locks datagen | xz -T 96 -3 > $A || { echo "pack FAILED"; exit 1; }
echo "[$(date +%T)] packed $(du -h $A | cut -f1)"
xz -t -T 96 $A || { echo "xz test FAILED"; exit 1; }
sha256sum $A | tee $A.sha256
cp $A /opt/stack/ && (cd /opt/stack && sed "s|$A|$(basename $A)|" $A.sha256 > $(basename $A).sha256 && sha256sum -c $(basename $A).sha256) || { echo "copy/verify FAILED"; exit 1; }
echo "[$(date +%T)] DONE -> /opt/stack/$(basename $A)"
