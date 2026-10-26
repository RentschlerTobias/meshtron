#!/bin/bash
# Copy the data of the mars sandbox (/work) next to the code in experiments/flow_topology (all gitignored).
# Idempotent (cp -u). Structure mirrors /work/meshtron_v2 so the scripts only need a path prefix change.
set -eu
FT=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
W=/work/meshtron_v2
cp_tree() { mkdir -p "$2"; (cd "$1" && tar cf - "${@:3}") | (cd "$2" && tar xf -); }
echo "[archive] -> $FT"
# datasets, checkpoints (without smoke / sanity runs), small result trees, logs
cp_tree $W $FT data
(cd $W && tar cf - --exclude='runs/*smoke*' --exclude='runs/*sanity*' runs) | (cd $FT && tar xf -)
cp_tree $W $FT viz vtk_val continuity
mkdir -p $FT/logs && cp -u $W/*.log $FT/logs/ 2>/dev/null || true
# chain: json / logs of every run, VTKs only of the final ones
(cd $W && tar cf - --exclude='*.vtk' chain) | (cd $FT && tar xf -)
cp_tree $W $FT chain/m4v2_val_full chain/m4_val_full
# blade: final states complete, the rest only reports / plots
mkdir -p $FT/blade/out
for d in oring_0131_v2 oring_0131_opt blocking_0131_gap_opt keepbl_0131_tube layers_0131_dbg; do
  cp_tree $W/blade/out $FT/blade/out $d
done
(cd $W/blade/out && find . \( -name '*.json' -o -name '*.png' \) -print0 | tar cf - --null -T -) | (cd $FT/blade/out && tar xf -)
cp_tree $W/blade $FT/blade keep_bl
# other sandbox folders
cp_tree /work $FT curves_post minimality literature
mkdir -p $FT/relabel && cp_tree /work/relabel $FT/relabel out audit.csv
cp_tree /work $FT datagen_papaya
# datagen: every run without tfi.vtk and without the dtOO case (except case/mesh.msh, the geometry)
mkdir -p $FT/datagen
cp -u /work/datagen_full/audit*.csv $FT/datagen/
(cd /work/datagen_full/datagen && tar cf - --exclude='runs/*/tfi.vtk' --exclude='runs/*/case' runs) | (cd $FT/datagen && tar xf -)
for r in /work/datagen_full/datagen/runs/*/; do
  n=$(basename "$r"); [ -f "$r/case/mesh.msh" ] || continue
  mkdir -p "$FT/datagen/runs/$n/case"; cp -u "$r/case/mesh.msh" "$FT/datagen/runs/$n/case/"
done
du -sh $FT/{data,runs,chain,blade/out,blade/keep_bl,viz,vtk_val,continuity,logs,curves_post,minimality,literature,relabel,datagen}
for f in data/flow_canon_full.pt runs/vtx_full_s0/best.pt datagen/runs/tistos_s1_0131/sample.npz; do
  a=$(sha1sum "$FT/$f" | cut -c1-12); case $f in datagen/*) s=/work/datagen_full/datagen/${f#datagen/};; *) s=$W/$f;; esac
  b=$(sha1sum "$s" | cut -c1-12); echo "[check] $f nfs=$a work=$b $([ "$a" = "$b" ] && echo OK || echo MISMATCH)"
done
echo "[archive] done"
