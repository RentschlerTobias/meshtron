#!/bin/bash
# Minimality check: re-run beam_collapse on the stored raw AlgoHex mesh with a wider / deeper beam
# (datagen: width 16, depth 10, guard struct). Same guard, so the result is directly comparable.
n=$1; R=/work/datagen_full/datagen/runs/$n; O=/work/minimality/$n; mkdir -p $O
E=/opt/stack/domain_partition_3D/experimentell/hex3d_algohex
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd $E && timeout -k 60 5400 /opt/venv-sci/bin/python beam_collapse.py $R/hex_hex_x.ovm --input-vtk $R/tet.vtk \
  --out $O/blocks.vtk --width 48 --depth 16 --guard struct --workers 2 > $O/beam.log 2>&1
echo "$n rc=$? $(grep -E '^\[beam\] result' $O/beam.log | grep -oE "'blocks': [0-9]+, 'cuboids': [0-9]+")"
