#!/bin/bash
# After run_flow.sh: conditioning probe for every flow_geo run (all seeds).
cd /work/meshtron_v2
while [ ! -f FLOW_DONE ]; do sleep 60; done
OMP_NUM_THREADS=1 taskset -c 7 /opt/venv-sci/bin/python probe_flow_conditioning.py runs/flow_*_s[0-9]/last.pt > probe_flow_conditioning.log 2>&1
touch FLOW_PROBE_DONE
