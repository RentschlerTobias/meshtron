#!/usr/bin/env python3
"""Evaluate flow_geo checkpoints on another item set (e.g. models trained on the old data,
evaluated on the new held-out geometries). The topology statistics (mean template, std) are
rebuilt from the checkpoint's OWN training data, as in training.

    python eval_flow.py --train-data data/flow_canon.pt --eval-data data/flow_canon_papaya.pt \
        --split val_new runs/flow_flow_all_s0/last.pt [...]
"""
import argparse
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/opt/stack/meshtron/experiments/flow_topology")
from flow_geo import FlowGeo, Topology, evaluate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--train-data", required=True)
ap.add_argument("--eval-data", required=True)
ap.add_argument("--split", default="val_new")
ap.add_argument("--n-samples", type=int, default=8)
ap.add_argument("ckpts", nargs="+")
a = ap.parse_args()
td = torch.load(a.train_data, weights_only=False)
items = torch.load(a.eval_data, weights_only=False)[a.split]
rmax = float(td["r_bounds"][1]); z0, z1 = (float(v) for v in td["z_bounds"])
lo, hi = torch.tensor([-rmax, -rmax, z0]), torch.tensor([rmax, rmax, z1])
for path in a.ckpts:
    ck = torch.load(path, weights_only=False); fa = ck["args"]
    tr = td["train"]
    if fa["train_frac"] < 1.0:
        rng = np.random.default_rng(fa["seed"])
        tr = [tr[i] for i in sorted(rng.choice(len(tr), max(1, round(fa["train_frac"] * len(tr))), replace=False))]
    tp = Topology(tr, lo, hi).to("cuda")
    m = FlowGeo(tp, fa["d"], fa["layers"], fa["heads"], fa["n_latent"], 0.0, curves=fa["parts"] == "all").cuda()
    m.load_state_dict(ck["model"])
    r = evaluate(m, items, fa["mode"], fa["steps"], a.n_samples, "cuda")
    print(json.dumps({"ckpt": path, "split": a.split, **{k: round(v, 2) for k, v in r.items()}}), flush=True)
