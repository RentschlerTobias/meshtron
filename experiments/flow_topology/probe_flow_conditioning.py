#!/usr/bin/env python3
"""Does the flow_geo model use its point-cloud conditioning? (flow_geo counterpart
of probe_conditioning.py)

For every val item, sample corners + curves with (a) its own surface points and
(b) the points of the next val item (same noise seed in both, so only the cloud
differs). Reports, in dataset units * 1000 (median over items):
  own          corner error to the item's GT
  other        corner error to the item's GT when fed the other item's points
  other->src   the same prediction against the other item's GT (should be ~own
               if the model follows the cloud)
  mean tpl     mean-template baseline
  spread       median over corners of the std across items: predictions vs GT
               (a template-copying model has a much smaller spread than GT)
Usage: probe_flow_conditioning.py runs/<flow_geo run>/last.pt [...]
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "/opt/stack/meshtron/experiments/flow_topology")
from flow_geo import FlowGeo, Topology, sample  # noqa: E402
from train import batch_points  # noqa: E402

data = torch.load("/opt/stack/meshtron/experiments/flow_topology/data/flow_canon.pt", weights_only=False)
rmax = float(data["r_bounds"][1]); z0, z1 = (float(v) for v in data["z_bounds"])
lo, hi = torch.tensor([-rmax, -rmax, z0]), torch.tensor([rmax, rmax, z1])
va = data["val"]


@torch.no_grad()
def gen(model, items, mode, steps, seed=1234):
    g = torch.Generator(device="cuda"); g.manual_seed(seed)
    out = []
    for i in range(0, len(items), 16):
        pts, pm = batch_points(items[i:i + 16], "cuda")
        V, _ = model.topo.decode(*sample(model, model.enc(pts, pm), mode, steps, g))
        out.append(V.cpu().numpy())
    return np.concatenate(out)


def err(P, items):
    G = np.stack([it["verts"].numpy() for it in items])
    return 1000 * np.linalg.norm(P - G, axis=-1).mean(1)


for path in sys.argv[1:]:
    ck = torch.load(path, weights_only=False); a = ck["args"]
    tr = data["train"]
    if a["train_frac"] < 1.0:
        rng = np.random.default_rng(a["seed"])
        tr = [tr[i] for i in sorted(rng.choice(len(tr), max(1, round(a["train_frac"] * len(tr))), replace=False))]
    tp = Topology(tr, lo, hi).to("cuda")
    m = FlowGeo(tp, a["d"], a["layers"], a["heads"], a["n_latent"], 0.0, curves=a["parts"] == "all").cuda()
    m.load_state_dict(ck["model"]); m.eval()
    rolled_src = va[1:] + va[:1]                      # item k gets the points of item k+1
    own, rolled = gen(m, va, a["mode"], a["steps"]), gen(m, rolled_src, a["mode"], a["steps"])
    mean = tp.vmean.cpu().numpy()[None].repeat(len(va), 0)
    G = np.stack([it["verts"].numpy() for it in va])
    print(f"{path} ({a['mode']}/{a['parts']}), {len(va)} val items")
    print(f"  corner err x1000, median: own {np.median(err(own, va)):.1f} | other {np.median(err(rolled, va)):.1f}"
          f" | other->src {np.median(err(rolled, rolled_src)):.1f} | mean tpl {np.median(err(mean, va)):.1f}")
    print(f"  spread x1000: predictions {1000*np.median(np.linalg.norm(own.std(0), axis=1)):.1f}"
          f" | GT {1000*np.median(np.linalg.norm(G.std(0), axis=1)):.1f}", flush=True)
