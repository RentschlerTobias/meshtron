#!/usr/bin/env python3
"""Flow dataset with post-processed curve targets (domain_partition_3D curve_postprocess.py side files).

Takes an existing flow dataset (canonical vertex order, e.g. data/flow_canon_full.pt) and replaces per item
  verts  <- post_vertices      (vertices inside a feature line moved onto their seam)
  eoff   <- from post_edge_ctrl (feature edges: G1 cubics on the exact seam arcs; others follow moved corners)
and adds `feature` [E] (bool, feature edge) and `post_ok`. Items whose post-processing is incomplete (a feature
edge without arc, a remaining kink > 1 deg, a vertex moved by more than --max-move) keep their original
targets with post_ok = False, so the validation splits stay complete and comparable.

    python build_flow_dataset_post.py --data data/flow_canon_full.pt --curves /work/curves_post \
        --out data/flow_canon_full_post.pt
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, "/opt/stack/meshtron")
from meshtron.data.polytron_blocks import ctrl_to_offsets  # noqa: E402

SRC = {"tistos": "/work/datagen_full/datagen/runs", "machine": "/work/relabel/out"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--curves", required=True, help="directory with <name>_curves.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-move", type=float, default=0.1, help="largest accepted vertex move (dataset units)")
    a = ap.parse_args()
    data = torch.load(a.data, weights_only=False)
    stats = Counter()
    for sp in [k for k in data if isinstance(data[k], list)]:
        for it in data[sp]:
            name = it["name"]
            raw = np.load(os.path.join(SRC[name.split("_")[0]], name, "sample.npz"), allow_pickle=False)
            cfile = os.path.join(a.curves, f"{name}_curves.npz")
            if not os.path.exists(cfile):
                it["post_ok"] = False; stats["no_curves"] += 1; continue
            post = np.load(cfile, allow_pickle=False)
            q = json.loads(str(post["post_quality"]))
            Vc = it["verts"].numpy().astype(np.float64)
            d, perm = cKDTree(raw["vertices"]).query(Vc)              # canonical id -> raw id
            assert d.max() < 1e-6 and len(set(perm.tolist())) == len(Vc), name
            ctrl = {(int(p), int(r)): c for (p, r), c in zip(raw["edges"], post["post_edge_ctrl"])}
            featd = {(int(p), int(r)): bool(f) for (p, r), f in zip(raw["edges"], post["post_feature_edge"])}
            E = it["edges"].numpy().astype(np.int64)
            V = post["post_vertices"][perm]
            eoff = np.zeros((len(E), 6)); feat = np.zeros(len(E), bool)
            for k, (p, r) in enumerate(E):
                c1, c2 = ctrl[(int(perm[p]), int(perm[r]))]
                eoff[k] = ctrl_to_offsets(V[p], V[r], c1, c2)
                feat[k] = featd[(int(perm[p]), int(perm[r]))]
            ok = (q["feature_arcs"] == q["feature_edges"]
                  and max(q["kink_after_max"].values(), default=0.0) <= 1.0
                  and q["vertex_moved_max"] <= a.max_move)
            it["feature"] = torch.tensor(feat)
            it["post_ok"] = bool(ok)
            if ok:
                it["verts_raw"], it["eoff_raw"] = it["verts"], it["eoff"]
                it["verts"] = torch.tensor(V, dtype=torch.float32)
                it["eoff"] = torch.tensor(eoff, dtype=torch.float32)
            stats[f"{sp}_ok" if ok else f"{sp}_kept_raw"] += 1
    torch.save(data, a.out)
    print(f"wrote {a.out}: {dict(stats)}")


if __name__ == "__main__":
    main()
