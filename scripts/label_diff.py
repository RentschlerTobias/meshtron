"""Which boundary-face labels differ between two samples of the same topology
(blocks matched by nearest centroid)? Debug aid for topo code mismatches."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from meshtron.data.topo_row_plan import load_npz_sample as load  # noqa: E402

# usage: label_diff.py <a/sample.npz> <b/sample.npz>

a, b = load(sys.argv[1]), load(sys.argv[2])
def table(s):
    V = s["vertices_cartesian"].numpy(); out = {}
    for f, l in s["face_label"].items():
        out[tuple(np.round(V[list(f)].mean(0), 2))] = l
    return out
ta, tb = table(a), table(b)
ka = np.array(list(ta)); kb = np.array(list(tb))
print("boundary faces:", len(ta), len(tb), "labels a:", np.unique(list(ta.values()), return_counts=True))
for k, l in ta.items():
    j = np.argmin(np.linalg.norm(kb - np.array(k), axis=1))
    if tb[tuple(kb[j])] != l:
        print(f"face at {k}: {l} vs {tb[tuple(kb[j])]} (match dist {np.linalg.norm(kb[j]-k):.3f})")
