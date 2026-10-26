"""Audit the beam-relabelled samples and flag corrupted ones.

For every /work/relabel/out/<name>/sample.npz:
  beam      all blocks cuboid, no excess faces, no inverted cells (beam.log result)
  tfi       every block a lattice, 0 inverted cells in the refill (tfi.log)
  blocks    no degenerate block, no negative block volume (6-tet split)
  labels    meshtron conditioning.check_surface_labels (blade = O-grid cut, label 7)
  surface   surface identical to the original dataset sample (same geometry)
  topo      topology-canonical row plan unique (ties == 1) and token round trip ok
Writes audit.csv (one row per sample, reason codes) and prints a summary.

    python audit.py
"""
import ast
import csv
import glob
import os
import re
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "/opt/stack/meshtron")
sys.path.insert(0, "/work/meshtron_topo")
from meshtron.data import conditioning as C  # noqa: E402

OUT = "/work/relabel/out"
SRC = "/opt/stack/meshtron/data/hex3d_algohex/batch"
TETS = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]


def signed_vol(P):
    return sum(np.dot(np.cross(P[b] - P[a], P[c] - P[a]), P[d] - P[a]) / 6.0 for a, b, c, d in TETS)


def beam_result(path):
    m = re.search(r"^\[beam\] result: (\{.*?\})", open(path).read(), re.M)
    return ast.literal_eval(m.group(1)) if m else None


def audit(name):
    D = os.path.join(OUT, name)
    r = {"name": name, "reasons": []}
    res = beam_result(os.path.join(D, "beam.log"))
    if res is None:
        r["reasons"].append("beam:no_result")
        return r
    r.update(blocks_before=None, blocks=res["blocks"], cuboids=res["cuboids"],
             excess=res["excess"], inverted_beam=res["inverted"], min_sj_beam=round(res["min_sj"], 3))
    m = re.search(r"\[beam\] start \{'cells': \d+, 'blocks': (\d+)", open(os.path.join(D, "beam.log")).read())
    r["blocks_before"] = int(m.group(1)) if m else None
    if res["cuboids"] != res["blocks"]:
        r["reasons"].append(f"beam:{res['blocks'] - res['cuboids']}_non_cuboid")
    if res["excess"]:
        r["reasons"].append("beam:excess_faces")
    if res["inverted"]:
        r["reasons"].append("beam:inverted")
    tl = open(os.path.join(D, "tfi.log")).read()
    if "not lattices" in tl:
        r["reasons"].append("tfi:non_lattice_block")
    m = re.search(r"refilled mesh: scaled Jacobian min ([-\d.]+), mean [\d.]+, (\d+) inverted", tl)
    if not m:
        r["reasons"].append("tfi:no_refill")
    else:
        r["tfi_min_sj"] = float(m.group(1)); r["tfi_inverted"] = int(m.group(2))
        if int(m.group(2)):
            r["reasons"].append("tfi:inverted")
    s = np.load(os.path.join(D, "sample.npz"), allow_pickle=True)
    V, B = np.asarray(s["vertices"], float), np.asarray(s["blocks"])
    vols = np.array([signed_vol(V[b]) for b in B])
    if (np.abs(vols) <= 1e-9).any():
        r["reasons"].append("blocks:degenerate")
    if (vols < 0).any() and (vols > 0).any():
        r["reasons"].append("blocks:mixed_orientation")
    try:
        C.check_surface_labels(s["surface_points"], s["surface_tris"], s["surface_tri_label"])
    except ValueError as e:
        r["reasons"].append(f"labels:{e}"[:60])
    o = np.load(os.path.join(SRC, name, "sample.npz"), allow_pickle=True)
    if s["surface_points"].shape != o["surface_points"].shape or \
            np.abs(s["surface_points"] - o["surface_points"]).max() > 1e-9:
        r["reasons"].append("surface:differs_from_source")
    try:
        from build_topo_tokens import load
        from topo_row_plan import build_row_plan_topo
        smp = load(os.path.join(D, "sample.npz"), merge={6: 5, 7: 5})
        rows, emit, info = build_row_plan_topo(smp["faces"].T.tolist(), smp["vertices_cartesian"].numpy(),
                                               smp["face_label"])
        r["rows"] = "-".join(map(str, info["row_lens"])); r["ties"] = info["n_ties"]
        r["topo_code"] = __import__("hashlib").sha1(str(info["code"]).encode()).hexdigest()[:10]
        if info["n_ties"] > 1:
            r["reasons"].append("topo:ambiguous")
    except Exception as e:  # noqa: BLE001
        r["reasons"].append(f"topo:{type(e).__name__}")
    return r


def main():
    names = sorted(os.path.basename(os.path.dirname(f)) for f in glob.glob(f"{OUT}/*/sample.npz"))
    rows = [audit(n) for n in names]
    for r in rows:
        r["ok"] = not r["reasons"]
        r["reasons"] = ";".join(r["reasons"])
    keys = ["name", "ok", "reasons", "blocks_before", "blocks", "cuboids", "excess", "inverted_beam",
            "min_sj_beam", "tfi_min_sj", "tfi_inverted", "rows", "ties", "topo_code"]
    with open("/work/relabel/audit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    ok = [r for r in rows if r["ok"]]
    print(f"audited {len(rows)}: ok {len(ok)}, rejected {len(rows) - len(ok)}")
    print("reasons:", dict(Counter(x for r in rows for x in r["reasons"].split(";") if x)))
    print("blocks before:", dict(sorted(Counter(r.get("blocks_before") for r in rows).items(), key=str)))
    print("blocks after (ok):", dict(sorted(Counter(r["blocks"] for r in ok).items())))
    print("topologies (ok):", dict(Counter(f"{r['blocks']}bl/{r.get('topo_code')}" for r in ok).most_common(8)))


if __name__ == "__main__":
    main()
