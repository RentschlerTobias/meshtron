"""Build the v2 training set from the beam-relabelled samples.

Per sample (only rows with ok=True in the relabel audit.csv, see _paths.py):
  tokens   HexaRow cart tokens in the topology-canonical row order
           (meshtron/data/topo_row_plan.py, --merge-cut-labels semantics)
  points   ALL surface points, [P, 4 + 7] float16:
           (r', sin th, cos th, z') + multi-hot of the incident surface labels
           1..7 (7 = O-grid cut = blade, see meshtron conditioning.SURFACE_LABELS)
  verts    GT vertex positions in canonical order = first appearance in the
           topo emission (for the corner-error metric)
  vtok     Polytron stage 1: [start] 3 cart tokens per vertex, each vertex ONCE,
           canonical order [stop]
  conn     Polytron stage 2: [B, 8] pointers into the canonical vertex list
           (block order = emission order; identical for one topology)
Split: geometry-disjoint (one machine = one geometry for the n2000 set),
seeded, val_frac of the machines.

    BLOCKGEN_SAMPLES=<relabel>/out python scripts/blockgen_v2/build_dataset.py \
        --out runs/blockgen_v2/data/v2_cart.pt [--topo 49a0142abb]
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths  # noqa: E402,F401  (puts the repo root on sys.path)
from meshtron.data import conditioning as C  # noqa: E402
from meshtron.data.hexa_row_tokenizer import HexaRowTokenizer  # noqa: E402
from meshtron.data.topo_row_plan import build_row_plan_topo  # noqa: E402
from meshtron.data.topo_row_plan import load_npz_sample as load  # noqa: E402


def point_labels(n, tris, tri_label):
    m = np.zeros((n, 7), np.float32)
    for lab in range(1, 8):
        sel = tri_label == lab
        if sel.any():
            m[np.unique(tris[sel]), lab - 1] = 1.0
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--audit", default=_paths.AUDIT)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topo", default=None, help="keep only this topo_code from audit.csv (e.g. 49a0142abb)")
    a = ap.parse_args()

    ok = [r["name"] for r in csv.DictReader(open(a.audit))
          if r["ok"] == "True" and (a.topo is None or r["topo_code"] == a.topo)]
    print(f"audit ok samples: {len(ok)}")
    samples = []
    for n in sorted(ok):
        path = os.path.join(_paths.need_samples(), n, "sample.npz")
        s = load(path, merge={6: 5, 7: 5})
        rows, emit, info = build_row_plan_topo(s["faces"].T.tolist(), s["vertices_cartesian"].numpy(),
                                               s["face_label"])
        raw = np.load(path, allow_pickle=True)
        C.check_surface_labels(raw["surface_points"], raw["surface_tris"], raw["surface_tri_label"])
        samples.append({"name": n, "s": s, "plan": (rows, emit), "info": info, "raw": raw})

    allv = np.vstack([np.asarray(x["raw"]["surface_points"]) for x in samples])
    r = np.hypot(allv[:, 0], allv[:, 1])
    pr, pz = 0.02 * np.ptp(r), 0.02 * np.ptp(allv[:, 2])
    rb = (float(r.min() - pr), float(r.max() + pr))
    zb = (float(allv[:, 2].min() - pz), float(allv[:, 2].max() + pz))
    tok = HexaRowTokenizer(r_bounds=rb, z_bounds=zb)

    items = []
    for x in samples:
        toks = tok.tokenize(x["s"], emit_override=x["plan"], coords="cart")
        vpt, blk = tok.detokenize(toks, coords="cart")
        if len(blk) != x["s"]["blocks"]:
            print("SKIP round trip", x["name"]); continue
        rows, emit = x["plan"]
        order, idx = [], {}
        conn = []
        for row in rows:
            for b in row:
                for v in emit[b]:
                    if v not in idx:
                        idx[v] = len(order); order.append(v)
                conn.append([idx[v] for v in emit[b]])
        Vc = x["s"]["vertices_cartesian"]
        vtok = [tok.core.start_token]
        for v in order:
            vtok += tok.quant_vertex_tokens(v, None, Vc, coords="cart")
        vtok.append(tok.core.stop_token)
        P = np.asarray(x["raw"]["surface_points"], np.float64)
        rr = np.hypot(P[:, 0], P[:, 1]); th = np.arctan2(P[:, 1], P[:, 0])
        geo = np.stack([(rr - rb[0]) / (rb[1] - rb[0]), np.sin(th), np.cos(th),
                        (P[:, 2] - zb[0]) / (zb[1] - zb[0])], 1)
        lab = point_labels(len(P), np.asarray(x["raw"]["surface_tris"]), np.asarray(x["raw"]["surface_tri_label"]))
        items.append({"name": x["name"], "machine": x["name"].rsplit("_n", 1)[0],
                      "tokens": torch.tensor(toks, dtype=torch.int16),
                      "points": torch.tensor(np.hstack([geo, lab]), dtype=torch.float16),
                      "verts": torch.tensor(Vc.numpy()[order], dtype=torch.float32),
                      "vtok": torch.tensor(vtok, dtype=torch.int16),
                      "conn": torch.tensor(conn, dtype=torch.int16),
                      "blocks": int(x["s"]["blocks"]),
                      "rows": "-".join(map(str, x["info"]["row_lens"]))})

    machines = sorted({it["machine"] for it in items})
    rng = np.random.default_rng(a.seed)
    val_m = set(rng.permutation(machines)[:max(1, int(round(len(machines) * a.val_frac)))])
    out = {"train": [it for it in items if it["machine"] not in val_m],
           "val": [it for it in items if it["machine"] in val_m],
           "vocab": tok.core.vocab_size, "r_bounds": rb, "z_bounds": zb, "coords": "cart", "npt": 3,
           "special": {"start": tok.core.start_token, "stop": tok.core.stop_token,
                       "sep": tok.core.sep_token, "pad": tok.core.pad_token},
           "row_order": "topo", "point_features": "r,sin,cos,z + multi-hot labels 1..7 (7 = O-grid cut)"}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(out, a.out)
    from collections import Counter
    print(f"wrote {a.out}: train {len(out['train'])} / val {len(out['val'])} | vocab {out['vocab']} | "
          f"tokens/sample {sorted(Counter(len(i['tokens']) for i in items).items())} | "
          f"points/sample {min(len(i['points']) for i in items)}..{max(len(i['points']) for i in items)} | "
          f"row structures {dict(Counter(i['rows'] for i in items))}")


if __name__ == "__main__":
    main()
