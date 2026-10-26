"""Tokenize sample.npz block structures with the topology-canonical row plan.

Same output format as meshtron's scripts/build_hexarow_tokens.py
({train, val, vocab, r_bounds, z_bounds, coords, npt}, items {tokens, name,
blocks, level}), plus per-item `topo` (row lengths, code hash, ties) so
label stability can be audited. Every sample is round-tripped through
HexaRowTokenizer.detokenize and must give back its blocks.

  python build_topo_tokens.py --out tok.pt [--coords cart] [--no-labels] \
      path/to/*/sample.npz ...
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/opt/stack/meshtron")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meshtron.data.domain_extractor_3d import to_cylindrical  # noqa: E402
from meshtron.data.hexa_row_tokenizer import HexaRowTokenizer  # noqa: E402
from topo_row_plan import build_row_plan_topo, face_labels_from_npz  # noqa: E402

_TETS = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]


def signed_vol(P):
    return sum(np.dot(np.cross(P[b] - P[a], P[c] - P[a]), P[d] - P[a]) / 6.0 for a, b, c, d in _TETS)


def load(path, labels=True, merge=None):
    s = np.load(path, allow_pickle=True)
    V = np.asarray(s["vertices"], dtype=np.float64)
    blks = np.asarray(s["blocks"], dtype=np.int64).copy()
    for i in range(len(blks)):          # same orientation fix as clean_base_npz.py
        if signed_vol(V[blks[i]]) < 0:
            blks[i] = blks[i][::-1]
    name = os.path.basename(os.path.dirname(os.path.abspath(path)))
    fl = face_labels_from_npz(s, blks.tolist(), V, merge) if labels else None
    return {"name": name, "blocks": len(blks), "vertices_cartesian": torch.tensor(V),
            "vertices_polar": torch.tensor(to_cylindrical(V), dtype=torch.float32),
            "faces": torch.tensor(blks.T), "face_label": fl}


def roundtrip_ok(tok, toks, s, coords):
    vpt, blk = tok.detokenize(toks, coords=coords)
    if coords != "cart":
        r, th, z = vpt[:, 0], vpt[:, 1], vpt[:, 2]
        vpt = torch.stack([r * torch.cos(th), r * torch.sin(th), z], 1)
    V = s["vertices_cartesian"].numpy()
    orig = np.array([V[b].mean(0) for b in s["faces"].T.tolist()])
    back = np.array([vpt[b].numpy().mean(0) for b in blk.tolist()])
    if len(orig) != len(back):
        return False, np.inf, False
    # nearest-centroid match both ways; a permutation is fine, a lost block is not
    d = np.linalg.norm(orig[:, None] - back[None], axis=-1)
    err = max(d.min(1).max(), d.min(0).max())
    bijective = len(set(d.argmin(1))) == len(orig)
    pos = all(signed_vol(vpt[b].numpy().astype(np.float64)) > 0 for b in blk.tolist())
    return bijective and err < 0.02, err, pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--coords", choices=("polar", "cart"), default="cart")
    ap.add_argument("--no-labels", action="store_true")
    ap.add_argument("--merge-cut-labels", action="store_true",
                    help="fold the three cut interfaces (5 BL hub, 6 BL shroud, 7 O-grid) into one label")
    ap.add_argument("--val-frac", type=float, default=1 / 7)
    a = ap.parse_args()

    merge = {6: 5, 7: 5} if a.merge_cut_labels else None
    data = [load(p, labels=not a.no_labels, merge=merge) for p in sorted(a.npz)]
    vp = torch.cat([s["vertices_polar"] for s in data])
    rb = (float(vp[:, 0].min()), float(vp[:, 0].max())); zb = (float(vp[:, 2].min()), float(vp[:, 2].max()))
    pr, pz = 0.02 * (rb[1] - rb[0]), 0.02 * (zb[1] - zb[0])
    rb, zb = (rb[0] - pr, rb[1] + pr), (zb[0] - pz, zb[1] + pz)
    tok = HexaRowTokenizer(r_bounds=rb, z_bounds=zb)

    items, fails = [], 0
    for s in data:
        try:
            rows, emit, info = build_row_plan_topo(s["faces"].T.tolist(), s["vertices_cartesian"].numpy(),
                                                   face_label=s["face_label"])
            toks = tok.tokenize(s, emit_override=(rows, emit), coords=a.coords)
            ok, err, pos = roundtrip_ok(tok, toks, s, a.coords)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {s['name']}: {type(e).__name__}: {e}"); fails += 1
            continue
        h = hashlib.sha1(str(info["code"]).encode()).hexdigest()[:10]
        print(f"{s['name']:26s} blocks={s['blocks']:3d} tok={len(toks):4d} rows={info['n_rows']} "
              f"lens={info['row_lens']} code={h} ties={info['n_ties']} "
              f"roundtrip={'ok' if ok else 'FAIL'} err={err:.4f} positive={pos}")
        fails += (not ok) or (not pos)
        items.append({"tokens": toks, "name": s["name"], "blocks": s["blocks"], "level": 0,
                      "topo": {"row_lens": info["row_lens"], "code": h, "ties": info["n_ties"]}})

    machines = sorted({it["name"].rsplit("_n", 1)[0] for it in items})
    val_m = set(machines[-max(1, int(len(machines) * a.val_frac)):])
    out = {"train": [it for it in items if it["name"].rsplit("_n", 1)[0] not in val_m],
           "val": [it for it in items if it["name"].rsplit("_n", 1)[0] in val_m],
           "vocab": tok.core.vocab_size, "r_bounds": rb, "z_bounds": zb, "coords": a.coords,
           "npt": 3 if a.coords == "cart" else 4, "row_order": "topo"}
    torch.save(out, a.out)
    print(f"wrote {a.out}: {len(out['train'])} train / {len(out['val'])} val, {fails} failures")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
