"""Tokenize polytron 3D datasets into hexa-row token-id streams for training.

Output .pt: dict with train/val lists of {tokens: list[int], name, blocks}.
Split is on machine granularity: sample order in the source .pt already
groups original + n2 + n3 augmentations per geometry, so we hold out one
geometry family per few samples (index-based % split documented in report).
"""
import argparse
import sys

import numpy as np
import torch

from hexa_row_tokenizer import HexaRowTokenizer


def bounds_from(data, pad=0.02):
    rs, zs = [], []
    for s in data:
        vp = np.array(s["vertices_polar"], dtype=np.float64)
        rs.append((vp[:, 0].min(), vp[:, 0].max()))
        zs.append((vp[:, 2].min(), vp[:, 2].max()))
    rmin = min(a for a, b in rs)
    rmax = max(b for a, b in rs)
    zmin = min(a for a, b in zs)
    zmax = max(b for a, b in zs)
    return (rmin - (rmax - rmin) * pad, rmax + (rmax - rmin) * pad), \
           (zmin - (zmax - zmin) * pad, zmax + (zmax - zmin) * pad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/polytron_data_3d_aug.pt")
    ap.add_argument("--out", default="data/hexarow_tokens_3d_aug.pt")
    ap.add_argument("--val-frac", type=float, default=1 / 7)
    args = ap.parse_args()

    data = torch.load(args.src, weights_only=False)
    rb, zb = bounds_from(data)
    print(f"r_bounds={tuple(round(v, 4) for v in rb)} z_bounds={tuple(round(v, 4) for v in zb)}")
    tok = HexaRowTokenizer(r_bounds=rb, z_bounds=zb)
    vocab = tok.core.vocab_size
    print("vocab size:", vocab)

    streams = []
    nskip = 0
    for i, s in enumerate(data):
        try:
            ids = tok.tokenize(s)
        except Exception as e:
            print(f"SKIP sample {i} ({s.get('name', '?')}): {type(e).__name__}: {e}")
            continue
        streams.append({"tokens": torch.tensor(ids, dtype=torch.long),
                        "name": s.get("name", f"sample{i}"),
                        "blocks": int(s["faces"].shape[1])})
        print(f"  sample {i}: blocks={streams[-1]['blocks']} tokens={len(ids)}")

    n_val = max(1, int(len(streams) * args.val_frac))
    val_idx = set(range(len(streams) - n_val, len(streams)))
    train = [s for j, s in enumerate(streams) if j not in val_idx]
    val = [s for j, s in enumerate(streams) if j in val_idx]
    torch.save({"train": train, "val": val, "vocab": vocab,
                "r_bounds": rb, "z_bounds": zb}, args.out)
    print(f"wrote {args.out}: {len(train)} train / {len(val)} val ({nskip} skips)")


if __name__ == "__main__":
    main()
