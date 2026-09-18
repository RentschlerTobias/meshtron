"""Tokenize polytron 3D datasets into hexa-row token-id streams for training.

Output .pt: dict with train/val lists of {tokens: list[int], name, blocks}.
Split is on machine granularity: sample order in the source .pt already
groups original + n2 + n3 augmentations per geometry, so we hold out one
geometry family per few samples (index-based % split documented in report).
"""
import argparse
import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from hexa_row_tokenizer import HexaRowTokenizer  # noqa: E402 (repo-root on sys.path)

_DATA = None
_TOK = None


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


def _proc_one(i):
    s = _DATA[i]
    try:
        ids = _TOK.tokenize(s)
    except Exception as e:
        return i, None, f"SKIP sample {i} ({s.get('name', '?')}): {type(e).__name__}: {e}"
    lv = s.get("subdiv_n", None)
    return i, {"tokens": torch.tensor(ids, dtype=torch.long),
               "name": s.get("name", f"sample{i}"),
               "blocks": int(s["faces"].shape[1]),
               "level": int(lv) if lv is not None and lv >= 1 else 1}, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/polytron_data_3d_aug.pt")
    ap.add_argument("--out", default="data/hexarow_tokens_3d_aug.pt")
    ap.add_argument("--val-frac", type=float, default=1 / 7)
    ap.add_argument("--jobs", type=int, default=min(24, mp.cpu_count()))
    ap.add_argument("--max-blocks", type=int, default=0,
                    help="skip samples with more blocks than this (0=off)")
    ap.add_argument("--ckpt-out", default=None,
                    help="incremental checkpoint .pt (name -> stream dict); "
                         "finished samples are skipped on resume")
    args = ap.parse_args()

    global _DATA, _TOK
    _DATA = torch.load(args.src, weights_only=False)
    if isinstance(_DATA, dict) and "samples" in _DATA:
        _DATA = _DATA["samples"]
    if args.max_blocks:
        before = len(_DATA)
        _DATA = [s for s in _DATA if int(s["faces"].shape[1]) <= args.max_blocks]
        print(f"--max-blocks={args.max_blocks}: {len(_DATA)}/{before} samples kept")
    rb, zb = bounds_from(_DATA)
    print(f"r_bounds={tuple(round(v, 4) for v in rb)} z_bounds={tuple(round(v, 4) for v in zb)}")
    _TOK = HexaRowTokenizer(r_bounds=rb, z_bounds=zb)
    vocab = _TOK.core.vocab_size
    print("vocab size:", vocab)

    done = {}
    if args.ckpt_out:
        import os
        if os.path.exists(args.ckpt_out):
            done = torch.load(args.ckpt_out, weights_only=False)
            print(f"resume: {len(done)} samples already tokenized")
        print(f"checkpoint every sample -> {args.ckpt_out}")

    results = [None] * len(_DATA)
    if args.ckpt_out:
        for i, s in enumerate(_DATA):
            nm = s.get("name", "?")
            if nm in done:
                results[i] = done[nm]
    nskip = 0
    run = [(i, s) for i, s in enumerate(_DATA) if s.get("name", "?") not in done]
    ctx = mp.get_context("fork")
    with ctx.Pool(args.jobs) as pool:
        asyncs = [pool.apply_async(_proc_one, (i,)) for i, s in run]
        for n, a in enumerate(asyncs, 1):
            i, stream, msg = a.get()
            if stream is not None and args.ckpt_out:
                import os, tempfile
                done[stream["name"]] = stream
                tmp = tempfile.mktemp(dir=os.path.dirname(args.ckpt_out))
                torch.save(done, tmp)
                os.replace(tmp, args.ckpt_out)
            results[i] = stream if stream is not None else msg
            if msg:
                print(msg)
                nskip += 1
            else:
                print(f"  [{n}/{len(run)}] {stream['name']}: "
                      f"{len(stream['tokens'])} tok, {stream['blocks']} blocks",
                      flush=True)

    streams = [r for r in results if isinstance(r, dict)]
    machines = sorted({s["name"].rsplit("_n", 1)[0] for s in streams})
    val_machines = set(machines[-max(1, int(len(machines) * args.val_frac)):])
    train = [s for s in streams if s["name"].rsplit("_n", 1)[0] not in val_machines]
    val = [s for s in streams if s["name"].rsplit("_n", 1)[0] in val_machines]
    torch.save({"train": train, "val": val, "vocab": vocab,
                "r_bounds": rb, "z_bounds": zb}, args.out)
    print(f"wrote {args.out}: {len(train)} train / {len(val)} val ({nskip} skips)")


if __name__ == "__main__":
    main()
