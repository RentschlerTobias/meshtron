"""Tokenize polytron 3D datasets into hexa-row token-id streams for training.

Output .pt: dict with train/val lists of {tokens: list[int], name, blocks}.
Split is on machine granularity: sample order in the source .pt already
groups original + n2 + n3 augmentations per geometry, so we hold out one
geometry family per few samples (index-based % split documented in report).
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from meshtron.data.hexa_row_tokenizer import HexaRowTokenizer  # noqa: E402 (repo-root on sys.path)

_DATA = None
_TOK = None
_COORDS = 'polar'  # fork-global: _proc_one liest hier; main() setzt vor Pool-Start
_FAMILY = False  # fork-global: Konditionierungsfelder je Item einbetten
_DIR_PREFIX = "batch"  # fork-global: Praefix fuer item["dir"]
_GEOMS = {}  # fork-global: name -> geom_id aus --geom-ids


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
        ids = _TOK.tokenize(s, coords=_COORDS)
    except Exception as e:
        return i, None, f"SKIP sample {i} ({s.get('name', '?')}): {type(e).__name__}: {e}"
    lv = s.get("subdiv_n", None)
    name = s.get("name", f"sample{i}")
    item = {"tokens": torch.tensor(ids, dtype=torch.long),
            "name": name,
            "blocks": int(s["faces"].shape[1]),
            "level": int(lv) if lv is not None and lv >= 1 else 1}
    if _FAMILY:
        item["dir"] = f"{_DIR_PREFIX}/{name}"
        item["coords"] = _COORDS
        item["surface_points"] = s.get("surface_points")
        item["is_blade"] = s.get("is_blade")
        item["is_band"] = s.get("is_band")
        item["geom_id"] = _GEOMS.get(name.rsplit("_n", 1)[0])
    return i, item, ""


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
    ap.add_argument("--only", type=int, default=-1,
                    help="nur diesen Index aus der geladenen Sample-Liste behalten "
                         "(-1=aus); bounds & train/val-split leiten sich daraus ab "
                         "(single-mesh overfit)")
    ap.add_argument("--coords", choices=("polar", "cart"), default="polar",
                    help="Vertex-Token-Koordinaten: polar (4/Vert) oder cart (3/Vert); "
                         "cart braucht vertices_cartesian im Sample")
    ap.add_argument("--family", action="store_true",
                    help="Konditionierungsfelder (dir/surface_points/is_blade/"
                         "is_band/geom_id/coords) je Item einbetten")
    ap.add_argument("--dir-prefix", default="batch",
                    help="Praefix des npz-Verzeichnisses fuer item['dir']; "
                         "npz liegt unter data/hex3d_algohex/<dir>/sample.npz")
    ap.add_argument("--geom-ids", default="data/geom_ids.json",
                    help="JSON name->geom_id; '_meta' wird ignoriert")
    args = ap.parse_args()

    global _DATA, _TOK, _COORDS, _FAMILY, _DIR_PREFIX, _GEOMS
    _COORDS = args.coords
    _FAMILY = bool(args.family)
    _DIR_PREFIX = args.dir_prefix
    _DATA = torch.load(args.src, weights_only=False)
    if isinstance(_DATA, dict) and "samples" in _DATA:
        _DATA = _DATA["samples"]
    if args.only >= 0:
        if args.only >= len(_DATA):
            raise IndexError(f"--only {args.only} ausserhalb [0, {len(_DATA)})")
        kept = _DATA[args.only]
        if not kept.get("name"):
            # Original-Index als Name sichern: Train-Src-Matching laeuft ueber
            # fallback-Namen f"sample{i}" mit voller Listenindexierung.
            kept["name"] = f"sample{args.only}"
        print(f"--only {args.only}: 1/{len(_DATA)} samples behalten "
              f"({kept.get('name', '?')})")
        _DATA = [kept]
    if args.coords == 'cart':
        missing = [s.get('name', '?') for s in _DATA
                   if s.get('vertices_cartesian') is None]
        if missing:
            raise ValueError(f"--coords cart: vertices_cartesian fehlt in samples: "
                             f"{missing}")
    if args.family:
        import os
        if os.path.exists(args.geom_ids):
            _GEOMS = json.loads(open(args.geom_ids).read())
            _GEOMS.pop("_meta", None)
        else:
            _GEOMS = {}
        missing = [s.get('name', '?') for s in _DATA
                   if s.get('surface_points') is None]
        if missing:
            raise ValueError(f"--family: surface_points fehlt in: {missing[:5]}")
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
    if streams and (not train or not val):
        only = streams[0]
        print(f"single-mesh overfit: train==val duplication ({only['name']})")
        train = [only]
        val = [only]
    npt = 3 if args.coords == 'cart' else 4
    payload = {"train": train, "val": val, "vocab": vocab,
               "r_bounds": rb, "z_bounds": zb,
               "coords": args.coords, "npt": npt}
    if _FAMILY:
        payload["family"] = f"{args.coords}-v1"
    torch.save(payload, args.out)
    print(f"wrote {args.out}: {len(train)} train / {len(val)} val ({nskip} skips)")


if __name__ == "__main__":
    main()
