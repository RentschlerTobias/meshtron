"""
export_augmented.py

Erzeugt den augmentierten Trainingssatz: nur die 6-Face-Meshes aus dem Quelldatensatz,
je isotrop in n x n unterteilt (transfinite Coons, siehe augment_subdivide.py). Die
Blade-Lens bleibt ein Loch (nie unterteilt). Original + Augmentationen.

  ~/Environments/meshtron/bin/python export_augmented.py \
      --data domain_data_10k.pt --out domain_data_aug.pt --ns 2 3 4

Default (7988 6-Face x [orig + n=2,3,4]) -> ~31900 Meshes, ~2.5 GB, ~7 min.
"""


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import time
import torch

from augment_subdivide import build_subdivided_mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='domain_data_10k.pt')
    ap.add_argument('--out', default='domain_data_aug.pt')
    ap.add_argument('--ns', type=int, nargs='+', default=[2, 3, 4],
                    help='Unterteilungsfaktoren (isotrop n x n)')
    ap.add_argument('--no-original', action='store_true',
                    help='Original-6-Face NICHT behalten')
    ap.add_argument('--no-smooth', action='store_true',
                    help='C1-Kanten-Glaettung AUS (default: an)')
    ap.add_argument('--limit', type=int, default=None, help='nur erste K 6-Face (Test)')
    args = ap.parse_args()

    smooth = not args.no_smooth
    print(f"Lade {args.data} ...")
    data = torch.load(args.data, weights_only=False)
    six = [d for d in data if d['faces'].shape[1] == 6]
    if args.limit:
        six = six[:args.limit]
    print(f"6-Face-Meshes: {len(six)}  |  ns={args.ns}  "
          f"keep_original={not args.no_original}  smooth={smooth}")

    out = []
    t0 = time.time()
    for i, d in enumerate(six):
        if not args.no_original:
            out.append(d)
        for n in args.ns:
            out.append(build_subdivided_mesh(d, n, smooth=smooth))
        if (i + 1) % 500 == 0:
            el = time.time() - t0
            eta = el / (i + 1) * (len(six) - i - 1)
            print(f"  {i+1}/{len(six)}  ({len(out)} out)  {el:.0f}s  ETA {eta:.0f}s")

    print(f"Fertig: {len(out)} Meshes in {time.time()-t0:.0f}s. Speichere {args.out} ...")
    torch.save(out, args.out)
    from collections import Counter
    fc = Counter(int(d['faces'].shape[1]) for d in out)
    print("Facecount-Verteilung:", dict(sorted(fc.items())))
    print("done.")


if __name__ == '__main__':
    main()
