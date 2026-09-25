"""
viz_augment_gallery.py

Galerie zufaelliger (augmentierter) Meshes: zieht zufaellige 6-Face-Blades und
unterteilt jedes mit zufaelligem n in {1,2,3,4} (1 = Original). Zeigt die Faces
gefuellt; der weisse Schlitz = Blade-Loch bleibt offen.

  ~/Environments/meshtron/bin/python viz_augment_gallery.py \
      [--data domain_data_10k.pt] [--rows 4] [--cols 5] [--seed 0]
-> figures/augment/03_random_gallery.png
"""


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from meshtron.data.augment_subdivide import build_subdivided_mesh


def draw_filled(ax, d):
    vc = d['vertices_cartesian'].numpy()
    faces = d['faces'].numpy().T
    e2s = d['edge_to_streamline']
    cmap = plt.get_cmap('tab20')
    for fi, q in enumerate(faces):
        loop = []
        for k in range(4):
            u, v = int(q[k]), int(q[(k + 1) % 4])
            c = e2s.get((u, v))
            if c is None:
                c = e2s.get((v, u))
                c = np.asarray(c, float)[::-1] if c is not None else np.stack([vc[u], vc[v]])
            else:
                c = np.asarray(c, float)
            loop.append(c)
        poly = np.vstack(loop)
        ax.fill(poly[:, 0], poly[:, 1], color=cmap(fi % 20), alpha=0.6, ec='k', lw=0.4)
    ax.set_aspect('equal'); ax.axis('off')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='domain_data_10k.pt')
    ap.add_argument('--rows', type=int, default=4)
    ap.add_argument('--cols', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ns', type=int, nargs='+', default=[1, 2, 3, 4])
    ap.add_argument('--no-smooth', action='store_true', help='C1-Glaettung aus')
    ap.add_argument('--out', default='figures/augment/03_random_gallery.png')
    args = ap.parse_args()
    smooth = not args.no_smooth

    random.seed(args.seed); np.random.seed(args.seed)
    print(f"Lade {args.data} ...")
    data = torch.load(args.data, weights_only=False)
    six = [d for d in data if d['faces'].shape[1] == 6]
    print(f"6-Face-Meshes: {len(six)}")

    k = args.rows * args.cols
    picks = random.sample(range(len(six)), min(k, len(six)))

    fig, axes = plt.subplots(args.rows, args.cols, figsize=(2.9 * args.cols, 2.9 * args.rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, idx in zip(axes, picks):
        n = random.choice(args.ns)
        d = six[idx] if n == 1 else build_subdivided_mesh(six[idx], n, smooth=smooth)
        draw_filled(ax, d)
        F = d['faces'].shape[1]
        ax.set_title(f"#{idx}  n={n}  ({F}F)", fontsize=9)
    for ax in axes[len(picks):]:
        ax.axis('off')

    fig.suptitle(f"Zufaellige (augmentierte) Meshes — n in {args.ns}, weiss = Blade-Loch",
                 fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches='tight')
    print("wrote", args.out)


if __name__ == '__main__':
    main()
