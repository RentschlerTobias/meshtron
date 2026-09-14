"""
viz_pointcloud_mesh.py

Plots: Eingabe-Punktwolke (tri_coordinates, gefaerbt nach Label) NEBEN dem
zugehoerigen 6-Face-Mesh (gefuellt, gekruemmte Kanten, Blade-Loch offen).
Mehrere Beispiele untereinander.

  ~/Environments/meshtron/bin/python viz_pointcloud_mesh.py \
      [--data domain_data_10k.pt] [--n 6] [--seed 0]
  -> figures/pointcloud_mesh/pairs.png
"""


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

LAB_COLOR = {0: '#d62728', 1: '#ff7f0e', 2: '#7fb3d5'}
LAB_NAME = {0: 'Ecke (Label 0)', 1: 'Rand (Label 1)', 2: 'Feld (Label 2)'}


def edge_curve(d, u, v):
    e2s = d['edge_to_streamline']; vc = d['vertices_cartesian'].numpy()
    c = e2s.get((u, v))
    if c is not None:
        return np.asarray(c, float)
    c = e2s.get((v, u))
    if c is not None:
        return np.asarray(c, float)[::-1]
    return np.stack([vc[u], vc[v]])


def draw_pointcloud(ax, d):
    tri = d['tri_coordinates'].numpy()
    for lab in (2, 1, 0):
        m = tri[:, 2] == lab
        ax.scatter(tri[m, 0], tri[m, 1], s=(34 if lab == 0 else 7),
                   c=LAB_COLOR[lab], edgecolors='k' if lab == 0 else 'none',
                   linewidths=0.6, zorder=3 if lab == 0 else 1, label=LAB_NAME[lab])
    ax.set_aspect('equal'); ax.axis('off')


def draw_mesh(ax, d):
    faces = d['faces'].numpy().T; cmap = plt.get_cmap('tab20')
    for fi, q in enumerate(faces):
        loop = [edge_curve(d, int(q[k]), int(q[(k + 1) % 4])) for k in range(4)]
        poly = np.vstack(loop)
        ax.fill(poly[:, 0], poly[:, 1], color=cmap(fi % 20), alpha=0.65, ec='k', lw=0.6)
    vc = d['vertices_cartesian'].numpy()
    ax.scatter(vc[:, 0], vc[:, 1], s=18, c='k', zorder=4)
    ax.set_aspect('equal'); ax.axis('off')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='domain_data_10k.pt')
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='figures/pointcloud_mesh/pairs.png')
    args = ap.parse_args()

    print(f"Lade {args.data} ...")
    data = torch.load(args.data, weights_only=False)
    six = [d for d in data if d['faces'].shape[1] == 6]
    print(f"6F-Meshes: {len(six)}")
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(six), size=min(args.n, len(six)), replace=False)

    n = len(picks)
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    axes = np.atleast_2d(axes)
    for r, idx in enumerate(picks):
        d = six[idx]
        draw_pointcloud(axes[r, 0], d)
        draw_mesh(axes[r, 1], d)
        Np = d['tri_coordinates'].shape[0]
        axes[r, 0].set_title(f"#{idx}  Punktwolke ({Np} Punkte)", fontsize=10)
        axes[r, 1].set_title(f"#{idx}  6-Face-Mesh (Blade-Loch offen)", fontsize=10)
    axes[0, 0].legend(loc='upper right', fontsize=7, framealpha=0.9)
    fig.suptitle("Eingabe-Punktwolke (Label-gefaerbt)  vs.  6-Face-Mesh", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches='tight')
    print("wrote", args.out)


if __name__ == '__main__':
    main()
