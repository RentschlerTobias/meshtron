"""
viz_blockstruct.py

Diagnose der 6-Face-Blockstruktur als Grundlage fuer die Daten-Augmentation
(transfinite Sub-Unterteilung der Bloecke, siehe Plan). Zeigt:
  - die 6 higher-order Bloecke mit ihren ECHTEN gekruemmten Kanten (edge_to_streamline),
  - Blockzentren mit Face-Index + lokale (u,v)-Orientierung (Pfeile),
  - Vertices nummeriert,
  - die Blade-Lens-Kante (Druck/Saug, nicht-invers) fett rot.

Aufruf: ~/Environments/meshtron/bin/python viz_blockstruct.py [--data domain_data.pt] [--idx N]
-> figures/augment/00_blockstruct.png
"""


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

OUT = "figures/augment"


def curve(e2s, u, v, P0, P1):
    """gekruemmte Kante u->v aus edge_to_streamline, Fallback Gerade."""
    c = e2s.get((u, v))
    if c is None:
        c = e2s.get((v, u))
        if c is not None:
            c = np.asarray(c, float)[::-1]
    if c is None:
        t = np.linspace(0, 1, 20)[:, None]
        return (1 - t) * P0 + t * P1
    return np.asarray(c, float)


def blade_edges(e2s, vc, thresh=0.02):
    """Menge gerichteter Kanten (u,v), deren Gegenrichtung NICHT die Inverse ist."""
    out = set()
    for (u, v) in e2s:
        if (v, u) in e2s:
            a = np.asarray(e2s[(u, v)], float)
            b = np.asarray(e2s[(v, u)], float)[::-1]
            if a.shape == b.shape and np.max(np.linalg.norm(a - b, axis=1)) > thresh:
                out.add((u, v)); out.add((v, u))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='domain_data.pt')
    ap.add_argument('--idx', type=int, default=None, help='Mesh-Index (default: erstes 6-Face)')
    args = ap.parse_args()

    data = torch.load(args.data, weights_only=False)
    if args.idx is None:
        idx = next(i for i, d in enumerate(data) if d['faces'].shape[1] == 6)
    else:
        idx = args.idx
    d = data[idx]
    vc = d['vertices_cartesian'].numpy()
    faces = d['faces'].numpy().T          # [F,4]
    e2s = d['edge_to_streamline']
    blade = blade_edges(e2s, vc)

    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 8))
    cmap = plt.get_cmap('tab10')

    for fi, quad in enumerate(faces):
        # Blockflaeche (dezent) + gekruemmte Raender
        loop = []
        for k in range(4):
            u, v = int(quad[k]), int(quad[(k + 1) % 4])
            c = curve(e2s, u, v, vc[u], vc[v])
            loop.append(c)
            is_blade = (u, v) in blade
            ax.plot(c[:, 0], c[:, 1],
                    color='crimson' if is_blade else cmap(fi % 10),
                    lw=3.4 if is_blade else 1.8,
                    zorder=4 if is_blade else 2)
        poly = np.vstack(loop)
        ax.fill(poly[:, 0], poly[:, 1], color=cmap(fi % 10), alpha=0.08, zorder=0)
        # Blockzentrum + Face-Index
        cen = vc[quad].mean(0)
        ax.text(cen[0], cen[1], f"F{fi}", fontsize=15, fontweight='bold',
                ha='center', va='center', color=cmap(fi % 10), zorder=6,
                bbox=dict(boxstyle='round', fc='white', ec=cmap(fi % 10), alpha=0.85))
        # lokale (u,v)-Orientierung: Pfeil entlang Kante0 (u) und Kante3 (v)
        p0 = vc[int(quad[0])]
        du = vc[int(quad[1])] - p0
        dv = vc[int(quad[3])] - p0
        ax.annotate('', p0 + 0.35 * du, p0, arrowprops=dict(arrowstyle='-|>', color='#333', lw=1.3), zorder=5)
        ax.annotate('', p0 + 0.35 * dv, p0, arrowprops=dict(arrowstyle='-|>', color='#777', lw=1.3), zorder=5)

    # Vertices
    for i, p in enumerate(vc):
        ax.scatter([p[0]], [p[1]], s=45, color='black', zorder=7)
        ax.annotate(f"v{i}", p, textcoords='offset points', xytext=(5, 5),
                    fontsize=9, zorder=8)

    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(f"6-Face-Blockstruktur (mesh {idx})\n"
                 "rot = Blade-Lens (Druck/Saug), grau/schwarz = lokale (v,u)-Achsen je Block",
                 fontsize=11)
    fig.tight_layout()
    path = f"{OUT}/00_blockstruct.png"
    fig.savefig(path, dpi=130, bbox_inches='tight')
    print("wrote", path)
    print("Blade-Kanten (gerichtet):", sorted(blade))


if __name__ == '__main__':
    main()
