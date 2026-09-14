"""
viz_pipeline_doc.py

Ausfuehrliche Schritt-fuer-Schritt-Dokumentation der Polytron-Pipeline (HO-Quad-
Block-Generator). Erzeugt eine Galerie, die jeden Prozessschritt visualisiert:

  0  Uebersicht (Punktwolke -> S1 Vertices -> S2 Faces -> S3 Geometrie -> Mesh)
  1  Eingabe: Punktwolke tri_coordinates, gefaerbt nach Label (Ecke/Rand/Feld)
  2  Stufe 1: eindeutige Vertices (polar r,theta), kanonische Sortier-Reihenfolge
  3  Stufe 2: Pointer-Faces = O-Grid-Blockstruktur (6 Bloecke, Blade-Loch offen)
  4  Stufe 3: HO-Kantengeometrie = cubic_bezier-Kontrollpunkte an einer Kante
  5  Rekonstruktion: gefuelltes Mesh mit gekruemmten Kanten
  6  Daten-Augmentation: 6F-Original + transfinite Unterteilung n=2,3,4

  ~/Environments/meshtron/bin/python viz_pipeline_doc.py \
      [--data domain_data_10k.pt] [--idx 0] -> figures/pipeline_doc/*.png
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from polytron_tokenizer import PolytronTokenizer
from augment_subdivide import build_subdivided_mesh

LAB_COLOR = {0: '#d62728', 1: '#ff7f0e', 2: '#7fb3d5'}
LAB_NAME = {0: 'Label 0: Ecke/Singularitaet', 1: 'Label 1: Rand', 2: 'Label 2: Feld'}


# ---- Bausteine -------------------------------------------------------
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
    for lab in (2, 1, 0):                                 # Feld zuerst, Ecken oben
        m = tri[:, 2] == lab
        ax.scatter(tri[m, 0], tri[m, 1], s=(30 if lab == 0 else 8),
                   c=LAB_COLOR[lab], edgecolors='k' if lab == 0 else 'none',
                   linewidths=0.6, zorder=3 if lab == 0 else 1, label=LAB_NAME[lab])
    ax.set_aspect('equal'); ax.axis('off')


def draw_vertices(ax, d, tok):
    tri = d['tri_coordinates'].numpy()
    ax.scatter(tri[:, 0], tri[:, 1], s=5, c='#cccccc', zorder=1)
    vc = d['vertices_cartesian'].numpy()
    order = tok._sort_order(d['vertices_polar'])
    for rank, oi in enumerate(order):
        ax.scatter(*vc[oi], s=70, c='#1f77b4', edgecolors='k', linewidths=0.8, zorder=3)
        ax.annotate(str(rank), vc[oi], fontsize=7, color='white', ha='center',
                    va='center', zorder=4)
    ax.set_aspect('equal'); ax.axis('off')


def draw_skeleton(ax, d):
    """S2: Blockstruktur mit GERADEN Kanten (Topologie, vor Geometrie)."""
    vc = d['vertices_cartesian'].numpy(); faces = d['faces'].numpy().T
    cmap = plt.get_cmap('tab20')
    for fi, q in enumerate(faces):
        poly = vc[[int(q[0]), int(q[1]), int(q[2]), int(q[3])]]
        ax.fill(poly[:, 0], poly[:, 1], color=cmap(fi % 20), alpha=0.5, zorder=1)
    for fi, q in enumerate(faces):
        for k in range(4):
            u, v = int(q[k]), int(q[(k + 1) % 4])
            ax.plot(*zip(vc[u], vc[v]), color='k', lw=1.0, zorder=2)
    ax.scatter(vc[:, 0], vc[:, 1], s=40, c='#1f77b4', edgecolors='k',
               linewidths=0.7, zorder=3)
    ax.set_aspect('equal'); ax.axis('off')


def draw_curved(ax, d, fill=True):
    """Gekruemmte Kanten (S3 / finales Mesh)."""
    faces = d['faces'].numpy().T; cmap = plt.get_cmap('tab20')
    for fi, q in enumerate(faces):
        loop = [edge_curve(d, int(q[k]), int(q[(k + 1) % 4])) for k in range(4)]
        poly = np.vstack(loop)
        if fill:
            ax.fill(poly[:, 0], poly[:, 1], color=cmap(fi % 20), alpha=0.6,
                    ec='k', lw=0.6)
        else:
            ax.plot(poly[:, 0], poly[:, 1], color='k', lw=0.9)
    ax.set_aspect('equal'); ax.axis('off')


def draw_stage3_detail(ax, d, tok):
    """Eine Kante: Endpunkte P0/P1, Kontrollpunkte B1/B2, Sehne, cubic_bezier vs
    echte Streamline."""
    vc = d['vertices_cartesian'].numpy(); faces = d['faces'].numpy()
    e2s = d['edge_to_streamline']
    # laengste gekruemmte Kante waehlen
    best = None; best_len = -1
    F = faces.shape[1]
    for fi in range(F):
        for k in range(4):
            u, v = int(faces[k, fi]), int(faces[(k + 1) % 4, fi])
            pts = e2s.get((u, v))
            if pts is None:
                continue
            pts = np.asarray(pts, float)
            dev = np.linalg.norm(pts - np.linspace(pts[0], pts[-1], len(pts)), axis=1).max()
            if dev > best_len:
                best_len = dev; best = (u, v, pts)
    u, v, pts = best
    P0, P1 = vc[u], vc[v]
    B1, B2 = tok._fit_cubic_bezier(P0, P1, pts)
    uh, nh, L = tok._chord_frame(P0, P1)
    t = np.linspace(0, 1, 60)[:, None]
    curve = ((1 - t) ** 3 * P0 + 3 * (1 - t) ** 2 * t * B1 +
             3 * (1 - t) * t ** 2 * B2 + t ** 3 * P1)
    ax.plot([P0[0], P1[0]], [P0[1], P1[1]], '--', color='gray', lw=1.0, label='Sehne')
    ax.plot(pts[:, 0], pts[:, 1], color='#2ca02c', lw=3, alpha=0.5, label='echte Streamline')
    ax.plot(curve[:, 0], curve[:, 1], color='#d62728', lw=1.4, label='cubic_bezier')
    for P, name, col in ((P0, 'P0', 'k'), (P1, 'P1', 'k')):
        ax.scatter(*P, s=60, c=col, zorder=4); ax.annotate(name, P, fontsize=9,
                                                           xytext=(4, 4), textcoords='offset points')
    for B, name in ((B1, 'B1'), (B2, 'B2')):
        ax.scatter(*B, s=60, c='#d62728', marker='s', zorder=4)
        ax.annotate(name, B, fontsize=9, xytext=(4, 4), textcoords='offset points',
                    color='#d62728')
        ax.plot([P0[0] if name == 'B1' else P1[0], B[0]],
                [P0[1] if name == 'B1' else P1[1], B[1]], ':', color='#d62728', lw=0.8)
    ax.legend(fontsize=7, loc='best'); ax.set_aspect('equal'); ax.axis('off')


# ---- Galerie ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='domain_data_10k.pt')
    ap.add_argument('--idx', type=int, default=0)
    ap.add_argument('--outdir', default='figures/pipeline_doc')
    args = ap.parse_args()

    tok = PolytronTokenizer(repr_mode='cubic_bezier')
    print(f"Lade {args.data} ...")
    data = torch.load(args.data, weights_only=False)
    six = [d for d in data if d['faces'].shape[1] == 6]
    d = six[args.idx]
    os.makedirs(args.outdir, exist_ok=True)

    _, meta = tok.tokenize(d)
    M, F = meta['M'], meta['F']
    print(f"Mesh #{args.idx}: M={M} Vertices, F={F} Faces, "
          f"Np={d['tri_coordinates'].shape[0]} Punkte")

    # ---- 0 Uebersicht (5 Panels) ----
    fig, ax = plt.subplots(1, 5, figsize=(21, 4.4))
    draw_pointcloud(ax[0], d); ax[0].set_title("Eingabe\nPunktwolke + Label", fontsize=11)
    draw_vertices(ax[1], d, tok); ax[1].set_title("Stufe 1\nVertices (r, theta)", fontsize=11)
    draw_skeleton(ax[2], d); ax[2].set_title("Stufe 2\nPointer-Faces (O-Grid)", fontsize=11)
    draw_stage3_detail(ax[3], d, tok); ax[3].set_title("Stufe 3\nHO-Kantengeometrie", fontsize=11)
    draw_curved(ax[4], d); ax[4].set_title("Ausgabe\nHO-Quad-Mesh", fontsize=11)
    for i in range(4):
        ax[i].annotate("", xy=(1.02, 0.5), xytext=(0.98, 0.5), xycoords='axes fraction',
                       arrowprops=dict(arrowstyle='-|>', color='k', lw=1.6))
    fig.suptitle("Polytron Pipeline: punktwolken-konditionierter HO-Quad-Block-Generator",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    p = f"{args.outdir}/00_overview.png"; fig.savefig(p, dpi=110, bbox_inches='tight')
    print("wrote", p); plt.close(fig)

    # ---- 1 Punktwolke (gross, mit Legende) ----
    fig, a = plt.subplots(figsize=(6, 6))
    draw_pointcloud(a, d)
    a.legend(loc='upper right', fontsize=8, framealpha=0.9)
    a.set_title("Eingabe: Punktwolke (tri_coordinates), gefaerbt nach Label\n"
                "6 Ecken (Label 0) = exakte Block-Vertices", fontsize=11)
    fig.tight_layout(); p = f"{args.outdir}/01_pointcloud.png"
    fig.savefig(p, dpi=120, bbox_inches='tight'); print("wrote", p); plt.close(fig)

    # ---- 2 Stufe 1 Vertices ----
    fig, a = plt.subplots(figsize=(6, 6))
    draw_vertices(a, d, tok)
    a.set_title(f"Stufe 1: {M} eindeutige Vertices, kanonische Reihenfolge lexsort(theta, r)\n"
                "Zahl = Sortier-Rang = Token-Reihenfolge (r, sin, cos)", fontsize=10)
    fig.tight_layout(); p = f"{args.outdir}/02_stage1_vertices.png"
    fig.savefig(p, dpi=120, bbox_inches='tight'); print("wrote", p); plt.close(fig)

    # ---- 3 Stufe 2 Topologie ----
    fig, a = plt.subplots(figsize=(6, 6))
    draw_skeleton(a, d)
    a.set_title(f"Stufe 2: Pointer-Faces = {F} Bloecke (O-Grid um Blade-Loch)\n"
                "gerade Kanten = reine Topologie (4 Zeiger je Face)", fontsize=10)
    fig.tight_layout(); p = f"{args.outdir}/03_stage2_topology.png"
    fig.savefig(p, dpi=120, bbox_inches='tight'); print("wrote", p); plt.close(fig)

    # ---- 4 Stufe 3 Kanten-Detail ----
    fig, a = plt.subplots(figsize=(6, 6))
    draw_stage3_detail(a, d, tok)
    a.set_title("Stufe 3: HO-Kantengeometrie\ncubic_bezier mit 2 Kontrollpunkten "
                "(B1, B2) best-fit an Streamline", fontsize=10)
    fig.tight_layout(); p = f"{args.outdir}/04_stage3_edge_detail.png"
    fig.savefig(p, dpi=120, bbox_inches='tight'); print("wrote", p); plt.close(fig)

    # ---- 5 Rekonstruktion ----
    fig, a = plt.subplots(1, 2, figsize=(12, 6))
    draw_curved(a[0], d, fill=False); a[0].set_title("gekruemmte Kanten (Draht)", fontsize=10)
    draw_curved(a[1], d, fill=True); a[1].set_title("gefuelltes Mesh (Blade-Loch offen)", fontsize=10)
    fig.suptitle("Rekonstruktion: HO-Quad-Mesh aus Vertices + Faces + Geometrie", fontsize=12)
    fig.tight_layout(); p = f"{args.outdir}/05_reconstruction.png"
    fig.savefig(p, dpi=120, bbox_inches='tight'); print("wrote", p); plt.close(fig)

    # ---- 6 Augmentation ----
    fig, a = plt.subplots(1, 4, figsize=(18, 4.6))
    for c, n in enumerate([1, 2, 3, 4]):
        dm = d if n == 1 else build_subdivided_mesh(d, n, smooth=True)
        draw_curved(a[c], dm)
        Fn = dm['faces'].shape[1]
        a[c].set_title(f"{'Original' if n == 1 else f'n={n}'}  ({Fn} Faces)", fontsize=11)
    fig.suptitle("Daten-Augmentation: transfinite Coons-Unterteilung (Blade-Loch bleibt offen)",
                 fontsize=13)
    fig.tight_layout(); p = f"{args.outdir}/06_augmentation.png"
    fig.savefig(p, dpi=110, bbox_inches='tight'); print("wrote", p); plt.close(fig)

    print(f"\nGalerie fertig in {args.outdir}/ (7 Bilder).")


if __name__ == '__main__':
    main()
