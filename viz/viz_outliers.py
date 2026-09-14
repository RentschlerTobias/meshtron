"""
viz_outliers.py

Visualisiert die Ausreisser im Datensatz nach Face-Anzahl UND zeigt die
tatsaechliche (gekruemmte) Blockkanten-Geometrie aus edge_to_streamline.

Ausgabe: figures/outliers/
  - 00_distribution.png        Face-Count Histogramm (log)
  - gallery_*.png              3x3 Galerie ausgewaehlter Meshes (echte Kurven)
  - single_<idx>_<F>faces.png  Einzelbilder der Top-Ausreisser
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "figures/outliers"
os.makedirs(OUT, exist_ok=True)
C_EDGE = "#33475b"
C_CURVE = "#1f77b4"
C_PARALLEL = "#d62728"


def draw_mesh_true(ax, d, title=None, show_parallel=True):
    """Zeichnet die ECHTEN gekruemmten Blockkanten (edge_to_streamline)."""
    e2s = d["edge_to_streamline"]
    xy = d["vertices_cartesian"].numpy()

    # twin-Erkennung: (u,v) vs (v,u) reversal-gleich?
    drawn = set()
    for (u, v), pts in e2s.items():
        if (u, v) in drawn:
            continue
        a = np.asarray(pts, float)
        color, lw, z = C_CURVE, 1.3, 2
        if (v, u) in e2s:
            b = np.asarray(e2s[(v, u)], float)[::-1]
            twin = a.shape == b.shape and np.allclose(a, b, atol=1e-6)
            if twin:
                drawn.add((v, u))       # nicht doppelt zeichnen
            elif show_parallel:
                # echte Parallelkante (z.B. Druck/Saugseite) -> rot hervorheben
                color, lw, z = C_PARALLEL, 2.0, 4
        ax.plot(a[:, 0], a[:, 1], color=color, lw=lw, zorder=z)
        drawn.add((u, v))

    ax.scatter(xy[:, 0], xy[:, 1], s=16, color=C_EDGE, zorder=5)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)


def main():
    print("Lade domain_data_10k.pt ...")
    data = torch.load("domain_data_10k.pt", weights_only=False)
    nf = np.array([d["faces"].shape[1] for d in data])
    nv = np.array([d["vertices_polar"].shape[0] for d in data])

    # --- 1) Verteilung ---
    fig, ax = plt.subplots(figsize=(9, 4.5))
    u, c = np.unique(nf, return_counts=True)
    ax.bar(u, c, color=C_CURVE, width=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Anzahl Faces pro Mesh")
    ax.set_ylabel("Anzahl Meshes (log)")
    ax.set_title(f"Face-Count Verteilung im 10k-Datensatz (bimodal)\n"
                 f"6 Faces: {c[u==6][0]} Meshes ({100*c[u==6][0]/len(nf):.0f}%),  "
                 f"Rest: langer Schwanz bis {nf.max()} Faces")
    for val, cnt in zip(u, c):
        if cnt < 50 or val == 6:
            ax.text(val, cnt, str(cnt), ha="center", va="bottom", fontsize=6.5)
    fig.savefig(f"{OUT}/00_distribution.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote 00_distribution.png")

    # --- 2) Galerie: von typisch bis extrem ---
    # ein Repraesentant je Ziel-Facecount
    targets = [6, 14, 18, 20, 25, 30, 33, 34, 35]
    picks = []
    for t in targets:
        idxs = np.where(nf == t)[0]
        if len(idxs):
            picks.append(int(idxs[np.argmax(nv[idxs])]))  # der mit meisten Verts
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for ax, idx in zip(axes.flat, picks):
        d = data[idx]
        draw_mesh_true(ax, d,
                       title=f"idx {idx}:  {d['faces'].shape[1]} Faces, "
                             f"{d['vertices_polar'].shape[0]} Verts")
    for ax in axes.flat[len(picks):]:
        ax.axis("off")
    fig.suptitle("Ausreisser-Galerie: echte gekrümmte Blockkanten\n"
                 "blau = normale/interne Kante,  ROT = echte Parallelkante "
                 "(2 verschiedene Splines, gleiche Endpunkte)", fontsize=12)
    fig.savefig(f"{OUT}/gallery_by_facecount.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote gallery_by_facecount.png")

    # --- 3) Einzelbilder der 3 groessten Ausreisser ---
    top = np.argsort(nf)[-3:][::-1]
    for idx in top:
        idx = int(idx)
        d = data[idx]
        fig, ax = plt.subplots(figsize=(7, 7))
        draw_mesh_true(ax, d,
                       title=f"Ausreisser idx {idx}: {d['faces'].shape[1]} Faces, "
                             f"{d['vertices_polar'].shape[0]} Verts\n"
                             "rot = echte Parallelkante (Multigraph)")
        name = f"single_{idx}_{d['faces'].shape[1]}faces.png"
        fig.savefig(f"{OUT}/{name}", dpi=130, bbox_inches="tight")
        plt.close(fig)
        print("wrote", name)

    print("\nFertig ->", OUT)


if __name__ == "__main__":
    main()
