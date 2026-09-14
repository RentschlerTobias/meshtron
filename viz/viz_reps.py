"""
viz_reps.py

Repraesentanten-Bilder der beiden Face-Klassen, die im gefilterten Trainingssatz
bleiben: 6-Face (trivialer O-Grid, 80% des 10k-Satzes) und 12-Face (naechste Stufe).
Rest (11,13,14,...,35 Faces) wird rausgekickt.

Zeichnet die ECHTEN gekruemmten Blockkanten (edge_to_streamline) + rote
Parallelkanten (Blade Druck/Saug).

Ausgabe: figures/reps/
  - rep_6faces.png
  - rep_12faces.png
  - reps_side_by_side.png
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

from viz_outliers import draw_mesh_true   # zeichnet echte Kurven + rote Parallelkante

OUT = "figures/reps"
os.makedirs(OUT, exist_ok=True)


def main():
    print("Lade domain_data_10k.pt ...")
    data = torch.load("domain_data_10k.pt", weights_only=False)
    nf = np.array([d["faces"].shape[1] for d in data])
    nv = np.array([d["vertices_polar"].shape[0] for d in data])

    n_keep = int(((nf == 6) | (nf == 12)).sum())
    print(f"Filter: behalte nur 6- & 12-Face -> {n_keep}/{len(nf)} Meshes "
          f"({100*n_keep/len(nf):.1f}%);  6f={int((nf==6).sum())}  12f={int((nf==12).sum())}")

    picks = {}
    for F in (6, 12):
        idxs = np.where(nf == F)[0]
        picks[F] = int(idxs[np.argmax(nv[idxs])])   # Repraesentant mit meisten Verts

    # Einzelbilder
    for F, idx in picks.items():
        d = data[idx]
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        draw_mesh_true(ax, d,
                       title=f"Repraesentant {F}-Face  (idx {idx}, "
                             f"{d['vertices_polar'].shape[0]} Verts)\n"
                             "blau = Block-/Innenkante,  ROT = Parallelkante (Blade Druck/Saug)")
        fig.savefig(f"{OUT}/rep_{F}faces.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote rep_{F}faces.png")

    # Side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, F in zip(axes, (6, 12)):
        d = data[picks[F]]
        draw_mesh_true(ax, d,
                       title=f"{F} Faces / {d['vertices_polar'].shape[0]} Verts "
                             f"(idx {picks[F]})")
    fig.suptitle("Gefilterter Trainingssatz: nur 6- & 12-Face-Meshes (echte gekrümmte Kanten)\n"
                 f"behalten {n_keep}/{len(nf)} Meshes; Rest (11/13/14/.../35 Faces) verworfen",
                 fontsize=12)
    fig.savefig(f"{OUT}/reps_side_by_side.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote reps_side_by_side.png")
    print("\nFertig ->", OUT)


if __name__ == "__main__":
    main()
