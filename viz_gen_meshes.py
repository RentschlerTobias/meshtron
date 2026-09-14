"""
viz_gen_meshes.py

Laedt das beste (bereits trainierte) S1-Modell und generiert Meshes fuer
verschiedene Facecounts (n=1,2,3,4 -> 6/24/54/96 Faces). Pro Mesh 3 Plots:
  Punktwolke (Eingabe)  |  generiert  |  true (GT)
KEIN Training. S1 erzeugt nur die VERTICES; das "generierte" Mesh sind die
generierten Vertices auf der (deterministischen) O-Grid-Topologie des jeweiligen
Facecounts, gerade Kanten (S2/S3-Koepfe sind hier nicht geladen).

  ~/Environments/meshtron/bin/python viz_gen_meshes.py \
      --ckpt runs_planb/s1_var.pt --data domain_data_smoke.pt --per-fc 2
  -> figures/gen_meshes/gen_meshes.png
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from polytron_tokenizer import PolytronTokenizer
import polytron_vertex_model as vh

LAB_COLOR = {0: '#d62728', 1: '#ff7f0e', 2: '#7fb3d5'}


def draw_pointcloud(ax, d):
    tri = d['tri_coordinates'].numpy()
    for lab in (2, 1, 0):
        m = tri[:, 2] == lab
        ax.scatter(tri[m, 0], tri[m, 1], s=(30 if lab == 0 else 6),
                   c=LAB_COLOR[lab], edgecolors='k' if lab == 0 else 'none',
                   linewidths=0.5, zorder=3 if lab == 0 else 1)
    ax.set_aspect('equal'); ax.axis('off')


def edge_curve(d, u, v):
    e2s = d['edge_to_streamline']; vc = d['vertices_cartesian'].numpy()
    c = e2s.get((u, v))
    if c is not None:
        return np.asarray(c, float)
    c = e2s.get((v, u))
    if c is not None:
        return np.asarray(c, float)[::-1]
    return np.stack([vc[u], vc[v]])


def draw_true(ax, d):
    faces = d['faces'].numpy().T; cmap = plt.get_cmap('tab20')
    for fi, q in enumerate(faces):
        loop = [edge_curve(d, int(q[k]), int(q[(k + 1) % 4])) for k in range(4)]
        poly = np.vstack(loop)
        ax.fill(poly[:, 0], poly[:, 1], color=cmap(fi % 20), alpha=0.65, ec='k', lw=0.5)
    ax.set_aspect('equal'); ax.axis('off')


def draw_gen(ax, cart_abs, faces_new):
    """Generierte Vertices (absolut) + O-Grid-Topologie (new-Indizes), gerade Kanten."""
    cmap = plt.get_cmap('tab20')
    for fi, q in enumerate(faces_new.T):
        poly = cart_abs[[int(q[0]), int(q[1]), int(q[2]), int(q[3])]]
        ax.fill(poly[:, 0], poly[:, 1], color=cmap(fi % 20), alpha=0.65, ec='k', lw=0.5)
    ax.scatter(cart_abs[:, 0], cart_abs[:, 1], s=10, c='k', zorder=4)
    ax.set_aspect('equal'); ax.axis('off')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='runs_planb/s1_var.pt')
    ap.add_argument('--data', default='domain_data_smoke.pt')
    ap.add_argument('--per-fc', type=int, default=2, help='Meshes je Facecount')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='figures/gen_meshes/gen_meshes.png')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = PolytronTokenizer(repr_mode='cubic_bezier')

    print(f"Lade Modell {args.ckpt} ...")
    ck = torch.load(args.ckpt, weights_only=False, map_location=device)
    model = vh.VertexGen(ck['vocab'], d=ck['d_model'], n_enc=ck['n_enc'],
                         n_dec=ck['n_dec'], max_len=ck['max_len'],
                         start_id=ck['start']).to(device)
    model.load_state_dict(ck['model']); model.eval()
    START, STOP = ck['start'], ck['stop']
    print(f"  best ep{ck.get('epoch')}, val-loss {ck.get('val_loss')}")

    print(f"Lade {args.data} ...")
    data = torch.load(args.data, weights_only=False)
    rng = np.random.default_rng(args.seed)

    # je Facecount per_fc Meshes waehlen, bei denen S1 die richtige Vertexzahl trifft
    picks = []                                  # (d, n, faces_new, cart_gen)
    for fc in vh.FACECOUNTS:
        pool = [d for d in data if d['faces'].shape[1] == fc]
        rng.shuffle(pool)
        found = 0
        for d in pool:
            n = vh.FC2N[fc]
            vp = d['vertices_polar']
            order = tok._sort_order(vp)
            old2new = np.empty(len(order), dtype=np.int64); old2new[order] = np.arange(len(order))
            faces_new = old2new[d['faces'].numpy()]         # [4,F] new-Indizes
            pts = torch.tensor(d['tri_coordinates'].numpy(), dtype=torch.float32)
            polar = vh.s1_generate(model, pts, n, tok, START, STOP, device)
            if polar.shape[0] != len(order):                # falsche Anzahl -> naechstes
                continue
            r, th = polar[:, 0], polar[:, 1]
            cart = np.stack([r * np.cos(th), r * np.sin(th)], 1) + d['center'].numpy()[None]
            picks.append((d, fc, faces_new, cart))
            found += 1
            if found >= args.per_fc:
                break
        print(f"  {fc}F: {found} Meshes mit korrekter Vertexzahl")

    nrows = len(picks)
    fig, axes = plt.subplots(nrows, 3, figsize=(11, 3.4 * nrows))
    axes = np.atleast_2d(axes)
    for r, (d, fc, faces_new, cart) in enumerate(picks):
        draw_pointcloud(axes[r, 0], d)
        draw_gen(axes[r, 1], cart, faces_new)
        draw_true(axes[r, 2], d)
        axes[r, 0].set_ylabel(f"{fc}F (n={vh.FC2N[fc]})", fontsize=11, rotation=90)
    axes[0, 0].set_title("Punktwolke (Eingabe)", fontsize=12)
    axes[0, 1].set_title("generiert (S1-Vertices + O-Grid)", fontsize=12)
    axes[0, 2].set_title("true (GT)", fontsize=12)
    fig.suptitle(f"S1 (best ep{ck.get('epoch')}) — generierte Meshes je Facecount",
                 fontsize=13, y=1.0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches='tight')
    print("wrote", args.out)


if __name__ == '__main__':
    main()
