"""
plot_domain_pipeline.py

DEPRECATED: part of the MeshtronDomain family (failed experiment, superseded by
Polytron / viz_twostage.py, viz_pipeline_doc.py). Kept for reference only, not
maintained or extended.

Visualisiert die komplette Domain-Partition Pipeline in einem Figure-Grid:
  1. Eingabe: Point Cloud + Blocking Nodes + Streamlines
  2. Polar Plot: Vertices in Polarkoordinaten
  3. Token-Sequenz: Farbkodierte Tokens (8 pro Vertex-Platz)
  4. Blocked Mesh: Faces + Hermite-Spline Kanten
  5. Transfinite: Feines Quad-Mesh
  6. Vergleich: Ground Truth vs. Rekonstruiert
"""


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from pathlib import Path

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TOOLS = os.path.normpath(os.path.join(
    _HERE, '..', 'domain_partition', 'domain_partition_2D', 'tools'))
sys.path.insert(0, _HERE)
sys.path.insert(0, _TOOLS)

from tokenizer_domain import DomainTokenizer
from meshtron.legacy.reconstruct_domain import reconstruct_blocked_mesh, reconstruct_domain
from transfinite_interpolation import Transfinite_Interpolation


def plot_pipeline(mesh_data, tokenizer, output_dir='./figures/pipeline', mesh_idx=0):
    """Erstellt ein 6-Subplot-Figure fuer eine Mesh-Pipeline."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Daten vorbereiten
    # --------------------------------------------------
    vertices_cart = mesh_data['vertices_cartesian'].numpy()
    vertices_pol = mesh_data['vertices_polar'].numpy()
    faces = mesh_data['faces']  # [4, n_faces]
    edge_index = mesh_data['edge_index'].numpy()
    edge_tangents = mesh_data['edge_tangents'].numpy()
    tri_coords = mesh_data['tri_coordinates'].numpy()
    center = mesh_data['center'].numpy()
    edge_to_streamline = mesh_data['edge_to_streamline']

    # Tokenisierung
    tokens = tokenizer.tokenize(mesh_data)
    output = tokenizer.detokenize(tokens)

    # --------------------------------------------------
    # Figure erstellen
    # --------------------------------------------------
    fig = plt.figure(figsize=(24, 14))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # === Subplot 1: Eingabe (Point Cloud + Blocking + Streamlines) ===
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('1. Eingabe: Point Cloud + Blocking Nodes + Streamlines', fontsize=12, fontweight='bold')

    # Point cloud (nur x,y)
    pc_mask = tri_coords[:, 2] != 2
    pc_boundary = tri_coords[pc_mask, :2]
    pc_interior = tri_coords[~pc_mask, :2]
    ax1.scatter(pc_interior[:, 0], pc_interior[:, 1], s=3, c='lightgrey', alpha=0.5, label='Interior')
    ax1.scatter(pc_boundary[:, 0], pc_boundary[:, 1], s=5, c='black', alpha=0.7, label='Boundary')

    # Blocking nodes
    ax1.scatter(vertices_cart[:, 0], vertices_cart[:, 1], s=100, c='red', zorder=5, marker='o', edgecolors='darkred', linewidth=1.5)
    for i, (x, y) in enumerate(vertices_cart):
        ax1.text(x, y, str(i), fontsize=8, ha='center', va='center', color='white', fontweight='bold', zorder=6)

    # Streamlines
    for (u, v), pts in edge_to_streamline.items():
        pts_arr = np.asarray(pts)
        color = plt.cm.tab20((u * 3 + v) % 20 / 20)
        ax1.plot(pts_arr[:, 0], pts_arr[:, 1], color=color, linewidth=1.5, alpha=0.8)

    # Faces als transparente Füllung
    for fi in range(faces.shape[1]):
        face = faces[:, fi].numpy()
        coords = vertices_cart[face]
        color = plt.cm.Set3(fi / max(faces.shape[1] - 1, 1))
        ax1.fill(coords[:, 0], coords[:, 1], color=color, alpha=0.2, edgecolor='grey', linewidth=0.5)

    ax1.set_aspect('equal')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend(loc='upper right', fontsize=8)

    # === Subplot 2: Polar Koordinaten ===
    ax2 = fig.add_subplot(gs[0, 1], projection='polar')
    ax2.set_title('2. Vertices in Polarkoordinaten', fontsize=12, fontweight='bold', pad=20)

    r = vertices_pol[:, 0]
    theta = vertices_pol[:, 1]
    colors = plt.cm.tab20(np.linspace(0, 1, len(r)))

    for i in range(len(r)):
        ax2.plot([0, theta[i]], [0, r[i]], color=colors[i], linewidth=1.5, alpha=0.7)
        ax2.scatter(theta[i], r[i], s=100, c=[colors[i]], zorder=5, edgecolors='black', linewidth=1)
        ax2.text(theta[i], r[i], str(i), fontsize=8, ha='center', va='center', fontweight='bold')

    ax2.set_rticks(np.linspace(0, r.max(), 4))
    ax2.set_rlabel_position(22.5)
    ax2.grid(True)

    # === Subplot 3: Token-Sequenz ===
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title(f'3. Token-Sequenz ({len(tokens)} Tokens)', fontsize=12, fontweight='bold')

    # Zeige Tokens als farbige Rechtecke
    n_show = min(len(tokens), 240)
    tokens_show = np.array(tokens[:n_show])

    # Reshape zu Grid
    n_cols = 24
    n_rows = (n_show + n_cols - 1) // n_cols
    grid = np.full((n_rows, n_cols), np.nan)
    grid.flat[:n_show] = tokens_show

    # Farbkodierung: 8 Token-Typen
    # Typ-Labels fuer jeden Token im 8-Takt
    type_names = ['r', 'θsin', 'θcos', 't_norm', 'α_in sin', 'α_in cos', 'α_out sin', 'α_out cos']
    type_colors = plt.cm.Set2(np.linspace(0, 1, 8))

    # Erstelle ein Bild mit Typ-Farben
    img = np.zeros((n_rows, n_cols, 3))
    for i in range(n_show):
        row = i // n_cols
        col = i % n_cols
        tok = tokens_show[i]
        # Bestimme Typ (Position modulo 8, nach Start-Tokens)
        # Einfacher: wir nutzen den Token-Wert selbst fuer Farbe
        if tok >= tokenizer.coord_vocab_size:
            img[row, col] = [0.8, 0.8, 0.8]  # Special token: grey
        else:
            # Normalisiere auf [0,1]
            val = tok / tokenizer.coord_vocab_size
            img[row, col] = plt.cm.viridis(val)[:3]

    ax3.imshow(img, aspect='auto', interpolation='nearest')
    ax3.set_xticks([])
    ax3.set_yticks([])

    # Legende
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=plt.cm.viridis(i/7)[:3], edgecolor='black', label=f'Type {i}') for i in range(8)]
    ax3.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=7, ncol=1)

    # === Subplot 4: Blocked Mesh (mit Hermite Splines) ===
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_title('4. Blocked Mesh + Hermite-Spline Kanten', fontsize=12, fontweight='bold')

    blocked_mesh = reconstruct_blocked_mesh(output, center)
    nodes = blocked_mesh.x.numpy()
    faces_blk = blocked_mesh.faces.numpy()
    edge_pts = blocked_mesh.edge_subdomain_points

    # Zeichne Faces
    for fi in range(faces_blk.shape[1]):
        face = faces_blk[:, fi]
        coords = nodes[face]
        color = plt.cm.Set3(fi / max(faces_blk.shape[1] - 1, 1))
        ax4.fill(coords[:, 0], coords[:, 1], color=color, alpha=0.3, edgecolor='grey', linewidth=0.5)

    # Zeichne Hermite-Spline Kanten
    for i, pts in enumerate(edge_pts):
        color = plt.cm.tab20(i / max(len(edge_pts) - 1, 1))
        ax4.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2, alpha=0.9)

    # Vertices
    ax4.scatter(nodes[:, 0], nodes[:, 1], s=80, c='red', zorder=5, edgecolors='darkred', linewidth=1.5)
    for i, (x, y) in enumerate(nodes):
        ax4.text(x, y, str(i), fontsize=8, ha='center', va='center', color='white', fontweight='bold', zorder=6)

    ax4.set_aspect('equal')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')

    # === Subplot 5: Transfinite Interpolation ===
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_title('5. Transfinite Interpolation (feines Quad-Mesh)', fontsize=12, fontweight='bold')

    try:
        interpolator = Transfinite_Interpolation(blocked_mesh, mesh_size=0.4)
        quad_mesh = interpolator.quad_mesh
        q_nodes = quad_mesh.x.numpy()
        q_faces = quad_mesh.faces.numpy()

        for fi in range(q_faces.shape[1]):
            face = q_faces[:, fi]
            coords = q_nodes[face]
            color = plt.cm.Set3(fi / max(q_faces.shape[1] - 1, 1))
            ax5.fill(coords[:, 0], coords[:, 1], color=color, alpha=0.3, edgecolor='grey', linewidth=0.3)

        ax5.set_aspect('equal')
        ax5.set_xlabel('x')
        ax5.set_ylabel('y')
    except Exception as e:
        ax5.text(0.5, 0.5, f'Transfinite fehlgeschlagen:\n{e}', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)

    # === Subplot 6: Vergleich GT vs. Rekonstruktion ===
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_title('6. Vergleich: Ground Truth (blau) vs. Rekonstruiert (rot)', fontsize=12, fontweight='bold')

    # GT: Original fine mesh
    original_meshes = torch.load(os.path.join(_HERE, 'checkpoint_mesh_100.pt'), weights_only=False)
    gt_mesh = original_meshes[mesh_idx]
    gt_nodes = gt_mesh.quad_coordinates.numpy()
    gt_faces = gt_mesh.quad_faces.numpy()

    for fi in range(gt_faces.shape[1]):
        face = gt_faces[:, fi]
        coords = gt_nodes[face]
        ax6.fill(coords[:, 0], coords[:, 1], color='lightblue', alpha=0.4, edgecolor='blue', linewidth=0.3)

    # Rekonstruktion
    if 'quad_mesh' in dir():
        for fi in range(q_faces.shape[1]):
            face = q_faces[:, fi]
            coords = q_nodes[face]
            ax6.fill(coords[:, 0], coords[:, 1], color='lightcoral', alpha=0.3, edgecolor='red', linewidth=0.3)

    ax6.set_aspect('equal')
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')

    # --------------------------------------------------
    # Speichern
    # --------------------------------------------------
    fig.suptitle(f'Domain-Partition Pipeline — Mesh {mesh_idx} | {faces.shape[1]} Faces | {len(tokens)} Tokens',
                 fontsize=16, fontweight='bold', y=0.98)

    output_file = output_dir / f'pipeline_mesh_{mesh_idx}.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Gespeichert: {output_file}")


def plot_token_evolution(tokenizer, mesh_data, output_dir='./figures/tokens'):
    """Visualisiert wie die Token-Sequenz aufgebaut wird (face by face)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vertices_cart = mesh_data['vertices_cartesian'].numpy()
    faces = mesh_data['faces'].numpy()

    # Nur fuer Strategy 0 (keine Kompression)
    if tokenizer.sorting_strategy != 0:
        print("Token evolution plot nur fuer Strategy 0")
        return

    n_faces = faces.shape[1]
    n_places_per_face = 4
    tokens_per_place = 8

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for step in range(min(n_faces, 6)):
        ax = axes[step]
        ax.set_title(f'Schritt {step + 1}: {step + 1} Face(s) tokenisiert', fontsize=11, fontweight='bold')

        # Zeichne alle Vertices
        ax.scatter(vertices_cart[:, 0], vertices_cart[:, 1], s=50, c='lightgrey', zorder=1)

        # Zeichne tokenisierte Faces farbig
        for fi in range(step + 1):
            face = faces[:, fi]
            coords = vertices_cart[face]
            color = plt.cm.Set3(fi / max(n_faces - 1, 1))
            ax.fill(coords[:, 0], coords[:, 1], color=color, alpha=0.4, edgecolor='grey', linewidth=1)

            # Markiere die 4 Vertex-Plaetze mit ihren Token-Farben
            for vi_idx, vi in enumerate(face):
                prev = face[(vi_idx - 1) % 4]
                next_v = face[(vi_idx + 1) % 4]

                x, y = vertices_cart[vi]
                # Kleine farbige Kreise fuer die 8 Tokens
                for ti in range(8):
                    angle = 2 * np.pi * ti / 8
                    radius = 0.015
                    cx = x + radius * np.cos(angle)
                    cy = y + radius * np.sin(angle)
                    tok_color = plt.cm.Set2(ti / 7)
                    ax.scatter(cx, cy, s=30, c=[tok_color], zorder=5, edgecolors='black', linewidth=0.5)

        # Markiere noch-nicht-tokenisierte Faces als Outline
        for fi in range(step + 1, n_faces):
            face = faces[:, fi]
            coords = vertices_cart[face]
            ax.plot(list(coords[:, 0]) + [coords[0, 0]],
                    list(coords[:, 1]) + [coords[0, 1]],
                    'k--', linewidth=0.8, alpha=0.3)

        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    plt.tight_layout()
    output_file = output_dir / 'token_evolution.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Gespeichert: {output_file}")


def plot_comparison_grid(tokenizer, data, indices=[0, 10, 20, 30], output_dir='./figures/comparison'):
    """Vergleicht GT vs. Rekonstruktion fuer mehrere Meshes."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = len(indices)
    fig, axes = plt.subplots(n, 2, figsize=(12, 5 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for row, idx in enumerate(indices):
        mesh_data = data[idx]
        center = mesh_data['center'].numpy()

        # GT
        original_meshes = torch.load(os.path.join(_HERE, 'checkpoint_mesh_100.pt'), weights_only=False)
        gt_mesh = original_meshes[idx]
        gt_nodes = gt_mesh.quad_coordinates.numpy()
        gt_faces = gt_mesh.quad_faces.numpy()

        ax_gt = axes[row, 0]
        ax_gt.set_title(f'Mesh {idx} — Ground Truth', fontsize=11, fontweight='bold')
        for fi in range(gt_faces.shape[1]):
            face = gt_faces[:, fi]
            coords = gt_nodes[face]
            ax_gt.fill(coords[:, 0], coords[:, 1], color='lightblue', alpha=0.5, edgecolor='blue', linewidth=0.3)
        ax_gt.set_aspect('equal')

        # Rekonstruktion
        tokens = tokenizer.tokenize(mesh_data)
        output = tokenizer.detokenize(tokens)
        try:
            quad_mesh = reconstruct_domain(output, center, transfinite_divisions=5)
            q_nodes = quad_mesh.x.numpy()
            q_faces = quad_mesh.faces.numpy()

            ax_rec = axes[row, 1]
            ax_rec.set_title(f'Mesh {idx} — Rekonstruiert', fontsize=11, fontweight='bold')
            for fi in range(q_faces.shape[1]):
                face = q_faces[:, fi]
                coords = q_nodes[face]
                ax_rec.fill(coords[:, 0], coords[:, 1], color='lightcoral', alpha=0.5, edgecolor='red', linewidth=0.3)
            ax_rec.set_aspect('equal')
        except Exception as e:
            ax_rec = axes[row, 1]
            ax_rec.text(0.5, 0.5, f'Fehler: {e}', ha='center', va='center', transform=ax_rec.transAxes)
            ax_rec.set_xlim(0, 1)
            ax_rec.set_ylim(0, 1)

    plt.tight_layout()
    output_file = output_dir / 'comparison_grid.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Gespeichert: {output_file}")


if __name__ == '__main__':
    # Daten laden
    data = torch.load(os.path.join(_HERE, 'domain_data.pt'), weights_only=False)

    # Tokenizer
    tok = DomainTokenizer(quantization_r=512, quantization_a=256,
                         sorting_strategy=0, embedding_mode=0, verbose=False)

    # Pipeline-Plots fuer Mesh 0, 10, 20
    for idx in [0, 10, 20]:
        print(f"\n=== Plotting Mesh {idx} ===")
        plot_pipeline(data[idx], tok, mesh_idx=idx)

    # Token-Evolution
    print("\n=== Token Evolution ===")
    plot_token_evolution(tok, data[0])

    # Vergleichs-Grid
    print("\n=== Comparison Grid ===")
    plot_comparison_grid(tok, data, indices=[0, 5, 10, 15])

    print("\nAlle Plots erstellt!")
