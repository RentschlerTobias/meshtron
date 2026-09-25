"""
plot_domain_and_pipeline.py

DEPRECATED: part of the MeshtronDomain family (failed experiment, superseded by
Polytron / viz_twostage.py, viz_pipeline_doc.py). Kept for reference only, not
maintained or extended.

Plot: Domain data + Tokenize -> Detokenize -> Hermite Spline pipeline
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TOOLS = os.path.normpath(os.path.join(
    _HERE, '..', 'domain_partition', 'domain_partition_2D', 'tools'))
sys.path.insert(0, _HERE)
sys.path.insert(0, _TOOLS)

import numpy as np
import torch
import matplotlib.pyplot as plt
from tokenizer_domain import DomainTokenizer
from meshtron.legacy.reconstruct_domain import (reconstruct_blocked_mesh, reconstruct_domain,
                                reconstruct_domain_coons)

# Daten laden
data = torch.load(os.path.join(_HERE, 'domain_data.pt'), weights_only=False)
mesh = data[0]
center = mesh['center'].numpy()

# Tokenizer
tok = DomainTokenizer(quantization_r=512, quantization_a=256,
                       sorting_strategy=0, embedding_mode=0, verbose=False)

# Tokenisierung
tokens = tok.tokenize(mesh)
output = tok.detokenize(tokens)

# Rekonstruktion
blocked_mesh = reconstruct_blocked_mesh(output, center)
# Per-Block Coons-TFI (wie domain_partition_3D) -> Airfoil-Loch bleibt hohl
quad_mesh = reconstruct_domain_coons(output, center, n=11)

# --------------------------------------------------
# Plot 1: Domain Data (Eingabe)
# --------------------------------------------------
fig1, ax1 = plt.subplots(1, 1, figsize=(10, 10))

vertices_cart = mesh['vertices_cartesian'].numpy()
faces = mesh['faces']  # [4, n_faces]
tri_coords = mesh['tri_coordinates'].numpy()
edge_to_streamline = mesh['edge_to_streamline']

# Point cloud
pc_mask = tri_coords[:, 2] != 2
pc_boundary = tri_coords[pc_mask, :2]
pc_interior = tri_coords[~pc_mask, :2]
ax1.scatter(pc_interior[:, 0], pc_interior[:, 1], s=3, c='lightgrey', alpha=0.5, label='Interior')
ax1.scatter(pc_boundary[:, 0], pc_boundary[:, 1], s=5, c='black', alpha=0.7, label='Boundary')

# Blocking nodes
ax1.scatter(vertices_cart[:, 0], vertices_cart[:, 1], s=120, c='red', zorder=5, marker='o', edgecolors='darkred', linewidth=2)
for i, (x, y) in enumerate(vertices_cart):
    ax1.text(x, y, str(i), fontsize=9, ha='center', va='center', color='white', fontweight='bold', zorder=6)

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
    ax1.fill(coords[:, 0], coords[:, 1], color=color, alpha=0.25, edgecolor='grey', linewidth=0.5)

ax1.set_aspect('equal')
ax1.set_title(f'Domain Data — Mesh 0 | {faces.shape[1]} Faces | {len(vertices_cart)} Vertices', fontsize=14, fontweight='bold')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(_HERE, 'figures', 'domain_data_mesh0.png'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: figures/domain_data_mesh0.png")

# --------------------------------------------------
# Plot 2: Tokenize -> Detokenize -> Hermite Spline Pipeline
# --------------------------------------------------
fig2 = plt.figure(figsize=(20, 12))
gs = fig2.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

# === Subplot A: Original Blocking Mesh ===
ax_a = fig2.add_subplot(gs[0, 0])
ax_a.set_title('A. Original Blocking Mesh', fontsize=12, fontweight='bold')
for fi in range(faces.shape[1]):
    face = faces[:, fi].numpy()
    coords = vertices_cart[face]
    color = plt.cm.Set3(fi / max(faces.shape[1] - 1, 1))
    ax_a.fill(coords[:, 0], coords[:, 1], color=color, alpha=0.3, edgecolor='grey', linewidth=0.5)
ax_a.scatter(vertices_cart[:, 0], vertices_cart[:, 1], s=80, c='red', zorder=5, edgecolors='darkred', linewidth=1.5)
for i, (x, y) in enumerate(vertices_cart):
    ax_a.text(x, y, str(i), fontsize=8, ha='center', va='center', color='white', fontweight='bold', zorder=6)
ax_a.set_aspect('equal')
ax_a.set_xlabel('x')
ax_a.set_ylabel('y')

# === Subplot B: Token Sequence ===
ax_b = fig2.add_subplot(gs[0, 1])
ax_b.set_title(f'B. Token Sequence ({len(tokens)} Tokens)', fontsize=12, fontweight='bold')
n_show = min(len(tokens), 240)
tokens_show = np.array(tokens[:n_show])
n_cols = 24
n_rows = (n_show + n_cols - 1) // n_cols
grid = np.full((n_rows, n_cols), np.nan)
grid.flat[:n_show] = tokens_show

img = np.zeros((n_rows, n_cols, 3))
for i in range(n_show):
    row = i // n_cols
    col = i % n_cols
    tok_val = tokens_show[i]
    if tok_val >= tok.coord_vocab_size:
        img[row, col] = [0.8, 0.8, 0.8]
    else:
        val = tok_val / tok.coord_vocab_size
        img[row, col] = plt.cm.viridis(val)[:3]

ax_b.imshow(img, aspect='auto', interpolation='nearest')
ax_b.set_xticks([])
ax_b.set_yticks([])

# === Subplot C: Detokenized Vertex Places (Polar -> Cartesian) ===
ax_c = fig2.add_subplot(gs[0, 2])
ax_c.set_title('C. Detokenized Vertex Places', fontsize=12, fontweight='bold')

places = output['vertex_places']
faces_as_places = output['faces_as_places']

# Reconstruct cartesian from detokenized polar
from meshtron.legacy.reconstruct_domain import polar_to_cartesian, merge_duplicate_vertices
cartesian = polar_to_cartesian(places, center)
unique_verts, place_to_unique = merge_duplicate_vertices(cartesian, threshold=1e-3)

# Build faces with unique indices
n_faces_recon = len(faces_as_places)
faces_recon = np.zeros((4, n_faces_recon), dtype=int)
for fi, face in enumerate(faces_as_places):
    for i, p in enumerate(face):
        faces_recon[i, fi] = place_to_unique[p]

# Farbe je rekonstruierter Face nach passender ORIGINAL-Face (gleicher Block =
# gleiche Farbe wie in Subplot A), damit A und C direkt vergleichbar sind.
orig_faces = faces.numpy()  # [4, nf]
orig_centroids = vertices_cart[orig_faces.T].mean(axis=1)  # [nf, 2]
for fi in range(faces_recon.shape[1]):
    face = faces_recon[:, fi]
    coords = unique_verts[face]
    rc = coords.mean(axis=0)
    oi = int(np.argmin(np.linalg.norm(orig_centroids - rc, axis=1)))
    color = plt.cm.Set3(oi / max(orig_faces.shape[1] - 1, 1))
    ax_c.fill(coords[:, 0], coords[:, 1], color=color, alpha=0.3, edgecolor='grey', linewidth=0.5)

ax_c.scatter(unique_verts[:, 0], unique_verts[:, 1], s=80, c='red', zorder=5, edgecolors='darkred', linewidth=1.5)
for i, (x, y) in enumerate(unique_verts):
    ax_c.text(x, y, str(i), fontsize=8, ha='center', va='center', color='white', fontweight='bold', zorder=6)
ax_c.set_aspect('equal')
ax_c.set_xlabel('x')
ax_c.set_ylabel('y')

# === Subplot D: Hermite Spline Edges ===
ax_d = fig2.add_subplot(gs[1, 0])
ax_d.set_title('D. Hermite Spline Edges', fontsize=12, fontweight='bold')

nodes_blk = blocked_mesh.x.numpy()
faces_blk = blocked_mesh.faces.numpy()
edge_pts = blocked_mesh.edge_subdomain_points

for fi in range(faces_blk.shape[1]):
    face = faces_blk[:, fi]
    coords = nodes_blk[face]
    color = plt.cm.Set3(fi / max(faces_blk.shape[1] - 1, 1))
    ax_d.fill(coords[:, 0], coords[:, 1], color=color, alpha=0.2, edgecolor='grey', linewidth=0.5)

for i, pts in enumerate(edge_pts):
    color = plt.cm.tab20(i / max(len(edge_pts) - 1, 1))
    ax_d.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2.5, alpha=0.9)

ax_d.scatter(nodes_blk[:, 0], nodes_blk[:, 1], s=80, c='red', zorder=5, edgecolors='darkred', linewidth=1.5)
for i, (x, y) in enumerate(nodes_blk):
    ax_d.text(x, y, str(i), fontsize=8, ha='center', va='center', color='white', fontweight='bold', zorder=6)
ax_d.set_aspect('equal')
ax_d.set_xlabel('x')
ax_d.set_ylabel('y')

# === Subplot E: Transfinite Interpolation ===
ax_e = fig2.add_subplot(gs[1, 1])
ax_e.set_title('E. Transfinite Interpolation', fontsize=12, fontweight='bold')

q_nodes = quad_mesh.x.numpy()
q_faces = quad_mesh.faces.numpy()
for fi in range(q_faces.shape[1]):
    face = q_faces[:, fi]
    coords = q_nodes[face]
    color = plt.cm.Set3(fi / max(q_faces.shape[1] - 1, 1))
    ax_e.fill(coords[:, 0], coords[:, 1], color=color, alpha=0.3, edgecolor='grey', linewidth=0.3)
ax_e.set_aspect('equal')
ax_e.set_xlabel('x')
ax_e.set_ylabel('y')

# === Subplot F: Comparison GT vs Reconstructed ===
ax_f = fig2.add_subplot(gs[1, 2])
ax_f.set_title('F. Ground Truth (blau) vs Reconstructed (rot)', fontsize=12, fontweight='bold')

# GT
original_meshes = torch.load(os.path.join(_HERE, 'checkpoint_mesh_100.pt'), weights_only=False)
gt_mesh = original_meshes[0]
gt_nodes = gt_mesh.quad_coordinates.numpy()
gt_faces = gt_mesh.quad_faces.numpy()

for fi in range(gt_faces.shape[1]):
    face = gt_faces[:, fi]
    coords = gt_nodes[face]
    ax_f.fill(coords[:, 0], coords[:, 1], color='lightblue', alpha=0.4, edgecolor='blue', linewidth=0.3)

for fi in range(q_faces.shape[1]):
    face = q_faces[:, fi]
    coords = q_nodes[face]
    ax_f.fill(coords[:, 0], coords[:, 1], color='lightcoral', alpha=0.3, edgecolor='red', linewidth=0.3)

ax_f.set_aspect('equal')
ax_f.set_xlabel('x')
ax_f.set_ylabel('y')

fig2.suptitle('Tokenize -> Detokenize -> Hermite Spline -> Transfinite Interpolation Pipeline',
              fontsize=16, fontweight='bold', y=0.98)
plt.savefig(os.path.join(_HERE, 'figures', 'tokenize_pipeline_mesh0.png'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: figures/tokenize_pipeline_mesh0.png")
print("Done!")
