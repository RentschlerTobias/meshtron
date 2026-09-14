"""
python visualize_tokenizer.py <mesh.pt> [mesh_idx|start:end] [n_faces] [alpha] [strategy]

mesh_idx: single index (e.g. "0") or Python-style slice "start:end" (e.g. "0:5").
strategy: 1 (full 8 tokens/face) or 2 (row-compressed + EOR). Default: 2.
Range produces one PNG per mesh: viz_<idx>.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenizer_v2 import Tokenizer2D
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, FancyBboxPatch
import matplotlib.pyplot as plt
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')

PALETTE = [
    '#e41a1c', '#4daf4a', '#377eb8', '#ff7f00', '#984ea3', '#a65628',
    '#f781bf', '#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e',
    '#e6ab02', '#a6761d', '#000000', '#8dd3c7', '#fb8072', '#80b1d3',
]


def viz(mesh_path, mesh_idx=0, n_faces=None, alpha=0.30, strategy=2, out='viz.png'):
    mesh = torch.load(mesh_path, weights_only=False)[mesh_idx]
    verts = mesh.x[:, :2]
    tok = Tokenizer2D(quantization_levels=256, sorting_strategy=strategy)
    # [n_total, 4], list[(s,e)] or None
    ordered, rows = tok._order_quads(verts, mesh.faces)

    n_total = ordered.shape[0]
    n = n_total if n_faces is None else min(n_faces, n_total)
    v = verts.numpy()

    # If strategy 1 didn't return rows, synthesize a single-row layout
    if rows is None:
        rows = [(0, n_total)]

    # Truncate rows to the displayed faces
    rows_shown = []
    for s, e in rows:
        if s >= n:
            break
        rows_shown.append((s, min(e, n)))
    if not rows_shown:
        rows_shown = [(0, n)]

    # Persistent color per unique vertex (first-appearance order)
    col_idx = {}
    for fi in range(n):
        for vi in range(4):
            vidx = ordered[fi, vi].item()
            if vidx not in col_idx:
                col_idx[vidx] = len(col_idx)

    def vc(vidx): return PALETTE[col_idx[vidx] % len(PALETTE)]

    # First-occurrence flag per (face, vertex-position)
    seen, first = set(), {}
    for fi in range(n):
        for vi in range(4):
            vidx = ordered[fi, vi].item()
            first[(fi, vi)] = vidx not in seen
            seen.add(vidx)

    # For strategy 2: mark which (face, vi) tokens are implicit (dropped during emission)
    # = first 2 vertices of every non-row-start face.
    row_starts = {s for s, _ in rows_shown}
    implicit = {(fi, vi): (strategy == 2 and fi not in row_starts and vi < 2)
                for fi in range(n) for vi in range(4)}

    # Layout: one line per compression row in the bottom panel.
    # Each line shows all faces in that row, followed by an EOR badge.
    max_row_len = max(e - s for s, e in rows_shown)
    seq_h = max(2.0, len(rows_shown) * 0.95)

    fig = plt.figure(figsize=(11, 8.5 + seq_h))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[8.5, seq_h], hspace=0.04)
    ax = fig.add_subplot(gs[0])

    # Full mesh (face-order coolwarm, all faces)
    cmap = plt.cm.coolwarm
    for fi in range(n_total):
        coords = v[ordered[fi].numpy()]
        c = cmap(fi / max(n_total - 1, 1))
        ax.add_patch(Polygon(coords, closed=True,
                             fc=(*c[:3], 0.18), ec='#888', lw=0.5, zorder=1))
        ax.text(*coords.mean(0), str(fi + 1), ha='center', va='center',
                fontsize=5, color='#444', zorder=2)

    # Tint each compression row with a light background colour to highlight
    # the row structure in the mesh plot (strategy 2 only).
    if strategy == 2:
        ROW_TINTS = ['#fde0dc', '#dce8fc', '#dcfce0',
                     '#fcf3dc', '#f0dcfc', '#dcfcf3']
        for ri, (s, e) in enumerate(rows_shown):
            row_verts = []
            for fi in range(s, e):
                row_verts.extend(v[ordered[fi].numpy()])
            if not row_verts:
                continue
            arr = np.array(row_verts)
            xmin, ymin = arr.min(0)
            xmax, ymax = arr.max(0)
            pad = 0.005
            ax.add_patch(plt.Rectangle((xmin - pad, ymin - pad),
                                       xmax - xmin + 2 * pad,
                                       ymax - ymin + 2 * pad,
                                       fc=ROW_TINTS[ri % len(ROW_TINTS)],
                                       ec='none', alpha=0.35, zorder=0))

    # Mini-element per face: vertices shifted alpha-fraction towards centroid
    for fi in range(n):
        coords = v[ordered[fi].numpy()]
        inner = coords * (1 - alpha) + coords.mean(0) * alpha

        ax.add_patch(Polygon(inner, closed=True, fc='none',
                     ec='#333', lw=0.7, zorder=3))

        for vi in range(4):
            vidx = ordered[fi, vi].item()
            is_first = first[(fi, vi)]
            is_implicit = implicit[(fi, vi)]
            face_color = '#bbb' if is_implicit else vc(vidx)
            ax.plot(*inner[vi], 'o' if is_first else 's', color=face_color,
                    ms=8 if is_first else 5,
                    mec='black', mew=0.4,
                    alpha=0.45 if is_implicit else 1.0,
                    zorder=4)

    ax.autoscale()
    ax.set_aspect('equal')
    ax.axis('off')
    legend_items = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#888',
               markeredgecolor='black', markersize=10, label='First occurrence'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#888',
               markeredgecolor='black', markersize=7, label='Repeated vertex'),
    ]
    if strategy == 2:
        legend_items.append(
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#bbb',
                   markeredgecolor='black', markersize=7,
                   label='Implicit (not emitted)'))
    ax.legend(handles=legend_items, loc='upper right',
              fontsize=8, framealpha=0.9)
    ax.set_title(
        f'{mesh_path}  [mesh {mesh_idx}]  –  Strategy {strategy}'
        f'   ({n}/{n_total} faces, {len(rows_shown)} compression rows, α={alpha})',
        fontsize=9,
    )

    # Token sequence – one line per compression row, EOR badge at row end.
    ax_s = fig.add_subplot(gs[1])
    # x axis: max_row_len faces + 1 EOR slot
    n_slots = max_row_len + 1
    ax_s.set_xlim(0, n_slots)
    ax_s.set_ylim(0, len(rows_shown))
    ax_s.axis('off')

    dx = 0.18  # spacing between tokens within a face slot
    slot_w = 1.0

    for ri, (s, e) in enumerate(rows_shown):
        row_y = len(rows_shown) - 1 - ri  # top-to-bottom
        # Row label
        ax_s.text(-0.15, row_y + 0.5, f'row {ri}',
                  ha='right', va='center', fontsize=8, color='#666',
                  fontstyle='italic')

        for slot, fi in enumerate(range(s, e)):
            x0 = slot * slot_w + 0.05

            # Light background for row-start face to emphasise it
            if fi in row_starts and strategy == 2:
                ax_s.add_patch(FancyBboxPatch(
                    (x0 - 0.04, row_y + 0.05), slot_w * 0.92, 0.9,
                    boxstyle='round,pad=0.01',
                    fc='#f5f5f5', ec='#ccc', lw=0.5, zorder=0))

            for vi in range(4):
                vidx = ordered[fi, vi].item()
                xn = x0 + vi * dx
                is_imp = implicit[(fi, vi)]
                color = '#aaa' if is_imp else vc(vidx)
                weight = 'normal' if is_imp else 'bold'
                style = 'italic' if is_imp else 'normal'
                label = f'$v_{vi + 1}^{{{fi + 1}}}$'
                if is_imp:
                    label = f'$\\langle v_{vi + 1}^{{{fi + 1}}} \\rangle$'
                ax_s.text(xn, row_y + 0.70, label,
                          ha='center', va='center', fontsize=9,
                          color=color, fontweight=weight, fontstyle=style)
                if first[(fi, vi)] and not is_imp:
                    ax_s.plot(xn, row_y + 0.25, 'o', color=color,
                              ms=5, mec='black', mew=0.4)
                if is_imp:
                    # subtle marker showing it would be filled by previous exit edge
                    ax_s.plot(xn, row_y + 0.25, 'x',
                              color='#aaa', ms=4, mew=1.0)

        # EOR badge after the row's last face
        if strategy == 2:
            xeor = (e - s) * slot_w + 0.20
            ax_s.add_patch(FancyBboxPatch(
                (xeor - 0.05, row_y + 0.20), 0.55, 0.55,
                boxstyle='round,pad=0.02',
                fc='#fff2c5', ec='#c0a23a', lw=0.8, zorder=0))
            ax_s.text(xeor + 0.22, row_y + 0.47, 'EOR',
                      ha='center', va='center', fontsize=8.5,
                      color='#7a5a00', fontweight='bold')

    legend_text = '●  First occurrence of unique vertex'
    if strategy == 2:
        legend_text += '       ⟨v⟩, ×  implicit (= reversed exit edge of prev face)       EOR  end-of-row token'
    ax_s.text(0.0, -0.10 / max(1, len(rows_shown)), legend_text,
              fontsize=7.5, ha='left', va='top', color='#555')

    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f'→ {out}')


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'centered_blades_cleaned.pt'
    midx_arg = sys.argv[2] if len(sys.argv) > 2 else '0'
    nf = int(sys.argv[3]) if len(sys.argv) > 3 else None
    alpha = float(sys.argv[4]) if len(sys.argv) > 4 else 0.30
    strat = int(sys.argv[5]) if len(sys.argv) > 5 else 2

    if ':' in midx_arg:
        a, b = midx_arg.split(':')
        indices = list(range(int(a), int(b)))
    else:
        indices = [int(midx_arg)]

    for i in indices:
        out = f'viz_s{strat}_{i}.png' if len(
            indices) > 1 else f'viz_s{strat}.png'
        viz(path, i, nf, alpha, strategy=strat, out=out)
