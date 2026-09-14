"""
python visualize_paper.py <mesh.pt> <mesh_idx> [alpha] [n_faces] [out.png]

Side-by-side comparison: strat 0 (left) vs strat 3 (right).
Face order color-coded blue→red (low to high).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenizer_v2 import Tokenizer2D
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, FancyBboxPatch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')

ROW_TINTS = ['#fde0dc', '#dce8fc', '#dcfce0',
             '#fcf3dc', '#f0dcfc', '#dcfcf3']


def viz_paper(mesh_path, mesh_idx=6, alpha=0.30, n_faces=None, out='comparison.png'):
    mesh = torch.load(mesh_path, weights_only=False)[mesh_idx]
    verts = mesh.x[:, :2]
    v = verts.numpy()
    n_total = mesh.faces.shape[1]

    tok0 = Tokenizer2D(quantization_levels=256, sorting_strategy=0)
    tok3 = Tokenizer2D(quantization_levels=256, sorting_strategy=3)

    ordered0, rows0 = tok0._order_quads(verts, mesh.faces)
    ordered3, rows3 = tok3._order_quads(verts, mesh.faces)

    n = n_total if n_faces is None else min(n_faces, n_total)

    cmap = plt.cm.coolwarm

    def face_color(fi, ordered):
        return cmap(fi / max(n_total - 1, 1))

    max_row_len = max(e - s for s, e in rows3) if rows3 else n

    n_chunks = (n_total + max_row_len - 1) // max_row_len
    pseudo_rows = [(i * max_row_len, min((i + 1) * max_row_len, n_total))
                   for i in range(n_chunks)]

    seq_h = max(2.0, len(rows3) * 0.95)

    fig = plt.figure(figsize=(16, 8 + seq_h))
    gs = GridSpec(2, 2, figure=fig,
                  width_ratios=[1, 1],
                  height_ratios=[8, seq_h],
                  hspace=0.0, wspace=0.0)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax_s0 = fig.add_subplot(gs[1, 0])
    ax_s1 = fig.add_subplot(gs[1, 1])

    for ax in [ax0, ax1]:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width, pos.height + 0.02])

    def col_idx(ordered_faces, n):
        seen = {}
        idx = 0
        for fi in range(n):
            for vi in range(4):
                vidx = ordered_faces[fi, vi].item()
                if vidx not in seen:
                    seen[vidx] = idx
                    idx += 1
        return seen

    ci0 = col_idx(ordered0, n)
    ci3 = col_idx(ordered3, n)

    def vc(vidx, col_idx_map):
        return ['#e41a1c', '#4daf4a', '#377eb8', '#ff7f00', '#984ea3', '#a65628',
                '#f781bf', '#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e',
                '#e6ab02', '#a6761d', '#000000', '#8dd3c7', '#fb8072', '#80b1d3'][col_idx_map.get(vidx, 0) % 18]

    def first_occurrence(ordered_faces, n):
        seen, first = set(), {}
        for fi in range(n):
            for vi in range(4):
                vidx = ordered_faces[fi, vi].item()
                first[(fi, vi)] = vidx not in seen
                seen.add(vidx)
        return first

    first0 = first_occurrence(ordered0, n)
    first3 = first_occurrence(ordered3, n)

    for fi in range(n):
        fc0 = face_color(fi, ordered0)
        fc3 = face_color(fi, ordered3)

        coords0 = v[ordered0[fi].numpy()]
        coords3 = v[ordered3[fi].numpy()]

        ax0.add_patch(Polygon(coords0, closed=True,
                              fc=(*fc0[:3], 0.18), ec='#888', lw=0.5, zorder=1))
        ax1.add_patch(Polygon(coords3, closed=True,
                              fc=(*fc3[:3], 0.18), ec='#888', lw=0.5, zorder=1))

        ax0.text(*coords0.mean(0), str(fi + 1), ha='center', va='center',
                 fontsize=7, color='#444', zorder=2)
        ax1.text(*coords3.mean(0), str(fi + 1), ha='center', va='center',
                 fontsize=7, color='#444', zorder=2)

        inner0 = coords0 * (1 - alpha) + coords0.mean(0) * alpha
        inner3 = coords3 * (1 - alpha) + coords3.mean(0) * alpha

        ax0.add_patch(Polygon(inner0, closed=True, fc='none',
                      ec='#333', lw=0.7, zorder=3))
        ax1.add_patch(Polygon(inner3, closed=True, fc='none',
                      ec='#333', lw=0.7, zorder=3))

        for vi in range(4):
            vidx0 = ordered0[fi, vi].item()
            vidx3 = ordered3[fi, vi].item()

            is_first0 = first0[(fi, vi)]
            is_first3 = first3[(fi, vi)]

            color0 = vc(vidx0, ci0)
            color3 = vc(vidx3, ci3)

            ax0.plot(*inner0[vi], 'o' if is_first0 else 's',
                     color=color0 if is_first0 else '#bbb',
                     ms=8 if is_first0 else 5,
                     mec='black', mew=0.4,
                     zorder=4)

            ax1.plot(*inner3[vi], 'o' if is_first3 else 's',
                     color=color3 if is_first3 else '#bbb',
                     ms=8 if is_first3 else 5,
                     mec='black', mew=0.4,
                     zorder=4)

    for ri, (s, e) in enumerate(rows3):
        row_verts = []
        for fi in range(s, min(e, n)):
            row_verts.extend(v[ordered3[fi].numpy()])
        if not row_verts:
            continue
        arr = np.array(row_verts)
        xmin, ymin = arr.min(0)
        xmax, ymax = arr.max(0)
        pad = 0.005
        ax1.add_patch(plt.Rectangle((xmin - pad, ymin - pad),
                                    xmax - xmin + 2 * pad,
                                    ymax - ymin + 2 * pad,
                                    fc=ROW_TINTS[ri % len(ROW_TINTS)],
                                    ec='none', alpha=0.35, zorder=0))

    ax0.autoscale()
    ax1.autoscale()
    ax0.set_aspect('equal')
    ax1.set_aspect('equal')

    x0, x1 = ax0.get_xlim()
    y0, y1 = ax0.get_ylim()
    ax1.set_xlim(x0, x1)
    ax1.set_ylim(y0, y1)

    ax0.axis('off')
    ax1.axis('off')

    fig.patch.set_alpha(0)
    for ax in [ax0, ax1, ax_s0, ax_s1]:
        ax.patch.set_alpha(0)

    cbar_w = 0.025
    pos1 = ax1.get_position()
    cbar_bottom = pos1.y0 + 0.02
    cbar_height = (pos1.y1 - pos1.y0) * 0.9
    cbar_ax = fig.add_axes(
        [1 - 0.04 + 0.005, cbar_bottom, cbar_w, cbar_height])

    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=0, vmax=n_total - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, shrink=0.9, pad=0.02)
    cbar.set_label('face order: low → high', fontsize=11)

    row_starts3 = {s for s, _ in rows3}
    dx = 0.28
    slot_w = 1.2

    def draw_seq(ax, ordered_faces, rows, row_starts, n_display, has_eor, show_row_labels, col_idx_map, first_occ, left_offset=0, max_rows=None):
        n_rows_to_show = max_rows if max_rows is not None else len(rows)
        ax.set_xlim(left_offset - 0.5, max_row_len + 1)
        ax.set_ylim(0, len(rows))
        ax.axis('off')

        for ri, (s, e) in enumerate(rows):
            if ri >= n_rows_to_show:
                if ri == n_rows_to_show:
                    row_y = len(rows) - 1 - ri
                    ax.text(max_row_len / 2 + left_offset, row_y + 0.5, '...',
                            ha='center', va='center', fontsize=18, color='#999')
                break

            row_y = len(rows) - 1 - ri

            if has_eor and show_row_labels:
                ax.text(left_offset - 0.15, row_y + 0.5, f'row {ri}',
                        ha='right', va='center', fontsize=10, color='#666',
                        fontstyle='italic')

            e_adj = min(e, n_display)
            for slot, fi in enumerate(range(s, e_adj)):
                x0 = slot * slot_w + 0.05 + left_offset

                if fi in row_starts and has_eor:
                    ax.add_patch(FancyBboxPatch(
                        (x0 - 0.04, row_y + 0.05), slot_w * 0.92, 0.9,
                        boxstyle='round,pad=0.01',
                        fc='#f5f5f5', ec='#ccc', lw=0.5, zorder=0))

                if not has_eor:
                    ax.add_patch(FancyBboxPatch(
                        (x0 - 0.04, row_y + 0.05), slot_w * 0.92, 0.9,
                        boxstyle='round,pad=0.01',
                        fc='none', ec='#ccc', lw=0.5, zorder=0))

                for vi in range(4):
                    vidx = ordered_faces[fi, vi].item()
                    xn = x0 + vi * dx
                    label = f'$v_{vi + 1}^{{{fi + 1}}}$'
                    ax.text(xn, row_y + 0.70, label,
                            ha='center', va='center', fontsize=11,
                            color='#333', fontweight='bold')

                    is_first = first_occ[(fi, vi)]
                    vertex_color = vc(vidx, col_idx_map)
                    if is_first:
                        ax.plot(xn, row_y + 0.25, 'o',
                                color=vertex_color,
                                ms=8, mec='black', mew=0.4, zorder=4)
                    elif not has_eor:
                        ax.plot(xn, row_y + 0.25, 's',
                                color='#bbb',
                                ms=5, mec='black', mew=0.4, zorder=4)

            if has_eor:
                xeor = (e_adj - s) * slot_w + 0.20 + left_offset
                ax.add_patch(FancyBboxPatch(
                    (xeor - 0.05, row_y + 0.20), 0.55, 0.55,
                    boxstyle='round,pad=0.02',
                    fc='#fff2c5', ec='#c0a23a', lw=0.8, zorder=0))
                ax.text(xeor + 0.22, row_y + 0.47, '★',
                        ha='center', va='center', fontsize=12,
                        color='#7a5a00', fontweight='bold')

    draw_seq(ax_s0, ordered0, pseudo_rows, set(), n,
             has_eor=False, show_row_labels=False, col_idx_map=ci0, first_occ=first0, left_offset=-0.3, max_rows=2)
    draw_seq(ax_s1, ordered3, rows3, row_starts3, n,
             has_eor=True, show_row_labels=False, col_idx_map=ci3, first_occ=first3, left_offset=0.2, max_rows=2)

    legend_items = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#888',
               markeredgecolor='black', markersize=8, label='first occurrence'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#888',
               markeredgecolor='black', markersize=7, label='repeated token'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#c0a23a',
               markeredgecolor='black', markersize=10, label='end-of-row-token'),
    ]
    fig.legend(handles=legend_items, loc='center', ncol=3,
               fontsize=11, framealpha=0.9, bbox_to_anchor=(0.5, 0.52))

    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f'→ {out}')


def try_float(x):
    try:
        return float(x)
    except:
        return None


def try_int(x):
    try:
        return int(x)
    except:
        return None


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'centered_blades_cleaned.pt'
    midx_arg = sys.argv[2] if len(sys.argv) > 2 else '6'

    args = sys.argv[3:] if len(sys.argv) > 3 else []
    alpha, nf, out = 0.30, None, 'comparison.png'
    for a in args:
        if nf is None and try_int(a) is not None:
            nf = try_int(a)
        elif try_float(a) is not None:
            alpha = try_float(a)
        else:
            out = a

    if ':' in midx_arg:
        a, b = midx_arg.split(':')
        indices = list(range(int(a), int(b)))
    else:
        indices = [int(midx_arg)]

    for i in indices:
        o = f'comparison_{i}.png' if len(indices) > 1 else out
        viz_paper(path, i, alpha, n_faces=None, out=o)
