"""Hexa-row demo: click-through animation of the hexa-row block tokenization.

Block geometry (hexaeder blocks, one step per block) revealed in row-encoded
tokenization order: row-head blocks emit 8 verts (entry+exit ring), continuations
only the exit ring (4 verts). EOR after every row, layers bottom-to-top.
Token bar: one slot per block (4 dots per vertex) + EOR slots between rows.

Usage:
    uv run python tokenization_animation_3d.py --hexa --idx 1 \
        --data data/polytron_data_3d_smoke.pt --out-dir embeds3d
"""
from __future__ import annotations

import torch

from tokenization_animation_3d import (_golden,
                                       render_html, order_color, step_color,
                                       INK, GRID_EDGE, ACTIVE)
import tokenization_animation_3d as base
from hexa_row_tokenizer import HexaRowTokenizer, build_row_plan, roundtrip

HEX_TRIS = ((0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7),
            (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
            (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7))


def _block_tris(verts8):
    i, j, k = [], [], []
    for a, b, c in HEX_TRIS:
        i.append(verts8[a]); j.append(verts8[b]); k.append(verts8[c])
    return i, j, k


def build_steps_hexa(mesh, tok: HexaRowTokenizer, max_blocks: int | None = None,
                     start_rule: str = 'min_theta'):
    """Steps per block in row-encoded emission order (rows reioniert)."""
    blks = mesh['faces'].T.tolist()
    rows, emit = build_row_plan(blks, mesh['vertices_cartesian'],
                                edges=mesh.get('edge_index'), start_rule=start_rule)
    vp = mesh['vertices_polar']
    steps = []
    for r, row in enumerate(rows):
        for bi, b in enumerate(row):
            ids = list(emit[b] if bi == 0 else emit[b][4:8])
            toks = []
            for vid in ids:
                toks += tok.quant_vertex_tokens(int(vid), vp)
            steps.append(dict(blk=blks[b], emit=ids, row=r,
                              row_end=(bi == len(row) - 1),
                              head=(bi == 0), n_tok=4 * len(ids), toks=toks))
        if max_blocks is not None and len(steps) >= max_blocks:
            steps = [s for s in steps if s['row'] <= r]
            rows = rows[:r + 1]
            break
    return rows, steps


def build_slots_hexa(steps):
    """('sos',) / ('block', i) / ('eor',) / ('cont',) slots for the token bar."""
    slots = [("sos",)]
    for i, st in enumerate(steps):
        slots.append(("block", i))
        if st["row_end"] and i != len(steps) - 1:
            slots.append(("eor",))
    slots.append(("cont",))
    return slots


def build_figure_hexa(mesh, steps, slots, total_blocks: int):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n_steps = len(steps)
    n_slots = len(slots)
    V = mesh['vertices_cartesian']
    x, y, z = V[:, 0].tolist(), V[:, 1].tolist(), V[:, 2].tolist()

    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.74, 0.26], vertical_spacing=0.10,
        specs=[[{"type": "scatter3d"}], [{"type": "xy"}]])

    # static all-blocks ghost + global edges
    gi, gj, gk = [], [], []
    for b in mesh['faces'].T.tolist():
        ti, tj, tk = _block_tris([int(v) for v in b])
        gi += ti; gj += tj; gk += tk
    fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=gi, j=gj, k=gk,
                            color="#b9b9b9", opacity=0.08, flatshading=True,
                            hoverinfo="skip"), row=1, col=1)
    edges = set()
    for b in mesh['faces'].T.tolist():
        b = [int(v) for v in b]
        for a, c in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7),
                     (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)):
            edges.add((min(b[a], b[c]), max(b[a], b[c])))
    ex, ey, ez = [], [], []
    for u, v in sorted(edges):
        ex += [x[u], x[v], None]
        ey += [y[u], y[v], None]
        ez += [z[u], z[v], None]
    fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines",
                               line=dict(color=GRID_EDGE, width=2), hoverinfo="skip"),
                  row=1, col=1)

    # Achsenkreuz x/y/z durch den Ursprung (center=0), gestrichelt + Label
    L = 1.15 * max(max(abs(float(c)) for c in V[:, k]) for k in range(3))
    fig.add_trace(go.Scatter3d(
        x=[-L, L, None, 0, 0, None, 0, 0, 0],
        y=[0, 0, None, -L, L, None, 0, 0, 0],
        z=[0, 0, None, 0, 0, None, 0, 0, L],
        mode="lines", line=dict(width=3, dash="dot", color="#9aa0a6"),
        hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter3d(
        mode="text", text=["x", "y", "z"], textfont=dict(size=12, color="#555555"),
        x=[L, 0, 0], y=[0, L, 0], z=[0, 0, L],
        hoverinfo="skip"), row=1, col=1)

    # dynamic placeholders: n_steps block meshes + active + corners + overview + dots
    empty3d = lambda: go.Mesh3d(x=x, y=y, z=z, i=[], j=[], k=[], opacity=0.0)
    for _ in range(n_steps):
        fig.add_trace(empty3d(), row=1, col=1)
    fig.add_trace(empty3d(), row=1, col=1)   # active block
    fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="markers+text", text=[],
                               textposition="top center",
                               textfont=dict(size=9, color=INK),
                               marker=dict(size=6, color=[], symbol=[],
                                           line=dict(color=INK, width=1)),
                               hoverinfo="text"), row=1, col=1)
    fig.add_trace(empty3d(), row=1, col=1)   # overview
    fig.data[-1].colorscale = [[0.0, "#3b4cc0"], [0.5, "#f2f2f2"], [1.0, "#b40426"]]
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers", marker=dict(size=6, color=[]),
                             hoverinfo="skip"), row=2, col=1)

    def dyn_data(fidx):
        data = []
        for i in range(n_steps):                       # revealed blocks
            if i < fidx:
                ti, tj, tk = _block_tris(steps[i]["blk"])
                data.append(go.Mesh3d(x=x, y=y, z=z, i=ti, j=tj, k=tk,
                                      opacity=0.55, color=order_color(i, n_steps),
                                      flatshading=True, hoverinfo="skip"))
            else:
                data.append(empty3d())
        if 0 < fidx <= n_steps:                        # active block
            ti, tj, tk = _block_tris(steps[min(fidx, n_steps) - 1]["blk"])
            data.append(go.Mesh3d(x=x, y=y, z=z, i=ti, j=tj, k=tk,
                                  color=base.ACTIVE, opacity=0.45,
                                  flatshading=True, hoverinfo="skip"))
        else:
            data.append(empty3d())
        seen = {}                                      # vertex markers
        cx, cy, cz, cc, ct, cs, ht = [], [], [], [], [], [], []
        for i in range(fidx):
            for k, vv in enumerate(steps[i]["emit"]):
                first = vv not in seen
                if first:
                    seen[vv] = i
                cx.append(x[vv]); cy.append(y[vv]); cz.append(z[vv])
                cc.append(_golden(seen[vv]))
                cs.append("circle" if first else "square")
                ct.append(str(k) if i == fidx - 1 else "")
                ht.append("v%d · B%d · pos %d" % (vv, i + 1, k))
        data.append(go.Scatter3d(x=cx, y=cy, z=cz, mode="markers+text", text=ct,
                                 textposition="top center",
                                 textfont=dict(size=10, color=INK),
                                 marker=dict(size=6, color=cc, symbol=cs,
                                             line=dict(color=INK, width=1)),
                                 hovertext=ht, hoverinfo="text"))
        if fidx >= n_steps:                            # overview final frame
            oi, oj, ok_, it_all = [], [], [], []
            for i, st in enumerate(steps):
                ti, tj, tk = _block_tris(st["blk"])
                oi += ti; oj += tj; ok_ += tk
                it_all += [i] * len(ti)
            data.append(go.Mesh3d(x=x, y=y, z=z, i=oi, j=oj, k=ok_,
                                  intensity=it_all, opacity=0.85,
                                  flatshading=True, showscale=False,
                                  hoverinfo="skip"))
        else:
            data.append(empty3d())
        slot_of = {i: slots.index(("block", i)) for i in range(n_steps)}
        seen_till = set()                              # Sequenz-Strip: 1 Glyph je Vertex
        bx, by, bs, bc, bt = [], [], [], [], []
        for i in range(fidx):
            st = steps[i]
            n_e = len(st["emit"])
            for k, vv in enumerate(st["emit"]):
                bx.append(slot_of[i] + 0.1 + (0.8 * k / max(n_e - 1, 1)))
                by.append(0.5)
                bs.append("square" if vv in seen_till else "circle")
                bc.append(step_color(i, n_steps))
                bt.append("v%d · B%d · pos %d" % (vv, i + 1, k))
                seen_till.add(vv)
        data.append(go.Scatter(x=bx, y=by, mode="markers",
                               marker=dict(symbol=bs, size=8, color=bc,
                                           line=dict(color=INK, width=1)),
                               hovertext=bt, hoverinfo="text"))
        return data

    total_toks = sum(st["n_tok"] for st in steps) + len({s['row'] for s in steps})
    n_heads = sum(1 for st in steps if st["head"])
    dup_saved = 4 * (n_steps - n_heads)
    frames = []
    max_step = n_steps + 1
    for tlen in range(max_step):
        fidx = tlen
        a = steps[min(fidx, n_steps) - 1] if fidx > 0 else None
        if fidx >= n_steps:
            info = (f"Übersicht: {total_blocks} Blöcke · "
                    f"{len({st['row'] for st in steps})} Reihen · "
                    f"{total_toks} tokens · dup-verts gespart: {dup_saved}")
        elif a is not None:
            info = (f"block {fidx}/{total_blocks}  ·  row {a['row'] + 1} "
                    f"({'head, 8 verts' if a['head'] else 'exit ring, 4 verts'})"
                    f"  ·  tokens this step: {a['n_tok']}"
                    + ("  ·  end of row -> EOR" if a["row_end"] else ""))
        else:
            info = (f"block 0/{total_blocks} ({total_toks} tokens total) — "
                    "press \u2192 to tokenize")
        layout = go.Layout(annotations=[go.layout.Annotation(
            xref="paper", yref="paper", x=0.0, y=1.0, showarrow=False, text=info,
            font=dict(size=13, color=INK), xanchor="left", yanchor="top")])
        frames.append(go.Frame(name=str(tlen), data=dyn_data(fidx), layout=layout))

    shapes, bar_annos = [], []
    for jx, slot in enumerate(slots):
        w = 0.06 if slot[0] == "eor" else 0.90
        shapes.append(go.layout.Shape(type="rect", xref="x", yref="y",
                                      x0=jx + (0.5 - w / 2), x1=jx + (0.5 + w / 2),
                                      y0=-0.06, y1=1.06,
                                      line=dict(width=1 if slot[0] == "eor" else 1,
                                                color="#c8c2bd"),
                                      fillcolor="rgba(0,0,0,0)"))
        anno = {"sos": "SOS", "eor": "EOR", "cont": "..."}.get(slot[0])
        if anno is None:
            anno = f"B{slot[1] + 1}"
        bar_annos.append(go.layout.Annotation(xref="x", yref="y", x=jx + 0.5, y=-0.14,
                                              xanchor="center", yanchor="top",
                                              showarrow=False,
                                              font=dict(size=12, color=INK),
                                              text=anno))
    pad = [go.layout.Annotation(xref="paper", yref="paper", x=-10, y=-10,
                                showarrow=False, text="") for _ in bar_annos]
    for fr in frames:
        fr.layout.annotations = pad + [fr.layout.annotations[0]]

    static_json = [ax.to_plotly_json() for ax in fig.data[:4]]

    def slot_label(s):
        lbl = {"sos": "SOS", "eor": "EOR", "cont": "..."}.get(s[0])
        return lbl if lbl is not None else f"B{s[1] + 1}"

    slot_text = [slot_label(s) for s in slots]
    labels_trace = go.Scatter(x=[j + 0.5 for j in range(n_slots)], y=[-0.12] * n_slots,
                              text=slot_text, mode="text",
                              textfont=dict(size=14, color=INK),
                              hoverinfo="skip").to_plotly_json()
    frames_json = [
        {"data": static_json + [labels_trace] + [tr.to_plotly_json() for tr in fr.data],
         "anno": fr.layout.annotations[-1].to_plotly_json(),
         "annoIdx": len(fig.layout.annotations)}
        for fr in frames]
    fig.frames = None

    fig.layout.shapes = shapes
    fig.layout.annotations = bar_annos + [go.layout.Annotation(
        xref="paper", yref="paper", x=-10, y=-10, showarrow=False,
        text="", font=dict(size=13, color=INK))]
    fig.update_xaxes(visible=False, range=[0, max(n_slots, 2)], row=2, col=1)
    fig.update_yaxes(visible=False, range=[-0.35, 1.15], row=2, col=1)
    fig.update_layout(
        height=860, margin=dict(l=10, r=10, t=30, b=34),
        template="plotly_white", font=dict(color=INK), showlegend=False,
        uirevision="keep",
        scene=dict(aspectmode="data",
                   xaxis=dict(visible=False), yaxis=dict(visible=False),
                   zaxis=dict(visible=False)),
    )
    return fig, max_step - 1, frames_json


def generate_hexa(out_dir, path, idx, start_rule='min_theta'):
    from hexa_row_tokenizer import HexaRowTokenizer

    data = torch.load(path, weights_only=False)
    mesh = data[idx]
    rpol = torch.cat([d['vertices_polar'][:, 0] for d in data])
    zpol = torch.cat([d['vertices_polar'][:, 2] for d in data])
    rb = (float(rpol.min() * 0.98), float(rpol.max() * 1.02))
    zb = (float(zpol.min() - 0.02 * abs(zpol.min()) - 0.01 if zpol.min() < 0
                else zpol.min() * 0.98),
          float(zpol.max() + 0.02 * abs(zpol.max())))
    tok = HexaRowTokenizer(r_bounds=rb, z_bounds=zb)
    rows, steps = build_steps_hexa(mesh, tok, start_rule=start_rule)
    slots = build_slots_hexa(steps)
    import os
    os.makedirs(out_dir, exist_ok=True)
    fig, max_idx, frames_json = build_figure_hexa(
        mesh, steps, slots, mesh['faces'].shape[1])
    out = os.path.join(out_dir, f"tokenization_3d_hexarow_{start_rule}_idx{idx}.html")
    render_html(fig, out, max_idx, frames_json)
    print(f"[hexarow] blocks={len(steps)} rows={len(rows)} slots={len(slots)} "
          f"frames={max_idx + 1}")
    print(f"-> {out}")
