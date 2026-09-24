"""Interactive 3D tokenization demo (plotly HTML), 3D port of tokenization_animation.py (2D).

Click-through animation of the 3D hexa-block surface tokenization:
  row 1: quad mesh (3D). Faces are revealed in tokenization order:
         strat 1 = dir_class rows, full 12-token emission (Meshtron-style face ordering)
         strat 2 = row-encoded emission (row start 12 tokens, mid-row 6, eor per row,
                   layers bottom-to-top)
  row 2: token bar -- one slot per face (12/6 token dots per slot), SOS / EOR / cont boxes.

Usage:
    uv run python tokenization_animation_3d.py \
        --data data/quadtron_data_3d_smoke.pt --idx 1 --max-faces 12 --out-dir embeds3d
"""
from __future__ import annotations

import argparse
import colorsys
import json
import re
import os

import plotly.graph_objects as go
import numpy as np
import torch
from plotly.subplots import make_subplots

from meshtron.data.tokenizer_v2 import Tokenizer2D

Q = 512                      # quantization levels (matches dataset tokenization)

ORDER_STOPS = [(0.0, (59, 76, 192)), (0.5, (242, 242, 242)), (1.0, (180, 4, 38))]
STEP_FROM = (0, 136, 176)    # teal
STEP_TO = (214, 0, 108)      # magenta
ACTIVE = "#d6006c"
GRID_EDGE = "#9a9490"
INK = "#201e1d"


def _golden(i: int) -> str:
    h = (i * 0.618033988749895) % 1.0
    r, g, b = colorsys.hls_to_rgb(h, 0.55, 0.85)
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"


def order_color(i: int, n: int) -> str:
    t = i / max(n - 1, 1)
    lo, hi = ORDER_STOPS[0][1], ORDER_STOPS[1][1]
    for j in range(len(ORDER_STOPS) - 1):
        if ORDER_STOPS[j][0] <= t <= ORDER_STOPS[j + 1][0]:
            f = (t - ORDER_STOPS[j][0]) / (ORDER_STOPS[j + 1][0] - ORDER_STOPS[j][0])
            lo, hi = ORDER_STOPS[j][1], ORDER_STOPS[j + 1][1]
    c = tuple(int(l + (h - l) * f) for l, h in zip(lo, hi))
    return f"rgb({c[0]},{c[1]},{c[2]})"


def step_color(i: int, n: int) -> str:
    t = i / max(n - 1, 1)
    c = tuple(int(a + (b - a) * t) for a, b in zip(STEP_FROM, STEP_TO))
    return f"rgb({c[0]},{c[1]},{c[2]})"


def load_sample(path: str, idx: int):
    data = torch.load(path, weights_only=False)
    s = data[idx]
    vertices = s.x.detach().cpu().numpy().astype(float)
    quads = s.faces.detach().cpu().numpy().astype(int)
    dc = getattr(s, "dir_class", None)
    if dc is None:                      # 2D domain meshes carry no direction class
        dir_class = np.zeros(quads.shape[1], dtype=int)
    else:
        dir_class = dc.detach().cpu().numpy().astype(int)
    return len(data), vertices, quads, dir_class


def build_steps(tok: Tokenizer2D, vertices, quads, dir_class, strategy: int,
                max_faces: int, dim: int = 3):
    """Tokenization steps for the first ``max_faces`` faces (row-boundary clipped).

    Returns (rows, steps). Each step: face corner vertex ids, dir_class label,
    row index, tokens emitted at this step (strat2: 12 row-start / 6 mid-row).
    """
    vt = torch.from_numpy(vertices)
    qt = torch.from_numpy(quads)
    dc_t = torch.from_numpy(dir_class)
    sorted_quads, rows = tok._order_quads(vt, qt, dc_t)
    # dim=2 returns [n,4], dim=3 returns [4,n] - normalise to [4,n]
    if sorted_quads.shape[0] != 4 and sorted_quads.shape[1] == 4:
        sorted_quads = sorted_quads.T
    if rows is None:
        # lexicographic emission has no row structure - one continuous run
        rows = [(0, int(sorted_quads.shape[1]))]
    rows = [(s, e) for s, e in rows]

    kept, used = [], 0
    for s, e in rows:
        if used >= max_faces:
            break
        k = min(e - s, max_faces - used)
        kept.append((s, s + k))
        used += k
    rows = kept
    total_faces = quads.shape[1]

    # ordered position -> original face id (dir_class lookup)
    orig_of = {}
    for i, f in enumerate(quads.T.tolist()):
        orig_of.setdefault(frozenset(f), i)

    coord_seq = tok._quads_to_coords(vt, sorted_quads)
    quant, _bounds = tok._quantize_coords(coord_seq)

    steps = []
    for r, (s, e) in enumerate(rows):
        for fi in range(s, e):
            face = sorted_quads[:, fi].tolist()
            # Only the row-compressed strategies drop the two shared vertices;
            # lexicographic emission restates all four corners of every face.
            v_start = (0 if fi == s else 2) if strategy in (2, 3) else 0
            verts = list(range(v_start, 4))
            toks = []
            for v in verts:
                toks += [int(c) for c in quant[fi * 4 + v].tolist()]
            n_tok = len(verts) * dim
            gfi = orig_of[frozenset(face)]
            steps.append(dict(face=face, verts=verts, row=r, row_end=(fi == e - 1),
                              label=int(dir_class[gfi]), n_tok=n_tok, toks=toks))
    return rows, steps


def build_slots(steps, strategy: int):
    """('sos',) / ('face', i) / ('eor',) / ('cont',) slot list for the token bar."""
    slots = [("sos",)]
    for i, st in enumerate(steps):
        slots.append(("face", i))
        if strategy == 2 and st["row_end"] and i != len(steps) - 1:
            slots.append(("eor",))
    slots.append(("cont",))
    return slots


def build_figure(vertices, quads, steps, slots, strategy: int, total_faces: int):
    n_steps = len(steps)
    n_slots = len(slots)
    x, y, z = vertices[:, 0].tolist(), vertices[:, 1].tolist(), vertices[:, 2].tolist()
    faces = quads.T  # [F', 4] rows

    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.74, 0.26], vertical_spacing=0.10,
        specs=[[{"type": "scatter3d"}], [{"type": "xy"}]])

    # ---- static mesh: all quads light gray + unique edges --------------------
    gray = "#b9b9b9"
    i0 = [int(q[0]) for q in faces]
    i2 = [int(q[2]) for q in faces]
    j1 = [int(q[1]) for q in faces]
    k3 = [int(q[3]) for q in faces]
    fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i0, j=j1, k=i2, color=gray, opacity=0.10,
                            flatshading=True, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i2, j=j1, k=k3, color=gray, opacity=0.10,
                            flatshading=True, hoverinfo="skip"), row=1, col=1)
    edges = set()
    for f in faces:
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
            u, v = sorted((int(f[a]), int(f[b])))
            edges.add((u, v))
    ex, ey, ez = [], [], []
    for u, v in sorted(edges):
        ex += [x[u], x[v], None]
        ey += [y[u], y[v], None]
        ez += [z[u], z[v], None]
    fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines",
                               line=dict(color=GRID_EDGE, width=2), hoverinfo="skip"), row=1, col=1)

    # ---- dynamic traces (fixed count/order): n_steps revealed faces,
    # 1 active face, 1 corner markers, 1 overview, 1 token-bar dots -----------
    empty3d = lambda: go.Mesh3d(x=x, y=y, z=z, i=[], j=[], k=[], opacity=0.0)
    for _ in range(n_steps):
        fig.add_trace(empty3d(), row=1, col=1)
    fig.add_trace(empty3d(), row=1, col=1)   # active
    fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="markers+text", text=[],
                               textposition="top center",
                               textfont=dict(size=9, color=INK),
                               marker=dict(size=5, color=[], symbol=[],
                                           line=dict(color=INK, width=1)),
                               hoverinfo="text"), row=1, col=1)
    fig.add_trace(empty3d(), row=1, col=1)   # overview
    fig.data[-1].colorscale = [[0.0, "#3b4cc0"], [0.5, "#f2f2f2"], [1.0, "#b40426"]]
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers", marker=dict(size=6, color=[]),
                             hoverinfo="skip"), row=2, col=1)  # token dots

    def dyn_data(fidx: int):
        data = []
        for i in range(n_steps):                       # revealed faces
            if i < fidx:
                f = steps[i]["face"]
                data.append(go.Mesh3d(x=x, y=y, z=z, i=[f[0], f[1], f[2]],
                                      j=[f[1], f[2], f[3]], k=[f[2], f[3], f[0]],
                                      opacity=0.55, color=order_color(i, n_steps),
                                      flatshading=True, hoverinfo="skip"))
            else:
                data.append(empty3d())
        if 0 < fidx <= n_steps:                        # active face
            f = steps[min(fidx, n_steps) - 1]["face"]
            data.append(go.Mesh3d(x=x, y=y, z=z, i=[f[0], f[1], f[2]],
                                  j=[f[1], f[2], f[3]], k=[f[2], f[3], f[0]],
                                  color=ACTIVE, opacity=0.45, flatshading=True,
                                  hoverinfo="skip"))
        else:
            data.append(empty3d())
        seen = {}                                      # corner markers
        cx, cy, cz, cc, ct, cs = [], [], [], [], [], []
        for i in range(fidx):
            st = steps[i]
            for v_pos in range(4):
                v_start = st["verts"][0]
                if v_pos < v_start:
                    continue
                vv = st["face"][v_pos]
                if vv not in seen:
                    seen[vv] = i
                cx.append(x[vv]); cy.append(y[vv]); cz.append(z[vv])
                cc.append(_golden(seen[vv]))
                cs.append("circle" if seen[vv] == i else "square")
                ct.append("v%d" % vv)
        data.append(go.Scatter3d(x=cx, y=cy, z=cz, mode="markers+text", text=ct,
                                 textposition="top center", textfont=dict(size=9, color=INK),
                                 marker=dict(size=5, color=cc, symbol=cs,
                                             line=dict(color=INK, width=1)),
                                 hoverinfo="text"))
        if fidx >= n_steps:                            # overview (final frame only)
            fi_all, fj_all, fk_all, it_all = [], [], [], []
            for i, st in enumerate(steps):
                f = st["face"]
                fi_all += [f[0], f[1], f[2]]
                fj_all += [f[1], f[2], f[3]]
                fk_all += [f[2], f[3], f[0]]
                it_all += [i] * 3
            data.append(go.Mesh3d(x=x, y=y, z=z, i=fi_all, j=fj_all, k=fk_all,
                                  intensity=it_all, opacity=0.85, flatshading=True,
                                  showscale=False, hoverinfo="skip"))
        else:
            data.append(empty3d())
        slot_of = {i: slots.index(("face", i)) for i in range(n_steps)}
        # One marker per emitted vertex - not per token. Circle = this vertex is
        # stated for the first time, square = it was already emitted earlier and
        # is being repeated. Same convention and colours as the mesh markers.
        bx, by, bc, bs = [], [], [], []
        seen_b = {}
        for i in range(fidx):
            st = steps[i]
            v_start = st["verts"][0]
            for v_pos in range(4):
                if v_pos < v_start:
                    continue
                vv = st["face"][v_pos]
                if vv not in seen_b:
                    seen_b[vv] = i
                n_emit = 4 - v_start
                k = v_pos - v_start
                bx.append(slot_of[i] + 0.18 + 0.64 * (k / max(n_emit - 1, 1)))
                by.append(0.40)
                bc.append(_golden(seen_b[vv]))
                bs.append("circle" if seen_b[vv] == i else "square")
        data.append(go.Scatter(x=bx, y=by, mode="markers",
                               marker=dict(size=9, color=bc, symbol=bs,
                                           line=dict(color=INK, width=1)),
                               hoverinfo="skip"))
        return data

    frames = []
    max_step = n_steps + 1            # last step = overview frame
    for tlen in range(max_step):
        fidx = tlen
        a = steps[min(fidx, n_steps) - 1] if fidx > 0 else None
        if a is not None:
            acc = sum(st["n_tok"] for st in steps[:min(fidx, n_steps)])
            info = (f"face {fidx}/{total_faces}"
                    f"  ·  accumulated tokens: {acc}")
        else:
            info = f"face 0/{total_faces} — press → to tokenize"
        layout = go.Layout(annotations=[go.layout.Annotation(
            xref="paper", yref="paper", x=0.0, y=1.0, showarrow=False, text=info,
            font=dict(size=20, color=INK), xanchor="left", yanchor="top")])
        frames.append(go.Frame(name=str(tlen), data=dyn_data(fidx), layout=layout))

    # static traces package empty dynamic placeholders; the page's initial
    # render() call jumps to frame "0" and fills them
    # ---- token bar static boxes/labels --------------------------------------

    # ---- token bar static boxes/labels --------------------------------------
    shapes, bar_annos = [], []
    for jx, slot in enumerate(slots):
        shapes.append(go.layout.Shape(type="rect", xref="x", yref="y",
                                      x0=jx + 0.10, x1=jx + 0.90, y0=-0.06, y1=1.06,
                                      line=dict(width=1, color="#c8c2bd"),
                                      fillcolor="rgba(0,0,0,0)"))
        anno = {"sos": "SOS", "eor": "EOR", "cont": "..."}.get(slot[0])
        if anno is None:
            anno = str(slot[1] + 1)
        bar_annos.append(go.layout.Annotation(xref="x", yref="y", x=jx + 0.5, y=-0.14,
                                              xanchor="center", yanchor="top",
                                              showarrow=False,
                                              font=dict(size=13, color=INK), text=anno))
    pad = [go.layout.Annotation(xref="paper", yref="paper", x=-10, y=-10,
                                showarrow=False, text="") for _ in bar_annos]
    for fr in frames:
        fr.layout.annotations = pad + [fr.layout.annotations[0]]

    # frames ship as JSON for Plotly.react (animate() mismatches trace order in
    # mixed mesh3d/scatter figures); fig itself stays on placeholders
    static_json = [ax.to_plotly_json() for ax in fig.data[:3]]
    def slot_label(s):
        lbl = {"sos": "SOS", "eor": "EOR", "cont": "..."}.get(s[0])
        return lbl if lbl is not None else str(s[1] + 1)

    slot_text = [slot_label(s) for s in slots]
    labels_trace = go.Scatter(x=[j + 0.5 for j in range(n_slots)], y=[-0.12] * n_slots,
                              text=slot_text, mode="text",
                              textfont=dict(size=15, color=INK),
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


def render_html(fig, out_path: str, max_idx: int, frames_json, info_div=False):
    div_id = "tokviz3d"
    plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn",
                            default_width="100%", default_height="560px")
    m = re.search(r'Plotly\.animate\(\'([^\']+)\'',
                  plot_html) or re.search(r'playly.*id="([^"]+)"', plot_html)
    mm = re.search(r'<div[^>]*class="plotly-graph-div[^"]*"[^>]*id="([^"]+)"', plot_html) \
        or re.search(r'id="([^"]+)" class="plotly-graph-div', plot_html) \
        or m
    div_src = mm.group(1) if mm else None
    if div_src and div_src != div_id:
        plot_html = re.sub(rf"Plotly\.animate\('{div_src}', null\);?", "", plot_html)
        plot_html = plot_html.replace(div_src, div_id)
    if f'id="{div_id}"' not in plot_html:
        plot_html = plot_html.replace('class="plotly-graph-div"',
                                      f'id="{div_id}" class="plotly-graph-div"', 1)
    frames_json_src = json.dumps(frames_json).replace("</", "<\\/")

    html = """<!doctype html>
<html><head><meta charset="utf-8">
<style>
  body { margin:0; background:#fbfaf8; font-family: ui-sans-serif, Helvetica, Arial, sans-serif; }
  .wrap { max-width: 1840px; margin: 0 auto; padding: 8px 12px 18px; }
  #controls { display:flex; gap:10px; padding:6px 4px 4px; }
  button.step { flex:1; height:76px; font-size:30px; border:1.5px solid #9a9490;
    border-radius:10px; background:#eae9e9; color:#201e1d; cursor:pointer;
    transition:border-color .15s ease, color .15s ease; }
  button.step:hover:not(:disabled) { border-color:#d6006c; color:#d6006c; }
  button.step:disabled { opacity:.35; cursor:default; }
  .progress { height:8px; border-radius:5px; margin:2px 4px 0;
    background:linear-gradient(90deg,#0088b0,#d6006c); position:relative; overflow:hidden; }
  .progress .mask { position:absolute; right:0; top:0; height:100%; width:0%;
    background:#fbfaf8; transition: width .18s ease; }
</style></head>
<body><div class="wrap">
TOKINFO_TOP_HERE
TOKROW_OPEN_HERE
PLOTLY_HTML_HERE
TOKINFO_SIDE_HERE
TOKROW_CLOSE_HERE
<div class="progress"><div id="progressMask" class="mask"></div></div>
<div id="controls">
  <button id="prevBtn" class="step">&larr;</button>
  <button id="nextBtn" class="step">&rarr;</button>
</div>
<script>var TOKFRAMES = FRAMES_JSON;</script>
</div>
<script>
var gd;
var qs = new URLSearchParams(window.location.search);
var idx = Math.min(parseInt(qs.get('step') || '0', 10) || 0, MAXI), maxIdx = MAXI;
function render() {
  if (!gd) gd = document.getElementById('TOKDIV');
  var f = TOKFRAMES[idx];
  var ti = document.getElementById('tokinfo');
  var L = JSON.parse(JSON.stringify(gd.layout));
  if (ti) { ti.innerHTML = f.anno.text; }
  else { if (!L.annotations) L.annotations = []; L.annotations[f.annoIdx] = f.anno; }
  Plotly.react(gd, f.data, L, {displayModeBar: true, responsive: false, scrollZoom: true});
  document.getElementById('progressMask').style.width = (100 - (idx / maxIdx) * 100) + '%';
  document.getElementById('prevBtn').disabled = (idx === 0);
  document.getElementById('nextBtn').disabled = (idx === maxIdx);
}
document.getElementById('prevBtn').addEventListener('click', function(){ if (idx > 0) { idx -= 1; render(); } });
document.getElementById('nextBtn').addEventListener('click', function(){ if (idx < maxIdx) { idx += 1; render(); } });
window.addEventListener('keydown', function(e){
  if (e.key === 'ArrowRight' && idx < maxIdx) { idx += 1; render(); }
  if (e.key === 'ArrowLeft' && idx > 0) { idx -= 1; render(); }
});
window.addEventListener('load', function(){
  gd = document.getElementById('TOKDIV');
  if (gd) render();
});
</script>
</body></html>"""
    top = side = row_open = row_close = ''
    if info_div in ('left', 'right'):
        plot_html = ('<div style="flex:1 1 auto; min-width:0;">'
                     + plot_html + '</div>')
        row_open = ('<div style="display:flex; gap:12px;'
                    ' align-items:flex-start;">')
        row_close = '</div>'
        side = ('<div id="tokinfo" style="flex:0 0 auto; white-space:nowrap;'
                ' line-height:1.5; padding:24px 0 0 0; font-size:18px;'
                ' color:#201e1d;"></div>')
        if info_div == 'left':
            row_open, side = row_open + side, ''
    elif info_div:
        top = ('<div id="tokinfo" style="min-height:26px; padding:6px 4px 2px;'
               ' font-size:20px; white-space:nowrap; color:#201e1d;"></div>')
    html = (html.replace("TOKINFO_TOP_HERE", top)
            .replace("TOKROW_OPEN_HERE", row_open)
            .replace("PLOTLY_HTML_HERE", plot_html)
            .replace("TOKINFO_SIDE_HERE", side)
            .replace("TOKROW_CLOSE_HERE", row_close)
            .replace("TOKDIV", div_id).replace("MAXI", str(max_idx)).replace(
                "FRAMES_JSON", frames_json_src))
    with open(out_path, "w") as f:
        f.write(html)



INSET = 0.26          # how far the per-face vertex markers sit inside the quad
SLOTS_PER_ROW = 6     # face slots per line in the token strip


def _inset(px, py, face, v_pos, t=INSET):
    """Marker position for one face corner, pulled towards the face centroid.

    The markers belong to a *face*, not to the mesh vertex: the same vertex shows
    up once per adjacent face, and drawing them on the shared corner would stack
    them on top of each other. Insetting keeps each face's four markers legible
    and visibly grouped - the convention the earlier versions used.
    """
    cx = sum(px[v] for v in face) / 4.0
    cy = sum(py[v] for v in face) / 4.0
    v = face[v_pos]
    return px[v] + t * (cx - px[v]), py[v] + t * (cy - py[v])


def _slot_cell(j: int):
    """Grid cell (column, row) of slot j in the wrapped token strip."""
    return j % SLOTS_PER_ROW, j // SLOTS_PER_ROW


def build_figure_2d(vertices, quads, steps, slots, strategy: int, total_faces: int):
    """Planar variant: real 2D axes instead of a top-down 3D scene.

    A scatter3d scene fits the bounding *sphere* of its box into the viewport, so
    a flat square mesh never fills more than ~70 % of the panel no matter what the
    camera does. On 2D axes the ranges decide the size outright.
    """
    n_steps, n_slots = len(steps), len(slots)
    px, py = vertices[:, 0].tolist(), vertices[:, 1].tolist()
    faces = quads.T

    fig = make_subplots(rows=2, cols=1, row_heights=[0.76, 0.24],
                        vertical_spacing=0.07,
                        specs=[[{"type": "xy"}], [{"type": "xy"}]])

    def ring(face):
        idx = list(face) + [face[0]]
        return [px[int(v)] for v in idx], [py[int(v)] for v in idx]

    # ---- static: every quad faint, plus the edge graph ----------------------
    ax_, ay_ = [], []
    for f in faces:
        rx, ry = ring(f)
        ax_ += rx + [None]
        ay_ += ry + [None]
    fig.add_trace(go.Scatter(x=ax_, y=ay_, mode="lines", fill="toself",
                             fillcolor="rgba(185,185,185,0.10)",
                             line=dict(color=GRID_EDGE, width=1),
                             hoverinfo="skip"), row=1, col=1)

    empty2d = lambda: go.Scatter(x=[], y=[], mode="lines", fill="toself",
                                 line=dict(width=0), hoverinfo="skip")
    for _ in range(n_steps):
        fig.add_trace(empty2d(), row=1, col=1)
    fig.add_trace(empty2d(), row=1, col=1)                      # active face
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines+markers", hoverinfo="skip",
                             line=dict(color=INK, width=1.4),
                             marker=dict(size=9, color=INK, symbol="arrow",
                                         angleref="previous")),
                  row=1, col=1)                                  # emission order
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers", hoverinfo="skip",
                             marker=dict(size=11, color=[], symbol=[],
                                         line=dict(color=INK, width=1))),
                  row=1, col=1)                                  # corner markers
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", fill="toself",
                             fillcolor="rgba(128,128,128,0.32)",
                             line=dict(color="#6f6f6f", width=1.5, dash="dash"),
                             hoverinfo="skip"), row=2, col=1)    # grey "deletable" backing
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers",
                             marker=dict(size=9, color=[], symbol=[],
                                         line=dict(color=INK, width=1)),
                             hoverinfo="skip"), row=2, col=1)    # vertex bar

    def dyn_data(fidx: int):
        data = []
        for i in range(n_steps):                                 # revealed faces
            if i < fidx:
                rx, ry = ring(steps[i]["face"])
                data.append(go.Scatter(x=rx, y=ry, mode="lines", fill="toself",
                                       fillcolor=order_color(i, n_steps),
                                       opacity=0.62, line=dict(width=0),
                                       hoverinfo="skip"))
            else:
                data.append(empty2d())
        if 0 < fidx <= n_steps:                                  # active face
            rx, ry = ring(steps[min(fidx, n_steps) - 1]["face"])
            data.append(go.Scatter(x=rx, y=ry, mode="lines", fill="toself",
                                   fillcolor=ACTIVE, opacity=0.5,
                                   line=dict(width=0), hoverinfo="skip"))
        else:
            data.append(empty2d())

        seen, mx, my, mc, ms = {}, [], [], [], []                # corner markers
        ox, oy = [], []                                          # emission order
        for i in range(fidx):
            st = steps[i]
            v_start = st["verts"][0]
            for v_pos in range(4):
                if v_pos < v_start:
                    continue
                vv = st["face"][v_pos]
                if vv not in seen:
                    seen[vv] = i
                gx, gy = _inset(px, py, st["face"], v_pos)
                mx.append(gx); my.append(gy)
                mc.append(_golden(seen[vv]))
                ms.append("circle" if seen[vv] == i else "square")
                ox.append(gx); oy.append(gy)
            ox.append(None); oy.append(None)     # break between faces
        data.append(go.Scatter(x=ox, y=oy, mode="lines+markers", hoverinfo="skip",
                               line=dict(color=INK, width=1.4),
                               marker=dict(size=9, color=INK, symbol="arrow",
                                           angleref="previous")))
        data.append(go.Scatter(x=mx, y=my, mode="markers", hoverinfo="skip",
                               marker=dict(size=11, color=mc, symbol=ms,
                                           line=dict(color=INK, width=1))))

        bx, by, bc, bs = [], [], [], []                          # vertex strip
        rx, ry = [], []                                          # grey deletable boxes
        seen_b = {}
        for i in range(fidx):
            st = steps[i]
            v_start = st["verts"][0]
            col, row = _slot_cell(i)
            if v_start > 0:
                x0, x1, yc = col + 0.06, col + 0.51, -row
                rx += [x0, x1, x1, x0, None]
                ry += [yc + 0.24, yc + 0.24, yc - 0.24, yc - 0.24, None]
            for v_pos in range(4):
                # Shared corners reappear grey-backed: they repeat the previous
                # face's exit edge in reversed order and are the tokens deleted.
                vv = st["face"][v_pos]
                if vv not in seen_b:
                    seen_b[vv] = i
                bx.append(col + 0.18 + 0.64 * (v_pos / 3))
                by.append(-row)
                bc.append(_golden(seen_b[vv]))
                bs.append("circle" if seen_b[vv] == i else "square")
        data.append(go.Scatter(x=rx, y=ry, mode="lines", fill="toself",
                               xaxis="x2", yaxis="y2",
                               fillcolor="rgba(128,128,128,0.32)",
                               line=dict(color="#6f6f6f", width=1.5, dash="dash"),
                               hoverinfo="skip"))
        data.append(go.Scatter(x=bx, y=by, mode="markers", hoverinfo="skip",
                               xaxis="x2", yaxis="y2",
                               marker=dict(size=9, color=bc, symbol=bs,
                                           line=dict(color=INK, width=1))))
        return data

    frames = []
    max_step = n_steps + 1
    for tlen in range(max_step):
        fidx = tlen
        a = steps[min(fidx, n_steps) - 1] if fidx > 0 else None
        if a is not None:
            acc = sum(st["n_tok"] for st in steps[:min(fidx, n_steps)])
            info = (f"face {fidx}/{total_faces}"
                    f"<br>accumulated tokens: {acc}")
        else:
            info = f"face 0/{total_faces} \u2014 press \u2192 to tokenize"
        frames.append(go.Frame(name=str(tlen), data=dyn_data(fidx),
                               layout=go.Layout(annotations=[go.layout.Annotation(
                                   xref="paper", yref="paper", x=0.0, y=1.0,
                                   showarrow=False, text=info,
                                   font=dict(size=20, color=INK), align="left",
                                   xanchor="left", yanchor="top")])))

    # No boxes and no face numbers: the grouping is carried by the spacing,
    # and dropping both frees the vertical room the mesh needs.
    shapes, bar_annos = [], []

    pad = [go.layout.Annotation(xref="paper", yref="paper", x=-10, y=-10,
                                showarrow=False, text="") for _ in bar_annos]
    for fr in frames:
        fr.layout.annotations = pad + [fr.layout.annotations[0]]

    static_json = [fig.data[0].to_plotly_json()]
    frames_json = [
        {"data": static_json + [tr.to_plotly_json() for tr in fr.data],
         "anno": fr.layout.annotations[-1].to_plotly_json(),
         "annoIdx": len(bar_annos)}
        for fr in frames]
    fig.frames = None
    fig.layout.shapes = shapes
    fig.layout.annotations = bar_annos + [go.layout.Annotation(
        xref="paper", yref="paper", x=-10, y=-10, showarrow=False, text="",
        font=dict(size=20, color=INK))]

    xs, ys = vertices[:, 0], vertices[:, 1]
    mx_, my_ = 0.03 * (xs.max() - xs.min()), 0.03 * (ys.max() - ys.min())
    fig.update_xaxes(visible=False, range=[xs.min() - mx_, xs.max() + mx_],
                     row=1, col=1)
    fig.update_yaxes(visible=False, range=[ys.min() - my_, ys.max() + my_],
                     scaleanchor="x", scaleratio=1, row=1, col=1)
    n_rows_strip = -(-n_steps // SLOTS_PER_ROW)
    fig.update_xaxes(visible=False, range=[0, SLOTS_PER_ROW], row=2, col=1)
    fig.update_yaxes(visible=False,
                     range=[-(n_rows_strip - 1) - 0.45, 0.45], row=2, col=1)
    fig.update_layout(height=672, margin=dict(l=6, r=6, t=26, b=8),
                      template="plotly_white", font=dict(color=INK),
                      showlegend=False, uirevision="keep")
    return fig, max_step - 1, frames_json


def generate(out_dir, data, idx, max_faces, strategies, dim=3, tag=''):
    os.makedirs(out_dir, exist_ok=True)
    data_n, vertices, quads, dir_class = load_sample(data, idx)
    if dim == 2:
        # Planar meshes: keep x/y only. Column 3 is an attribute, not a
        # coordinate, and the 2D tokenizer indexes coordinates positionally.
        vertices = vertices[:, :2].copy()
    print(f"sample idx={idx}/{data_n}  verts={len(vertices)} quads={quads.shape[1]}")
    for strategy in strategies:
        tok = Tokenizer2D(Q, verbose=False, sorting_strategy=strategy, dim=dim,
                          n_start_end_tokens_repeat=8)
        rows, steps = build_steps(tok, vertices, quads, dir_class, strategy,
                                  max_faces, dim=dim)
        slots = build_slots(steps, strategy)
        builder = build_figure_2d if dim == 2 else build_figure
        fig, max_idx, frames_json = builder(vertices, quads, steps, slots, strategy,
                                            quads.shape[1])
        out = os.path.join(out_dir, f"tokenization{tag}_strat{strategy}_idx{idx}.html")
        render_html(fig, out, max_idx, frames_json)
        print(f"[strat {strategy}] faces={len(steps)} rows={len(rows)} "
              f"slots={len(slots)} frames={max_idx + 1}")
        print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/quadtron_data_3d_smoke.pt")
    ap.add_argument("--idx", type=int, default=1)
    ap.add_argument("--max-faces", type=int, default=12)
    ap.add_argument("--strategies", default="1,2")
    ap.add_argument("--out-dir", default="embeds3d")
    ap.add_argument("--hexa", action="store_true",
                    help="hexa-row mode: polytron .pt, blocks in row-order, EOR per row")
    ap.add_argument("--hexa-start", default="min_theta", choices=["min_theta", "max_theta"],
                    help="hexa-row start rule: max-r in quadrant [-90,0), tie-break angle")
    ap.add_argument("--gran", default="row", choices=["row", "block", "face"],
                    help="hexa-row granularity: row-block dedup, block = 1 step/block, "
                         "face = 1 step/quad (Meshtron-style, no dedup)")
    ap.add_argument("--eoe", action="store_true",
                    help="emit EOE (sep2) element token after each block/quad")
    ap.add_argument("--dim", type=int, default=3, choices=[2, 3],
                    help="tokenizer dimension; 2 for planar domain meshes "
                         "(no dir_class needed)")
    ap.add_argument("--tag", default="_3d", help="filename infix")
    ap.add_argument("--coords", default="polar", choices=["polar", "cart"],
                    help="vertex coords: polar (4 tok/vert: r,sin,cos,z) | "
                         "cart (3 tok/vert: x,y,z)")
    ap.add_argument("--no-bar", action="store_true",
                    help="hexa-row mode: omit the token-sequence panel, 3D mesh only")
    a = ap.parse_args()
    if a.hexa:
        from tokenization_animation_3d_hexa import generate_hexa
        generate_hexa(a.out_dir, a.data, a.idx, start_rule=a.hexa_start,
                      granularity=a.gran, eoe=a.eoe, coords=a.coords,
                      show_bar=not a.no_bar)
        return
    generate(a.out_dir, a.data, a.idx, a.max_faces,
             [int(s) for s in a.strategies.split(",")], dim=a.dim, tag=a.tag)


if __name__ == "__main__":
    main()
