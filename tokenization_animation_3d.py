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
import torch
from plotly.subplots import make_subplots

from tokenizer_v2 import Tokenizer2D

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
    dir_class = s.dir_class.detach().cpu().numpy().astype(int)
    return len(data), vertices, quads, dir_class


def build_steps(tok: Tokenizer2D, vertices, quads, dir_class, strategy: int, max_faces: int):
    """Tokenization steps for the first ``max_faces`` faces (row-boundary clipped).

    Returns (rows, steps). Each step: face corner vertex ids, dir_class label,
    row index, tokens emitted at this step (strat2: 12 row-start / 6 mid-row).
    """
    vt = torch.from_numpy(vertices)
    qt = torch.from_numpy(quads)
    dc_t = torch.from_numpy(dir_class)
    sorted_quads, rows = tok._order_quads(vt, qt, dc_t)
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
            v_start = 0 if fi == s else 2
            verts = list(range(v_start, 4))
            toks = []
            for v in verts:
                toks += [int(c) for c in quant[fi * 4 + v].tolist()]
            n_tok = (12 if fi == s else 6) if strategy == 2 else 12
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
        bx, by, bc = [], [], []                        # token bar dots
        for i in range(fidx):
            st = steps[i]
            for kk, tv in enumerate(st["toks"]):
                offs = kk / max(st["n_tok"] - 1, 1)
                bx.append(slot_of[i] + 0.2 + 0.6 * offs)
                by.append(tv / Q)
                bc.append(step_color(i, n_steps))
        data.append(go.Scatter(x=bx, y=by, mode="markers", marker=dict(size=6, color=bc),
                               hoverinfo="skip"))
        return data

    frames = []
    max_step = n_steps + 1            # last step = overview frame
    for tlen in range(max_step):
        fidx = tlen
        a = steps[min(fidx, n_steps) - 1] if fidx > 0 else None
        if a is not None:
            info = (f"face {fidx}/{total_faces}  ·  row {a['row'] + 1}"
                    f" (dir_class {a['label']})  ·  tokens this step: {a['n_tok']}"
                    + ("  ·  end of row -> EOR" if (strategy == 2 and a["row_end"]) else ""))
        else:
            info = f"face 0/{total_faces} — press → to tokenize"
        layout = go.Layout(annotations=[go.layout.Annotation(
            xref="paper", yref="paper", x=0.0, y=1.0, showarrow=False, text=info,
            font=dict(size=13, color=INK), xanchor="left", yanchor="top")])
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
    # frames: pad annotations so the info annotation index stays stable
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


def render_html(fig, out_path: str, max_idx: int, frames_json):
    div_id = "tokviz3d"
    plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn",
                            default_width="100%", default_height="740px")
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
  .wrap { max-width: 1280px; margin: 0 auto; padding: 8px 12px 18px; }
  #controls { display:flex; align-items:center; gap:14px; padding:10px 4px 6px; }
  button.step { width:56px; height:44px; font-size:26px; border:1px solid #d8d2cc;
    border-radius:10px; background:#fff; color:#201e1d; cursor:pointer; }
  button.step:disabled { opacity:.25; cursor:default; }
  .progress { flex:1; height:10px; border-radius:6px;
    background:linear-gradient(90deg,#0088b0,#d6006c); position:relative; overflow:hidden; }
  .progress .mask { position:absolute; right:0; top:0; height:100%; width:0%;
    background:#fbfaf8; transition: width .18s ease; }
  .legend { display:flex; align-items:center; gap:8px; font-size:12px; color:#201e1d; }
  .bar { width:12px; height:64px; border-radius:4px;
    background:linear-gradient(180deg,#3b4cc0,#f2f2f2 50%,#b40426); }
</style></head>
<body><div class="wrap">
<div id="controls">
  <button id="prevBtn" class="step">&larr;</button>
  <button id="nextBtn" class="step">&rarr;</button>
  <div class="progress"><div id="progressMask" class="mask"></div></div>
  <div class="legend"><span>order:</span><div class="bar"></div><span>first&nbsp;&hellip;&nbsp;last</span></div>
</div>
PLOTLY_HTML_HERE
<script>var TOKFRAMES = FRAMES_JSON;</script>
</div>
<script>
var gd;
var qs = new URLSearchParams(window.location.search);
var idx = Math.min(parseInt(qs.get('step') || '0', 10) || 0, MAXI), maxIdx = MAXI;
function render() {
  if (!gd) gd = document.getElementById('TOKDIV');
  var f = TOKFRAMES[idx];
  var L = JSON.parse(JSON.stringify(gd.layout));
  L.annotations[f.annoIdx] = f.anno;
  Plotly.react(gd, f.data, L, {displayModeBar: false, responsive: false});
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
    html = html.replace("PLOTLY_HTML_HERE", plot_html).replace(
        "TOKDIV", div_id).replace("MAXI", str(max_idx)).replace(
        "FRAMES_JSON", frames_json_src)
    with open(out_path, "w") as f:
        f.write(html)


def generate(out_dir, data, idx, max_faces, strategies):
    os.makedirs(out_dir, exist_ok=True)
    data_n, vertices, quads, dir_class = load_sample(data, idx)
    print(f"sample idx={idx}/{data_n}  verts={len(vertices)} quads={quads.shape[1]}")
    for strategy in strategies:
        tok = Tokenizer2D(Q, verbose=False, sorting_strategy=strategy, dim=3,
                          n_start_end_tokens_repeat=8)
        rows, steps = build_steps(tok, vertices, quads, dir_class, strategy, max_faces)
        slots = build_slots(steps, strategy)
        fig, max_idx, frames_json = build_figure(vertices, quads, steps, slots, strategy,
                                                 quads.shape[1])
        out = os.path.join(out_dir, f"tokenization_3d_strat{strategy}_idx{idx}.html")
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
    a = ap.parse_args()
    if a.hexa:
        from tokenization_animation_3d_hexa import generate_hexa
        generate_hexa(a.out_dir, a.data, a.idx, start_rule=a.hexa_start)
        return
    generate(a.out_dir, a.data, a.idx, a.max_faces,
             [int(s) for s in a.strategies.split(",")])


if __name__ == "__main__":
    main()
