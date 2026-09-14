"""
viz_twostage.py

Erzeugt eine Bildserie, die den Zwei-Stufen-Tokenizer (polytron_tokenizer.py)
Schritt fuer Schritt erklaert: Tokenisierung -> Sequenz -> Detokenisierung -> Round-trip.

Ausgabe: figures/twostage/01..08_*.png
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from polytron_tokenizer import PolytronTokenizer

OUT = "figures/twostage"
os.makedirs(OUT, exist_ok=True)

# Farbschema fuer Token-Typen
C_R = "#1f77b4"      # r
C_TS = "#2ca02c"     # theta_sin
C_TC = "#17becf"     # theta_cos
C_IDX = "#ff7f0e"    # vertex_idx (Pointer)
C_SPEC = "#7f7f7f"   # special tokens
C_FACE = "#cfe3f7"
C_EDGE = "#33475b"


def load_mesh(idx=0):
    data = torch.load("domain_data_10k.pt", weights_only=False)
    return data[idx]


def cart(mesh):
    return mesh["vertices_cartesian"].numpy()


def recon_cart(r, th, center):
    return np.array([center[0] + r * np.cos(th), center[1] + r * np.sin(th)])


def draw_faces(ax, xy, faces, labels=None, label_color="k", face_color=C_FACE,
               edge_color=C_EDGE, lw=1.6, alpha=0.85):
    F = faces.shape[1]
    for fi in range(F):
        ring = faces[:, fi]
        poly = xy[ring]
        ax.fill(poly[:, 0], poly[:, 1], color=face_color, alpha=alpha, zorder=1)
        cyc = np.vstack([poly, poly[0]])
        ax.plot(cyc[:, 0], cyc[:, 1], color=edge_color, lw=lw, zorder=2)
    ax.scatter(xy[:, 0], xy[:, 1], s=90, color="white", edgecolors=edge_color,
               zorder=3, linewidths=1.5)
    if labels is not None:
        for i, (x, y) in enumerate(xy):
            ax.text(x, y, str(labels[i]), ha="center", va="center",
                    fontsize=8, color=label_color, zorder=4, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")


def savefig(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------
def fig1_input(mesh):
    xy = cart(mesh)
    faces = mesh["faces"].numpy()
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_faces(ax, xy, faces, labels=list(range(xy.shape[0])))
    ax.set_title("1) Eingabe: Blockstruktur\n"
                 f"M = {xy.shape[0]} eindeutige Ecken,  F = {faces.shape[1]} Quad-Faces\n"
                 "Zahlen = globale Vertex-Indizes (schon dedupliziert)", fontsize=11)
    savefig(fig, "01_input_mesh.png")


def fig2_polar(mesh):
    xy = cart(mesh)
    center = mesh["center"].numpy()
    vp = mesh["vertices_polar"].numpy()
    faces = mesh["faces"].numpy()
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_faces(ax, xy, faces, labels=None, alpha=0.35)
    ax.plot(center[0], center[1], "P", color="crimson", ms=12, zorder=5)
    ax.text(center[0], center[1] - 0.04, "center", color="crimson",
            ha="center", va="top", fontsize=9)
    for i, (x, y) in enumerate(xy):
        ax.plot([center[0], x], [center[1], y], color=C_R, lw=0.7, alpha=0.6, zorder=1)
        r, th = vp[i]
        ax.text(x, y, f"r={r:.2f}\nθ={np.degrees(th):.0f}°", ha="center", va="center",
                fontsize=6.5, zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=C_R, lw=0.6))
    ax.set_title("2) Vertices → Polarkoordinaten (r, θ) relativ zum Zentrum\n"
                 "θ wird als (sin, cos)-Paar kodiert (kein Wrap-around bei 0/2π)", fontsize=11)
    savefig(fig, "02_polar.png")


def fig3_sort(mesh, tok):
    xy = cart(mesh)
    vp = mesh["vertices_polar"]
    faces = mesh["faces"].numpy()
    order = tok._sort_order(vp)             # new_pos -> old_idx
    old2new = np.empty(len(order), dtype=int)
    old2new[order] = np.arange(len(order))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # links: alte Indizes
    draw_faces(axes[0], xy, faces, labels=list(range(xy.shape[0])), alpha=0.5)
    axes[0].set_title("vorher: globale Original-Indizes", fontsize=10)
    # rechts: neue sortierte Indizes (lexsort nach (θ, r))
    draw_faces(axes[1], xy, faces, labels=list(old2new), alpha=0.5,
               face_color="#ffe9d6", label_color="#a3560d")
    axes[1].set_title("nachher: neue Indizes = Reihenfolge im Vertex-Stream", fontsize=10)
    fig.suptitle("3) Stufe-1-Sortierung: Vertices lexikografisch nach (θ, r)\n"
                 "→ deterministische Reihenfolge; old→new-Mapping remappt später die Faces",
                 fontsize=11)
    savefig(fig, "03_sort.png")


def fig4_quantize(mesh, tok):
    vp = mesh["vertices_polar"]
    order = tok._sort_order(vp)
    r_min, r_max = tok.R_MIN, tok.R_MAX            # FIXE Bounds (nicht per-Mesh)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis("off")
    ax.set_title("4) Stufe-1-Quantisierung: jeder Vertex → 3 Tokens  [r, θsin, θcos]\n"
                 f"Qr={tok.Qr} (r-Bins),  Qa={tok.Qa} (Winkel-Bins),  "
                 f"r-Bounds FIX [{r_min:.2f},{r_max:.2f}]", fontsize=11)

    header = ["new-Idx", "r", "θ (°)", "→ r_tok", "→ θsin_tok", "→ θcos_tok"]
    rows = []
    for new_i in range(min(6, len(order))):
        old_i = order[new_i]
        r = float(vp[old_i, 0]); th = float(vp[old_i, 1])
        r_tok = tok._q_scalar(r, r_min, r_max) + tok.off_r
        ts, tc = tok._q_angle(th)
        rows.append([new_i, f"{r:.3f}", f"{np.degrees(th):.1f}",
                     r_tok, ts + tok.off_ts, tc + tok.off_tc])

    tbl = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.6)
    # Spaltenfarben
    for (r_, c_), cell in tbl.get_celld().items():
        if r_ == 0:
            cell.set_facecolor("#eeeeee"); cell.set_text_props(fontweight="bold")
        if c_ == 3 and r_ > 0: cell.set_facecolor("#dce9f7")
        if c_ == 4 and r_ > 0: cell.set_facecolor("#dcf0dc")
        if c_ == 5 and r_ > 0: cell.set_facecolor("#d6f2f5")
    ax.text(0.5, -0.02, "(nur die ersten 6 von M Vertices gezeigt)",
            transform=ax.transAxes, ha="center", fontsize=8, color="gray")
    savefig(fig, "04_stage1_quantize.png")


def fig5_pointers(mesh, tok):
    xy = cart(mesh)
    vp = mesh["vertices_polar"]
    faces = mesh["faces"].numpy()
    order = tok._sort_order(vp)
    old2new = np.empty(len(order), dtype=int); old2new[order] = np.arange(len(order))
    faces_new = old2new[faces]              # [4,F]

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    draw_faces(ax, xy, faces, labels=list(old2new), alpha=0.4,
               face_color="#ffe9d6", label_color="#a3560d")

    # ein Face hervorheben + Pointer-Pfeile
    hi = 0
    ring_old = faces[:, hi]
    poly = xy[ring_old]
    ax.fill(poly[:, 0], poly[:, 1], color="#ff7f0e", alpha=0.35, zorder=2)
    cx, cy = poly[:, 0].mean(), poly[:, 1].mean()
    for k in range(4):
        p = xy[ring_old[k]]
        arr = FancyArrowPatch((cx, cy), (p[0], p[1]), arrowstyle="-|>",
                              mutation_scale=13, color=C_IDX, lw=1.6, zorder=5)
        ax.add_patch(arr)
    ax.text(cx, cy, f"Face 0\n→ {[int(x) for x in faces_new[:, hi]]}", ha="center", va="center",
            fontsize=8, zorder=6, color="#7a3c00", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_IDX))

    # Liste aller Faces
    lines = "\n".join(f"Face {fi}: {[int(x) for x in faces_new[:, fi]]}"
                      for fi in range(faces.shape[1]))
    ax.text(1.02, 0.5, "Stufe 2 — Pointer-Faces\n(Indizes in den Vertex-Stream):\n\n" + lines,
            transform=ax.transAxes, va="center", ha="left", fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff6ec", ec=C_IDX))
    ax.set_title("5) Stufe 2: jedes Quad = 4 ZEIGER auf Stufe-1-Vertices\n"
                 "→ kann keine ungültigen Indizes erzeugen = Gültigkeit per Konstruktion",
                 fontsize=11)
    savefig(fig, "05_stage2_pointers.png")


def _tok_color(tok_obj, t):
    if t >= tok_obj.start_token:
        return C_SPEC
    if t < tok_obj.off_ts:
        return C_R
    if t < tok_obj.off_tc:
        return C_TS
    if t < tok_obj.off_idx:
        return C_TC
    return C_IDX


def fig6_sequence(mesh, tok):
    toks, meta = tok.tokenize(mesh)
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    n = len(toks)
    per_row = 40
    rows = int(np.ceil(n / per_row))
    dx = 1.0 / per_row
    dy = 0.7 / rows
    for i, t in enumerate(toks):
        r_ = i // per_row; c_ = i % per_row
        x = c_ * dx; y = 0.78 - r_ * dy
        col = _tok_color(tok, t)
        ax.add_patch(Rectangle((x, y), dx * 0.92, dy * 0.8, color=col,
                               alpha=0.85, ec="white", lw=0.4))
        if t == tok.start_token: lbl = "S"
        elif t == tok.end_token: lbl = "E"
        elif t == tok.sep_token: lbl = "|"
        elif t == tok.sep2_token: lbl = "‖"
        else: lbl = str(t)
        ax.text(x + dx * 0.46, y + dy * 0.4, lbl, ha="center", va="center",
                fontsize=5.5, color="white", fontweight="bold")

    legend = [("r", C_R), ("θsin", C_TS), ("θcos", C_TC),
              ("vertex_idx (Pointer)", C_IDX), ("special (S/|/E)", C_SPEC)]
    for j, (name, col) in enumerate(legend):
        x = 0.02 + j * 0.20
        ax.add_patch(Rectangle((x, 0.02), 0.02, 0.04, color=col))
        ax.text(x + 0.025, 0.04, name, va="center", fontsize=8)

    ax.set_title(f"6) Fertige Token-Sequenz  (Länge {n})\n"
                 "[S] (r,θsin,θcos)×M  |  (i0..i3)×F  ‖  (αs,tn,αe,tn)×4F geom  [E]",
                 fontsize=11)
    savefig(fig, "06_sequence.png")


def fig7_detokenize(mesh, tok):
    toks, meta = tok.tokenize(mesh)
    verts, faces_new, _ = tok.detokenize(toks, meta["r_min"], meta["r_max"])
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axis("off")
    ax.set_title("7) Detokenisierung: Sequenz zurück in Geometrie + Topologie", fontsize=11)

    steps = [
        ("① Sequenz an [S] | [E] trennen",
         "Vertex-Sektion  ←→  Face-Sektion (Trenner = |)"),
        ("② Stufe 1: je 3 Tokens → ein Vertex",
         "r_tok → r (dequant.),  (θsin,θcos) → θ = atan2(sin,cos)"),
        ("③ Stufe 2: je 4 Tokens → ein Face",
         "Pointer-Indizes zeigen direkt in die Vertexliste"),
        ("④ Ergebnis",
         f"{len(verts)} Vertices,  {len(faces_new)} Faces  →  reconstruct_domain_coons"),
    ]
    y = 0.85
    for title, body in steps:
        ax.text(0.03, y, title, fontsize=10.5, fontweight="bold", color=C_EDGE)
        ax.text(0.06, y - 0.07, body, fontsize=9.5, family="monospace")
        y -= 0.21
    ax.text(0.5, -0.02, f"erste rekonstruierte Faces: {faces_new[:3]}",
            transform=ax.transAxes, ha="center", fontsize=8, color="gray")
    savefig(fig, "07_detokenize.png")


def fig8_roundtrip(mesh, tok):
    xy = cart(mesh)
    vp = mesh["vertices_polar"]
    center = mesh["center"].numpy()
    faces = mesh["faces"].numpy()
    order = tok._sort_order(vp)

    toks, meta = tok.tokenize(mesh)
    verts, faces_new, _ = tok.detokenize(toks, meta["r_min"], meta["r_max"])

    # rekonstruierte kartesische Positionen (in new-Reihenfolge)
    xy_rec_new = np.array([recon_cart(r, th, center) for (r, th) in verts])
    # zurueck in old-Reihenfolge fuer den Vergleich
    xy_rec = np.empty_like(xy_rec_new)
    xy_rec[order] = xy_rec_new
    faces_rec_old = order[np.array(faces_new)].T   # new-idx -> old-idx, [4,F]

    max_err = float(np.max(np.linalg.norm(xy_rec - xy, axis=1)))
    topo_ok = np.array_equal(np.sort(faces_rec_old, axis=0), np.sort(faces, axis=0))

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    # Original solide
    for fi in range(faces.shape[1]):
        p = np.vstack([xy[faces[:, fi]], xy[faces[0, fi]]])
        ax.plot(p[:, 0], p[:, 1], color=C_EDGE, lw=2.2, zorder=2,
                label="Original" if fi == 0 else None)
    # Rekonstruiert gestrichelt
    for fi in range(faces_rec_old.shape[1]):
        p = np.vstack([xy_rec[faces_rec_old[:, fi]], xy_rec[faces_rec_old[0, fi]]])
        ax.plot(p[:, 0], p[:, 1], color=C_IDX, lw=1.4, ls="--", zorder=3,
                label="Rekonstruiert" if fi == 0 else None)
    ax.scatter(xy[:, 0], xy[:, 1], s=40, color=C_EDGE, zorder=4)
    ax.set_aspect("equal"); ax.axis("off"); ax.legend(loc="upper right", fontsize=9)
    ax.set_title("8) Round-trip: Original vs. Rekonstruiert\n"
                 f"Topologie exakt: {'JA ✓' if topo_ok else 'NEIN ✗'}   |   "
                 f"max. Positionsfehler: {max_err:.2e}\n"
                 "(Fehler = reine Quantisierung, mit höherem Qr/Qa beliebig kleiner)", fontsize=11)
    savefig(fig, "08_roundtrip.png")


C_PARALLEL = "#d62728"


def _is_twin(a, b, atol=1e-6):
    b = np.asarray(b, float)[::-1]
    a = np.asarray(a, float)
    return a.shape == b.shape and np.allclose(a, b, atol=atol)


def fig9_curved(mesh, tok):
    """Stufe 3: echte gekruemmte Half-Edge-Kurven aus der Sequenz rekonstruiert,
    Parallelkanten (Blade Druck/Saug) rot. Vergleich gegen edge_to_streamline."""
    center = mesh["center"].numpy()
    e2s = mesh["edge_to_streamline"]
    order = tok._sort_order(mesh["vertices_polar"])

    toks, meta = tok.tokenize(mesh)
    verts, faces_new, geom = tok.detokenize(
        toks, meta["r_min"], meta["r_max"], meta["tn_min"], meta["tn_max"])
    edges = tok.reconstruct_edges(verts, faces_new, geom, center, n=60)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))

    # --- links: 1. Ordnung (gerade Quads) = was Frame 1-8 zeigt ---
    xy_new = np.array([[center[0] + r * np.cos(th), center[1] + r * np.sin(th)]
                       for (r, th) in verts])
    for quad in faces_new:
        ring = xy_new[np.array(quad + [quad[0]])]
        axes[0].plot(ring[:, 0], ring[:, 1], color=C_EDGE, lw=1.6)
        axes[0].fill(xy_new[quad][:, 0], xy_new[quad][:, 1], color=C_FACE, alpha=0.6)
    axes[0].scatter(xy_new[:, 0], xy_new[:, 1], s=30, color=C_EDGE, zorder=5)
    axes[0].set_aspect("equal"); axes[0].axis("off")
    axes[0].set_title("Stufe 1+2: gerade Quads (Kantenverlauf fehlt)", fontsize=10)

    # --- rechts: Stufe 3 gekruemmte Half-Edges, Parallelkanten rot ---
    # GT-streamline zum Vergleich (duenn grau)
    for (ug, vg), pts in e2s.items():
        p = np.asarray(pts, float)
        axes[1].plot(p[:, 0], p[:, 1], color="#bbbbbb", lw=3.2, zorder=1, alpha=0.7)

    n_par = 0
    for e in edges:
        ug = int(order[e["u"]]); vg = int(order[e["v"]])
        color, lw, z = C_R, 1.6, 3
        # echte Parallelkante? twin existiert, ist aber KEINE Umkehrung
        if (vg, ug) in e2s and (ug, vg) in e2s:
            if not _is_twin(e2s[(ug, vg)], e2s[(vg, ug)]):
                color, lw, z = C_PARALLEL, 2.4, 5
                n_par += 1
        c = e["curve"]
        axes[1].plot(c[:, 0], c[:, 1], color=color, lw=lw, zorder=z)
    axes[1].scatter(xy_new[:, 0], xy_new[:, 1], s=30, color=C_EDGE, zorder=6)
    axes[1].set_aspect("equal"); axes[1].axis("off")
    axes[1].set_title("Stufe 3: rekonstruierte gekrümmte Half-Edges\n"
                      f"grau = GT-Streamline,  blau = Hermite,  "
                      f"ROT = {n_par // 2} Parallelkante(n) (Blade Druck/Saug)", fontsize=10)

    fig.suptitle("9) Warum Stufe 3: der Kantenverlauf.  Pro gerichteter Half-Edge "
                 "eine Hermite-Kurve (αstart, tn, αend, tn).\n"
                 "Multigraph-treu: keine {u,v}-Deduplikation → Druck-/Saugseite "
                 "am Blade bleiben getrennte Kurven.", fontsize=11)
    savefig(fig, "09_curved_faces.png")


def pick_curved_edge(mesh, target_sag=0.10):
    """Waehlt eine MODERAT gekruemmte gerichtete Face-Kante (Sagitta/Chord nahe
    target_sag), unter Ausschluss winziger/entarteter Kanten (chord >= Median)."""
    xy = cart(mesh)
    faces_g = mesh["faces"].numpy()
    e2s = mesh["edge_to_streamline"]
    cands = []
    for fi in range(faces_g.shape[1]):
        for k in range(4):
            p0 = int(faces_g[k, fi]); p1 = int(faces_g[(k + 1) % 4, fi])
            if (p0, p1) not in e2s:
                continue
            c = np.asarray(e2s[(p0, p1)], float)
            P0, P1 = xy[p0], xy[p1]
            chord = float(np.linalg.norm(P1 - P0)) + 1e-9
            sag = float(np.max(np.abs(np.cross(P1 - P0, c - P0)) / chord)) / chord
            cands.append((p0, p1, chord, sag))
    med_chord = np.median([c[2] for c in cands])
    ok = [c for c in cands if c[2] >= med_chord] or cands
    p0, p1, _, _ = min(ok, key=lambda c: abs(c[3] - target_sag))
    return p0, p1


def fig10_edge_tokenize(mesh, tok):
    """Zoom auf EINE moderat gekruemmte Half-Edge: wie (α_start, tn_start,
    α_end, tn_end) zu 6 Tokens werden. Der Schritt, den Blockbilder verstecken."""
    from matplotlib.patches import FancyArrowPatch, Rectangle
    xy = cart(mesh)
    ei = mesh["edge_index"].numpy()
    et = mesh["edge_tangents"].numpy()
    e2s = mesh["edge_to_streamline"]
    tan = {(int(ei[0, e]), int(ei[1, e])): et[e] for e in range(ei.shape[1])}
    tn_min, tn_max = tok.TN_MIN, tok.TN_MAX        # FIXE Bounds (nicht per-Mesh)

    p0, p1 = pick_curved_edge(mesh)
    a_s, tn_s, a_e, tn_e = tan[(p0, p1)]
    P0, P1 = xy[p0], xy[p1]
    curve = np.asarray(e2s[(p0, p1)], float)

    # Quantisierung (echte Tokenizer-Methoden)
    ts0, tc0 = tok._q_angle(a_s); ts1, tc1 = tok._q_angle(a_e)
    qtn0 = tok._q_scalar(tn_s, tn_min, tn_max); qtn1 = tok._q_scalar(tn_e, tn_min, tn_max)
    tok_seq = [ts0 + tok.off_ts, tc0 + tok.off_tc, qtn0 + tok.off_r,
               ts1 + tok.off_ts, tc1 + tok.off_tc, qtn1 + tok.off_r]

    fig, (axg, axt) = plt.subplots(1, 2, figsize=(13.5, 6.6),
                                   gridspec_kw={"width_ratios": [1, 1.15]})

    # --- links: Geometrie mit Tangenten ---
    axg.plot([P0[0], P1[0]], [P0[1], P1[1]], ls="--", color="#999999", lw=1.4,
             zorder=1, label="Sehne (Chord)")
    axg.plot(curve[:, 0], curve[:, 1], color=C_R, lw=2.6, zorder=3, label="echte Kanten-Kurve")
    scale = 0.9   # Tangenten-Pfeile visuell skaliert
    T0 = tn_s * np.array([np.cos(a_s), np.sin(a_s)])
    T1 = tn_e * np.array([np.cos(a_e), np.sin(a_e)])
    for P, T, lab, col in [(P0, T0, "start", C_TS), (P1, -T1, "end", C_TC)]:
        tip = P + scale * T
        axg.add_patch(FancyArrowPatch((P[0], P[1]), (tip[0], tip[1]), arrowstyle="-|>",
                                      mutation_scale=16, color=col, lw=2.2, zorder=5))
    axg.scatter([P0[0], P1[0]], [P0[1], P1[1]], s=70, color=C_EDGE, zorder=6)
    axg.annotate("P0 (start)", P0, textcoords="offset points", xytext=(-8, -14), fontsize=9)
    axg.annotate("P1 (end)", P1, textcoords="offset points", xytext=(-6, 8), fontsize=9)
    axg.text(P0[0] + scale * T0[0], P0[1] + scale * T0[1],
             f"  T0\n  α_start={np.degrees(a_s):.0f}°\n  tn={tn_s:.3f}",
             fontsize=8, color=C_TS, va="center")
    axg.text(P1[0] - scale * T1[0], P1[1] - scale * T1[1],
             f"  T1\n  α_end={np.degrees(a_e):.0f}°\n  tn={tn_e:.3f}",
             fontsize=8, color=C_TC, va="center")
    axg.set_aspect("equal"); axg.axis("off"); axg.legend(loc="best", fontsize=8)
    axg.set_title(f"eine gerichtete Half-Edge  p{p0}→p{p1}\n"
                  "Hermite: T0=tn·[cosα,sinα] am Start,  T1 am Ende", fontsize=10)

    # --- rechts: Quantisierungs-Tabelle + Token-Chips ---
    axt.axis("off")
    axt.set_title("→ 6 Tokens (genau das, was das Modell lernt)", fontsize=10)
    header = ["Groesse", "Wert", "→ (sin,cos) bzw. norm.", "→ Token"]
    rows = [
        ["α_start", f"{np.degrees(a_s):.1f}°",
         f"sin={np.sin(a_s):+.2f} cos={np.cos(a_s):+.2f}", f"{ts0+tok.off_ts}, {tc0+tok.off_tc}"],
        ["tn_start", f"{tn_s:.3f}",
         f"({tn_s:.3f}-{tn_min:.2f})/({tn_max:.2f}-{tn_min:.2f})", f"{qtn0+tok.off_r}"],
        ["α_end", f"{np.degrees(a_e):.1f}°",
         f"sin={np.sin(a_e):+.2f} cos={np.cos(a_e):+.2f}", f"{ts1+tok.off_ts}, {tc1+tok.off_tc}"],
        ["tn_end", f"{tn_e:.3f}",
         f"({tn_e:.3f}-{tn_min:.2f})/({tn_max:.2f}-{tn_min:.2f})", f"{qtn1+tok.off_r}"],
    ]
    tbl = axt.table(cellText=rows, colLabels=header, loc="upper center", cellLoc="center",
                    colWidths=[0.16, 0.14, 0.42, 0.20])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 1.7)
    rowcol = [C_TS, C_R, C_TC, C_R]
    for (rr, cc), cell in tbl.get_celld().items():
        if rr == 0:
            cell.set_facecolor("#eeeeee"); cell.set_text_props(fontweight="bold")
        elif cc == 3:
            cell.set_facecolor(rowcol[rr - 1]); cell.set_alpha(0.35)

    # Token-Chips (die fertige 6er-Gruppe fuer diese Kante)
    chip_cols = [C_TS, C_TC, C_R, C_TS, C_TC, C_R]
    chip_lab = ["αs.sin", "αs.cos", "tn_s", "αe.sin", "αe.cos", "tn_e"]
    y0 = 0.16
    axt.text(0.5, y0 + 0.14, "Kanten-Token-Gruppe (6 Tokens):",
             transform=axt.transAxes, ha="center", fontsize=9.5, fontweight="bold")
    for j, (t, col, lab) in enumerate(zip(tok_seq, chip_cols, chip_lab)):
        x = 0.09 + j * 0.14
        axt.add_patch(Rectangle((x, y0), 0.12, 0.09, transform=axt.transAxes,
                                color=col, alpha=0.85, ec="white"))
        axt.text(x + 0.06, y0 + 0.045, str(t), transform=axt.transAxes,
                 ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        axt.text(x + 0.06, y0 - 0.03, lab, transform=axt.transAxes,
                 ha="center", va="center", fontsize=7, color="#444")
    axt.text(0.5, y0 - 0.09,
             "grün/cyan = Winkel (sin,cos-Paar),   blau = Tangenten-Betrag tn (skalar)",
             transform=axt.transAxes, ha="center", fontsize=8, color="gray")

    fig.suptitle("10) Wie eine KANTEN-KURVE tokenisiert wird (Stufe 3, pro Half-Edge)\n"
                 "Blockbilder verstecken das — hier explizit: Kurve → (α,tn) → 6 Tokens", fontsize=11)
    savefig(fig, "10_edge_tokenize.png")


def fig11_compare_methods(mesh, tok):
    """Direktvergleich Hermite (6 Tok, kubisch) vs Bezier k=2 (4 Tok, quadratisch)
    auf DERSELBEN Kante, gegen die GT-Streamline."""
    from matplotlib.patches import Rectangle
    xy = cart(mesh)
    ei = mesh["edge_index"].numpy(); et = mesh["edge_tangents"].numpy()
    e2s = mesh["edge_to_streamline"]
    tan = {(int(ei[0, e]), int(ei[1, e])): et[e] for e in range(ei.shape[1])}
    tn_min, tn_max = tok.TN_MIN, tok.TN_MAX        # FIXE Bounds (nicht per-Mesh)

    p0, p1 = pick_curved_edge(mesh)
    a_s, tn_s, a_e, tn_e = tan[(p0, p1)]
    P0, P1 = xy[p0], xy[p1]
    gt = np.asarray(e2s[(p0, p1)], float)
    chord = np.linalg.norm(P1 - P0) + 1e-9

    T0 = tn_s * np.array([np.cos(a_s), np.sin(a_s)])
    T1 = tn_e * np.array([np.cos(a_e), np.sin(a_e)])
    c_herm = tok._hermite(P0, P1, T0, T1, len(gt))
    c_bez = tok._quad_bezier(P0, P1, [np.cos(a_s), np.sin(a_s)],
                             [np.cos(a_e), np.sin(a_e)], len(gt))
    # cubic_bezier: best-fit 2 Kontrollpunkte an die Streamline
    B1, B2 = tok._fit_cubic_bezier(P0, P1, gt)
    c_cub = tok._cubic_bezier_curve(P0, P1, B1, B2, len(gt))

    def relerr(c):
        return float(np.max(np.linalg.norm(c - gt, axis=1)) / chord)
    e_h, e_b, e_c = relerr(c_herm), relerr(c_bez), relerr(c_cub)

    # Token-Zahlen fuer diese eine Kante
    ht = PolytronTokenizer(quantization_r=tok.Qr, quantization_a=tok.Qa,
                           max_vertices=8, repr_mode='hermite')
    ct = PolytronTokenizer(quantization_r=tok.Qr, quantization_a=tok.Qa,
                           max_vertices=8, repr_mode='cubic_bezier')
    ts0, tc0 = ht._q_angle(a_s); ts1, tc1 = ht._q_angle(a_e)
    q0 = ht._q_scalar(tn_s, tn_min, tn_max); q1 = ht._q_scalar(tn_e, tn_min, tn_max)
    toks_h = [ts0 + ht.off_ts, tc0 + ht.off_tc, q0 + ht.off_r,
              ts1 + ht.off_ts, tc1 + ht.off_tc, q1 + ht.off_r]
    toks_b = [ts0 + ht.off_ts, tc0 + ht.off_tc, ts1 + ht.off_ts, tc1 + ht.off_tc]
    uh, nh, L = ct._chord_frame(P0, P1)
    s1 = np.dot(B1 - P0, uh) / L; h1 = np.dot(B1 - P0, nh) / L
    s2 = np.dot(B2 - P0, uh) / L; h2 = np.dot(B2 - P0, nh) / L
    toks_c = [ct._q_scalar(s1, ct.S_MIN, ct.S_MAX) + ct.off_r,
              ct._q_scalar(h1, ct.H_MIN, ct.H_MAX) + ct.off_r,
              ct._q_scalar(s2, ct.S_MIN, ct.S_MAX) + ct.off_r,
              ct._q_scalar(h2, ct.H_MIN, ct.H_MAX) + ct.off_r]

    fig, (axg, axi) = plt.subplots(1, 2, figsize=(13.5, 6.6),
                                   gridspec_kw={"width_ratios": [1.1, 1]})
    axg.plot([P0[0], P1[0]], [P0[1], P1[1]], ls="--", color="#bbbbbb", lw=1.2, label="Sehne")
    axg.plot(gt[:, 0], gt[:, 1], color="#444444", lw=4.5, alpha=0.30, label="GT-Streamline")
    axg.plot(c_herm[:, 0], c_herm[:, 1], color=C_R, lw=1.8,
             label=f"Hermite (6 Tok)  {e_h*100:.1f}%")
    axg.plot(c_bez[:, 0], c_bez[:, 1], color="#e6550d", lw=1.8, ls=(0, (4, 2)),
             label=f"Bézier k=2 (4 Tok)  {e_b*100:.1f}%")
    axg.plot(c_cub[:, 0], c_cub[:, 1], color="#2ca02c", lw=1.8, ls=(0, (1, 1)),
             label=f"cubic_bezier best-fit (4 Tok)  {e_c*100:.1f}%")
    axg.scatter([P0[0], P1[0]], [P0[1], P1[1]], s=55, color=C_EDGE, zorder=6)
    sag = np.max(np.abs(np.cross(P1 - P0, gt - P0)) / chord) / chord
    axg.set_aspect("equal"); axg.axis("off"); axg.legend(loc="best", fontsize=8.5)
    axg.set_title(f"selbe Kante p{p0}→p{p1}  (Sagitta/Chord ≈ {sag:.2f})", fontsize=10)

    axi.axis("off")
    lines = [
        ("Hermite  (kubisch, aus Extractor-Tangenten)", C_R),
        (f"  6 Tok → {toks_h}", None),
        (f"  α+tn.  Fehler {e_h*100:.1f}%  (Satz median 1.3%)", None),
        ("", None),
        ("Bézier k=2  (quadratisch)", "#e6550d"),
        (f"  4 Tok → {toks_b}", None),
        ("  nur α, Kontrollpkt = Tangentenschnitt (0 Freiheit)", None),
        (f"  Fehler {e_b*100:.1f}%  (Satz median 6.0%)", None),
        ("", None),
        ("cubic_bezier  (kubisch, best-fit)", "#2ca02c"),
        (f"  4 Tok → {toks_c}", None),
        ("  2 freie Kontrollpkt (s1,h1,s2,h2) LSQ an Streamline", None),
        (f"  Fehler {e_c*100:.1f}%  (Satz median 0.7%)", None),
        ("", None),
        ("→ cubic_bezier gewinnt BEIDE Achsen: 4 Tok UND genauer.", "#111"),
        ("Wendepunkt-Limit theoretisch — Daten haben 0 S-Kanten,", "#666"),
        ("also nicht der Grund; Grund = freie best-fit Kontrollpkt.", "#666"),
    ]
    y = 0.96
    for txt, col in lines:
        axi.text(0.02, y, txt, transform=axi.transAxes, fontsize=9.3,
                 family="monospace" if txt.startswith("  ") else "sans-serif",
                 fontweight="bold" if col in (C_R, "#e6550d", "#2ca02c") else "normal",
                 color=col if col else "#111")
        y -= 0.056
    fig.suptitle("11) Methodenvergleich: Hermite vs Bézier k=2 vs cubic_bezier (gleiche Kante, gegen GT)",
                 fontsize=11)
    savefig(fig, "11_compare_methods.png")


if __name__ == "__main__":
    mesh = load_mesh(0)
    tok = PolytronTokenizer(max_vertices=64)
    fig1_input(mesh)
    fig2_polar(mesh)
    fig3_sort(mesh, tok)
    fig4_quantize(mesh, tok)
    fig5_pointers(mesh, tok)
    fig6_sequence(mesh, tok)
    fig7_detokenize(mesh, tok)
    fig8_roundtrip(mesh, tok)
    fig9_curved(mesh, tok)
    # Frames 10/11: moderat gekruemmte Innenkante braucht ein curvigeres Mesh
    # (das 6-Face-Grid hat nur gerade Kanten + die extreme Blade-Linse).
    mesh_edge = load_mesh(340)      # 20-Face, Kante mit Sagitta/Chord ~0.12
    fig10_edge_tokenize(mesh_edge, tok)
    fig11_compare_methods(mesh_edge, tok)
    print("\nFertig. 11 Frames in", OUT)
