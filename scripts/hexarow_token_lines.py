"""hexarow_token_lines.py

Token-Anordnung des HexaRow-Encoding als eine Zeile pro Row ausgeben
(txt + html). Quads (r, sin, cos, z) gruppiert, Verts gelabelt, specials
markiert.

  uv run python scripts/hexarow_token_lines.py --idx 10 \
      --data data/polytron_data_3d_aug.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meshtron.data.hexa_row_tokenizer import HexaRowTokenizer

SP_NAMES = {}


def dataset_bounds(data):
    rpol = torch.cat([d["vertices_polar"][:, 0] for d in data])
    zpol = torch.cat([d["vertices_polar"][:, 2] for d in data])
    rp = (rpol.max() - rpol.min()) * 0.02
    zp = (zpol.max() - zpol.min()) * 0.02
    return ((float(rpol.min()) - rp, float(rpol.max()) + rp),
            (float(zpol.min()) - zp, float(zpol.max()) + zp))


def row_lines(mesh, tok, out_base):
    vp = mesh["vertices_polar"]
    blks = mesh["faces"].T.tolist()
    rows, emit = None, None
    from meshtron.data import hexa_row_tokenizer as hrt
    rows, emit = hrt.build_row_plan(blks, mesh["vertices_cartesian"],
                                    edges=mesh.get("edge_index"))
    vmap = {}  # (r,ts,tc,z) tokens -> label index

    def vid_of(vid, quad):
        key = tuple(quad)
        if key not in vmap:
            vmap[key] = vid
        return vmap[key]

    lines_txt, lines_html = [], []
    names = {(tok.core.start_token): "START", tok.core.end_token: "END",
             tok.core.sep_token: "EOR", tok.core.sep2_token: "SEP2",
             tok.core.stop_token: "STOP", tok.core.pad_token: "PAD"}
    c = tok.core

    def quads_of(vids, fname, bid):
        out = []
        for vi, vid in enumerate(vids):
            r = float(vp[vid, 0]); th = float(vp[vid, 1]); z = float(vp[vid, 2])
            ts, tc = c._q_angle(th)
            quad = [int(c._q_scalar(r, c.R_MIN, c.R_MAX) + c.off_r),
                    int(ts + c.off_ts), int(tc + c.off_tc),
                    int(c._q_scalar(z, c.Z_MIN, c.Z_MAX) + c.off_r)]
            lab = vid
            out.append((lab, quad))
        return out

    txt_parts, html_parts = [], []
    for ri, row in enumerate(rows):
        txt = [f"row {ri:3d} n blocks {len(row):2d}: "]
        htm = [f"<b>row {ri}</b> ({len(row)} blocks): "]
        consumption = "start"
        for bi, b in enumerate(row):
            vids = emit[b] if bi == 0 else emit[b][4:8]
            qs = quads_of(vids, "entry+exit" if bi == 0 else "exit", b)
            label = f"B{b}" + ("" if bi == 0 else "+")
            seg_txt = f"[{label} " + " ".join(
                f"v{lab}({q[0]},{q[1]},{q[2]},{q[3]})" for lab, q in qs) + "]"
            seg_html = f"[<span style='color:#006'>B{b}{'' if bi == 0 else '+'}</span> " + " ".join(
                f"<span style='color:#060'>v{lab}</span>​({q[0]},{q[1]},{q[2]},{q[3]})"
                for lab, q in qs) + "]"
            txt.append(seg_txt); htm.append(seg_html)
        lines_txt.append(" ".join(txt))
        lines_html.append("<div>" + " ".join(htm) + "</div>")
    lines_txt.append("STOP")
    lines_html.append("<div><b>STOP</b></div>")

    with open(out_base + ".txt", "w") as fh:
        fh.write("\n".join(lines_txt) + "\n")
    html = ("<!doctype html><meta charset='utf-8'><body style='font-family:monospace;font-size:11px'>"
            + "<pre>" + "\n".join(lines_html) + "</pre></body>")
    with open(out_base + ".html", "w") as fh:
        fh.write(html)
    return lines_txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/polytron_data_3d_aug.pt")
    ap.add_argument("--idx", type=int, default=10)
    ap.add_argument("--start", default="min_theta", choices=["min_theta", "max_theta"])
    ap.add_argument("--out-base", default="")
    a = ap.parse_args()

    data = torch.load(a.data, weights_only=False)
    rb, zb = dataset_bounds(data)
    tok = HexaRowTokenizer(r_bounds=rb, z_bounds=zb)
    base = a.out_base or (a.data.replace("/", "_").replace(".pt", "")
                          + f"_idx{a.idx}_toklines")
    lines = row_lines(data[a.idx], tok, base)
    print(f"{len(lines)} row-lines -> {base}.txt / {base}.html")
    for l in lines[:4]:
        print(l[:200] + (" ..." if len(l) > 200 else ""))


if __name__ == "__main__":
    main()
