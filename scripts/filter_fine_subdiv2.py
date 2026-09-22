"""Filter data/fine polytron samples to subdiv_n==2 and render 10 random previews.

Candidates with subdiv_n 3/4 are dropped (too many tokens for retraining).
Output: data/fine/polytron_subdiv2.pt (filtered list) and one PNG per
randomly drawn candidate under data/fine/preview_subdiv2/.
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "fine" / "polytron_data_3d_fine.pt"


def draw_block_edges(vc: np.ndarray, faces: np.ndarray, ax, color: str) -> None:
    hexes = np.asarray(faces).reshape(-1, 8)
    for h in hexes:
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0),
                     (4, 5), (5, 6), (6, 7), (7, 4),
                     (0, 4), (1, 5), (2, 6), (3, 7)):
            ax.plot(*vc[[h[a], h[b]]].T, color=color, lw=0.6, alpha=0.85)


def render(item: dict, path: Path, title: str) -> None:
    vc = np.asarray(item.get("vertices_cartesian",
                             item.get("vertices_cart")))
    faces = np.asarray(item["faces"])
    sp = item.get("surface_points")
    fig = plt.figure(figsize=(12, 5))
    views = [("x", "y", 0, 1), ("x", "z", 0, 2), ("theta-proxy", "z", 2, 1)]
    for k, (xl, yl, i, j) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, k)
        if "surface_points" in item:
            sp = np.asarray(item["surface_points"])
            ax.scatter(sp[:, i], sp[:, j], s=0.4, c="lightgray", alpha=0.3)
        hexes = faces.reshape(-1, 8)
        draw = [("gray", h) for h in hexes]
        for _, h in draw:
            for e in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
                      (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)):
                ax.plot(vc[h[list(e)], i], vc[h[list(e)], j],
                        color="#2255aa", lw=0.8)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    items = torch.load(SRC, weights_only=False)
    keep = [it for it in items if it.get("subdiv_n") == 2]
    out = ROOT / "data" / "fine" / "polytron_subdiv2.pt"
    torch.save(keep, out)
    print(f"filtered {len(items)} -> {len(keep)} samples (subdiv_n==2): {out}")

    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(keep), size=min(args.n, len(keep)), replace=False)
    outdir = ROOT / "data" / "fine" / "preview_subdiv2"
    outdir.mkdir(parents=True, exist_ok=True)
    for rank, i in enumerate(int(p) for p in picks):
        it = keep[i]
        vc = np.asarray(it.get("vertices_cartesian", it.get("vertices_cart")))
        nf = len(np.asarray(it["faces"]))
        png = outdir / f"pick{rank:02d}_src{i:04d}_{nf}hex.png"
        render(it, png, f"subdiv2 sample src#{i} hexes={nf} verts={len(vc)}")
        print(f"png {png.name} (src idx {i}, {nf} hexes, {len(vc)} verts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
