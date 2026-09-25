"""Dump a mapping-debug VTK: generated blocks, seam curves, their corner
vertices, and the corner->target mapping lines with labels.

Reproduces the exact production state (idx, k=8, fixed seeds) so the visual
check shows precisely what the back-mapping did.

Usage: uv run python scripts/dump_mapping_debug.py [--idx 687] [--k 8]
"""
from __future__ import annotations

import argparse
import os
import sys
import types

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from meshtron.data import conditioning  # noqa: E402
from meshtron.geometry.block_mapping import (SnapConfigV2, score_candidate,  # noqa: E402
                           snap_corners_v2)
from meshtron.training.generate import detokenize_safe, generate  # noqa: E402
from meshtron.geometry.geometry_features import FeatureModelV2  # noqa: E402
from meshtron.data.hexa_row_tokenizer import HexaRowTokenizer  # noqa: E402
from meshtron.geometry.mesh_validation import validate_generated_mesh  # noqa: E402
from meshtron.geometry.seam_graph import ep_clusters  # noqa: E402
from scripts.compare_viz import _to_cart  # noqa: E402
from scripts.map_generated_blocks import _resolve_item  # noqa: E402
from scripts.eval_family import load_model  # noqa: E402

PART_HEX, PART_SEAM, PART_SEAM_CORNER, PART_GEN_CORNER, PART_LINE = 1, 2, 3, 4, 5
TIER_ID = {"vertex": 0, "edge": 1, "surface": 2}

HEX, POLY, VERTEX, LINE = 12, 4, 1, 2


def _write_vtk(path, pts, cells, ctypes, cell_scalars, point_scalars):
    """Legacy ASCII VTK with int scalar columns on cells and points."""
    with open(path, "w") as fh:
        fh.write(f"# vtk DataFile Version 3.0\nmeshtron mapping debug {path}\n"
                 "ASCII\nDATASET UNSTRUCTURED_GRID\n")
        fh.write(f"POINTS {len(pts)} double\n")
        for p in pts:
            fh.write(f"{p[0]:.9f} {p[1]:.9f} {p[2]:.9f}\n")
        fh.write(f"CELLS {len(cells)} {sum(len(c) + 1 for c in cells)}\n")
        for c in cells:
            fh.write(f"{len(c)} " + " ".join(str(i) for i in c) + "\n")
        fh.write(f"CELL_TYPES {len(cells)}\n")
        for t in ctypes:
            fh.write(f"{int(t)}\n")
        fh.write(f"CELL_DATA {len(cells)}\n")
        for name, vals in cell_scalars.items():
            fh.write(f"SCALARS {name} int 1\nLOOKUP_TABLE default\n")
            for v in vals:
                fh.write(f"{int(v)}\n")
        fh.write(f"POINT_DATA {len(pts)}\n")
        for name, vals in point_scalars.items():
            fh.write(f"SCALARS {name} int 1\nLOOKUP_TABLE default\n")
            for v in vals:
                fh.write(f"{int(v)}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", default="data/hexarow_tokens_family_cart.pt")
    ap.add_argument("--ckpt", default="data/grpo_cart_step300.pt")
    ap.add_argument("--idx", type=int, default=687)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--tol-v", type=float, default=0.06)
    ap.add_argument("--tol-e", type=float, default=0.04)
    ap.add_argument("--feature-cache", default=os.path.join(ROOT, "data", "features"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    ck, cfg, coords, _npt, rb, zb, model, max_len, _ = load_model(
        os.path.join(ROOT, args.ckpt), dev)
    tok = HexaRowTokenizer(r_bounds=rb, z_bounds=zb)
    core = tok.core
    stop_id = core.stop_token
    specials = {core.start_token, core.end_token, core.sep_token,
                core.sep2_token, core.stop_token, core.pad_token}
    use_slot = "slot.weight" in ck["model"]
    cap = max_len - 1

    item = _resolve_item(os.path.join(ROOT, args.tokens), args.idx)
    dir_ = str(item["dir"])
    npz_path = os.path.join(ROOT, "data", "hex3d_algohex", dir_, "sample.npz")
    fm = FeatureModelV2(npz_path, cache_dir=os.path.join(ROOT, args.feature_cache))
    target = types.SimpleNamespace(curves=fm.seam_curves,
                                   surface_nearest=fm.surface_nearest)

    rng = np.random.default_rng(args.seed)
    pts, _ = conditioning.build_cloud(item, cfg["n_points"], rb, zb, rng,
                                      blade_weight=3.0)
    pc = torch.as_tensor(pts[None], dtype=torch.float32, device=dev)
    fc = torch.tensor([float(item["blocks"])], device=dev)

    snap_cfg = SnapConfigV2(tol_v=args.tol_v, tol_e=args.tol_e)
    candidates: list[dict] = []
    for i in range(args.k):
        torch.manual_seed(args.seed + i)
        seq = generate(model, pc, fc, core.start_token, stop_id, core.sep_token,
                       cap, args.temperature, 0, dev, dtype, specials,
                       use_slot, tok=tok, constrained=True, coords=coords)
        res, _ = detokenize_safe(seq, tok, stop_id, coords=coords)
        if res is None:
            continue
        vpt, blk = res
        vcart = _to_cart(vpt.numpy(), coords)
        blocks = blk.numpy()
        C_snap, records = snap_corners_v2(target, vcart[blocks], snap_cfg)
        snapped_v = vcart.copy()
        snapped_v[blocks] = C_snap
        validation = validate_generated_mesh(snapped_v, blocks)
        mean_d, min_j = score_candidate(C_snap, records)
        if validation.valid and min_j > 0.0:
            candidates.append({"blocks": blocks, "C_snap": C_snap,
                               "records": records, "mean_d": mean_d, "i": i,
                               "min_j": min_j})
    if not candidates:
        raise SystemExit("no quality-passing rollout for the chosen seed set")
    cand = min(candidates, key=lambda c: c["mean_d"])
    blocks, C_snap, records = cand["blocks"], cand["C_snap"], cand["records"]
    nb = int(blocks.shape[0])
    print(f"[debug] {dir_} rollout {cand['i']}: blocks={nb} "
          f"mean_d={cand['mean_d']:.5f} minJ={cand['min_j']:.5f}")
    seam = fm.seam_curves

    pts_parts: list[np.ndarray] = []
    cells: list[list[int]] = []
    ctypes: list[int] = []
    part_col: list[int] = []
    tier_col: list[int] = []
    cid_col: list[int] = []
    labels: list[int] = []
    n_all = 0

    def add(points, part, new_cells, new_types, tiers, cids, pt_labels) -> None:
        nonlocal n_all
        base = n_all
        pts_parts.append(np.asarray(points, float))
        labels.extend(pt_labels)
        for cell, ctype, tier, cid in zip(new_cells, new_types, tiers, cids):
            cells.append([base + int(j) for j in cell])
            ctypes.append(ctype)
            part_col.append(part)
            tier_col.append(tier)
            cid_col.append(cid)
        n_all += len(points)

    # Part 1: generated snapped blocks. C_snap rows are per-block continuous:
    # corner points are C_snap.flatten() rows b*8+k in hex corner order, so
    # the hex cell is [8b + k].
    hex_pts = C_snap.reshape(-1, 3)
    add(hex_pts, PART_HEX,
        [[8 * b + k for k in range(8)] for b in range(nb)],
        [HEX] * nb, [-1] * nb, [-1] * nb, [-1] * len(hex_pts))

    # Part 2: seam curves, one polyline cell per curve.
    for c in range(int(seam.n_curves)):
        lo, hi = int(seam.offset[c]), int(seam.offset[c + 1])
        add(seam.pts[lo:hi], PART_SEAM, [list(range(hi - lo))], [POLY],
            [-1], [c], [-1] * (hi - lo))

    # Part 3: feature-edge corner vertices (deduped seam endpoints, labeled
    # with their junction cluster id from ep_clusters).
    ep_pts = np.asarray(seam.ep_pt, float)
    cl = np.asarray(ep_clusters(seam), int)
    seen: set[int] = set()
    corner_rows: list[int] = []
    for e in range(len(ep_pts)):
        if int(cl[e]) not in seen:
            seen.add(int(cl[e]))
            corner_rows.append(e)
    add([ep_pts[e] for e in corner_rows], PART_SEAM_CORNER,
        [[j] for j in range(len(corner_rows))], [VERTEX] * len(corner_rows),
        [-1] * len(corner_rows), [-1] * len(corner_rows),
        [int(cl[e]) for e in corner_rows])

    # Part 4: generated block corner vertices (deduped); labels filled after
    # the line part is added, by matching each record to its deduped corner.
    flat = C_snap.reshape(-1, 3)
    uniq, inv = np.unique(np.round(flat, 6), axis=0, return_inverse=True)
    base4 = n_all
    corner_cid = np.full(len(uniq), -1, dtype=int)
    add([p for p in uniq], PART_GEN_CORNER,
        [[j] for j in range(len(uniq))], [VERTEX] * len(uniq),
        [-1] * len(uniq), [-1] * len(uniq), corner_cid.tolist())

    # Part 5: mapping lines generated corner -> snap target.
    tgt = np.asarray([np.asarray(r["target"], float) for r in records])
    pair_pts = np.concatenate([flat, tgt], axis=0)
    line_tiers = [TIER_ID.get(r["tier"], -1) for r in records]
    line_cids = [int(r["curve_id"]) for r in records]
    add(pair_pts, PART_LINE,
        [[j, len(flat) + j] for j in range(len(flat))],
        [LINE] * len(flat), line_tiers, line_cids, [-1] * len(pair_pts))

    for j, r in enumerate(records):
        corner_cid[int(inv[j])] = int(r["curve_id"])
    labels[base4:base4 + len(uniq)] = corner_cid.tolist()

    all_pts = np.concatenate(pts_parts, axis=0)
    out = args.out or os.path.join(
        ROOT, "data", f"map_batch__{os.path.basename(dir_)}", "mapping_debug.vtk")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    _write_vtk(out, all_pts, cells, ctypes,
               {"part": part_col, "tier": tier_col, "cid": cid_col},
               {"label": labels})
    print(f"saved {out}")
    print(f"parts: 1={ctypes.count(HEX)} hexes, 2={int(seam.n_curves)} seams, "
          f"3={len(corner_rows)} seam corners, 4={len(uniq)} generated corners, "
          f"5={ctypes.count(LINE)} mapping lines")
    print("ParaView: Threshold on 'part' (1 blocks, 2 seams, 3 seam corners, "
          "4 generated corners, 5 mapping lines); color points by 'label' "
          "(junction id / snapped curve id), lines by 'cid'/'tier'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
