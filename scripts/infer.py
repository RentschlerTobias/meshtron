"""infer.py — geometry in, CFD mesh out. The whole pipeline in one command.

    uv run python scripts/infer.py --npz data/hex3d_algohex/batch/X/sample.npz \
        --ckpt data/grpo_cart_step300.pt --blocks 24 --k 8

Every other entry point into this repo starts from something pre-computed: a
token file, a sample index, an existing blocking. This one starts from the
geometry alone, which is what inference on a new machine actually means:

  1  geometry     read the labelled surface out of the npz          01_geometry.vtk
  2  point cloud  the conditioning the transformer sees             02_cloud.vtk
  3  transformer  k rollouts -> coarse blocks, no geometry yet      03_blocks_raw.vtk
  4  snap         corners pulled onto seams and patches             04_blocks_snapped.vtk
  5  conform      edges routed, faces projected, TFI fills          05_cfd.vtk
  6  report       every number of every stage                       report.json

Each stage writes a VTK, so a failure is inspectable in ParaView instead of
being a number in a log, and the six files walk an audience through the
pipeline in order.

--blocks is an INPUT, not something inferred: the block count conditions the
model (FiLM) and the npz does not carry it. For a known sample the ground
truth count is in the tokens; for a genuinely new geometry it is a choice, so
--blocks-sweep takes several counts and keeps the best mesh.

Exit 0 mesh written; 2 no rollout survived; 3 geometry unusable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from meshtron.data import conditioning  # noqa: E402
from meshtron.data.hexa_row_tokenizer import HexaRowTokenizer  # noqa: E402
from meshtron.geometry.block_mapping import (SnapConfigV2,  # noqa: E402
                                             score_candidate, snap_corners_v2,
                                             tier_counts)
from meshtron.geometry.conform import (ConformOptions,  # noqa: E402
                                       collapsed_faces, conform_blocks)
from meshtron.geometry.geometry_features import FeatureModelV2  # noqa: E402
from meshtron.geometry.mesh_validation import validate_generated_mesh  # noqa: E402
from meshtron.training.generate import detokenize_safe, generate  # noqa: E402
from scripts.compare_viz import _to_cart, _write_parts_vtk  # noqa: E402
from scripts.eval_family import load_model  # noqa: E402

LABEL_NAMES = {1: "inlet", 2: "outlet", 3: "periodic_lo", 4: "periodic_hi",
               5: "hub", 6: "shroud", 7: "blade_hull"}


def write_surface_vtk(path: str, fm: FeatureModelV2, stem: str) -> dict:
    """Stage 1: the labelled surface as it comes out of the npz, one scalar per
    triangle so the seven patches are separable in ParaView. This is the only
    geometry the pipeline ever sees -- if a patch is missing here, nothing
    downstream can invent it."""
    P, T, L = fm.surface_points, fm.surface_tris, fm.surface_tri_label
    with open(path, "w") as fh:
        fh.write("# vtk DataFile Version 2.0\n"
                 f"meshtron {stem} npz labelled surface\nASCII\n"
                 "DATASET UNSTRUCTURED_GRID\n")
        fh.write(f"POINTS {len(P)} double\n")
        for q in P:
            fh.write(f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f}\n")
        fh.write(f"CELLS {len(T)} {4 * len(T)}\n")
        for t in T:
            fh.write(f"3 {t[0]} {t[1]} {t[2]}\n")
        fh.write(f"CELL_TYPES {len(T)}\n")
        for _ in T:
            fh.write("5\n")
        fh.write(f"CELL_DATA {len(T)}\nSCALARS patch_label int 1\n"
                 "LOOKUP_TABLE default\n")
        for v in L:
            fh.write(f"{int(v)}\n")
    present = {int(k): int(v) for k, v in
               zip(*np.unique(np.asarray(L), return_counts=True))}
    return {"points": int(len(P)), "tris": int(len(T)),
            "labels": {LABEL_NAMES.get(k, str(k)): v
                       for k, v in sorted(present.items())}}


def write_cloud_vtk(path: str, pts_cart: np.ndarray, is_blade, stem: str) -> None:
    """Stage 2: the conditioning cloud in real coordinates, with the blade flag
    the oversampling uses. What the picture shows is what the model is given --
    no more."""
    with open(path, "w") as fh:
        fh.write("# vtk DataFile Version 2.0\n"
                 f"meshtron {stem} conditioning cloud\nASCII\n"
                 "DATASET UNSTRUCTURED_GRID\n")
        fh.write(f"POINTS {len(pts_cart)} double\n")
        for q in pts_cart:
            fh.write(f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f}\n")
        fh.write(f"CELLS {len(pts_cart)} {2 * len(pts_cart)}\n")
        for i in range(len(pts_cart)):
            fh.write(f"1 {i}\n")
        fh.write(f"CELL_TYPES {len(pts_cart)}\n")
        for _ in pts_cart:
            fh.write("1\n")
        fh.write(f"POINT_DATA {len(pts_cart)}\nSCALARS is_blade int 1\n"
                 "LOOKUP_TABLE default\n")
        flag = (np.zeros(len(pts_cart), dtype=int) if is_blade is None
                else np.asarray(is_blade, dtype=int))
        for v in flag:
            fh.write(f"{int(v)}\n")


def cloud_from_npz(fm: FeatureModelV2, n_points: int, rb, zb, rng,
                   blade_weight: float):
    """The conditioning item built from the npz alone.

    build_cloud wants a sample dict; the two fields it reads (surface_points,
    is_blade) both come out of the geometry, so a new machine needs no dataset
    entry to be conditioned on.
    """
    item = {"surface_points": np.asarray(fm.surface_points, dtype=np.float64),
            "is_blade": conditioning.point_is_blade(
                len(fm.surface_points), fm.surface_tris, fm.surface_tri_label)}
    pts, blade_mask = conditioning.build_cloud(item, n_points, rb, zb, rng,
                                              blade_weight=blade_weight)
    return item, pts, blade_mask


def main() -> int:
    ap = argparse.ArgumentParser(
        description="geometry (npz) -> conditioning -> transformer -> CFD mesh")
    ap.add_argument("--npz", required=True, help="labelled surface geometry")
    ap.add_argument("--ckpt", default="data/grpo_cart_step300.pt")
    ap.add_argument("--blocks", type=int, default=0,
                    help="block count to condition on (0 = read fm.blocks, "
                         "which only exists for a sample that has a blocking)")
    ap.add_argument("--blocks-sweep", default="",
                    help="comma separated counts to try instead of --blocks")
    ap.add_argument("--k", type=int, default=8, help="rollouts per count")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-h", type=float, default=0.05)
    ap.add_argument("--blend-boundary", type=int, default=4)
    ap.add_argument("--no-geodesic", action="store_true")
    ap.add_argument("--no-project-faces", action="store_true")
    ap.add_argument("--blade-weight", type=float, default=3.0)
    ap.add_argument("--feature-cache", default=os.path.join(ROOT, "data", "features"))
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    stem = os.path.basename(os.path.dirname(os.path.abspath(args.npz)))
    out_dir = args.out_dir or os.path.join(ROOT, "data", f"infer_{stem}")
    os.makedirs(out_dir, exist_ok=True)
    rep: dict = {"npz": args.npz, "ckpt": args.ckpt, "stem": stem,
                 "stages": {}}

    # ---- stage 1: geometry -------------------------------------------------
    fm = FeatureModelV2(args.npz, cache_dir=args.feature_cache)
    g1 = write_surface_vtk(os.path.join(out_dir, "01_geometry.vtk"), fm, stem)
    rep["stages"]["1_geometry"] = g1
    print(f"[1] geometry {stem}: {g1['points']} pts, {g1['tris']} tris, "
          f"patches {g1['labels']}")
    if len(g1["labels"]) < 5:
        print(f"abort: only {len(g1['labels'])} patches labelled; the mapping "
              f"needs the hull patches to route on", file=sys.stderr)
        return 3

    # ---- model -------------------------------------------------------------
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    ck, cfg, coords, npt, rb, zb, model, max_len, missing = load_model(
        args.ckpt, dev)
    tok = HexaRowTokenizer(r_bounds=rb, z_bounds=zb)
    core = tok.core
    specials = {core.start_token, core.end_token, core.sep_token,
                core.sep2_token, core.stop_token, core.pad_token}
    use_slot = "slot.weight" in ck["model"]
    rep["model"] = {"coords": coords, "n_points": int(cfg["n_points"]),
                    "layers": int(cfg["layers"]), "d": int(cfg["d"]),
                    "use_slot": bool(use_slot), "max_len": int(max_len),
                    "device": dev,
                    "missing_keys": list(missing.missing_keys)}
    print(f"[.] model {os.path.basename(args.ckpt)}: coords={coords} "
          f"d={cfg['d']} layers={cfg['layers']} slot={use_slot} dev={dev}")

    # ---- stage 2: conditioning cloud --------------------------------------
    rng = np.random.default_rng(args.seed)
    item, pts, blade_mask = cloud_from_npz(fm, int(cfg["n_points"]), rb, zb,
                                           rng, args.blade_weight)
    sp = item["surface_points"]
    vis_rng = np.random.default_rng(args.seed)
    vis_idx = vis_rng.choice(len(sp), size=int(cfg["n_points"]),
                             replace=len(sp) < int(cfg["n_points"]))
    write_cloud_vtk(os.path.join(out_dir, "02_cloud.vtk"), sp[vis_idx],
                    item["is_blade"][vis_idx], stem)
    rep["stages"]["2_cloud"] = {
        "n_points": int(len(pts)), "blade_weight": args.blade_weight,
        "blade_fraction_surface": float(np.mean(item["is_blade"])),
        "blade_fraction_sampled": (float(np.mean(blade_mask))
                                   if blade_mask is not None else None)}
    print(f"[2] cloud: {len(pts)} points, blade {np.mean(item['is_blade']):.1%} "
          f"of surface -> {np.mean(blade_mask):.1%} of samples "
          f"(weight {args.blade_weight})")
    pc = torch.as_tensor(pts[None], dtype=torch.float32, device=dev)

    # ---- stage 3: transformer ---------------------------------------------
    if args.blocks_sweep:
        counts = [int(c) for c in args.blocks_sweep.split(",") if c.strip()]
    elif args.blocks > 0:
        counts = [args.blocks]
    else:
        counts = [int(len(fm.blocks))]
        print(f"[3] --blocks not given, using the blocking in the npz "
              f"({counts[0]} blocks); a new geometry needs --blocks")
    cands: list[dict] = []
    rollouts: list[dict] = []
    for nb_cond in counts:
        fc = torch.tensor([float(nb_cond)], device=dev)
        for i in range(args.k):
            torch.manual_seed(args.seed + i)
            seq = generate(model, pc, fc, core.start_token, core.stop_token,
                           core.sep_token, max_len - 1, args.temperature,
                           1 if args.k == 1 else 0, dev, dtype, specials,
                           use_slot, tok=tok, constrained=True, coords=coords)
            res, trim = detokenize_safe(seq, tok, core.stop_token, coords=coords)
            r = {"cond_blocks": nb_cond, "rollout": i, "n_tokens": len(seq),
                 "stopped": bool(seq[-1] == core.stop_token),
                 "detok_ok": res is not None}
            if res is None:
                r["error"] = f"detokenize failed: {trim}"
                rollouts.append(r)
                print(f"[3] cond={nb_cond} r{i}: detok FAILED")
                continue
            vpt, blk = res
            vcart = _to_cart(vpt.numpy(), coords)
            blocks = blk.numpy()
            val = validate_generated_mesh(vcart, blocks)
            r.update({"n_blocks": int(blocks.shape[0]),
                      "structural_valid": bool(val.valid),
                      "errors": list(val.errors)})
            r["_v"], r["_b"] = vcart, blocks
            rollouts.append(r)
            cands.append(r)
            print(f"[3] cond={nb_cond} r{i}: tok={len(seq)} "
                  f"blocks={blocks.shape[0]} "
                  f"{'valid' if val.valid else 'INVALID: ' + str(val.errors[:1])}")
    rep["stages"]["3_transformer"] = {
        "counts": counts, "k": args.k, "temperature": args.temperature,
        "rollouts": [{k: v for k, v in r.items() if not k.startswith("_")}
                     for r in rollouts]}
    if not cands:
        json.dump(rep, open(os.path.join(out_dir, "report.json"), "w"), indent=2)
        print("abort: no rollout detokenized", file=sys.stderr)
        return 2

    # ---- stage 4: snap onto the geometry ----------------------------------
    target = types.SimpleNamespace(curves=fm.seam_curves,
                                   surface_nearest=fm.surface_nearest)
    for r in cands:
        C = r["_v"][r["_b"]]
        C_snap, records = snap_corners_v2(target, C, SnapConfigV2())
        mean_d, min_j = score_candidate(C_snap, records)
        n_coll = collapsed_faces(r["_b"], C_snap)
        r.update({"mean_snap_dist": round(float(mean_d), 6),
                  "min_hex_jacobian": round(float(min_j), 6),
                  "collapsed_faces": int(n_coll),
                  "tier_counts": tier_counts(records)})
        r["_C"], r["_Csnap"], r["_rec"] = C, C_snap, records
        print(f"[4] cond={r['cond_blocks']} r{r['rollout']}: "
              f"mean_snap={mean_d:.5f} minJ={min_j:.5f} collapsed={n_coll} "
              f"tiers={r['tier_counts']}")
    ok = [r for r in cands
          if r["structural_valid"] and r["collapsed_faces"] == 0]
    if not ok:
        ok = [r for r in cands if r["collapsed_faces"] == 0]
    if not ok:
        rep["stages"]["4_snap"] = {"chosen": None,
                                  "note": "every candidate has collapsed "
                                          "block faces, refill would fail"}
        json.dump(rep, open(os.path.join(out_dir, "report.json"), "w"), indent=2)
        print("abort: no candidate survived snapping", file=sys.stderr)
        return 2
    best = min(ok, key=lambda r: r["mean_snap_dist"])
    v_snap = best["_v"].copy()
    v_snap[best["_b"]] = best["_Csnap"]
    blist = [[int(j) for j in b] for b in best["_b"]]
    _write_parts_vtk(os.path.join(out_dir, "03_blocks_raw.vtk"),
                     [(best["_v"], blist, 1, 12)],
                     f"meshtron {stem} generated blocks, before snapping")
    _write_parts_vtk(os.path.join(out_dir, "04_blocks_snapped.vtk"),
                     [(best["_v"], blist, 1, 12), (v_snap, blist, 2, 12)],
                     f"meshtron {stem} blocks (1=raw, 2=snapped)")
    rep["stages"]["4_snap"] = {
        "chosen": {k: v for k, v in best.items() if not k.startswith("_")},
        "n_candidates": len(cands), "n_usable": len(ok)}
    print(f"[4] chosen: cond={best['cond_blocks']} r{best['rollout']}, "
          f"{best['n_blocks']} blocks, mean_snap={best['mean_snap_dist']:.6f}")

    # ---- stage 5: conform and fill ---------------------------------------
    opt = ConformOptions(target_h=args.target_h,
                         geodesic=not args.no_geodesic,
                         project_faces=not args.no_project_faces,
                         blend_boundary=args.blend_boundary)
    out = conform_blocks(fm, best["_b"], best["_Csnap"], best["_rec"],
                         os.path.join(out_dir, "05_cfd"), opt, stem=stem)
    bnd = out["boundary"]
    rep["stages"]["5_conform"] = {
        "options": vars(opt),
        "cells": int(out["cells_after"]),
        "watertight": bool(out["watertight"]),
        "inverted": int(out["inverted_curved"]),
        "boundary": bnd, "buckets": out.get("buckets"),
        "route_stats": out["route_stats"]}
    print(f"[5] cfd: cells={out['cells_after']} "
          f"watertight={out['watertight']} inverted={out['inverted_curved']} "
          f"({100.0 * out['inverted_curved'] / max(1, out['cells_after']):.2f}%)")
    print(f"[5] boundary vs npz surface: max={bnd['max']:.3e} "
          f"p99={bnd.get('p99', float('nan')):.3e} n={bnd.get('n')}")

    # ---- stage 6: report -------------------------------------------------
    rep["verdict"] = {
        "mesh": out["out_vtk"],
        "sits_on_geometry": bool(bnd["max"] <= 1e-3),
        "watertight": bool(out["watertight"]),
        "all_cells_valid": bool(out["inverted_curved"] == 0),
        "note": "sits_on_geometry and watertight are gates; inverted cells "
                "are a known open item of the mapping (edge shape), not of "
                "the geometry"}
    rpath = os.path.join(out_dir, "report.json")
    json.dump(rep, open(rpath, "w"), indent=2)
    for f in ("01_geometry.vtk", "02_cloud.vtk", "03_blocks_raw.vtk",
              "04_blocks_snapped.vtk", "05_cfd_refill.vtk"):
        p = os.path.join(out_dir, f)
        if os.path.exists(p):
            print(f"saved {p}")
    print(f"saved {rpath}")
    return 0 if (rep["verdict"]["sits_on_geometry"]
                 and rep["verdict"]["watertight"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
