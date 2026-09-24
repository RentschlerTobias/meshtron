"""map_generated_blocks.py — generierte Blockstruktur auf die reale Geometrie
mappen, bewerten, besten Kandidaten TFI-fuellen -> CFD-fertiges VTK.

Pipeline: tokens-Item -> conditioning.build_cloud (Paritaet zu eval_family)
-> k Rollouts (generate.generate) -> detokenize_safe -> Ecken auf das
sample.npz-Feature-Modell snappen (block_mapping, Dimension-Prioritaet)
-> Score (mean snap dist, min det J) -> argmin unter den qualitaetsgueltigen
-> Vergleichs-VTK (GT | gesnappt | Punktwolke), Roh-VTK und ein 3-Block-VTK
  compare3.vtk (part 1=true, 2=generated roh, 3=back_mapped gesnappt) sowie
  TFI-CFD-VTK + summary.json.

compare3.vtk stellt die drei Hex-Mengen in EINER Datei dar (kein Punktwolken-
Part) — Threshold auf 'part' zeigt true | raw | back-mapped direkt uebereinander.

  uv run python scripts/map_generated_blocks.py \
    --tokens data/hexarow_tokens_family_cart.pt \
    --ckpt data/grpo_cart_step300.pt --idx 687 --k 8

Exit 2 nur, wenn KEIN Kandidat die Validitaet besteht.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import types
from collections import defaultdict

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from meshtron.data import conditioning  # noqa: E402
from meshtron.geometry.block_mapping import SnapConfigV2, score_candidate, snap_corners_v2, tier_counts  # noqa: E402
from meshtron.geometry.curved_bridge import chord_baseline, refill_curved  # noqa: E402
from meshtron.training.generate import detokenize_safe, generate  # noqa: E402
from meshtron.geometry.geometry_features import FeatureModelV2  # noqa: E402
from meshtron.data.hexa_row_tokenizer import HexaRowTokenizer  # noqa: E402
from meshtron.geometry.mesh_validation import validate_generated_mesh  # noqa: E402
from meshtron.geometry.seam_graph import SeamNavigator  # noqa: E402
from scripts.compare_viz import _to_cart, _write_compare_vtk, _write_parts_vtk  # noqa: E402
from scripts.eval_family import load_model  # noqa: E402
from scripts.snap_selftest import _arc_targets  # noqa: E402


def _seam_path_fn(seam, records, stats, tol=0.12):
    """Seam-graph route callback for sample_on_curve (step (c) shim mode).

    Keys routes by the exact snapped corner positions (records' targets). For
    corners that fell to surface tier near a seam (d<=tol), falls back to
    seam.nearest, adds the unseen arc position to the navigator and routes
    anyway - this covers block edges hugging blade-foot loops that the snap
    tiers missed. Anything else returns None (caller's chord)."""
    arc = _arc_targets(seam, records)
    corner_ts: dict[int, list[float]] = defaultdict(list)
    pos: dict[bytes, tuple[int, float]] = {}
    for (cid, t), r in zip(arc, records):
        if cid < 0:
            continue
        corner_ts[cid].append(t)
        pos[np.round(np.asarray(r["target"], float), 12).tobytes()] = (cid, t)
    nav = SeamNavigator(seam, dict(corner_ts))

    def target_of(p):
        p = np.asarray(p, float)
        hit = pos.get(np.round(p, 12).tobytes())
        if hit is not None:
            return hit
        d, c, t, _ = seam.nearest(p[None])
        if float(d[0]) > tol:
            return None
        cid, t = int(c[0]), float(t[0])
        nav.add_point(cid, t)
        return cid, t

    def path_fn(p0, p1, n):
        a = target_of(p0)
        b = target_of(p1)
        if a is None or b is None:
            return None
        res = nav.path_between(p0, a[0], a[1], p1, b[0], b[1], n)
        if res is not None:
            stats["routes"] += 1
        return res
    return path_fn


def _resolve_item(tokens_path: str, idx: int) -> dict:
    ds = torch.load(tokens_path, weights_only=False)
    if "train" not in ds or "val" not in ds:
        raise SystemExit("--tokens muss eine Familien-Tokenfile mit train/val sein")
    order = list(ds["train"]) + list(ds["val"])
    if not 0 <= idx < len(order):
        raise SystemExit(f"--idx {idx} ausserhalb 0..{len(order) - 1}")
    return order[idx]


def main() -> int:
    ap = argparse.ArgumentParser(description="generated blocks -> real geometry -> CFD")
    ap.add_argument("--tokens", default="data/hexarow_tokens_family_cart.pt")
    ap.add_argument("--ckpt", default="data/grpo_cart_step300.pt")
    ap.add_argument("--idx", type=int, required=True)
    ap.add_argument("--k", type=int, default=8, help="Rollouts: 1=greedy, >1=stochastisch")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol-v", type=float, default=0.06)
    ap.add_argument("--tol-e", type=float, default=0.04)
    ap.add_argument("--target-h", type=float, default=0.08)
    ap.add_argument("--feature-cache", default=os.path.join(ROOT, "data", "features"),
                    help="Verzeichnis des FeatureModel-v2-Cache (data/features)")
    ap.add_argument("--band-weight", type=float, default=None,
                    help="opt-in oversample of the O-grid band surface (label 7); "
                         "None = parity conditioning, no band oversample")
    ap.add_argument("--chord-fallback", action="store_true",
                    help="TFI im Chord-Modus (tfi_bridge) statt Kurven/Coons")
    ap.add_argument("--curve-target", choices=["seam", "legacy"], default="seam",
                    help="snap target curves: seam=geometry seams (default), "
                         "legacy=fm.curves incl. GT block edges (old behavior)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    ck, cfg, coords, npt, rb, zb, model, max_len, missing = load_model(args.ckpt, dev)
    if missing.missing_keys:
        print(f"note: fehlende CKPT-Keys: {missing.missing_keys}")
    tok = HexaRowTokenizer(r_bounds=rb, z_bounds=zb)
    core = tok.core
    stop_id = core.stop_token
    specials = {core.start_token, core.end_token, core.sep_token,
                core.sep2_token, core.stop_token, core.pad_token}
    use_slot = "slot.weight" in ck["model"]
    cap = max_len - 1

    item = _resolve_item(args.tokens, args.idx)
    name, dir_ = item.get("name", f"sample{args.idx}"), item.get("dir")
    blocks_gt = int(item["blocks"])
    npz_path = os.path.join(ROOT, "data", "hex3d_algohex", str(dir_), "sample.npz")
    if not os.path.exists(npz_path):
        raise SystemExit(f"sample.npz fehlt: {npz_path}")
    fm = FeatureModelV2(npz_path, cache_dir=args.feature_cache)
    target = (types.SimpleNamespace(curves=fm.seam_curves,
                                    surface_nearest=fm.surface_nearest)
              if args.curve_target == "seam" else fm)
    safe = name.replace("/", "_")
    out_dir = args.out_dir or os.path.join(ROOT, "data", f"map_{safe}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"item idx={args.idx} name={name} dir={dir_} blocks_gt={blocks_gt} "
          f"coords={coords} npt={npt} cap={cap} use_slot={use_slot}")
    print(f"feature model v2: {len(fm.vertices)} GT-Ecken, "
          f"{fm.edge_curves.n_curves} Blockkanten-Kurven, "
          f"{fm.seam_curves.n_curves} Seam-Kurven, "
          f"{len(fm.surface_tris)} Flaeschentris -> {npz_path}")

    # Conditioning exakt wie eval_family/compare_viz: cfg-Werte, blade_weight=3.0.
    rng = np.random.default_rng(args.seed)
    if args.band_weight is not None:
        sample = dict(item)
        sample["is_band"] = conditioning.point_is_band(
            len(fm.surface_points), fm.surface_tris, fm.surface_tri_label)
    else:
        sample = item
    pts, _ = conditioning.build_cloud(sample, cfg["n_points"], rb, zb, rng,
                                      blade_weight=3.0,
                                      band_weight=args.band_weight)
    pc = torch.as_tensor(pts[None], dtype=torch.float32, device=dev)
    fc = torch.tensor([float(blocks_gt)], device=dev)
    # Punktwolke fuer Part 3, identisch zu compare_viz (gleiche rng-Reihenfolge
    # nach build_cloud); Fallback ohne surface_points ist die rohe Cloud-Inverse.
    sp = item.get("surface_points")
    if sp is not None:
        sp = sp.detach().cpu().numpy() if hasattr(sp, "detach") else sp
        sp = np.asarray(sp, dtype=np.float64)
        pc_vis = sp[rng.choice(len(sp), size=cfg["n_points"],
                               replace=len(sp) < cfg["n_points"])]
    else:
        th = np.arctan2(pts[:, 1], pts[:, 2])
        r_real = rb[0] + pts[:, 0] * (rb[1] - rb[0])
        pc_vis = np.stack([r_real * np.cos(th), r_real * np.sin(th),
                           zb[0] + pts[:, 3] * (zb[1] - zb[0])], axis=-1)

    k = max(1, args.k)
    greedy = k == 1
    top_k = 1 if greedy else 0
    snap_cfg = SnapConfigV2(tol_v=args.tol_v, tol_e=args.tol_e)
    tfi_mode = "chord_fallback" if args.chord_fallback else "curved"
    print(f"generation: k={k} {'greedy top_k=1' if greedy else 'stochastic top_k=0'} "
          f"temperature={args.temperature} seed={args.seed} | snap tol_v={args.tol_v} "
          f"tol_e={args.tol_e} target_h={args.target_h} tfi={tfi_mode}")

    rollouts: list[dict] = []
    candidates: list[dict] = []
    for i in range(k):
        torch.manual_seed(args.seed + i)
        seq = generate(model, pc, fc, core.start_token, stop_id, core.sep_token,
                       cap, args.temperature, top_k, dev, dtype, specials,
                       use_slot, tok=tok, constrained=True, coords=coords)
        stopped = seq[-1] == stop_id
        res, trim = detokenize_safe(seq, tok, stop_id, coords=coords)
        rec = {"rollout": i, "n_tokens": len(seq), "stopped": bool(stopped),
               "trim": trim is not None, "detok_ok": res is not None,
               "n_blocks": 0, "quality_passing": False, "errors": []}
        if res is None:
            rec["errors"].append(f"detokenize fehlgeschlagen: {trim}")
            print(f"rollout {i}: detok FEHLGESCHLAGEN ({trim})")
            rollouts.append(rec)
            continue
        vpt, blk = res
        vcart = _to_cart(vpt.numpy(), coords)
        blocks = blk.numpy()
        C = vcart[blocks]                              # (nb,8,3) VTK-Hex-Ecken
        C_snap, records = snap_corners_v2(target, C, snap_cfg)
        snapped_v = vcart.copy()
        snapped_v[blocks] = C_snap
        validation = validate_generated_mesh(snapped_v, blocks)
        mean_d, min_j = score_candidate(C_snap, records)
        rec.update({"n_blocks": int(blocks.shape[0]),
                    "tier_counts": tier_counts(records),
                    "mean_snap_dist": round(mean_d, 6),
                    "min_hex_jacobian": round(min_j, 6),
                    "structural_valid": bool(validation.valid),
                    "errors": list(validation.errors)})
        rec["quality_passing"] = bool(validation.valid and min_j > 0.0)
        rec["_C_snap"] = C_snap
        rec["_records"] = records
        rec["_snapped_v"] = snapped_v
        rec["_raw_v"] = vcart
        rec["_blocks"] = blocks
        candidates.append(rec)
        print(f"rollout {i}: tok={len(seq)} stop={'ja' if stopped else 'NEIN'} "
              f"blocks={blocks.shape[0]} tiers={rec['tier_counts']} "
              f"mean_d={mean_d:.5f} minJ={min_j:.5f} "
              f"{'OK' if rec['quality_passing'] else 'invalid'}")
        rollouts.append({kk: vv for kk, vv in rec.items() if not kk.startswith("_")})

    passing = [c for c in candidates if c["quality_passing"]]
    # Bekanntes Datensatz-Artefakt: die groben GT-Bloecke selbst enthalten
    # gefaltete Zellen (min det J < 0, dokumentiert fuer Block [4,7]); strikte
    # Positivitaet ist damit fuer manche Geometrien unerreichbar. Fallback:
    # unter den strukturell gueltigen Kandidaten waehlen und das markieren.
    positivity_fallback = False
    if not passing:
        passing = [c for c in candidates if c["structural_valid"]]
        positivity_fallback = len(passing) > 0
    summary = {"item": {"idx": args.idx, "name": name, "dir": dir_,
                        "blocks_gt": blocks_gt, "coords": coords},
               "params": {"k": k, "temperature": args.temperature, "seed": args.seed,
                          "tol_v": args.tol_v, "tol_e": args.tol_e,
                       "target_h": args.target_h, "n_points": cfg["n_points"],
                           "blade_weight": 3.0, "greedy": greedy,
                           "band_weight": args.band_weight,
                           "curve_target": args.curve_target},
               "rollouts": rollouts, "n_candidates": len(candidates),
               "n_quality_passing": sum(c["quality_passing"] for c in candidates),
               "positivity_fallback": positivity_fallback}
    if not passing:
        summary["chosen"] = None
        with open(os.path.join(out_dir, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"KEIN Kandidat gueltig ({len(candidates)}/{k} detok, 0 passing)")
        print(f"wrote {os.path.join(out_dir, 'summary.json')}")
        return 2

    best = min(passing, key=lambda c: c["mean_snap_dist"])
    blocks = best["_blocks"]
    gt_blocks = [[int(j) for j in b] for b in fm.blocks]

    compare = os.path.join(out_dir, "compare.vtk")
    _write_compare_vtk(compare, fm.vertices, gt_blocks, best["_snapped_v"],
                       [[int(j) for j in b] for b in blocks], pc_vis)
    raw = os.path.join(out_dir, "compare_raw.vtk")
    _write_compare_vtk(raw, fm.vertices, gt_blocks, best["_raw_v"],
                       [[int(j) for j in b] for b in blocks], pc_vis)
    # Drei Hex-Mengen in EINER Datei, kein Punktwolken-Part (Threshold auf part).
    compare3 = os.path.join(out_dir, "compare3.vtk")
    _write_parts_vtk(compare3,
                     [(fm.vertices, gt_blocks, 1, 12),
                      (best["_raw_v"], [[int(j) for j in b] for b in blocks], 2, 12),
                      (best["_snapped_v"], [[int(j) for j in b] for b in blocks], 3, 12)],
                     f"meshtron {name} r{best['rollout']} (true|generated|back_mapped)")
    Csnap = best["_C_snap"]
    route_stats = {"routes": 0}
    path_fn = (_seam_path_fn(fm.seam_curves, best["_records"], route_stats)
               if args.curve_target == "seam" else None)
    cfd = os.path.join(out_dir, "cfd_refill.vtk")
    curved_path = os.path.join(out_dir, "cfd_curved.vtk")
    chord_path = os.path.join(out_dir, "cfd_chord.vtk")
    rep_curved = refill_curved(Csnap, args.target_h, curved_path, fm=target,
                               path_fn=path_fn)
    rep_chord = chord_baseline(Csnap, args.target_h, chord_path)
    inv_c, inv_ch = rep_curved["inverted_curved"], rep_chord["inverted_chord"]
    if args.chord_fallback:
        effective, reason = "chord", "--chord-fallback"
    elif inv_c <= inv_ch:
        effective, reason = "curved", None
    else:
        effective, reason = "chord", f"curved inverted {inv_c} > chord {inv_ch}"
    tfi_report = rep_curved if effective == "curved" else rep_chord
    shutil.copyfile(curved_path if effective == "curved" else chord_path, cfd)
    curved_edges = os.path.splitext(curved_path)[0] + "_edges.vtk"
    cfd_edges = os.path.join(out_dir, "cfd_refill_edges.vtk")
    if os.path.exists(curved_edges):
        shutil.copyfile(curved_edges, cfd_edges)

    summary["chosen"] = {kk: vv for kk, vv in best.items() if not kk.startswith("_")}
    summary["curve_target"] = args.curve_target
    summary["coverage"] = {"seam_graph_routes": route_stats["routes"],
                           **rep_curved.get("buckets", {})}
    summary["tfi"] = tfi_report
    summary["tfi_effective"] = {"mode": effective, "reason": reason}
    summary["tfi_compare"] = {
        "chosen_mode": tfi_mode, "inverted_curved": int(inv_c),
        "inverted_chord": int(inv_ch),
        "watertight_curved": bool(rep_curved["watertight"]),
        "effective": effective, "cells": int(tfi_report["cells_after"])}
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"--- chosen rollout {best['rollout']}: mean_d={best['mean_snap_dist']:.6f} "
          f"minJ={best['min_hex_jacobian']:.6f} tiers={best['tier_counts']} "
          f"blocks={best['n_blocks']}")
    print(f"--- tfi[effective={effective}]: mode={tfi_report['mode']} "
          f"watertight={tfi_report['watertight']} cells={tfi_report['cells_after']} "
          f"bnd_faces={tfi_report['boundary_faces']} reason={reason}")
    print(f"--- INVERTED curved={inv_c} vs chord={inv_ch} (headline)")
    print(f"saved {compare}  (part: 1=GT, 2=gesnappt, 3=Punktwolke)")
    print(f"saved {raw}      (part: 1=GT, 2=roh/ungesnappt, 3=Punktwolke)")
    print(f"saved {compare3} (part: 1=true, 2=generated roh, 3=back_mapped gesnappt)")
    print(f"saved {cfd}      (CFD-Volumen, TFI)")
    if os.path.exists(cfd_edges):
        print(f"saved {cfd_edges} (1D Blockkanten-Kurven, coverage={summary['coverage']})")
    print(f"saved {os.path.join(out_dir, 'summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
