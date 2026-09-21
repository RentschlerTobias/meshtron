"""map_generated_blocks.py — generierte Blockstruktur auf die reale Geometrie
mappen, bewerten, besten Kandidaten TFI-fuellen -> CFD-fertiges VTK.

Pipeline: tokens-Item -> conditioning.build_cloud (Paritaet zu eval_family)
-> k Rollouts (generate.generate) -> detokenize_safe -> Ecken auf das
sample.npz-Feature-Modell snappen (block_mapping, Dimension-Prioritaet)
-> Score (mean snap dist, min det J) -> argmin unter den qualitaetsgueltigen
-> Vergleichs-VTK (GT | gesnappt | Punktwolke), Roh-VTK und TFI-CFD-VTK
+ summary.json.

  uv run python scripts/map_generated_blocks.py \
    --tokens data/hexarow_tokens_family_cart.pt \
    --ckpt data/grpo_cart_step300.pt --idx 687 --k 8

Exit 2 nur, wenn KEIN Kandidat die Validitaet besteht.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import conditioning  # noqa: E402
from block_mapping import FeatureModel, SnapConfig, score_candidate, tier_counts  # noqa: E402
from generate import detokenize_safe, generate  # noqa: E402
from hexa_row_tokenizer import HexaRowTokenizer  # noqa: E402
from mesh_validation import validate_generated_mesh  # noqa: E402
from scripts.compare_viz import _to_cart, _write_compare_vtk  # noqa: E402
from scripts.eval_family import load_model  # noqa: E402
from tfi_bridge import refill_cfd  # noqa: E402


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
    fm = FeatureModel(npz_path)
    safe = name.replace("/", "_")
    out_dir = args.out_dir or os.path.join(ROOT, "data", f"map_{safe}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"item idx={args.idx} name={name} dir={dir_} blocks_gt={blocks_gt} "
          f"coords={coords} npt={npt} cap={cap} use_slot={use_slot}")
    print(f"feature model: {len(fm.vertices)} GT-Ecken, {len(fm.edges)} Feature-Kanten, "
          f"{len(fm.surface_tris)} Flaeschentris -> {npz_path}")

    # Conditioning exakt wie eval_family/compare_viz: cfg-Werte, blade_weight=3.0.
    rng = np.random.default_rng(args.seed)
    pts, _ = conditioning.build_cloud(item, cfg["n_points"], rb, zb, rng,
                                      blade_weight=3.0)
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
    snap_cfg = SnapConfig(tol_v=args.tol_v, tol_e=args.tol_e)
    print(f"generation: k={k} {'greedy top_k=1' if greedy else 'stochastic top_k=0'} "
          f"temperature={args.temperature} seed={args.seed} | snap tol_v={args.tol_v} "
          f"tol_e={args.tol_e} target_h={args.target_h}")

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
        C_snap, records = fm.snap_corners(C, snap_cfg)
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
                          "blade_weight": 3.0, "greedy": greedy},
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
    cfd = os.path.join(out_dir, "cfd_refill.vtk")
    tfi_report = refill_cfd(best["_C_snap"], args.target_h, cfd,
                            title=f"meshtron {name} r{best['rollout']} -> CFD (TFI)")

    summary["chosen"] = {kk: vv for kk, vv in best.items() if not kk.startswith("_")}
    summary["tfi"] = tfi_report
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"--- chosen rollout {best['rollout']}: mean_d={best['mean_snap_dist']:.6f} "
          f"minJ={best['min_hex_jacobian']:.6f} tiers={best['tier_counts']} "
          f"blocks={best['n_blocks']}")
    print(f"--- tfi: mode={tfi_report['mode']} watertight={tfi_report['watertight']} "
          f"cells={tfi_report['cells_after']} bnd_faces={tfi_report['boundary_faces']}")
    print(f"saved {compare}  (part: 1=GT, 2=gesnappt, 3=Punktwolke)")
    print(f"saved {raw}      (part: 1=GT, 2=roh/ungesnappt, 3=Punktwolke)")
    print(f"saved {cfd}      (CFD-Volumen, TFI)")
    print(f"saved {os.path.join(out_dir, 'summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
