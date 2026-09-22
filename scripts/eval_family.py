"""eval_family.py — P1-3 Evaluationsharness (Val-Split, k stochastisch + 1 greedy).

Nutzt generate.generate()/detokenize_safe() und train_hexarow_full.GPTCond
unveraendert (kein Decode-/KV-Reimplementat). Conditioning ueber
conditioning.build_cloud, blade_weight=3.0, n_points/bounds/coords/npt nur aus
dem CKPT (Train-/Inferenz-Paritaet). Definitionen: grammar_valid = STOP vor Cap;
mesh_valid = detokenize_safe -> validate_generated_mesh(..., blocks_gt); valid =
beides; validity@k = gepoolte valid-Rate ueber die k Rollouts; mode_match =
Blockmenge (frozenset 1e-3-gerundeter Vertex-Tuples je Block) schneidet eine
detokisierte GT-Mode der Geometrie; edit_proxy = 1 - SequenceMatcher-Ratio
(sekundaer, kein Gate). Friction: generate.py kennt kein top_p -> --top-p nur
Provenance, stochastisch top_k=0, greedy top_k=1. Items ohne surface_points
(Alt-Tokenfiles) brauchen --src (Name-Lookup).

  uv run python scripts/eval_family.py --ckpt data/hexarow_sft_cart.pt \
      --tokens data/hexarow_tokens_family_cart.pt --split val --k 8
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch

from conditioning import build_cloud
from generate import detokenize_safe, generate
from hexa_row_tokenizer import HexaRowTokenizer
from mesh_validation import hex_min_jacobian, validate_generated_mesh
from train_hexarow_full import GPTCond


def load_model(ckpt, dev):
    """GPTCond exakt wie generate.py.main aus dem CKPT (keine CLI-Raten)."""
    ck = torch.load(ckpt, weights_only=False)
    cfg = ck["cfg"]
    coords = (ck.get("coords")
              or (cfg.get("coords") if isinstance(cfg, dict) else None) or "polar")
    npt = 3 if coords == "cart" else 4
    rb = tuple(float(v) for v in ck["r_bounds"])
    zb = tuple(float(v) for v in ck["z_bounds"])
    max_len = int(ck["model"]["pos.weight"].shape[0])
    model = GPTCond(ck["vocab"], cfg["d"], cfg["layers"], cfg["heads"], max_len,
                    ck["pad_id"], 0.0, cfg["n_points"], cfg["n_latent"],
                    npt=npt).to(dev)
    missing = model.load_state_dict(ck["model"], strict=False)
    model.eval()
    return ck, cfg, coords, npt, rb, zb, model, max_len, missing


def block_set(vpt, blk):
    """Blockmenge als frozenset auf 1e-3 gerundeter Vertex-Tuples je Block."""
    v = vpt.cpu().numpy() if hasattr(vpt, "cpu") else np.asarray(vpt)
    b = blk.cpu().numpy() if hasattr(blk, "cpu") else np.asarray(blk)
    return {frozenset(tuple(np.round(v[i], 3).tolist()) for i in row) for row in b}


def evaluate_seq(seq, gt_tokens, blocks_gt, gt_modes, tok, coords, stop_id):
    """Alle Rollout-Metriken aus einer Token-Sequenz (gemeinsam fuer k+greedy)."""
    rec = {"stopped": bool(seq[-1] == stop_id), "n_tokens": len(seq),
           "trim": False, "mesh_valid": False, "valid": False, "n_blocks": 0,
           "block_exact": False, "block_within1": False, "mode_match": False,
           "mean_min_detJ": None,
           "edit_proxy": round(1.0 - difflib.SequenceMatcher(
               None, gt_tokens, seq, autojunk=False).ratio(), 6)}
    res, trim = detokenize_safe(seq, tok, stop_id, coords=coords)
    rec["trim"] = trim is not None
    if res is not None:
        vpt, blk = res
        rec["n_blocks"] = int(blk.shape[0])
        rec["block_exact"] = rec["n_blocks"] == blocks_gt
        rec["block_within1"] = abs(rec["n_blocks"] - blocks_gt) <= 1
        vc = vpt.numpy()
        vcart = (vc if coords == "cart" else np.stack(
            [vc[:, 0] * np.cos(vc[:, 1]), vc[:, 0] * np.sin(vc[:, 1]),
             vc[:, 2]], axis=-1))
        rec["mesh_valid"] = bool(validate_generated_mesh(
            vcart, blk.numpy(), expected_blocks=blocks_gt).valid)
        if rec["n_blocks"]:
            rec["mean_min_detJ"] = float(
                np.mean(hex_min_jacobian(vcart[blk.numpy()])))
        gs = block_set(vpt, blk)
        rec["mode_match"] = any(bool(gs & m) for m in gt_modes)
    rec["valid"] = rec["stopped"] and rec["mesh_valid"]
    return rec


def flush(path, obj):
    """Atomarer Zwischenstand: Crash nach Stunde 3 behaelt alle prior Items."""
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description="P1-3 Familien-Eval (k+1 Rollouts)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--split", default="val", choices=["val", "train", "all"])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default="data/eval_family.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--src", default="data/polytron_data_3d_smoke.pt",
                    help="Conditioning-Fallback per 'name' fuer Items ohne "
                         "surface_points (Alt-Tokenfiles); '' deaktiviert")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    ds = torch.load(args.tokens, weights_only=False)
    items = (list(ds["train"]) + list(ds["val"]) if args.split == "all"
             else list(ds[args.split]))
    if args.limit:
        items = items[: args.limit]

    src_by_name = {}
    if any(it.get("surface_points") is None for it in items):
        if not args.src:
            raise SystemExit("Items ohne surface_points brauchen --src")
        src = torch.load(args.src, weights_only=False)
        if isinstance(src, dict) and "samples" in src:
            src = src["samples"]
        src_by_name = {s.get("name", f"sample{i}"): s
                       for i, s in enumerate(src)}

    ck, cfg, coords, npt, rb, zb, model, max_len, missing = load_model(
        args.ckpt, dev)
    tok = HexaRowTokenizer(r_bounds=rb, z_bounds=zb)
    core = tok.core
    stop_id = core.stop_token
    specials = {core.start_token, core.end_token, core.sep_token,
                core.sep2_token, core.stop_token, core.pad_token}
    use_slot = "slot.weight" in ck["model"]
    cap = max_len - 1
    if missing.missing_keys:
        print(f"note: fehlende CKPT-Keys: {missing.missing_keys}")
    if args.top_p < 1.0:
        print(f"note: generate.py kennt kein top_p -> --top-p {args.top_p} nur "
              f"Provenance; stochastisch=Temp-Multinomial (top_k=0), greedy top_k=1")
    print(f"ckpt d={cfg['d']} L={cfg['layers']} H={cfg['heads']} coords={coords} "
          f"npt={npt} cap={cap} | {len(items)} items x (k={args.k}+1) "
          f"use_slot={use_slot}")

    prov = {"ckpt": args.ckpt, "tokens": args.tokens, "split": args.split,
            "k": args.k, "temperature": args.temperature, "top_p": args.top_p,
            "seed": args.seed, "n_points": cfg["n_points"], "blade_weight": 3.0,
            "coords": coords, "npt": npt, "limit": args.limit}
    out = {"provenance": {}, "aggregates": {}, "items": []}
    if os.path.exists(args.out_json):
        try:
            old = json.load(open(args.out_json))
            if all(old.get("provenance", {}).get(kk) == v
                   for kk, v in prov.items()):
                out = old
                print(f"resume: {len(out['items'])} Items bereits evaluiert")
        except (OSError, ValueError):
            pass
    done = {it["name"] for it in out["items"]}
    prov["timestamp"] = datetime.now(timezone.utc).isoformat()
    out["provenance"] = prov

    # GT-Modes je Geometrie aus detokenisierten GT-Tokens (Blockmengen)
    gt_modes = defaultdict(list)
    for it in items:
        gres, _ = detokenize_safe(it["tokens"].tolist(), tok, stop_id,
                                  coords=coords)
        if gres is not None:
            gt_modes[it.get("geom_id") or it["name"]].append(block_set(*gres))

    t0, ntok = time.time(), 0
    print(f"{'item':<34}{'gt':>4}{'grd':>5}{'valid':>7}{'blocks':>10}{'mode':>6}{'edit':>8}")
    for idx, it in enumerate(items):
        if it["name"] in done:
            continue
        torch.manual_seed(args.seed + idx)
        raw = (it if it.get("surface_points") is not None
               else src_by_name.get(it["name"]))
        if raw is None:
            raise SystemExit(f"kein Conditioning fuer '{it['name']}'")
        pts, _ = build_cloud(raw, cfg["n_points"], rb, zb,
                             np.random.default_rng(args.seed + idx),
                             blade_weight=3.0)
        pc = torch.as_tensor(pts[None], dtype=torch.float32, device=dev)
        fc = torch.tensor([float(it["blocks"])], device=dev)
        gtok = it["tokens"].tolist()
        modes = gt_modes[it.get("geom_id") or it["name"]]

        def run(top_k, rollouts):
            nonlocal ntok
            seq = generate(model, pc, fc, core.start_token, stop_id,
                           core.sep_token, cap, args.temperature, top_k, dev,
                           dtype, specials, use_slot, tok=tok, constrained=True,
                           coords=coords)
            ntok += len(seq)
            if rollouts is not None:
                rollouts.append(evaluate_seq(seq, gtok, int(it["blocks"]),
                                             modes, tok, coords, stop_id))
            return seq

        rolls = []
        for _ in range(args.k):
            run(0, rolls)
        greedy = evaluate_seq(run(1, None), gtok, int(it["blocks"]), modes, tok,
                              coords, stop_id)

        nb = [r["n_blocks"] for r in rolls]
        detj = [r["mean_min_detJ"] for r in rolls
                if r["mean_min_detJ"] is not None]
        rec = {"name": it["name"], "geom_id": it.get("geom_id"),
               "blocks_gt": int(it["blocks"]), "rollouts": rolls,
               "greedy": greedy,
               "valid_rollouts": sum(r["valid"] for r in rolls),
               "mesh_valid_rollouts": sum(r["mesh_valid"] for r in rolls),
               "stopped_rollouts": sum(r["stopped"] for r in rolls),
               "mode_rollouts": sum(r["mode_match"] for r in rolls),
               "mean_blocks_gen": round(float(np.mean(nb)), 2) if nb else 0.0,
               "mean_min_detJ": round(float(np.mean(detj)), 6) if detj else None}
        rec["validity_rate"] = round(rec["valid_rollouts"] / max(1, args.k), 4)
        rec["mode_hit"] = rec["mode_rollouts"] > 0
        out["items"].append(rec)
        done.add(it["name"])
        flush(args.out_json, out)
        print(f"{it['name'][:34]:<34}{it['blocks']:>4}"
              f"{'Y' if greedy['stopped'] else 'N':>5}"
              f"{rec['valid_rollouts']:>4}/{args.k:<2}"
              f"{min(nb) if nb else 0:>4}-{max(nb) if nb else 0:<4}"
              f"{'Y' if rec['mode_hit'] else 'N':>6}"
              f"{np.mean([r['edit_proxy'] for r in rolls]):>8.3f}")

    recs = out["items"]
    total_r = sum(len(r["rollouts"]) for r in recs)
    geoms = defaultdict(bool)
    for r in recs:
        geoms[r.get("geom_id") or r["name"]] |= r["mode_hit"]
    detjs = [ro["mean_min_detJ"] for r in recs for ro in r["rollouts"]
             if ro["mean_min_detJ"] is not None]
    edits = [ro["edit_proxy"] for r in recs for ro in r["rollouts"]]
    out["aggregates"] = {
        "n_items": len(recs), "n_rollouts": total_r, "n_geoms": len(geoms),
        "runtime_s": round(time.time() - t0, 1),
        "validity_rate_at_k": round(sum(r["valid_rollouts"] for r in recs) / max(1, total_r), 4),
        "mesh_valid_rate": round(sum(r["mesh_valid_rollouts"] for r in recs) / max(1, total_r), 4),
        "stop_rate": round(sum(r["stopped_rollouts"] for r in recs) / max(1, total_r), 4),
        "greedy_valid_rate": round(sum(r["greedy"]["valid"] for r in recs) / max(1, len(recs)), 4),
        "mode_coverage": round(sum(geoms.values()) / max(1, len(geoms)), 4),
        "mean_blocks_gen": round(float(np.mean([r["mean_blocks_gen"] for r in recs])), 2) if recs else 0.0,
        "mean_blocks_gt": round(float(np.mean([r["blocks_gt"] for r in recs])), 2) if recs else 0.0,
        "mean_min_detJ": round(float(np.mean(detjs)), 6) if detjs else None,
        "mean_edit_proxy": round(float(np.mean(edits)), 4) if edits else None,
        "tokens_per_s": round(ntok / max(1e-9, time.time() - t0), 1)}
    flush(args.out_json, out)
    print("--- aggregate ---")
    for kk, v in out["aggregates"].items():
        print(f"  {kk}: {v}")
    print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
