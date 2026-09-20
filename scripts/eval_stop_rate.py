"""eval_stop_rate.py

Evaluiert die STOP-Rate des HexaRow-Modells auf Validierungs-Samples.
Basis fuer RL GO/NoGO-Entscheidung.

  uv run python scripts/eval_stop_rate.py --n 2
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from generate import detokenize_safe, generate, load_sample, mesh_to_polar
from hexa_row_tokenizer import HexaRowTokenizer
from train_hexarow_full import GPTCond, sample_points


def _gt_block_count(faces_t):
    """faces_t kann [8,F] oder [F,8] sein -> groessere Dim = Blockzahl."""
    if faces_t is None:
        return None
    a, b = faces_t.shape[0], faces_t.shape[1]
    return int(b if b >= a else a)


def main():
    ap = argparse.ArgumentParser(
        description="HexaRow STOP-Rate auf Val-Samples")
    ap.add_argument("--ckpt", default="data/hexarow_full_model_3090.pt")
    ap.add_argument("--src", default="data/hexarow_refill_coarse_v2.pt")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--n-points", type=int, default=1000)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=40000)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32

    ck = torch.load(args.ckpt, weights_only=False)
    cfg = ck["cfg"]
    rb = tuple(float(v) for v in ck["r_bounds"])
    zb = tuple(float(v) for v in ck["z_bounds"])
    max_len = int(ck["model"]["pos.weight"].shape[0])
    if args.max_tokens > max_len - 2:
        print(f"note: --max-tokens {args.max_tokens} > pos-Matrix ({max_len}) "
              f"-> gekappt auf {max_len - 2} (sonst pos-Embedding-Overlauf)")
        args.max_tokens = max_len - 2

    model = GPTCond(ck["vocab"], cfg["d"], cfg["layers"], cfg["heads"], max_len,
                    ck["pad_id"], 0.0, cfg["n_points"], cfg["n_latent"]).to(dev)
    missing = model.load_state_dict(ck["model"], strict=False)
    if missing.missing_keys:
        print(f"note: fehlende CKPT-Keys: {missing.missing_keys}")
    model.eval()
    tok = HexaRowTokenizer(r_bounds=rb, z_bounds=zb)
    core = tok.core
    specials = {core.start_token, core.end_token, core.sep_token,
                core.sep2_token, core.stop_token, core.pad_token}
    use_slot = "slot.weight" in ck["model"]
    if not use_slot:
        print("note: CKPT ohne slot.weight -> decode ohne Slot-Embedding")

    src = torch.load(args.src, weights_only=False)
    if isinstance(src, dict) and "samples" in src:
        src = src["samples"]
    n_avail = len(src)
    n_eval = min(args.n, n_avail)
    print(f"ckpt d={cfg['d']} L={cfg['layers']} H={cfg['heads']} | "
          f"src={args.src} n_avail={n_avail} n_eval={n_eval}")

    counts = dict(stop=0, cap=0, error=0, skipped=0, detok_ok=0, trim=0)
    tok_total = 0
    rows_total = 0
    gen_blocks_total = 0
    gt_total = 0
    gt_n = 0

    for i in range(n_eval):
        xyz, blocks, name, faces_t = load_sample(src, i)
        if blocks <= 0:
            print(f"{i:>3} {name:<24} blocks={blocks:<4} SKIP (no blocks)")
            counts["skipped"] += 1
            continue

        rng = np.random.default_rng(args.seed)
        pts = sample_points(mesh_to_polar(xyz), args.n_points, rb, zb, rng)
        pc = torch.as_tensor(pts[None], dtype=torch.float32, device=dev)
        fc = torch.tensor([float(blocks)], device=dev)

        try:
            seq = generate(model, pc, fc, core.start_token, core.stop_token,
                           core.sep_token, args.max_tokens, args.temperature,
                           args.top_k, dev, dtype, specials, use_slot)
        except RuntimeError as e:
            print(f"{i:>3} {name:<24} blocks={blocks:<4} ERROR generate: {e}")
            counts["error"] += 1
            continue

        stopped = seq[-1] == core.stop_token
        if stopped:
            counts["stop"] += 1
        else:
            counts["cap"] += 1

        detok_ok, trim, gen_blocks = False, False, 0
        try:
            (_vpt, blk), trim_warn = detokenize_safe(seq, tok, core.stop_token)
            if trim_warn is not None:
                trim = True
                counts["trim"] += 1
            detok_ok = True
            counts["detok_ok"] += 1
            gen_blocks = int(blk.shape[0])
        except AssertionError as e:
            print(f"{i:>3} {name:<24} blocks={blocks:<4} ERROR detok: {e}")

        n_tok = len(seq)
        n_rows = seq.count(core.sep_token)
        tok_total += n_tok
        rows_total += n_rows
        gen_blocks_total += gen_blocks
        gt_n_blocks = _gt_block_count(faces_t)
        if gt_n_blocks is not None:
            gt_total += gt_n_blocks
            gt_n += 1

        print(f"{i:>3} {name:<24} blocks={blocks}/{gt_n_blocks if gt_n_blocks is not None else '?'} "
              f"stop={'Y' if stopped else 'N'} tok={n_tok} rows={n_rows} "
              f"detok={'OK' if detok_ok else 'FAIL'} trim={'Y' if trim else 'N'} "
              f"gen_blocks={gen_blocks}")

    total = counts["stop"] + counts["cap"] + counts["error"]
    stop_pct = (100.0 * counts["stop"] / total) if total else 0.0
    print("--- summary ---")
    print(f"total={total} skipped={counts['skipped']}")
    print(f"stop={counts['stop']} ({stop_pct:.1f}%)")
    print(f"cap={counts['cap']}")
    print(f"error={counts['error']}")
    print(f"detok_ok={counts['detok_ok']} trim={counts['trim']}")
    if total:
        print(f"mean_tokens={tok_total/total:.1f}")
        print(f"mean_rows={rows_total/total:.1f}")
        print(f"mean_gen_blocks={gen_blocks_total/total:.1f}")
    if gt_n:
        print(f"mean_gt_blocks={gt_total/gt_n:.1f}")


if __name__ == "__main__":
    main()
