#!/usr/bin/env python3
"""Rebuild the h05 cart token file with the corrected row tokenizer.

The original build had two silent corruption sources:
1. Lex fallback for exit rings (incomplete wireframe) -> scrambled corner
   orders (net-inverted blocks).
2. No quantization-collision guard -> blocks reusing a vertex index.

This script re-tokenizes every source sample with the FIXED tokenizer
(plain tokenize, no emit_override), validates the decode against the
expected block count, and DROPS any item whose GT token stream does not
decode to a valid mesh (including items raising DegenerateBlockError at
tokenize time). Items are only replaced inside the original name set;
split membership and item order are preserved.

Usage:
  uv run python scripts/build_consistent_h05_tokens.py \
      [--tokens PATH] [--source PATH] [--out PATH] [--limit N]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch

from hexa_row_tokenizer import DegenerateBlockError, HexaRowTokenizer
from mesh_validation import validate_generated_mesh


def validate_tokens(tok: HexaRowTokenizer, tokens, expected_blocks: int):
    try:
        verts, blocks = tok.detokenize(list(int(t) for t in tokens),
                                       granularity='row', coords='cart')
    except Exception as e:  # noqa: BLE001 - report, don't crash the sweep
        return False, f"detok error {type(e).__name__}: {e}"
    res = validate_generated_mesh(np.asarray(verts), np.asarray(blocks),
                                  expected_blocks=expected_blocks)
    return bool(res.valid), "; ".join(res.errors[:3]) or "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="data/hexarow_tokens_h05_family_cart.pt")
    ap.add_argument("--source", default="data/fine/polytron_data_3d_h05_from_batch.pt")
    ap.add_argument("--out", default="data/hexarow_tokens_h05_family_cart_fixed.pt")
    ap.add_argument("--limit", type=int, default=None,
                    help="only rebuild the first N items per split (smoke test)")
    args = ap.parse_args()

    payload = torch.load(args.tokens, map_location="cpu", weights_only=False)
    src_items = torch.load(args.source, map_location="cpu", weights_only=False)
    src_by_name = {it["name"]: it for it in src_items}

    rb = tuple(float(v) for v in payload["r_bounds"])
    zb = tuple(float(v) for v in payload["z_bounds"])
    tok = HexaRowTokenizer(r_bounds=rb, z_bounds=zb)

    arms = {k: list(payload[k]) for k in ("train", "val")}
    total = sum(len(v) for v in arms.values())
    if args.limit:
        for k in arms:
            arms[k] = arms[k][:args.limit]
    sel_n = sum(len(v) for v in arms.values())
    print(f"rebuilding {sel_n}/{total} items | r_bounds={rb} z_bounds={zb}", flush=True)

    kept = {"train": 0, "val": 0}
    dropped = []
    n_valid_old = 0
    t0 = time.time()
    done = 0
    for arm, items in arms.items():
        new_list = []
        for item in items:
            src = src_by_name.get(item["name"])
            if src is None:
                dropped.append((arm, item["name"], "no-source", ""))
                done += 1
                continue
            try:
                old_ok, old_err = validate_tokens(tok, item["tokens"], item["blocks"])
                n_valid_old += int(old_ok)
                new_tokens = tok.tokenize(src, coords="cart")
            except DegenerateBlockError as e:
                dropped.append((arm, item["name"], "tokenize", str(e)[:100]))
                done += 1
                continue
            ok_new, err_new = validate_tokens(tok, new_tokens, item["blocks"])
            if not ok_new:
                dropped.append((arm, item["name"], "decode", err_new[:100]))
                done += 1
                continue
            new_item = dict(item)
            new_item["tokens"] = torch.tensor(new_tokens, dtype=torch.long)
            new_list.append(new_item)
            done += 1
            if done % 50 == 0:
                print(f"  [{done}/{sel_n}] {time.time() - t0:.0f}s | "
                      f"dropped {len(dropped)} | old-valid {n_valid_old}",
                      flush=True)
        arms[arm] = new_list
        kept[arm] = len(new_list)

    print("\n=== RESULT ===")
    print(f"kept: train {kept['train']}, val {kept['val']} "
          f"(of {total} original)")
    print(f"dropped: {len(dropped)}")
    for arm, name, reason, msg in dropped[:60]:
        print(f"  [{arm}] {name}: {reason}: {msg}")
    if len(dropped) > 60:
        print(f"  ... and {len(dropped) - 60} more")
    print(f"GT decode validity  OLD {n_valid_old}/{sel_n}  "
          f"NEW {sel_n - len(dropped)}/{sel_n} (kept items valid by construction)")

    out_payload = dict(payload)
    out_payload["train"], out_payload["val"] = arms["train"], arms["val"]
    out_payload["rebuild_note"] = (
        "fixed-tokenizer rebuild (block-local axial pairing + head-ring "
        "positivity + quantization guard); items with invalid GT decode "
        "dropped via scripts/build_consistent_h05_tokens.py")
    torch.save(out_payload, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
