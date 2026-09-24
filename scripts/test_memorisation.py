"""test_memorisation.py — does the model reproduce what it was trained on?

The question this answers: when validation loss rises while training loss keeps
falling, is the architecture wrong, or is it working and merely failing to
generalise? The two have opposite consequences -- one means fixing the model,
the other means getting more geometries -- and the loss curve alone cannot tell
them apart.

The test: generate greedily (no sampling, so what comes out is what the model
learned) for geometries FROM THE TRAINING SPLIT and for geometries from the
validation split, and compare both against their ground-truth blockings.

    reproduces train, fails val   ->  the transformer works, it overfits
    fails both                    ->  something is broken upstream of training

Metrics, per item:

    blocks_exact      the generated blocking has the ground-truth block count
    blocks_matched    fraction of GT blocks reproduced exactly, as sets of
                      corners rounded to 1e-3 (eval_family.block_set)
    token_prefix      how far the greedy sequence agrees with the GT tokens
                      before the first divergence, as a fraction
    corner_chamfer    symmetric mean nearest-neighbour distance between the
                      generated and GT corner clouds, in geometry units --
                      a graded version of blocks_matched that does not collapse
                      to 0 when a blocking is close but not identical

    uv run python scripts/test_memorisation.py --n 6
    uv run python scripts/test_memorisation.py --ckpt data/grpo_cart_step300.pt

Exit 0 always: this is a measurement, not a gate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from meshtron.data import conditioning  # noqa: E402
from meshtron.data.hexa_row_tokenizer import HexaRowTokenizer  # noqa: E402
from meshtron.training.generate import detokenize_safe, generate  # noqa: E402
from scripts.compare_viz import _to_cart  # noqa: E402
from scripts.eval_family import block_set, load_model  # noqa: E402


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric mean nearest-neighbour distance between two point sets."""
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float(0.5 * (d.min(axis=1).mean() + d.min(axis=0).mean()))


def token_prefix(gen: list[int], gt: list[int]) -> float:
    """Fraction of the GT token sequence the greedy output matches before it
    first diverges. 1.0 means the model reproduced the sequence verbatim."""
    n = min(len(gen), len(gt))
    k = 0
    while k < n and gen[k] == gt[k]:
        k += 1
    return k / max(1, len(gt))


def teacher_forced_acc(item, model, tok, cfg, rb, zb, dev, seed: int) -> dict:
    """Next-token accuracy with the ground truth fed in at every position.

    This is what the training loss measures, and it is a different quantity from
    what generation produces: one wrong token compounds, so a model can score
    high here and still diverge from the GT on its first free step. Separating
    the two is what tells memorisation apart from a structurally-correct
    generator.
    """
    import torch.nn.functional as F
    rng = np.random.default_rng(seed)
    pts, _ = conditioning.build_cloud(item, cfg["n_points"], rb, zb, rng,
                                      blade_weight=3.0)
    pc = torch.as_tensor(pts[None], dtype=torch.float32, device=dev)
    fc = torch.tensor([float(item["blocks"])], device=dev)
    t = torch.as_tensor([int(x) for x in item["tokens"].tolist()],
                        dtype=torch.long, device=dev)[None]
    with torch.no_grad():
        logits = model(t[:, :-1], pc, fc)
    pred = logits.argmax(-1)
    tgt = t[:, 1:]
    acc = float((pred == tgt).float().mean())
    loss = float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                 tgt.reshape(-1)))
    return {"tf_acc": round(acc, 4), "tf_loss": round(loss, 4)}


def evaluate_item(item, model, tok, core, cfg, rb, zb, coords, dev, dtype,
                  use_slot, cap, specials, seed: int) -> dict:
    """Greedy generation for one dataset item, scored against its own GT."""
    rng = np.random.default_rng(seed)
    pts, _ = conditioning.build_cloud(item, cfg["n_points"], rb, zb, rng,
                                      blade_weight=3.0)
    pc = torch.as_tensor(pts[None], dtype=torch.float32, device=dev)
    blocks_gt = int(item["blocks"])
    fc = torch.tensor([float(blocks_gt)], device=dev)
    torch.manual_seed(seed)
    # top_k=1 is greedy: no sampling, so the output is the model's own argmax
    # path and the comparison is not blurred by temperature.
    seq = generate(model, pc, fc, core.start_token, core.stop_token,
                   core.sep_token, cap, 1.0, 1, dev, dtype, specials,
                   use_slot, tok=tok, constrained=True, coords=coords)
    gt_toks = [int(t) for t in item["tokens"].tolist()]
    out = {"name": item["name"], "blocks_gt": blocks_gt,
           "n_tokens_gen": len(seq), "n_tokens_gt": len(gt_toks),
           "token_prefix": round(token_prefix([int(t) for t in seq], gt_toks), 4),
           "stopped": bool(seq[-1] == core.stop_token)}
    res, trim = detokenize_safe(seq, tok, core.stop_token, coords=coords)
    if res is None:
        out.update({"detok_ok": False, "blocks_exact": False,
                    "blocks_matched": 0.0, "corner_chamfer": float("nan")})
        return out
    vpt, blk = res
    out["detok_ok"] = True
    out["n_blocks"] = int(blk.shape[0])
    out["blocks_exact"] = bool(out["n_blocks"] == blocks_gt)

    # GT blocking from its own tokens, so both sides go through the identical
    # quantise/dequantise path and the comparison cannot be biased by it.
    gt_res, _ = detokenize_safe(gt_toks + [core.stop_token], tok,
                               core.stop_token, coords=coords)
    if gt_res is None:
        out.update({"blocks_matched": float("nan"),
                    "corner_chamfer": float("nan")})
        return out
    gvpt, gblk = gt_res
    gen_set = block_set(vpt, blk)
    gt_set = block_set(gvpt, gblk)
    out["blocks_matched"] = round(len(gen_set & gt_set) / max(1, len(gt_set)), 4)
    A = _to_cart(vpt.numpy(), coords)[blk.numpy()].reshape(-1, 3)
    B = _to_cart(gvpt.numpy(), coords)[gblk.numpy()].reshape(-1, 3)
    out["corner_chamfer"] = round(chamfer(np.unique(A, axis=0),
                                         np.unique(B, axis=0)), 6)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="reproduce trained geometries vs held-out ones")
    ap.add_argument("--ckpt", default="data/hexarow_sft_cart_ep584.pt")
    ap.add_argument("--tokens", default="data/hexarow_tokens_family_cart.pt")
    ap.add_argument("--n", type=int, default=6, help="items per split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    ck, cfg, coords, npt, rb, zb, model, max_len, miss = load_model(args.ckpt, dev)
    tok = HexaRowTokenizer(r_bounds=rb, z_bounds=zb)
    core = tok.core
    specials = {core.start_token, core.end_token, core.sep_token,
                core.sep2_token, core.stop_token, core.pad_token}
    use_slot = "slot.weight" in ck["model"]
    ds = torch.load(args.tokens, weights_only=False)
    print(f"ckpt {os.path.basename(args.ckpt)} coords={coords} "
          f"slot={use_slot} dev={dev} | tokens "
          f"{os.path.basename(args.tokens)}: {len(ds['train'])} train / "
          f"{len(ds['val'])} val")
    if miss.missing_keys:
        print(f"note: missing checkpoint keys: {miss.missing_keys}")

    report: dict = {"ckpt": args.ckpt, "tokens": args.tokens, "splits": {}}
    for split in ("train", "val"):
        items = list(ds[split])[:args.n]
        rows = []
        print(f"\n--- {split} ({len(items)} items, greedy)")
        for it in items:
            r = evaluate_item(it, model, tok, core, cfg, rb, zb, coords, dev,
                              dtype, use_slot, max_len - 1, specials, args.seed)
            r.update(teacher_forced_acc(it, model, tok, cfg, rb, zb, dev,
                                        args.seed))
            rows.append(r)
            print(f"  {r['name'][:34]:34s} blocks {r.get('n_blocks', '-'):>3}"
                  f"/{r['blocks_gt']:<3} matched {r.get('blocks_matched', 0):.2f}"
                  f"  prefix {r['token_prefix']:.2f}"
                  f"  chamfer {r.get('corner_chamfer', float('nan')):.4f}"
                  f"  tf-acc {r['tf_acc']:.3f}")
        ok = [r for r in rows if r.get("detok_ok")]
        agg = {
            "n": len(rows), "detok_ok": len(ok),
            "blocks_exact": round(np.mean([r["blocks_exact"] for r in ok]), 4)
            if ok else 0.0,
            "blocks_matched": round(float(np.nanmean(
                [r["blocks_matched"] for r in ok])), 4) if ok else 0.0,
            "token_prefix": round(float(np.mean(
                [r["token_prefix"] for r in rows])), 4),
            "corner_chamfer": round(float(np.nanmean(
                [r["corner_chamfer"] for r in ok])), 6) if ok else float("nan"),
            "tf_acc": round(float(np.mean([r["tf_acc"] for r in rows])), 4),
            "tf_loss": round(float(np.mean([r["tf_loss"] for r in rows])), 4),
        }
        report["splits"][split] = {"aggregate": agg, "items": rows}
        print(f"  => blocks_exact {agg['blocks_exact']:.2f}  "
              f"blocks_matched {agg['blocks_matched']:.3f}  "
              f"token_prefix {agg['token_prefix']:.3f}  "
              f"chamfer {agg['corner_chamfer']:.4f}  "
              f"tf_acc {agg['tf_acc']:.3f}  tf_loss {agg['tf_loss']:.3f}")

    tr = report["splits"]["train"]["aggregate"]
    va = report["splits"]["val"]["aggregate"]
    print(f"\n{'metric':18s} {'train':>9s} {'val':>9s}   reading")
    for k, better_low in (("tf_acc", False), ("tf_loss", True),
                          ("blocks_exact", False), ("blocks_matched", False),
                          ("token_prefix", False), ("corner_chamfer", True)):
        t, v = tr[k], va[k]
        gap = ("train better" if (t > v) != better_low else
               "val better" if t != v else "same")
        print(f"{k:18s} {t:9.4f} {v:9.4f}   {gap}")
    print("\ntf_* is teacher forced, the quantity the training loss measures. "
          "The rest is free generation. A model can overfit on the first and "
          "still not reproduce a trained blocking on the second, because one "
          "wrong token compounds -- so read the two blocks separately.")
    if args.out:
        json.dump(report, open(args.out, "w"), indent=2)
        print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
