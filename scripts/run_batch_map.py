"""Batch E2E over several machines: per-machine coverage + inverted table."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENS = os.path.join(ROOT, "data", "hexarow_tokens_family_cart.pt")
SCRIPT = os.path.join(ROOT, "scripts", "map_generated_blocks.py")


def _item_index(tokens_path: str, dir_name: str) -> int:
    import torch
    ds = torch.load(tokens_path, weights_only=False)
    order = list(ds["train"]) + list(ds["val"])
    for i, item in enumerate(order):
        if str(item.get("dir")) == f"batch/{dir_name}":
            return i
    raise SystemExit(f"dir {dir_name} not in tokens file")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True,
                    help="sample dirs under data/hex3d_algohex/batch/")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--band-weight", type=float, default=None,
                    help="optional conditioning band oversampling (label 7)")
    args = ap.parse_args()

    rows = []
    for dir_ in args.dirs:
        idx = _item_index(TOKENS, dir_)
        out_dir = os.path.join(ROOT, "data", f"map_batch__{dir_}")
        cmd = [sys.executable, SCRIPT, "--tokens", TOKENS, "--ckpt",
               os.path.join(ROOT, "data", "grpo_cart_step300.pt"),
               "--idx", str(idx), "--k", str(args.k)]
        if args.band_weight is not None:
            cmd += ["--band-weight", str(args.band_weight)]
        print(f"=== {dir_} idx={idx} -> {out_dir}", flush=True)
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if proc.returncode not in (0, 2):
            print(proc.stdout[-2000:], proc.stderr[-2000:], flush=True)
            rows.append({"dir": dir_, "idx": idx, "error": f"rc={proc.returncode}"})
            continue
        summ_path = os.path.join(out_dir, "summary.json")
        if not os.path.exists(summ_path):
            rows.append({"dir": dir_, "idx": idx, "error": "no summary.json"})
            continue
        s = json.load(open(summ_path))
        cov = s.get("coverage", {})
        tfi = s.get("tfi_compare", {})
        rows.append({
            "dir": dir_, "idx": idx, "n_blocks": (s.get("chosen") or {}).get("n_blocks"),
            "mean_d": (s.get("chosen") or {}).get("mean_snap_dist"),
            "curve_target": s.get("curve_target"),
            "seam_routes": cov.get("seam_graph_routes"),
            "edges_1d_curve": cov.get("edges_1d_curve"),
            "edges_total": cov.get("edges_total"),
            "inverted_curved": tfi.get("inverted_curved"),
            "inverted_chord": tfi.get("inverted_chord"),
            "effective": tfi.get("effective"),
            "positivity_fallback": s.get("positivity_fallback"),
        })
        print(json.dumps(rows[-1]), flush=True)

    table = os.path.join(ROOT, "data", "map_batch_summary.json")
    with open(table, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"saved {table}")
    for r in rows:
        if "error" in r:
            print(f"{r['dir']}: ERROR {r['error']}")
            continue
        print(f"{r['dir']}: blocks={r['n_blocks']} mean_d={r['mean_d']} "
              f"routes={r['seam_routes']} curved={r['edges_1d_curve']}/{r['edges_total']} "
              f"inv_curved={r['inverted_curved']} inv_chord={r['inverted_chord']} "
              f"effective={r['effective']} fallback={r['positivity_fallback']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
