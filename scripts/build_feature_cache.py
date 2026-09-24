#!/usr/bin/env python3
"""build_feature_cache.py — FeatureModel-v2-Cache fuer alle Keep-Runs.

Quelle: `data/dedup_inventory.json` (`keep`-Liste), npz unter
`data/hex3d_algohex/<dir>/sample.npz`. Pro Run wird `FeatureModelV2` gebaut
und nach `data/features/<dir>.pt` geschrieben (multiprocessing Pool).

Statistik: gebaute/skipped/Fehler, Timing, Bucket-Histogramm (1D-Kurvenkanten,
geschlossene/offene Seam-Kurven, degenerierte Kanten).

  uv run python scripts/build_feature_cache.py --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import get_context

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _worker(job: tuple[str, str]) -> dict:
    npz, cache_dir = job
    try:
        from meshtron.geometry.geometry_features import FeatureModelV2
        t0 = time.time()
        fm = FeatureModelV2(npz, cache_dir=cache_dir)
        import numpy as np
        segs = [fm.edge_curves.segment(c) for c in range(fm.edge_curves.n_curves)]
        chord_only = int(sum(1 for Q in segs if len(Q) < 3))
        zero_len = int(sum(1 for Q in segs
                           if len(Q) >= 2 and np.linalg.norm(Q[-1] - Q[0]) < 1e-9))
        seam_closed = int(fm.seam_curves.closed.sum())
        return {"ok": True, "run": os.path.basename(os.path.dirname(npz)),
                "edge_curves": int(fm.edge_curves.n_curves),
                "seam_curves": int(fm.seam_curves.n_curves),
                "seam_closed": seam_closed,
                "seam_open": int(fm.seam_curves.n_curves - seam_closed),
                "chord_only_edges": chord_only, "zero_length_edges": zero_len,
                "sec": float(time.time() - t0)}
    except Exception as exc:  # pro Run weiterlaufen, Fehler sammeln
        return {"ok": False, "run": os.path.basename(os.path.dirname(npz)),
                "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", default=os.path.join(ROOT, "data",
                                                        "dedup_inventory.json"))
    ap.add_argument("--npz-root", default=os.path.join(ROOT, "data",
                                                       "hex3d_algohex"))
    ap.add_argument("--cache-dir", default=os.path.join(ROOT, "data", "features"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    inv = json.load(open(args.inventory))
    keep = [r["dir"] for r in inv["runs"] if r.get("keep")]
    if args.limit:
        keep = keep[:args.limit]
    jobs = [(os.path.join(args.npz_root, d, "sample.npz"), args.cache_dir)
            for d in keep]
    exist = [j for j in jobs if os.path.exists(j[0])]
    missing = len(jobs) - len(exist)
    print(f"[cache] keep-runs {len(keep)}, npz vorhanden {len(exist)}, "
          f"fehlend {missing}, cache_dir {args.cache_dir}")

    os.makedirs(args.cache_dir, exist_ok=True)
    t0 = time.time()
    results: list[dict] = []
    with get_context("spawn").Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, exist, chunksize=4)):
            results.append(r)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(exist)}  ({time.time() - t0:.0f}s)")
    dt = time.time() - t0

    ok = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]
    stats = {
        "n_keep": len(keep), "n_npz_missing": missing, "n_built": len(ok),
        "n_failed": len(bad), "seconds": round(dt, 1),
        "sec_per_run": round(dt / max(len(exist), 1), 3),
        "edge_curves_total": int(sum(r["edge_curves"] for r in ok)),
        "seam_curves_total": int(sum(r["seam_curves"] for r in ok)),
        "seam_closed_total": int(sum(r["seam_closed"] for r in ok)),
        "seam_open_total": int(sum(r["seam_open"] for r in ok)),
        "chord_only_edges_total": int(sum(r["chord_only_edges"] for r in ok)),
        "zero_length_edges_total": int(sum(r["zero_length_edges"] for r in ok)),
        "edge_curves_hist": dict(Counter(r["edge_curves"] for r in ok)),
        "chord_only_hist": dict(Counter(r["chord_only_edges"] for r in ok)),
        "errors": [r["error"] for r in bad[:20]],
    }
    print("[cache] " + json.dumps({k: v for k, v in stats.items()
                                   if k not in ("edge_curves_hist",
                                                "chord_only_hist")}, indent=1))
    print("[cache] edge_curves histogram:", stats["edge_curves_hist"])
    print("[cache] chord_only histogram:", stats["chord_only_hist"])
    out = os.path.join(args.cache_dir, "build_stats.json")
    json.dump(stats, open(out, "w"), indent=1)
    print(f"[cache] wrote {out}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
