#!/usr/bin/env python3
"""Tables for curve_opt.py runs: mesh metrics and kinks per variant (medians over the items)."""
import json
import sys

import numpy as np

VARIANTS = ("pred", "pred_seam", "pred_opt", "pred_gar", "gt", "gt_seam", "gt_opt", "gt_gar")


def main() -> int:
    for path in sys.argv[1:]:
        rows = json.load(open(path))
        print(f"\n## {path}  ({len(rows)} items)\n")
        print("| variant | meshed | uncovered med | inv % med | no inverted | sj_p1 med | sj_min med | nonortho_max med "
              "| ring kink med / p90 / max | open feature kink p90 | ambiguous kink med |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for v in VARIANTS:
            ok = [r[v] for r in rows if v in r and "error" not in r[v]]
            if not ok:
                continue
            g = lambda k: np.array([m[k] for m in ok])  # noqa: E731
            kk = lambda c: np.concatenate([m["kinks"].get(c, []) for m in ok] or [[np.nan]])  # noqa: E731
            ring, opn, amb = kk("feature_closed"), kk("feature_open"), kk("ambiguous")
            print(f"| {v} | {len(ok)}/{len(rows)} | {np.median(g('uncovered')):.4f} | {100 * np.median(g('inv_share')):.3f} "
                  f"| {int((g('inv_share') == 0).sum())} | {np.median(g('sj_p1')):.3f} | {np.median(g('sj_min')):.3f} "
                  f"| {np.median(g('nonortho_max')):.1f} | {np.median(ring):.1f} / {np.percentile(ring, 90):.1f} / {ring.max():.1f} "
                  f"| {np.percentile(opn, 90):.1f} | {np.median(amb):.1f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
