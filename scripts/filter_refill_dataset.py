"""Filter refill-derived samples into a training dataset.

Rules (outlier policy):
1. sample name carries target h like ..._h0.1; parse it.
2. blocks (base cell count of the structure) <= --max-blocks (users rule).
3. effective cell size h_eff = (bbox_volume / n_cells)^(1/3) must be within
   [--ratio, +ratio] of the target h (catches structures where the fill
   silently produced far more/fewer cells than requested; v16m is 2.3x off).
4. hex_tet_volumes gate: negative sub-tet volume sum must stay below
   --rel-neg of the total (bbox) volume. Planar degenerate tets (~1e-6
   absolute) pass; real folded cells (h0.5 coarses, ~5-9% of volume)
   fail. Zero-threshold would drop every sample on float noise.

Writes a flat list (compat with trainer + token builder).
"""

import argparse
import os
import re
import sys

import numpy as np
import torch


def _faces8(s):
    f = s["faces"]
    return f if f.shape[0] == 8 else f.T


def neg_tets(s):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scripts.validate_3d_dataset import hex_tet_volumes
    v = hex_tet_volumes(_faces8(s).numpy(), s["vertices_cartesian"].numpy())
    return float(v[v < 0].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/hexarow_refill_coarse.pt")
    ap.add_argument("--out", default="data/hexarow_refill_coarse_v1.pt")
    ap.add_argument("--max-blocks", type=int, default=30)
    ap.add_argument("--ratio", type=float, default=2.0,
                    help="allowed h_eff/target h deviation factor")
    ap.add_argument("--rel-neg", type=float, default=1e-3,
                    help="max |neg-tet volume sum| / bbox volume")
    a = ap.parse_args()

    d = torch.load(a.src, weights_only=False)
    samples = d["samples"] if isinstance(d, dict) and "samples" in d else d

    keep, drop = [], []
    for s in samples:
        name = s.get("name", "sample?")
        m = re.search(r"_h([0-9.]+)$", name)
        h_t = float(m.group(1)) if m else None
        f = _faces8(s)
        cells = int(f.shape[1])
        bi = s.get("block_ids", None)
        blocks = len(set(bi.tolist())) if bi is not None else 0

        # effective cell size from bbox volume
        v = s["vertices_cartesian"].numpy()
        vol = float(np.prod(v.max(0) - v.min(0)))
        h_e = (vol / max(cells, 1)) ** (1 / 3)

        nt = neg_tets(s)
        reasons = []
        if a.max_blocks and blocks > a.max_blocks:
            reasons.append(f"blocks {blocks}>{a.max_blocks}")
        if h_t:
            r = h_e / h_t
            if not (1 / a.ratio <= r <= a.ratio):
                reasons.append(f"h_eff {h_e:.3f} vs target {h_t} (x{r:.2f})")
        if nt < -a.rel_neg * vol:
            reasons.append(f"neg vol {nt:.4f} ({-nt/vol:.2e} rel)")
        line = (f"{name:34s} blocks={blocks:4d} cells={cells:7d} "
                f"h_eff={h_e:.3f} target={h_t} negvol={nt:.3g}")
        if reasons:
            drop.append(f"DROPPED {line}  [{', '.join(reasons)}]")
        else:
            keep.append(s)
            print(f"KEPT    {line}")

    for dline in drop:
        print(dline)
    torch.save({"samples": keep,
                "meta": {"source": a.src, "kept": len(keep), "dropped": len(drop)}},
               a.out)
    print(f"wrote {a.out}: {len(keep)} samples kept, {len(drop)} dropped")


if __name__ == "__main__":
    main()
