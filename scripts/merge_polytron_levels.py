"""Merge polytron-level datasets (baseline + TFI subdivisions) into one file.

Dedupes baseline samples by name (they are rebuilt identically per level by
augment_subdivide_3d) and guarantees a 'subdiv_n' key on every sample
(baseline -> None). Output feeds scripts/build_hexarow_tokens.py.
"""

import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    datasets = []
    for p in a.src:
        d = torch.load(p, map_location="cpu", weights_only=False)
        datasets.append(d)
        print(f"{p}: {len(d)} samples")

    merged, seen = [], set()
    skipped = 0
    import numpy as np

    for d in datasets:
        for s in d:
            name = s.get("name")
            s.setdefault("subdiv_n", None)
            if name is None:
                cn = s.get("center")
                base = (tuple(np.round(np.asarray(cn).ravel(), 6)),) \
                    if cn is not None else (id(s),)
                fp = base + (int(s["faces"].shape[1]), len(s["vertices_cartesian"]))
            else:
                fp = (name,)
            if s["subdiv_n"] is None and fp in seen:
                skipped += 1
                continue
            seen.add(fp)
            merged.append(s)
    torch.save(merged, a.out)
    print(f"merged -> {a.out}: {len(merged)} samples, {skipped} baselines "
          f"deduped")


if __name__ == "__main__":
    main()
