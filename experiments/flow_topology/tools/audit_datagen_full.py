"""Audit the new tistos_s1 samples (local extract of datagen_tistos_s1_papaya_2026-10-12.tar.xz) with the
unchanged checks of /work/relabel/audit.py. The 'surface identical to the source sample' check is
void here (new geometries): SRC is pointed at the same directory. Writes audit.csv next to this file.

    python audit_papaya.py
"""
import csv
import glob
import os
import sys
from collections import Counter

sys.path.insert(0, "/work/relabel")
import audit as A  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
A.OUT = A.SRC = os.path.join(HERE, "datagen", "runs")


def main():
    names = sorted(os.path.basename(os.path.dirname(f)) for f in glob.glob(f"{A.OUT}/*/sample.npz"))
    rows = [A.audit(n) for n in names]
    for r in rows:
        r["ok"] = not r["reasons"]
        r["reasons"] = ";".join(r["reasons"])
    keys = ["name", "ok", "reasons", "blocks_before", "blocks", "cuboids", "excess", "inverted_beam",
            "min_sj_beam", "tfi_min_sj", "tfi_inverted", "rows", "ties", "topo_code"]
    with open(os.path.join(HERE, "audit.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    ok = [r for r in rows if r["ok"]]
    print(f"audited {len(rows)}: ok {len(ok)}, rejected {len(rows) - len(ok)}")
    print("reasons:", dict(Counter(x for r in rows for x in r["reasons"].split(";") if x)))
    print("blocks before:", dict(sorted(Counter(r.get("blocks_before") for r in rows).items(), key=str)))
    print("blocks after (ok):", dict(sorted(Counter(r["blocks"] for r in ok).items())))
    print("topologies (ok):", dict(Counter(f"{r['blocks']}bl/{r.get('topo_code')}" for r in ok).most_common(8)))


if __name__ == "__main__":
    main()
