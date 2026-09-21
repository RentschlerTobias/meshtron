"""Grid-identity deduplication for the AlgoHex n-parameter sweep.

Every variant run dir (machine_XXXX_nYYYY) under data/hex3d_algohex/batch/ and
its sibling batch_t19_sweep/ carries sample.npz with vertices [V,3] and hex
block connectivity [B,8]. grid_id is the blake2b-16 digest of the canonical
cell complex: quantise vertices to 1e-4, relabel node indices by lexicographic
order of the quantised coordinates, apply the relabel to the block tuples, sort
the 8 node indices of every block and sort the block list lexicographically, then
hash the canonical block bytes. The digest is invariant to node numbering and
block order - grid identity is the cell complex, not the ordering.

Geometry identity is read from data/geom_ids.json. Exactly one run per
(geom_id, grid_id) is kept: highest n in the preferred root, batch before
batch_t19_sweep (n8000 > n2000 > n4000 > n1000 > any other n); ties resolve to
the lexicographically first dir. A machine appears in both roots, so
batch/machine_X_n2000 and sweep/machine_X_n1000 compete when their grid matches.
Missing or corrupt npz files are recorded and skipped, never fatal. Deterministic:
sorted discovery, sorted hashing, json sort_keys=True, no timestamps; a re-run is
byte-identical. Fork pool, same style as scripts/dedupe_geometry.py.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "hex3d_algohex"
SCAN_DIRS = ("batch", "batch_t19_sweep")
GEOM_IDS = ROOT / "data" / "geom_ids.json"
OUT_JSON = ROOT / "data" / "dedup_inventory.json"
OUT_MD = ROOT / "reports" / "dataset_dedup.md"
VARIANT_RE = re.compile(r"^(machine_\d+)_n(\d+)$")
QUANT = 1.0e4
DIGEST = 16
ROOT_RANK = {"batch": 0, "batch_t19_sweep": 1}
KEEP_POLICY = (
    "keep one run per (geom_id,grid_id), prefer n8000 over n2000; "
    "sweep n4000 over n1000; tie -> lexicographically first root"
)


class NpzError(Exception):
    """sample.npz missing the required keys or with malformed array shapes."""


def grid_id(rel: str) -> tuple[str, int, int]:
    """Canonical grid digest plus original (n_blocks, n_vertices) of one run."""
    with np.load(DATA / rel / "sample.npz") as data:
        if "vertices" not in data.files or "blocks" not in data.files:
            raise NpzError("sample.npz lacks vertices/blocks")
        vertices = np.asarray(data["vertices"], dtype=np.float64)
        blocks = np.asarray(data["blocks"], dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise NpzError(f"vertices shape {vertices.shape}")
    if blocks.ndim != 2 or blocks.shape[1] != 8:
        raise NpzError(f"blocks shape {blocks.shape}")
    if blocks.size and (blocks.min() < 0 or blocks.max() >= vertices.shape[0]):
        raise NpzError("block node index out of range")

    quant = np.round(vertices * QUANT).astype(np.int64)
    order = np.lexsort((quant[:, 2], quant[:, 1], quant[:, 0]))
    rank = np.empty(quant.shape[0], dtype=np.int64)
    rank[order] = np.arange(quant.shape[0], dtype=np.int64)

    canon_blocks = rank[blocks]
    canon_blocks.sort(axis=1)
    keys = tuple(canon_blocks[:, c] for c in range(canon_blocks.shape[1] - 1, -1, -1))
    canon_blocks = canon_blocks[np.lexsort(keys)]

    return (
        hashlib.blake2b(canon_blocks.tobytes(), digest_size=DIGEST).hexdigest(),
        int(blocks.shape[0]),
        int(vertices.shape[0]),
    )


def _proc(job: tuple[str, str, str | None]) -> tuple:
    rel, machine, geom = job
    try:
        gid, n_blocks, n_vertices = grid_id(rel)
        return rel, machine, geom, gid, n_blocks, n_vertices, None
    except Exception as exc:  # boundary catch: one bad file never kills the run
        return rel, machine, geom, None, 0, 0, f"{type(exc).__name__}: {exc}"


def discover(geom_ids: dict[str, str | None]) -> tuple[list, dict[str, str]]:
    jobs: list[tuple[str, str, str | None]] = []
    failed: dict[str, str] = {}
    for d in SCAN_DIRS:
        base = DATA / d
        if not base.is_dir():
            raise SystemExit(f"scan dir missing: {base}")
        for entry in sorted(base.iterdir()):
            m = VARIANT_RE.match(entry.name) if entry.is_dir() else None
            if m is None:
                continue
            rel = f"{d}/{entry.name}"
            if (entry / "sample.npz").is_file():
                jobs.append((rel, m.group(1), geom_ids.get(m.group(1))))
            else:
                failed[rel] = "FileNotFoundError: sample.npz not found"
    return jobs, failed


def assign_keep(runs: list[dict]) -> None:
    groups: dict[tuple, list[dict]] = {}
    for r in runs:
        groups.setdefault((r["geom_id"], r["grid_id"]), []).append(r)
    for members in groups.values():
        best = min(
            members, key=lambda r: (ROOT_RANK[r["root"]], -int(r["n"][1:]), r["root"], r["dir"])
        )
        for r in members:
            r["keep"] = r is best


def build_geometries(runs: list[dict]) -> list[dict]:
    by_geom: dict[str | None, list[dict]] = {}
    for r in runs:
        by_geom.setdefault(r["geom_id"], []).append(r)
    geometries = [
        {
            "geom_id": geom,
            "n_runs": len(rs),
            "n_unique_grids": len({r["grid_id"] for r in rs}),
            "kept_grids": sorted({r["grid_id"] for r in rs if r["keep"]}),
        }
        for geom, rs in by_geom.items()
    ]
    geometries.sort(key=lambda g: (g["geom_id"] is not None, g["geom_id"] or ""))
    return geometries


def root_rows(runs: list[dict]) -> list[tuple[str, int, int, int, int]]:
    rows = []
    for root in SCAN_DIRS:
        rs = [r for r in runs if r["root"] == root]
        rows.append((root, len(rs), len({r["grid_id"] for r in rs}),
                     sum(r["keep"] for r in rs), sum(not r["keep"] for r in rs)))
    return rows


def write_markdown(meta: dict, runs: list[dict], geometries: list[dict], failed: int) -> None:
    hist = Counter(g["n_unique_grids"] for g in geometries)
    hist_txt = "{" + ", ".join(f"{k}: {hist[k]}" for k in sorted(hist)) + "}"
    multi = [g for g in geometries if g["n_runs"] >= 2]
    identical = sum(1 for g in multi if g["n_unique_grids"] == 1)
    pct = 100.0 * identical / len(multi) if multi else 0.0
    verdict = "dedup win" if pct >= 50 else "ueberwiegend verschiedene Grids"
    dropped = meta["n_runs_scanned"] - sum(1 for r in runs if r["keep"])
    share = 100.0 * dropped / meta["n_runs_scanned"] if meta["n_runs_scanned"] else 0.0

    lines = [
        "# Dataset-Deduplizierung AlgoHex n-Sweep",
        "",
        "Grid-Identitaet: blake2b-16 ueber den kanonikalisierten Zellkomplex "
        "(Knoten-Relabeling nach quantisierten Koordinaten, Block-Sortierung, "
        f"Quantisierung 1/{QUANT:.0e}). Geometrie-Identitaet aus "
        "`data/geom_ids.json` (read-only).",
        "",
        f"Keep-Policy: {KEEP_POLICY}.",
        "",
        "## Runs pro Scan-Root",
        "",
        "| Root | Runs | Unique Grids | Behalten | Verworfen |",
        "|---|---:|---:|---:|---:|",
    ]
    for root, n, uniq, kept, drop in root_rows(runs):
        lines.append(f"| {root} | {n} | {uniq} | {kept} | {drop} |")
    lines += [
        "",
        f"Variante-Verzeichnisse gescannt: {meta['n_runs_scanned'] + failed}; "
        f"gelesen: {meta['n_runs_scanned']}; fehlgeschlagen (npz fehlt/korrupt): "
        f"{meta['n_files_failed']}.",
        "",
        "## Verteilung n_unique_grids pro Geometrie",
        "",
        f"`{hist_txt}`",
        "",
        "## Duplikate",
        "",
        f"- Runs insgesamt: {meta['n_runs_scanned']}",
        f"- Unique Grids: {meta['n_unique_grids']}",
        f"- Geometrien: {len(geometries)}",
        f"- Geometrien mit >= 2 Runs: {len(multi)}",
        f"- Als Grid-Duplikat verworfen: {dropped} ({share:.1f} %)",
        "",
        "## Antwort: n-Sweep",
        "",
        f"{identical}/{len(multi)} Geometrien mit >= 2 Runs liefern genau ein "
        f"Grid ({pct:.1f} %) - {verdict}.",
        "",
        "## Sonderfaelle",
        "",
        f"- machine_0006 hat geom_id null; Runs ohne Geometrie: "
        f"{sum(1 for r in runs if r['geom_id'] is None)}.",
        "",
        "## Implikation P1",
        "",
        f"Trainingsset nach Dedup: {meta['n_runs_scanned'] - dropped} Runs "
        f"(ein Run pro (geom_id, grid_id)); {meta['n_unique_grids']} eindeutige "
        "Grids statt der rohen Run-Zahl. Ohne Dedup wuerden identische Grids "
        "mehrfach gewichtet.",
        "",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))


def main() -> None:
    geom_ids: dict[str, str | None] = json.loads(GEOM_IDS.read_text())
    geom_ids.pop("_meta", None)
    jobs, failed = discover(geom_ids)
    print(f"scan roots: {', '.join(SCAN_DIRS)}")
    print(f"variant dirs discovered: {len(jobs) + len(failed)}  with npz: {len(jobs)}")

    runs: list[dict] = []
    parse_errors = dict(failed)
    with mp.get_context("fork").Pool(mp.cpu_count()) as pool:
        for n, (rel, machine, geom, gid, n_blocks, n_vertices, err) in enumerate(
            pool.imap(_proc, jobs), 1
        ):
            root, name = rel.split("/")
            if err is None:
                runs.append(
                    {
                        "dir": rel,
                        "machine": machine,
                        "n": f"n{VARIANT_RE.match(name).group(2)}",
                        "root": root,
                        "geom_id": geom,
                        "grid_id": gid,
                        "keep": False,
                        "n_blocks": n_blocks,
                        "n_vertices": n_vertices,
                    }
                )
            else:
                parse_errors[rel] = err
            if n % 100 == 0 or n == len(jobs):
                print(f"  [{n}/{len(jobs)}] {rel}", flush=True)

    runs.sort(key=lambda r: (r["machine"], r["root"], int(r["n"][1:])))
    assign_keep(runs)
    geometries = build_geometries(runs)
    meta = {
        "n_runs_scanned": len(runs),
        "n_files_failed": len(parse_errors),
        "n_unique_grids": len({r["grid_id"] for r in runs}),
        "duplicate_policy": KEEP_POLICY,
    }
    write_markdown(meta, runs, geometries, len(parse_errors))
    for r in runs:
        r.pop("root")
    payload = {"_meta": meta, "runs": runs, "geometries": geometries}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")

    kept = sum(1 for r in runs if r["keep"])
    dropped = len(runs) - kept
    share = 100.0 * dropped / len(runs) if runs else 0.0
    print("\nsummary")
    print(f"  {'variant dirs discovered':<26}{len(jobs) + len(failed)}")
    print(f"  {'runs read':<26}{len(runs)}")
    print(f"  {'files failed':<26}{len(parse_errors)}")
    print(f"  {'unique grids':<26}{meta['n_unique_grids']}")
    print(f"  {'kept runs':<26}{kept}")
    print(f"  {'dropped as duplicates':<26}{dropped} ({share:.1f} %)")
    print(f"  {'geometries':<26}{len(geometries)}")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
