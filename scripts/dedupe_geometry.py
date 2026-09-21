"""Blade-surface geometry identity for the AlgoHex batch machines.

Every base machine dir (machine_XXXX) carries the input surface in
machine_XXXX_tet.vtk (ASCII UNSTRUCTURED_GRID, CELL_DATA SCALARS color).
The blade surface is the set of cells tagged color==5. Identity is the
blake2b-16 digest of the blade vertex coordinates quantized to 1e-4,
deduplicated and sorted lexicographically, so it is stable across AlgoHex
n-parameter runs of one geometry.

Scans data/hex3d_algohex/batch/ and its sibling batch_t19_sweep/ (both roots).
The n-variant dirs (_n2000/_n8000, _n1000/_n4000) hold no tet.vtk of their own,
so every run of a machine resolves to the base dir's geometry; the emitted map
is keyed by base machine id. Deterministic: sorted discovery, sorted hashing,
json sort_keys=True, no timestamps.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "hex3d_algohex"
SCAN_DIRS = ("batch", "batch_t19_sweep")
OUT = ROOT / "data" / "geom_ids.json"
MACHINE_RE = re.compile(r"^machine_\d+$")
BLADE_TAG = 5
QUANT = 1.0e4
DIGEST = 16


class VtkError(Exception):
    """Malformed or truncated ASCII VTK input."""


def _seek(lines: list[str], prefix: str, start: int) -> int:
    for i in range(start, len(lines)):
        if lines[i].startswith(prefix):
            return i
    raise VtkError(f"section {prefix!r} not found")


def blade_geom_id(path: str) -> str | None:
    """blake2b hex of quantized blade vertices, or None if no color-5 cells."""
    lines = Path(path).read_text().splitlines()
    i = _seek(lines, "POINTS", 0)
    npts = int(lines[i].split()[1])
    pts = np.array(
        [[float(x) for x in lines[i + 1 + k].split()] for k in range(npts)],
        dtype=np.float64,
    )
    j = _seek(lines, "CELLS", i + npts + 1)
    ncell = int(lines[j].split()[1])
    m = _seek(lines, "LOOKUP_TABLE", j + 1)
    col = np.fromiter(
        (int(lines[m + 1 + t]) for t in range(ncell)),
        dtype=np.int64,
        count=ncell,
    )
    blade = np.flatnonzero(col == BLADE_TAG)
    if blade.size == 0:
        return None
    idx = np.fromiter(
        (int(v) for t in blade for v in lines[j + 1 + int(t)].split()[1:]),
        dtype=np.int64,
    )
    rows = np.round(pts[np.unique(idx)] * QUANT).astype(np.int64)
    rows = np.unique(rows, axis=0)
    return hashlib.blake2b(rows.tobytes(), digest_size=DIGEST).hexdigest()


def _proc(job: tuple[str, str, str]) -> tuple[str, str, str | None, str | None]:
    machine, source, path = job
    try:
        return machine, source, blade_geom_id(path), None
    except Exception as exc:  # boundary catch: one bad file never kills the run
        return machine, source, None, f"{type(exc).__name__}: {exc}"


def discover() -> dict[str, list[str]]:
    machines: dict[str, list[str]] = {}
    for d in SCAN_DIRS:
        base = DATA / d
        if not base.is_dir():
            raise SystemExit(f"scan dir missing: {base}")
        for entry in sorted(base.iterdir()):
            if entry.is_dir() and MACHINE_RE.match(entry.name):
                machines.setdefault(entry.name, []).append(d)
    return machines


def build_jobs(
    machines: dict[str, list[str]],
) -> tuple[list[tuple[str, str, str]], list[str]]:
    jobs: list[tuple[str, str, str]] = []
    no_tet: list[str] = []
    for name in sorted(machines):
        found = False
        for d in machines[name]:
            path = DATA / d / name / f"{name}_tet.vtk"
            if path.is_file():
                jobs.append((name, d, str(path)))
                found = True
        if not found:
            no_tet.append(name)
    return jobs, no_tet


def main() -> None:
    machines = discover()
    jobs, no_tet = build_jobs(machines)
    print(f"scan roots: {', '.join(SCAN_DIRS)}")
    print(f"machines: {len(machines)}  tet files: {len(jobs)}  dirs without tet: {len(no_tet)}")

    per_machine: dict[str, list[tuple[str, str | None]]] = {n: [] for n in machines}
    parse_errors: dict[str, str] = {
        n: "tet.vtk not found in either scan root" for n in no_tet
    }
    ctx = mp.get_context("fork")
    with ctx.Pool(mp.cpu_count()) as pool:
        for n, (machine, source, gid, err) in enumerate(pool.imap(_proc, jobs), 1):
            if err is None:
                per_machine[machine].append((source, gid))
            else:
                parse_errors.setdefault(machine, err)
            if n % 100 == 0 or n == len(jobs):
                tag = gid[:12] if gid else ("no-blade" if err is None else "error")
                print(f"  [{n}/{len(jobs)}] {machine} ({source}): {tag}", flush=True)

    geom_ids: dict[str, str | None] = {}
    blade_missing: list[str] = []
    conflicts: dict[str, list[str]] = {}
    for name in sorted(machines):
        vals = per_machine[name]
        if name in parse_errors and not vals:
            geom_ids[name] = None
            continue
        gids = sorted({g for _, g in vals if g is not None})
        if not gids:
            geom_ids[name] = None
            blade_missing.append(name)
            continue
        if len(gids) > 1:
            conflicts[name] = gids
        geom_ids[name] = gids[0]

    groups: dict[str, list[str]] = {}
    for name, gid in geom_ids.items():
        if gid is not None:
            groups.setdefault(gid, []).append(name)
    clusters = sorted(
        ({"geom_id": g, "size": len(v), "machines": sorted(v)} for g, v in groups.items()),
        key=lambda c: (-c["size"], c["geom_id"]),
    )

    meta = {
        "n_machines_scanned": len(machines),
        "n_files_scanned": len(jobs),
        "n_unique_geometries": len(groups),
        "blade_missing": len(blade_missing),
        "blade_missing_names": sorted(blade_missing),
        "parse_errors": dict(sorted(parse_errors.items())),
        "clusters": clusters,
        "geom_conflicts": dict(sorted(conflicts.items())),
        "scan_dirs": list(SCAN_DIRS),
        "blade_tag": BLADE_TAG,
        "quantization": QUANT,
        "digest": f"blake2b-{DIGEST}",
    }
    payload = {"_meta": meta, **{k: geom_ids[k] for k in sorted(geom_ids)}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")

    print("\nsummary")
    print(f"  {'scanned machines':<22}{len(machines)}")
    print(f"  {'tet files read':<22}{len(jobs)}")
    print(f"  {'unique geometries':<22}{len(groups)}")
    print(f"  {'blade_missing':<22}{len(blade_missing)}")
    print(f"  {'parse_errors':<22}{len(parse_errors)}")
    print(f"  {'geom conflicts':<22}{len(conflicts)}")
    print("top clusters by size:")
    for c in clusters[:3]:
        print(f"  {c['geom_id'][:16]}  {c['size']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
