#!/usr/bin/env python3
"""
validate_3d_dataset.py

Validate and characterise the 3D hex-block mesh dataset under
data/hex3d_algohex/batch/ that feeds the meshtron transformer.

Pure-numpy (no torch, no network). Loads every `sample.npz` under `batch/*/`,
never lets one bad file abort the run, and emits:

  * a human summary on stdout,
  * a machine-readable JSON report (per-sample + dataset-level metrics).

Rerunnable and deterministic. Only ever WRITES the JSON report path; it never
modifies any dataset file.

Usage:
    python3 scripts/validate_3d_dataset.py
    python3 scripts/validate_3d_dataset.py --out-json /tmp/val.json --limit 20
"""

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH = REPO_ROOT / "data" / "hex3d_algohex" / "batch"
DEFAULT_OUT_JSON = DEFAULT_BATCH / "validation_summary.json"

# ---- Topology constants (VTK_HEXAHEDRON corner ordering) -------------------
# Corners: 0-3 bottom face CCW (z=-1), 4-7 top face CCW (z=+1).
HEX_FACES = [
    (0, 3, 2, 1), (4, 5, 6, 7),
    (0, 1, 5, 4), (1, 2, 6, 5),
    (2, 3, 7, 6), (3, 0, 4, 7),
]
HEX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]
# 6-tetrahedron decomposition around body diagonal 0-6 (valid for a convex hex).
TETS = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]

# Keys in the reference sample.npz (smoke2000). Only name + ndim are pinned;
# exact shapes are validated relative to the other arrays.
EXPECTED_KEYS = {
    "vertices": 2, "blocks": 2, "quad_faces": 2, "edges": 2,
    "edge_ctrl": 3, "edge_polyline": 2, "edge_polyline_offset": 1,
    "dir_class": 1, "params": 0, "quality": 0, "provenance": 0,
    "dir_class_count": 1, "surface_points": 2, "surface_tris": 2,
    "surface_tri_label": 1,
}
FLOAT_KEYS = {"vertices", "edge_ctrl", "edge_polyline", "surface_points"}


# ---- Hashing ----------------------------------------------------------------

def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def hash_vertices(vertices: np.ndarray) -> str:
    """Order-invariant hash of the vertex coordinate multiset (exact-duplicate
    detection at a fixed resolution)."""
    return _sha256(np.ascontiguousarray(np.sort(vertices, axis=0)).tobytes())


def hash_params(params) -> str:
    """Resolution-independent geometry identity from the `params` payload.

    Canonicalises the JSON (sort_keys) so formatting differences do not split
    one geometry into two hashes. Falls back to the raw string."""
    if isinstance(params, np.ndarray):
        params = params.item()
    raw = str(params)
    try:
        return _sha256(json.dumps(json.loads(raw), sort_keys=True).encode("utf-8"))
    except Exception:
        return _sha256(raw.encode("utf-8"))


# ---- Topology helpers -------------------------------------------------------

def _sorted_edge(u, v):
    return (int(u), int(v)) if u < v else (int(v), int(u))


def block_face_multiset(blocks):
    fc = Counter()
    for b in blocks:
        for f in HEX_FACES:
            fc[tuple(sorted(int(b[c]) for c in f))] += 1
    return fc


def block_edge_multiset(blocks):
    ec = Counter()
    for b in blocks:
        for (u, v) in HEX_EDGES:
            ec[_sorted_edge(b[u], b[v])] += 1
    return ec


def quad_face_set(quad_faces):
    return {tuple(sorted(int(q[c]) for c in range(4))) for q in quad_faces}


def quad_edge_multiset(quad_faces):
    qe = Counter()
    for q in quad_faces:
        for i in range(4):
            qe[_sorted_edge(q[i], q[(i + 1) % 4])] += 1
    return qe


def hex_tet_volumes(blocks, vertices):
    """(F,6) signed tet volumes of the 6-tet decomposition of each block."""
    V = vertices[blocks]
    cols = []
    for (a, b, c, d) in TETS:
        p0, p1, p2, p3 = V[:, a], V[:, b], V[:, c], V[:, d]
        m = np.stack([p1 - p0, p2 - p0, p3 - p0], axis=-1)
        cols.append(np.linalg.det(m) / 6.0)
    return np.stack(cols, axis=1)


def split_name(name):
    if name.startswith("machine_"):
        base, res = name.rsplit("_n", 1)
        return base, int(res)
    if name.startswith("T1_9"):
        return "T1_9", int(name.rsplit("_n", 1)[1])
    return name, None


def parse_json_scalar(arr):
    try:
        return json.loads(str(arr.item()))
    except Exception:
        return {}


def parse_json_file(path: Path):
    try:
        with open(path, "r") as fh:
            return json.load(fh)
    except Exception:
        return None


# ---- Per-sample validation --------------------------------------------------

def validate_sample(name, data):
    rec = {
        "name": name, "status": "ok", "reasons": [], "warnings": [],
        "keys_missing": [], "keys_extra": [],
    }
    hard, warn = rec["reasons"], rec["warnings"]
    mid, res = split_name(name)
    rec["machine"] = mid
    rec["resolution"] = res
    rec["is_t1"] = name.startswith("T1_9")

    keys = set(data.keys())
    rec["keys_missing"] = sorted(set(EXPECTED_KEYS) - keys)
    rec["keys_extra"] = sorted(keys - set(EXPECTED_KEYS))
    if rec["keys_missing"]:
        hard.append(f"missing keys: {rec['keys_missing']}")

    v = data.get("vertices")
    b = data.get("blocks")
    q = data.get("quad_faces")
    e = data.get("edges")
    ec = data.get("edge_ctrl")
    ep = data.get("edge_polyline")
    epo = data.get("edge_polyline_offset")
    dc = data.get("dir_class")
    sp = data.get("surface_points")
    st = data.get("surface_tris")
    dcc = data.get("dir_class_count")

    if v is None or b is None:
        rec["status"] = "failed"
        return rec

    N, F = v.shape[0], b.shape[0]
    rec.update({
        "n_vertices": int(N), "n_blocks": int(F),
        "n_quad_faces": int(q.shape[0]) if q is not None else None,
        "n_edges": int(e.shape[0]) if e is not None else None,
        "n_surface_points": int(sp.shape[0]) if sp is not None else None,
        "n_surface_tris": int(st.shape[0]) if st is not None else None,
    })

    for key, arr, ndim in (
        ("vertices", v, 2), ("blocks", b, 2), ("quad_faces", q, 2),
        ("edges", e, 2), ("edge_ctrl", ec, 3), ("edge_polyline", ep, 2),
        ("edge_polyline_offset", epo, 1), ("dir_class", dc, 1),
        ("surface_points", sp, 2), ("surface_tris", st, 2),
        ("surface_tri_label", data.get("surface_tri_label"), 1),
        ("dir_class_count", dcc, 1),
    ):
        if arr is not None and arr.ndim != ndim:
            hard.append(f"{key}: ndim {arr.ndim} != {ndim}")
    if v is not None and v.ndim == 2 and v.shape[1] != 3:
        hard.append(f"vertices: shape {v.shape} (want [N,3])")
    if b is not None and b.ndim == 2 and b.shape[1] != 8:
        hard.append(f"blocks: shape {b.shape} (want [F,8])")

    # finiteness
    for k in FLOAT_KEYS:
        arr = data.get(k)
        if arr is not None and np.issubdtype(arr.dtype, np.floating):
            if not np.isfinite(arr).all():
                nbad = int(arr.size - np.isfinite(arr).sum())
                hard.append(f"{k}: {nbad} non-finite values")

    # integer index ranges
    if b is not None and b.size and (b.min() < 0 or b.max() >= N):
        hard.append(f"blocks: index out of [0,{N})")
    if q is not None and q.size and (q.min() < 0 or q.max() >= N):
        hard.append(f"quad_faces: index out of [0,{N})")
    if e is not None and e.size and (e.min() < 0 or e.max() >= N):
        hard.append(f"edges: index out of [0,{N})")
    if st is not None and sp is not None and st.size and (st.min() < 0 or st.max() >= sp.shape[0]):
        hard.append("surface_tris: index out of range")

    # edge-array consistency
    if e is not None:
        E = e.shape[0]
        if ec is not None and ec.shape[0] != E:
            hard.append(f"edge_ctrl: {ec.shape[0]} != E={E}")
        if dc is not None and dc.shape[0] != E:
            hard.append(f"dir_class: {dc.shape[0]} != E={E}")
        if epo is not None:
            if epo.shape[0] != E + 1:
                hard.append(f"edge_polyline_offset: {epo.shape[0]} != E+1={E + 1}")
            else:
                if int(epo[0]) != 0:
                    hard.append("edge_polyline_offset: first element != 0")
                if not np.all(epo[1:] >= epo[:-1]):
                    hard.append("edge_polyline_offset: not monotonic non-decreasing")
                if ep is not None and int(epo[-1]) != ep.shape[0]:
                    hard.append("edge_polyline_offset: last != len(edge_polyline)")
                if ep is not None and (epo.min() < 0 or epo.max() > ep.shape[0]):
                    hard.append("edge_polyline_offset: out of range")

    # dir_class range + histogram (NOTE: the number of direction classes is
    # NOT fixed at 11 -- it varies per sample; 9..14 observed).
    if dc is not None and dc.size:
        lo, hi = int(dc.min()), int(dc.max())
        rec["dir_class_range"] = [lo, hi]
        rec["n_dir_classes"] = hi + 1
        if lo < 0:
            hard.append(f"dir_class: negative value (min={lo})")
        else:
            rec["dir_class_values"] = sorted(int(x) for x in np.unique(dc))
        rec["dir_class_hist"] = [int(x) for x in np.bincount(dc, minlength=hi + 1)]
    if dcc is not None and dc is not None and dc.size:
        if dcc.shape[0] != int(dc.max()) + 1:
            hard.append(f"dir_class_count: length {dcc.shape[0]} != max(dir_class)+1={int(dc.max()) + 1}")
        rec["dir_class_count_sum"] = int(np.asarray(dcc).sum())

    # duplicate vertices
    if v is not None and v.size:
        _, cnt = np.unique(np.round(v, 12), axis=0, return_counts=True)
        rec["n_duplicate_vertices"] = int((cnt > 1).sum())

    # ---- geometry integrity ----
    if b is not None and b.size and v is not None:
        vols = hex_tet_volumes(b, v)
        block_vol = vols.sum(axis=1)
        min_tet = vols.min(axis=1)
        rec["block_volume"] = {
            "min": float(block_vol.min()), "median": float(np.median(block_vol)),
            "max": float(block_vol.max()), "sum": float(block_vol.sum()),
        }
        # A block whose 6-tet signed sum is <= 0 is inverted (negative volume)
        # or collapsed (zero volume). Additionally flag near-zero positive
        # volumes (|vol| <= 1e-9) as collapsed: the smallest volume in a
        # healthy sample is ~0.02, so anything below 1e-9 is a numerical sliver.
        eps = 1e-9
        n_inverted = int((block_vol < -eps).sum())
        n_collapsed = int((np.abs(block_vol) <= eps).sum())
        rec["n_inverted_blocks"] = n_inverted
        rec["n_collapsed_blocks"] = n_collapsed
        rec["min_tet_volume"] = float(min_tet.min())
        if n_inverted + n_collapsed:
            hard.append(
                f"{n_inverted + n_collapsed} degenerate block(s) "
                f"({n_inverted} inverted, {n_collapsed} collapsed)"
            )
        med = float(np.median(np.abs(block_vol)))
        if med > 0:
            rec["min_volume_ratio"] = float(np.abs(block_vol).min() / med)

        face_counts = block_face_multiset(b)
        rec["n_interior_faces"] = int(sum(1 for c in face_counts.values() if c == 2))
        rec["n_boundary_faces"] = int(sum(1 for c in face_counts.values() if c == 1))
        rec["n_nonmanifold_faces"] = int(sum(1 for c in face_counts.values() if c > 2))
        if rec["n_nonmanifold_faces"]:
            hard.append(f"{rec['n_nonmanifold_faces']} face(s) shared by >2 blocks (non-manifold)")

        boundary_set = {k for k, c in face_counts.items() if c == 1}
        if q is not None:
            qset = quad_face_set(q)
            bnd_not_in_quad = boundary_set - qset
            quad_not_bnd = qset - boundary_set
            rec["n_boundary_faces_not_in_quad"] = len(bnd_not_in_quad)
            rec["n_quad_faces_not_boundary"] = len(quad_not_bnd)
            if len(q) != len(qset):
                hard.append(f"{len(q) - len(qset)} duplicate quad_faces")
            if bnd_not_in_quad:
                hard.append(f"{len(bnd_not_in_quad)} boundary block face(s) missing from quad_faces")
            if quad_not_bnd:
                hard.append(f"{len(quad_not_bnd)} quad_face(s) are not block boundary faces")

        be = block_edge_multiset(b)
        rec["edge_valence_histogram"] = dict(sorted((int(k), int(v)) for k, v in Counter(be.values()).items()))
        rec["n_edges_valence1"] = int(sum(1 for c in be.values() if c == 1))

        if e is not None:
            eset = Counter(tuple(sorted(map(int, row))) for row in e)
            edges_ok = (set(eset.keys()) == set(be.keys())) and all(c == 2 for c in eset.values())
            rec["edges_match_block_edges"] = bool(edges_ok)
            rec["n_undirected_edges"] = len(eset)
            if not edges_ok:
                hard.append("edges array != 2x directed block edges")

        if q is not None:
            qe = quad_edge_multiset(q)
            n_nm_bnd = int(sum(1 for c in qe.values() if c != 2))
            rec["n_nonmanifold_boundary_edges"] = n_nm_bnd
            if n_nm_bnd:
                hard.append(f"{n_nm_bnd} boundary edge(s) shared by != 2 boundary faces (non-manifold)")

    # quality / params payloads
    qual = parse_json_scalar(data.get("quality"))
    if qual:
        rec["quality"] = {
            "fit_residual_median": qual.get("fit_residual_median"),
            "fit_residual_p95": qual.get("fit_residual_p95"),
            "degenerate_edges": qual.get("degenerate_edges"),
            "inflection_edges": qual.get("inflection_edges"),
            "class_disagreements": qual.get("class_disagreements"),
            "edges_undirected": qual.get("edges_undirected"),
            "blocks_without_lattice": qual.get("blocks_without_lattice"),
        }
        if qual.get("blocks_without_lattice"):
            warn.append(f"blocks_without_lattice={qual['blocks_without_lattice']}")
        if qual.get("class_disagreements"):
            warn.append(f"class_disagreements={qual['class_disagreements']}")
    raw_params = str(data["params"].item()).strip() if "params" in data else ""
    rec["params_empty"] = raw_params in ("", "{}")
    rec["params_hash"] = hash_params(data["params"])
    rec["vertex_hash"] = hash_vertices(v)

    if hard:
        rec["status"] = "failed"
    return rec


# ---- Cross-checks -----------------------------------------------------------

def reconcile(batch: Path):
    entries = sorted(os.listdir(batch))
    bare = sorted(e for e in entries if e.startswith("machine_") and "_n" not in e)
    res = sorted(e for e in entries if (e.startswith("machine_") and "_n" in e) or e.startswith("T1_9_n"))
    other = sorted(e for e in entries if not (e.startswith("machine_") or e.startswith("T1_9_")))
    out = {
        "bare_machine_dirs": len(bare),
        "resolution_dirs": len(res),
        "t1_dirs": sum(1 for e in res if e.startswith("T1_9")),
        "other_entries": other,
    }

    man = []
    mpath = batch / "manifest.txt"
    if mpath.exists():
        for line in mpath.read_text().splitlines():
            p = line.split()
            if len(p) >= 3:
                man.append({
                    "tet": p[0], "n": int(p[1]),
                    "outdir": os.path.basename(p[2]),
                    "params": p[3] if len(p) > 3 else None,
                })
    out["manifest_lines"] = len(man)
    out["manifest_unique_machines"] = len({os.path.basename(m["tet"]).replace("_tet.vtk", "") for m in man})

    hist = {}
    hpath = batch / "sample_history.tsv"
    n_hist_lines = 0
    if hpath.exists():
        for line in hpath.read_text().splitlines():
            if line.startswith("#"):
                continue
            n_hist_lines += 1
            parts = line.split("\t")
            if len(parts) >= 4:
                hist[parts[1]] = parts[3]  # chronological -> last status wins
    out["history_lines"] = n_hist_lines
    out["history_unique_runs"] = len(hist)
    out["history_final_status"] = dict(sorted(Counter(hist.values()).items()))

    man_keys = {m["outdir"] for m in man}
    npz_dirs = {e for e in res if (batch / e / "sample.npz").exists()}
    out["npz_on_disk"] = len(npz_dirs)
    out["npz_machine"] = sum(1 for e in npz_dirs if e.startswith("machine_"))
    out["npz_t1"] = sum(1 for e in npz_dirs if e.startswith("T1_9"))

    missing = sorted(man_keys - npz_dirs)
    out["manifest_runs_missing_npz"] = len(missing)
    out["missing_npz"] = [{"run": m, "final_status": hist.get(m)} for m in missing]
    out["manifest_minus_history"] = len(man_keys - set(hist.keys()))
    return out, npz_dirs


def parse_side_files(batch: Path, name: str):
    """Parse blocks.divisions.json + hex_hex_metrics_*.json next to a sample."""
    d = batch / name
    div = parse_json_file(d / "blocks.divisions.json")
    metrics = None
    for f in d.glob("hex_hex_metrics_*.json"):
        metrics = parse_json_file(f)
        break
    result = {}
    if div:
        counts = div.get("counts", {})
        result["divisions_counts"] = {int(k): int(v) for k, v in counts.items()}
        result["divisions_target_h"] = div.get("target_h")
        tbl = div.get("table", [])
        result["divisions_n_classes"] = len(tbl)
        result["divisions_h_eff_median"] = float(np.median([t.get("h_eff", 0) for t in tbl])) if tbl else None
    if metrics:
        prm = metrics.get("Parametrization", {})
        result["hex_hex_metrics"] = {
            "final_energy": prm.get("final_energy"),
            "final_energy_seamless": prm.get("final_energy_seamless"),
            "valid_volume": prm.get("valid_volume"),
            "n_invalid_param_tets": prm.get("n_invalid_param_tets"),
            "n_invalid_valencies": prm.get("n_invalid_valencies"),
            "time_total": metrics.get("time_total"),
        }
    return result


# ---- Main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after N samples")
    args = ap.parse_args()

    batch = args.batch
    recon, npz_dirs = reconcile(batch)

    records, load_failures = [], []
    for name in sorted(npz_dirs):
        p = batch / name / "sample.npz"
        try:
            z = np.load(p, allow_pickle=False)
            data = {k: z[k] for k in z.files}
        except Exception as exc:
            load_failures.append({"name": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        try:
            rec = validate_sample(name, data)
            rec.update(parse_side_files(batch, name))
        except Exception as exc:
            load_failures.append({"name": name, "error": f"validate: {type(exc).__name__}: {exc}"})
            continue
        records.append(rec)
        if args.limit and len(records) >= args.limit:
            break

    n_ok = sum(1 for r in records if r["status"] == "ok")
    n_failed = sum(1 for r in records if r["status"] == "failed")

    # duplicates
    params_groups = defaultdict(list)
    for r in records:
        params_groups[r["params_hash"]].append(r["name"])
    vertex_groups = defaultdict(list)
    for r in records:
        vertex_groups[r["vertex_hash"]].append(r["name"])
    dup_vertices = {h: sorted(v) for h, v in vertex_groups.items() if len(v) > 1}
    dup_params = {h: sorted(v) for h, v in params_groups.items() if len(v) > 1}

    # resolution pairing (machines only)
    machine_res = defaultdict(list)
    for r in records:
        if r["name"].startswith("machine_"):
            machine_res[r["machine"]].append(r)
    all_machine_ids = set(machine_res.keys())
    for m in recon["missing_npz"]:
        base, _ = split_name(m["run"])
        all_machine_ids.add(base)
    pairing = {"machines_with_2_res": 0, "machines_with_1_res": 0, "machines_with_0_res": 0,
               "mismatched_params_pairs": [], "machines": {}}
    for mid in sorted(all_machine_ids):
        rs = machine_res.get(mid, [])
        hashes = {r["params_hash"] for r in rs}
        entry = {
            "n_res": len(rs),
            "res": sorted(r["name"] for r in rs),
            "params_consistent": len(hashes) <= 1,
            "status": [r["status"] for r in rs],
        }
        pairing["machines"][mid] = entry
        if len(rs) == 2:
            pairing["machines_with_2_res"] += 1
            if len(hashes) > 1:
                pairing["mismatched_params_pairs"].append(mid)
        elif len(rs) == 1:
            pairing["machines_with_1_res"] += 1
        else:
            pairing["machines_with_0_res"] += 1

    ok_records = [r for r in records if r["status"] == "ok"]

    def _dist(field):
        vals = [r[field] for r in ok_records if r.get(field) is not None]
        if not vals:
            return None
        return {"min": int(min(vals)), "median": float(np.median(vals)),
                "max": int(max(vals)), "mean": float(np.mean(vals))}

    dir_class_hist = Counter()
    for r in ok_records:
        h = r.get("dir_class_hist")
        if h:
            for c, n in enumerate(h):
                dir_class_hist[c] += n

    div_hist = Counter()
    for r in ok_records:
        dc = r.get("divisions_counts")
        if dc:
            div_hist.update(dc)

    n_dir_classes_dist = Counter(r.get("n_dir_classes") for r in ok_records if r.get("n_dir_classes") is not None)

    dataset = {
        "n_records": len(records),
        "n_ok": n_ok,
        "n_failed": n_failed,
        "n_load_failures": len(load_failures),
        "distributions": {
            "n_blocks": _dist("n_blocks"),
            "n_vertices": _dist("n_vertices"),
            "n_edges": _dist("n_edges"),
            "n_quad_faces": _dist("n_quad_faces"),
            "n_surface_points": _dist("n_surface_points"),
            "n_surface_tris": _dist("n_surface_tris"),
            "n_interior_faces": _dist("n_interior_faces"),
        },
        "dir_class_histogram": {str(k): int(v) for k, v in sorted(dir_class_hist.items())},
        "n_dir_classes_distribution": {str(k): int(v) for k, v in sorted(n_dir_classes_dist.items())},
        "divisions_histogram": {str(k): int(v) for k, v in sorted(div_hist.items())},
        "n_distinct_params_hashes": len(params_groups),
        "n_params_duplicate_groups": len(dup_params),
        "n_vertex_duplicate_groups": len(dup_vertices),
        "n_distinct_machine_geometries": len({r["machine"] for r in ok_records if r["name"].startswith("machine_")}),
    }

    bv_min = [r["block_volume"]["min"] for r in ok_records if r.get("block_volume")]
    bv_med = [r["block_volume"]["median"] for r in ok_records if r.get("block_volume")]
    min_vol_ratio = [r["min_volume_ratio"] for r in ok_records if r.get("min_volume_ratio") is not None]
    fit_p95 = [r["quality"]["fit_residual_p95"] for r in ok_records
               if r.get("quality") and r["quality"].get("fit_residual_p95") is not None]
    degenerate_edges = [r["quality"]["degenerate_edges"] for r in ok_records
                        if r.get("quality") and r["quality"].get("degenerate_edges") is not None]
    dataset["distributions"]["block_volume_min"] = {
        "min": float(min(bv_min)), "median": float(np.median(bv_min)), "max": float(max(bv_min)),
    } if bv_min else None
    dataset["distributions"]["block_volume_median"] = {
        "min": float(min(bv_med)), "median": float(np.median(bv_med)), "max": float(max(bv_med)),
    } if bv_med else None
    dataset["distributions"]["min_volume_ratio"] = {
        "min": float(min(min_vol_ratio)), "median": float(np.median(min_vol_ratio)), "max": float(max(min_vol_ratio)),
    } if min_vol_ratio else None
    dataset["distributions"]["fit_residual_p95"] = {
        "min": float(min(fit_p95)), "median": float(np.median(fit_p95)), "max": float(max(fit_p95)),
    } if fit_p95 else None
    dataset["distributions"]["degenerate_edges"] = {
        "min": int(min(degenerate_edges)), "median": float(np.median(degenerate_edges)), "max": int(max(degenerate_edges)),
        "n_samples_nonzero": int(sum(1 for x in degenerate_edges if x > 0)),
    } if degenerate_edges else None
    dataset["n_samples_with_degenerate_blocks"] = int(
        sum(1 for r in records if r.get("n_inverted_blocks", 0) + r.get("n_collapsed_blocks", 0) > 0))

    summary = {
        "meta": {
            "generated_by": "scripts/validate_3d_dataset.py",
            "batch_root": str(batch),
            "schema_reference": "data/hex3d_algohex/smoke2000/sample.npz",
            "expected_keys": sorted(EXPECTED_KEYS),
        },
        "reconciliation": recon,
        "load_failures": load_failures,
        "dataset": dataset,
        "duplicates": {
            "params_hash_duplicate_groups": dup_params,
            "vertex_hash_duplicate_groups": dup_vertices,
        },
        "resolution_pairing": pairing,
        "per_sample": records,
        "exclusion_list": (
            [{"name": r["name"], "reasons": r["reasons"]} for r in records if r["status"] == "failed"]
            + [{"name": f["name"], "reasons": [f["error"]]} for f in load_failures]
        ),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(summary, fh, indent=1, default=str)

    print("=" * 72)
    print("3D hex-block dataset validation")
    print("=" * 72)
    print(f"batch root      : {batch}")
    print(f"on-disk npz     : {recon['npz_on_disk']}  (machine={recon['npz_machine']}, T1_9={recon['npz_t1']})")
    print(f"loaded          : {len(records)}")
    print(f"ok              : {n_ok}")
    print(f"failed          : {n_failed}")
    print(f"load failures   : {len(load_failures)}")
    print(f"distinct params hashes : {len(params_groups)}  (params-dup groups: {len(dup_params)})")
    print(f"vertex-dup groups      : {len(dup_vertices)}")
    print(f"samples w/ degenerate blk: {dataset['n_samples_with_degenerate_blocks']}")
    print("-" * 72)
    print("size distributions (ok samples) [min / median / max]:")
    for k, v in dataset["distributions"].items():
        if v and isinstance(v, dict) and "median" in v:
            print(f"  {k:22s}: {v['min']:>10g} / {v['median']:>10.3g} / {v['max']:>10g}")
    print("-" * 72)
    print("n_dir_classes distribution:", dataset["n_dir_classes_distribution"])
    print("dir_class histogram:", dataset["dir_class_histogram"])
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
