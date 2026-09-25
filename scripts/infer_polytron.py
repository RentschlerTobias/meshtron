#!/usr/bin/env python3
"""infer_polytron.py -- geometry in, curvilinear block structure and CFD mesh out.

    uv run python scripts/infer_polytron.py \
        --npz data/hex3d_algohex/batch/machine_0034_n2000/sample.npz --blocks 12

Only the labelled surface of the npz is read -- the blocking stored next to it
is never touched, so this is exactly what a new machine goes through:

  1  geometry    labelled surface                               01_geometry.vtk
  2  cloud       conditioning points, xyz + patch multi-hot     02_cloud.vtk
  3  vertices    VertexModel, k rollouts                        03_vertices.vtk
  4  structure   BlockModel pointers + CurveModel Beziers       04_structure.vtk
  5  mesh        curved TFI, boundary mapped onto the patches   05_mesh.vtk
                 (+ 05_mesh_edges.vtk, the curves the fill used)
                 mesh and npz surface in one file, scalar source  06_mesh_vs_geometry.vtk
  6  report      every number of every stage                    report.json

`--blocks` is an input: the block count conditions all three models and the
geometry does not carry it. `--blocks-sweep 12,16,22` tries several and keeps
the best mesh. Exit 0 mesh written, 2 no rollout could be filled.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from meshtron.data.polytron_blocks import (build_cloud, decode_seq, load_npz,  # noqa: E402
                                           point_labels)
from meshtron.geometry.polytron_tfi import (mesh_candidate, rank_key,  # noqa: E402
                                            read_hex_vtk)
from meshtron.training.polytron_sample import run_chain  # noqa: E402
from meshtron.training.train_polytron import load_stage  # noqa: E402
from scripts.eval_polytron import write_structure_vtk  # noqa: E402


def _write_points(path, P, arrays, title):
    with open(path, "w") as fh:
        fh.write(f"# vtk DataFile Version 2.0\n{title}\nASCII\n"
                 "DATASET UNSTRUCTURED_GRID\n")
        fh.write(f"POINTS {len(P)} double\n")
        for q in P:
            fh.write(f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f}\n")
        fh.write(f"CELLS {len(P)} {2 * len(P)}\n")
        for i in range(len(P)):
            fh.write(f"1 {i}\n")
        fh.write(f"CELL_TYPES {len(P)}\n" + "1\n" * len(P))
        fh.write(f"POINT_DATA {len(P)}\n")
        for name, a in arrays.items():
            fh.write(f"SCALARS {name} int 1\nLOOKUP_TABLE default\n")
            fh.write("".join(f"{int(v)}\n" for v in a))


def _write_surface(path, P, T, L):
    with open(path, "w") as fh:
        fh.write("# vtk DataFile Version 2.0\nlabelled surface\nASCII\n"
                 "DATASET UNSTRUCTURED_GRID\n")
        fh.write(f"POINTS {len(P)} double\n")
        for q in P:
            fh.write(f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f}\n")
        fh.write(f"CELLS {len(T)} {4 * len(T)}\n")
        for t in T:
            fh.write(f"3 {t[0]} {t[1]} {t[2]}\n")
        fh.write(f"CELL_TYPES {len(T)}\n" + "5\n" * len(T))
        fh.write(f"CELL_DATA {len(T)}\nSCALARS patch_label int 1\n"
                 "LOOKUP_TABLE default\n")
        fh.write("".join(f"{int(v)}\n" for v in L))


def _annotate_mesh(path):
    """Rewrite the fill VTK with scaled_jacobian / inverted next to block_id."""
    import meshio
    from meshtron.geometry.curved_bridge import _load
    _tfi, _ev, _bc, cb = _load()
    m = meshio.read(path)
    P, H = read_hex_vtk(path)
    bid = np.concatenate([np.asarray(d).ravel() for d in m.cell_data["block_id"]])
    sj = cb.scaled_jacobians(P, H)
    meshio.write(path, meshio.Mesh(P, [("hexahedron", H)], cell_data={
        "block_id": [bid.astype(np.int32)], "scaled_jacobian": [sj],
        "inverted": [(sj <= 0).astype(np.int32)]}), binary=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--npz", required=True)
    ap.add_argument("--blocks", type=int, default=None)
    ap.add_argument("--blocks-sweep", default=None)
    ap.add_argument("--prefix", default=os.path.join(ROOT, "data", "polytron_final"))
    ap.add_argument("--which", default="last", choices=("last", "best"))
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--beam", type=int, default=8,
                    help="vertex beam width (0 = greedy + sampling)")
    ap.add_argument("--target-h", type=float, default=0.08)
    ap.add_argument("--n-points", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-project", action="store_true")
    ap.add_argument("--out-dir", default=None)
    for st in ("vertex", "block", "curve"):
        ap.add_argument(f"--{st}-ckpt", default=None,
                        help=f"override the {st} checkpoint of --prefix")
    args = ap.parse_args()
    counts = ([int(x) for x in args.blocks_sweep.split(",")] if args.blocks_sweep
              else [args.blocks])
    if counts == [None]:
        raise SystemExit("--blocks or --blocks-sweep is required")
    stem = os.path.basename(os.path.dirname(os.path.abspath(args.npz)))
    out = args.out_dir or os.path.join(ROOT, "data", f"polytron_infer_{stem}")
    os.makedirs(out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models, spec, meta, specs = {}, None, {}, {}
    for st in ("vertex", "block", "curve"):
        path = getattr(args, f"{st}_ckpt") or f"{args.prefix}_{st}_{args.which}.pt"
        m, spec, ck = load_stage(path, dev)
        models[st] = m
        specs[st] = spec
        meta[st] = {"epoch": ck["epoch"], "val_loss": ck["val_loss"]}

    import time
    timing = {}
    t_start = time.perf_counter()
    raw = load_npz(args.npz)
    SP, ST, SL = raw["surface_points"], raw["surface_tris"], raw["surface_tri_label"]
    _write_surface(os.path.join(out, "01_geometry.vtk"), SP, ST, SL)
    lab = point_labels(len(SP), ST, SL)
    rng = np.random.default_rng(args.seed)
    cloud = build_cloud(SP, lab, args.n_points, specs["vertex"], rng)
    xyz = (cloud[:, :3] + 1) / 2 * (np.asarray(spec.hi) - np.asarray(spec.lo)) \
        + np.asarray(spec.lo)
    _write_points(os.path.join(out, "02_cloud.vtk"), xyz,
                  {"patch_label": np.where(cloud[:, 3:].sum(1) >= 2, 0,
                                           1 + np.argmax(cloud[:, 3:], 1)),
                   "on_weighted_patch": cloud[:, 3 + specs["vertex"].weight_label - 1] > 0,
                   "seam": cloud[:, 3:].sum(1) >= 2}, "conditioning cloud")
    t0 = time.perf_counter()
    pc = {st: torch.as_tensor(build_cloud(SP, lab, args.n_points, sp,
                                          np.random.default_rng(args.seed)))[None].to(dev)
          for st, sp in specs.items()}
    timing["cloud"] = time.perf_counter() - t0

    report = {"npz": os.path.abspath(args.npz), "models": meta,
              "target_h": args.target_h, "k": args.k, "temp": args.temp,
              "candidates": []}
    best = None
    for nb in counts:
        seqs = run_chain(models, pc, nb, spec, k=args.k, temp=args.temp,
                         top_p=args.top_p, seed=args.seed, beam=args.beam,
                         timing=timing)
        for ci, seq in enumerate(seqs):
            if seq is None:
                report["candidates"].append({"blocks": nb, "cand": ci,
                                             "error": "fewer than 8 vertices"})
                continue
            tmp = os.path.join(out, f"_cand_nb{nb}_{ci}.vtk")
            r = mesh_candidate(seq, spec, raw, args.target_h, tmp,
                               project=not args.no_project)
            r.update(blocks=nb, cand=ci)
            timing["tfi_fill"] = timing.get("tfi_fill", 0.0) + r.get("seconds_fill", 0.0)
            timing["metrics_not_pipeline"] = (timing.get("metrics_not_pipeline", 0.0)
                                              + r.get("seconds_metrics", 0.0))
            report["candidates"].append(r)
            t = r.get("tfi") or {}
            s = r.get("surface") or {}
            print(f"nb {nb:2d} cand {ci}: V {len(seq.vq):3d}  "
                  f"nonmanifold {r['structure']['nonmanifold_faces']}  "
                  f"dup {r['structure']['duplicate_blocks']}  "
                  f"tfi {'ok' if t else 'FAIL'}  wt {t.get('watertight')}  "
                  f"inv {t.get('inverted_curved')}/{t.get('cells_after')}  "
                  f"uncov {s.get('uncovered_share', float('nan')):.3f}"
                  + (f"  {r.get('error')}" if r.get("error") else ""), flush=True)
            if best is None or rank_key(r) < rank_key(best[0]):
                best = (r, seq, tmp)
    keep = set() if best is None else {best[2], os.path.splitext(best[2])[0] + "_edges.vtk"}
    for f in os.listdir(out):
        if f.startswith("_cand_") and os.path.join(out, f) not in keep:
            os.remove(os.path.join(out, f))
    if best is None or best[0].get("tfi") is None:
        json.dump(report, open(os.path.join(out, "report.json"), "w"), indent=1,
                  default=float)
        print("no rollout could be filled")
        return 2
    r, seq, tmp = best
    V, B, C = decode_seq(seq, spec, n_edge_pts=32)
    _write_points(os.path.join(out, "03_vertices.vtk"), V,
                  {"vertex_id": np.arange(len(V)),
                   "used": np.isin(np.arange(len(V)), B)}, "generated corners")
    write_structure_vtk(os.path.join(out, "04_structure.vtk"), [(V, B, C, 0)],
                        "generated block structure: straight hex cells + "
                        "Bezier edges (kind 1)")
    os.replace(tmp, os.path.join(out, "05_mesh.vtk"))
    e = os.path.splitext(tmp)[0] + "_edges.vtk"
    if os.path.exists(e):
        os.replace(e, os.path.join(out, "05_mesh_edges.vtk"))
    _annotate_mesh(os.path.join(out, "05_mesh.vtk"))
    from scripts.polytron_compare_vtk import write_compare
    write_compare(os.path.join(out, "05_mesh.vtk"), args.npz,
                  os.path.join(out, "06_mesh_vs_geometry.vtk"))
    timing["total_wall_incl_vtk_io"] = time.perf_counter() - t_start
    pipe = sum(v for k, v in timing.items()
               if k in ("cloud", "vertex_model", "block_model", "curve_model", "tfi_fill"))
    timing["pipeline_cloud_to_mesh"] = pipe
    one = len([c for c in report["candidates"] if "structure" in c]) or 1
    timing["per_candidate_fill"] = timing.get("tfi_fill", 0.0) / one
    report["timing_s"] = timing
    report["chosen"] = {"blocks": r["blocks"], "cand": r["cand"],
                        "structure": r["structure"], "tfi": r["tfi"],
                        "surface": r["surface"]}
    json.dump(report, open(os.path.join(out, "report.json"), "w"), indent=1,
              default=float)
    t, s = r["tfi"], r["surface"]
    print(f"\nchosen nb {r['blocks']} cand {r['cand']}: {t['cells_after']} cells, "
          f"watertight {t['watertight']}, inverted {t['inverted_curved']} "
          f"({t['inverted_curved'] / t['cells_after']:.2%}), on-surface max "
          f"{s['on_surface_max']:.1e}, uncovered {s['uncovered_share']:.3f}")
    print("timing [s]: " + "  ".join(f"{k} {v:.2f}" for k, v in timing.items()))
    print(f"artifacts in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
