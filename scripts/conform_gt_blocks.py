"""conform_gt_blocks.py -- plan v3: conform the GT coarse block structure onto the
npz geometry (machine_0034_n2000 or any npz path).

Goal (verbatim from user): "Wir wollen die Blockstruktur auf unsere npz-Geometrie
uebertragen. Mehr nicht."

Reuses the proven production path of scripts/map_generated_blocks.py
(--curve-target seam) but feeds it the GT coarse blocks instead of generated ones:

  fm = FeatureModelV2(npz)                     # cache data/features (format 2)
  target = SimpleNamespace(curves=seam, surface_nearest=...)   # SEAM-ONLY shim
  C = fm.vertices[fm.blocks]                   # (nb,8,3) VTK hex corner order
  C_snap, records = snap_corners_v2(target, C, SnapConfigV2())
  path_fn = _seam_path_fn(seam, records, stats)
  rep = refill_curved(C_snap, target_h=10.0, ..., fm=target, path_fn=path_fn)

CRITICAL constraints (deep research, plan v3):
  - NEVER pass the full fm as snap/refill target: _gt_edge_curve would return
    block-complex edge polylines up to 0.454 off the npz surface. The seam-only
    shim routes through fm.seam_curves instead (boundary dist 5.5e-8 measured).
  - No analytic snap constants (plan v3 amendment A1; Momus ranking: highest
    silent-failure risk).
  - target_h=10.0 (>> max GT edge 1.41) => 1x1x1 lattice => 12 coarse cells.

Artifacts (data/features_debug/<stem>_gt_conform_*):
  conform_refill.vtk   12 curved hexes (Coons+Gordon-Hall TFI)
  conform_edges.vtk    curved block-edge polylines (84 unique welded)
  conform_compare.vtk  part 1=GT corners, 2=snapped corners
  conform_summary.json numeric tripwires

Tripwires (plan v3 amendment A2, fail with exit 2 on violation):
  - max boundary->npz-surface distance <= 1e-3
  - watertight
  - seam routes reported (informational; 0 => path_fn produced no routes)

Known pre-existing dataset artifact (scripts/map_generated_blocks.py:241-244):
  GT coarse blocks contain folded cells (min det J < 0, documented for block
  [4,7] on this geometry); inverted cells are reported but NOT a tripwire.

Usage:
  python scripts/conform_gt_blocks.py [--npz PATH] [--out-dir DIR]
                                      [--geodesic] [--surface-project]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from meshtron.geometry.block_mapping import SnapConfigV2, snap_corners_v2  # noqa: E402
from meshtron.geometry.conform import (ConformOptions,  # noqa: E402
                                       _boundary_edge_pred, _read_vtk_mesh,
                                       collapsed_faces, conform_blocks)
from meshtron.geometry.geometry_features import FeatureModelV2  # noqa: E402
from scripts.compare_viz import _write_parts_vtk  # noqa: E402

DEFAULT_NPZ = os.path.join(ROOT, "data", "hex3d_algohex", "batch",
                           "machine_0034_n2000", "sample.npz")
DEFAULT_OUT = os.path.join(ROOT, "data", "features_debug")
BOUNDARY_TOL = 1e-3
WALK_ARC_MULT = 1.5      # walked path may exceed the chord by this factor
WALK_ID_BASE = 910000    # synthetic curve_id for multi-patch walked edges



def main() -> int:
    ap = argparse.ArgumentParser(
        description="conform GT coarse block structure onto npz geometry")
    ap.add_argument("--npz", default=DEFAULT_NPZ)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--target-h", type=float, default=10.0,
                    help=">> max GT edge length => 1x1x1 lattice (coarse)")
    ap.add_argument("--feature-cache", default=os.path.join(ROOT, "data", "features"))
    ap.add_argument("--geodesic", action="store_true",
                    help="route non-seam block edges as shortest paths ON "
                         "their associated npz patch (plan v4, D4). A patch "
                         "path cannot cross the blade footprint, so edges run "
                         "AROUND the blade; no arc/chord guard. Writes "
                         "<prefix>_geodesic_edges.vtk + _routing.json")
    ap.add_argument("--reject-interior-faces", action="store_true",
                    help="try to reject single-owner faces that lie inside the "
                         "domain (block-level T-junctions). OFF by default: by "
                         "the time the test runs, the edges of such a wall have "
                         "already been routed onto the geometry, so it neither "
                         "catches machine_0387_n8000's two T-junctions nor "
                         "avoids false positives on clean samples -- and a "
                         "wrongly rejected face is neither projected nor "
                         "measured, which puts a hole in the conformity number. "
                         "Use scripts/detect_block_tjunctions.py on blocks.vtk "
                         "to gate the dataset instead.")
    ap.add_argument("--no-seam-snap", action="store_true",
                    help="keep the seam navigator's raw path instead of "
                         "snapping it back onto the seam polyline and the npz "
                         "surface (measured 4.9e-03 off both).")
    ap.add_argument("--allow-collapsed", action="store_true",
                    help="attempt samples with collapsed block faces; they "
                         "normally fail in refill_curved with "
                         "'keine D4-Orientierung'.")
    ap.add_argument("--seam-tol", type=float, default=1e-9,
                    help="a corner without a seam record in the snap may only "
                         "be routed along a seam when it lies this close to "
                         "one. The legacy 0.12 fallback started the arc up to "
                         "0.078 away from the block corner, which distorts the "
                         "Coons boundary row; with the geodesic router those "
                         "edges belong on a patch instead.")
    ap.add_argument("--project-faces", action="store_true",
                    help="pull the interior of every domain boundary face onto "
                         "its npz patch before Gordon-Hall (plan v4 S2). Coons "
                         "faces only interpolate their four edges, so on a "
                         "curved patch the interior cuts the chord.")
    ap.add_argument("--blade-clearance", type=float, default=0.0,
                    help="keep geodesic edges this far away from the blade "
                         "patch (label 7). 0 = plain shortest path, which "
                         "hugs the blade root exactly as the GT edges do.")
    ap.add_argument("--clearance-chord-frac", type=float, default=0.25,
                    help="cap the effective blade clearance of an edge at this "
                         "fraction of its own chord; a short edge cannot bow a "
                         "full clearance away without creasing.")
    ap.add_argument("--blend-boundary", type=int, default=0, metavar="N",
                    help="pull routed boundary edges towards the shape of "
                         "their parallel rails, N passes. A shortest path is "
                         "the wrong shape for a block edge; four passes cut "
                         "inverted cells by a third to two thirds with the "
                         "boundary error unchanged. Only edges belonging "
                         "unambiguously to one patch are touched.")
    ap.add_argument("--blend-interior", action="store_true",
                    help="give interior chord edges a shape blended from the "
                         "parallel rails of their direction class (plan v4). "
                         "Boundary edges untouched -- their patch already "
                         "determines them.")
    ap.add_argument("--surface-project", action="store_true",
                    help="project non-seam block edges lying on a single npz "
                         "patch onto that patch; edges crossing onto another "
                         "patch mid-way are walked along the separating seam "
                         "curve (guarded; default off)")
    args = ap.parse_args()

    base = os.path.splitext(os.path.basename(args.npz))[0]
    stem = (os.path.basename(os.path.dirname(os.path.abspath(args.npz)))
            if base == "sample" else base)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, f"{stem}_gt_conform")

    fm = FeatureModelV2(args.npz, cache_dir=args.feature_cache)
    seam = fm.seam_curves
    # Seam-only shim (plan v3): full fm would bypass seams via _gt_edge_curve.
    target = types.SimpleNamespace(curves=seam, surface_nearest=fm.surface_nearest)

    C = fm.vertices[fm.blocks].astype(np.float64)  # (nb, 8, 3) VTK hex order
    nb = int(C.shape[0])

    C_snap, records = snap_corners_v2(target, C, SnapConfigV2())
    from meshtron.geometry import curved_bridge  # noqa: F401  (inserts the hex3d path for tfi)
    curved_bridge._load()
    n_coll = collapsed_faces(fm.blocks, C_snap)
    if n_coll and not args.allow_collapsed:
        print(f"skip {stem}: {n_coll} collapsed block faces (degenerate block; "
              f"clean_base_npz.py drops these samples too). "
              f"Use --allow-collapsed to try anyway.", file=sys.stderr)
        return 3
    tier = {}
    for rec in records:
        tier[rec["tier"]] = tier.get(rec["tier"], 0) + 1

    opt = ConformOptions(
        target_h=args.target_h, geodesic=args.geodesic,
        project_faces=args.project_faces, blend_boundary=args.blend_boundary,
        seam_tol=args.seam_tol, no_seam_snap=args.no_seam_snap,
        blade_clearance=args.blade_clearance,
        clearance_chord_frac=args.clearance_chord_frac,
        blend_interior=args.blend_interior,
        surface_project=args.surface_project,
        reject_interior_faces=args.reject_interior_faces)
    out = conform_blocks(fm, fm.blocks, C_snap, records, prefix, opt, stem=stem)
    curved_path = out["out_vtk"]
    rep = {k: v for k, v in out.items()
           if k not in ("boundary", "route_stats", "routing_debug", "kink_deg",
                        "out_vtk", "out_boundary_vtk", "out_edges_debug_vtk")}
    route_stats = out["route_stats"]
    bnd_stats = out["boundary"]
    max_bnd = bnd_stats["max"]
    compare_path = prefix + "_compare.vtk"
    gt_blocks = [[int(j) for j in b] for b in fm.blocks]
    snapped_v = np.array(fm.vertices, dtype=np.float64)
    snapped_v[fm.blocks] = C_snap  # welded vertex j gets corner C_snap[r,c]
    _write_parts_vtk(compare_path,
                     [(fm.vertices, gt_blocks, 1, 12),
                      (snapped_v, gt_blocks, 2, 12)],
                     f"meshtron {stem} GT conform (1=GT corners, 2=snapped)")

    routing_debug = out["routing_debug"]
    kink_all = out["kink_deg"]
    if routing_debug:
        with open(prefix + "_routing.json", "w") as fh:
            json.dump(routing_debug, fh, indent=2)
    snap_moves = np.linalg.norm(C_snap.reshape(-1, 3) - C.reshape(-1, 3), axis=1)
    summary = {
        "npz": args.npz,
        "n_blocks": nb,
        "snap": {"max_move": float(snap_moves.max()),
                 "mean_move": float(snap_moves.mean()),
                 "tier_counts": {str(k): v for k, v in sorted(tier.items())}},
        "seam_routes": route_stats["routes"],
        "surface_projected": {
            "enabled": bool(args.surface_project),
            "edges": int(route_stats["edges_surface_projected"])},
        "multi_patch_walk": {
            "enabled": bool(args.surface_project),
            "edges": int(route_stats["edges_walked_multi_patch"])},
        "single_owner_faces_rejected": int(
            rep.get("single_owner_faces_rejected", 0)),
        "project_faces": {
            "enabled": bool(args.project_faces),
            "faces": int(route_stats.get("faces_projected", 0)),
            "by_label": route_stats.get("faces_projected_labels", {})},
        "kink": {
            "max_deg": float(max(kink_all, default=0.0)),
            "p90_deg": float(np.percentile(kink_all, 90)) if kink_all else 0.0,
            "edges_over_20deg": int(sum(1 for k in kink_all if k > 20.0))},
        "blend_boundary": {
            "passes": int(args.blend_boundary),
            "edges_touched": int(route_stats.get("edges_blend_boundary", 0))},
        "blend_interior": {
            "enabled": bool(args.blend_interior),
            "edges": int(route_stats.get("edges_blended", 0))},
        # Edges routed ON the blade patch itself sit at distance 0 by
        # definition; they are excluded from the clearance statistic.
        "blade_clearance": {
            "value": float(args.blade_clearance),
            "min_median_off_blade_edges": float(min(
                [d["blade_dist_med"] for d in routing_debug
                 if d.get("blade_dist_med") is not None
                 and d.get("chosen") not in (7,)], default=-1.0))},
        "geodesic": {
            "enabled": bool(args.geodesic),
            "edges": int(route_stats.get("edges_geodesic", 0)),
            "failed": int(sum(1 for d in routing_debug
                              if d["chosen"] is None)),
            "max_arc_over_chord": float(max(
                [d["arc_over_chord"] for d in routing_debug
                 if d.get("arc_over_chord") is not None], default=0.0)),
            "max_dist_patch": float(max(
                [d["max_dist_patch"] for d in routing_debug
                 if d.get("max_dist_patch") is not None], default=0.0))},
        "refill": rep,
        "boundary": bnd_stats,
        "tripwires": {
            "max_boundary_surface_dist": max_bnd,
            "tolerance": BOUNDARY_TOL,
            "pass": bool(max_bnd <= BOUNDARY_TOL and rep["watertight"]),
            "watertight": bool(rep["watertight"]),
        },
        "known_artifact": {
            "inverted_cells_gt_dataset": int(rep["inverted_curved"]),
            "note": "GT coarse blocks contain folded cells (documented, "
                    "scripts/map_generated_blocks.py:241-244); not a tripwire"},
    }
    summary_path = prefix + "_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"blocks={nb} snap_max_move={snap_moves.max():.6f} "
          f"tiers={summary['snap']['tier_counts']} routes={route_stats['routes']} "
          f"surface_projected={route_stats['edges_surface_projected']} "
          f"walked_multi_patch={route_stats['edges_walked_multi_patch']} "
          f"geodesic={route_stats.get('edges_geodesic', 0)} "
          f"blended={route_stats.get('edges_blended', 0)}")
    print(f"refill: watertight={rep['watertight']} cells={rep['cells_after']} "
          f"inverted={rep['inverted_curved']} (known GT dataset artifact) "
          f"buckets={rep.get('buckets')}")
    print(f"tripwire boundary_dist={max_bnd:.3e} (tol {BOUNDARY_TOL:.0e}) "
          f"pass={summary['tripwires']['pass']}")
    print(f"saved {curved_path}")
    print(f"saved {rep.get('out_edges_vtk')}")
    print(f"saved {compare_path}  (part: 1=GT corners, 2=snapped corners)")
    print(f"saved {summary_path}")
    return 0 if summary["tripwires"]["pass"] else 2




if __name__ == "__main__":
    raise SystemExit(main())
