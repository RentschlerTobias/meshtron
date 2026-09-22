#!/usr/bin/env python3
"""test_feature_model.py — Tests fuer das Kurvenmodell (FeatureModel v2).

(a) analytischer Wuerfel, 6 Labels -> exakt 12 Seam-Kurven + Endpunkte
(b) reales npz: GT -> edge_ctrl -> curved_refill, watertight, inverted <= chord
(c) Frames: Planar-Residual < 1e-9, hub/shroud radial, Innenraum-Vorzeichen
(d) Snap-Injektivitaet (v28/v38-Kollision) + GT-Idempotenz
(e) Format-Konformitaet gegen den export_sample._edge_records-Vertrag

Keine GPU. Plain asserts, exit 0/1 wie die uebrigen Tests.
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from block_mapping import snap_corners_v2, tier_counts  # noqa: E402
from curved_bridge import chord_baseline, refill_curved  # noqa: E402
from edge_curves import build_structures, emit_edge_records  # noqa: E402
from geometry_features import FeatureModelV2, plane_fit_residual  # noqa: E402

NPZ = os.path.join(ROOT, "data", "hex3d_algohex", "batch",
                   "machine_0034_n2000", "sample.npz")
CACHE = os.path.join(ROOT, "data", "features")


def _fm() -> FeatureModelV2:
    return FeatureModelV2(NPZ, cache_dir=CACHE)


def _cube_surface():
    C = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    tris, labels = [], []
    for li, (a, b, c, d) in enumerate(faces):
        tris += [[a, b, c], [a, c, d]]
        labels += [li, li]
    return C, np.array(tris, np.int64), np.array(labels, np.int64)


def test_cube_has_twelve_seam_curves() -> None:
    from curve_model import extract_seam_curves
    C, tris, labels = _cube_surface()
    cs = extract_seam_curves(C, tris, labels)
    assert cs.n_curves == 12, cs.n_curves
    assert len(cs.ep_pt) == 24, len(cs.ep_pt)
    assert all(len(cs.segment(c)) == 2 for c in range(12))
    junctions = {tuple(p) for p in cs.ep_pt}
    assert len(junctions) == 8, junctions


def test_real_npz_curved_roundtrip_beats_chord() -> None:
    fm = _fm()
    C = fm.vertices[fm.blocks].astype(float)
    Csnap, rec = snap_corners_v2(fm, C)
    assert np.allclose(Csnap, C, atol=1e-12)
    assert tier_counts(rec)["vertex"] == C.size // 3
    out = "/tmp/opencode/_test_curved.vtk"
    rc = refill_curved(Csnap, 0.08, out, fm=fm)
    rb = chord_baseline(Csnap, 0.08, "/tmp/opencode/_test_chord.vtk")
    assert rc["watertight"], rc
    assert rb["watertight"], rb
    assert rc["inverted_curved"] <= rb["inverted_chord"], (rc["inverted_curved"],
                                                           rb["inverted_chord"])
    assert rc["cells_after"] == rb["cells_after"]


def test_frame_planar_fit_and_radial_and_interior() -> None:
    t = np.linspace(0, 2 * np.pi, 40)
    planar = np.stack([np.cos(t), np.sin(t), np.full_like(t, 0.3)], axis=1)
    assert plane_fit_residual(planar) < 1e-9
    fm = _fm()
    fr = fm.frames
    radial_ids = [c for c in range(fm.edge_curves.n_curves)
                  if fr.radial[c] and len(fm.edge_curves.segment(c)) >= 3]
    assert radial_ids, "keine radialen Kantenkurven gefunden"
    c = radial_ids[0]
    lo = int(fm.edge_curves.offset[c])
    Q = fm.edge_curves.segment(c)
    tmid = float(fm.edge_curves.arclen[lo + len(Q) // 2])
    p = Q[len(Q) // 2]
    T, N, B = fr.frame(c, tmid, p)
    for v in (T, N, B):
        assert abs(np.linalg.norm(v) - 1) < 1e-9
    assert abs(T @ N) < 1e-9 and abs(T @ B) < 1e-9 and abs(N @ B) < 1e-9
    radial = np.array([p[0], p[1], 0.0])
    radial /= np.linalg.norm(radial)
    assert abs(N @ radial) > 0.999, (N, radial)
    assert abs(N[2]) < 0.05, N
    assert (fm.interior_ref - p) @ N > 0
    for c2 in range(0, fm.edge_curves.n_curves, 7):
        if len(fm.edge_curves.segment(c2)) < 2:
            continue
        lo2 = int(fm.edge_curves.offset[c2])
        p2 = fm.edge_curves.segment(c2)[0]
        t2 = float(fm.edge_curves.arclen[lo2])
        _T, N2, _B = fr.frame(c2, t2, p2)
        assert (fm.interior_ref - p2) @ N2 > 0


def test_snap_v2_injective_and_idempotent() -> None:
    fm = _fm()
    C = fm.vertices[fm.blocks]
    Cs, rec = snap_corners_v2(fm, C)
    assert np.allclose(Cs, C, atol=1e-12)
    assert max(r["dist"] for r in rec) < 1e-12
    assert all(r["tier"] == "vertex" for r in rec)
    v28 = np.array([0.5811, 0.1352, 1.4866])
    v38 = np.array([0.5884, 0.0987, 1.4357])
    Cs2, rec2 = snap_corners_v2(fm, np.stack([v28, v38])[None])
    assert rec2[1]["tier"] == "vertex"
    assert rec2[0]["tier"] in ("edge", "surface")
    assert not np.allclose(Cs2[0, 0], Cs2[0, 1])
    assert rec2[0]["curve_id"] >= -1 and rec2[1]["curve_id"] >= 0


def test_format_conformance_vs_edge_records() -> None:
    fm = _fm()
    counts = {0: 2}
    cof = {(r, ax): 0 for r in range(len(fm.blocks)) for ax in (0, 1, 2)}
    st = build_structures(fm, fm.blocks, fm.vertices[fm.blocks], fm.blocks,
                          counts, cof)
    out = emit_edge_records(fm, fm.blocks, fm.vertices[fm.blocks], fm.blocks, st)
    E = len(st.edge_curve)
    assert out["edges"].shape == (2 * E, 2)
    assert out["edge_ctrl"].shape == (2 * E, 2, 3)
    assert len(out["dir_class"]) == 2 * E
    assert int(out["edge_polyline_offset"][-1]) == len(out["edge_polyline"])
    for i in range(0, 2 * E, 2):
        a, b = out["edge_ctrl"][i], out["edge_ctrl"][i + 1]
        assert np.allclose(a[0], b[1]) and np.allclose(a[1], b[0])
        assert np.array_equal(out["edges"][i][::-1], out["edges"][i + 1])
    import export_sample as es
    import block_edges as be
    for i in range(0, min(6, 2 * E), 2):
        j0 = int(out["edge_polyline_offset"][i])
        j1 = int(out["edge_polyline_offset"][i + 1])
        pts = out["edge_polyline"][j0:j1]
        B1, B2 = out["edge_ctrl"][i]
        curve = be.bezier_points(pts[0], B1, B2, pts[-1], 200)
        chord = float(np.linalg.norm(pts[-1] - pts[0]))
        assert be.max_dist_to_curve(pts, curve) / chord <= 0.05
    s = es.load(NPZ)
    for i in range(0, min(4, len(s["edges"])), 2):
        pts = s["edge_polyline"][s["edge_polyline_offset"][i]:
                                 s["edge_polyline_offset"][i + 1]]
        B1, B2 = s["edge_ctrl"][i]
        curve = be.bezier_points(pts[0], B1, B2, pts[-1], 200)
        chord = float(np.linalg.norm(pts[-1] - pts[0]))
        assert be.max_dist_to_curve(pts, curve) / chord <= 0.05


if __name__ == "__main__":
    fns = (test_cube_has_twelve_seam_curves,
           test_real_npz_curved_roundtrip_beats_chord,
           test_frame_planar_fit_and_radial_and_interior,
           test_snap_v2_injective_and_idempotent,
           test_format_conformance_vs_edge_records)
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print("ALL PASS")
