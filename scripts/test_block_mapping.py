"""test_block_mapping.py — Unit-Tests fuer das Feature-Snapping (keine GPU).

Nutzt machine_0034_n2000 (idx 687, 12 Bloecke, 40 GT-Ecken) als Feature-Modell.
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from meshtron.geometry.block_mapping import FeatureModel, SnapConfig, score_candidate, tier_counts  # noqa: E402

NPZ = os.path.join(ROOT, "data", "hex3d_algohex", "batch",
                   "machine_0034_n2000", "sample.npz")


def _fm() -> FeatureModel:
    return FeatureModel(NPZ)


def _perp_dir_on_blade_edge(fm: FeatureModel, v_idx: int) -> tuple[np.ndarray, int]:
    """Einheitsrichtung senkrecht zur naechsten Blade-Kante an GT-Ecke v_idx."""
    v = fm.vertices[v_idx]
    _, ii = fm.seg_tree.query(v, k=10)
    ii = np.atleast_1d(ii)
    ab = fm.seg_b[ii] - fm.seg_a[ii]
    t = np.clip(((v - fm.seg_a[ii]) * ab).sum(-1)
                / np.maximum((ab * ab).sum(-1), 1e-30), 0.0, 1.0)
    p = fm.seg_a[ii] + t[..., None] * ab
    seg = int(ii[int(np.argmin(np.linalg.norm(v - p, axis=-1)))])
    tang = fm.seg_b[seg] - fm.seg_a[seg]
    tang = tang / np.linalg.norm(tang)
    near = int(np.argmin(np.linalg.norm(fm.surface_points - v, axis=1)))
    tw = fm.surface_tris[(fm.surface_tris == near).any(axis=1)]
    nrm = np.cross(fm.surface_points[tw[:, 1]] - fm.surface_points[tw[:, 0]],
                   fm.surface_points[tw[:, 2]] - fm.surface_points[tw[:, 0]])
    nrm = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
    n = nrm.mean(0)
    n = n / np.linalg.norm(n)
    d = n - (n @ tang) * tang
    if np.linalg.norm(d) < 1e-6:
        d = np.cross(tang, [0.0, 0.0, 1.0])
    return d / np.linalg.norm(d), seg


def _blade_interior_point(fm: FeatureModel) -> tuple[np.ndarray, int]:
    """Segmentmitte einer Blade-Interface-Kante, > 0.08 von jeder GT-Ecke weg."""
    centroids = fm.surface_points[fm.surface_tris].mean(axis=1)
    from scipy.spatial import cKDTree
    stree = cKDTree(centroids)
    for e in range(len(fm.edge_offsets) - 1):
        pts = fm.edge_polyline[fm.edge_offsets[e]:fm.edge_offsets[e + 1]]
        _, idx = stree.query(pts)
        labs = fm.surface_tri_label[idx]
        val, cnt = np.unique(labs, return_counts=True)
        if int(val[np.argmax(cnt)]) not in (5, 6):
            continue
        for k in range(len(pts) - 1):
            m = 0.5 * (pts[k] + pts[k + 1])
            if fm.vertex_tree.query(m)[0] > 0.08:
                return m, e
    raise AssertionError("kein Blade-Innenpunkt gefunden")


def _surface_normal(fm: FeatureModel, q: np.ndarray) -> np.ndarray:
    _, idx = fm.tri_tree.query(q, k=12)
    tw = fm.surface_tris[np.atleast_1d(idx)]
    nrm = np.cross(fm.surface_points[tw[:, 1]] - fm.surface_points[tw[:, 0]],
                   fm.surface_points[tw[:, 2]] - fm.surface_points[tw[:, 0]])
    nrm = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
    n = nrm.mean(0)
    return n / np.linalg.norm(n)


def test_snap_idempotent_on_gt_corners() -> None:
    # Given das GT-Feature-Modell und seine Blockecken
    fm = _fm()
    C = fm.vertices[fm.blocks]
    # When alle Ecken gesnappt werden
    Cs, rec = fm.snap_corners(C)
    # Then bleiben sie exakt liegen und sind Feature-Punkte
    assert np.allclose(Cs, C, atol=1e-12)
    assert max(r["dist"] for r in rec) < 1e-12
    assert all(r["tier"] == "vertex" for r in rec)
    assert tier_counts(rec)["vertex"] == C.size // 3


def test_gt_vertex_displaced_snaps_back_to_feature_point() -> None:
    # Given eine GT-Ecke auf einer Blade-Kante, 0.03 entlang der Flaechennormale
    fm = _fm()
    d, _seg = _perp_dir_on_blade_edge(fm, 0)
    q = (fm.vertices[0] + 0.03 * d).reshape(1, 1, 3)
    # When mit Default-tol_v=0.06 gesnappt wird
    Cs, rec = fm.snap_corners(q)
    # Then gewinnt der Feature-Punkt (Nutzer-Prioritaet) und zieht exakt zurueck
    assert rec[0]["tier"] == "vertex"
    assert abs(rec[0]["dist"] - 0.03) < 1e-6
    assert np.allclose(Cs[0, 0], fm.vertices[0], atol=1e-12)


def test_blade_curve_outranks_surface_when_vertex_tol_is_tight() -> None:
    # Given dieselbe um 0.03 versetzte Ecke, aber tol_v unter dem Versatz
    fm = _fm()
    d, _seg = _perp_dir_on_blade_edge(fm, 0)
    cfg = SnapConfig(tol_v=0.01, tol_e=0.04)
    # When gesnappt wird
    _, rec = fm.snap_corners((fm.vertices[0] + 0.03 * d).reshape(1, 1, 3), cfg)
    # Then greift die Feature-Kurve, nicht die Flaeche
    assert rec[0]["tier"] == "edge"
    assert abs(rec[0]["dist"] - 0.03) < 1e-6
    # And 0.05 liegt ausserhalb tol_e -> Flaeche
    _, rec2 = fm.snap_corners((fm.vertices[0] + 0.05 * d).reshape(1, 1, 3), cfg)
    assert rec2[0]["tier"] == "surface"


def test_blade_curve_interior_prefers_edge_over_surface() -> None:
    # Given ein Punkt im Inneren einer Blade-Kante, weit weg von GT-Ecken
    fm = _fm()
    m, _e = _blade_interior_point(fm)
    N = _surface_normal(fm, m)
    # When 0.03/0.05 entlang der Flaechennormale versetzt und gesnappt wird
    _, r03 = fm.snap_corners((m + 0.03 * N).reshape(1, 1, 3))
    _, r05 = fm.snap_corners((m + 0.05 * N).reshape(1, 1, 3))
    # Then bevorzugt 0.03 die Kurve, 0.05 faellt auf die Flaeche
    assert r03[0]["tier"] == "edge"
    assert abs(r03[0]["dist"] - 0.03) < 1e-4
    assert r05[0]["tier"] == "surface"


def test_score_ordering_sanity() -> None:
    # Given GT-Ecken und eine um 0.03 verschobene Kopie
    fm = _fm()
    C = fm.vertices[fm.blocks]
    C_shift = C.copy()
    C_shift[:, :, 2] += 0.03
    # When beide gesnappt und bewertet werden
    Cs0, rec0 = fm.snap_corners(C)
    Cs1, rec1 = fm.snap_corners(C_shift)
    mean0, _ = score_candidate(Cs0, rec0)
    mean1, _ = score_candidate(Cs1, rec1)
    # Then ist der Versatz strikt schlechter (groesserer mean snap dist)
    assert mean0 == 0.0
    assert mean1 > mean0


def test_vertex_collision_demotes_loser() -> None:
    # Given zwei verschiedene Rohpunkte (idx-687 v28/v38) an derselben GT-Ecke
    fm = _fm()
    v28 = np.array([0.5811, 0.1352, 1.4866])
    v38 = np.array([0.5884, 0.0987, 1.4357])
    C = np.stack([v28, v38])[None]
    # When gesnappt wird
    Cs, rec = fm.snap_corners(C)
    # Then gewinnt der naehere (v38, Index 1) den Feature-Punkt, der andere
    # steigt auf edge/surface ab und die Ziele fallen nicht zusammen
    assert rec[1]["tier"] == "vertex" and rec[1]["feature_id"] == 39
    assert rec[0]["tier"] in ("edge", "surface")
    assert not np.allclose(Cs[0, 0], Cs[0, 1])


def test_shared_raw_coordinate_gets_identical_record() -> None:
    # Given dieselbe Rohkoordinate als Ecke in zwei verschiedenen Bloecken
    fm = _fm()
    shared = fm.vertices[0].copy()
    far = np.array([10.0, 10.0, 10.0])
    C = np.full((2, 8, 3), far)
    C[0, 0] = shared
    C[1, 5] = shared
    # When gesnappt wird
    _, rec = fm.snap_corners(C)
    a, b = rec[0], rec[13]
    # Then sind tier/feature_id/dist/target identisch (Scatter-Konsistenz)
    assert a["tier"] == b["tier"] == "vertex"
    assert a["feature_id"] == b["feature_id"]
    assert a["dist"] == b["dist"]
    assert a["target"] == b["target"]


if __name__ == "__main__":
    for fn in (test_snap_idempotent_on_gt_corners,
               test_gt_vertex_displaced_snaps_back_to_feature_point,
               test_blade_curve_outranks_surface_when_vertex_tol_is_tight,
               test_blade_curve_interior_prefers_edge_over_surface,
               test_score_ordering_sanity,
               test_vertex_collision_demotes_loser,
               test_shared_raw_coordinate_gets_identical_record):
        fn()
        print(f"PASS {fn.__name__}")
