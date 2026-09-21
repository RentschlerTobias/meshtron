"""tfi_bridge.py — gesnappte Blockecken -> CFD-Volumengitter via TFI-Bruecke.

Import-only: das externe hex3d_algohex-Repo wird per sys.path eingebunden und
NIE editiert. Aufrufsequenz exakt wie in dessen tests/test_refill.py:27-52:

    weld -> build_topology -> lat (1x1x1-Lattice) -> direction_classes
         -> solve_block_divisions(target_h) -> refill_complex -> check_watertight
         -> export_vtk.write_vtk

Refill-Modus v1 = "chord": das Lattice traegt nur die 8 Blockecken, also werden
die sechs Randflaechen als gerade Sehnen resampled und der Innenraum per
Gordon-Hall gefuellt. Krumme Feature-Kanten (edge_ctrl im sample.npz) sind der
Ausbaupfad; v1 dokumentiert das bewusst als Naherung.

Modi:
  "conforming"           direction_classes hat Bloecke ueber gemeinsame Flaechen
                         gekoppelt (Face-Sharing der generierten Struktur).
  "independent_fallback" direction_classes/solve warf oder lieferte keine
                         Kopplung -> jede Blockachse eigene Klasse, keine
                         Konformitaet erzwungen; weld+refill laeuft trotzdem.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HEX3D_REPO = ("/home/t1dde/hydrostack_pipeline/stack/domain_partition_3D/"
              "experimentell/hex3d_algohex")


def _load_bridge():
    if HEX3D_REPO not in sys.path:
        sys.path.insert(0, HEX3D_REPO)
    import base_complex as bc
    import export_vtk as ev
    import tfi
    return tfi, ev, bc


def _lattice_vert(row: np.ndarray, corner: tuple) -> np.ndarray:
    """(8,) VTK-Hex-Indexzeile -> (2,2,2) Lattice in tfi.CORNER-Konvention."""
    v = np.empty((2, 2, 2), dtype=np.int64)
    for li, (di, dj, dk) in enumerate(corner):
        v[di, dj, dk] = row[li]
    return v


def _independent_classes(lat: dict) -> list[list[tuple[int, int]]]:
    return [[(r, ax)] for r in sorted(lat) for ax in (0, 1, 2)]


def refill_cfd(corners: np.ndarray, target_h: float, out_vtk: str) -> dict:
    """(nb,8,3) gesnappte Ecken -> CFD-VTK. Liefert Report-dict."""
    tfi, ev, bc = _load_bridge()
    C = np.asarray(corners, dtype=np.float64)
    if C.ndim != 3 or C.shape[1:] != (8, 3):
        raise ValueError(f"corners shape {C.shape} != (nb,8,3)")
    nb = C.shape[0]
    # Geteilte Punktwolke: identische Koordinaten -> identischer Index (weld).
    P, remap = tfi.weld(C.reshape(-1, 3))
    H = remap.reshape(nb, 8)
    B = np.arange(nb, dtype=np.int64)
    f2h, _e2h = bc.build_topology(H)
    lat = {r: (np.ones(3, dtype=int), _lattice_vert(H[r], tfi.CORNER))
           for r in range(nb)}

    report: dict = {"mode": "conforming", "chord_mode": True,
                    "n_blocks": nb, "n_points_welded": int(len(P)),
                    "target_h": float(target_h), "fallback_reason": None}
    try:
        classes = tfi.direction_classes(lat, f2h, H, B, verbose=False)
        linked = any(len(c) > 1 for c in classes)
        counts = tfi.solve_block_divisions(lat, classes, P, target_h,
                                           verbose=False)
        if not linked:
            report["mode"] = "independent_fallback"
            report["fallback_reason"] = "no face-sharing links between blocks"
    except Exception as exc:  # MILP/Topologie kann auf nicht-konformen Bloecken scheitern
        classes = _independent_classes(lat)
        counts = tfi.solve_block_divisions(lat, classes, P, target_h,
                                           verbose=False)
        report["mode"] = "independent_fallback"
        report["fallback_reason"] = f"{type(exc).__name__}: {exc}"

    report["n_classes"] = len(classes)
    pts, Hn, Bn, rep = tfi.refill_complex(P, H, B, f2h, lat, classes, counts,
                                          verbose=False)
    report.update(rep)
    ok, bnd = tfi.check_watertight(pts, Hn, verbose=False)
    report["watertight"] = bool(ok)
    report["boundary_faces"] = int(bnd)
    ev.write_vtk(out_vtk, pts, Hn, [12] * len(Hn), Bn, "block_id",
                 "meshtron generated->CFD (TFI)")
    report["out_vtk"] = os.path.abspath(out_vtk)
    return report
