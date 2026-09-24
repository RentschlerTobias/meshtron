"""geometry_features.py — Frames + FeatureModel v2 (Kurven + Snapping-Basis).

Provenienz-Befund (code-verifiziert, Report): `edge_polyline`/`edge_ctrl` sind
NICHT Spline-Fitting der Design-System-Triangulierung, sondern aus dem AlgoHex-
Blockkomplex (`export_sample.py:138-179` -> `tfi.load_blocks`/`tfi.lattices` ->
`block_edges.edge_chains`/`fit_edges`). Die Blockkanten-Polylinien bleiben die
geometrie-treuen Kurven fuers Block-Mapping (Endpunkte == GT-Blockecken).

Kurven-Primitive liegen in `curve_model` (LOC-Trennung). Hier: Frame (T aus
dgamma/dt, N radial fuer hub/shroud oder lokaler Plane-Fit, N global zum
Innenraum), FeatureModelV2 (Blockkanten- + Seam-Kurven, Oberflaeche, Cache
data/features/<run>.pt).
"""
from __future__ import annotations

import os

import numpy as np
from scipy.spatial import cKDTree

from meshtron.geometry.curve_model import (AXIS, CurveSet, block_edge_curves, concat,
                         extract_seam_curves)

CACHE_FORMAT = 2
CACHE_KEYS = ("pts", "curve_of", "offset", "arclen", "closed",
              "label_lo", "label_hi", "ep_pt", "ep_curve", "ep_t", "ep_vertex")


def _plane_normal(Q: np.ndarray) -> np.ndarray:
    """Kleinste Singulaerrichtung der zentrierten Kurvenpunkte (Plane-Fit)."""
    if len(Q) < 3:
        return AXIS.copy()
    X = Q - Q.mean(0)
    n = np.linalg.svd(X, full_matrices=False)[2][-1]
    return n / max(np.linalg.norm(n), 1e-30)


def plane_fit_residual(Q: np.ndarray) -> float:
    """Max. Abstand der Kurvenpunkte zur Bestfit-Ebene / Chordlaenge.

    Geschlossene Kurven (Chord ~ 0) werden gegen ihre Bounding-Box-Diagonale
    normiert, sonst explodiert der Quotient."""
    Q = np.asarray(Q, float)
    if len(Q) < 3:
        return 0.0
    chord = float(np.linalg.norm(Q[-1] - Q[0]))
    extent = float(np.linalg.norm(Q.max(0) - Q.min(0)))
    scale = chord if chord > 1e-6 * max(extent, 1e-12) else extent
    if scale <= 1e-12:
        return 0.0
    X = Q - Q.mean(0)
    n = np.linalg.svd(X, full_matrices=False)[2][-1]
    return float(np.abs(X @ n).max() / scale)


class FrameModel:
    """T/N/B je Kurve; N radial (hub/shroud) oder Plane-Fit, global nach innen."""

    def __init__(self, cs: CurveSet, interior_ref: np.ndarray) -> None:
        self.cs = cs
        self.interior = np.asarray(interior_ref, float)
        self.radial = np.zeros(cs.n_curves, bool)
        self.plane_n = np.zeros((cs.n_curves, 3))
        for c in range(cs.n_curves):
            Q = cs.segment(c)
            self.plane_n[c] = _plane_normal(Q)
            if len(Q) >= 3:
                r = np.linalg.norm(Q[:, :2], axis=1)
                self.radial[c] = bool(np.std(r) / max(np.mean(r), 1e-12) < 0.02)
            else:
                self.radial[c] = True

    def frame(self, curve: int, t: float, p: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(T, N, B) am Punkt `p` auf Kurve `curve` bei Bogenlaenge `t`."""
        Q = self.cs.segment(curve)
        s = self.cs.arclen[int(self.cs.offset[curve]):int(self.cs.offset[curve + 1])]
        if len(Q) >= 2:
            i = int(np.clip(np.searchsorted(s, t) - 1, 0, len(Q) - 2))
            T = Q[i + 1] - Q[i]
        else:
            T = np.array([1.0, 0.0, 0.0])
        T = T / max(np.linalg.norm(T), 1e-30)
        if self.radial[curve]:
            N = np.array([p[0], p[1], 0.0])
            if np.linalg.norm(N) < 1e-9:
                N = self.plane_n[curve].copy()
        else:
            N = self.plane_n[curve].copy()
        N = N - (N @ T) * T
        if np.linalg.norm(N) < 1e-9:
            N = np.cross(T, AXIS)
        N = N / max(np.linalg.norm(N), 1e-30)
        if (self.interior - p) @ N < 0:
            N = -N
        B = np.cross(T, N)
        return T, N, B / max(np.linalg.norm(B), 1e-30)


def _run_key(path: str) -> str:
    return os.path.basename(os.path.dirname(os.path.abspath(path))).replace("/", "__")


def cache_path(npz_path: str, cache_dir: str | None) -> str:
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(npz_path))), "features")
    return os.path.join(cache_dir, _run_key(npz_path) + ".pt")


def _cs_to_dict(cs: CurveSet) -> dict:
    return {k: getattr(cs, k) for k in CACHE_KEYS}


class FeatureModelV2:
    """npz -> Blockkanten-Kurven + Seam-Kurven + Frames (+ optionaler Cache)."""

    def __init__(self, npz_path: str | os.PathLike, cache_dir: str | None = None) -> None:
        import torch
        self.path = str(npz_path)
        cpath = cache_path(self.path, cache_dir)
        if os.path.exists(cpath):
            blob = torch.load(cpath, weights_only=False)
            if blob.get("format") == CACHE_FORMAT:
                self._base_from(blob)
                self.edge_curves = CurveSet(**blob["edge_curves"])
                self.seam_curves = CurveSet(**blob["seam_curves"])
                self._post()
                return
        with np.load(self.path, allow_pickle=False) as z:
            self.vertices = np.asarray(z["vertices"], float)
            self.blocks = np.asarray(z["blocks"], np.int64)
            self.edges = np.asarray(z["edges"], np.int64)
            self.edge_polyline = np.asarray(z["edge_polyline"], float)
            self.edge_offset = np.asarray(z["edge_polyline_offset"], np.int64)
            self.edge_ctrl = np.asarray(z["edge_ctrl"], float)
            self.surface_points = np.asarray(z["surface_points"], float)
            self.surface_tris = np.asarray(z["surface_tris"], np.int64)
            self.surface_tri_label = np.asarray(z["surface_tri_label"], np.int64)
        self.edge_curves = block_edge_curves(self.edges, self.edge_polyline,
                                             self.edge_offset)
        self.seam_curves = extract_seam_curves(
            self.surface_points, self.surface_tris, self.surface_tri_label)
        self._post()
        os.makedirs(os.path.dirname(cpath), exist_ok=True)
        blob = self._base_dict()
        blob["edge_curves"] = _cs_to_dict(self.edge_curves)
        blob["seam_curves"] = _cs_to_dict(self.seam_curves)
        torch.save(blob, cpath)

    def _base_dict(self) -> dict:
        return {"format": CACHE_FORMAT, "vertices": self.vertices,
                "blocks": self.blocks, "edges": self.edges,
                "edge_polyline": self.edge_polyline,
                "edge_offset": self.edge_offset, "edge_ctrl": self.edge_ctrl,
                "surface_points": self.surface_points,
                "surface_tris": self.surface_tris,
                "surface_tri_label": self.surface_tri_label}

    def _base_from(self, blob: dict) -> None:
        self.vertices = blob["vertices"]; self.blocks = blob["blocks"]
        self.edges = blob["edges"]; self.edge_polyline = blob["edge_polyline"]
        self.edge_offset = blob["edge_offset"]; self.edge_ctrl = blob["edge_ctrl"]
        self.surface_points = blob["surface_points"]
        self.surface_tris = blob["surface_tris"]
        self.surface_tri_label = blob["surface_tri_label"]

    def _post(self) -> None:
        self.interior_ref = self.surface_points.mean(0)
        self.frames = FrameModel(self.edge_curves, self.interior_ref)
        self.curves = concat(self.edge_curves, self.seam_curves)
        self.tri_tree = cKDTree(self.surface_points[self.surface_tris].mean(axis=1))
        self.vtree = cKDTree(self.vertices)
        self.edge_lookup = {(int(a), int(b)): i
                            for i, (a, b) in enumerate(self.edges)}

    def surface_nearest(self, q: np.ndarray, k: int = 16):
        """(...,3) -> (dist, Dreiecksindex, punkt) auf naechstem surface_tris."""
        import sys
        hexrepo = ("/home/t1dde/hydrostack_pipeline/stack/domain_partition_3D/"
                   "experimentell/hex3d_algohex")
        if hexrepo not in sys.path:
            sys.path.insert(0, hexrepo)
        from clean_blocks import _closest_point_on_tris
        q = np.asarray(q, float).reshape(-1, 3)
        _, cand = self.tri_tree.query(q, k=min(k, len(self.surface_tris)))
        cand = np.atleast_2d(cand)
        T = self.surface_tris[cand]
        d2, p = _closest_point_on_tris(q, self.surface_points[T[..., 0]],
                                       self.surface_points[T[..., 1]],
                                       self.surface_points[T[..., 2]])
        best = np.argmin(d2, axis=1)
        rows = np.arange(len(q))
        return np.sqrt(d2[rows, best]), cand[rows, best], p[rows, best]
