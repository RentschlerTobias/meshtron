"""block_mapping.py — generierte Block-Ecken auf das reale Feature-Modell snappen.

Das Ziel ist nicht die naechste Flaeche, sondern die naechste *Feature*-Struktur
(Nutzer-Vorgabe). Deshalb eine Dimensionsprioritaet statt reiner NN-Suche:

  1. Feature-Punkt  = GT-Blockeckpunkt  (LE/TE-Spitzen, Wand-Junction-Vertices)
     naechster Punkt <= tol_v -> tier "vertex".
  2. Feature-Kurve  = naechstes Segment der edge_polyline (Blade-Profilkanten,
     sonstige Feature-Kanten), Segmentprojektion <= tol_e -> tier "edge".
  3. Flaeche        = naechstes Dreieck der surface_tris -> tier "surface".

Die Reihenfolge ist bewusst kein argmin: ein Punkt 0.03 neben einer Blade-Kante
bleibt auf der Kante, auch wenn das naechste Flaeschendreieck naeher liegt.
Das schuetzt die CFD-relevante Kantentreue an LE/TE, wo eine reine
Flaechenprojektion die Kurve glattbuegelt.

Auf dem GT-Modell ist das Snappen idempotent: C == vertices[blocks] liefert
dist 0, tier "vertex" fuer jede Ecke.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# _closest_point_on_tris: exakte Dreiecksprojektion aus dem TFI-Repo (Import,
# niemals editieren) — dieselbe Routine, die clean_blocks fuer die Boundary
# nutzt, also dieselbe Metrik.
HEX3D_REPO = ("/home/t1dde/hydrostack_pipeline/stack/domain_partition_3D/"
              "experimentell/hex3d_algohex")
if HEX3D_REPO not in sys.path:
    sys.path.insert(0, HEX3D_REPO)
from clean_blocks import _closest_point_on_tris  # noqa: E402


@dataclass(frozen=True, slots=True)
class SnapConfig:
    """Toleranzen der Dimensionsprioritaet (physikalische Einheiten)."""

    tol_v: float = 0.06   # Feature-Punkt (GT-Blockecke)
    tol_e: float = 0.04   # Feature-Kurve (edge_polyline-Segment)
    k_seg: int = 16       # Segmentkandidaten fuer die cKDTree-Vorauswahl
    k_tri: int = 16       # Dreieckskandidaten fuer die surface-Projektion


def _project_to_segments(q: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Naechster Punkt auf Segmenten [n,k,3] fuer q [n,1,3] (vektorisiert)."""
    ab = b - a
    t = np.clip(((q - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-30),
                0.0, 1.0)
    return a + t[..., None] * ab


class FeatureModel:
    """sample.npz: GT-Blockecken, Feature-Kurven (edge_polyline) und -Flaechen."""

    def __init__(self, npz_path: str | os.PathLike) -> None:
        with np.load(npz_path) as z:
            self.path = str(npz_path)
            self.vertices = np.asarray(z["vertices"], dtype=np.float64)
            self.blocks = np.asarray(z["blocks"], dtype=np.int64)
            self.edge_offsets = np.asarray(z["edge_polyline_offset"], dtype=np.int64)
            self.edge_polyline = np.asarray(z["edge_polyline"], dtype=np.float64)
            self.edges = np.asarray(z["edges"], dtype=np.int64)
            self.surface_points = np.asarray(z["surface_points"], dtype=np.float64)
            self.surface_tris = np.asarray(z["surface_tris"], dtype=np.int64)
            self.surface_tri_label = np.asarray(z["surface_tri_label"], dtype=np.int64)
        self._build_segments()
        self._build_trees()

    def _build_segments(self) -> None:
        """Pro edge_polyline-Kante alle Segmente sammeln (start/ende + edge id)."""
        starts, edge_of = [], []
        for e in range(len(self.edge_offsets) - 1):
            lo, hi = int(self.edge_offsets[e]), int(self.edge_offsets[e + 1])
            idx = np.arange(lo, max(lo, hi - 1))
            starts.append(idx)
            edge_of.append(np.full(len(idx), e, dtype=np.int64))
        starts = (np.concatenate(starts) if starts
                  else np.empty(0, dtype=np.int64))
        self.seg_a = self.edge_polyline[starts]
        self.seg_b = self.edge_polyline[starts + 1]
        self.seg_edge = (np.concatenate(edge_of) if edge_of
                         else np.empty(0, dtype=np.int64))
        self.seg_mid = 0.5 * (self.seg_a + self.seg_b)

    def _build_trees(self) -> None:
        self.vertex_tree = cKDTree(self.vertices)
        self.seg_tree = cKDTree(self.seg_mid)
        centroids = self.surface_points[self.surface_tris].mean(axis=1)
        self.tri_tree = cKDTree(centroids)

    def snap_corners(self, corners: np.ndarray, cfg: SnapConfig | None = None
                     ) -> tuple[np.ndarray, list[dict]]:
        """(nb,8,3) -> (gesnappt, pro-Ecke {tier, dist, feature_id}).

        feature_id: vertex-Index | edge-Index | Dreiecksindex (surface).
        """
        cfg = cfg or SnapConfig()
        C = np.asarray(corners, dtype=np.float64)
        flat = C.reshape(-1, 3)
        n = len(flat)
        snapped = flat.copy()
        dists = np.zeros(n)
        fids = np.full(n, -1, dtype=np.int64)
        tier = np.full(n, "surface", dtype=object)

        # 1) Feature-Punkte
        dv, iv = self.vertex_tree.query(flat, k=1)
        sel = dv <= cfg.tol_v
        snapped[sel] = self.vertices[iv[sel]]
        dists[sel] = dv[sel]
        fids[sel] = iv[sel]
        tier[sel] = "vertex"

        # 2) Feature-Kurven (nur was nicht schon ein Feature-Punkt ist)
        rem = np.flatnonzero(~sel)
        if len(rem):
            q = flat[rem]
            _, cand = self.seg_tree.query(q, k=min(cfg.k_seg, len(self.seg_mid)))
            cand = np.atleast_2d(cand)
            p = _project_to_segments(q[:, None, :], self.seg_a[cand],
                                     self.seg_b[cand])
            d = np.linalg.norm(q[:, None, :] - p, axis=-1)
            best = np.argmin(d, axis=1)
            rows = np.arange(len(q))
            dbest = d[rows, best]
            hit = dbest <= cfg.tol_e
            tgt = rem[hit]
            snapped[tgt] = p[rows[hit], best[hit]]
            dists[tgt] = dbest[hit]
            fids[tgt] = self.seg_edge[cand[rows[hit], best[hit]]]
            tier[tgt] = "edge"

        # 3) Flaechen-Fallback
        rem2 = np.flatnonzero(tier == "surface")
        if len(rem2):
            q = flat[rem2]
            _, cand = self.tri_tree.query(q, k=min(cfg.k_tri,
                                                   len(self.surface_tris)))
            cand = np.atleast_2d(cand)
            T = self.surface_tris[cand]
            d2, p = _closest_point_on_tris(
                q, self.surface_points[T[..., 0]],
                self.surface_points[T[..., 1]], self.surface_points[T[..., 2]])
            best = np.argmin(d2, axis=1)
            rows = np.arange(len(q))
            snapped[rem2] = p[rows, best]
            dists[rem2] = np.sqrt(d2[rows, best])
            fids[rem2] = cand[rows, best]

        records = [{"tier": str(tier[i]), "dist": float(dists[i]),
                    "feature_id": int(fids[i])} for i in range(n)]
        return snapped.reshape(C.shape), records


def tier_counts(records: list[dict]) -> dict[str, int]:
    """Wie viele Ecken pro Prioritaetsstufe (vertex/edge/surface)."""
    out = {"vertex": 0, "edge": 0, "surface": 0}
    for r in records:
        out[r["tier"]] = out.get(r["tier"], 0) + 1
    return out


def score_candidate(corners: np.ndarray, records: list[dict]
                    ) -> tuple[float, float]:
    """(mean snap dist, min det(J)) fuer Blockecken (nb,8,3) NACH dem Snappen."""
    from mesh_validation import hex_min_jacobian
    mean_dist = float(np.mean([r["dist"] for r in records])) if records else 0.0
    jac = hex_min_jacobian(np.asarray(corners, dtype=np.float64))
    return mean_dist, float(jac.min())
