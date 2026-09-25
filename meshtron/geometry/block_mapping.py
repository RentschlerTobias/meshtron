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

    def _edge_nearest(self, q: np.ndarray, cfg: SnapConfig):
        """(n,3) -> (dist, edge_id, punkt) auf der naechsten edge_polyline."""
        _, cand = self.seg_tree.query(q, k=min(cfg.k_seg, len(self.seg_mid)))
        cand = np.atleast_2d(cand)
        p = _project_to_segments(q[:, None, :], self.seg_a[cand],
                                 self.seg_b[cand])
        d = np.linalg.norm(q[:, None, :] - p, axis=-1)
        best = np.argmin(d, axis=1)
        rows = np.arange(len(q))
        return d[rows, best], self.seg_edge[cand[rows, best]], p[rows, best]

    def _surface_nearest(self, q: np.ndarray, cfg: SnapConfig):
        """(n,3) -> (dist, Dreiecksindex, punkt) auf der naechsten surface_tris."""
        _, cand = self.tri_tree.query(q, k=min(cfg.k_tri, len(self.surface_tris)))
        cand = np.atleast_2d(cand)
        T = self.surface_tris[cand]
        d2, p = _closest_point_on_tris(q, self.surface_points[T[..., 0]],
                                       self.surface_points[T[..., 1]],
                                       self.surface_points[T[..., 2]])
        best = np.argmin(d2, axis=1)
        rows = np.arange(len(q))
        return np.sqrt(d2[rows, best]), cand[rows, best], p[rows, best]

    def snap_corners(self, corners: np.ndarray, cfg: SnapConfig | None = None
                     ) -> tuple[np.ndarray, list[dict]]:
        """(nb,8,3) -> (gesnappt, pro-Ecke {tier, feature_id, dist, target}).

        Injizierend auf Feature-Punkten: jede GT-Ecke darf von hoechstens EINER
        eindeutigen generierten Ecke beansprucht werden, sonst kollabiert eine
        Kante auf Laenge 0. Identische Rohkoordinaten (geteilter Vertex ueber
        mehrere Bloecke) zaehlen als EINE Stimme und bekommen identische
        Records, damit der Scatter snapped_v[blocks]=C_snap konsistent bleibt.
        Verlierer steigen eine Stufe ab (edge innerhalb tol_e, sonst surface).
        feature_id: vertex-Index | edge-Index | Dreiecksindex (surface).
        """
        cfg = cfg or SnapConfig()
        C = np.asarray(corners, dtype=np.float64)
        flat = C.reshape(-1, 3)
        uniq, inv = np.unique(flat, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        g = len(uniq)

        dv, iv = self.vertex_tree.query(uniq, k=1)
        de, ie, pe = self._edge_nearest(uniq, cfg)
        dist, fid, target = self._surface_nearest(uniq, cfg)
        tier = np.full(g, 2, dtype=np.int8)

        # Feature-Punkt: naechster Anspruch gewinnt, Gleichstand -> kleinster
        # Eckindex der Gruppe (deterministisch). Alle anderen fallen weiter.
        first = np.full(g, np.iinfo(np.int64).max, dtype=np.int64)
        np.minimum.at(first, inv, np.arange(len(flat), dtype=np.int64))
        winners = np.zeros(g, bool)
        claimed: set[int] = set()
        for gi in np.lexsort((first, dv)):
            if dv[gi] > cfg.tol_v:
                break
            f = int(iv[gi])
            if f not in claimed:
                claimed.add(f)
                winners[gi] = True

        target, fid, dist = target.copy(), fid.copy(), dist.copy()
        if winners.any():
            tier[winners] = 0
            target[winners] = self.vertices[iv[winners]]
            fid[winners] = iv[winners]
            dist[winners] = dv[winners]
        edge_ok = (~winners) & (de <= cfg.tol_e)
        tier[edge_ok] = 1
        target[edge_ok] = pe[edge_ok]
        fid[edge_ok] = ie[edge_ok]
        dist[edge_ok] = de[edge_ok]

        names = np.array(["vertex", "edge", "surface"])
        snapped = target[inv]
        records = [{"tier": str(names[tier[inv[i]]]), "feature_id": int(fid[inv[i]]),
                    "dist": float(dist[inv[i]]),
                    "target": target[inv[i]].tolist()} for i in range(len(flat))]
        return snapped.reshape(C.shape), records


def tier_counts(records: list[dict]) -> dict[str, int]:
    """Wie viele Ecken pro Prioritaetsstufe (vertex/edge/surface)."""
    out = {"vertex": 0, "edge": 0, "surface": 0}
    for r in records:
        out[r["tier"]] = out.get(r["tier"], 0) + 1
    return out


@dataclass(frozen=True, slots=True)
class SnapConfigV2(SnapConfig):
    """Toleranzen des Kurven-Snappings: Kurven-ENDPUNKT | Kurve | Flaeche."""


def snap_corners_v2(fm, corners: np.ndarray, cfg: SnapConfigV2 | None = None
                    ) -> tuple[np.ndarray, list[dict]]:
    """(nb,8,3) -> (gesnappt, Records) gegen das Geometrie-Kurvenmodell.

    Vertex-Tier zielt auf Kurven-ENDPUNKTE (`FeatureModelV2.curves.ep_pt`), die
    ueber die deduplizierte Punktmenge injizierend beansprucht werden — sonst
    kollabiert eine Kante auf Laenge 0. Die `claimed`-Semantik ist identisch zu
    `FeatureModel.snap_corners`; zusaetzlich tragen die Records `curve_id`/`t`
    (Bogenlaenge am Ziel), was die Kurven-Rekonstruktion speist.
    feature_id: deduplizierter Endpunkt | Kurvenindex | Dreiecksindex.
    """
    cfg = cfg or SnapConfigV2()
    C = np.asarray(corners, dtype=np.float64)
    flat = C.reshape(-1, 3)
    uniq, inv = np.unique(flat, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    g = len(uniq)
    cs = fm.curves
    eu, eu_inv = (np.unique(cs.ep_pt, axis=0, return_inverse=True)
                  if len(cs.ep_pt) else (np.zeros((0, 3)), np.zeros(0, np.int64)))
    eu_inv = np.asarray(eu_inv).reshape(-1)
    dv, iv = cs.nearest_endpoint(uniq)
    eid = eu_inv[iv] if len(eu_inv) else np.zeros(g, np.int64)
    de, ce, te, pe = cs.nearest(uniq)
    ds, isurf, ps = fm.surface_nearest(uniq, cfg.k_tri)
    tier = np.full(g, 2, dtype=np.int8)

    first = np.full(g, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(first, inv, np.arange(len(flat), dtype=np.int64))
    winners = np.zeros(g, bool)
    claimed: set[int] = set()
    for gi in np.lexsort((first, dv)):
        if dv[gi] > cfg.tol_v:
            break
        f = int(eid[gi])
        if f not in claimed:
            claimed.add(f)
            winners[gi] = True

    target, fid, dist = ps.copy(), isurf.copy(), ds.copy()
    cid = np.full(g, -1, np.int64)
    tt = np.zeros(g)
    if winners.any():
        tier[winners] = 0
        target[winners] = eu[eid[winners]]
        fid[winners] = eid[winners]
        dist[winners] = dv[winners]
    edge_ok = (~winners) & (de <= cfg.tol_e)
    tier[edge_ok] = 1
    target[edge_ok] = pe[edge_ok]
    fid[edge_ok] = ce[edge_ok]
    dist[edge_ok] = de[edge_ok]
    cid[edge_ok] = ce[edge_ok]
    tt[edge_ok] = te[edge_ok]
    cid[winners] = cs.ep_curve[iv[winners]]
    tt[winners] = cs.ep_t[iv[winners]]

    names = np.array(["vertex", "edge", "surface"])
    snapped = target[inv]
    records = [{"tier": str(names[tier[inv[i]]]), "feature_id": int(fid[inv[i]]),
                "dist": float(dist[inv[i]]), "curve_id": int(cid[inv[i]]),
                "t": float(tt[inv[i]]), "target": target[inv[i]].tolist()}
               for i in range(len(flat))]
    return snapped.reshape(C.shape), records


def score_candidate(corners: np.ndarray, records: list[dict]
                    ) -> tuple[float, float]:
    """(mean snap dist, min det(J)) fuer Blockecken (nb,8,3) NACH dem Snappen."""
    from meshtron.geometry.mesh_validation import hex_min_jacobian
    mean_dist = float(np.mean([r["dist"] for r in records])) if records else 0.0
    jac = hex_min_jacobian(np.asarray(corners, dtype=np.float64))
    return mean_dist, float(jac.min())
