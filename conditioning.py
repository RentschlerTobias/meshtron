"""Conditioning-Bausteine fuer HexaRow 3D (geteilt von Training und Inferenz).

Eine einzige Wahrheitsquelle fuer Punktwolken-Erzeugung und Geometrie-Split:
  - build_cloud: Oberflaechenpunkte -> normalisierte [n,4]-Wolke
    (r, sin, cos, z), optional Blade-oversampled (x3).
  - split_by_geometry: 90/10-Split mit GEOM-LEVEL-Disjunktheit.

Paritaet: build_cloud(..., weights=None) erzeugt bit-identisch das, was
train_hexarow_full.sample_points(surface_cloud(raw), ...) erzeugt. Gewichtung
ist ein ADDITIVER Output (Default weights=None -> altes Verhalten unveraendert).
Das schliesst die Drift-Klasse "Train- und Inferenz-Wolke unterscheiden sich"
strukturell aus.
"""
from __future__ import annotations

import numpy as np

# tet_prep.py Label-Mapping: 1=hub, 2=shroud, 3=inlet, 4=outlet, 5=blade,
# 6=periodic_A, 7=periodic_B. Label 5 in realer sample.npz verifiziert.
BLADE_LABEL = 5


def polar_from_xyz(p: np.ndarray) -> np.ndarray:
    """[N,3] xyz -> [N,3] (r, theta, z), float64."""
    p = np.asarray(p, dtype=np.float64)
    return np.stack([np.hypot(p[:, 0], p[:, 1]),
                     np.arctan2(p[:, 1], p[:, 0]), p[:, 2]], axis=-1)


def surface_cloud(raw: dict) -> np.ndarray:
    """Konditionierungs-Quelle [M,3] polar (r,theta,z): komplette Geometrie aus
    surface_points (xyz), Fallback vertices_polar. Identisch zu
    train_hexarow_full.surface_cloud."""
    sp = raw.get("surface_points")
    if sp is not None:
        p = np.asarray(sp.detach().cpu().numpy() if hasattr(sp, "detach") else sp,
                       dtype=np.float64)
        return polar_from_xyz(p)
    vp = raw["vertices_polar"]
    vp = vp.detach().cpu().numpy() if hasattr(vp, "detach") else vp
    return np.asarray(vp, dtype=np.float64)


def point_is_blade(n_points: int, tris: np.ndarray, tri_label: np.ndarray,
                   blade_label: int = BLADE_LABEL) -> np.ndarray:
    """Per-Punkt-Blade-Flag: Punkt ist Blade, wenn er an einem Blade-Triangle
    (label==blade_label) haengt. tris [T,3] Indizes in surface_points."""
    mask = np.zeros(int(n_points), dtype=bool)
    tris = np.asarray(tris, dtype=np.int64)
    sel = np.asarray(tri_label) == blade_label
    if sel.any():
        mask[tris[sel].ravel()] = True
    return mask


def _normalize(p: np.ndarray, rb: tuple, zb: tuple) -> np.ndarray:
    """(r,theta,z) -> [n,4] (r', sin, cos, z'); Spaltenreihenfolge wie
    train_hexarow_full.sample_points."""
    r = (p[:, 0] - rb[0]) / max(1e-9, rb[1] - rb[0])
    z = (p[:, 2] - zb[0]) / max(1e-9, zb[1] - zb[0])
    return np.stack([r, np.sin(p[:, 1]), np.cos(p[:, 1]), z], axis=-1)


def build_cloud(sample: dict, n_points: int, rb: tuple, zb: tuple, rng,
                blade_weight: float | None = None):
    """Punktwolke [n,4] aus dem Sample. weights=None -> bit-identisch zu
    sample_points(surface_cloud(sample), ...); sonst Blade-Punkte mit
    blade_weight (z.B. 3.0) oversampled, zweiter Output = is_blade-Maske der
    gezogenen Punkte. Ohne is_blade im Sample -> uniform (kein Oversample)."""
    vp = surface_cloud(sample)
    is_blade = sample.get("is_blade")
    if blade_weight is None or is_blade is None:
        idx = rng.choice(len(vp), size=n_points, replace=len(vp) < n_points)
        return _normalize(vp[idx], rb, zb), None
    is_blade = np.asarray(is_blade, dtype=bool)
    w = np.where(is_blade, float(blade_weight), 1.0)
    idx = rng.choice(len(vp), size=n_points, replace=True, p=w / w.sum())
    return _normalize(vp[idx], rb, zb), is_blade[idx]


def split_by_geometry(items: list[dict], val_frac: float = 0.1,
                      seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Geometrie-disjunkter Split: identische geom_id nie auf beiden Seiten.
    Geometrien sortiert, dann geshuffelt (seed) -> deterministisch."""
    geoms = sorted({it["geom_id"] for it in items})
    if len(geoms) <= 1:
        return list(items), list(items)
    perm = np.random.default_rng(seed).permutation(len(geoms))
    n_val = max(1, min(len(geoms) - 1, int(round(len(geoms) * val_frac))))
    val_geoms = {geoms[int(i)] for i in perm[:n_val]}
    train = [it for it in items if it["geom_id"] not in val_geoms]
    val = [it for it in items if it["geom_id"] in val_geoms]
    return train, val
