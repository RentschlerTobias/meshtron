"""Conditioning building blocks for HexaRow 3D (shared by training and inference).

Single source of truth for point-cloud generation and geometry split:
  - build_cloud: surface points -> normalized [n,4] cloud (r, sin, cos, z),
    optional blade oversampling (x3).
  - split_by_geometry: 90/10 split with GEOMETRY-LEVEL disjointness.

Parity: build_cloud(..., weights=None) produces bit-identically what
train_hexarow_full.sample_points(surface_cloud(raw), ...) produces. Weighting
is an ADDITIVE output (default weights=None -> old behavior unchanged).
This structurally excludes the drift class "training and inference clouds
differ".
"""
from __future__ import annotations

import numpy as np

# tet_prep.py label mapping: 1=hub, 2=shroud, 3=inlet, 4=outlet, 5=blade,
# 6=periodic_A, 7=periodic_B. Label 5 verified in the real sample.npz.
BLADE_LABEL = 5

# O-grid cut band surface around the blade row (npz surface label verified:
# r spans hub..shroud, z limited to the band, theta ±45 deg). Only used at
# generation time via band_weight; training data is untouched.
BAND_LABEL = 7


def polar_from_xyz(p: np.ndarray) -> np.ndarray:
    """[N,3] xyz -> [N,3] (r, theta, z), float64."""
    p = np.asarray(p, dtype=np.float64)
    return np.stack([np.hypot(p[:, 0], p[:, 1]),
                     np.arctan2(p[:, 1], p[:, 0]), p[:, 2]], axis=-1)


def surface_cloud(raw: dict) -> np.ndarray:
    """Conditioning source [M,3] polar (r,theta,z): complete geometry from
    surface_points (xyz), fallback vertices_polar. Identical to
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
    """Per-point blade flag: a point is blade if it hangs on a blade triangle
    (label==blade_label). tris [T,3] indices into surface_points."""
    mask = np.zeros(int(n_points), dtype=bool)
    tris = np.asarray(tris, dtype=np.int64)
    sel = np.asarray(tri_label) == blade_label
    if sel.any():
        mask[tris[sel].ravel()] = True
    return mask


def point_is_band(n_points: int, tris: np.ndarray, tri_label: np.ndarray,
                  band_label: int = BAND_LABEL) -> np.ndarray:
    """Per-point flag for the O-grid cut band (label band_label). Same logic
    as point_is_blade; tris index into surface_points."""
    mask = np.zeros(int(n_points), dtype=bool)
    tris = np.asarray(tris, dtype=np.int64)
    sel = np.asarray(tri_label) == band_label
    if sel.any():
        mask[tris[sel].ravel()] = True
    return mask


def _normalize(p: np.ndarray, rb: tuple, zb: tuple) -> np.ndarray:
    """(r,theta,z) -> [n,4] (r', sin, cos, z'); column order as in
    train_hexarow_full.sample_points."""
    r = (p[:, 0] - rb[0]) / max(1e-9, rb[1] - rb[0])
    z = (p[:, 2] - zb[0]) / max(1e-9, zb[1] - zb[0])
    return np.stack([r, np.sin(p[:, 1]), np.cos(p[:, 1]), z], axis=-1)


def build_cloud(sample: dict, n_points: int, rb: tuple, zb: tuple, rng,
                blade_weight: float | None = None,
                band_weight: float | None = None):
    """Point cloud [n,4] from the sample. weights=None -> bit-identical to
    sample_points(surface_cloud(sample), ...); otherwise blade points are
    oversampled with blade_weight (e.g. 3.0), second output = is_blade mask of
    the drawn points. Without is_blade in the sample -> uniform (no
    oversampling). band_weight (opt-in, generation only) oversamples the
    O-grid band points compositionally: band points keep their blade_weight
    and gain an extra band_weight factor; requires is_band in sample."""
    vp = surface_cloud(sample)
    is_blade = sample.get("is_blade")
    is_band = sample.get("is_band")
    if blade_weight is None or is_blade is None or (
            band_weight is not None and is_band is None):
        idx = rng.choice(len(vp), size=n_points, replace=len(vp) < n_points)
        return _normalize(vp[idx], rb, zb), None
    is_blade = np.asarray(is_blade, dtype=bool)
    w = np.where(is_blade, float(blade_weight), 1.0)
    if band_weight is not None:
        is_band = np.asarray(is_band, dtype=bool)
        w = np.where(is_band, w * float(band_weight), w)
    idx = rng.choice(len(vp), size=n_points, replace=True, p=w / w.sum())
    return _normalize(vp[idx], rb, zb), is_blade[idx]


def split_by_geometry(items: list[dict], val_frac: float = 0.1,
                      seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Geometry-disjoint split: identical geom_id never on both sides.
    Geometries sorted, then shuffled (seed) -> deterministic."""
    geoms = sorted({it["geom_id"] for it in items})
    if len(geoms) <= 1:
        return list(items), list(items)
    perm = np.random.default_rng(seed).permutation(len(geoms))
    n_val = max(1, min(len(geoms) - 1, int(round(len(geoms) * val_frac))))
    val_geoms = {geoms[int(i)] for i in perm[:n_val]}
    train = [it for it in items if it["geom_id"] not in val_geoms]
    val = [it for it in items if it["geom_id"] in val_geoms]
    return train, val
