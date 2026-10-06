"""mesh_quality.py -- checkMesh-style quality metrics of a refilled (post-TFI) hex mesh,
and the [0, 1] score R_mesh used by meshtron.training.rewards_v2.

Metrics (OpenFOAM checkMesh semantics where one exists):
  sj_min, sj_p1     scaled Jacobian at the cell corners (1 = perfect, <= 0 = inverted)
  n_inverted        cells with any corner scaled Jacobian <= 0
  nonortho_max/mean angle between the face normal and the owner->neighbour centre
                    vector over internal faces, degrees (checkMesh warns > 70)
  skew_max          checkMesh skewness: distance between the face centre and the
                    intersection of the centre-centre line with the face, divided by
                    the centre-centre distance (checkMesh warns > 4)
  aspect_max        longest / shortest cell edge

R_mesh = 0 if any cell is inverted, else the mean of three ramps:
  clip(sj_p1 / 0.5), clip((80 - nonortho_max) / 40), clip((1.5 - skew_max) / 1.0)
(GT TFI meshes of the relabelled n2000 set: sj_p1 0.62-0.83, nonortho_max 51-72,
 skew_max 0.44-0.73 -> R_mesh 0.69-0.91)
"""
from __future__ import annotations

import sys

import numpy as np

# Same local vertex order as meshtron.geometry.mesh_validation (VTK hexahedron).
HEX_FACES = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
# For every corner: its three neighbours along the local u, v, w directions.
CORNER_NB = ((1, 3, 4), (2, 0, 5), (3, 1, 6), (0, 2, 7),
             (7, 5, 0), (4, 6, 1), (5, 7, 2), (6, 4, 3))


def scaled_jacobian_corners(cells: np.ndarray) -> np.ndarray:
    """[N,8,3] -> [N,8] scaled Jacobian at the corners (det of the unit edge triad)."""
    nb = np.asarray(CORNER_NB)
    e = cells[:, nb, :] - cells[:, :, None, :]                      # [N,8,3,3]
    n = np.linalg.norm(e, axis=-1, keepdims=True)
    e = e / np.maximum(n, 1e-30)
    return np.linalg.det(e)


def metrics(points: np.ndarray, hexes: np.ndarray) -> dict:
    p = np.asarray(points, np.float64); h = np.asarray(hexes, np.int64)
    cells = p[h]
    sj = scaled_jacobian_corners(cells)
    cc = cells.mean(1)
    # internal faces: owner / neighbour pairs by sorted vertex key
    f = h[:, np.asarray(HEX_FACES)].reshape(-1, 4)                   # [6N,4]
    owner = np.repeat(np.arange(len(h)), 6)
    key = np.sort(f, axis=1)
    order = np.lexsort(key.T[::-1])
    key, f, owner = key[order], f[order], owner[order]
    same = (key[1:] == key[:-1]).all(1)
    i = np.flatnonzero(same)
    fo, a, b = f[i], owner[i], owner[i + 1]
    q = p[fo]
    fc = q.mean(1)
    n = np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 1]); n /= np.linalg.norm(n, axis=1, keepdims=True)
    d = cc[b] - cc[a]; dl = np.linalg.norm(d, axis=1)
    cosang = np.abs((n * d).sum(1)) / dl
    nonortho = np.degrees(np.arccos(np.clip(cosang, 0, 1)))
    # skewness: intersection of the centre line with the face plane
    t = ((fc - cc[a]) * n).sum(1) / np.where(np.abs((d * n).sum(1)) < 1e-30, 1e-30, (d * n).sum(1))
    x = cc[a] + t[:, None] * d
    skew = np.linalg.norm(fc - x, axis=1) / dl
    e = np.linalg.norm(cells[:, [1, 2, 3, 0, 5, 6, 7, 4, 4, 5, 6, 7]] - cells[:, [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3]], axis=-1)
    return dict(n_cells=len(h), sj_min=float(sj.min()), sj_p1=float(np.percentile(sj.min(1), 1)),
                n_inverted=int((sj <= 0).any(1).sum()),
                nonortho_max=float(nonortho.max()), nonortho_mean=float(nonortho.mean()),
                skew_max=float(skew.max()), aspect_max=float((e.max(1) / np.maximum(e.min(1), 1e-30)).max()))


def r_mesh(m: dict) -> float:
    if m["n_inverted"]:
        return 0.0
    ramps = (np.clip(m["sj_p1"] / 0.5, 0, 1), np.clip((80 - m["nonortho_max"]) / 40, 0, 1),
             np.clip((1.5 - m["skew_max"]) / 1.0, 0, 1))
    return float(np.mean(ramps))


def from_vtk(path: str) -> dict:
    import meshio
    m = meshio.read(path)
    hexes = np.concatenate([c.data for c in m.cells if c.type == "hexahedron"])
    return metrics(m.points, hexes)


if __name__ == "__main__":
    for path in sys.argv[1:]:
        mm = from_vtk(path)
        print(path, {k: round(v, 3) if isinstance(v, float) else v for k, v in mm.items()}, "R_mesh", round(r_mesh(mm), 3))
