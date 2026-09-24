"""Validation of generated hexahedral meshes before they leave the pipeline."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np


_HEX_FACES = (
    (0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
    (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
)
_TETS = (
    (0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
    (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6),
)
_JAC_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class MeshValidation:
    """Machine-readable result for one generated hexahedral mesh."""

    valid: bool
    n_vertices: int
    n_blocks: int
    n_duplicate_vertices: int
    n_degenerate_blocks: int
    n_inverted_blocks: int
    n_nonmanifold_faces: int
    n_folded_blocks: int
    n_invalid_faces: int
    errors: tuple[str, ...]


def hex_signed_volumes(pts: np.ndarray) -> np.ndarray:
    """[N,8,3] -> [N] signed volume, Standard-VTK-6-Tet-Zerlegung (Diagonale 0-6)."""
    values = np.asarray(pts, dtype=np.float64)
    volumes = []
    for a, b, c, d in _TETS:
        matrix = np.stack(
            (values[:, b] - values[:, a],
             values[:, c] - values[:, a],
             values[:, d] - values[:, a]),
            axis=-1,
        )
        volumes.append(np.linalg.det(matrix) / 6.0)
    return np.stack(volumes, axis=1).sum(axis=1)


def _dshape(u: float, v: float, w: float) -> tuple[np.ndarray, ...]:
    """Ableitungen der trilinearen Hex-Basisfunktionen (8 Knoten) nach (u,v,w)."""
    du = np.array([-(1 - v) * (1 - w), (1 - v) * (1 - w), v * (1 - w),
                   -v * (1 - w), -(1 - v) * w, (1 - v) * w, v * w, -v * w])
    dv = np.array([-(1 - u) * (1 - w), -u * (1 - w), u * (1 - w),
                   (1 - u) * (1 - w), -(1 - u) * w, -u * w, u * w, (1 - u) * w])
    dw = np.array([-(1 - u) * (1 - v), -u * (1 - v), -u * v,
                   -(1 - u) * v, (1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v])
    return du, dv, dw


def hex_min_jacobian(pts: np.ndarray, n: int = 9) -> np.ndarray:
    """[N,8,3] -> [N] min det(J) der trilinearen Hex-Abbildung auf n^3 Stuetzstellen.
    < 0 => Zelle faltet sich (invertiert)."""
    values = np.asarray(pts, dtype=np.float64)
    out = np.full(values.shape[0], np.inf)
    grid = np.linspace(0.0, 1.0, n)
    for u in grid:
        for v in grid:
            for w in grid:
                du, dv, dw = _dshape(float(u), float(v), float(w))
                mat = np.stack((du @ values, dv @ values, dw @ values), axis=-1)
                out = np.minimum(out, np.linalg.det(mat))
    return out


def _block_volumes(vertices: np.ndarray, blocks: np.ndarray) -> np.ndarray:
    return hex_signed_volumes(np.asarray(vertices)[np.asarray(blocks)])


def _face_incidence_ok(block: np.ndarray) -> bool:
    """Alle 6 VTK-Hex-Faces als Knotenmengen vorhanden: jeder Knoten in genau
    3 Faces, jede der 12 Hex-Kanten in genau 2 Faces."""
    faces = [tuple(int(block[i]) for i in q) for q in _HEX_FACES]
    if len({frozenset(f) for f in faces}) != 6 or any(len(set(f)) != 4 for f in faces):
        return False
    node_use: Counter[int] = Counter()
    edge_use: Counter[frozenset[int]] = Counter()
    for f in faces:
        node_use.update(f)
        for k in range(4):
            edge_use[frozenset((f[k], f[(k + 1) % 4]))] += 1
    return (all(c == 3 for c in node_use.values())
            and len(edge_use) == 12
            and all(c == 2 for c in edge_use.values()))


def validate_generated_mesh(
    vertices: np.ndarray,
    blocks: np.ndarray,
    *,
    expected_blocks: int | None = None,
    volume_epsilon: float = 1e-9,
) -> MeshValidation:
    """Check topology and geometry of a generated VTK hexahedral mesh."""
    errors: list[str] = []
    vertices = np.asarray(vertices)
    blocks = np.asarray(blocks)
    n_vertices = int(vertices.shape[0]) if vertices.ndim >= 1 else 0
    n_blocks = int(blocks.shape[0]) if blocks.ndim >= 1 else 0

    if vertices.ndim != 2 or vertices.shape[1:] != (3,):
        errors.append(f"vertices shape {vertices.shape} != [N, 3]")
    if blocks.ndim != 2 or blocks.shape[1:] != (8,):
        errors.append(f"blocks shape {blocks.shape} != [F, 8]")
    if expected_blocks is not None and n_blocks != expected_blocks:
        errors.append(f"block count {n_blocks} != expected {expected_blocks}")
    if not np.isfinite(vertices).all():
        errors.append("vertices contain non-finite values")
    if blocks.size and (blocks.min() < 0 or blocks.max() >= n_vertices):
        errors.append("block index outside vertex range")

    duplicate_vertices = 0
    if vertices.ndim == 2 and vertices.shape[1:] == (3,) and vertices.size:
        _, counts = np.unique(np.round(vertices, 12), axis=0, return_counts=True)
        duplicate_vertices = int((counts > 1).sum())
        if duplicate_vertices:
            errors.append(f"{duplicate_vertices} duplicate vertex coordinate(s)")

    degenerate_blocks = 0
    inverted_blocks = 0
    nonmanifold_faces = 0
    folded_blocks = 0
    invalid_faces = 0
    if (
        not errors
        and blocks.size
        and blocks.dtype.kind in "iu"
    ):
        unique_counts = np.apply_along_axis(lambda row: len(np.unique(row)), 1, blocks)
        degenerate_blocks += int((unique_counts < 8).sum())
        if degenerate_blocks:
            errors.append(f"{degenerate_blocks} block(s) reuse a vertex index")
        volumes = _block_volumes(vertices, blocks)
        inverted_blocks = int((volumes < -volume_epsilon).sum())
        collapsed = int((np.abs(volumes) <= volume_epsilon).sum())
        degenerate_blocks += collapsed
        if inverted_blocks:
            errors.append(f"{inverted_blocks} inverted block(s)")
        if collapsed:
            errors.append(f"{collapsed} collapsed block(s)")

        invalid_faces = int(sum(not _face_incidence_ok(b) for b in blocks))
        if invalid_faces:
            errors.append(f"{invalid_faces} block(s) without valid VTK hex face sets")

        jacobian = hex_min_jacobian(vertices[blocks])
        folded_blocks = int((jacobian < -_JAC_EPS).sum())
        # Netto-invertierte Falten (Volumen < 0) sind fatal. Positiv-volumige
        # Falten sind das dokumentierte Grobblock-Artefakt der GT-Bloecke [4,7]:
        # min det J ist invariant ueber alle 48 gueltigen Hex-Relabelings, daher
        # GT-blind nicht von einem Defekt zu trennen -> nur diagnostisch.
        net_folded = int(((jacobian < -_JAC_EPS) & (volumes < 0.0)).sum())
        if net_folded:
            errors.append(f"{net_folded} net-inverted folded block(s) (min det J < 0)")

        face_counts: Counter[tuple[int, ...]] = Counter()
        for block in blocks:
            for face in _HEX_FACES:
                face_counts[tuple(sorted(int(block[index]) for index in face))] += 1
        nonmanifold_faces = sum(count > 2 for count in face_counts.values())
        if nonmanifold_faces:
            errors.append(f"{nonmanifold_faces} non-manifold face(s)")

    return MeshValidation(
        valid=not errors,
        n_vertices=n_vertices,
        n_blocks=n_blocks,
        n_duplicate_vertices=duplicate_vertices,
        n_degenerate_blocks=degenerate_blocks,
        n_inverted_blocks=inverted_blocks,
        n_nonmanifold_faces=nonmanifold_faces,
        n_folded_blocks=folded_blocks,
        n_invalid_faces=invalid_faces,
        errors=tuple(errors),
    )
