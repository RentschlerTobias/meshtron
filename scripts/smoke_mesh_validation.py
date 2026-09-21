#!/usr/bin/env python3
"""Deterministic smoke test for generated hexahedral mesh validation."""

from __future__ import annotations

import numpy as np

from mesh_validation import validate_generated_mesh


def cube() -> tuple[np.ndarray, np.ndarray]:
    """Return one positively oriented unit hexahedron."""
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float64)
    return vertices, np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)


def main() -> None:
    """Exercise valid, duplicate-index, inverted, and count checks."""
    vertices, blocks = cube()
    valid = validate_generated_mesh(vertices, blocks, expected_blocks=1)
    assert valid.valid, valid.errors

    duplicate = blocks.copy()
    duplicate[0, 7] = 0
    duplicate_result = validate_generated_mesh(vertices, duplicate)
    assert not duplicate_result.valid
    assert duplicate_result.n_degenerate_blocks == 1

    inverted = blocks[:, [0, 3, 2, 1, 4, 7, 6, 5]]
    inverted_result = validate_generated_mesh(vertices, inverted)
    assert not inverted_result.valid
    assert inverted_result.n_inverted_blocks == 1
    assert inverted_result.n_folded_blocks >= 1

    folded = blocks.copy()
    folded[0, 0], folded[0, 6] = folded[0, 6], folded[0, 0]
    folded_result = validate_generated_mesh(vertices, folded)
    assert not folded_result.valid
    assert folded_result.n_folded_blocks >= 1

    count_result = validate_generated_mesh(vertices, blocks, expected_blocks=2)
    assert not count_result.valid
    assert "expected 2" in count_result.errors[0]

    nonmanifold = np.repeat(blocks, 3, axis=0)
    nonmanifold_result = validate_generated_mesh(vertices, nonmanifold)
    assert not nonmanifold_result.valid
    assert nonmanifold_result.n_nonmanifold_faces == 6
    print("mesh validation smoke: PASS")


if __name__ == "__main__":
    main()
