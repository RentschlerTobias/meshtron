"""
domain_extractor.py

Extrahiert aus den domain_partition Meshes die Tangenten aus edge_to_streamline,
konvertiert Vertices und Tangenten in Polarkoordinaten und speichert ein
vorverarbeitetes Dataset.

Output pro Mesh:
    - vertices_polar:      [n_verts, 2]  -> (r, theta)
    - edges_polar:         dict{(u,v): (t_norm, alpha_start, alpha_end)}
    - faces:               [4, n_faces]
    - edge_to_streamline:  Original (zur Rekonstruktion)
    - tri_coordinates:     Point cloud for conditioning
    - center:              [2] (reference point for polar coordinates)
    - bounds:              [x_min, y_min, x_max, y_max]
"""

import torch
import numpy as np
from torch_geometric.data import Data
from typing import Dict, Tuple, List
from pathlib import Path
from tqdm import tqdm


def fit_tangent(points: np.ndarray, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fitte eine Gerade durch die ersten (bzw. letzten) k Punkte und
    gib Richtungsvektor (normalisiert) + Norm zurück.

    Args:
        points: [N, 2] array von Punkten entlang einer Kante.
        k: Anzahl Punkte für den Fit.

    Returns:
        direction: [2] normalisierter Richtungsvektor.
        norm:      Skalare Länge der Sehne (||p[-1] - p[0]||) als Proxy
                   für die "Stärke" der Tangente.
    """
    if len(points) < 2:
        # Fallback: Null-Vektor
        return np.zeros(2, dtype=np.float32), 0.0

    # Erste k Punkte
    n = min(k, len(points))
    pts = points[:n]  # [n, 2]

    # Lineare Regression: y ~ x
    x = pts[:, 0]
    y = pts[:, 1]

    if n == 2:
        dx = x[1] - x[0]
        dy = y[1] - y[0]
    else:
        # Polyfit 1. Grades
        # Wir fitten x(t) und y(t) gegen Parameter t=0..n-1
        t = np.arange(n, dtype=np.float32)
        # Robust gegen vertikale Linien: wir fitten beide gegen t
        dx = np.polyfit(t, x, 1)[0]
        dy = np.polyfit(t, y, 1)[0]

    direction = np.array([dx, dy], dtype=np.float32)
    norm_dir = np.linalg.norm(direction)
    if norm_dir < 1e-12:
        direction = np.zeros(2, dtype=np.float32)
    else:
        direction = direction / norm_dir

    # Norm = Sehnenlänge als Proxy für Tangentialstärke
    chord = points[-1] - points[0]
    norm = float(np.linalg.norm(chord))

    return direction, norm


def compute_tangents(edge_to_streamline: Dict[Tuple[int, int], np.ndarray],
                     k: int = 3) -> Dict[Tuple[int, int], Tuple[np.ndarray, float, np.ndarray, float]]:
    """
    Berechnet für jede gerichtete Kante:
        (t_start_dir, t_start_norm, t_end_dir, t_end_norm)

    Args:
        edge_to_streamline: dict{(u,v): ndarray[N,2]}.
        k: Fit-Länge.

    Returns:
        dict{(u,v): (dir_start, norm_start, dir_end, norm_end)}
    """
    tangents = {}
    for (u, v), pts in edge_to_streamline.items():
        pts_arr = np.asarray(pts, dtype=np.float32)

        # Start-Tangente (am u, in Richtung v)
        dir_start, norm_start = fit_tangent(pts_arr, k=k)

        # End-Tangente (am v, in Richtung v) -> fit über letzte k Punkte
        pts_rev = pts_arr[::-1]
        dir_end_rev, norm_end = fit_tangent(pts_rev, k=k)
        # dir_end_rev zeigt von v zurück zu u; wir wollen Richtung u->v
        dir_end = -dir_end_rev

        tangents[(u, v)] = (dir_start, norm_start, dir_end, norm_end)

    return tangents


def cartesian_to_polar(vertices: torch.Tensor,
                       center: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Konvertiere 2D-Koordinaten in Polarkoordinaten relativ zu center.

    Returns:
        r:     [n_verts]
        theta: [n_verts] in [0, 2π)
        bounds: [x_min, y_min, x_max, y_max] der ORIGINALEN Koordinaten
    """
    rel = vertices - center  # [n, 2]
    r = torch.norm(rel, dim=1)
    theta = torch.atan2(rel[:, 1], rel[:, 0])  # [-π, π]
    theta = torch.where(theta < 0, theta + 2 * np.pi, theta)  # [0, 2π)

    x_min, y_min = torch.min(vertices, dim=0).values
    x_max, y_max = torch.max(vertices, dim=0).values
    bounds = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)

    return r, theta, bounds


def extract_mesh_data(mesh: Data) -> Dict:
    """
    Extrahiert alle relevanten Daten aus einem domain_partition Mesh.
    """
    blocking_nodes = mesh.blocking_nodes  # [n_verts, 2]
    blocking_faces = mesh.blocking_faces  # [4, n_faces]
    edge_to_streamline = mesh.edge_to_streamline  # dict
    tri_coordinates = mesh.tri_coordinates  # [N, 3] (col 2 = boundary flag)

    # Zentrum: Schwerpunkt der triangulierten Koordinaten (nur Boundary + Interior)
    # oder Schwerpunkt der blocking_nodes
    center = torch.mean(blocking_nodes, dim=0)  # [2]

    # Vertices -> Polar
    r, theta, bounds = cartesian_to_polar(blocking_nodes, center)

    # Tangenten
    tangents = compute_tangents(edge_to_streamline, k=3)

    # Für jede gerichtete Kante: berechne alpha (Winkel des Richtungsvektors)
    n_edges = len(tangents)
    edge_index = torch.zeros((2, n_edges), dtype=torch.long)
    edge_tangents = torch.zeros((n_edges, 4), dtype=torch.float32)  # [alpha_s, t_norm_s, alpha_e, t_norm_e]

    for i, ((u, v), (dir_start, norm_start, dir_end, norm_end)) in enumerate(tangents.items()):
        alpha_start = float(np.arctan2(dir_start[1], dir_start[0]))
        if alpha_start < 0:
            alpha_start += 2 * np.pi

        alpha_end = float(np.arctan2(dir_end[1], dir_end[0]))
        if alpha_end < 0:
            alpha_end += 2 * np.pi

        edge_index[0, i] = u
        edge_index[1, i] = v
        edge_tangents[i, 0] = alpha_start
        edge_tangents[i, 1] = norm_start
        edge_tangents[i, 2] = alpha_end
        edge_tangents[i, 3] = norm_end

    return {
        'vertices_cartesian': blocking_nodes,
        'vertices_polar': torch.stack([r, theta], dim=1),  # [n_verts, 2]
        'faces': blocking_faces,
        'edge_index': edge_index,
        'edge_tangents': edge_tangents,
        'edge_to_streamline': edge_to_streamline,
        'tri_coordinates': tri_coordinates,
        'center': center,
        'bounds': bounds,
    }


def main(input_path: str = '/root/repos/meshtron/checkpoint_mesh_100.pt',
         output_path: str = '/root/repos/meshtron/domain_data.pt'):
    print(f"Loading meshes from {input_path}")
    meshes = torch.load(input_path, weights_only=False)
    if not isinstance(meshes, list):
        meshes = [meshes]

    print(f"Processing {len(meshes)} meshes...")
    processed = []
    for i in tqdm(range(len(meshes))):
        mesh = meshes[i]
        data = extract_mesh_data(mesh)
        processed.append(data)

    torch.save(processed, output_path)
    print(f"Saved preprocessed data to {output_path}")

    # Quick validation print
    d = processed[0]
    print(f"\nSample mesh stats:")
    print(f"  n_verts: {d['vertices_polar'].shape[0]}")
    print(f"  n_faces: {d['faces'].shape[1]}")
    print(f"  n_edges: {d['edge_index'].shape[1]}")
    print(f"  center: {d['center']}")
    print(f"  bounds: {d['bounds']}")


if __name__ == '__main__':
    main()
