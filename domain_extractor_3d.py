"""
domain_extractor_3d.py

Converts the tistos 3D domain-partition samples (raw, neutral geometry --
see the exporter's own docstring in
domain_partition_3D/experimentell/hex3d_algohex/export_sample.py, "this
repository writes neutral geometry, meshtron owns the ML format") into the
two preprocessed datasets the 3D pipeline consumes:

    quadtron_data_3d.pt  -- list[torch_geometric.data.Data], one per sample,
                             same .x/.faces/.tri_coordinates/.dir_class shape
                             as the existing 2D blade dataset (just dim=3
                             columns), consumed directly by dataset.py:MeshData.
    polytron_data_3d.pt  -- list[dict], one per sample, same key layout as the
                             existing 2D domain_data.pt (vertices_polar,
                             vertices_cartesian, faces, edge_index, edge_ctrl,
                             edge_to_streamline, center), consumed by
                             polytron_chain.py / polytron_tokenizer.py.

Coordinate frame: cylindrical (r, theta, z) about the z-axis, theta =
atan2(y, x), no re-centering. This is not a guess -- it matches the existing
convention already used for this same turbomachinery data one level up in
the pipeline: domain_partition_3D/dp3d/unwrap_surface.py:8 does
`theta = atan2(y, x)` for "a cylindrical turbine surface (hub or shroud)".
`sample.npz`'s own `params` dict is empty (no axis hint shipped), so this
convention is inherited rather than re-derived from the file itself.

Per-face `dir_class` (Quadtron row-grouping signal, see
tokenizer_v2.Tokenizer2D._order_quads_by_dir_class): the raw `dir_class` array
in sample.npz is per-EDGE (one entry per directed edge, 214 of them), not
per-face. There is no single canonical direction for a quad face -- each face
has two edge-direction pairs (e.g. one "row" direction, one "column"
direction). This script uses a simple, deterministic heuristic: each face's
row-key = the dir_class of its first edge (corner 0 -> corner 1). This groups
faces that start off in the same structured direction; it is NOT a verified
topological row-strip discovery (that would need graph traversal comparable
to half_edge.py's 2D sweep, adapted for 3D direction classes) -- flagged here
as a documented approximation, not a guess passed off as ground truth.

Usage:
    python domain_extractor_3d.py --src ../domain_partition_3D/data/tistos_domain_partition \\
        --out-quadtron quadtron_data_3d.pt --out-polytron polytron_data_3d.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm


def _edge_lookup(edges: np.ndarray) -> dict:
    """(u,v) -> index into `edges` [E,2]."""
    return {(int(u), int(v)): e for e, (u, v) in enumerate(edges)}


def subsample_points(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random subsample, capped at `max_points`. Needed because
    Polytron's VertexGen (polytron_vertex_model.py) runs full self-attention
    directly over `tri_coordinates` with NO subsampling of its own (unlike
    Quadtron's dataset.py:MeshData.get_point_cloud, which subsamples at load
    time) -- the raw tistos `surface_points` is a full triangulated surface
    (up to ~12k points per sample), and self-attention memory is O(N^2): fed
    raw, this allocated 43GB on the very first training batch. The 2D
    pipeline never hit this because its tri_coordinates were already sparse
    (a few hundred points) coming out of domain_extractor.py."""
    if points.shape[0] <= max_points:
        return points
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def to_cylindrical(vertices_cart: np.ndarray) -> np.ndarray:
    """[N,3] cartesian (x,y,z) -> [N,3] cylindrical (r, theta, z), axis=z,
    theta=atan2(y,x). See module docstring for why this axis/convention."""
    x, y, z = vertices_cart[:, 0], vertices_cart[:, 1], vertices_cart[:, 2]
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    return np.stack([r, theta, z], axis=1)


def per_face_dir_class(quad_faces: np.ndarray, edges: np.ndarray,
                        dir_class: np.ndarray, lut: dict) -> np.ndarray:
    """[F] row-key per quad face -- dir_class of the face's first edge
    (corner 0 -> corner 1). See module docstring: documented heuristic, not a
    verified per-face topological row assignment."""
    out = np.zeros(quad_faces.shape[0], dtype=np.int64)
    for fi, face in enumerate(quad_faces):
        u, v = int(face[0]), int(face[1])
        e = lut.get((u, v))
        if e is None:
            e = lut.get((v, u))  # fallback: reversed twin carries the same class
        out[fi] = int(dir_class[e]) if e is not None else -1
    return out


def edge_to_streamline_from_polyline(edges: np.ndarray, edge_polyline: np.ndarray,
                                     edge_polyline_offset: np.ndarray) -> dict:
    """Rebuilds the {(u,v): [N,3] polyline} dict from the CSR-encoded
    edge_polyline/edge_polyline_offset arrays, mirroring edge_to_streamline's
    role in the 2D pipeline (ground truth for round-trip geometry error)."""
    out = {}
    for e, (u, v) in enumerate(edges):
        s, t = int(edge_polyline_offset[e]), int(edge_polyline_offset[e + 1])
        out[(int(u), int(v))] = edge_polyline[s:t]
    return out


def load_sample(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def build_quadtron_sample(s: dict) -> Data:
    """Cartesian vertices + boundary quad faces -- Quadtron/tokenizer_v2
    operates directly on cartesian coordinates (unlike Polytron, no polar
    conversion here), same as the existing 2D blade dataset."""
    vertices = torch.tensor(s['vertices'], dtype=torch.float32)          # [N,3]
    faces = torch.tensor(s['quad_faces'].T, dtype=torch.long)            # [4,F]
    tri_coordinates = torch.tensor(s['surface_points'], dtype=torch.float32)  # [Np,3]

    edges = s['edges']
    lut = _edge_lookup(edges)
    dc = per_face_dir_class(s['quad_faces'], edges, s['dir_class'], lut)
    dir_class = torch.tensor(dc, dtype=torch.long)                       # [F]

    return Data(x=vertices, faces=faces, tri_coordinates=tri_coordinates,
                dir_class=dir_class)


def build_polytron_sample(s: dict, rng: np.random.Generator, max_tri_points: int = 768) -> dict:
    """Cylindrical vertices + hex-block pointer targets + directed half-edge
    control points, same dict-key layout as the 2D domain_data.pt so
    polytron_chain.py / polytron_tokenizer.py need no further special-casing
    beyond PolytronTokenizer(dim=3, corners_per_block=8)."""
    vertices_cart = s['vertices'].astype(np.float64)
    vertices_polar = to_cylindrical(vertices_cart)

    # 'tri_coordinates' is the name polytron_vertex_model.py/polytron_geom_model.py
    # actually read (matches the 2D convention) -- pre-subsampled, see
    # subsample_points()'s docstring for why this (unlike Quadtron's raw,
    # dynamically-subsampled tri_coordinates) has to happen at extraction time.
    tri_coordinates = torch.tensor(
        subsample_points(s['surface_points'], max_tri_points, rng), dtype=torch.float32)

    return {
        'vertices_polar': torch.tensor(vertices_polar, dtype=torch.float32),    # [M,3]
        'vertices_cartesian': torch.tensor(vertices_cart, dtype=torch.float32),  # [M,3]
        'faces': torch.tensor(s['blocks'].T, dtype=torch.long),                 # [8,F]
        'edge_index': torch.tensor(s['edges'].T, dtype=torch.long),             # [2,E]
        'edge_ctrl': torch.tensor(s['edge_ctrl'], dtype=torch.float32),         # [E,2,3]
        'edge_to_streamline': edge_to_streamline_from_polyline(
            s['edges'], s['edge_polyline'], s['edge_polyline_offset']),
        'center': torch.tensor([0.0, 0.0, 0.0]),   # no re-centering, see module docstring
        'quad_faces': torch.tensor(s['quad_faces'].T, dtype=torch.long),        # [4,F'] boundary shell, kept for reference/plots
        'tri_coordinates': tri_coordinates,                                      # subsampled, max_tri_points
        'surface_points': torch.tensor(s['surface_points'], dtype=torch.float32),  # full, for reference/plots
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path,
                    default=Path('../domain_partition_3D/data/tistos_domain_partition'),
                    help='Directory containing one subfolder per sample (each with sample.npz).')
    ap.add_argument('--out-quadtron', type=Path, default=Path('quadtron_data_3d.pt'))
    ap.add_argument('--out-polytron', type=Path, default=Path('polytron_data_3d.pt'))
    ap.add_argument('--max-tri-points', type=int, default=768,
                    help="Cap on Polytron's tri_coordinates point count (its VertexGen "
                         "runs full self-attention over this with no subsampling of its "
                         "own -- see subsample_points()'s docstring).")
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    sample_paths = sorted(args.src.glob('*/sample.npz'))
    print(f"Found {len(sample_paths)} sample.npz files under {args.src}")

    quadtron_samples, polytron_samples = [], []
    n_failed = 0
    for p in tqdm(sample_paths, desc='extracting'):
        try:
            s = load_sample(p)
            quadtron_samples.append(build_quadtron_sample(s))
            polytron_samples.append(build_polytron_sample(s, rng, args.max_tri_points))
        except Exception as e:
            n_failed += 1
            print(f"  [SKIP] {p.parent.name}: {type(e).__name__}: {e}")

    print(f"\nExtracted {len(quadtron_samples)}/{len(sample_paths)} samples "
          f"({n_failed} failed/skipped).")

    torch.save(quadtron_samples, args.out_quadtron)
    torch.save(polytron_samples, args.out_polytron)
    print(f"Wrote {args.out_quadtron} ({len(quadtron_samples)} samples)")
    print(f"Wrote {args.out_polytron} ({len(polytron_samples)} samples)")

    # Bounds for PolytronTokenizer(r_bounds=..., z_bounds=...) -- these are FIXED,
    # dataset-wide bounds (see PolytronTokenizer's own fit_bounds() docstring for
    # why: the model can't dequantize per-mesh bounds it never saw at inference
    # time). Measured here since dim=3 has no edge_tangents for the existing
    # fit_bounds() classmethod to key off; printed for the caller to pass in.
    if polytron_samples:
        all_r = np.concatenate([d['vertices_polar'][:, 0].numpy() for d in polytron_samples])
        all_z = np.concatenate([d['vertices_polar'][:, 2].numpy() for d in polytron_samples])
        pad_r = (all_r.max() - all_r.min()) * 0.02
        pad_z = (all_z.max() - all_z.min()) * 0.02
        r_bounds = (float(all_r.min() - pad_r), float(all_r.max() + pad_r))
        z_bounds = (float(all_z.min() - pad_z), float(all_z.max() + pad_z))
        print(f"\nMeasured bounds for PolytronTokenizer(dim=3, ...):")
        print(f"  r_bounds={r_bounds}")
        print(f"  z_bounds={z_bounds}")


if __name__ == '__main__':
    main()
