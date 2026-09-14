from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
from matplotlib.path import Path
from tqdm import tqdm


class MeshData(Dataset):

    """
    Dataset für Quadtron
    """

    def __init__(self, meshes, tokenizer, max_seq_length=None, n_sample_points=1000, verbose=True, boundary_points_only=False, dim=2):
        """
        meshes: Liste von Mesh-Objekten
        tokenizer:  Tokenizer2D
        max_seq_length: Maximale Sequenzlänge (für Padding)
        dim: 2 oder 3 -- muss zu tokenizer.dim passen. Fuer dim=3 wird, falls vorhanden,
             mesh.dir_class an tokenizer.tokenize() durchgereicht (Row-Gruppierung, siehe
             Tokenizer2D._order_quads_by_dir_class).
        """
        assert dim == tokenizer.dim, f"dataset dim={dim} != tokenizer.dim={tokenizer.dim}"
        self.meshes = meshes
        self.tokenizer = tokenizer
        self.n_sample_points = n_sample_points
        self.dim = dim
        self.data = []
        self.face_count = []
        self.boundary_points_only = boundary_points_only
        if verbose == True:
            print(f"start tokenizing")
        for i in tqdm(range(len(meshes)), desc="meshes"):
            mesh = meshes[i]
            vertices = mesh.x[:, 0:dim]
            faces = mesh.faces
            dir_class = getattr(mesh, 'dir_class', None) if dim == 3 else None

            tokens = tokenizer.tokenize(vertices, faces, dir_class)
            num_faces = faces.size(1)

            self.data.append(tokens)
            self.face_count.append(num_faces)
        if max_seq_length is None:
            # self.max_seq_length = max(len(tokens) for tokens in self.data)
            self.max_seq_length = tokenizer.max_length_token_sequence
        else:
            self.max_seq_length = max_seq_length

        # self.min_seq_length = min(len(tokens) for tokens in self.data)
        self.min_seq_length = tokenizer.min_length_token_sequence
        print(
            f"\nMax Sequenzlänge: {self.max_seq_length}\nMin Sequenzlänge: {self.min_seq_length}")

    def get_point_cloud(self, mesh, n_sample_points):
        dim = self.dim
        all_coords = mesh.tri_coordinates[:, 0:dim]
        center = (all_coords.max(dim=0).values + all_coords.min(dim=0).values) / 2
        # uniform scale (same across axes) um Aspektverhältnis zu erhalten
        scale = (all_coords.max(dim=0).values - all_coords.min(dim=0).values).max().clamp(min=1e-6)

        if dim == 3:
            # 3D-Punktwolken (tistos surface_points) sind bereits die volle Oberflaeche --
            # kein Box/NACA-Loch-Split, kein Rejection-Sampling noetig. Nur sub-/upsamplen.
            n_points = all_coords.size(0)
            if n_points >= n_sample_points:
                idx = torch.randint(0, n_points, [n_sample_points])
                point_cloud = all_coords[idx, :]
            else:
                reps = (n_sample_points + n_points - 1) // n_points
                point_cloud = all_coords.repeat(reps, 1)[:n_sample_points]
            point_cloud = (point_cloud - center) / scale * 2
            return point_cloud

        mask = mesh.tri_coordinates[:, 2] != 2
        boundary_points = mesh.tri_coordinates[mask, 0:2]

        num_boundary_points = boundary_points.size(0)
        remaining = n_sample_points - num_boundary_points

        if num_boundary_points >= n_sample_points:
            random_idx = torch.randint(
                0, num_boundary_points, [n_sample_points])
            point_cloud = boundary_points[random_idx, :]
        else:
            # Trenne Box-Berandung von NACA-Berandung anhand der Box-Kanten.
            box_eps = 1e-4
            on_box = (
                (boundary_points[:, 0] <= box_eps)
                | (boundary_points[:, 0] >= 1 - box_eps)
                | (boundary_points[:, 1] <= box_eps)
                | (boundary_points[:, 1] >= 1 - box_eps)
            )
            naca_points = boundary_points[~on_box].numpy()

            # NACA-Polygon durch Winkelsortierung um den Centroid (star-convex bei Profilen).
            naca_polygon = None
            if naca_points.shape[0] >= 3:
                centroid = naca_points.mean(axis=0)
                angles = np.arctan2(
                    naca_points[:, 1] - centroid[1],
                    naca_points[:, 0] - centroid[0],
                )
                order = np.argsort(angles)
                naca_polygon = Path(naca_points[order])

            # Rejection-Sampling: uniform in [0,1]^2 minus NACA-Loch.
            kept = []
            n_kept = 0
            max_iter = 32
            for _ in range(max_iter):
                if n_kept >= remaining:
                    break
                batch = max(64, int((remaining - n_kept) * 1.5))
                cand = torch.rand(batch, 2)
                if naca_polygon is not None:
                    inside = naca_polygon.contains_points(cand.numpy())
                    cand = cand[~torch.from_numpy(inside)]
                kept.append(cand)
                n_kept += cand.size(0)

            interior_sampled = torch.cat(kept, dim=0)[:remaining]
            point_cloud = torch.cat([boundary_points, interior_sampled], dim=0)

        point_cloud = (point_cloud - center) / scale * 2  # -> [-1, 1] entlang längster Achse
        return point_cloud

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        point_cloud = self.get_point_cloud(self.meshes[idx], self.n_sample_points)
        num_faces = self.face_count[idx]

        pad_token = self.tokenizer.pad_token

        # Erstelle Input und Target für autoregressives Training
        if len(tokens) > self.max_seq_length:
            tokens = tokens[:self.max_seq_length]

        # Padding (auffuellen der token sequencen, damit alle die gleiche laenge haben)
        tokens_padded = tokens + [pad_token] * \
            (self.max_seq_length - len(tokens))
        tokens_tensor = torch.LongTensor(tokens_padded)

        # Input: alle außer letztes Token, Target: alle außer erstes Token
        input_tokens = tokens_tensor[:-1]
        target_tokens = tokens_tensor[1:]

        # Erstelle Padding Mask (wo NICHT gepaddet ist = 1)
        padding_mask = (input_tokens != pad_token).float()

        return {
            'input_tokens': input_tokens,
            'target_tokens': target_tokens,
            'padding_mask': padding_mask,
            'pad_token': pad_token,
            'point_cloud': torch.FloatTensor(point_cloud),
            'face_count': num_faces,
            'seq_length': len(tokens) - 1  # Tatsächliche Länge ohne Padding
        }
