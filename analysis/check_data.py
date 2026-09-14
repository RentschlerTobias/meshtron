
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sys

import numpy as np
import torch


class MeshDataChecker:
    """
    Loads a list of meshes from a .pt file and runs a configurable pipeline
    of checks/fixes on each mesh. Extend by adding methods prefixed with
    `_check_` — they are discovered automatically in definition order.

    Each check method signature:
        _check_<name>(self, mesh_idx: int, x: Tensor, faces: Tensor)
            -> (new_x: Tensor, new_faces: Tensor, message: str | None)

    Returning None as the message means no issue was found for that mesh.
    """

    def __init__(self, path: str, decimals: int = 5):
        self.path = path
        self.decimals = decimals
        self.meshes = torch.load(path, weights_only=False)

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------

    def _check_duplicate_vertices(
        self, i: int, x: torch.Tensor, faces: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, str | None]:
        coords_np = np.round(x[:, 0:2].numpy(), self.decimals)

        coord_to_id: dict = {}
        remap = np.empty(len(coords_np), dtype=int)
        keep: list[int] = []

        for j, (cx, cy) in enumerate(coords_np):
            k = (cx, cy)
            if k not in coord_to_id:
                coord_to_id[k] = len(keep)
                keep.append(j)
            remap[j] = coord_to_id[k]

        n_removed = len(coords_np) - len(keep)
        if n_removed == 0:
            return x, faces, None

        new_x = x[keep]
        new_faces = torch.from_numpy(remap[faces.numpy()])
        return new_x, new_faces, (
            f'duplicate_vertices: removed {n_removed} '
            f'({len(x)} → {len(new_x)})'
        )

    def _check_isolated_vertices(
        self, i: int, x: torch.Tensor, faces: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, str | None]:
        n_verts = len(x)
        referenced = faces.flatten().unique()

        if len(referenced) == n_verts:
            return x, faces, None

        mask = torch.zeros(n_verts, dtype=torch.bool)
        mask[referenced] = True

        new_idx = torch.full((n_verts,), -1, dtype=torch.long)
        new_idx[mask] = torch.arange(mask.sum())

        new_x = x[mask]
        new_faces = new_idx[faces]
        n_removed = n_verts - len(new_x)
        return new_x, new_faces, (
            f'isolated_vertices: removed {n_removed} '
            f'({n_verts} → {len(new_x)})'
        )

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def _discover_checks(self):
        return [
            getattr(self, name)
            for name in type(self).__dict__
            if name.startswith('_check_')
        ]

    def run(self) -> list:
        checks = self._discover_checks()
        working = [(m.x, m.faces) for m in self.meshes]

        for check_fn in checks:
            updated = []
            for i, (x, faces) in enumerate(working):
                new_x, new_faces, msg = check_fn(i, x, faces)
                if msg:
                    print(f'[mesh {i}] {msg}')
                updated.append((new_x, new_faces))
            working = updated

        cleaned = []
        for mesh, (new_x, new_faces) in zip(self.meshes, working):
            m = mesh.clone()
            m.x = new_x
            m.faces = new_faces
            cleaned.append(m)

        print(f'done — {len(self.meshes)} meshes processed')
        return cleaned

    def save(self, cleaned: list, out_path: str | None = None) -> str:
        if out_path is None:
            out_path = self.path.replace('.pt', '_cleaned.pt')
        torch.save(cleaned, out_path)
        print(f'saved to {out_path}')
        return out_path


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else './centered_blades_cleaned.pt'
    checker = MeshDataChecker(path)
    cleaned = checker.run()
    checker.save(cleaned)
