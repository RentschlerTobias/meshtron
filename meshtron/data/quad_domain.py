"""quad_domain.py — load the 2D quad-domain dataset into the shape MeshData wants.

The 2D corpus lives outside this repo, one file per batch of meshes:

    /home/t1dde/Duty/projects/meshtron/quad_domain_data/checkpoint_mesh_*.pt

Each file is a list of torch_geometric Data objects holding the same problem as
the 3D pipeline, one dimension down, plus the thing 3D has to infer:

    blocking_nodes    (12, 2)   the coarse blocking -- what gets tokenized
    blocking_faces    (4, 6)    six quad blocks, four corners each
    tri_coordinates   (N, 3)    the domain triangulation, conditioning source
    quad_coordinates  (M, 3)    the refined quad mesh (the target mesh)
    frame_field_u     (N, 2)    the frame field, which 3D has no equivalent of
    streamlines, singularities

MeshData reads `.x`, `.faces` and `.tri_coordinates`, and the 3D quad files in
data/ already come in that form. The fields above are the same data under
different names, so loading is renaming and nothing more -- no resampling, no
reordering, no coordinate change.

    from meshtron.data.quad_domain import load_quad_domain
    meshes = load_quad_domain(limit=8)          # default directory
    meshes = load_quad_domain("/some/where", limit=8)
"""
from __future__ import annotations

import glob
import os

import torch

DEFAULT_DIR = "/home/t1dde/Duty/projects/meshtron/quad_domain_data"
# blocking_nodes is 2D; tri_coordinates carries a third column that is constant
# for a planar domain. MeshData slices to [:, :dim] itself, so both stay whole.
REQUIRED = ("blocking_nodes", "blocking_faces", "tri_coordinates")


class QuadDomainMesh:
    """The three fields MeshData reads, under the names it reads them by.

    Deliberately not a torch_geometric Data: attribute assignment on Data
    carrying these keys is what would rename the originals in place. Keeping the
    adapter separate means the source objects are never mutated.
    """

    __slots__ = ("x", "faces", "tri_coordinates", "source", "index")

    def __init__(self, raw, source: str = "", index: int = 0):
        self.x = raw["blocking_nodes"]
        self.faces = raw["blocking_faces"]
        self.tri_coordinates = raw["tri_coordinates"]
        self.source = source
        self.index = index

    def __repr__(self) -> str:
        return (f"QuadDomainMesh({tuple(self.x.shape)} nodes, "
                f"{self.faces.shape[1]} blocks, "
                f"{self.tri_coordinates.shape[0]} tri points, "
                f"{os.path.basename(self.source)}#{self.index})")


def quad_domain_files(root: str = DEFAULT_DIR) -> list[str]:
    """The batch files, sorted, so a limit always selects the same meshes."""
    return sorted(glob.glob(os.path.join(root, "checkpoint_mesh_*.pt")))


def load_quad_domain(root: str = DEFAULT_DIR, limit: int = 0,
                     max_files: int = 0) -> list[QuadDomainMesh]:
    """Meshes from the 2D corpus, adapted for MeshData.

    limit      stop after this many meshes (0 = all; the corpus is 5.2 GB, so
               a limit is the normal case)
    max_files  stop after this many batch files (0 = as many as limit needs)

    Skips any entry missing one of the three fields instead of failing the
    whole load, and says how many were skipped.
    """
    files = quad_domain_files(root)
    if not files:
        raise FileNotFoundError(
            f"no checkpoint_mesh_*.pt under {root} -- the 2D corpus is not in "
            f"this repo; pass its directory as root")
    out: list[QuadDomainMesh] = []
    skipped = 0
    for fi, path in enumerate(files):
        if max_files and fi >= max_files:
            break
        batch = torch.load(path, weights_only=False)
        for i, raw in enumerate(batch):
            keys = raw.keys() if hasattr(raw, "keys") else ()
            if any(k not in keys for k in REQUIRED):
                skipped += 1
                continue
            out.append(QuadDomainMesh(raw, path, i))
            if limit and len(out) >= limit:
                break
        if limit and len(out) >= limit:
            break
    if skipped:
        print(f"note: skipped {skipped} entries missing one of {REQUIRED}")
    if not out:
        raise RuntimeError(f"no usable mesh in {root} (skipped {skipped})")
    return out


if __name__ == "__main__":
    ms = load_quad_domain(limit=4)
    print(f"{len(ms)} meshes from {len(quad_domain_files())} files")
    for m in ms:
        print(" ", m)
