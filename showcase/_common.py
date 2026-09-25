"""Shared helpers for the showcase scripts.

Nothing here is part of the pipeline -- it only loads things, prints them in a
readable shape and writes VTK, so the walkthrough scripts stay about the
pipeline itself.
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

HEX3D = (
    "/home/t1dde/hydrostack_pipeline/stack/domain_partition_3D/"
    "experimentell/hex3d_algohex"
)
if HEX3D not in sys.path:
    sys.path.insert(0, HEX3D)

DATA = os.path.join(ROOT, "data")
BATCH = os.path.join(DATA, "hex3d_algohex", "batch")
OUT = os.path.join(DATA, "showcase")

# A machine that carries every artifact and has no block-level T-junctions.
MACHINE = "machine_0034_n2000"
# The pair used throughout: a coarse-level model and the tokens it was trained
# on. Its cfg carries no "npt" and no "coords", which dates it before the slot
# embedding was added -- so it must be fed with slot=None (SLOT_TRAINED below).
# Feeding a slot it never saw costs about 30 points of next-token accuracy.
# data/hexarow_h05_model.pt is the newer model, trained on the h=0.5 subdivided
# blockings (68 blocks, 1689 tokens per sample) against
# data/hexarow_tokens_h05_family_cart.pt.
CKPT = os.path.join(DATA, "hexarow_batch_model_100.pt")
TOKENS = os.path.join(DATA, "hexarow_batch_tokens.pt")
SRC = os.path.join(DATA, "polytron_batch_clean.pt")
SLOT_TRAINED = False

PATCHES = {
    1: "inlet",
    2: "outlet",
    3: "periodic",
    4: "periodic",
    5: "hub",
    6: "shroud",
    7: "blade hull",
}


def head(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def show(name, obj, n: int = 3) -> None:
    """One line about an array, tensor, list or dict -- shape, dtype, range."""
    import torch

    if torch.is_tensor(obj):
        a = obj.detach().cpu().numpy()
        kind = "tensor"
    elif isinstance(obj, np.ndarray):
        a, kind = obj, "array"
    elif isinstance(obj, dict):
        print(f"{name:28s} dict, {len(obj)} keys: {sorted(map(str, obj))[:6]}")
        return
    elif isinstance(obj, (list, tuple)):
        print(
            f"{name:28s} {type(obj).__name__}, {len(obj)} entries, "
            f"first: {str(obj[0])[:44] if obj else '-'}"
        )
        return
    else:
        print(f"{name:28s} {type(obj).__name__}  {str(obj)[:48]}")
        return
    rng = (
        f"  [{a.min():.4g} .. {a.max():.4g}]"
        if a.size and np.issubdtype(a.dtype, np.number)
        else ""
    )
    print(f"{name:28s} {kind} {str(tuple(a.shape)):18s} {str(a.dtype):9s}{rng}")
    if a.ndim <= 2 and a.size and n:
        flat = a[:n] if a.ndim == 1 else a[:n]
        with np.printoptions(precision=4, suppress=True, linewidth=110):
            for row in np.atleast_1d(flat):
                print(f"{'':28s}   {row}")


def outdir() -> str:
    os.makedirs(OUT, exist_ok=True)
    return OUT


def write_hexes(path, P, H, arrays=None, title="showcase"):
    """Hex mesh as legacy VTK with optional cell arrays."""
    P = np.asarray(P, float)
    H = np.asarray(H, int)
    with open(path, "w") as fh:
        fh.write(f"# vtk DataFile Version 2.0\n{title}\nASCII\n")
        fh.write("DATASET UNSTRUCTURED_GRID\n")
        fh.write(f"POINTS {len(P)} double\n")
        for p in P:
            fh.write("%.9f %.9f %.9f\n" % tuple(p))
        fh.write(f"CELLS {len(H)} {9 * len(H)}\n")
        for c in H:
            fh.write("8 " + " ".join(str(int(x)) for x in c) + "\n")
        fh.write(f"CELL_TYPES {len(H)}\n")
        for _ in H:
            fh.write("12\n")
        if arrays:
            fh.write(f"CELL_DATA {len(H)}\n")
            for name, vals in arrays.items():
                v = np.asarray(vals)
                isint = np.issubdtype(v.dtype, np.integer)
                fh.write(f"SCALARS {name} {'int' if isint else 'double'} 1\n")
                fh.write("LOOKUP_TABLE default\n")
                for x in v:
                    fh.write(("%d\n" % x) if isint else ("%.9e\n" % x))
    print(f"wrote {path}  ({len(P)} points, {len(H)} cells)")


def write_points(path, P, arrays=None, title="showcase points"):
    P = np.asarray(P, float)[:, :3]
    with open(path, "w") as fh:
        fh.write(f"# vtk DataFile Version 2.0\n{title}\nASCII\n")
        fh.write("DATASET UNSTRUCTURED_GRID\n")
        fh.write(f"POINTS {len(P)} double\n")
        for p in P:
            fh.write("%.9f %.9f %.9f\n" % tuple(p))
        fh.write(f"CELLS {len(P)} {2 * len(P)}\n")
        for i in range(len(P)):
            fh.write(f"1 {i}\n")
        fh.write(f"CELL_TYPES {len(P)}\n")
        for _ in range(len(P)):
            fh.write("1\n")
        if arrays:
            fh.write(f"POINT_DATA {len(P)}\n")
            for name, vals in arrays.items():
                v = np.asarray(vals)
                isint = np.issubdtype(v.dtype, np.integer)
                fh.write(f"SCALARS {name} {'int' if isint else 'double'} 1\n")
                fh.write("LOOKUP_TABLE default\n")
                for x in v:
                    fh.write(("%d\n" % x) if isint else ("%.9e\n" % x))
    print(f"wrote {path}  ({len(P)} points)")


def load_model(device="cpu"):
    """The trained GPTCond exactly as generation loads it."""
    from scripts.eval_family import load_model as _lm

    return _lm(CKPT, device)


def cloud_to_xyz(cloud, rb, zb):
    """Model-space cloud (r', sin, cos, z') back to xyz in machine coordinates
    (inverse of _normalize in meshtron.data.conditioning)."""
    r = cloud[:, 0] * (rb[1] - rb[0]) + rb[0]
    z = cloud[:, 3] * (zb[1] - zb[0]) + zb[0]
    return np.stack([r * cloud[:, 2], r * cloud[:, 1], z], axis=-1)


def scaled_jacobians(P, H):
    import clean_blocks as cb

    return cb.scaled_jacobians(np.asarray(P, float), np.asarray(H, int))
