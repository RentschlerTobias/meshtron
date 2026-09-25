"""Write the drawn conditioning cloud (with optional band oversample) as VTK."""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from meshtron.data import conditioning  # noqa: E402
from meshtron.geometry.geometry_features import FeatureModelV2  # noqa: E402
from scripts.eval_family import load_model  # noqa: E402
from scripts.map_generated_blocks import _resolve_item  # noqa: E402


def polar_to_xyz(p: np.ndarray) -> np.ndarray:
    r, th, z = p[:, 0], p[:, 1], p[:, 2]
    return np.stack([r * np.cos(th), r * np.sin(th), z], axis=-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default=os.path.join(ROOT, "data", "hexarow_tokens_family_cart.pt"))
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "data", "grpo_cart_step300.pt"))
    ap.add_argument("--idx", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--band-weight", type=float, default=None)
    ap.add_argument("--all", action="store_true",
                    help="write the COMPLETE surface cloud (all points, no draw)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--feature-cache", default=os.path.join(ROOT, "data", "features"))
    args = ap.parse_args()

    ck, cfg, coords, npt, rb, zb, model, max_len, missing = load_model(args.ckpt, "cpu")
    item = _resolve_item(args.tokens, args.idx)
    dir_ = str(item.get("dir")).split("/")[-1]
    npz_path = os.path.join(ROOT, "data", "hex3d_algohex", str(item.get("dir")), "sample.npz")
    fm = FeatureModelV2(npz_path, cache_dir=args.feature_cache)

    rng = np.random.default_rng(args.seed)
    sample = dict(item)
    is_band = conditioning.point_is_band(
        len(fm.surface_points), fm.surface_tris, fm.surface_tri_label)
    sample["is_band"] = is_band

    vp = conditioning.surface_cloud(sample)
    is_blade = np.asarray(item["is_blade"], dtype=bool)
    w = np.where(is_blade, 3.0, 1.0)
    if args.band_weight is not None:
        w = np.where(is_band, w * float(args.band_weight), w)
    n_curves = len(vp)
    tag = f"{args.band_weight if args.band_weight is not None else 0.0:.2f}"
    if args.all:
        idx = np.arange(n_curves)
        out = args.out or os.path.join(ROOT, "data", f"map_batch__{dir_}",
                                       f"cloud_full_band{tag}.vtk")
    else:
        n = int(cfg["n_points"])
        idx = rng.choice(n_curves, size=n, replace=True, p=w / w.sum())
        out = args.out or os.path.join(ROOT, "data", f"map_batch__{dir_}",
                                       f"cloud_band{tag}.vtk")
    cls = np.where(is_band[idx] & is_blade[idx], 2,
                   np.where(is_band[idx], 3, np.where(is_blade[idx], 1, 0)))
    xyz = polar_to_xyz(vp[idx])
    wsel = w[idx]
    n_pts = len(idx)
    with open(out, "w") as fh:
        fh.write("# vtk DataFile Version 3.0\nconditioning cloud (drawn)\n"
                 "ASCII\nDATASET POLYDATA\n")
        fh.write(f"POINTS {n_pts} double\n")
        for q in xyz:
            fh.write(f"{q[0]:.8f} {q[1]:.8f} {q[2]:.8f}\n")
        fh.write(f"VERTICES {n_pts} {2 * n_pts}\n")
        for i in range(n_pts):
            fh.write(f"1 {i}\n")
        fh.write(f"POINT_DATA {n_pts}\nSCALARS cls int 1\nLOOKUP_TABLE default\n")
        fh.write("\n".join(str(int(v)) for v in cls) + "\n")
        fh.write(f"SCALARS weight double 1\nLOOKUP_TABLE default\n")
        fh.write("\n".join(f"{v:.4f}" for v in wsel) + "\n")
    print(f"saved {out}  cls: 0=uniform 1=blade(hub-label) 2=band+blade 3=band "
          f"n={n_pts} band_weight={args.band_weight} all={args.all}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
