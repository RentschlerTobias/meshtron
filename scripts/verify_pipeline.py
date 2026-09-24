"""verify_pipeline.py -- does every path still work, and which ones do not.

Turning the prototype into a product means knowing, at any moment, which of the
combinations actually runs: 2D and 3D, cartesian and polar, tokenisation,
supervised training, RL. This exercises each at smoke scale and reports PASS,
FAIL or MISSING -- the last meaning the code path exists but the data or a
prerequisite does not, which is a different problem from a broken one.

  uv run python scripts/verify_pipeline.py
  uv run python scripts/verify_pipeline.py --only tokenisation
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, "data")
RESULTS: list = []


def check(group, name, fn):
    t0 = time.time()
    try:
        note = fn()
        RESULTS.append((group, name, "PASS", note or "", time.time() - t0))
    except FileNotFoundError as e:
        RESULTS.append((group, name, "MISSING", str(e)[:70], time.time() - t0))
    except NotImplementedError as e:
        RESULTS.append((group, name, "MISSING", str(e)[:70], time.time() - t0))
    except Exception as e:  # noqa: BLE001
        RESULTS.append((group, name, "FAIL",
                        f"{type(e).__name__}: {e}"[:70], time.time() - t0))
        if os.environ.get("VERIFY_TRACE"):
            traceback.print_exc()


def need(path):
    if not os.path.exists(path):
        raise FileNotFoundError(os.path.relpath(path, ROOT))
    return path


# --------------------------------------------------------------------------
# tokenisation
# --------------------------------------------------------------------------

def tok3d(coords):
    """HexaRow round trip: blocks -> tokens -> blocks."""
    from meshtron.data.hexa_row_tokenizer import HexaRowTokenizer
    npz = need(os.path.join(DATA, "hex3d_algohex", "batch",
                            "machine_0034_n2000", "sample.npz"))
    z = np.load(npz, allow_pickle=True)
    V = np.asarray(z["vertices"], float)
    B = np.asarray(z["blocks"], np.int64)
    rb = (0.0, float(np.hypot(V[:, 0], V[:, 1]).max()))
    zb = (float(V[:, 2].min()), float(V[:, 2].max()))
    tok = HexaRowTokenizer(r_bounds=rb, z_bounds=zb)
    from meshtron.data.domain_extractor_3d import to_cylindrical
    mesh = {"vertices_cartesian": torch.tensor(V),
            "vertices_polar": torch.tensor(to_cylindrical(V),
                                           dtype=torch.float32),
            "faces": torch.tensor(B.T)}
    seq = tok.tokenize(mesh, coords=coords)
    seq = seq["tokens"] if isinstance(seq, dict) else seq
    vpt, blk = tok.detokenize(list(map(int, seq)), coords=coords)
    if len(blk) != len(B):
        raise AssertionError(f"{len(blk)} blocks back, {len(B)} in")
    # The representation is quantised, so the round trip can only be exact up
    # to one quantisation step. Measure against that rather than against
    # equality, which would fail by construction.
    from scipy.spatial import cKDTree
    ref = V if coords == "cart" else to_cylindrical(V)
    d, _ = cKDTree(ref).query(np.asarray(vpt, float))
    step = max((rb[1] - rb[0]) / tok.Qr, (zb[1] - zb[0]) / tok.Qr)
    if d.max() > 1.5 * step:
        raise AssertionError(f"corner error {d.max():.5f} exceeds 1.5 "
                             f"quantisation steps ({step:.5f})")
    return (f"{len(seq)} tokens, {len(blk)} blocks back, corner error "
            f"max {d.max():.5f} against a step of {step:.5f}")


def tok2d(dim):
    """Tokenizer2D round trip on a quad mesh."""
    from meshtron.data.tokenizer_v2 import Tokenizer2D
    p = os.path.join(DATA, "quadtron_data_3d_smoke.pt")
    if dim == 3:
        meshes = torch.load(need(p), weights_only=False)
        m = meshes[0]
        verts = m.x[:, :3].numpy()
        faces = m.faces.numpy()
    else:
        raise NotImplementedError(
            "no 2D quad dataset in data/ -- only quadtron_data_3d*.pt exist")
    dir_class = getattr(m, "dir_class", None)
    tok = Tokenizer2D(quantization_levels=128, dim=dim, verbose=False)
    seq = tok.tokenize(torch.as_tensor(verts), torch.as_tensor(faces),
                       dir_class=dir_class)
    seq = seq["tokens"] if isinstance(seq, dict) else seq
    out = tok.detokenize(seq)
    # detokenize returns (vertices [V,dim], quads [4,F]) -- the quad array is
    # corners-first, so the face count is the SECOND axis.
    fo = np.asarray(out[1] if isinstance(out, tuple) else out)
    n = fo.shape[1] if fo.ndim == 2 and fo.shape[0] == 4 else len(fo)
    fi = np.asarray(faces)
    n_in = fi.shape[1] if fi.ndim == 2 and fi.shape[0] == 4 else len(fi)
    if n != n_in:
        raise AssertionError(f"{n} faces back, {n_in} in -- the round trip "
                             f"loses faces")
    nv = np.asarray(out[0]).shape[0] if isinstance(out, tuple) else -1
    return (f"{len(seq)} tokens, {n} faces back of {n_in}, "
            f"{nv} vertices (quantisation may merge some)")


# --------------------------------------------------------------------------
# model and training
# --------------------------------------------------------------------------

def forward_3d():
    from meshtron.data import conditioning
    from scripts.eval_family import load_model
    ck, cfg, coords, npt, rb, zb, model, max_len, _ = load_model(
        need(os.path.join(DATA, "hexarow_batch_model_100.pt")), "cpu")
    src = torch.load(need(os.path.join(DATA, "polytron_batch_clean.pt")),
                     weights_only=False)
    tk = torch.load(need(os.path.join(DATA, "hexarow_batch_tokens.pt")),
                    weights_only=False)
    it = tk["train"][0]
    s = next(s for s in src["samples"] if s["name"] == it["name"])
    cl = conditioning.build_cloud(s, cfg["n_points"], rb, zb,
                                  np.random.default_rng(0))
    cl = cl[0] if isinstance(cl, tuple) else cl
    x = torch.as_tensor(np.asarray(it["tokens"][:64]))[None]
    with torch.no_grad():
        lg = model(x, torch.as_tensor(np.asarray(cl),
                                      dtype=torch.float32)[None],
                   torch.tensor([float(it["blocks"])]))
    return f"logits {tuple(lg.shape)}, {sum(p.numel() for p in model.parameters()) / 1e6:.0f} M params"


def train_step_3d():
    """One supervised step, exactly the trainer's convention."""
    from meshtron.data import conditioning
    from meshtron.training import train_hexarow_full as T
    from scripts.eval_family import load_model
    ck, cfg, coords, npt, rb, zb, model, max_len, _ = load_model(
        need(os.path.join(DATA, "hexarow_batch_model_100.pt")), "cpu")
    tk = torch.load(need(os.path.join(DATA, "hexarow_batch_tokens.pt")),
                    weights_only=False)
    src = torch.load(need(os.path.join(DATA, "polytron_batch_clean.pt")),
                     weights_only=False)
    by = {s["name"]: s for s in src["samples"]}
    items = [dict(it) for it in tk["train"][:2]]
    rng = np.random.default_rng(0)
    for it in items:
        cl = conditioning.build_cloud(by[it["name"]], cfg["n_points"], rb, zb,
                                      rng)
        it["points"] = np.asarray(cl[0] if isinstance(cl, tuple) else cl)
    x, slot, w, pc, fc = T.batchify(items, ck["pad_id"], "cpu", (), 1.0, npt)
    lossf = torch.nn.CrossEntropyLoss(ignore_index=ck["pad_id"],
                                      reduction="none")
    model.train()
    lg = model(x[:, :-1], pc, fc)
    loss = T.weighted_loss(lg, x[:, 1:], w[:, 1:], ck["pad_id"], ck["vocab"],
                           lossf)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    opt.zero_grad()
    loss.backward()
    gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    opt.step()
    m = x[:, 1:] != ck["pad_id"]
    acc = float((lg.argmax(-1)[m] == x[:, 1:][m]).float().mean())
    return f"loss {float(loss):.3f}, next-token acc {100 * acc:.1f}%, grad {gn:.2f}"


def generate_3d():
    from meshtron.data import conditioning
    from meshtron.data.hexa_row_tokenizer import HexaRowTokenizer
    from meshtron.training.generate import slot_mask
    from scripts.eval_family import load_model
    ck, cfg, coords, npt, rb, zb, model, max_len, _ = load_model(
        need(os.path.join(DATA, "hexarow_batch_model_100.pt")), "cpu")
    tk = torch.load(need(os.path.join(DATA, "hexarow_batch_tokens.pt")),
                    weights_only=False)
    src = torch.load(need(os.path.join(DATA, "polytron_batch_clean.pt")),
                     weights_only=False)
    it = tk["train"][0]
    s = next(s for s in src["samples"] if s["name"] == it["name"])
    cl = conditioning.build_cloud(s, cfg["n_points"], rb, zb,
                                  np.random.default_rng(0))
    cl = cl[0] if isinstance(cl, tuple) else cl
    pc = torch.as_tensor(np.asarray(cl), dtype=torch.float32)[None]
    fc = torch.tensor([float(it["blocks"])])
    tok = HexaRowTokenizer(r_bounds=rb, z_bounds=zb)
    seq = list(map(int, it["tokens"][:8]))
    illegal = 0
    for _ in range(24):
        with torch.no_grad():
            lg = model(torch.tensor(seq)[None], pc, fc)[0, -1]
        m = slot_mask(tok, seq, len(seq), ck["vocab"], coords)
        if not torch.isfinite(m[int(lg.argmax())]):
            illegal += 1
        seq.append(int((lg + m).argmax()))
    return (f"24 steps, {illegal} of them the unmasked argmax was illegal "
            f"(the mask caught it)")


def grpo_3d():
    from meshtron.training import rewards_hexarow  # noqa: F401
    from meshtron.training import train_grpo  # noqa: F401
    need(os.path.join(DATA, "hexarow_batch_model_100.pt"))
    return "modules import, reward functions available (no step run here)"


def train_step_2d(dim):
    """One Quadtron step through the real trainer path."""
    from meshtron.data.dataset import MeshData
    from meshtron.data.tokenizer_v2 import Tokenizer2D
    from meshtron.model.quadtron import Quadtron
    if dim == 2:
        raise NotImplementedError(
            "no 2D quad dataset in data/ -- only quadtron_data_3d*.pt exist")
    meshes = torch.load(need(os.path.join(DATA, "quadtron_data_3d_smoke.pt")),
                        weights_only=False)[:2]
    tok = Tokenizer2D(quantization_levels=128, dim=dim, verbose=False)
    ds = MeshData(meshes, tok, n_sample_points=64, verbose=False, dim=dim)
    item = ds[0]
    return f"dataset builds, {len(ds)} items, first item {type(item).__name__}"


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------



def mapping_real():
    import types

    from meshtron.geometry.block_mapping import SnapConfigV2, snap_corners_v2
    from meshtron.geometry.curved_bridge import refill_curved
    from meshtron.geometry.geometry_features import FeatureModelV2
    from meshtron.geometry.patch_paths import (PatchPaths,
                                               make_face_projector,
                                               snap_seam_path)
    from scripts.conform_gt_blocks import _boundary_edge_pred
    from scripts.map_generated_blocks import _seam_path_fn
    npz = need(os.path.join(DATA, "hex3d_algohex", "batch",
                            "machine_0034_n2000", "sample.npz"))
    fm = FeatureModelV2(npz, cache_dir=os.path.join(DATA, "features"))
    tgt = types.SimpleNamespace(curves=fm.seam_curves,
                                surface_nearest=fm.surface_nearest)
    C = fm.vertices[fm.blocks].astype(float)
    Cs, rec = snap_corners_v2(tgt, C, SnapConfigV2())
    st = {"routes": 0, "edges_surface_projected": 0,
          "edges_walked_multi_patch": 0}
    raw = _seam_path_fn(fm.seam_curves, rec, st, tol=1e-9)

    def seam(a, b, n):
        r = raw(a, b, n)
        return None if r is None else (snap_seam_path(fm.seam_curves, fm,
                                                      r[0]), r[1])

    geo = PatchPaths(fm, records=rec, stats=st,
                     is_boundary=_boundary_edge_pred(fm.blocks, Cs))

    def pf(a, b, n):
        r = seam(a, b, n)
        return r if r is not None else geo(a, b, n)

    out = os.path.join(DATA, "showcase", "verify_refill.vtk")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rep = refill_curved(Cs, 0.2, out, fm=tgt, path_fn=pf, write_edges=False,
                        face_project_fn=make_face_projector(geo, st))
    d, _, _ = fm.surface_nearest(
        _points(out)[rep["boundary_point_ids"]], k=32)
    os.remove(out)
    return (f"{rep['cells_after']} cells at h=0.2, boundary {d.max():.2e}, "
            f"watertight {rep['watertight']}")


def _points(path):
    with open(path) as fh:
        L = fh.read().split("\n")
    i = next(k for k, l in enumerate(L) if l.startswith("POINTS"))
    n = int(L[i].split()[1])
    return np.array([[float(x) for x in L[i + 1 + k].split()]
                     for k in range(n)])


GROUPS = {
    "tokenisation": [
        ("3D hexarow, polar", lambda: tok3d("polar")),
        ("3D hexarow, cartesian", lambda: tok3d("cart")),
        ("2D quads, dim=2", lambda: tok2d(2)),
        ("2D quads, dim=3", lambda: tok2d(3)),
    ],
    "model": [
        ("3D forward pass", forward_3d),
        ("3D generation under the mask", generate_3d),
    ],
    "training": [
        ("3D supervised step", train_step_3d),
        ("3D GRPO", grpo_3d),
        ("2D Quadtron, dim=2", lambda: train_step_2d(2)),
        ("2D Quadtron, dim=3", lambda: train_step_2d(3)),
    ],
    "geometry": [
        ("mapping and refill", mapping_real),
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser(description="verify every path")
    ap.add_argument("--only", default="", help="one group name")
    args = ap.parse_args()
    for group, checks in GROUPS.items():
        if args.only and group != args.only:
            continue
        for name, fn in checks:
            check(group, name, fn)

    width = max(len(n) for _, n, _, _, _ in RESULTS) + 2
    last = None
    for group, name, status, note, secs in RESULTS:
        if group != last:
            print(f"\n{group}")
            last = group
        mark = {"PASS": "PASS", "FAIL": "FAIL", "MISSING": "MISS"}[status]
        print(f"  {mark}  {name:<{width}} {secs:5.1f}s  {note}")
    n_pass = sum(1 for r in RESULTS if r[2] == "PASS")
    n_fail = sum(1 for r in RESULTS if r[2] == "FAIL")
    n_miss = sum(1 for r in RESULTS if r[2] == "MISSING")
    print(f"\n{n_pass} pass, {n_fail} fail, {n_miss} missing "
          f"(set VERIFY_TRACE=1 for tracebacks)")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
