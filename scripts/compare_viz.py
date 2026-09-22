"""Vergleichs-VTK: True-Mesh + generierte Hexe + Conditioning-Punktwolke.

Ein UNSTRUCTURED_GRID mit CELL_DATA 'part': 1=true, 2=generated, 3=punktwolke.
ParaView: Threshold(part) + Coloring je Teil.

Zwei Modi:
  Legacy (--seq): fertige Token-Sequenz detokenisieren (Default wie bisher).
  Generation (--ckpt --tokens --idx): Item aus train+val aufloesen, Conditioning
  via conditioning.build_cloud (blade_weight=3.0 Paritaet), Rollouts mit
  generate.generate() greedy (--k 1) oder stochastisch (--k>1).

Beispiele:
  uv run python scripts/compare_viz.py \
    --src data/polytron_data_3d_smoke.pt --idx 3 \
    --seq data/seq_overfit_polar.pt --tokens data/hexarow_overfit_1sample.pt \
    --out data/compare_overfit_polar.vtk

  uv run python scripts/compare_viz.py \
    --tokens data/hexarow_tokens_family_cart.pt \
    --ckpt data/grpo_cart_step300.pt --idx 687 --out data/gen_m0034.vtk

  uv run python scripts/compare_viz.py --ckpt data/grpo_cart_step300.pt \
    --tokens data/hexarow_tokens_family_cart.pt --idx 683 \
    --temperature 0.7 --k 2 --out data/gen_m0005.vtk
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import conditioning  # noqa: E402
from generate import (  # noqa: E402
    detokenize_safe,
    generate,
    load_sample,
    mesh_to_polar,
    sample_points,
)
from hexa_row_tokenizer import HexaRowTokenizer  # noqa: E402
from mesh_validation import validate_generated_mesh  # noqa: E402
from scripts.eval_family import load_model  # noqa: E402

LEGACY_SEQ = "data/seq_overfit_polar.pt"
LEGACY_OUT = "data/compare_overfit_polar.vtk"
LEGACY_IDX = 3


def _hexes_to_vtk_blocks(faces_t: np.ndarray) -> list[list[int]]:
    return [[int(i) for i in row] for row in faces_t]


def _to_cart(v: np.ndarray, coords: str) -> np.ndarray:
    """(r,theta,z) -> xyz; cart bleibt unveraendert."""
    return (v if coords == "cart"
            else np.stack([v[:, 0] * np.cos(v[:, 1]),
                           v[:, 0] * np.sin(v[:, 1]), v[:, 2]], axis=-1))


def _write_parts_vtk(path, parts, title) -> None:
    """Legacy-ASCII-VTK aus beliebig vielen Teilen (punkte, bloecke, part_id, cell_type).

    Ein Part = ein Satz Punkte + Hex-Zellen; die Punkteindizes der Zellen werden
    um den Punkt-Offset des jeweiligen Parts verschoben. CELL_DATA 'part' traegt
    die part_id, CELL_TYPES den VTK-Zelltyp (12=Hex, 1=Punkt). Mehrere Part-
    Bloecke koennen disjunkte Punktmengen sein (ParaView: Threshold auf part).
    """
    parts = [(np.asarray(p[0], dtype=np.float64), p[1], int(p[2]), int(p[3]))
             for p in parts]
    m = sum(len(p[0]) for p in parts)
    V = np.concatenate([p[0] for p in parts], axis=0)

    cells: list[str] = []
    types: list[str] = []
    scalars: list[str] = []
    base = 0
    for _, blocks, part, ctype in parts:
        for b in blocks:
            cells.append(f"{len(b)} " + " ".join(str(i + base) for i in b))
            types.append(str(ctype))
            scalars.append(str(part))
        base += len(next(p for p in parts if p[2] == part)[0])

    assert len(cells) == sum(len(p[1]) for p in parts), "zell-payment fehlerhaft"

    with open(path, "w") as fh:
        fh.write("# vtk DataFile Version 2.0\n")
        fh.write(f"{title}\n")
        fh.write("ASCII\nDATASET UNSTRUCTURED_GRID\n")
        fh.write(f"POINTS {m} double\n")
        for p in V:
            fh.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        total = sum(len(c.split()) for c in cells)
        fh.write(f"CELLS {len(cells)} {total}\n")
        fh.write("\n".join(cells) + "\n")
        fh.write(f"CELL_TYPES {len(types)}\n" + "\n".join(types) + "\n")
        fh.write(f"CELL_DATA {len(scalars)}\n")
        fh.write("SCALARS part int 1\nLOOKUP_TABLE default\n")
        fh.write("\n".join(scalars) + "\n")


def _write_compare_vtk(path, xyz, gt_blocks, gen_cart, gen_blocks, pc) -> None:
    """3-teiliges VTK schreiben (part 1=true, 2=generated, 3=punktwolke).
    Byte-identisch zum Legacy-Pfad (delegiert an _write_parts_vtk)."""
    parts = [(xyz, gt_blocks, 1, 12),
             (gen_cart, gen_blocks, 2, 12),
             (pc, [[k] for k in range(len(pc))], 3, 1)]
    _write_parts_vtk(path, parts, "meshtron compare (true|generated|pointcloud)")


def _run_legacy(args) -> int:
    """Unveraenderter Legacy-Pfad: --seq vorgegebene Token-Sequenz detokenisieren."""
    seq_path = args.seq if args.seq is not None else LEGACY_SEQ
    idx = LEGACY_IDX if args.idx is None else args.idx
    out = args.out if args.out is not None else LEGACY_OUT

    tk_pt = torch.load(args.tokens, map_location="cpu", weights_only=False)
    rb, zb = tuple(tk_pt["r_bounds"]), tuple(tk_pt["z_bounds"])
    coords = tk_pt.get("coords", "polar")

    src = torch.load(args.src, weights_only=False)
    obj = src["samples"] if isinstance(src, dict) and "samples" in src else src
    xyz, _nblocks, name, faces_t, surf = load_sample(obj, idx, with_surface=True)
    assert faces_t is not None, "sample ohne faces"
    gt_blocks = _hexes_to_vtk_blocks(np.asarray(faces_t, dtype=np.int64))

    seq = torch.load(seq_path, map_location="cpu", weights_only=False)
    seq = seq.tolist() if hasattr(seq, "tolist") else list(seq)
    tok = HexaRowTokenizer(r_bounds=rb, z_bounds=zb)
    res, trim = detokenize_safe(seq, tok, tok.core.stop_token, coords=coords)
    assert res is not None, f"detokenize fehlgeschlagen: {trim}"
    vpt, blk = res
    vnp = vpt.numpy()
    gen_cart = (vnp if coords == "cart"
                else np.stack([vnp[:, 0] * np.cos(vnp[:, 1]),
                               vnp[:, 0] * np.sin(vnp[:, 1]),
                               vnp[:, 2]], axis=-1))
    gen_blocks = [[int(i) for i in b] for b in blk.numpy()]

    rng = np.random.default_rng(0)
    cloud_xyz = surf if surf is not None else xyz
    pts = sample_points(mesh_to_polar(cloud_xyz), args.n_points, rb, zb, rng)
    r01, s01, c01, z01 = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    th = np.arctan2(s01, c01)
    r_real = rb[0] + r01 * (rb[1] - rb[0])
    pc = np.stack([r_real * np.cos(th),
                   r_real * np.sin(th),
                   zb[0] + z01 * (zb[1] - zb[0])], axis=-1)

    _write_compare_vtk(out, xyz, gt_blocks, gen_cart, gen_blocks, pc)

    print(f"mesh={name} coords={coords} | true blocks={len(gt_blocks)} "
          f"generated blocks={len(gen_blocks)} punktwolke={len(pc)}")
    if trim:
        print(f"note: seq wurde beim detokenize getrimmt: {trim}")
    print(f"saved {out}  (part: 1=true, 2=generated, 3=punktwolke)")
    return 0


def _run_generation(args, ap) -> int:
    """Generation-Modus: Item aufloesen, Rollouts sampeln, 3-teiliges VTK schreiben."""
    if args.idx is None:
        ap.error("--ckpt braucht --idx")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    ck, cfg, coords, npt, rb, zb, model, max_len, missing = load_model(args.ckpt, dev)
    if missing.missing_keys:
        print(f"note: fehlende CKPT-Keys: {missing.missing_keys}")
    tok = HexaRowTokenizer(r_bounds=rb, z_bounds=zb)
    core = tok.core
    stop_id = core.stop_token
    specials = {core.start_token, core.end_token, core.sep_token,
                core.sep2_token, core.stop_token, core.pad_token}
    use_slot = "slot.weight" in ck["model"]
    cap = max_len - 1

    ds = torch.load(args.tokens, weights_only=False)
    if "train" not in ds or "val" not in ds:
        ap.error("--tokens muss eine Familien-Tokenfile mit train/val sein")
    order = list(ds["train"]) + list(ds["val"])
    if not 0 <= args.idx < len(order):
        ap.error(f"--idx {args.idx} ausserhalb 0..{len(order) - 1}")
    item = order[args.idx]
    name = item.get("name", f"sample{args.idx}")
    blocks = int(item["blocks"])
    print(f"item idx={args.idx} name={name} blocks={blocks} coords={coords} "
          f"npt={npt} cap={cap} use_slot={use_slot}")

    # Conditioning exakt wie eval_family/generate: cfg-Werte, blade_weight=3.0.
    rng = np.random.default_rng(args.seed)
    pts, _ = conditioning.build_cloud(item, cfg["n_points"], rb, zb, rng,
                                      blade_weight=3.0)
    pc = torch.as_tensor(pts[None], dtype=torch.float32, device=dev)
    fc = torch.tensor([float(blocks)], device=dev)

    # True-Teil = detokisierte GT-Tokens (START/STOP wie im Legacy-Pfad behandelt).
    gres, gtrim = detokenize_safe(item["tokens"].tolist(), tok, stop_id,
                                  coords=coords)
    assert gres is not None, f"GT-detokenize fehlgeschlagen: {gtrim}"
    gvpt, gblk = gres
    gt_cart = _to_cart(gvpt.numpy(), coords)
    gt_blocks = [[int(i) for i in b] for b in gblk.numpy()]

    # Punktwolke = rohe surface_points. Das ist exakt die Quellmenge, aus der
    # conditioning.surface_cloud (via build_cloud) zieht; der ungewichtete Draw
    # aus DEMSELBEN rng (build_cloud hat oben schon konsumiert -> Paritaet bleibt)
    # zeigt dieselbe Geometrie, ohne die Blade-Gewichtung zu duplizieren.
    sp = item.get("surface_points")
    if sp is not None:
        sp = sp.detach().cpu().numpy() if hasattr(sp, "detach") else sp
        sp = np.asarray(sp, dtype=np.float64)
        sel = rng.choice(len(sp), size=cfg["n_points"],
                         replace=len(sp) < cfg["n_points"])
        pc_vis = sp[sel]
    else:
        th = np.arctan2(pts[:, 1], pts[:, 2])
        r_real = rb[0] + pts[:, 0] * (rb[1] - rb[0])
        pc_vis = np.stack([r_real * np.cos(th), r_real * np.sin(th),
                           zb[0] + pts[:, 3] * (zb[1] - zb[0])], axis=-1)

    out_base = args.out if args.out else f"data/compare_{name}.vtk"
    stem, ext = os.path.splitext(out_base)
    k = max(1, args.k)
    greedy = k == 1
    top_k = 1 if greedy else 0
    print(f"generation: k={k} "
          f"{'greedy top_k=1' if greedy else 'stochastic top_k=0'} "
          f"temperature={args.temperature} seed={args.seed} "
          f"n_points={cfg['n_points']}")

    n_valid = 0
    for i in range(k):
        seq = generate(model, pc, fc, core.start_token, stop_id,
                       core.sep_token, cap, args.temperature, top_k, dev,
                       dtype, specials, use_slot, tok=tok, constrained=True,
                       coords=coords)
        print(f"rollout {i}: tokens={len(seq)} "
              f"stop={'ja' if seq[-1] == stop_id else 'NEIN (cap)'} "
              f"rows={seq.count(core.sep_token)}")
        res, trim = detokenize_safe(seq, tok, stop_id, coords=coords)
        if res is None:
            print(f"rollout {i}: detokenize FEHLGESCHLAGEN ({trim})")
            continue
        vpt, blk = res
        vcart = _to_cart(vpt.numpy(), coords)
        validation = validate_generated_mesh(vcart, blk.numpy(),
                                             expected_blocks=blocks)
        print(f"rollout {i}: mesh validation: "
              f"{'valid' if validation.valid else 'invalid'} "
              f"(blocks={validation.n_blocks}, verts={validation.n_vertices})")
        path = out_base if greedy else f"{stem}_r{i}{ext}"
        _write_compare_vtk(path, gt_cart, gt_blocks, vcart,
                           [[int(j) for j in b] for b in blk.numpy()], pc_vis)
        print(f"saved {path}  (part: 1=true, 2=generated, 3=punktwolke)")
        n_valid += int(validation.valid)
        if greedy and not validation.valid:
            print("INVALID GENERATED MESH: " + "; ".join(validation.errors))
            return 2
    print(f"done: {n_valid}/{k} valide Rollouts")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/polytron_data_3d_smoke.pt")
    ap.add_argument("--idx", type=int, default=None)
    ap.add_argument("--seq", default=None, help="Legacy-Modus: Token-Sequenz .pt")
    ap.add_argument("--tokens", default="data/hexarow_overfit_1sample.pt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-points", type=int, default=1000)
    ap.add_argument("--ckpt", default="", help="Generation-Modus: Checkpoint .pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--k", type=int, default=1,
                    help="Rollouts: 1=greedy (1 Datei), >1=stochastisch")
    args = ap.parse_args()

    if args.ckpt and args.seq is not None:
        ap.error("--ckpt (Generation) und --seq (Legacy) schliessen sich aus")
    if args.ckpt:
        return _run_generation(args, ap)
    return _run_legacy(args)


if __name__ == "__main__":
    raise SystemExit(main())
