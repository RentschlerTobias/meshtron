#!/usr/bin/env python3
"""End-to-end chain on the canonical-topology v2 set: how good are the meshes?

Polytron stage 1 (vertex model, greedy) -> stage 2 (pointer net on the generated
corners) -> snap all corners -> learned edge curves (curve head) -> curved TFI.
Same mesh metrics as scripts/infer_chain_order.py (polytron_tfi.mesh_candidate)
plus the post-TFI quality score of meshtron/geometry/tfi_mesh_quality.py.

Variants per item (one greedy rollout each):
  gen          generated corners + predicted connectivity            (the pipeline)
  gen_gtconn   generated corners + GT connectivity                   (isolates stage 1)
  mean         mean train corners + GT connectivity                  (no-learning baseline)
  gen_nosnap   generated corners + predicted connectivity, no snap
  gt_canon_curves  GT corners/blocks in the generator's canonical order + curve head
  gt_curves    GT blocks in raw sample order + curve head            (curve head alone)
  gt           GT blocks + GT curves                                 (ceiling)

    BLOCKGEN_SAMPLES=<relabel>/out PYTHONPATH=<dp3d hex3d_algohex> \
    python scripts/blockgen_v2/infer_chain_v2.py --data <v2 canon set>.pt --split train --n 34 \
        --vertex <run>/last.pt --conn <run>/last.pt --curve <curve head>_last.pt --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths  # noqa: E402  (puts the repo root on sys.path)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from meshtron.geometry.tfi_mesh_quality import from_vtk, r_mesh  # noqa: E402
from meshtron.model.blockgen import BlockGen  # noqa: E402
from meshtron.training import train_blockgen as T  # noqa: E402
from meshtron.data.hexa_row_tokenizer import HexaRowTokenizer  # noqa: E402
from meshtron.data.polytron_blocks import PolySeq, build_cloud as poly_cloud, encode_sample, point_labels  # noqa: E402
from meshtron.geometry.block_mapping import SnapConfigV2, snap_corners_v2  # noqa: E402
from meshtron.geometry.geometry_features import FeatureModelV2  # noqa: E402
from meshtron.geometry.polytron_tfi import mesh_candidate  # noqa: E402
from meshtron.training.polytron_sample import predict_curves  # noqa: E402
from meshtron.training.train_polytron import load_stage  # noqa: E402

VARIANTS = ("gen", "gen_nosnap", "gen_gtconn", "mean", "gt_canon_curves", "gt_curves", "gt")


def load_raw(name):
    with np.load(os.path.join(_paths.need_samples(), name, "sample.npz"), allow_pickle=True) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def snap_all(V, B, fm):
    target = type("T", (), {})()
    target.curves, target.surface_nearest = fm.seam_curves, fm.surface_nearest
    C, _ = snap_corners_v2(target, V[B], SnapConfigV2(tol_v=0.06, tol_e=0.04))
    V2 = V.copy()
    for vid, c in zip(B.reshape(-1), C.reshape(-1, 3)):
        V2[vid] = c
    return V2


@torch.no_grad()
def decode_conn(model, V, it, dev, max_steps):
    Vt = torch.as_tensor(V, dtype=torch.float32, device=dev)[None]
    vmask = torch.ones(1, len(V), dtype=torch.bool, device=dev)
    pts, pmask = T.batch_points([it], dev)
    seq = [-1]
    for _ in range(max_steps):
        lg = model(Vt, vmask, torch.tensor([seq], device=dev).clamp(min=-1), pts, pmask)
        nxt = int(lg[0, -1].argmax()); seq.append(nxt)
        if nxt == len(V):
            break
    p = [x for x in seq[1:] if x != len(V)]
    if len(p) % 8:
        return None
    return np.array(p, np.int64).reshape(-1, 8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="v2 dataset (build_dataset.py --topo ...)")
    ap.add_argument("--vertex", required=True)
    ap.add_argument("--conn", required=True)
    ap.add_argument("--curve", required=True)
    ap.add_argument("--split", default="val", choices=("train", "val"))
    ap.add_argument("--n", type=int, default=34)
    ap.add_argument("--target-h", type=float, default=0.08)
    ap.add_argument("--curve-n-points", type=int, default=2048)
    ap.add_argument("--feature-cache", default=_paths.FEATURE_CACHE)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    dev = "cuda"
    os.makedirs(a.out_dir, exist_ok=True); os.makedirs(a.feature_cache, exist_ok=True)

    data = torch.load(a.data, weights_only=False)
    sp = data["special"]; specials = set(sp.values())
    tok = HexaRowTokenizer(r_bounds=tuple(data["r_bounds"]), z_bounds=tuple(data["z_bounds"]))
    ck = torch.load(a.vertex, weights_only=False); va = ck["args"]
    max_len = ck["model"]["pos.weight"].shape[0]          # as trained (dataset-dependent)
    vm = BlockGen(data["vocab"], va["d"], va["layers"], va["heads"], max_len, sp["pad"], 0.0, va["n_latent"]).to(dev)
    vm.load_state_dict(ck["model"]); vm.eval()
    ck = torch.load(a.conn, weights_only=False); ca = ck["args"]
    cmax = ck["model"]["pos.weight"].shape[0]
    cm = T.PointerNet(ca["d"], ca["layers"], ca["heads"], cmax, dropout=0.0).to(dev)
    cm.load_state_dict(ck["model"]); cm.eval()
    curve_model, spec, ck2 = load_stage(a.curve, dev)
    print(f"vertex {a.vertex} | conn {a.conn} | curve {a.curve} (epoch {ck2['epoch']})", flush=True)
    mean_V = np.stack([it["verts"].numpy() for it in data["train"]]).mean(0)

    items = data[a.split][:a.n]
    rows = []
    for it in items:
        raw = load_raw(it["name"])
        fm = FeatureModelV2(os.path.join(_paths.need_samples(), it["name"], "sample.npz"), cache_dir=a.feature_cache)
        lab = point_labels(len(raw["surface_points"]), raw["surface_tris"], raw["surface_tri_label"])
        pc = torch.as_tensor(np.array(poly_cloud(raw["surface_points"], lab, a.curve_n_points, spec,
                                                 np.random.default_rng(0)))[None], dtype=torch.float32, device=dev)
        pts, pmask = T.batch_points([it], dev)
        out = T.generate(vm, pts, pmask, sp["start"], sp["stop"], specials, len(it["vtok"]) + 40)[0].tolist()
        if sp["stop"] in out[1:]:
            out = out[:out.index(sp["stop"], 1)]
        body = [t for t in out[1:] if t not in specials]
        Vg = T.dequant_vertices(tok, body[:len(body) // 3 * 3])
        Vgt, Bgt = it["verts"].numpy().astype(np.float64), it["conn"].numpy().astype(np.int64).reshape(-1, 8)
        row = {"name": it["name"], "n_verts_gen": len(Vg),
               "corner_err": float(1000 * np.linalg.norm(Vg - Vgt, axis=1).mean()) if len(Vg) == len(Vgt) else None}
        Bg = decode_conn(cm, Vg, it, dev, cmax) if len(Vg) else None
        row["conn_exact"] = bool(Bg is not None and Bg.shape == Bgt.shape and (Bg == Bgt).all())

        def chain(V, B, tag, snap=True):
            V2 = snap_all(V, B, fm) if snap else V
            vq = spec.quant_xyz(V2)
            e, cq = predict_curves(curve_model, vq, B, pc, len(B), spec)
            return mesh_candidate(PolySeq(vq, B, e, cq), spec, raw, a.target_h,
                                  os.path.join(a.out_dir, f"{it['name']}_{tag}.vtk"))

        cand = {}
        if Bg is not None and len(Vg):
            cand["gen"] = chain(Vg, Bg, "gen")
            cand["gen_nosnap"] = chain(Vg, Bg, "gen_nosnap", snap=False)
        if len(Vg) == len(Vgt):
            cand["gen_gtconn"] = chain(Vg, Bgt, "gen_gtconn")
        cand["mean"] = chain(mean_V, Bgt, "mean")
        cand["gt_canon_curves"] = chain(Vgt, Bgt, "gt_canon_curves", snap=False)
        # curve head alone / ceiling on the raw GT block structure
        vq = spec.quant_xyz(raw["vertices"])
        e, cq = predict_curves(curve_model, vq, raw["blocks"], pc, len(raw["blocks"]), spec)
        cand["gt_curves"] = mesh_candidate(PolySeq(vq, raw["blocks"], e, cq), spec, raw, a.target_h,
                                           os.path.join(a.out_dir, f"{it['name']}_gt_curves.vtk"))
        cand["gt"] = mesh_candidate(encode_sample(raw["vertices"], raw["blocks"], raw["edges"], raw["edge_ctrl"], spec),
                                    spec, raw, a.target_h, os.path.join(a.out_dir, f"{it['name']}_gt.vtk"))
        for v in VARIANTS:
            r = cand.get(v)
            if r is None or r.get("tfi") is None:
                row[v] = None
                continue
            q = from_vtk(os.path.join(a.out_dir, f"{it['name']}_{v}.vtk"))
            row[v] = {"watertight": bool(r["tfi"]["watertight"]),
                      "inv_share": r["tfi"]["inverted_curved"] / max(1, r["tfi"]["cells_after"]),
                      "uncovered": float(r["surface"]["uncovered_share"]),
                      "on_surface": float(r["surface"]["on_surface_median"]),
                      "sj_p1": q["sj_p1"], "nonortho_max": q["nonortho_max"], "R_mesh": r_mesh(q)}
        rows.append(row)
        print(f"{it['name']:22s} err {row['corner_err'] if row['corner_err'] is None else round(row['corner_err'], 1)} "
              f"conn {row['conn_exact']:d} | " + " | ".join(
                  f"{v} " + ("-" if row[v] is None else f"unc {row[v]['uncovered']:.3f} inv {100*row[v]['inv_share']:.2f}% "
                                                         f"R {row[v]['R_mesh']:.2f}") for v in VARIANTS), flush=True)

    summ = {"split": a.split, "n": len(rows),
            "corner_err_median": float(np.median([r["corner_err"] for r in rows if r["corner_err"] is not None])),
            "conn_exact": int(sum(r["conn_exact"] for r in rows))}
    for v in VARIANTS:
        x = [r[v] for r in rows if r[v]]
        summ[v] = None if not x else {
            "meshed": len(x), "watertight": int(sum(y["watertight"] for y in x)),
            "uncovered_median": float(np.median([y["uncovered"] for y in x])),
            "inv_median_pct": 100 * float(np.median([y["inv_share"] for y in x])),
            "no_inverted": int(sum(y["inv_share"] == 0 for y in x)),
            "R_mesh_median": float(np.median([y["R_mesh"] for y in x]))}
    print(json.dumps(summ, indent=1))
    json.dump({"summary": summ, "rows": rows}, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
