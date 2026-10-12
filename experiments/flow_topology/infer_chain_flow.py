#!/usr/bin/env python3
"""End-to-end chain for the flow-matching geometry model (flow_geo.py): corners
AND Bezier edge curves come from one model, the topology (blocks) is the GT
canonical one (stage A = the single topology). Same snap, curved TFI and mesh
metrics as infer_chain_v2.py, so the summaries are directly comparable with
chain/long_val (AR vertex model + pointer conn + curve head).

Variants per item:
  flow         one sample (fixed seed), corners snapped, curves re-attached to the
               snapped corners through their chord-relative offsets
  flow_nosnap  the same sample without snapping
  flow_best    best of --n-best samples by a GT-free score on the target geometry
               (uncovered surface share + 10 * inverted share), snapped
  mean_tpl     mean train corners + mean train curve offsets, snapped (baseline)
  gt           GT corners + GT curves (ceiling)

    PYTHONPATH=<dp3d hex3d_algohex> python infer_chain_flow.py --ckpt runs/flow_flow_all_s0/last.pt \
        --split val --out-dir chain/flow_val
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, "/opt/stack/meshtron")
sys.path.insert(0, "/work/meshtron_v2")
sys.path.insert(0, "/work/rl_reward")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import train as T  # noqa: E402
from flow_geo import FlowGeo, Topology, sample  # noqa: E402
from infer_chain_v2 import RELABEL, snap_all  # noqa: E402
from mesh_quality import from_vtk, r_mesh  # noqa: E402
from meshtron.data.polytron_blocks import encode_sample, offsets_to_ctrl  # noqa: E402
from meshtron.geometry.geometry_features import FeatureModelV2  # noqa: E402
from meshtron.geometry.polytron_tfi import mesh_candidate  # noqa: E402
from meshtron.training.train_polytron import load_stage  # noqa: E402

VARIANTS = ("flow", "flow_nosnap", "flow_best", "mean_tpl", "gt")


def to_seq(V, B, edges, off, spec):
    """Continuous corners + chord offsets -> PolySeq (both edge directions recorded)."""
    c1, c2 = offsets_to_ctrl(V[edges[:, 0]], V[edges[:, 1]], off)
    E = np.concatenate([edges, edges[:, ::-1]])
    ctrl = np.concatenate([np.stack([c1, c2], 1), np.stack([c2, c1], 1)])
    return encode_sample(V, B, E, ctrl, spec)


def metrics(r, vtk):
    if r is None or r.get("tfi") is None:
        return None
    q = from_vtk(vtk)
    return {"watertight": bool(r["tfi"]["watertight"]),
            "inv_share": r["tfi"]["inverted_curved"] / max(1, r["tfi"]["cells_after"]),
            "uncovered": float(r["surface"]["uncovered_share"]),
            "on_surface": float(r["surface"]["on_surface_median"]),
            "sj_p1": q["sj_p1"], "nonortho_max": q["nonortho_max"], "R_mesh": r_mesh(q)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="/work/meshtron_v2/data/flow_canon.pt")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--spec-from", default="/work/meshtron_v2/runs/curve_order_last.pt",
                    help="checkpoint whose PolytronSpec the chain uses (same as infer_chain_v2)")
    ap.add_argument("--split", default="val", choices=("train", "val", "val_new"))
    ap.add_argument("--src-root", default=RELABEL, help="directory with <name>/sample.npz of the evaluated items")
    ap.add_argument("--eval-data", default=None, help="take the --split items from this file (default --data)")
    ap.add_argument("--n", type=int, default=34)
    ap.add_argument("--n-best", type=int, default=4)
    ap.add_argument("--target-h", type=float, default=0.08)
    ap.add_argument("--feature-cache", default="/work/curve/features_v2")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    dev = "cuda"
    os.makedirs(a.out_dir, exist_ok=True)

    data = torch.load(a.data, weights_only=False)
    ck = torch.load(a.ckpt, weights_only=False); fa = ck["args"]
    rmax = float(data["r_bounds"][1]); z0, z1 = (float(v) for v in data["z_bounds"])
    lo, hi = torch.tensor([-rmax, -rmax, z0]), torch.tensor([rmax, rmax, z1])
    tr = data["train"]
    if fa["train_frac"] < 1.0:
        rng = np.random.default_rng(fa["seed"])
        tr = [tr[i] for i in sorted(rng.choice(len(tr), max(1, round(fa["train_frac"] * len(tr))), replace=False))]
    tp = Topology(tr, lo, hi).to(dev)
    model = FlowGeo(tp, fa["d"], fa["layers"], fa["heads"], fa["n_latent"], 0.0, curves=fa["parts"] == "all").to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    _, spec, _ = load_stage(a.spec_from, dev)
    edges = tp.edges.cpu().numpy()
    print(f"flow {a.ckpt} ({fa['mode']}/{fa['parts']}) | spec {a.spec_from}", flush=True)

    gen = torch.Generator(device=dev); gen.manual_seed(1234)
    rows = []
    ev = torch.load(a.eval_data, weights_only=False) if a.eval_data else data
    for it in ev[a.split][:a.n]:
        npz = os.path.join(a.src_root, it["name"], "sample.npz")
        with np.load(npz, allow_pickle=True) as zf:
            raw = {k: np.asarray(zf[k]) for k in zf.files}
        fm = FeatureModelV2(npz, cache_dir=a.feature_cache)
        B = it["conn"].numpy().astype(np.int64)
        pts, pmask = T.batch_points([it], dev)
        with torch.no_grad():
            z = model.enc(pts, pmask)
            n_s = 1 if fa["mode"] == "regress" else max(1, a.n_best)
            samples = [tp.decode(*sample(model, z, fa["mode"], fa["steps"], gen)) for _ in range(n_s)]
        samples = [(v[0].cpu().numpy().astype(np.float64), o[0].cpu().numpy().astype(np.float64)) for v, o in samples]
        if fa["parts"] != "all":                 # verts-only model: mean curves
            samples = [(v, tp.omean.cpu().numpy().astype(np.float64)) for v, _ in samples]
        Vgt = it["verts"].numpy().astype(np.float64)
        row = {"name": it["name"], "corner_err": float(1000 * np.linalg.norm(samples[0][0] - Vgt, axis=1).mean())}

        def chain(V, off, tag, snap=True):
            V2 = snap_all(V, B, fm) if snap else V
            vtk = os.path.join(a.out_dir, f"{it['name']}_{tag}.vtk")
            return metrics(mesh_candidate(to_seq(V2, B, edges, off, spec), spec, raw, a.target_h, vtk), vtk)

        row["flow"] = chain(*samples[0], "flow")
        row["flow_nosnap"] = chain(*samples[0], "flow_nosnap", snap=False)
        best = None
        for k, s in enumerate(samples):
            m = row["flow"] if k == 0 else chain(*s, f"flow_s{k}")
            sc = None if m is None else m["uncovered"] + 10 * m["inv_share"]
            if sc is not None and (best is None or sc < best[0]):
                best = (sc, k, m)
        row["flow_best"] = None if best is None else best[2]
        row["best_k"] = None if best is None else best[1]
        row["mean_tpl"] = chain(tp.vmean.cpu().numpy().astype(np.float64), tp.omean.cpu().numpy().astype(np.float64), "mean_tpl")
        vtk = os.path.join(a.out_dir, f"{it['name']}_gt.vtk")
        row["gt"] = metrics(mesh_candidate(encode_sample(raw["vertices"], raw["blocks"], raw["edges"], raw["edge_ctrl"], spec),
                                           spec, raw, a.target_h, vtk), vtk)
        rows.append(row)
        print(f"{it['name']:22s} err {row['corner_err']:.1f} | " + " | ".join(
            f"{v} " + ("-" if row[v] is None else f"unc {row[v]['uncovered']:.3f} inv {100*row[v]['inv_share']:.2f}%")
            for v in VARIANTS), flush=True)

    summ = {"split": a.split, "n": len(rows), "ckpt": a.ckpt,
            "corner_err_median": float(np.median([r["corner_err"] for r in rows]))}
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
