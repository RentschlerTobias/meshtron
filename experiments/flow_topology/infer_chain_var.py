#!/usr/bin/env python3
"""M1 check of VARIABLE_TOPOLOGY_PLAN.md: stage 3 (flow_geo_var.py) on corners of the current flow model.

Per item, same chain as infer_chain_flow.py (snap, curved TFI, mesh metrics):
  flowp      corners + curves of the fixed-topology flow model (flow_geo.py), the reference
  var        stage 3 started from the flowp corners (curves new, corners refined)
  var_gtc    stage 3 started from the GT corners (curve quality alone)
  gt         GT corners + curves

    PYTHONPATH=<dp3d hex3d_algohex> python infer_chain_var.py --var-ckpt runs/var_s0/last.pt --split val_new \
        --src-root /work/datagen_papaya/runs --out-dir chain/var_m1_valnew
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
import flow_geo_var as FV  # noqa: E402
from flow_geo import sample as sample_fixed  # noqa: E402
from infer_chain_v2 import RELABEL, snap_all  # noqa: E402
from infer_chain_flow import metrics, to_seq  # noqa: E402
import continuity as C  # noqa: E402
from meshtron.data.polytron_blocks import encode_sample  # noqa: E402
from meshtron.geometry.geometry_features import FeatureModelV2  # noqa: E402
from meshtron.geometry.polytron_tfi import mesh_candidate  # noqa: E402
from meshtron.training.train_polytron import load_stage  # noqa: E402

VARIANTS = ("flowp", "var", "var_gtc", "gt")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="/work/meshtron_v2/data/flow_canon_papaya.pt")
    ap.add_argument("--var-ckpt", required=True)
    ap.add_argument("--flow-ckpt", default="/work/meshtron_v2/runs/flowp_flow_all_s0/last.pt")
    ap.add_argument("--flow-data", default="/work/meshtron_v2/data/flow_canon_papaya.pt",
                    help="the data the fixed-topology model was trained on (its template statistics)")
    ap.add_argument("--spec-from", default="/work/meshtron_v2/runs/curve_order_last.pt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--src-root", default=RELABEL)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--target-h", type=float, default=0.08)
    ap.add_argument("--feature-cache", default="/work/curve/features_v2")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    dev = "cuda"
    os.makedirs(a.out_dir, exist_ok=True)
    data = torch.load(a.data, weights_only=False)
    fixed, tp, fa = C.load_flow(a.flow_ckpt, torch.load(a.flow_data, weights_only=False))
    var, nrm, va = FV.load(a.var_ckpt, data)
    _, spec, _ = load_stage(a.spec_from, dev)
    edges_fixed = tp.edges.cpu().numpy()
    gen = torch.Generator(device=dev); gen.manual_seed(1234)
    rows = []
    for it in data[a.split][:a.n]:
        root = {"tistos": "/work/datagen_full/datagen/runs", "machine": RELABEL}.get(it["name"].split("_")[0], a.src_root) \
            if a.src_root == "auto" else a.src_root
        npz = os.path.join(root, it["name"], "sample.npz")
        with np.load(npz, allow_pickle=True) as zf:
            raw = {k: np.asarray(zf[k]) for k in zf.files}
        fm = FeatureModelV2(npz, cache_dir=a.feature_cache)
        B = it["conn"].numpy().astype(np.int64)
        E = it["edges"].numpy().astype(np.int64)
        assert np.array_equal(E, edges_fixed), "M1 runs on the fixed-topology data"
        pts, pmask = T.batch_points([it], dev)
        with torch.no_grad():
            z = fixed.enc(pts, pmask)
            v, o = tp.decode(*sample_fixed(fixed, z, fa["mode"], fa["steps"], gen))
        V1 = v[0].cpu().double().numpy(); O1 = o[0].cpu().double().numpy()
        b = FV.collate([it], dev)
        with torch.no_grad():
            zv = var.enc(pts, pmask)
            out = {}
            for tag, V0 in (("var", torch.as_tensor(V1, dtype=torch.float32)), ("var_gtc", it["verts"].float())):
                xv, xo = FV.sample(var, zv, b, nrm.enc_v(V0[None].to(dev)), va["steps"], gen)
                out[tag] = (nrm.dec_v(xv)[0].cpu().double().numpy(), nrm.dec_o(xo)[0].cpu().double().numpy())
        Vgt = it["verts"].numpy().astype(np.float64)
        # G1 kinks at the continuity pairs (closed loops = O-rings) of every variant's cubics
        from meshtron.data.polytron_blocks import offsets_to_ctrl
        from curve_opt import closed_chain_edges
        vpatch = [set((np.flatnonzero(l.numpy() > 0.5) + 1).tolist()) for l in it["vlab"]]
        g1, g2 = offsets_to_ctrl(Vgt[E[:, 0]], Vgt[E[:, 1]], it["eoff"].numpy().astype(np.float64))
        det = C.detect(Vgt, B, E, (g1, g2), fm, vpatch)
        closed = closed_chain_edges(det["pairs"])

        def kinks(V, off):
            k1, k2 = offsets_to_ctrl(V[E[:, 0]], V[E[:, 1]], off)
            ks = [(C.kink(V, E, k1, k2, v, e, f), e in closed and f in closed) for (v, e, f) in det["pairs"]]
            return {"ring": [k for k, c in ks if c], "open": [k for k, c in ks if not c]}
        row = {"name": it["name"],
               "corner_flowp": float(1000 * np.linalg.norm(V1 - Vgt, axis=1).mean()),
               "corner_var": float(1000 * np.linalg.norm(out["var"][0] - Vgt, axis=1).mean())}

        def chain(V, off, tag):
            vtk = os.path.join(a.out_dir, f"{it['name']}_{tag}.vtk")
            V2 = snap_all(V, B, fm)
            return metrics(mesh_candidate(to_seq(V2, B, E, off, spec), spec, raw, a.target_h, vtk), vtk)

        row["kinks"] = {"flowp": kinks(V1, O1), "var": kinks(*out["var"]), "var_gtc": kinks(*out["var_gtc"]),
                        "gt": kinks(Vgt, it["eoff"].numpy().astype(np.float64))}
        row["flowp"] = chain(V1, O1, "flowp")
        row["var"] = chain(*out["var"], "var")
        row["var_gtc"] = chain(*out["var_gtc"], "var_gtc")
        vtk = os.path.join(a.out_dir, f"{it['name']}_gt.vtk")
        row["gt"] = metrics(mesh_candidate(encode_sample(raw["vertices"], raw["blocks"], raw["edges"], raw["edge_ctrl"], spec),
                                           spec, raw, a.target_h, vtk), vtk)
        rows.append(row)
        print(f"{it['name']:22s} corner flowp {row['corner_flowp']:.1f} var {row['corner_var']:.1f} | " + " | ".join(
            f"{k} " + ("-" if row[k] is None else f"unc {row[k]['uncovered']:.3f} inv {100*row[k]['inv_share']:.2f}%")
            for k in VARIANTS), flush=True)
        json.dump(rows, open(os.path.join(a.out_dir, "rows.json"), "w"), indent=1)
    summ = {"corner_flowp_med": float(np.median([r["corner_flowp"] for r in rows])),
            "corner_var_med": float(np.median([r["corner_var"] for r in rows]))}
    for k in VARIANTS:
        ok = [r[k] for r in rows if r[k] is not None]
        summ[k] = {"meshed": len(ok), "watertight": sum(m["watertight"] for m in ok),
                   "uncovered_mean": float(np.mean([m["uncovered"] for m in ok])),
                   "uncovered_median": float(np.median([m["uncovered"] for m in ok])),
                   "inv_mean_pct": float(100 * np.mean([m["inv_share"] for m in ok])),
                   "no_inverted": sum(m["inv_share"] == 0 for m in ok),
                   "sj_p1_median": float(np.median([m["sj_p1"] for m in ok])),
                   "ring_kink_median": float(np.median([x for r in rows for x in r["kinks"][k]["ring"]])),
                   "ring_kink_p90": float(np.percentile([x for r in rows for x in r["kinks"][k]["ring"]], 90))}
    json.dump(summ, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
