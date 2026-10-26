#!/usr/bin/env python3
"""Generic continuity detection on a hex block structure (step 1 of HANDOFF_2026-10-12).

Which block edges should continue G1 through a vertex is decided from the block topology and the
patch structure of the input geometry only (no label numbers, no geometry-specific rules):

  3d       two edges at a vertex that share no block face, each the other's only such partner
           (grid line of the block structure through a regular vertex)
  2d       two boundary edges at a vertex that share no boundary quad, mutually unique, both inside
           the same patch (grid line of the boundary quad mesh)
  feature  two feature edges (adjacent boundary quads on different patches) at a vertex that lie on
           the same patch pair and whose feature curve has no kink there; edges on a feature follow
           the exact seam curve of the input geometry (FeatureModelV2.seam_curves)
  corner   vertex with >= 3 patches, a seam junction or a kink in the feature curve: no pairing
  (vertices where the rules leave several candidates are singular: no forced pairing, the
   candidates are listed as 'ambiguous')

A boundary edge takes the first rule that applies in the order feature > 2d > 3d. Pairs are joined
into chains (grid lines / feature lines); per pair the kink angle of the cubic Bezier edges at the
shared vertex is measured for GT and (optionally) for one flow_geo sample.

    PYTHONPATH=<dp3d hex3d_algohex> python continuity.py --data data/flow_canon_papaya.pt \
        --split val_new --src-root /opt/stack/meshtron/experiments/flow_topology/datagen_papaya/runs --ckpt runs/flowp_flow_all_s0/last.pt \
        --out continuity
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, "/opt/stack/meshtron")
sys.path.insert(0, "/opt/stack/meshtron/experiments/flow_topology")

import meshio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from meshtron.data.polytron_blocks import bezier, offsets_to_ctrl  # noqa: E402
from meshtron.geometry.geometry_features import FeatureModelV2  # noqa: E402

RELABEL = "/opt/stack/meshtron/experiments/flow_topology/relabel/out"
# VTK hex faces (same as meshtron.geometry.mesh_validation._HEX_FACES)
HEX_FACES = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
KIND = {"3d": 0, "2d": 1, "feature": 2}


# ------------------------------------------------------------------ topology
def block_faces(B):
    """All block faces (sorted vertex tuples) -> number of incident blocks; face list per block."""
    cnt = collections.Counter()
    per = []
    for b in B:
        fs = [tuple(int(b[i]) for i in f) for f in HEX_FACES]
        per.append(fs)
        for f in fs:
            cnt[tuple(sorted(f))] += 1
    return cnt, per


def face_edges(f):
    return {tuple(sorted((f[i], f[(i + 1) % 4]))) for i in range(4)}


def edge_face_sets(edges, faces):
    """edge index -> set of face ids (faces given as ordered quads)."""
    eid = {tuple(sorted(map(int, e))): k for k, e in enumerate(edges)}
    out = [set() for _ in edges]
    for fi, f in enumerate(faces):
        for e in face_edges(f):
            if e in eid:
                out[eid[e]].add(fi)
    return out


def mutual_pairs(edges, cand_edges, share):
    """Pairs at a vertex among cand_edges that share no element (face / quad), mutually unique.

    Returns (pairs, ambiguous): pairs = {(v, e1, e2)}, ambiguous = [(v, [candidate pairs])]."""
    inc = collections.defaultdict(list)
    for k in cand_edges:
        for v in edges[k]:
            inc[int(v)].append(k)
    pairs, amb = set(), []
    for v, es in inc.items():
        part = {e: [f for f in es if f != e and not (share[e] & share[f])] for e in es}
        cand = sorted({tuple(sorted((e, f))) for e, fs in part.items() for f in fs})
        uniq = [(e, f) for e, f in cand if part[e] == [f] and part[f] == [e]]
        for e, f in uniq:
            pairs.add((v, e, f))
        rest = [p for p in cand if p not in uniq]
        if rest:
            amb.append((v, rest))
    return pairs, amb


# ------------------------------------------------------------------ patches
def quad_patch(quads, V, edges, ctrl, fm, vpatch=None):
    """Patch label of each boundary quad: the patch all four corners lie on (vpatch = per-vertex
    patch sets); if that is not unique, the surface label nearest to the curved quad centre
    (mean of the four Bezier edge midpoints)."""
    eid = {tuple(sorted(map(int, e))): k for k, e in enumerate(edges)}
    lab = []
    for q in quads:
        if vpatch is not None:
            common = set.intersection(*(vpatch[v] for v in q))
            if len(common) == 1:
                lab.append(common.pop())
                continue
        mids = []
        for a, b in ((q[i], q[(i + 1) % 4]) for i in range(4)):
            k = eid[tuple(sorted((a, b)))]
            P = bezier(V[edges[k, 0]], V[edges[k, 1]], ctrl[0][k], ctrl[1][k], 3)
            mids.append(P[1])
        _, tri, _ = fm.surface_nearest(np.mean(mids, 0)[None])
        lab.append(int(fm.surface_tri_label[tri[0]]))
    return np.array(lab)


def seam_turn(fm, p, h):
    """Turning angle (deg) of the nearest seam curve at p over arclength window +-h, and whether p
    is an open end (junction) of that curve; (None, None) if no seam is near."""
    cs = fm.seam_curves
    if cs.n_curves == 0:
        return None, None
    d, c, t, _ = cs.nearest(p[None])
    c, t = int(c[0]), float(t[0])
    lo, hi = int(cs.offset[c]), int(cs.offset[c + 1])
    L = float(cs.arclen[hi - 1])
    closed = bool(cs.closed[c])
    if not closed and (t < 0.5 * h or t > L - 0.5 * h):
        return None, True
    ta, tb = t - h, t + h
    if closed:
        ta, tb = ta % L, tb % L
    else:
        ta, tb = max(ta, 0.0), min(tb, L)
    A, = cs.sample_segment(c, ta, ta, 1); Bp, = cs.sample_segment(c, tb, tb, 1)
    P, = cs.sample_segment(c, t, t, 1)
    u, w = P - A, Bp - P
    cosv = u @ w / max(np.linalg.norm(u) * np.linalg.norm(w), 1e-30)
    return float(np.degrees(np.arccos(np.clip(cosv, -1, 1)))), False


# ------------------------------------------------------------------ detection
def detect(V, B, edges, ctrl, fm, vpatch=None, kink_deg=30.0):
    """Continuity pairs for one block structure. ctrl = (c1, c2) per edge (GT geometry),
    vpatch = patch set per vertex (from the sample's surface labels) or None.

    A feature curve has a corner at v if it turns by more than kink_deg over +-h and the turn does
    not shrink when h is halved (h = seam polyline spacing): a smooth curve, also a tight hairpin,
    turns ~linearly in h, a corner keeps its angle."""
    cnt, per = block_faces(B)
    allf = [f for fs in per for f in fs]
    uniq_faces = list({tuple(sorted(f)): f for f in allf}.values())
    bquads = [f for f in uniq_faces if cnt[tuple(sorted(f))] == 1]
    share3 = edge_face_sets(edges, uniq_faces)
    share2 = edge_face_sets(edges, bquads)
    bedge = [k for k in range(len(edges)) if share2[k]]
    qlab = quad_patch(bquads, V, edges, ctrl, fm, vpatch)

    # patches per edge (from its boundary quads) and per vertex
    epat = [set(int(qlab[q]) for q in share2[k]) for k in range(len(edges))]
    feat = [k for k in bedge if len(epat[k]) >= 2]
    vpat = collections.defaultdict(set)
    for qi, q in enumerate(bquads):
        for v in q:
            vpat[v].add(int(qlab[qi]))

    # corners: >= 3 patches, seam junction or kink of the feature curve
    sp = fm.seam_curves.pts
    same = fm.seam_curves.curve_of[1:] == fm.seam_curves.curve_of[:-1]
    h = float(np.median(np.linalg.norm(np.diff(sp, axis=0), axis=1)[same])) if same.any() else 0.0
    corner, vinfo = set(), {}
    for v in {int(x) for k in feat for x in edges[k]}:
        turn, junction = seam_turn(fm, V[v], h)
        turn2, _ = seam_turn(fm, V[v], 2 * h)
        kinked = turn is not None and turn > kink_deg and turn > 0.75 * (turn2 or 0.0)
        vinfo[v] = {"patches": len(vpat[v]), "seam_turn": turn, "seam_turn_2h": turn2, "junction": junction}
        if len(vpat[v]) >= 3 or junction or turn is None or kinked:
            corner.add(v)

    # feature pairs: same patch pair, no shared boundary quad, not at a corner, mutually unique
    fpairs, famb = set(), []
    finc = collections.defaultdict(list)
    for k in feat:
        for v in edges[k]:
            finc[int(v)].append(k)
    for v, es in finc.items():
        if v in corner:
            continue
        part = {e: [f for f in es if f != e and epat[e] == epat[f] and not (share2[e] & share2[f])] for e in es}
        cand = sorted({tuple(sorted((e, f))) for e, fs in part.items() for f in fs})
        uniq = [(e, f) for e, f in cand if part[e] == [f] and part[f] == [e]]
        fpairs |= {(v, e, f) for e, f in uniq}
        if [p for p in cand if p not in uniq]:
            famb.append((v, [p for p in cand if p not in uniq]))

    p2, a2 = mutual_pairs(edges, bedge, share2)
    p3, a3 = mutual_pairs(edges, range(len(edges)), share3)

    # combine: feature > 2d > 3d; a boundary edge already paired at v by a stronger rule keeps it
    pairs, used = {}, set()
    for kind, P in (("feature", fpairs), ("2d", p2), ("3d", p3)):
        for v, e, f in sorted(P):
            if (v, e) in used or (v, f) in used:
                continue
            if kind != "feature" and v in corner and (e in feat or f in feat):
                continue                                   # never run a grid line through a feature corner
            if kind == "2d" and (e in feat or f in feat or epat[e] != epat[f]):
                continue                                   # a boundary grid line stays inside one patch
            if kind == "3d" and (e in feat or f in feat):
                continue
            pairs[(v, e, f)] = kind
            used |= {(v, e), (v, f)}
    amb = {"3d": a3, "2d": a2, "feature": famb}
    return {"pairs": pairs, "ambiguous": amb, "corner": corner, "feat": set(feat), "bedge": set(bedge),
            "vinfo": vinfo, "n_bquads": len(bquads), "p3_raw": p3}


def chains(pairs, edges):
    """Join paired edges through their shared vertices into chains (lists of edge ids)."""
    nb = collections.defaultdict(list)
    for (v, e, f), kind in pairs.items():
        nb[e].append(f); nb[f].append(e)
    seen, out = set(), []
    for s in range(len(edges)):
        if s in seen or s not in nb:
            continue
        comp, stack = [], [s]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x); comp.append(x); stack += nb[x]
        out.append(sorted(comp))
    return out


def kink(V, edges, c1, c2, v, e, f):
    """Angle (deg) between the tangent arriving at v along e and leaving v along f."""
    def tan_out(k):                                        # tangent leaving v along edge k
        a, b = edges[k]
        t = c1[k] - V[a] if a == v else c2[k] - V[b]
        return t / max(np.linalg.norm(t), 1e-30)
    u, w = -tan_out(e), tan_out(f)
    return float(np.degrees(np.arccos(np.clip(u @ w, -1, 1))))


# ------------------------------------------------------------------ output
def write_vtk(path, V, edges, c1, c2, det, ch, n_pts=17):
    kind_of = {}
    for (v, e, f), kd in det["pairs"].items():
        for k in (e, f):
            kind_of[k] = max(kind_of.get(k, -1), KIND[kd])
    chain_of = {e: i for i, c in enumerate(ch) for e in c}
    pts, cells, cid, ckind, eid, isfeat = [V], [], [], [], [], []
    base = len(V)
    for k, (a, b) in enumerate(edges):
        P = bezier(V[a], V[b], c1[k], c2[k], n_pts)
        pts.append(P)
        idx = base + np.arange(n_pts)
        seg = np.stack([idx[:-1], idx[1:]], 1)
        cells.append(seg); base += n_pts
        n = len(seg)
        cid += [chain_of.get(k, -1)] * n; ckind += [kind_of.get(k, -1)] * n
        eid += [k] * n; isfeat += [int(k in det["feat"])] * n
    role = np.zeros(len(V), int)                           # 0 regular, 1 corner, 2 ambiguous
    for v in det["corner"]:
        role[v] = 1
    for lst in det["ambiguous"].values():
        for v, _ in lst:
            if role[v] == 0:
                role[v] = 2
    m = meshio.Mesh(np.concatenate(pts), [("vertex", np.arange(len(V))[:, None]), ("line", np.concatenate(cells))],
                    cell_data={"chain": [-np.ones(len(V), int), np.array(cid)],
                               "kind": [-np.ones(len(V), int), np.array(ckind)],
                               "edge": [-np.ones(len(V), int), np.array(eid)],
                               "feature": [np.zeros(len(V), int), np.array(isfeat)],
                               "vrole": [role, -np.ones(len(cid), int)]})
    m.write(path, binary=True)


def load_flow(ckpt, data, dev="cuda"):
    """flow_geo model + topology exactly as infer_chain_flow.py builds them."""
    from flow_geo import FlowGeo, Topology
    ck = torch.load(ckpt, weights_only=False); fa = ck["args"]
    rmax = float(data["r_bounds"][1]); z0, z1 = (float(v) for v in data["z_bounds"])
    lo, hi = torch.tensor([-rmax, -rmax, z0]), torch.tensor([rmax, rmax, z1])
    tr = data["train"]
    if fa["train_frac"] < 1.0:
        rng = np.random.default_rng(fa["seed"])
        tr = [tr[i] for i in sorted(rng.choice(len(tr), max(1, round(fa["train_frac"] * len(tr))), replace=False))]
    tp = Topology(tr, lo, hi).to(dev)
    model = FlowGeo(tp, fa["d"], fa["layers"], fa["heads"], fa["n_latent"], 0.0, curves=fa["parts"] == "all").to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    return model, tp, fa


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="/opt/stack/meshtron/experiments/flow_topology/data/flow_canon_papaya.pt")
    ap.add_argument("--split", default="val_new", choices=("train", "val", "val_new"))
    ap.add_argument("--src-root", default=RELABEL)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--ckpt", default=None, help="flow_geo checkpoint: also measure kinks of one sample")
    ap.add_argument("--kink-deg", type=float, default=30.0, help="feature-curve turn above this = corner")
    ap.add_argument("--feature-cache", default="/opt/stack/meshtron/experiments/flow_topology/cache/features_v2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = torch.load(a.data, weights_only=False)

    model = None
    if a.ckpt:
        from flow_geo import sample
        model, tp, fa = load_flow(a.ckpt, data)
        gen = torch.Generator(device="cuda"); gen.manual_seed(1234)

    rows = []
    for it in data[a.split][:a.n]:
        name = it["name"]
        npz = os.path.join(a.src_root, name, "sample.npz")
        fm = FeatureModelV2(npz, cache_dir=a.feature_cache)
        V = it["verts"].numpy().astype(np.float64)
        B = it["conn"].numpy().astype(np.int64)
        E = it["edges"].numpy().astype(np.int64)
        c1, c2 = offsets_to_ctrl(V[E[:, 0]], V[E[:, 1]], it["eoff"].numpy().astype(np.float64))
        vpatch = [set((np.flatnonzero(l.numpy() > 0.5) + 1).tolist()) for l in it["vlab"]]   # column k = label k+1
        det = detect(V, B, E, (c1, c2), fm, vpatch, kink_deg=a.kink_deg)
        ch = chains(det["pairs"], E)
        row = {"name": name, "n_pairs": collections.Counter(det["pairs"].values()),
               "n_p3_raw": len(det["p3_raw"]), "n_corner": len(det["corner"]), "n_feat_edges": len(det["feat"]),
               "n_chains": len(ch), "chain_len": sorted((len(c) for c in ch), reverse=True),
               "n_ambiguous": {k: sum(len(p) for _, p in v) for k, v in det["ambiguous"].items()},
               "pairs": [], "ambiguous": [], "vinfo": {int(v): i for v, i in det["vinfo"].items()}}
        geo = {"gt": (V, c1, c2)}
        if model is not None:
            from infer_chain_v2 import snap_all
            from train import batch_points
            pts, pmask = batch_points([it], "cuda")
            with torch.no_grad():
                z = model.enc(pts, pmask)
                v, o = tp.decode(*sample(model, z, fa["mode"], fa["steps"], gen))
            Vp = snap_all(v[0].cpu().numpy().astype(np.float64), B, fm)
            p1, p2 = offsets_to_ctrl(Vp[E[:, 0]], Vp[E[:, 1]], o[0].cpu().numpy().astype(np.float64))
            geo["pred"] = (Vp, p1, p2)
        for (v, e, f), kd in sorted(det["pairs"].items()):
            row["pairs"].append({"v": v, "e": int(e), "f": int(f), "kind": kd,
                                 **{f"kink_{g}": kink(G[0], E, G[1], G[2], v, e, f) for g, G in geo.items()}})
        for kd, lst in det["ambiguous"].items():
            for v, cand in lst:
                for e, f in cand:
                    row["ambiguous"].append({"v": v, "e": int(e), "f": int(f), "rule": kd,
                                             **{f"kink_{g}": kink(G[0], E, G[1], G[2], v, e, f) for g, G in geo.items()}})
        d = os.path.join(a.out, a.split, name)
        os.makedirs(d, exist_ok=True)
        for g, G in geo.items():
            write_vtk(os.path.join(d, f"chains_{g}.vtk"), G[0], E, G[1], G[2], det, ch)
        rows.append(row)
        kg = [p["kink_gt"] for p in row["pairs"]]
        print(f"{name:22s} pairs {dict(row['n_pairs'])} raw3d {row['n_p3_raw']} corners {row['n_corner']} "
              f"chains {row['n_chains']} amb {row['n_ambiguous']} | gt kink med {np.median(kg):.1f} "
              f"p90 {np.percentile(kg, 90):.1f} max {np.max(kg):.1f}", flush=True)

    def stats(x):
        x = np.asarray(x)
        return None if not len(x) else {"n": int(len(x)), "median": float(np.median(x)),
                                         "p90": float(np.percentile(x, 90)), "max": float(x.max())}
    summ = {}
    for g in (("gt", "pred") if model is not None else ("gt",)):
        for kd in ("feature", "2d", "3d"):
            summ[f"{g}_{kd}"] = stats([p[f"kink_{g}"] for r in rows for p in r["pairs"] if p["kind"] == kd])
        summ[f"{g}_ambiguous"] = stats([p[f"kink_{g}"] for r in rows for p in r["ambiguous"]])
    with open(os.path.join(a.out, f"{a.split}.json"), "w") as fh:
        json.dump({"summary": summ, "rows": rows}, fh, indent=1, default=lambda o: dict(o) if isinstance(o, collections.Counter) else str(o))
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
