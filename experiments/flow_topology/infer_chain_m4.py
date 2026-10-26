#!/usr/bin/env python3
"""M4 of VARIABLE_TOPOLOGY_PLAN.md: the whole chain without any GT information.

  stage 1   corners as a set of predicted size (flow_vertex.py)
  snap      block-free snap onto junction / seam / surface (pointer_conn2.snap_free)
  stage 2   set pointer (pointer_conn2.py): greedy + K sampled block structures; every candidate goes through the
            geometry-first feature assignment (domain_partition_3D curve_postprocess.py, branch feat/feature-curves):
            candidates whose feature curves are not completely covered are ranked last, then all corners used,
            then log-likelihood
  features  of the chosen structure: corners onto their seams / junctions, feature edges = exact seam arcs with
            G1 cubics (known tokens for stage 3)
  stage 3   flow_geo_var.py model with known tokens (geometric corners + feature edges fixed)
  mesh      feature edges as the exact arcs, the rest as stage-3 cubics -> curved TFI; then the TFI curve
            optimisation (curve_opt.optimise, hinge) with the arcs fixed
Metrics per item: block count, topology isomorphic to GT, feature completeness, mesh metrics before / after the
optimisation, O-ring kinks.

    PYTHONPATH=<dp3d hex3d_algohex> python infer_chain_m4.py --split val_full --out-dir chain/m4_val_full
"""
import argparse
import importlib.util
import json
import os
import sys

sys.path.insert(0, "/opt/stack/meshtron")
sys.path.insert(0, "/opt/stack/meshtron/experiments/flow_topology")
sys.path.insert(0, "/opt/stack/meshtron/experiments/flow_topology/rl_reward")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from infer_chain_v2 import snap_all  # noqa: E402,F401  (import order: before meshtron.geometry)
import continuity as C  # noqa: E402
import curve_opt as CO  # noqa: E402
import flow_geo_var as FV  # noqa: E402
import flow_vertex as FVX  # noqa: E402
import pointer_conn2 as P2  # noqa: E402
from build_flow_dataset import surface_multihot  # noqa: E402
from eval_ptr_iso import complex_graph  # noqa: E402
from meshtron.data.polytron_blocks import ctrl_to_offsets  # noqa: E402
from meshtron.geometry.geometry_features import FeatureModelV2  # noqa: E402
from meshtron.geometry.polytron_tfi import SurfaceProjector  # noqa: E402
from train import batch_points  # noqa: E402

CP_PATH = "/opt/stack/dp3d_keep_bl/experimentell/hex3d_algohex/curve_postprocess.py"
_spec = importlib.util.spec_from_file_location("curve_postprocess_fc", CP_PATH)
cp = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(cp)
HEX_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))


VTK_FACES = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
TETS = ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6))


def seam_order(P, fm, tol=1e-6):
    """Corners lying on a seam curve (after the snap) and, per seam, the pairs of corners that are consecutive
    along it: only those may be joined by a block edge on that seam. Returns (seam_of, allowed) with
    seam_of[i] = set of curves of corner i, allowed = set of sorted corner pairs."""
    cs = fm.seam_curves
    diag = float(np.linalg.norm(np.ptp(np.asarray(fm.surface_points), 0)))
    seam_of = [set() for _ in P]
    on = {}
    for c in range(cs.n_curves):
        Q = cs.segment(c)
        if len(Q) < 2:
            continue
        from scipy.spatial import cKDTree
        d, j = cKDTree(Q).query(P)
        # distance to the polyline (not only its nodes): project onto the two adjacent segments
        hit = []
        for i in np.flatnonzero(d < 0.05):
            best = np.inf
            for a_, b_ in ((max(j[i] - 1, 0), j[i]), (j[i], min(j[i] + 1, len(Q) - 1))):
                ab = Q[b_] - Q[a_]; t = np.clip((P[i] - Q[a_]) @ ab / max(ab @ ab, 1e-30), 0, 1)
                best = min(best, np.linalg.norm(Q[a_] + t * ab - P[i]))
            if best < 1e-3 * diag:
                hit.append(int(i))
        if len(hit) >= 2:
            s_ = cs.arclen[cs.offset[c]:cs.offset[c + 1]]
            # arclength position of each corner on this curve
            tt = []
            for i in hit:
                dd = np.linalg.norm(Q - P[i], axis=1); k = int(dd.argmin()); tt.append(s_[k])
            order = [hit[k] for k in np.argsort(tt)]
            on[c] = order
            for i in hit:
                seam_of[i].add(c)
    allowed = set()
    for c, order in on.items():
        for a_, b_ in zip(order[:-1], order[1:]):
            allowed.add(tuple(sorted((a_, b_))))
        if cs.closed[c] and len(order) > 2:
            allowed.add(tuple(sorted((order[0], order[-1]))))
    return seam_of, allowed


def orientation_ok(V, B):
    """All blocks with positive volume and every interior face traversed in opposite directions by its two
    blocks (consistently oriented block complex): what the TFI's D4 face matching needs."""
    for b in B:
        X = V[np.asarray(b)]
        vol = sum(np.linalg.det(np.stack([X[t[1]] - X[t[0]], X[t[2]] - X[t[0]], X[t[3]] - X[t[0]]])) for t in TETS)
        if vol <= 0:
            return False
    seen = {}
    for b in B:
        for f in VTK_FACES:
            q = [int(b[i]) for i in f]
            key = tuple(sorted(q))
            k = q.index(min(q)); cyc = tuple(q[k:] + q[:k])
            if key in seen:
                o = seen[key]
                rev = (cyc[0],) + tuple(reversed(cyc[1:]))
                if o != rev:
                    return False
            else:
                seen[key] = cyc
    return True


def decode_masked(ptr, e, z, vmask, seam_of, allowed, temp=0.0, gen=None, max_blocks=256):
    """pointer_conn2.decode plus the seam mask: an edge of the block (hex edge between two placed slots) that
    joins two corners of the same seam is allowed only if they are consecutive on that seam."""
    import torch.nn.functional as F_
    M = int(vmask[0].sum()); dev = e.device
    max_blocks = min(max_blocks, (ptr.pos.num_embeddings - 2) // 8)
    nbr = {s_: [t for (u, v) in HEX_EDGES for s2, t in ((u, v), (v, u)) if s2 == s_] for s_ in range(8)}
    seq, blocks, face_count, ll = [-1], [], {}, 0.0
    for _ in range(8 * max_blocks + 1):
        lg = ptr.logits(e, z, vmask, torch.tensor([seq], device=dev))[0, -1, :M + 1].float().clone()
        pos = (len(seq) - 1) % 8
        if pos:
            lg[M] = -1e9
            cur = seq[len(seq) - pos:]
            for c_ in cur:
                lg[c_] = -1e9
            for c_ in range(M):
                for t in nbr[pos]:
                    if t < pos:
                        o = cur[t]
                        if seam_of[c_] & seam_of[o] and tuple(sorted((c_, o))) not in allowed:
                            lg[c_] = -1e9
        lp = F_.log_softmax(lg, -1)
        nxt = int(torch.multinomial(F_.softmax(lg / temp, -1), 1, generator=gen)) if temp > 0 else int(lg.argmax())
        if lg[nxt] < -1e8:
            break
        ll += float(lp[nxt])
        if nxt == M:
            break
        seq.append(nxt)
        if (len(seq) - 1) % 8 == 0:
            b = seq[-8:]
            keys = [tuple(sorted(b[i] for i in f)) for f in VTK_FACES]
            if any(face_count.get(k, 0) >= 2 for k in keys):
                seq = seq[:-8]; break
            for k in keys:
                face_count[k] = face_count.get(k, 0) + 1
            blocks.append(b)
    return blocks, ll


def min_pair(X, diag):
    """Smallest corner-corner distance relative to the geometry diagonal."""
    D = np.linalg.norm(X[:, None] - X[None], axis=-1); np.fill_diagonal(D, np.inf)
    return float(D.min()) / diag


def dup_threshold(items, frac=0.5, q=1.0):
    """Duplicate threshold from the data: frac x the q-th percentile of the smallest GT corner spacing over the
    training geometries (a percentile, not the minimum: single tight GT outliers would make it useless)."""
    return frac * float(np.percentile([min_pair(it["verts"].double().numpy(), diag_of(it)) for it in items], q))


def diag_of(it):
    return float(np.linalg.norm(np.ptp(it["points"][:, :3].double().numpy(), 0)))


def snap_unique(P, fm, thr_abs, tol_v=0.06, tol_e=0.04):
    """pointer_conn2.snap_free without double occupancy: seam junctions are assigned one-to-one (Hungarian within
    tol_v); a corner that collides with another one only because of the snap (raw distance >= thr_abs) falls back
    to its own surface projection. Collisions already present in the raw corners are left for the resampling."""
    from scipy.optimize import linear_sum_assignment
    cs = fm.seam_curves
    J = np.asarray(cs.ep_pt, float).reshape(-1, 3)
    out = P.copy(); taken = {}
    if len(J):
        D = np.linalg.norm(P[:, None] - J[None], axis=-1)
        cost = np.where(D < tol_v, D, 1e6)
        r, c = linear_sum_assignment(cost)
        taken = {int(i): int(j) for i, j in zip(r, c) if cost[i, j] < 1e6}
    for i, p in enumerate(P):
        if i in taken:
            out[i] = J[taken[i]]; continue
        if cs.n_curves:
            d, _, _, q = cs.nearest(p[None])
            if d[0] < tol_e:
                out[i] = q[0]; continue
        out[i] = fm.surface_nearest(p[None])[2][0]
    for _ in range(3):
        D = np.linalg.norm(out[:, None] - out[None], axis=-1); np.fill_diagonal(D, np.inf)
        Dr = np.linalg.norm(P[:, None] - P[None], axis=-1)
        bad = [(i, j) for i, j in zip(*np.nonzero(np.triu(D < thr_abs))) if Dr[i, j] >= thr_abs]
        if not bad:
            break
        for i, j in bad:
            k = i if np.linalg.norm(out[i] - P[i]) > np.linalg.norm(out[j] - P[j]) else j
            out[k] = fm.surface_nearest(P[k][None])[2][0]
    return out


def straight_sample(V, B, raw):
    """Sample dict (export_sample layout) of a block structure with straight edges."""
    E = sorted({tuple(sorted((int(b[i]), int(b[j])))) for b in B for i, j in HEX_EDGES})
    edges, ctrl = [], []
    for a, b in E:
        for p, q in ((a, b), (b, a)):
            edges.append([p, q]); ctrl.append([(2 * V[p] + V[q]) / 3, (V[p] + 2 * V[q]) / 3])
    return {"vertices": V, "blocks": np.asarray(B), "edges": np.asarray(edges), "edge_ctrl": np.asarray(ctrl),
            **{k: raw[k] for k in ("surface_points", "surface_tris", "surface_tri_label")}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="/opt/stack/meshtron/experiments/flow_topology/data/flow_canon_full.pt")
    ap.add_argument("--post-data", default="/opt/stack/meshtron/experiments/flow_topology/data/flow_canon_full_post.pt")
    ap.add_argument("--split", default="val_full")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--vtx-ckpt", default="/opt/stack/meshtron/experiments/flow_topology/runs/vtx_full_s0/best.pt")
    ap.add_argument("--ptr-ckpt", default="/opt/stack/meshtron/experiments/flow_topology/runs/ptr2_nolab/ep50.pt")
    ap.add_argument("--var-ckpt", default="/opt/stack/meshtron/experiments/flow_topology/runs/varm_mask_s0/last.pt")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seam-mask", action="store_true", help="only consecutive seam corners may form a block edge")
    ap.add_argument("--names", default=None, help="comma-separated item names (subset)")
    ap.add_argument("--vtx-steps", type=int, default=128, help="Euler steps of stage 1")
    ap.add_argument("--chain-tries", type=int, default=8,
                    help="stage-1 draws until the corner set has no duplicate AND a pointer candidate is oriented, "
                         "feature-complete and uses all corners (1 = single draw)")
    ap.add_argument("--no-dup-check", action="store_true", help="do not skip duplicate corner sets before the pointer")
    ap.add_argument("--old-snap", action="store_true", help="pointer_conn2.snap_free (may stack corners)")
    ap.add_argument("--target-h", type=float, default=0.08)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    dev = "cuda"
    os.makedirs(a.out_dir, exist_ok=True)
    data = torch.load(a.data, weights_only=False)
    pdata = torch.load(a.post_data, weights_only=False)
    vm, vn, _ = FVX.load(a.vtx_ckpt, data)
    ck = torch.load(a.ptr_ckpt, weights_only=False); pa = ck["args"]
    ptr = P2.SetPointer2(pa["d"], pa["layers"], pa["heads"], ck["max_len"]).to(dev)
    ptr.load_state_dict(ck["model"]); ptr.eval()
    pnrm = FV.Norm(data, data["train"]).to(dev)
    var, vnrm, vargs = FV.load(a.var_ckpt, pdata)
    import networkx as nx
    from networkx.algorithms.isomorphism import categorical_node_match
    gen = torch.Generator(device=dev); gen.manual_seed(1234)
    thr = dup_threshold(data["train"])
    print(f"[m4] duplicate threshold {thr:.2e} x diag", flush=True)
    rows = []
    names = set(a.names.split(",")) if a.names else None
    for it in data[a.split][:a.n]:
        name = it["name"]
        if names is not None and name not in names:
            continue
        gen.manual_seed(1234 + sum(map(ord, name)))             # same stage-1 corners per item in every variant
        npz = P2.npz_of(name)
        raw = {k: np.asarray(v) for k, v in np.load(npz, allow_pickle=False).items()}
        fm = FeatureModelV2(npz, cache_dir=P2.FEAT_CACHE)
        S = raw["surface_points"]; tol = 6e-3 * np.linalg.norm(np.ptp(S, 0))
        row = {"name": name}
        pts, pmask = batch_points([it], dev)
        # ---- stage 1 + snap
        diag = diag_of(it)
        best, tries = None, []                                 # best = (key, V, B, post, q, scored, n, mp)
        for tr_ in range(a.chain_tries):
            gen.manual_seed(1234 + sum(map(ord, name)) + 7919 * tr_)      # try 0 = the earlier single draw
            with torch.no_grad():
                zv = vm.enc(pts, pmask)
                n = int(vm.count_logits(zv).argmax(-1)[0])
                P = vn.dec_v(FVX.sample(vm, zv, [n], a.vtx_steps, gen)[0]).cpu().double().numpy()
            Ps = P2.snap_free(P, fm) if a.old_snap else snap_unique(P, fm, thr * diag)
            mp = min_pair(Ps, diag); dup = mp < thr
            if dup and tr_ < a.chain_tries - 1 and not a.no_dup_check:
                tries.append({"try": tr_, "dup": True}); continue           # duplicate: no pointer, draw again
            # ---- stage 2 candidates, ranked by orientation, feature completeness, corner usage, likelihood
            with torch.no_grad():
                x = pnrm.enc_v(torch.tensor(Ps, dtype=torch.float32, device=dev))[None]
                vmask = torch.ones(1, len(Ps), dtype=torch.bool, device=dev)
                lab0 = torch.zeros(1, len(Ps), 7, device=dev)
                e, z = ptr.memory(x, vmask, pts, pmask, lab0)
                if a.seam_mask:
                    seam_of, allowed = seam_order(Ps, fm)
                    dec = lambda temp: decode_masked(ptr, e, z, vmask, seam_of, allowed, temp, gen)  # noqa: E731
                else:
                    dec = lambda temp: P2.decode(ptr, e, z, vmask, temp, gen)  # noqa: E731
                cands = [dec(0.0)] + [dec(0.7) for _ in range(a.k)]
            scored = []
            for blocks, ll in cands:
                if not blocks:
                    continue
                Bc = np.asarray(blocks)
                used = sorted({int(c) for b in Bc for c in b}); remap = {c: i for i, c in enumerate(used)}
                Bc = np.vectorize(remap.get)(Bc); Vc = Ps[used]
                try:
                    post = cp.postprocess(straight_sample(Vc, Bc, raw), verbose=False)
                    q = json.loads(str(post["post_quality"]))
                    complete = bool(q["feature_complete"])
                except Exception as ex:  # noqa: BLE001
                    post, q, complete = None, {"error": str(ex)}, False
                orient = orientation_ok(Vc, Bc)
                scored.append(((orient, complete, len(used) == len(Ps), not dup, ll), Vc, Bc, post, q))
            scored.sort(key=lambda s_: s_[0], reverse=True)
            ok = bool(scored) and all(scored[0][0][:4])
            tries.append({"try": tr_, "dup": bool(dup), "ok": ok, "n_complete": sum(s_[0][1] for s_ in scored)})
            if scored and (best is None or scored[0][0] > best[0]):
                best = (scored[0][0], *scored[0][1:], scored, n, mp)
            if ok:
                break
        row["tries"] = tries
        if best is None:
            row["error"] = "no candidate"; rows.append(row); continue
        (orient, complete, all_used, nodup, ll), V, B, post, q, scored, n, mp = best
        row.update(n_corners=n, chain_tries=len(tries), min_pair=mp, duplicate=not nodup)
        row["orientation_ok"] = orient
        row["n_candidates_orient"] = sum(s_[0][0] for s_ in scored)
        row.update(n_blocks=len(B), feature_complete=complete, n_candidates_complete=sum(s_[0][1] for s_ in scored),
                   topology_iso=bool(len(B) == len(it["conn"]) and nx.is_isomorphic(
                       complex_graph(B), complex_graph(it["conn"].long().numpy()), node_match=categorical_node_match("kind", None))))
        if post is None:
            row["error"] = q.get("error"); rows.append(row); continue
        # ---- features of the chosen structure -> known tokens
        Vp = post["post_vertices"]
        smp = straight_sample(V, B, raw)
        Eu, _, dmap = cp.undirected(smp)
        pc = np.zeros((len(Eu), 2, 3)); feat = np.zeros(len(Eu), bool); arcs = {}
        ao, apl = post["post_arc_offset"], post["post_arc_polyline"]
        for d, (k, rev) in enumerate(dmap):
            c_ = post["post_edge_ctrl"][d]
            pc[k] = (c_[1], c_[0]) if rev else (c_[0], c_[1])
            feat[k] |= bool(post["post_feature_edge"][d])
            if ao[d + 1] > ao[d] and k not in arcs:
                Pa = apl[ao[d]:ao[d + 1]]; arcs[k] = Pa[::-1] if rev else Pa
        role = post["post_vertex_role"]
        o = np.zeros((len(Eu), 6))
        for k, (p, q_) in enumerate(Eu):
            o[k] = ctrl_to_offsets(Vp[p], Vp[q_], pc[k, 0], pc[k, 1])
        item = {"verts": torch.tensor(Vp, dtype=torch.float32), "conn": torch.tensor(B), "edges": torch.tensor(Eu),
                "vlab": torch.tensor(surface_multihot(Vp, raw, tol)),
                "elab": torch.tensor(surface_multihot((Vp[Eu[:, 0]] + Vp[Eu[:, 1]]) / 2, raw, tol)),
                "eoff": torch.tensor(o, dtype=torch.float32), "points": it["points"], "feature": torch.tensor(feat),
                "kv": torch.tensor(role == cp.ROLE["corner"]), "ke": torch.tensor(feat)}
        b = FV.collate([item], dev)
        with torch.no_grad():
            zz = var.enc(pts, pmask)
            x1v = vnrm.enc_v(b["V"])
            xv, xo = FV.sample(var, zz, b, x1v, vargs["steps"], gen, x1o_known=vnrm.enc_o(b["O"]))
        V3 = vnrm.dec_v(xv)[0].cpu().double().numpy(); O3 = vnrm.dec_o(xo)[0].cpu().double().numpy()
        V3 = np.where((role == cp.ROLE["corner"])[:, None], Vp, P2.snap_free(V3, fm))
        for k in range(len(Eu)):                               # feature corners stay on their seam: use Vp there
            if feat[k]:
                V3[Eu[k]] = Vp[Eu[k]]
        from meshtron.data.polytron_blocks import offsets_to_ctrl
        c1, c2 = offsets_to_ctrl(V3[Eu[:, 0]], V3[Eu[:, 1]], O3)
        c1[feat], c2[feat] = pc[feat, 0], pc[feat, 1]
        cur = CO.to_curves(V3, Eu, c1, c2, arcs)
        row["mesh_tfi"] = CO.mesh_eval(V3, B, cur, raw, a.target_h, f"{a.out_dir}/{name}_tfi.vtk")
        # ---- TFI curve optimisation with the arcs fixed
        vpatch = [set((np.flatnonzero(l) + 1).tolist()) for l in surface_multihot(V3, raw, tol)]
        det = C.detect(V3, B, Eu, (c1, c2), fm, vpatch)
        proj = SurfaceProjector(S, raw["surface_tris"], raw["surface_tri_label"])
        cnt, per = C.block_faces(B)
        uf = list({tuple(sorted(f)): f for fs in per for f in fs}.values())
        bq = [f for f in uf if cnt[tuple(sorted(f))] == 1]
        qlab = C.quad_patch(bq, V3, Eu, (c1, c2), fm, vpatch)
        share2 = C.edge_face_sets(Eu, bq)
        epat = [set(int(qlab[qq]) for qq in share2[k]) for k in range(len(Eu))]
        try:
            o1, o2, _ = CO.optimise(V3, B, Eu, c1, c2, arcs, det, proj, epat)
            if not (np.isfinite(o1).all() and np.isfinite(o2).all()):
                raise FloatingPointError("non-finite control points after the optimisation")
        except Exception as ex:  # noqa: BLE001
            row["opt_error"] = f"{type(ex).__name__}: {ex}"
            o1, o2 = c1, c2                                   # fall back to the stage-3 curves
        cur8 = CO.to_curves(V3, Eu, o1, o2, arcs)
        row["mesh_opt"] = CO.mesh_eval(V3, B, cur8, raw, a.target_h, f"{a.out_dir}/{name}_opt.vtk")
        k8 = CO.pair_kinks(cur8, Eu, det)
        row["ring_kink_median"] = float(np.median(k8["feature_closed"])) if k8.get("feature_closed") else None
        rows.append(row)
        mt, mo = row["mesh_tfi"], row["mesh_opt"]
        if not (np.isfinite(V3).all() and np.isfinite(c1).all() and np.isfinite(c2).all()):
            row["stage3_nonfinite"] = True
        print(f"{name:20s} n {n} blocks {len(B)} iso {row['topology_iso']} complete {complete} "
              f"({row['n_candidates_complete']}/{len(scored)}) | tfi unc {mt.get('uncovered', -1):.3f} | "
              f"opt unc {mo.get('uncovered', -1):.3f} inv {100 * mo.get('inv_share', 0):.3f}% sj1 {mo.get('sj_p1', 0):.2f}",
              flush=True)
        json.dump(rows, open(f"{a.out_dir}/rows.json", "w"), indent=1, default=float)
    ok = [r for r in rows if "mesh_opt" in r and "uncovered" in r["mesh_opt"]]
    summ = {"n": len(rows), "topology_iso": float(np.mean([r.get("topology_iso", False) for r in rows])),
            "feature_complete": float(np.mean([r.get("feature_complete", False) for r in rows]))}
    for key in ("mesh_tfi", "mesh_opt"):
        mm = [r[key] for r in ok if "uncovered" in r[key]]
        summ[key] = {"uncovered_mean": float(np.mean([m["uncovered"] for m in mm])),
                     "uncovered_median": float(np.median([m["uncovered"] for m in mm])),
                     "inv_mean_pct": float(100 * np.mean([m["inv_share"] for m in mm])),
                     "no_inverted": int(sum(m["inv_share"] == 0 for m in mm)),
                     "sj_p1_median": float(np.median([m["sj_p1"] for m in mm])),
                     "nonortho_max_median": float(np.median([m["nonortho_max"] for m in mm])), "n": len(mm)}
    rk = [r["ring_kink_median"] for r in ok if r.get("ring_kink_median") is not None]
    summ["ring_kink_median"] = float(np.median(rk)) if rk else None
    json.dump(summ, open(f"{a.out_dir}/summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
