#!/usr/bin/env python3
"""Stage 2, round 2 (M3): set pointer robust to stage-1 corners.

Changes against pointer_conn.py (round 1: 62-71 % exact topology on stage-1 corners, failures = seam corners
that slide past each other):
  1. sliding augmentation: in training every corner moves inside its own geometry, not isotropically:
     seam corners (2 patches) along the seam tangent, surface corners (1 patch) in the tangent plane, geometric
     corners (>= 3 patches) stay; slide scale drawn per item up to --slide (stage-1 sliding median 0.04-0.07,
     p90 0.13-0.19), plus a small normal component;
  2. patch features: each corner carries its surface-patch multi-hot (training: dataset vlab; inference: from the
     snapped corner, same tolerance as the dataset);
  3. snap before the pointer (inference): onto a seam junction, else onto the nearest seam, else onto the surface
     (block-free, the blocks are not known yet);
  4. selection: greedy plus K sampled decodings, ranked by (all corners used, manifold) then log-likelihood.

    python pointer_conn2.py --data data/flow_canon_full.pt --vtx-ckpt runs/vtx_full_s0/best.pt --out runs/ptr2_s0
"""
import argparse
import copy
import json
import math
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

sys.path.insert(0, "/opt/stack/meshtron")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pointer_conn as PC  # noqa: E402
from build_flow_dataset import surface_multihot  # noqa: E402
from flow_geo_var import Norm  # noqa: E402
from train import batch_points, pad  # noqa: E402

SRC = {"tistos": "/opt/stack/meshtron/experiments/flow_topology/datagen/runs", "machine": "/opt/stack/meshtron/experiments/flow_topology/relabel/out"}
FEAT_CACHE = "/opt/stack/meshtron/experiments/flow_topology/cache/features_v2"


def npz_of(name):
    return os.path.join(SRC[name.split("_")[0]], name, "sample.npz")


class SetPointer2(PC.SetPointer):
    """SetPointer with the corner patch multi-hot as extra input."""

    def __init__(self, *args, n_lab=7, **kw):
        super().__init__(*args, **kw)
        d = self.venc[0].out_features
        self.venc = nn.Sequential(nn.Linear(3 * 13 + n_lab, d), nn.GELU(), nn.Linear(d, d))

    def memory(self, V, vmask, pts, pmask, lab=None):
        z = self.enc(pts, pmask)
        e = self.venc(torch.cat([PC.fourier(V), lab], -1))
        for blk in self.vblocks:
            e = blk(e, src_key_padding_mask=~vmask)
        return e, z


# ----------------------------------------------------------------------- geometry per item
def slide_projectors(it, fm, raw):
    """[M,3,3] projector of the allowed sliding per corner (seam: t t^T, surface: I - n n^T, corner: 0)."""
    V = it["verts"].double().numpy()
    nl = it["vlab"].sum(1).numpy()
    S = np.asarray(raw["surface_points"], float)
    tree = cKDTree(S)
    cs = fm.seam_curves
    out = np.zeros((len(V), 3, 3))
    for i, p in enumerate(V):
        if nl[i] >= 3:
            continue
        if nl[i] == 2 and cs.n_curves:
            _, c, t, _ = cs.nearest(p[None])
            c, t = int(c[0]), float(t[0])
            lo, hi = int(cs.offset[c]), int(cs.offset[c + 1])
            s = cs.arclen[lo:hi]; L = float(s[-1]); h = max(0.01 * L, 1e-4)
            a, b = cs.sample_segment(c, max(t - h, 0), min(t + h, L), 2)
            tg = b - a; tg /= max(np.linalg.norm(tg), 1e-30)
            out[i] = np.outer(tg, tg)
        elif nl[i] == 1:
            _, j = tree.query(p, k=32)
            Q = S[j] - S[j].mean(0)
            n = np.linalg.svd(Q, full_matrices=False)[2][-1]
            out[i] = np.eye(3) - np.outer(n, n)
        else:
            out[i] = np.eye(3)
    return out


def prepare(items):
    from meshtron.geometry.geometry_features import FeatureModelV2
    for it in items:
        if "slide" in it:
            continue
        raw = np.load(npz_of(it["name"]), allow_pickle=False)
        fm = FeatureModelV2(npz_of(it["name"]), cache_dir=FEAT_CACHE)
        it["slide"] = torch.tensor(slide_projectors(it, fm, raw), dtype=torch.float32)


def snap_free(P, fm, tol_v=0.06, tol_e=0.04):
    """Block-free snap of corners onto the geometry: seam junction, else seam, else surface."""
    cs = fm.seam_curves
    out = P.copy()
    J = np.asarray(cs.ep_pt, float).reshape(-1, 3)
    jt = cKDTree(J) if len(J) else None
    for i, p in enumerate(P):
        if jt is not None:
            d, j = jt.query(p)
            if d < tol_v:
                out[i] = J[j]; continue
        if cs.n_curves:
            d, _, _, q = cs.nearest(p[None])
            if d[0] < tol_e:
                out[i] = q[0]; continue
        out[i] = fm.surface_nearest(p[None])[2][0]
    return out


# ----------------------------------------------------------------------- batches / decoding
def corrupt_labels(L, rng, p):
    """Per corner with probability p: drop one of its labels (2 -> 1 patch) or add a label of another corner of
    the same item (1 -> 2 patches) -- the errors a snap onto the wrong entity makes."""
    L = L.clone()
    for i in np.flatnonzero(rng.random(len(L)) < p):
        on = torch.nonzero(L[i]).flatten().tolist()
        if len(on) >= 2 and rng.random() < 0.5:
            L[i, on[rng.integers(len(on))]] = 0
        else:
            other = torch.nonzero(L.sum(0)).flatten().tolist()
            L[i, other[rng.integers(len(other))]] = 1
    return L


def make_batch(items, nrm, dev, rng, slide=0.0, sigma_n=0.005, labels="gt", label_noise=0.0):
    Vs, Ls, tgs = [], [], []
    for it in items:
        M = len(it["verts"])
        perm = rng.permutation(M)
        inv = np.empty(M, np.int64); inv[perm] = np.arange(M)
        V = it["verts"].double().numpy()
        if slide > 0:
            st = rng.uniform(0, slide)
            eps = rng.standard_normal((M, 3)) * st
            V = V + np.einsum("mij,mj->mi", it["slide"].double().numpy(), eps) + sigma_n * rng.standard_normal((M, 3))
        Vs.append(torch.tensor(V[perm], dtype=torch.float32))
        L = it["vlab"].float()[torch.as_tensor(perm)]
        if labels == "none":
            L = torch.zeros_like(L)
        elif label_noise > 0:
            L = corrupt_labels(L, rng, label_noise)
        Ls.append(L)
        tgs.append(torch.as_tensor(inv)[it["conn"].long().reshape(-1)])
    V = pad(Vs, 0.0, torch.float32).to(dev)
    lab = pad(Ls, 0.0, torch.float32).to(dev)
    vmask = pad([torch.ones(len(v), dtype=torch.bool) for v in Vs], False, torch.bool).to(dev)
    Mx = V.shape[1]
    t = pad([torch.cat([g, torch.tensor([Mx])]) for g in tgs], -100).to(dev)
    ptr_in = torch.cat([torch.full((len(items), 1), -1, device=dev), t[:, :-1]], 1)
    ptr_in[ptr_in == -100] = 0
    return nrm.enc_v(V), lab, vmask, ptr_in, t


@torch.no_grad()
def decode(model, e, z, vmask, temp=0.0, gen=None, max_blocks=256):
    """Pointer decoding with validity masks; temp 0 = greedy. Returns (blocks, log-likelihood)."""
    M = int(vmask[0].sum()); dev = e.device
    max_blocks = min(max_blocks, (model.pos.num_embeddings - 2) // 8)
    seq, blocks, face_count, ll = [-1], [], {}, 0.0
    for _ in range(8 * max_blocks + 1):
        lg = model.logits(e, z, vmask, torch.tensor([seq], device=dev))[0, -1, :M + 1].float().clone()
        pos = (len(seq) - 1) % 8
        if pos:
            lg[M] = -1e9
            for c in seq[len(seq) - pos:]:
                lg[c] = -1e9
        lp = F.log_softmax(lg, -1)
        if temp > 0:
            nxt = int(torch.multinomial(F.softmax(lg / temp, -1), 1, generator=gen))
        else:
            nxt = int(lg.argmax())
        ll += float(lp[nxt])
        if nxt == M:
            break
        seq.append(nxt)
        if (len(seq) - 1) % 8 == 0:
            b = seq[-8:]
            keys = [tuple(sorted(b[i] for i in f)) for f in PC.FACES]
            if any(face_count.get(k, 0) >= 2 for k in keys):
                seq = seq[:-8]; break
            for k in keys:
                face_count[k] = face_count.get(k, 0) + 1
            blocks.append(b)
    return blocks, ll


def select(model, e, z, vmask, k, gen):
    """Greedy + k samples; rank by (all corners used, -#unused) then log-likelihood."""
    M = int(vmask[0].sum())
    cands = [decode(model, e, z, vmask)] + [decode(model, e, z, vmask, 0.7, gen) for _ in range(k)]
    def key(c):
        used = len({x for b in c[0] for x in b})
        return (used == M, used, c[1])
    return cands[0][0], max(cands, key=key)[0], [c[0] for c in cands]


# ----------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, nrm, items, dev, vtx=None, k=8, seed=1234, slide=0.0, labels="gt"):
    """Exact topology (isomorphism of block complexes) on GT corners, slid GT corners, or stage-1 corners
    (snapped + labelled). Reports greedy, selected (GT-free) and oracle (any candidate) rates."""
    import networkx as nx
    from networkx.algorithms.isomorphism import categorical_node_match
    from eval_ptr_iso import complex_graph
    from meshtron.geometry.geometry_features import FeatureModelV2
    import flow_vertex as FVX
    model.eval()
    rng = np.random.default_rng(seed); gen = torch.Generator(device=dev); gen.manual_seed(seed)
    acc = Counter()
    for it in items:
        pts, pmask = batch_points([it], dev)
        if vtx is None:
            x, lab, vmask, _, _ = make_batch([it], nrm, dev, rng, slide, labels=labels)
        else:
            vm, vn = vtx
            zv = vm.enc(pts, pmask)
            n = int(vm.count_logits(zv).argmax(-1)[0])
            P = vn.dec_v(FVX.sample(vm, zv, [n], 32, gen)[0]).cpu().double().numpy()
            fm = FeatureModelV2(npz_of(it["name"]), cache_dir=FEAT_CACHE)
            raw = np.load(npz_of(it["name"]), allow_pickle=False)
            P = snap_free(P, fm)
            tol = 6e-3 * np.linalg.norm(np.ptp(np.asarray(raw["surface_points"]), 0))
            lab = torch.tensor(surface_multihot(P, raw, tol), device=dev)[None]
            if labels == "none":
                lab = torch.zeros_like(lab)
            x = nrm.enc_v(torch.tensor(P, dtype=torch.float32, device=dev))[None]
            vmask = torch.ones(1, len(P), dtype=torch.bool, device=dev)
        e, z = model.memory(x, vmask, pts, pmask, lab)
        greedy, best, cands = select(model, e, z, vmask, k, gen)
        Gg = complex_graph(it["conn"].long().numpy())
        iso = lambda b: len(b) == len(it["conn"]) and nx.is_isomorphic(  # noqa: E731
            complex_graph(b), Gg, node_match=categorical_node_match("kind", None))
        acc["greedy"] += iso(greedy); acc["selected"] += iso(best); acc["oracle"] += any(iso(c) for c in cands)
        acc["n"] += 1
    n = acc.pop("n")
    return {k2: v / n for k2, v in acc.items()} | {"n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vtx-ckpt", required=True)
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--slide", type=float, default=0.12, help="max sliding scale (dataset units)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--labels", choices=("gt", "none"), default="gt", help="corner patch features on / off")
    ap.add_argument("--label-noise", type=float, default=0.0, help="per-corner label corruption in training")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = "cuda"
    data = torch.load(a.data, weights_only=False)
    tr, va = data["train"], data["val"]
    t0 = time.time()
    prepare(tr + [it for sp in data if sp.startswith("val") and isinstance(data[sp], list) for it in data[sp]])
    print(f"[ptr2] slide projectors {time.time() - t0:.0f}s", flush=True)
    nrm = Norm(data, tr).to(dev)
    max_len = max(it["conn"].numel() for it in tr) * 2 + 2
    model = SetPointer2(a.d, a.layers, a.heads, max_len).to(dev)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    os.makedirs(a.out, exist_ok=True)
    import flow_vertex as FVX
    vm, vn, _ = FVX.load(a.vtx_ckpt, data)
    print(f"[ptr2] params {sum(p.numel() for p in model.parameters())/1e6:.2f} M | train {len(tr)} | slide {a.slide}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * math.ceil(len(tr) / a.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 300) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
    hist, t0 = [], time.time()
    for ep in range(1, a.epochs + 1):
        model.train(); tot = 0.0; order = rng.permutation(len(tr))
        for i in range(0, len(tr), a.bs):
            chunk = [tr[j] for j in order[i:i + a.bs]]
            x, lab, vmask, ptr_in, t = make_batch(chunk, nrm, dev, rng, a.slide, labels=a.labels, label_noise=a.label_noise)
            pts, pmask = batch_points(chunk, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                e, z = model.memory(x, vmask, pts, pmask, lab)
                lg = model.logits(e, z, vmask, ptr_in)
            loss = F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), t.reshape(-1), ignore_index=-100)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - 0.999)
            tot += float(loss.detach())
        if ep % a.eval_every == 0 or ep == a.epochs:
            ev = evaluate(ema, nrm, va, dev, vtx=(vm, vn), k=a.k, labels=a.labels)
            torch.save({"model": ema.state_dict(), "args": vars(a), "max_len": max_len, "epoch": ep}, f"{a.out}/ep{ep}.pt")
            hist.append({"epoch": ep, "loss": tot / math.ceil(len(tr) / a.bs), **ev})
            print(f"[ptr2] ep {ep:4d} loss {hist[-1]['loss']:.4f} | val stage-1 {ev} ({time.time()-t0:.0f}s)", flush=True)
    torch.save({"model": ema.state_dict(), "args": vars(a), "max_len": max_len}, f"{a.out}/last.pt")
    summ = {}
    for sp in [k for k in data if k.startswith("val") and isinstance(data[k], list) and data[k]]:
        summ[sp] = {"stage1": evaluate(ema, nrm, data[sp], dev, vtx=(vm, vn), k=a.k, labels=a.labels)}
    summ["train_s"] = time.time() - t0
    print(f"[ptr2] RESULT {json.dumps(summ)}", flush=True)
    json.dump({"summary": summ, "history": hist}, open(f"{a.out}/result.json", "w"), indent=1)


if __name__ == "__main__":
    main()
