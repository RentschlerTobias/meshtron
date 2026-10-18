#!/usr/bin/env python3
"""Stage 3 of VARIABLE_TOPOLOGY_PLAN.md: Bezier edge curves (+ corner refinement) for a GIVEN block structure of
ANY size, by flow matching over vertex and edge tokens. Topology-agnostic version of flow_geo.py:

  - no fixed topology object: every item brings its own corners, blocks, edges; batches are padded + masked;
  - token features from the item's own graph only: surface-label multi-hot, label count, boundary flag, valence,
    random-walk positional encoding (RWPE, sign-free) of the block-edge graph; edge tokens get the label
    multi-hot of the edge and the RWPE of both end vertices;
  - self-attention with a learned bias per head and graph distance (hops in the vertex/edge incidence graph,
    Graphormer style) instead of the mean-template position;
  - global normalisation: corners mapped by the dataset box to [-1, 1], chord-relative offsets by one global
    mean / std (no per-token statistics);
  - source distribution: corners = input corners + sigma noise (input = GT in training, stage-1 corners at
    inference), offsets = pure noise. Rectified flow x_t = (1-t) x0 + t x1, target x1 - x0, logit-normal t.
    Edge tokens also see the current chord (midpoint, direction * length) of their corners.

    python flow_geo_var.py --data data/flow_canon_papaya.pt --out runs/var_s0
"""
import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flow_geo import bezier_pts, modulate  # noqa: E402
from model import Attn, PointPerceiver, mlp  # noqa: E402
from train import batch_points, pad  # noqa: E402

N_RW = 16          # random-walk PE steps
MAX_HOP = 8        # graph distance buckets 0..MAX_HOP (larger = MAX_HOP)
HEX_FACES = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))


# ----------------------------------------------------------------------- per-item graph features
def rwpe(n, edges, k=N_RW):
    """Diagonal of the random-walk matrix powers P^1..P^k (sign- and basis-free node encoding)."""
    A = np.zeros((n, n))
    A[edges[:, 0], edges[:, 1]] = 1; A[edges[:, 1], edges[:, 0]] = 1
    P = A / np.maximum(A.sum(1, keepdims=True), 1)
    out, Pk = [], np.eye(n)
    for _ in range(k):
        Pk = Pk @ P
        out.append(np.diag(Pk))
    return np.stack(out, 1)


def item_graph(it):
    """Topology features of one item (cached in the item dict under 'g')."""
    if "g" in it:
        return it["g"]
    V, E, B = it["verts"], it["edges"].long().numpy(), it["conn"].long().numpy()
    M, Ne = len(V), len(E)
    cnt = {}
    for b in B:
        for f in HEX_FACES:
            k = tuple(sorted(int(b[i]) for i in f)); cnt[k] = cnt.get(k, 0) + 1
    bverts = {v for k, c in cnt.items() if c == 1 for v in k}
    bedges = set()                                       # boundary edges from the ordered quad cycles
    for b in B:
        for f in HEX_FACES:
            q = [int(b[i]) for i in f]
            if cnt[tuple(sorted(q))] == 1:
                for i in range(4):
                    bedges.add(tuple(sorted((q[i], q[(i + 1) % 4]))))
    deg = np.bincount(E.ravel(), minlength=M)
    vl, el = it["vlab"].float().numpy(), it["elab"].float().numpy()
    rw = rwpe(M, E)
    vfeat = np.concatenate([vl, vl.sum(1, keepdims=True) / 3, np.array([[v in bverts] for v in range(M)], float),
                            np.eye(5)[np.clip(deg, 2, 6) - 2], rw], 1)
    efeat = np.concatenate([el, el.sum(1, keepdims=True) / 3,
                            np.array([[tuple(sorted(map(int, e))) in bedges] for e in E], float),
                            rw[E[:, 0]] + rw[E[:, 1]], np.abs(rw[E[:, 0]] - rw[E[:, 1]])], 1)
    # incidence graph: nodes 0..M-1 vertices, M..M+E-1 edges; edge node linked to its two vertices
    rows = np.concatenate([E[:, 0], E[:, 1], M + np.arange(Ne), M + np.arange(Ne)])
    cols = np.concatenate([M + np.arange(Ne), M + np.arange(Ne), E[:, 0], E[:, 1]])
    G = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(M + Ne, M + Ne))
    hop = shortest_path(G, unweighted=True, directed=False)
    hop = np.minimum(np.nan_to_num(hop, posinf=MAX_HOP), MAX_HOP).astype(np.int64)
    it["g"] = {"vfeat": torch.tensor(vfeat, dtype=torch.float32), "efeat": torch.tensor(efeat, dtype=torch.float32),
               "hop": torch.tensor(hop), "M": M, "E": Ne}
    return it["g"]


def permute_item(it, rng):
    """Same item with randomly renumbered corners: conn / edges renumbered, edges re-sorted (a < b), the two
    Bezier offset halves swapped where an edge flips direction; cached graph features permuted alongside."""
    g = item_graph(it)
    M = len(it["verts"])
    perm = rng.permutation(M)                                   # new id -> old id
    inv = np.empty(M, np.int64); inv[perm] = np.arange(M)       # old id -> new id
    E = it["edges"].long().numpy()
    En = inv[E]; flip = En[:, 0] > En[:, 1]; En = np.sort(En, 1)
    oe = np.lexsort((En[:, 1], En[:, 0]))                       # new edge j <- old edge oe[j]
    O = it["eoff"].clone()
    O[torch.as_tensor(flip)] = torch.cat([O[torch.as_tensor(flip)][:, 3:], O[torch.as_tensor(flip)][:, :3]], 1)
    out = dict(it)
    out["verts"] = it["verts"][torch.as_tensor(perm)]
    out["vlab"] = it["vlab"][torch.as_tensor(perm)]
    out["conn"] = torch.as_tensor(inv[it["conn"].long().numpy()])
    out["edges"] = torch.as_tensor(En[oe])
    out["eoff"] = O[torch.as_tensor(oe)]
    out["elab"] = it["elab"][torch.as_tensor(oe)]
    if "feature" in it:
        out["feature"] = it["feature"][torch.as_tensor(oe)]
    idx = torch.as_tensor(np.concatenate([perm, M + oe]))
    out["g"] = {"vfeat": g["vfeat"][torch.as_tensor(perm)], "efeat": g["efeat"][torch.as_tensor(oe)],
                "hop": g["hop"][idx][:, idx], "M": g["M"], "E": g["E"]}
    for k in ("kv", "ke"):
        out.pop(k, None)
    return out


def geo_corners(it):
    """Corners on >= 3 surface patches (seam junctions): known exactly from the geometry."""
    return it["vlab"].sum(1) >= 2.5


def known_masks(it, rng, policy):
    """(kv [M], ke [E]) bool masks of tokens fed as known.
    'all': geometric corners + feature edges; 'none'; 'recipe' (training): 60 % all, 20 % a random half of
    them, 10 % all + 10 % of the other edges, 10 % none."""
    M, E = len(it["verts"]), len(it["edges"])
    gv = geo_corners(it)
    fe = it["feature"].bool() if "feature" in it else torch.zeros(E, dtype=torch.bool)
    if policy == "none":
        return torch.zeros(M, dtype=torch.bool), torch.zeros(E, dtype=torch.bool)
    if policy == "all":
        return gv.clone(), fe.clone()
    r = rng.random()
    if r < 0.6:
        return gv.clone(), fe.clone()
    if r < 0.8:
        return gv & torch.as_tensor(rng.random(M) < 0.5), fe & torch.as_tensor(rng.random(E) < 0.5)
    if r < 0.9:
        return gv.clone(), fe | torch.as_tensor(rng.random(E) < 0.1)
    return torch.zeros(M, dtype=torch.bool), torch.zeros(E, dtype=torch.bool)


class Norm:
    """Global normalisation: corners by the dataset box, offsets by one mean / std over all train edges."""

    def __init__(self, data, train):
        rmax = float(data["r_bounds"][1]); z0, z1 = (float(v) for v in data["z_bounds"])
        lo, hi = torch.tensor([-rmax, -rmax, z0]), torch.tensor([rmax, rmax, z1])
        self.c, self.s = (lo + hi) / 2, (hi - lo) / 2
        O = torch.cat([it["eoff"] for it in train])
        self.om, self.os = O.mean(0), O.std(0).clamp(min=1e-6)

    def to(self, dev):
        for k in ("c", "s", "om", "os"):
            setattr(self, k, getattr(self, k).to(dev))
        return self

    def enc_v(self, V):
        return (V - self.c) / self.s

    def dec_v(self, x):
        return x * self.s + self.c

    def enc_o(self, O):
        return (O - self.om) / self.os

    def dec_o(self, x):
        return x * self.os + self.om


def collate(items, dev):
    """Padded batch: vertex / edge features, masks, edges (padded with 0), hop matrix of the token graph."""
    gs = [item_graph(it) for it in items]
    Mx, Ex = max(g["M"] for g in gs), max(g["E"] for g in gs)
    B = len(items)
    vf = pad([g["vfeat"] for g in gs], 0.0, torch.float32)
    ef = pad([g["efeat"] for g in gs], 0.0, torch.float32)
    vm = pad([torch.ones(g["M"], dtype=torch.bool) for g in gs], False, torch.bool)
    em = pad([torch.ones(g["E"], dtype=torch.bool) for g in gs], False, torch.bool)
    edges = pad([it["edges"].long() for it in items], 0)
    hop = torch.full((B, Mx + Ex, Mx + Ex), MAX_HOP, dtype=torch.long)
    for i, g in enumerate(gs):
        M, E = g["M"], g["E"]
        h = g["hop"]
        hop[i, :M, :M] = h[:M, :M]; hop[i, :M, Mx:Mx + E] = h[:M, M:]
        hop[i, Mx:Mx + E, :M] = h[M:, :M]; hop[i, Mx:Mx + E, Mx:Mx + E] = h[M:, M:]
    V = pad([it["verts"].float() for it in items], 0.0, torch.float32)
    O = pad([it["eoff"].float() for it in items], 0.0, torch.float32)
    kv = pad([it.get("kv", torch.zeros(g["M"], dtype=torch.bool)) for it, g in zip(items, gs)], False, torch.bool)
    ke = pad([it.get("ke", torch.zeros(g["E"], dtype=torch.bool)) for it, g in zip(items, gs)], False, torch.bool)
    return {k: x.to(dev) for k, x in dict(vf=vf, ef=ef, vm=vm, em=em, edges=edges, hop=hop, V=V, O=O,
                                          kv=kv, ke=ke).items()}


# ----------------------------------------------------------------------- model
class BiasAttn(nn.Module):
    """Self-attention with an additive per-head bias (graph distance) and key padding mask."""

    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)

    def forward(self, x, bias, kmask):
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        m = bias.masked_fill(~kmask[:, None, None, :], float("-inf"))
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=m.to(q.dtype))
        return self.o(a.transpose(1, 2).reshape(B, L, D))


class Block(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(d)
        self.ln3 = nn.LayerNorm(d, elementwise_affine=False)
        self.sa, self.ca, self.ff = BiasAttn(d, heads), Attn(d, heads), mlp(d)
        self.ada = nn.Linear(d, 6 * d)
        nn.init.zeros_(self.ada.weight); nn.init.zeros_(self.ada.bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, z, c, bias, kmask):
        s1, b1, g1, s2, b2, g2 = self.ada(F.silu(c))[:, None].chunk(6, -1)
        x = x + g1 * self.drop(self.sa(modulate(self.ln1(x), b1, s1), bias, kmask))
        x = x + self.drop(self.ca(self.ln2(x), z))
        return x + g2 * self.drop(self.ff(modulate(self.ln3(x), b2, s2)))


class FlowGeoVar(nn.Module):
    def __init__(self, vdim, edim, d=128, layers=4, heads=4, n_latent=32, dropout=0.0, point_dim=14):
        super().__init__()
        self.heads = heads
        self.enc = PointPerceiver(point_dim, d, heads, n_latent, 2)
        self.vin = nn.Linear(3, d)
        self.vtop = nn.Sequential(nn.Linear(vdim, d), nn.GELU(), nn.Linear(d, d))
        self.ein = nn.Linear(6 + 6, d)                       # offsets + current chord (midpoint, vector)
        self.etop = nn.Sequential(nn.Linear(edim, d), nn.GELU(), nn.Linear(d, d))
        self.kflag_v, self.kflag_e = nn.Embedding(2, d), nn.Embedding(2, d)     # token is known (fixed)
        for m in (self.kflag_v, self.kflag_e):
            nn.init.zeros_(m.weight)
        self.hop = nn.Embedding(MAX_HOP + 1, heads)
        nn.init.zeros_(self.hop.weight)
        self.tmlp = nn.Sequential(nn.Linear(64, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList(Block(d, heads, dropout) for _ in range(layers))
        self.lnf = nn.LayerNorm(d, elementwise_affine=False)
        self.adaf = nn.Linear(d, 2 * d)
        self.vout, self.eout = nn.Linear(d, 3), nn.Linear(d, 6)
        for m in (self.adaf, self.vout, self.eout):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def temb(self, t):
        f = torch.exp(-math.log(1000.0) * torch.arange(32, device=t.device) / 32)
        a = t[:, None] * 1000.0 * f
        return self.tmlp(torch.cat([a.sin(), a.cos()], -1))

    def forward(self, xv, xo, t, z, b):
        """xv [B,M,3] normalised corners, xo [B,E,6] normalised offsets, b = collate() batch."""
        Mx = xv.shape[1]
        ia, ib = b["edges"][..., 0], b["edges"][..., 1]
        Pa = torch.gather(xv, 1, ia[..., None].expand(-1, -1, 3))
        Pb = torch.gather(xv, 1, ib[..., None].expand(-1, -1, 3))
        hv = self.vin(xv) + self.vtop(b["vf"])
        he = self.ein(torch.cat([xo, (Pa + Pb) / 2, Pb - Pa], -1)) + self.etop(b["ef"])
        if "kv" in b:
            hv = hv + self.kflag_v(b["kv"].long())
            he = he + self.kflag_e(b["ke"].long())
        h = torch.cat([hv, he], 1)
        kmask = torch.cat([b["vm"], b["em"]], 1)
        bias = self.hop(b["hop"]).permute(0, 3, 1, 2)          # [B,heads,L,L]
        c = self.temb(t) + z.mean(1)
        for blk in self.blocks:
            h = blk(h, z, c, bias, kmask)
        sh, sc = self.adaf(F.silu(c))[:, None].chunk(2, -1)
        h = modulate(self.lnf(h), sh, sc)
        return self.vout(h[:, :Mx]), self.eout(h[:, Mx:])


@torch.no_grad()
def sample(model, z, b, x0v, steps, gen=None, x1o_known=None):
    """Euler integration from corners x0v (normalised, already noised; known corners clean) and noise offsets.
    Known tokens (b['kv'], b['ke']; offsets from x1o_known, normalised) stay fixed at their values."""
    xv = x0v
    xo = torch.randn(*b["edges"].shape[:2], 6, device=z.device, generator=gen)
    kv = b.get("kv"); ke = b.get("ke")
    if ke is not None and x1o_known is not None:
        xo = torch.where(ke[..., None], x1o_known, xo)
    for i in range(steps):
        t = torch.full((xv.shape[0],), i / steps, device=z.device)
        vv, vo = model(xv, xo, t, z, b)
        xv2, xo2 = xv + vv / steps, xo + vo / steps
        xv = torch.where(kv[..., None], xv, xv2) if kv is not None else xv2
        xo = torch.where(ke[..., None], xo, xo2) if ke is not None else xo2
    return xv, xo


# ----------------------------------------------------------------------- metrics
def item_errors(Vp, Op, Vg, Og, edges, vm, em):
    """per-item corner / curve / curve_gtc errors (dataset units * 1000), padded entries masked."""
    out = []
    for i in range(len(Vp)):
        M, E = int(vm[i].sum()), int(em[i].sum())
        e = edges[i, :E]
        vp, vg, op, og = Vp[i:i + 1, :M], Vg[i:i + 1, :M], Op[i:i + 1, :E], Og[i:i + 1, :E]
        corner = float((vp - vg).norm(dim=-1).mean()) * 1000
        curve = float((bezier_pts(vp, op, e) - bezier_pts(vg, og, e)).norm(dim=-1).mean()) * 1000
        gtc = float((bezier_pts(vg, op, e) - bezier_pts(vg, og, e)).norm(dim=-1).mean()) * 1000
        out.append((corner, curve, gtc))
    return out


@torch.no_grad()
def evaluate(model, nrm, items, sigma, steps, dev, bs=16, seed=1234, corners=None, known="none", permute=False):
    """Errors with input corners = GT + sigma noise (proxy for stage-1 corners) unless `corners` (list of
    [M,3] tensors in dataset units) is given. known: 'none' / 'all' (geometric corners + feature edges fed as
    known GT tokens); permute: renumber the corners of every item at random first."""
    model.eval()
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    rng = np.random.default_rng(seed)
    acc = {"corner_in": [], "corner": [], "curve": [], "curve_gtc": [], "curve_gtc_free": []}
    if permute:
        items = [permute_item(it, rng) for it in items]
    items = [dict(it, **dict(zip(("kv", "ke"), known_masks(it, rng, known)))) for it in items]
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        b = collate(chunk, dev)
        pts, pmask = batch_points(chunk, dev)
        z = model.enc(pts, pmask)
        x1v = nrm.enc_v(b["V"])
        if corners is None:
            x0v = x1v + sigma * torch.randn(x1v.shape, device=dev, generator=gen)
        else:
            x0v = nrm.enc_v(pad([c.float() for c in corners[i:i + bs]], 0.0, torch.float32).to(dev))
        x0v = torch.where(b["kv"][..., None], x1v, x0v)
        xv, xo = sample(model, z, b, x0v, steps, gen, x1o_known=nrm.enc_o(b["O"]))
        Vp, Op = nrm.dec_v(xv), nrm.dec_o(xo)
        Vin = nrm.dec_v(x0v)
        for (c, cu, g), (ci, _, _) in zip(item_errors(Vp, Op, b["V"], b["O"], b["edges"], b["vm"], b["em"]),
                                          item_errors(Vin, Op, b["V"], b["O"], b["edges"], b["vm"], b["em"])):
            acc["corner"].append(c); acc["curve"].append(cu); acc["curve_gtc"].append(g); acc["corner_in"].append(ci)
        # curve shape on GT corners over the FREE edges only (known edges are exact by construction)
        free = b["em"] & ~b["ke"]
        for j in range(len(chunk)):
            e = b["edges"][j]
            d = (bezier_pts(b["V"][j:j + 1], Op[j:j + 1], e) - bezier_pts(b["V"][j:j + 1], b["O"][j:j + 1], e)).norm(dim=-1).mean(-1)[0] * 1000
            acc["curve_gtc_free"].append(float(d[free[j]].mean()))
    return {k: float(np.median(v)) for k, v in acc.items()} | {"n": len(items)}


# ----------------------------------------------------------------------- main
def build(data, train, a, dev):
    nrm = Norm(data, train).to(dev)
    g = item_graph(train[0])
    model = FlowGeoVar(g["vfeat"].shape[1], g["efeat"].shape[1], a["d"], a["layers"], a["heads"],
                       a["n_latent"], a.get("dropout", 0.0)).to(dev)
    return model, nrm


def load(ckpt, data, dev="cuda"):
    ck = torch.load(ckpt, weights_only=False); a = ck["args"]
    model, nrm = build(data, data["train"], a, dev)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    assert not unexpected and all(k.startswith("kflag") for k in missing), (missing, unexpected)
    model.eval()
    return model, nrm, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--n-latent", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--sigma", type=float, default=0.06,
                    help="corner source noise in dataset units (~ stage-1 corner error 0.05)")
    ap.add_argument("--curve-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perm-aug", action="store_true", help="renumber the corners of every training item at random")
    ap.add_argument("--mask-policy", choices=("none", "recipe"), default="none",
                    help="known-token masks in training (geometric corners + feature edges)")
    ap.add_argument("--eval-ckpt", default=None, help="only evaluate this checkpoint (known x permutation matrix)")
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    if a.eval_ckpt:
        data = torch.load(a.data, weights_only=False)
        model, nrm, ma = load(a.eval_ckpt, data)
        sig = ma["sigma"] / float(nrm.s.mean())
        summ = {f"{sp}|known={kn}|perm={int(pm)}": evaluate(model, nrm, data[sp], sig, ma["steps"], "cuda", known=kn, permute=pm)
                for sp in ["val"] + [k for k in data if k.startswith("val_") and data[k]]
                for kn in ("none", "all") for pm in (False, True)}
        os.makedirs(a.out, exist_ok=True)
        json.dump({"summary": summ}, open(f"{a.out}/result.json", "w"), indent=1)
        print(f"[var] RESULT {json.dumps(summ)}", flush=True)
        return
    dev = "cuda"
    data = torch.load(a.data, weights_only=False)
    tr, va = data["train"], data["val"]
    t0 = time.time()
    for it in tr + va + data.get("val_new", []):
        item_graph(it)
    print(f"graph features {time.time() - t0:.0f}s", flush=True)
    model, nrm = build(data, tr, vars(a), dev)
    sig = a.sigma / float(nrm.s.mean())                      # source noise in normalised units
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    os.makedirs(a.out, exist_ok=True)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[var] params {n_par/1e6:.2f} M | train {len(tr)} val {len(va)} | sigma {a.sigma} (norm {sig:.4f})", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * math.ceil(len(tr) / a.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))

    def loss_on(chunk):
        if a.perm_aug:
            chunk = [permute_item(it, rng) for it in chunk]
        chunk = [dict(it, **dict(zip(("kv", "ke"), known_masks(it, rng, a.mask_policy)))) for it in chunk]
        b = collate(chunk, dev)
        pts, pmask = batch_points(chunk, dev)
        x1v, x1o = nrm.enc_v(b["V"]), nrm.enc_o(b["O"])
        B = len(chunk)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model.enc(pts, pmask)
            t = torch.sigmoid(torch.randn(B, device=dev))
            x0v = torch.where(b["kv"][..., None], x1v, x1v + sig * torch.randn_like(x1v))   # known: clean at all t
            x0o = torch.where(b["ke"][..., None], x1o, torch.randn_like(x1o))
            tv = t[:, None, None]
            pv, po = model((1 - tv) * x0v + tv * x1v, (1 - tv) * x0o + tv * x1o, t, z, b)
        # loss only on the free tokens
        vm = (b["vm"] & ~b["kv"])[..., None].float(); em = (b["em"] & ~b["ke"])[..., None].float()
        # corner velocity target is O(sigma): scale so both parts carry comparable weight
        lv = (((pv.float() - (x1v - x0v)) / sig) ** 2 * vm).sum() / (vm.sum().clamp(min=1) * 3)
        lo_ = ((po.float() - (x1o - x0o)) ** 2 * em).sum() / (em.sum().clamp(min=1) * 6)
        return lv, lo_

    hist, t0 = [], time.time()
    for ep in range(1, a.epochs + 1):
        model.train(); tot = np.zeros(2); order = rng.permutation(len(tr))
        for i in range(0, len(tr), a.bs):
            lv, lo_ = loss_on([tr[j] for j in order[i:i + a.bs]])
            loss = lv + a.curve_weight * lo_
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - a.ema)
            tot += [float(lv.detach()), float(lo_.detach())]
        if ep % a.eval_every == 0 or ep == a.epochs:
            nb = math.ceil(len(tr) / a.bs)
            ev = evaluate(ema, nrm, va, sig, a.steps, dev)
            hist.append({"epoch": ep, "loss_v": tot[0] / nb, "loss_curve": tot[1] / nb, **ev})
            print(f"[var] ep {ep:5d} loss v {tot[0]/nb:.4f} c {tot[1]/nb:.4f} | val corner in {ev['corner_in']:.1f} "
                  f"-> {ev['corner']:.1f} curve {ev['curve']:.1f} gtc {ev['curve_gtc']:.1f} ({time.time()-t0:.0f}s)", flush=True)
    torch.save({"model": ema.state_dict(), "args": vars(a)}, f"{a.out}/last.pt")
    summ = {}
    for sp in ["val"] + [k for k in data if k.startswith("val_") and data[k]]:
        for kn in ("none", "all"):
            for pm in (False, True):
                summ[f"{sp}|known={kn}|perm={int(pm)}"] = evaluate(ema, nrm, data[sp], sig, a.steps, dev,
                                                                     known=kn, permute=pm)
    print(f"[var] RESULT {json.dumps(summ)}", flush=True)
    json.dump({"summary": summ, "history": hist}, open(f"{a.out}/result.json", "w"), indent=1)
    return
    summ = {"val": evaluate(ema, nrm, va, sig, a.steps, dev)}
    for sp in [k for k in data if k.startswith("val_") and data[k]]:      # val_new, val_full, ...
        summ[sp] = evaluate(ema, nrm, data[sp], sig, a.steps, dev)
    summ["train"] = evaluate(ema, nrm, tr[:34], sig, a.steps, dev)
    summ.update(params_M=n_par / 1e6, n_train=len(tr), seed=a.seed, epochs=a.epochs, train_s=time.time() - t0)
    print(f"[var] RESULT {json.dumps(summ)}", flush=True)
    json.dump({"summary": summ, "history": hist}, open(f"{a.out}/result.json", "w"), indent=1)


if __name__ == "__main__":
    main()
