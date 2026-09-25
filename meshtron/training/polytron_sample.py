"""polytron_sample.py -- the Polytron chain at inference.

    cloud, n_blocks --VertexModel--> vertices --BlockModel--> blocks
                    --CurveModel--> edge curves --> PolySeq

Sampling is masked so that every emitted sequence is well-formed by
construction, the same idea as the structural mask of `generate.py`:

  vertices  (z, y, x) never decreases lexicographically, the order the data
            is sorted in; STOP only between vertices and after at least 8
  blocks    slot 0 of a block is >= slot 0 of the previous block (blocks are
            sorted), slots 1..7 are > slot 0 (slot 0 is the block minimum) and
            never repeat within the block; exactly n_blocks blocks

What the mask cannot promise -- that the blocks tile the domain, that every
vertex is used -- is scored afterwards by `structure_report`.
"""
from __future__ import annotations

import numpy as np
import torch

from meshtron.data.polytron_blocks import (HEX_EDGES, PolySeq, PolytronSpec,
                                           block_edges)
from meshtron.model.polytron import clear_cache


def _sample(logits, temp, top_p, gen):
    if temp <= 0:
        return logits.argmax(-1)
    p = torch.softmax(logits / temp, -1)
    if top_p < 1.0:
        sp, si = p.sort(-1, descending=True)
        keep = sp.cumsum(-1) - sp < top_p
        sp = sp * keep
        p = torch.zeros_like(p).scatter(-1, si, sp)
    return torch.multinomial(p / p.sum(-1, keepdim=True), 1, generator=gen)[:, 0]


def _vertex_allow(toks, step, q, done, min_verts, max_verts):
    """[R, q+1] bool: which tokens keep the (z, y, x) order non-decreasing,
    STOP only between vertices, after min_verts, forced at max_verts."""
    R, dev = toks.shape[0], toks.device
    STOP = q
    slot, nverts = step % 3, step // 3
    allow = torch.ones((R, q + 1), dtype=torch.bool, device=dev)
    if slot == 0:
        allow[:, STOP] = nverts >= min_verts
        if nverts == max_verts:
            allow[:, :q] = False
            allow[:, STOP] = True
    else:
        allow[:, STOP] = False
    if nverts > 0:
        vals = torch.arange(q, device=dev)
        prev = toks[:, (nverts - 1) * 3:nverts * 3]
        cur = toks[:, nverts * 3:nverts * 3 + slot]
        tie = torch.ones(R, dtype=torch.bool, device=dev)
        for j in range(slot):
            tie &= cur[:, j] == prev[:, j]
        allow[:, :q] &= ~(tie[:, None] & (vals[None] < prev[:, slot][:, None]))
    allow[done] = True                     # finished rows: token discarded
    return allow


def _unflatten(toks, STOP):
    out = []
    for r in toks.cpu().numpy():
        stop = np.where(r == STOP)[0]
        r = r[:stop[0]] if len(stop) else r
        r = r[:len(r) // 3 * 3].reshape(-1, 3)
        out.append(r[:, [2, 1, 0]].astype(np.int64))          # back to xyz
    return out


@torch.no_grad()
def sample_vertices(model, pc, nb, k=1, temp=1.0, top_p=0.9, max_verts=100,
                    min_verts=8, seed=0):
    """pc [1,P,F], nb int -> list of k arrays [M,3] quantised xyz."""
    clear_cache(model)
    dev = pc.device
    gen = torch.Generator(device=dev).manual_seed(seed)
    q, STOP, BOS = model.q, model.STOP, model.BOS
    lat = model.cloud(pc).expand(k, -1, -1)
    nbt = torch.full((k,), int(nb), device=dev)
    inp = torch.full((k, 1), BOS, device=dev)
    toks = torch.zeros((k, 0), dtype=torch.long, device=dev)
    done = torch.zeros(k, dtype=torch.bool, device=dev)
    for step in range(3 * max_verts + 1):
        valid = torch.ones_like(inp, dtype=torch.bool)
        logits = model(inp, valid, None, nbt, lat=lat)[:, -1].float()
        allow = _vertex_allow(toks, step, q, done, min_verts, max_verts)
        logits = logits.masked_fill(~allow, float("-inf"))
        nxt = _sample(logits, temp, top_p, gen)
        nxt = torch.where(done, torch.full_like(nxt, STOP), nxt)
        done |= nxt == STOP
        toks = torch.cat([toks, nxt[:, None]], 1)
        inp = torch.cat([inp, nxt[:, None]], 1)
        if bool(done.all()):
            break
    return _unflatten(toks, STOP)


@torch.no_grad()
def beam_vertices(model, pc, nb, beam=8, max_verts=100, min_verts=8):
    """Beam search over the vertex sequence -> list of `beam` arrays [M,3],
    most likely first, with their log-likelihoods.

    The first coordinates are the only ones the cloud alone has to pin down:
    on the training set token 1 (the first corner's y) is spread over a few
    bins, and one bin off, a memorised model continues with ANOTHER sample's
    structure. Every later token is then near-certain given a correct prefix
    and uncertain given a wrong one, so the sequence likelihood separates
    them -- which greedy decoding, committing at token 1, cannot."""
    clear_cache(model)
    dev = pc.device
    q, STOP, BOS = model.q, model.STOP, model.BOS
    lat1 = model.cloud(pc)
    toks = torch.zeros((1, 0), dtype=torch.long, device=dev)
    score = torch.zeros(1, device=dev)
    done = torch.zeros(1, dtype=torch.bool, device=dev)
    for step in range(3 * max_verts + 1):
        R = toks.shape[0]
        inp = torch.cat([torch.full((R, 1), BOS, device=dev), toks], 1)
        valid = torch.ones_like(inp, dtype=torch.bool)
        logits = model(inp, valid, None, torch.full((R,), int(nb), device=dev),
                       lat=lat1.expand(R, -1, -1))[:, -1].float()
        allow = _vertex_allow(toks, step, q, done, min_verts, max_verts)
        lp = torch.log_softmax(logits.masked_fill(~allow, float("-inf")), -1)
        # finished beams carry on with STOP at no cost, exactly once
        lp[done] = float("-inf")
        lp[done, STOP] = 0.0
        cand = (score[:, None] + lp).reshape(-1)
        top = cand.topk(min(beam, int(torch.isfinite(cand).sum())))
        rows, tok = top.indices // (q + 1), top.indices % (q + 1)
        toks = torch.cat([toks[rows], tok[:, None]], 1)
        score = top.values
        done = done[rows] | (tok == STOP)
        if bool(done.all()):
            break
    return _unflatten(toks, STOP), score.cpu().numpy()


@torch.no_grad()
def sample_blocks(model, vq, pc, nb, k=1, temp=1.0, top_p=0.9, seed=0):
    """vq [M,3] (one vertex set), nb int -> list of k arrays [nb,8]."""
    clear_cache(model)
    dev = pc.device
    gen = torch.Generator(device=dev).manual_seed(seed)
    M = len(vq)
    vqt = torch.as_tensor(vq, device=dev)[None]
    vvalid = torch.ones((1, M), dtype=torch.bool, device=dev)
    nbt = torch.full((1,), int(nb), device=dev)
    H = model.encode(vqt, vvalid, pc, nbt).expand(k, -1, -1)
    vvalid = vvalid.expand(k, -1)
    nbk = nbt.expand(k)
    ptr = torch.zeros((k, 0), dtype=torch.long, device=dev)
    idx = torch.arange(M, device=dev)
    for t in range(8 * int(nb)):
        ptr_in = torch.cat([torch.zeros((k, 1), dtype=torch.long, device=dev), ptr], 1)
        logits = model.decode(H, vvalid, ptr_in, nbk)[:, -1].float()
        slot = t % 8
        allow = torch.ones((k, M), dtype=torch.bool, device=dev)
        if slot == 0:
            allow &= idx[None] <= M - 8
            if t > 0:
                allow &= idx[None] >= ptr[:, t - 8][:, None]
        else:
            first = ptr[:, t - slot]
            allow &= idx[None] > first[:, None]
            for j in range(t - slot, t):
                allow &= idx[None] != ptr[:, j][:, None]
        logits = logits.masked_fill(~allow, float("-inf"))
        nxt = _sample(logits, temp, top_p, gen)
        ptr = torch.cat([ptr, nxt[:, None]], 1)
    return [r.reshape(-1, 8) for r in ptr.cpu().numpy()]


@torch.no_grad()
def predict_curves(model, vq, blocks, pc, nb, spec: PolytronSpec):
    """-> (edges [E,2], cq [E,6]) by argmax."""
    clear_cache(model)
    dev = pc.device
    edges = block_edges(blocks)
    vqt = torch.as_tensor(vq, device=dev)[None]
    vvalid = torch.ones((1, len(vq)), dtype=torch.bool, device=dev)
    et = torch.as_tensor(edges, device=dev)[None]
    evalid = torch.ones((1, len(edges)), dtype=torch.bool, device=dev)
    vnorm = torch.as_tensor(spec.norm_xyz(spec.dequant_xyz(vq)),
                            dtype=torch.float32, device=dev)[None]
    logits = model(vqt, vvalid, pc, torch.tensor([int(nb)], device=dev), et,
                   evalid, vnorm)
    return edges, logits[0].argmax(-1).cpu().numpy()


def structure_report(seq: PolySeq) -> dict:
    """What the sampling mask cannot guarantee, as numbers.

    face_owners: a block face owned by more than two blocks is non-manifold;
    distinct_blocks: the same 8 ids twice is a duplicate; unused_vertices are
    corners no block points to (dropped before the fill)."""
    B = np.asarray(seq.blocks)
    faces = {}
    quads = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5),
             (2, 3, 7, 6), (3, 0, 4, 7))
    for b in B:
        for f in quads:
            key = frozenset(int(b[i]) for i in f)
            faces[key] = faces.get(key, 0) + 1
    used = np.unique(B)
    return {"n_vertices": int(len(seq.vq)), "n_blocks": int(len(B)),
            "unused_vertices": int(len(seq.vq) - len(used)),
            "duplicate_blocks": int(len(B) - len({frozenset(b.tolist()) for b in B})),
            "nonmanifold_faces": int(sum(v > 2 for v in faces.values())),
            "interior_faces": int(sum(v == 2 for v in faces.values())),
            "boundary_faces": int(sum(v == 1 for v in faces.values())),
            "n_edges": int(len(seq.edges))}


def compact(seq: PolySeq) -> PolySeq:
    """Drop vertices no block uses and renumber."""
    used = np.unique(seq.blocks)
    if len(used) == len(seq.vq):
        return seq
    new = -np.ones(len(seq.vq), np.int64)
    new[used] = np.arange(len(used))
    B = new[seq.blocks]
    E = new[seq.edges]
    return PolySeq(vq=seq.vq[used], blocks=B, edges=E, cq=seq.cq)


@torch.no_grad()
def run_chain(models: dict, pc, nb, spec: PolytronSpec, k=4, temp=1.0,
              top_p=0.9, seed=0, gt_vertices=None, gt_blocks=None, beam=0,
              timing=None):
    """Full chain for one geometry, k rollouts. `gt_vertices` / `gt_blocks`
    short-circuit a stage with ground truth, to attribute errors to a stage.
    `pc` is one cloud for all stages or {stage: cloud}, when the stages were
    trained with different cloud settings (spec.seam_cloud)."""
    pcs = pc if isinstance(pc, dict) else {s: pc for s in ("vertex", "block", "curve")}
    import time
    tm = timing if timing is not None else {}

    def tick(key, t0):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        tm[key] = tm.get(key, 0.0) + time.perf_counter() - t0
        return time.perf_counter()

    t0 = time.perf_counter()
    if gt_vertices is not None:
        vsets = [np.asarray(gt_vertices)] * k
    else:
        # rollout 0 is always greedy, the rest sampled: ranking the k fills
        # can then never do worse than the greedy chain alone
        if beam:
            vsets, _ = beam_vertices(models["vertex"], pcs["vertex"], nb,
                                     beam=max(beam, k))
            vsets = vsets[:k]
        else:
            # rollout 0 greedy, the rest sampled: ranking the k fills can
            # then never do worse than the greedy chain alone
            vsets = sample_vertices(models["vertex"], pcs["vertex"], nb, 1, 0.0,
                                    1.0, seed=seed)
            if k > 1:
                vsets += sample_vertices(models["vertex"], pcs["vertex"], nb,
                                         k - 1, temp, top_p, seed=seed)
    t0 = tick("vertex_model", t0)
    out = []
    for i, vq in enumerate(vsets):
        if len(vq) < 8:
            out.append(None)
            continue
        if gt_blocks is not None:
            B = np.asarray(gt_blocks)
        else:
            B = sample_blocks(models["block"], vq, pcs["block"], nb, 1,
                              0.0 if i == 0 else temp, top_p,
                              seed=seed + 1000 * i)[0]
        t0 = tick("block_model", t0)
        edges, cq = predict_curves(models["curve"], vq, B, pcs["curve"], nb, spec)
        t0 = tick("curve_model", t0)
        out.append(PolySeq(vq=vq, blocks=B, edges=edges, cq=cq))
    return out


__all__ = ["sample_vertices", "beam_vertices", "sample_blocks", "predict_curves", "run_chain",
           "structure_report", "compact", "HEX_EDGES"]
