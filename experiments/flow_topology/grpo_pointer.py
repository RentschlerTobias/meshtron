#!/usr/bin/env python3
"""GRPO for the set pointer (pointer_conn2.SetPointer2) on REAL stage-1 corners.

Per step: B training geometries; for each, stage 1 (flow_vertex.py) generates corners, they are snapped
(block-free), and the pointer samples G block structures (temperature --temp, same validity masks as at inference,
optional seam mask). Reward per structure, no corner matching needed:
    w_orient * orientation_ok          (positive volumes, interior faces traversed oppositely)
  + w_feat   * feature_complete        (domain_partition_3D curve_postprocess, branch feat/feature-curves)
  + w_iso    * isomorphic to the GT    (training geometries have one)
  + w_used   * all corners used
Group-relative advantage A = (r - mean) / (std + eps); loss = -mean_t(A * log pi(a_t)) + beta * KL_k3(pi || ref)
over the sampled tokens, with the decoding masks re-applied when the log-probabilities are recomputed. Reward
evaluation runs in a process pool.

    PYTHONPATH=<dp3d hex3d_algohex> python grpo_pointer.py --out runs/ptr_grpo_s0 --steps 300
"""
import argparse
import copy
import json
import math
import os
import sys
import time
from multiprocessing import get_context

sys.path.insert(0, "/opt/stack/meshtron")
sys.path.insert(0, "/work/meshtron_v2")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import infer_chain_m4 as M4  # noqa: E402  (imports the import-order guard and the feature postprocess)
import flow_geo_var as FV  # noqa: E402
import flow_vertex as FVX  # noqa: E402
import pointer_conn2 as P2  # noqa: E402
from infer_chain_m4 import FeatureModelV2, batch_points  # noqa: E402


# ----------------------------------------------------------------------- rollouts with recorded masks
@torch.no_grad()
def rollout(ptr, e, z, vmask, seam, temp, gen, max_blocks=64):
    """Like infer_chain_m4.decode_masked, but records per step the list of masked ids (for the re-scoring)."""
    M = int(vmask[0].sum()); dev = e.device
    max_blocks = min(max_blocks, (ptr.pos.num_embeddings - 2) // 8)
    seam_of, allowed = seam if seam is not None else ([set()] * M, set())
    nbr = {s: [t for (u, v) in M4.HEX_EDGES for s2, t in ((u, v), (v, u)) if s2 == s] for s in range(8)}
    seq, blocks, face_count, masks = [-1], [], {}, []
    for _ in range(8 * max_blocks + 1):
        lg = ptr.logits(e, z, vmask, torch.tensor([seq], device=dev))[0, -1, :M + 1].float()
        pos = (len(seq) - 1) % 8
        bad = set()
        if pos:
            bad.add(M)
            cur = seq[len(seq) - pos:]
            bad |= set(cur)
            for c in range(M):
                for t in nbr[pos]:
                    if t < pos and seam_of[c] & seam_of[cur[t]] and tuple(sorted((c, cur[t]))) not in allowed:
                        bad.add(c)
        lg2 = lg.clone()
        if bad:
            lg2[list(bad)] = -1e9
        if (lg2 < -1e8).all():
            break
        nxt = int(torch.multinomial(F.softmax(lg2 / temp, -1), 1, generator=gen))
        masks.append(sorted(bad))
        if nxt == M:
            seq.append(M); break
        seq.append(nxt)
        if (len(seq) - 1) % 8 == 0:
            b = seq[-8:]
            keys = [tuple(sorted(b[i] for i in f)) for f in M4.VTK_FACES]
            if any(face_count.get(k, 0) >= 2 for k in keys):
                break
            for k in keys:
                face_count[k] = face_count.get(k, 0) + 1
            blocks.append(b)
    return seq[1:], masks, blocks


def token_logprobs(ptr, e, z, vmask, actions, masks):
    """log pi(a_t) for one sampled sequence under the same masks (teacher forcing, one forward pass)."""
    M = int(vmask[0].sum())
    ptr_in = torch.tensor([[-1] + actions[:-1]], device=e.device)
    lg = ptr.logits(e, z, vmask, ptr_in)[0, :, :M + 1].float()
    mk = torch.zeros_like(lg, dtype=torch.bool)
    for t, bad in enumerate(masks):
        if bad:
            mk[t, bad] = True
    lg = lg.masked_fill(mk, -1e9)
    return F.log_softmax(lg, -1).gather(1, torch.tensor(actions, device=e.device)[:, None])[:, 0]


# ----------------------------------------------------------------------- reward (worker process)
def reward_worker(args):
    V, blocks, gt_conn, npz, w = args
    import networkx as nx
    from networkx.algorithms.isomorphism import categorical_node_match
    from eval_ptr_iso import complex_graph
    out = {"orient": 0.0, "feat": 0.0, "iso": 0.0, "used": 0.0}
    if blocks:
        B = np.asarray(blocks)
        used = sorted({int(c) for b in B for c in b}); remap = {c: i for i, c in enumerate(used)}
        Bc = np.vectorize(remap.get)(B); Vc = V[used]
        out["used"] = float(len(used) == len(V))
        out["orient"] = float(M4.orientation_ok(Vc, Bc))
        try:
            raw = dict(np.load(npz, allow_pickle=False))
            post = M4.cp.postprocess(M4.straight_sample(Vc, Bc, raw), verbose=False)
            out["feat"] = float(json.loads(str(post["post_quality"]))["feature_complete"])
        except Exception:  # noqa: BLE001
            out["feat"] = 0.0
        out["iso"] = float(len(Bc) == len(gt_conn) and nx.is_isomorphic(
            complex_graph(Bc), complex_graph(gt_conn), node_match=categorical_node_match("kind", None)))
    out["r"] = w["orient"] * out["orient"] + w["feat"] * out["feat"] + w["iso"] * out["iso"] + w["used"] * out["used"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="/work/meshtron_v2/data/flow_canon_full.pt")
    ap.add_argument("--vtx-ckpt", default="/work/meshtron_v2/runs/vtx_full_s0/best.pt")
    ap.add_argument("--ptr-ckpt", default="/work/meshtron_v2/runs/ptr2_nolab/ep50.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=4, help="geometries per step")
    ap.add_argument("--group", type=int, default=8, help="rollouts per geometry")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--seam-mask", action="store_true")
    ap.add_argument("--w-orient", type=float, default=1.0)
    ap.add_argument("--w-feat", type=float, default=1.0)
    ap.add_argument("--w-iso", type=float, default=1.0)
    ap.add_argument("--w-used", type=float, default=0.2)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-n", type=int, default=34)
    ap.add_argument("--print-every", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = "cuda"
    os.makedirs(a.out, exist_ok=True)
    data = torch.load(a.data, weights_only=False)
    tr, va = data["train"], data["val"]
    vm, vn, _ = FVX.load(a.vtx_ckpt, data)
    ck = torch.load(a.ptr_ckpt, weights_only=False); pa = ck["args"]
    ptr = P2.SetPointer2(pa["d"], pa["layers"], pa["heads"], ck["max_len"]).to(dev)
    ptr.load_state_dict(ck["model"]); ptr.eval()                 # no dropout: rollout and update see the same policy
    ref = copy.deepcopy(ptr).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    nrm = FV.Norm(data, tr).to(dev)
    opt = torch.optim.AdamW(ptr.parameters(), lr=a.lr, weight_decay=0.0)
    w = {"orient": a.w_orient, "feat": a.w_feat, "iso": a.w_iso, "used": a.w_used}
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
    pool = get_context("spawn").Pool(a.workers)
    log, fms = [], {}

    def corners(it):
        pts, pmask = batch_points([it], dev)
        with torch.no_grad():
            zv = vm.enc(pts, pmask)
            n = int(vm.count_logits(zv).argmax(-1)[0])
            P = vn.dec_v(FVX.sample(vm, zv, [n], 32, gen)[0]).cpu().double().numpy()
        if it["name"] not in fms:
            fms[it["name"]] = FeatureModelV2(P2.npz_of(it["name"]), cache_dir=P2.FEAT_CACHE)
        fm = fms[it["name"]]
        Ps = P2.snap_free(P, fm)
        seam = M4.seam_order(Ps, fm) if a.seam_mask else None
        return Ps, seam, pts, pmask

    def evaluate(items):
        hit = 0
        for it in items:
            gen.manual_seed(1234 + sum(map(ord, it["name"])))
            Ps, seam, pts, pmask = corners(it)
            with torch.no_grad():
                x = nrm.enc_v(torch.tensor(Ps, dtype=torch.float32, device=dev))[None]
                vmask = torch.ones(1, len(Ps), dtype=torch.bool, device=dev)
                e, z = ptr.memory(x, vmask, pts, pmask, torch.zeros(1, len(Ps), 7, device=dev))
                if seam is not None:
                    blocks, _ = M4.decode_masked(ptr, e, z, vmask, *seam)
                else:
                    blocks, _ = P2.decode(ptr, e, z, vmask)
            r = reward_worker((Ps, blocks, it["conn"].long().numpy(), P2.npz_of(it["name"]), w))
            hit += r["iso"]
        return hit / max(1, len(items))

    t0 = time.time()
    ev0 = evaluate(va[:a.eval_n])
    print(f"[grpo] start | val greedy iso {ev0:.3f}", flush=True)
    log.append({"step": 0, "val_iso": ev0})
    for step in range(1, a.steps + 1):
        batch = [tr[i] for i in rng.choice(len(tr), a.batch, replace=False)]
        groups = []
        for it in batch:
            Ps, seam, pts, pmask = corners(it)
            x = nrm.enc_v(torch.tensor(Ps, dtype=torch.float32, device=dev))[None]
            vmask = torch.ones(1, len(Ps), dtype=torch.bool, device=dev)
            lab = torch.zeros(1, len(Ps), 7, device=dev)
            with torch.no_grad():
                e0, z0 = ptr.memory(x, vmask, pts, pmask, lab)
            rolls = [rollout(ptr, e0, z0, vmask, seam, a.temp, gen) for _ in range(a.group)]
            groups.append((it, Ps, x, vmask, pts, pmask, lab, rolls))
        jobs = [(g[1], r[2], g[0]["conn"].long().numpy(), P2.npz_of(g[0]["name"]), w) for g in groups for r in g[7]]
        rews = pool.map(reward_worker, jobs)
        loss_terms, kl_terms, stats = [], [], {"r": [], "orient": [], "feat": [], "iso": [], "used": []}
        j, n_inf = 0, 0
        for it, Ps, x, vmask, pts, pmask, lab, rolls in groups:
            R = np.array([rews[j + g]["r"] for g in range(len(rolls))])
            for k in stats:
                stats[k] += [rews[j + g][k] for g in range(len(rolls))]
            j += len(rolls)
            A = (R - R.mean()) / (R.std() + 1e-6)
            if R.std() < 1e-9:
                continue                                        # no signal in this group
            n_inf += 1
            e, z = ptr.memory(x, vmask, pts, pmask, lab)
            with torch.no_grad():
                er, zr = ref.memory(x, vmask, pts, pmask, lab)
            for (actions, masks, _), adv in zip(rolls, A):
                if not actions:
                    continue
                lp = token_logprobs(ptr, e, z, vmask, actions, masks)
                with torch.no_grad():
                    lpr = token_logprobs(ref, er, zr, vmask, actions, masks)
                loss_terms.append(-(float(adv) * lp).mean())
                d = lpr - lp
                kl_terms.append((torch.exp(d) - d - 1).mean())   # k3 estimator
        if loss_terms:
            loss = torch.stack(loss_terms).mean() + a.beta * torch.stack(kl_terms).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(ptr.parameters(), 1.0); opt.step()
            kl = float(torch.stack(kl_terms).mean().detach())
        else:
            kl = 0.0
        rec = {"step": step, **{k: float(np.mean(v)) for k, v in stats.items()}, "kl": kl, "informative": n_inf, "t": time.time() - t0}
        log.append(rec)
        if step % a.print_every == 0 or step == 1:
            print(f"[grpo] step {step:4d} | R {rec['r']:.3f} orient {rec['orient']:.2f} feat {rec['feat']:.2f} "
                  f"iso {rec['iso']:.2f} used {rec['used']:.2f} | inf {n_inf}/{len(groups)} kl {kl:.4f} ({rec['t']:.0f}s)", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            ev = evaluate(va[:a.eval_n])
            log.append({"step": step, "val_iso": ev})
            print(f"[grpo] step {step:4d} | val greedy iso {ev:.3f}", flush=True)
            torch.save({"model": ptr.state_dict(), "args": pa, "max_len": ck["max_len"], "grpo": vars(a), "step": step},
                       f"{a.out}/step{step}.pt")
        json.dump(log, open(f"{a.out}/log.json", "w"), indent=1)
    pool.close()


if __name__ == "__main__":
    main()
