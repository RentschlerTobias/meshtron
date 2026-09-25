"""train_polytron.py -- supervised training of one Polytron stage.

    uv run python -m meshtron.training.train_polytron --stage vertex
    uv run python -m meshtron.training.train_polytron --stage block
    uv run python -m meshtron.training.train_polytron --stage curve

Each stage is trained on its own with teacher forcing on the ground truth of
the stages before it (PolyGen trains its vertex and face model the same way).
The conditioning cloud is redrawn every time an item is seen, so the only
augmentation is which surface points the model gets to look at.

Two checkpoints per run: `<out>_best.pt` (lowest val loss) and `<out>_last.pt`.
With 683 training structures the model will overfit; `_last` is the one that
proves the architecture can carry a structure all the way to a mesh, `_best`
the one to judge generalisation with. Both carry the model kwargs and the
quantisation spec, so `load_stage` needs nothing else.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from meshtron.data.polytron_blocks import PolytronSpec, build_cloud  # noqa: E402
from meshtron.model.polytron import BlockModel, CurveModel, VertexModel  # noqa: E402

STAGES = {"vertex": VertexModel, "block": BlockModel, "curve": CurveModel}
IGN = -100


# --------------------------------------------------------------------------
# batching
# --------------------------------------------------------------------------

def _pad(arrs, val, dtype=torch.long):
    n = max(len(a) for a in arrs)
    shape = (len(arrs), n) + tuple(np.asarray(arrs[0]).shape[1:])
    out = torch.full(shape, val, dtype=dtype)
    for i, a in enumerate(arrs):
        if len(a):
            out[i, :len(a)] = torch.as_tensor(np.asarray(a))
    return out


def make_batch(items, stage, spec: PolytronSpec, rng, n_points, device):
    pcs = [build_cloud(it["surface_points"], it["point_labels"], n_points, spec, rng)
           for it in items]
    b = {"pc": torch.as_tensor(np.stack(pcs)).to(device),
         "nb": torch.tensor([it["n_blocks"] for it in items]).to(device)}
    if stage == "vertex":
        q = spec.q_vert
        seqs = [np.concatenate([VertexModel.flatten(it["vq"]), [q]]) for it in items]
        tgt = _pad(seqs, IGN)
        inp = torch.cat([torch.full((len(items), 1), q + 1), tgt[:, :-1]], 1)
        valid = torch.cat([torch.ones(len(items), 1, dtype=torch.bool),
                           tgt[:, :-1] != IGN], 1)
        inp = inp.masked_fill(inp == IGN, q + 1)
        b.update(inp=inp.to(device), valid=valid.to(device), tgt=tgt.to(device))
        return b
    vq = _pad([it["vq"] for it in items], 0)
    vvalid = _pad([np.ones(len(it["vq"]), bool) for it in items], False, torch.bool)
    b.update(vq=vq.to(device), vvalid=vvalid.to(device))
    if stage == "block":
        tgt = _pad([it["blocks"].reshape(-1) for it in items], IGN)
        inp = torch.cat([torch.zeros(len(items), 1, dtype=torch.long),
                         tgt[:, :-1].clamp(min=0)], 1)
        b.update(ptr_in=inp.to(device), tgt=tgt.to(device))
        return b
    edges = _pad([it["edges"] for it in items], 0)
    evalid = _pad([np.ones(len(it["edges"]), bool) for it in items], False, torch.bool)
    tgt = _pad([it["cq"] for it in items], IGN)
    vnorm = torch.as_tensor(spec.norm_xyz(spec.dequant_xyz(vq.numpy())),
                            dtype=torch.float32)
    b.update(edges=edges.to(device), evalid=evalid.to(device),
             tgt=tgt.to(device), vnorm=vnorm.to(device))
    return b


def soft_ce(logits, tgt, n_bins, sigma):
    """Cross-entropy against a Gaussian over neighbouring bins for targets
    < n_bins (coordinates, curve offsets); one-hot for the rest (STOP,
    pointers). A corner one bin off is then nearly as good as the exact
    bin, which is what it is geometrically."""
    m = tgt != IGN
    lg, t = logits[m], tgt[m]
    logp = torch.log_softmax(lg, -1)
    q = torch.zeros_like(logp)
    val = t < n_bins
    if val.any():
        idx = torch.arange(n_bins, device=lg.device, dtype=torch.float32)
        w = torch.exp(-0.5 * ((idx[None] - t[val, None].float()) / sigma) ** 2)
        q[val, :n_bins] = w / w.sum(-1, keepdim=True)
    q[~val] = torch.nn.functional.one_hot(t[~val], lg.shape[-1]).float()
    return -(q * logp).sum(-1).mean()


def jitter_inputs(b, q, p, max_bins):
    """Shift a share `p` of the coordinate INPUT tokens by up to `max_bins`
    (targets untouched). Without it the model reads the exact previous
    coordinates as a key into memorised sequences: one early token a bin off
    at inference and it continues with another sample's structure. With it,
    the prefix is only approximately informative and the cloud has to carry
    the position."""
    x = b["inp"]
    coord = x < q
    hit = coord & (torch.rand(x.shape, device=x.device) < p)
    d = torch.randint(-max_bins, max_bins + 1, x.shape, device=x.device)
    b["inp"] = torch.where(hit, (x + d).clamp(0, q - 1), x)


def forward_loss(model, stage, b, label_sigma=0.0):
    """(loss, token accuracy) for one batch."""
    if stage == "vertex":
        logits = model(b["inp"], b["valid"], b["pc"], b["nb"])
    elif stage == "block":
        logits = model(b["vq"], b["vvalid"], b["pc"], b["nb"], b["ptr_in"])
    else:
        logits = model(b["vq"], b["vvalid"], b["pc"], b["nb"], b["edges"],
                       b["evalid"], b["vnorm"])
    tgt = b["tgt"]
    lf = logits.float().reshape(-1, logits.shape[-1])
    if label_sigma > 0 and stage in ("vertex", "curve"):
        n_bins = lf.shape[-1] - 1 if stage == "vertex" else lf.shape[-1]
        loss = soft_ce(lf, tgt.reshape(-1), n_bins, label_sigma)
    else:
        loss = F.cross_entropy(lf, tgt.reshape(-1), ignore_index=IGN)
    m = tgt.reshape(-1) != IGN
    acc = (lf.argmax(-1) == tgt.reshape(-1))[m].float().mean()
    return loss, float(acc)


# --------------------------------------------------------------------------
# checkpoints
# --------------------------------------------------------------------------

def model_kwargs(stage, args, spec: PolytronSpec) -> dict:
    kw = dict(q=spec.q_vert, d=args.d, heads=args.heads, dropout=args.dropout)
    if stage == "vertex":
        kw.update(layers=args.layers)
        if args.point_memory:
            kw.update(point_memory=True)
    elif stage == "block":
        kw.update(enc_layers=args.layers, dec_layers=args.layers)
    else:
        kw.update(qc=spec.q_curve, enc_layers=max(2, args.layers // 2),
                  edge_layers=args.layers)
    return kw


def save_ckpt(path, model, stage, kw, spec, meta):
    torch.save({"stage": stage, "kwargs": kw, "spec": spec.to_json(),
                "state": model.state_dict(), **meta}, path)


def load_stage(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = STAGES[ck["stage"]](**ck["kwargs"]).to(device)
    m.load_state_dict(ck["state"])
    m.eval()
    return m, PolytronSpec.from_json(ck["spec"]), ck


# --------------------------------------------------------------------------
# loop
# --------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, stage, items, spec, bs, n_points, device, seed=123):
    model.eval()
    rng = np.random.default_rng(seed)
    tot, acc, n = 0.0, 0.0, 0
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        b = make_batch(chunk, stage, spec, rng, n_points, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            l, a = forward_loss(model, stage, b)
        tot += float(l) * len(chunk)
        acc += a * len(chunk)
        n += len(chunk)
    model.train()
    return tot / max(1, n), acc / max(1, n)


def train(args) -> dict:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    data = torch.load(args.data, weights_only=False)
    spec = PolytronSpec.from_json(data["spec"])
    spec = dataclasses.replace(spec, seam_cloud=bool(args.seam_cloud),
                               weight_label=int(args.weight_label))
    train_items, val_items = data["train"], data["val"]
    if args.limit:
        train_items = train_items[:args.limit]
        val_items = val_items[:max(1, args.limit // 4)]
    if args.val_is_train:
        val_items = train_items
    kw = model_kwargs(args.stage, args, spec)
    model = STAGES[args.stage](**kw).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd, betas=(0.9, 0.98))
    steps_per_ep = math.ceil(len(train_items) / args.bs)
    total = args.epochs * steps_per_ep
    warm = min(500, max(1, total // 20))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (
            0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))))
    start_ep, best = 1, float("inf")
    if args.resume and os.path.exists(args.out + "_last.pt"):
        ck = torch.load(args.out + "_last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["state"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_ep, best = ck["epoch"] + 1, ck.get("best_val", best)
        print(f"resumed at epoch {start_ep}")
    print(f"[{args.stage}] {n_par / 1e6:.2f} M params  train {len(train_items)} "
          f"val {len(val_items)}  {steps_per_ep} steps/epoch  device {device}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    log = []
    t0 = time.time()
    model.train()
    for ep in range(start_ep, args.epochs + 1):
        order = rng.permutation(len(train_items))
        el, ea = 0.0, 0.0
        for s in range(steps_per_ep):
            chunk = [train_items[j] for j in order[s * args.bs:(s + 1) * args.bs]]
            b = make_batch(chunk, args.stage, spec, rng, args.n_points, device)
            if args.input_jitter > 0 and args.stage == "vertex":
                jitter_inputs(b, spec.q_vert, args.input_jitter, args.jitter_bins)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss, acc = forward_loss(model, args.stage, b, args.label_sigma)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            el += float(loss)
            ea += acc
        el /= steps_per_ep
        ea /= steps_per_ep
        row = {"epoch": ep, "train_loss": el, "train_acc": ea,
               "lr": sched.get_last_lr()[0], "time": time.time() - t0}
        if ep % args.eval_every == 0 or ep == args.epochs:
            vl, va = evaluate(model, args.stage, val_items, spec, args.bs,
                              args.n_points, device)
            row.update(val_loss=vl, val_acc=va)
            meta = {"epoch": ep, "best_val": min(best, vl), "val_loss": vl,
                    "train_loss": el, "args": vars(args)}
            if vl < best:
                best = vl
                save_ckpt(args.out + "_best.pt", model, args.stage, kw, spec, meta)
            save_ckpt(args.out + "_last.pt", model, args.stage, kw, spec,
                      dict(meta, opt=opt.state_dict(), sched=sched.state_dict()))
            print(f"[{args.stage}] ep {ep:4d}  train {el:.4f} acc {ea:.3f}  "
                  f"val {vl:.4f} acc {va:.3f}  {time.time() - t0:.0f}s", flush=True)
        log.append(row)
    with open(args.out + "_log.json", "w") as fh:
        json.dump(log, fh)
    return {"final_train_loss": log[-1]["train_loss"], "best_val": best,
            "first_train_loss": log[0]["train_loss"], "log": log}


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=sorted(STAGES), required=True)
    ap.add_argument("--data", default=os.path.join(ROOT, "data",
                                                   "polytron_blocks_clean.pt"))
    ap.add_argument("--out", default=None,
                    help="checkpoint prefix (default data/polytron_<stage>)")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--n-points", type=int, default=2048)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--val-is-train", action="store_true",
                    help="evaluate on the training items (overfit runs)")
    ap.add_argument("--label-sigma", type=float, default=0.0,
                    help="Gaussian target width in bins (vertex/curve); 0 = one-hot. "
                         "Validation loss stays plain cross-entropy.")
    ap.add_argument("--input-jitter", type=float, default=0.0,
                    help="vertex stage: share of coordinate input tokens shifted")
    ap.add_argument("--jitter-bins", type=int, default=2)
    ap.add_argument("--point-memory", action="store_true",
                    help="vertex stage: decoder cross-attends every cloud point")
    ap.add_argument("--weight-label", type=int, default=7,
                    help="patch label oversampled 3x in the cloud (7 = O-grid "
                         "band; 5 = hub, what the first checkpoints used)")
    ap.add_argument("--seam-cloud", action="store_true",
                    help="conditioning cloud always carries every seam point "
                         "(stored in the checkpoint spec)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)
    if a.out is None:
        a.out = os.path.join(ROOT, "data", f"polytron_{a.stage}")
    return a


if __name__ == "__main__":
    r = train(parse())
    print(json.dumps({k: v for k, v in r.items() if k != "log"}))
