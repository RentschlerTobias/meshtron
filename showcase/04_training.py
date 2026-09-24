"""04 -- training, low level.

How a batch is built, what the loss weights do, and one optimiser step with the
gradients watched. Nothing here writes a checkpoint; it runs on CPU on a couple
of samples so every number can be inspected.
"""
# %% [0] setup
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

from meshtron.training import train_hexarow_full as T  # noqa: E402

ck, cfg, coords, npt, rb, zb, model, max_len, _ = C.load_model()
tokens_file = torch.load(C.TOKENS, weights_only=False)
items = tokens_file["train"][:8]
pad_id = ck["pad_id"]

C.head("[0] training configuration as trained")
for k in ("d", "layers", "heads", "n_points", "n_latent", "dropout",
          "token_budget", "batch_cap", "epochs", "lr", "wd", "warmup"):
    if k in cfg:
        print(f"   {k:14s} {cfg[k]}")

# %% [1] batching by token budget, not by sample count
# Sequences differ in length, so batches are formed to fill a token budget --
# long sequences give small batches and the other way round.
C.head("[1] batches")
batches = T.make_batches(items, budget=4096, cap=4)
print(f"   {len(items)} items -> {len(batches)} batches "
      f"(budget 4096 tokens, cap 4 samples)")
for bi, b in enumerate(batches):
    lens = [len(items[i]["tokens"]) for i in b]
    print(f"      batch {bi}: {len(b)} samples, lengths {lens}, "
          f"padded cost {max(lens) * len(b)}")

# %% [2] what batchify produces
# batchify wants the conditioning cloud on each item, so attach it first -- the
# trainer does the same, drawing a fresh cloud every epoch so the model never
# sees the exact same sampling twice.
from meshtron.data import conditioning  # noqa: E402

C.head("[2] batchify")
src = torch.load(C.SRC, weights_only=False)
by_name = {s["name"]: s for s in src["samples"]}
rng = np.random.default_rng(0)
for it in items:
    s = by_name.get(it["name"])
    cl = conditioning.build_cloud(s, cfg["n_points"], rb, zb, rng)
    it["points"] = np.asarray(cl[0] if isinstance(cl, tuple) else cl)

x, slot, w, pc, fc = T.batchify([items[i] for i in batches[0]], pad_id, "cpu",
                                specials=(), w_unit_end=1.0, npt=npt)
# the trainer feeds x[:, :-1] and targets x[:, 1:] -- same tensor, offset by
# one, and the weight and slot are cut the same way.
xin, y = x[:, :-1], x[:, 1:]
win, slotin = w[:, :-1], slot[:, :-1]
C.show("x  (input, cut)", xin, 1)
C.show("y  (target, shifted)", y, 1)
C.show("w  (per-token weight)", w, 1)
C.show("slot (coord id)", slot, 1)
C.show("pc (clouds)", pc, 0)
C.show("fc (block counts)", fc, 1)
print(f"   padding fills {(y == pad_id).float().mean() * 100:.1f}% of the "
      f"target")
print(f"   slot cycles 0..{npt - 1}: {slot[0, :12].tolist()}")

# %% [3] the slot embedding, and a checkpoint that predates it
# Position alone does not say whether a token is an r, a sin, a cos or a z.
# The slot id is embedded and added so the model knows which coordinate it is
# predicting -- the PolyGen trick. This checkpoint was trained BEFORE that was
# added, which its cfg gives away: no "npt", no "coords" key. Feeding it a slot
# it never saw costs about 30 points of next-token accuracy, measured below.
C.head("[3] why the slot id exists")
print(f"   cfg has npt: {'npt' in cfg}   cfg has coords: {'coords' in cfg}")
print(f"   -> this checkpoint is fed with slot="
      f"{'slot' if C.SLOT_TRAINED else 'None'}")
C.show("slot embedding", model.slot.weight, 0)
sims = torch.nn.functional.normalize(model.slot.weight, dim=-1)
print("   cosine similarity between slot embeddings:")
print((sims @ sims.T).detach().numpy().round(3))

# %% [4] the loss
# Cross entropy on the shifted sequence, weighted per token, padding excluded.
C.head("[4] loss")
model.train()
slot_arg = slotin if C.SLOT_TRAINED else None
logits = model(xin, pc, fc, slot=slot_arg)
lossf = torch.nn.CrossEntropyLoss(ignore_index=pad_id, reduction="none")
loss = T.weighted_loss(logits, y, win, pad_id, ck["vocab"], lossf)
pred = logits.argmax(-1)
real0 = y != pad_id
acc = float((pred[real0] == y[real0]).float().mean())
print(f"   logits {tuple(logits.shape)} -> loss {float(loss):.4f}, "
      f"next-token accuracy {100 * acc:.1f}%")
with torch.no_grad():
    other = model(xin, pc, fc, slot=(None if C.SLOT_TRAINED else slotin))
    lo = T.weighted_loss(other, y, win, pad_id, ck["vocab"], lossf)
    ao = float((other.argmax(-1)[real0] == y[real0]).float().mean())
print(f"   with the other slot convention: loss {float(lo):.4f}, "
      f"accuracy {100 * ao:.1f}%  <- the mismatch this checkpoint punishes")
print(f"   perplexity {float(torch.exp(loss.detach())):.1f} over "
      f"{ck['vocab']} ids")
flat = lossf(logits.reshape(-1, ck["vocab"]), y.reshape(-1)).reshape(y.shape)
real = y != pad_id
print(f"   per-token loss: p50 {flat[real].median():.4f}  "
      f"max {flat[real].max():.4f}")

# %% [5] where the model is still wrong
# The worst positions, decoded back to which coordinate slot they are.
C.head("[5] hardest positions")
vals, idx = flat[real].topk(6)
rows, cols = real.nonzero(as_tuple=True)
for v, i in zip(vals.tolist(), idx.tolist()):
    r, c = int(rows[i]), int(cols[i])
    print(f"   sample {r}, position {c:4d}, slot {int(slot[r, c])}, "
          f"target {int(y[r, c]):5d}, loss {v:.3f}")

# %% [6] one optimiser step
# Backward, gradient norms per module, one AdamW step, and the loss after.
C.head("[6] one step")
opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-4),
                        weight_decay=cfg.get("wd", 0.04))
opt.zero_grad()
loss.backward()
print("   gradient norm per module:")
for name, mod in model.named_children():
    g = [p.grad for p in mod.parameters() if p.grad is not None]
    if g:
        n = torch.sqrt(sum((x * x).sum() for x in g))
        print(f"      {name:10s} {n:10.4f}")
total = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
print(f"   total grad norm {total:.4f} (clipped at 1.0)")
opt.step()
model.eval()
with torch.no_grad():
    loss2 = T.weighted_loss(model(xin, pc, fc, slot=slot_arg), y, win, pad_id,
                            ck["vocab"], lossf)
print(f"   loss {loss.item():.4f} -> {loss2.item():.4f} after one step")

# %% [7] the learning-rate schedule
# Linear warmup then cosine decay; warmup matters because the conditioning
# vector multiplies every token embedding and can blow up early.
C.head("[7] schedule")
warm, epochs = cfg.get("warmup", 600), cfg.get("epochs", 15)
steps = warm * 6
base = cfg.get("lr", 1e-4)
xs = np.arange(0, steps, max(1, steps // 12))
for s in xs:
    lr = (base * s / warm if s < warm
          else base * 0.5 * (1 + np.cos(np.pi * (s - warm) / (steps - warm))))
    bar = "#" * int(40 * lr / base)
    print(f"   step {s:6d}  lr {lr:.2e}  {bar}")
