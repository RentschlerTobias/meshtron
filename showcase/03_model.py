"""03 -- the transformer, low level.

A decoder-only GPT with FiLM conditioning on a point cloud and the block count.
This script walks the architecture, one forward pass with every intermediate
shape, the structural mask that constrains sampling, and one generation step.
"""
# %% [0] setup
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

# Run as a file and __file__ gives the location. Paste a cell into a REPL and
# the code is stdin, so there is no __file__ -- then locate showcase/ from the
# working directory instead. Start the REPL in the repo root or in showcase/.
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.path.abspath("showcase" if os.path.isdir("showcase") else ".")

if not os.path.isfile(os.path.join(HERE, "_common.py")):
    raise RuntimeError(f"showcase/_common.py not found from {os.getcwd()!r} -- "
                       "start the REPL in the meshtron repo root")

if HERE not in sys.path:
    sys.path.insert(0, HERE)

import _common as C  # noqa: E402

ck, cfg, coords, npt, rb, zb, model, max_len, _ = C.load_model()
tokens_file = torch.load(C.TOKENS, weights_only=False)
item = next((it for it in tokens_file["train"] + tokens_file["val"]
             if it["name"] == C.MACHINE), tokens_file["train"][0])

C.head("[0] the checkpoint")
print(f"   vocab {ck['vocab']}  pad_id {ck['pad_id']}  max_len {max_len}")
print(f"   d {cfg['d']}  layers {cfg['layers']}  heads {cfg['heads']}")
print(f"   n_points {cfg['n_points']}  n_latent {cfg['n_latent']}  "
      f"coords {coords}  npt {npt}")
print(f"   parameters {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")

# %% [1] the module tree
# Three things feed the stack: token embedding, position embedding, and the
# conditioning vector. The head shares weights with the token embedding.
C.head("[1] modules")
for name, mod in model.named_children():
    n = sum(p.numel() for p in mod.parameters())
    print(f"   {name:10s} {type(mod).__name__:18s} {n / 1e6:7.2f} M")

print()
print("   head.weight is tok.weight:",
      model.head.weight is model.tok.weight)

# %% [2] the conditioning path
# The cloud goes through a per-point MLP, then attention pooling onto n_latent
# learned queries, then a mean. The block count goes through a small MLP. Both
# are summed into ONE vector per sample.
from meshtron.data import conditioning  # noqa: E402

C.head("[2] conditioning")
src = torch.load(C.SRC, weights_only=False)
sample = next(s for s in src["samples"] if s["name"] == C.MACHINE)
cloud = conditioning.build_cloud(sample, cfg["n_points"], rb, zb,
                                 np.random.default_rng(0))
cloud = cloud[0] if isinstance(cloud, tuple) else cloud
pc = torch.as_tensor(np.asarray(cloud), dtype=torch.float32)[None]
fc = torch.tensor([float(item["blocks"])])

with torch.no_grad():
    f = model.penc.mlp(pc)
    q = model.penc.q[None].expand(f.shape[0], -1, -1)
    a = F.scaled_dot_product_attention(q, f, f)
    pooled = model.penc.proj(a).mean(1)
    cvec = model.condition(pc, fc)

C.show("pc", pc, 0)
C.show("per-point features", f, 0)
C.show("latent queries", q, 0)
C.show("pooled", pooled, 0)
C.show("condition vector", cvec, 0)
print("   one vector per sample; it is applied to EVERY token position as")
print("   h = h * (1 + tanh(c)) + c  -- a FiLM scale and shift.")

# %% [3] one forward, with the intermediates
# Hooks on the blocks so the residual stream can be watched layer by layer.
C.head("[3] forward")
x = torch.as_tensor(np.asarray(item["tokens"][:96]))[None]
acts = []
hooks = [b.register_forward_hook(lambda m, i, o: acts.append(o.detach()))
         for b in model.blocks]
with torch.no_grad():
    logits = model(x, pc, fc)

for h in hooks:
    h.remove()

C.show("x", x, 0)
C.show("logits", logits, 0)
print()
print("   residual stream norm per layer:")
for i, h in enumerate(acts):
    print(f"      layer {i:2d}  mean |h| {h.norm(dim=-1).mean():8.3f}  "
          f"max {h.norm(dim=-1).max():8.3f}")

# %% [4] what the model predicts here
# Next-token distribution at the last position, against the true next token.
C.head("[4] next token")
probs = logits[0, -1].softmax(-1)
top = probs.topk(8)
true_next = int(item["tokens"][96])
print(f"   true next token {true_next}, model gives it "
      f"p={probs[true_next]:.4f} (rank "
      f"{int((probs > probs[true_next]).sum()) + 1})")
print("   top 8:")
for p, i in zip(top.values.tolist(), top.indices.tolist()):
    print(f"      {i:5d}  p={p:.4f}{'   <- truth' if i == true_next else ''}")

# %% [5] causal masking, verified
# Changing a LATER token must not change an earlier prediction.
C.head("[5] causality")
x2 = x.clone()
x2[0, -1] = (x2[0, -1] + 1) % ck["vocab"]
with torch.no_grad():
    l2 = model(x2, pc, fc)

d_last = (logits[0, -1] - l2[0, -1]).abs().max().item()
d_prev = (logits[0, :-1] - l2[0, :-1]).abs().max().item()
print(f"   changed the last token: last position moved {d_last:.4f}, "
      f"all earlier positions moved {d_prev:.2e}")

# %% [6] the structural mask
# Generation is not free: at each position only certain token ids are legal --
# a coordinate slot, a separator, the stop id. slot_mask encodes that, and it is
# the reason the model cannot emit a syntactically broken sequence.
from meshtron.training.generate import slot_mask  # noqa: E402
from meshtron.data.hexa_row_tokenizer import HexaRowTokenizer  # noqa: E402

C.head("[6] slot mask")
tok = HexaRowTokenizer(r_bounds=rb, z_bounds=zb)
seq = list(map(int, item["tokens"][:96]))
stop_id = ck.get("stop_id", None)
print("   the mask is additive: 0 for legal ids, -inf for the rest.")
print("   the legal range cycles with the coordinate slot, and the separator")
print("   and stop ids only open at row boundaries.\n")
for cnt in range(8):
    m = slot_mask(tok, seq[:cnt + 1], cnt, ck["vocab"], coords)
    ok = torch.isfinite(m)
    ids = ok.nonzero().flatten()
    span = f"{int(ids.min())}..{int(ids.max())}" if len(ids) else "-"
    print(f"   after {cnt + 1:2d} tokens (slot {cnt % npt}): "
          f"{int(ok.sum()):4d} of {ck['vocab']} ids legal, range {span}")

# %% [7] one sampling step by hand
# Mask, temperature, multinomial. The generation loop in generate.py is this
# repeated with a KV cache.
C.head("[7] one sampling step")
temperature = 0.9
with torch.no_grad():
    lg = model(torch.tensor(seq)[None], pc, fc)[0, -1]

m = slot_mask(tok, seq, len(seq), ck["vocab"], coords)
raw_top = int(lg.argmax())
lg = lg + m                      # additive: -inf kills the illegal ids
p = (lg / temperature).softmax(-1)
choice = int(torch.multinomial(p, 1))
print(f"   legal ids {int(torch.isfinite(m).sum())} of {ck['vocab']}")
print(f"   unmasked argmax {raw_top}, "
      f"{'legal' if torch.isfinite(m[raw_top]) else 'ILLEGAL -- masked away'}")
print(f"   sampled {choice} with p={p[choice]:.4f}, "
      f"greedy would take {int(p.argmax())}")

# %% [8] the KV cache
# generate.forward_cached appends one token at a time and reuses the keys and
# values of everything before it, which is what makes generation linear rather
# than quadratic in the sequence length.
import inspect  # noqa: E402

from meshtron.training.generate import forward_cached  # noqa: E402

C.head("[8] cached forward")
print(inspect.getsource(forward_cached))
