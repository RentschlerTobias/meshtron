"""
train_pointer.py

Echtes (nicht-overfit) train/val fuer den POINTER-FACE-Kopf (Stufe 2) mit
Batching + Padding. Skaliert polytron_pointer_model.py von batch=1 auf echte
Batches ueber den augmentierten Datensatz. Vertices werden als gegeben angenommen
(Stufe-1-Konditionierung kommt spaeter); getestet wird, ob das Pointer-Netz die
Face-Topologie GENERALISIERT (val face-exact), nicht nur auswendig lernt.

Padding-Schema:
  - vert_feats [B,Mmax,3] + vert_pad [B,Mmax] (True=Padding) -> Encoder ignoriert,
    Pointer-Logits an Pad-Spalten = -inf (Zeiger kann nie auf Padding zeigen).
  - ptr_targets [B,Lmax] mit -1 an Pad-Positionen -> CE ignore_index=-1,
    Decoder tgt_key_padding_mask.

  ~/Environments/meshtron/bin/python train_pointer.py \
      [--data domain_data_smoke.pt] [--d-model 256] [--batch 32] [--epochs 20]
"""

import argparse
import math
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from meshtron.data.polytron_tokenizer import PolytronTokenizer
from meshtron.legacy.polytron_pointer_model import PointerFaceModel, build_examples

PAD_TGT = -1


def collate(batch):
    """Liste von (vert_feats[M,D], ptr[L], faces[F,cpb]) -> gepolsterte Tensoren.
    D (Vertex-Feature-Breite: 3 fuer dim=2 r/sin/cos, 4 fuer dim=3 +z) wird aus
    den Daten selbst gelesen statt hartkodiert."""
    Mmax = max(vf.shape[0] for vf, _, _ in batch)
    Lmax = max(p.numel() for _, p, _ in batch)
    D = batch[0][0].shape[1]
    B = len(batch)
    vf = torch.zeros(B, Mmax, D)
    vpad = torch.ones(B, Mmax, dtype=torch.bool)          # True=Padding
    ptr = torch.full((B, Lmax), PAD_TGT, dtype=torch.long)
    tpad = torch.ones(B, Lmax, dtype=torch.bool)
    faces = []
    for i, (v, p, f) in enumerate(batch):
        M, L = v.shape[0], p.numel()
        vf[i, :M] = v; vpad[i, :M] = False
        ptr[i, :L] = p; tpad[i, :L] = False
        faces.append(f)
    return vf, vpad, ptr, tpad, faces


def run_epoch(model, order, examples, bs, opt, sched, clip, device, train=True,
              progress=False, epoch=0):
    model.train() if train else model.eval()
    tot_loss = tot_tok = tf_correct = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    steps = range(0, len(order), bs)
    nsteps = len(steps)
    show_bar = progress and sys.stderr.isatty()   # Balken nur im TTY; Log -> Prints
    bar = tqdm(steps, desc=f"ep{epoch} {'train' if train else 'val'}", unit="batch",
               leave=False, dynamic_ncols=True) if show_bar else steps
    log_every = max(1, nsteps // 10)
    with ctx:
        for bi, s in enumerate(bar):
            batch = [examples[i] for i in order[s:s + bs]]
            vf, vpad, ptr, tpad, _ = collate(batch)
            vf, vpad, ptr, tpad = vf.to(device), vpad.to(device), ptr.to(device), tpad.to(device)
            logits = model(vf, ptr, vert_pad=vpad, tgt_pad=tpad)   # [B,L,M]
            M = logits.size(-1)
            loss = F.cross_entropy(logits.reshape(-1, M), ptr.reshape(-1),
                                   ignore_index=PAD_TGT)
            if train:
                opt.zero_grad(); loss.backward()
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step(); sched.step()
            with torch.no_grad():
                m = ptr != PAD_TGT
                pred = logits.argmax(-1)
                tf_correct += float((pred[m] == ptr[m]).sum())
                ntok = int(m.sum())
                tot_loss += loss.item() * ntok; tot_tok += ntok
            if show_bar:
                bar.set_postfix(loss=f"{tot_loss/max(tot_tok,1):.3f}",
                                acc=f"{tf_correct/max(tot_tok,1):.3f}")
            elif progress and (bi % log_every == 0 or bi == nsteps - 1):
                print(f"  ep{epoch} {'train' if train else 'val'} {bi+1:4d}/{nsteps} "
                      f"loss {tot_loss/max(tot_tok,1):.3f} acc {tf_correct/max(tot_tok,1):.3f}",
                      flush=True)
    return tot_loss / tot_tok, tf_correct / tot_tok


@torch.no_grad()
def free_run_eval(model, examples, ids, device):
    """Autoregressives Free-Run (batch=1) auf Stichprobe: face-exact + quads-distinct."""
    model.eval()
    face_exact = face_total = quads_distinct = 0
    for i in ids:
        vf, ptr, faces = examples[i]
        gen = model.generate(vf.unsqueeze(0).to(device), ptr.numel())
        gen = np.array(gen).reshape(-1, 4)
        for gi, fi in zip(gen, faces):
            face_total += 1
            if np.array_equal(gi, fi):
                face_exact += 1
            if len(set(gi.tolist())) == 4:
                quads_distinct += 1
    return face_exact / max(face_total, 1), quads_distinct / max(face_total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='domain_data_smoke.pt')
    ap.add_argument('--d-model', type=int, default=256)
    ap.add_argument('--n-enc', type=int, default=4)
    ap.add_argument('--n-dec', type=int, default=4)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--warmup-frac', type=float, default=0.05)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--val-frac', type=float, default=0.1)
    ap.add_argument('--fr-every', type=int, default=5, help='Free-Run alle N Epochen')
    ap.add_argument('--fr-n', type=int, default=60, help='Free-Run Stichprobe')
    ap.add_argument('--limit', type=int, default=None, help='nur erste K Meshes')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--save', default=None, help='Modell + Meta hier speichern (.pt)')
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Lade {args.data} ...")
    data = torch.load(args.data, weights_only=False)
    if args.limit:
        data = data[:args.limit]
    max_v = max(d['vertices_polar'].shape[0] for d in data)
    tok = PolytronTokenizer(max_vertices=max_v + 16)
    print(f"baue Beispiele aus {len(data)} Meshes ...")
    t0 = time.time()
    examples = build_examples(data, tok)
    Ms = [e[0].shape[0] for e in examples]; Ls = [e[1].numel() for e in examples]
    print(f"{len(examples)} Beispiele in {time.time()-t0:.0f}s  |  M {min(Ms)}..{max(Ms)}  "
          f"|  Zeiger/Mesh {min(Ls)}..{max(Ls)}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(examples))
    n_val = max(1, int(args.val_frac * len(examples)))
    val_ids = perm[:n_val]; train_ids = perm[n_val:]
    print(f"split: {len(train_ids)} train / {len(val_ids)} val")

    model = PointerFaceModel(d_model=args.d_model, n_enc=args.n_enc,
                             n_dec=args.n_dec).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"PointerFaceModel d={args.d_model} enc={args.n_enc} dec={args.n_dec}  "
          f"Params {n_par/1e6:.2f}M  device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_ep = math.ceil(len(train_ids) / args.batch)
    total_steps = args.epochs * steps_per_ep
    warmup = max(1, int(args.warmup_frac * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else
        0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup))))

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    fr_ids = val_ids[:min(args.fr_n, len(val_ids))]
    tglob = time.time()
    best_val = float('inf'); best_ep = 0
    for ep in range(1, args.epochs + 1):
        tr_ids = rng.permutation(train_ids)
        tr_loss, tr_acc = run_epoch(model, tr_ids, examples, args.batch, opt, sched,
                                    args.clip, device, train=True, progress=True, epoch=ep)
        vl_loss, vl_acc = run_epoch(model, val_ids, examples, args.batch, None, None,
                                    0, device, train=False)
        vram = torch.cuda.max_memory_allocated() / 1e9 if device == 'cuda' else 0
        best = ""
        if args.save and vl_loss < best_val:            # bestes Modell speichern
            best_val = vl_loss; best_ep = ep
            os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
            torch.save({'model': model.state_dict(), 'd_model': args.d_model,
                        'n_enc': args.n_enc, 'n_dec': args.n_dec, 'kind': 'pointer',
                        'epoch': ep, 'val_loss': vl_loss}, args.save)
            best = "  *best*"
        if ep % args.fr_every == 0 or ep == args.epochs:
            fe, qd = free_run_eval(model, examples, fr_ids, device)
            frs = f"val-face-exact {fe:.3f} quads-distinct {qd:.3f}"
        else:
            frs = "val-face-exact -     quads-distinct -"
        print(f"ep {ep:3d}  tr-loss {tr_loss:.4f} tr-acc {tr_acc:.3f}  |  "
              f"val-loss {vl_loss:.4f} val-tf-acc {vl_acc:.3f}  {frs}  "
              f"lr {sched.get_last_lr()[0]:.1e} peakVRAM {vram:.2f}GB{best}")
    print(f"\nfertig in {time.time()-tglob:.0f}s.")
    if args.save:
        print(f"bestes Modell (ep{best_ep}, val-loss {best_val:.4f}) -> {args.save}")


if __name__ == '__main__':
    main()
