"""
vertex_head_prototype.py

STUFE 1: Punktwolke + AUFLOESUNG n -> Vertices ERZEUGEN (variabel). Konditioniert
auf die Punktwolke tri_coordinates [Np,3] (x,y in [0,1], Label {0=Ecke,1=Rand,
2=Feld}) UND das Aufloesungs-Level n (Subdivision-Faktor; Face-Zahl F = 6 n^2)
generiert das Modell autoregressiv den Vertex-Satz als polar-Tokens (r, sin theta,
cos theta), kanonische Reihenfolge lexsort(theta, r), variable Laenge via STOP.

Warum auf n konditionieren (nicht rohe Face-Zahl): n ist der eigentliche 1-DOF-
Knopf -- Geometrie kommt aus der Punktwolke, Aufloesung aus n. F=6n^2 ist nur ein
nichtlinearer Proxy. Kleiner linearer Bereich -> der sinusoidale FaceCountEncoder
generalisiert sauber und extrapoliert auf n=5,6 (feinere Gitter, nie trainiert).
Die Punktwolke ist fuer alle Subdivisions DERSELBEN Blade identisch -> ohne n
waere die Vertex-Zahl mehrdeutig. Konditionierung = FaceCountEncoder(n) als
zusaetzliches Memory-Token an die Punkt-Latents konkateniert (Meshtron-Muster).

  ~/Environments/meshtron/bin/python vertex_head_prototype.py \
      [--data domain_data_aug.pt] [--d-model 256] [--batch 64] [--epochs 40]
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

from prototype_twostage import TwoStageTokenizer
from faceCount_encoder import FaceCountEncoder

FACECOUNTS = [6, 24, 54, 96]
FC2N = {6: 1, 24: 2, 54: 3, 96: 4}          # F = 6 n^2  ->  n = sqrt(F/6)


# ------------------------------------------------------------------
# Daten
# ------------------------------------------------------------------
def build_vertex_examples(data, tok):
    """Pro Mesh (F in {6,24,54,96}): pts [Np,3], seq [3M+1] (r,ts,tc)*M + STOP,
    verts_cart [M,2], center [2], n (Aufloesungs-Level 1..4)."""
    Qr, Qa = tok.Qr, tok.Qa
    off_ts, off_tc = tok.off_ts, tok.off_tc
    STOP = Qr + 2 * Qa + 1
    examples = []
    for d in data:
        Fc = d['faces'].shape[1]
        if Fc not in FC2N:
            continue
        vp = d['vertices_polar'].numpy()
        order = tok._sort_order(d['vertices_polar'])
        seq = []
        for oi in order:
            r, th = float(vp[oi, 0]), float(vp[oi, 1])
            rt = tok._q_scalar(r, tok.R_MIN, tok.R_MAX)
            ts, tc = tok._q_angle(th)
            seq += [rt, ts + off_ts, tc + off_tc]
        seq.append(STOP)
        examples.append({
            'pts': torch.tensor(d['tri_coordinates'].numpy(), dtype=torch.float32),
            'seq': torch.tensor(seq, dtype=torch.long),
            'vc': d['vertices_cartesian'].numpy()[order],
            'center': d['center'].numpy(),
            'n': FC2N[Fc], 'fc': Fc})
    return examples


# ------------------------------------------------------------------
# Modell
# ------------------------------------------------------------------
class VertexGen(nn.Module):
    def __init__(self, vocab, d=256, heads=8, n_enc=4, n_dec=4, max_len=400,
                 start_id=0, res_max=8):
        super().__init__()
        self.start_id = start_id
        self.pt_proj = nn.Linear(3, d)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, activation='gelu',
                                         batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, n_enc)
        # Aufloesungs-Konditionierung: sinusoidal(n)+MLP -> ein Memory-Token
        self.res_enc = FaceCountEncoder(d, min_face_count=1, max_face_count=res_max)
        self.tok_emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        dec = nn.TransformerDecoderLayer(d, heads, 4 * d, activation='gelu',
                                         batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec, n_dec)
        self.head = nn.Linear(d, vocab)

    def encode(self, pts, ppad, n):
        """pts [B,Np,3], n [B] -> mem [B,Np+1,d], ppad_ext [B,Np+1] (Res-Token angehaengt)."""
        mem = self.encoder(self.pt_proj(pts), src_key_padding_mask=ppad)
        res = self.res_enc(n.float())[:, None, :]        # [B,1,d]
        mem = torch.cat([mem, res], dim=1)
        ppad_ext = torch.cat([ppad, torch.zeros(ppad.size(0), 1, dtype=torch.bool,
                                                device=ppad.device)], dim=1)
        return mem, ppad_ext

    def decode(self, mem, ppad_ext, dec_in):
        L = dec_in.size(1)
        causal = torch.triu(torch.full((L, L), float('-inf'), device=dec_in.device), 1)
        h = self.tok_emb(dec_in) + self.pos(torch.arange(L, device=dec_in.device))[None]
        out = self.decoder(h, mem, tgt_mask=causal, memory_key_padding_mask=ppad_ext)
        return self.head(out)

    def forward(self, pts, ppad, dec_in, n):
        mem, ppad_ext = self.encode(pts, ppad, n)
        return self.decode(mem, ppad_ext, dec_in)


# ------------------------------------------------------------------
# Gruppen-Maske: Decoder-Position -> erlaubte Token-Gruppe (variable Laenge)
# ------------------------------------------------------------------
def group_mask(logits, pos0, Qr, Qa, stop_id):
    """Ziel an Position pos0 (0-basiert). Gruppe zyklisch (pos0 mod 3): 0=r,1=sin,
    2=cos. STOP nur zu Beginn eines neuen Tripels (pos0 mod 3 == 0, pos0 > 0)."""
    V = logits.shape[-1]
    mask = torch.full((V,), float('-inf'), device=logits.device)
    g = pos0 % 3
    if g == 0:
        mask[0:Qr] = 0.0
        if pos0 > 0:
            mask[stop_id] = 0.0
    elif g == 1:
        mask[Qr:Qr + Qa] = 0.0
    else:
        mask[Qr + Qa:Qr + 2 * Qa] = 0.0
    return logits + mask


# ------------------------------------------------------------------
# Batching
# ------------------------------------------------------------------
def collate(batch, start_id, pad_id):
    Npmax = max(e['pts'].shape[0] for e in batch)
    Lmax = max(e['seq'].numel() for e in batch)
    B = len(batch)
    pts = torch.zeros(B, Npmax, 3); ppad = torch.ones(B, Npmax, dtype=torch.bool)
    din = torch.full((B, Lmax), pad_id, dtype=torch.long)
    tgt = torch.full((B, Lmax), pad_id, dtype=torch.long)
    n = torch.zeros(B, dtype=torch.long)
    for i, e in enumerate(batch):
        Np = e['pts'].shape[0]; pts[i, :Np] = e['pts']; ppad[i, :Np] = False
        s = e['seq']; li = s.numel()
        din[i, 0] = start_id
        din[i, 1:li] = s[:-1]
        tgt[i, :li] = s
        n[i] = e['n']
    return pts, ppad, din, tgt, n


def run_epoch(model, order, examples, bs, opt, sched, clip, device, start_id,
              pad_id, train=True, progress=False, epoch=0):
    model.train() if train else model.eval()
    tot = cnt = correct = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    steps = range(0, len(order), bs)
    nsteps = len(steps)
    # Live-Balken nur im echten TTY (interaktiv/salloc). In SLURM-Logs (Nicht-TTY)
    # wuerde tqdm 1 Zeile/Batch schreiben -> stattdessen ~10 saubere Prints/Epoche.
    show_bar = progress and sys.stderr.isatty()
    bar = tqdm(steps, desc=f"ep{epoch} {'train' if train else 'val'}", unit="batch",
               leave=False, dynamic_ncols=True) if show_bar else steps
    log_every = max(1, nsteps // 10)
    with ctx:
        for bi, s in enumerate(bar):
            batch = [examples[i] for i in order[s:s + bs]]
            pts, ppad, din, tgt, n = collate(batch, start_id, pad_id)
            pts, ppad, din, tgt, n = (pts.to(device), ppad.to(device), din.to(device),
                                      tgt.to(device), n.to(device))
            logits = model(pts, ppad, din, n)
            V = logits.size(-1)
            loss = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1),
                                   ignore_index=pad_id)
            if train:
                opt.zero_grad(); loss.backward()
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step(); sched.step()
            with torch.no_grad():
                m = tgt != pad_id
                pred = logits.argmax(-1)
                correct += float((pred[m] == tgt[m]).sum()); k = int(m.sum())
                tot += loss.item() * k; cnt += k
            if show_bar:
                bar.set_postfix(loss=f"{tot/max(cnt,1):.3f}", acc=f"{correct/max(cnt,1):.3f}")
            elif progress and (bi % log_every == 0 or bi == nsteps - 1):
                print(f"  ep{epoch} {'train' if train else 'val'} {bi+1:4d}/{nsteps} "
                      f"loss {tot/max(cnt,1):.3f} acc {correct/max(cnt,1):.3f}", flush=True)
    return tot / max(cnt, 1), correct / max(cnt, 1)


@torch.no_grad()
def s1_generate(model, pts, n, tok, START, STOP, device, cap=140):
    """Variable Generierung, konditioniert auf Aufloesung n. Returns polar [Mgen,2]."""
    Qr, Qa = tok.Qr, tok.Qa
    pts = pts.unsqueeze(0).to(device)
    ppad = torch.zeros(1, pts.size(1), dtype=torch.bool, device=device)
    nt_ = torch.tensor([n], device=device)
    mem, ppad_ext = model.encode(pts, ppad, nt_)
    maxpos = model.pos.num_embeddings                # Decoder-Input darf pos nicht sprengen
    seq = [START]
    for step in range(cap * 3):
        if len(seq) >= maxpos:                       # Laenge auf pos-Embedding begrenzen
            break
        din = torch.tensor(seq, device=device)[None]
        logits = model.decode(mem, ppad_ext, din)[0, -1]
        logits = group_mask(logits, step, Qr, Qa, STOP)
        nt = int(logits.argmax())
        if nt == STOP:
            break
        seq.append(nt)
    gen = seq[1:]
    M = len(gen) // 3
    polar = []
    for m in range(M):
        r = tok._dq_scalar(gen[3 * m], tok.R_MIN, tok.R_MAX)
        th = tok._dq_angle(gen[3 * m + 1] - Qr, gen[3 * m + 2] - (Qr + Qa))
        polar.append((r, th))
    return np.array(polar).reshape(-1, 2)


@torch.no_grad()
def vertex_eval(model, examples, ids, device, tok, START, STOP):
    """Free-Run: Vertexzahl korrekt? + Vertex-Fehler (bei korrekter Zahl), je Facecount."""
    model.eval()
    per = {f: {'n': 0, 'count_ok': 0, 'errs': []} for f in FACECOUNTS}
    for i in ids:
        e = examples[i]; fc = e['fc']
        polar = s1_generate(model, e['pts'], e['n'], tok, START, STOP, device)
        Mgt = e['vc'].shape[0]
        per[fc]['n'] += 1
        if polar.shape[0] == Mgt:
            per[fc]['count_ok'] += 1
            rec = np.stack([polar[:, 0] * np.cos(polar[:, 1]),
                            polar[:, 0] * np.sin(polar[:, 1])], 1) + e['center'][None]
            per[fc]['errs'].append(np.linalg.norm(rec - e['vc'], axis=1) / math.sqrt(2))
    return per


def fmt_eval(per):
    out = []
    for f in FACECOUNTS:
        p = per[f]
        if p['n'] == 0:
            continue
        co = p['count_ok'] / p['n']
        if p['errs']:
            e = np.concatenate(p['errs'])
            out.append(f"{f}F: count-ok {co:.2f} err% med {100*np.median(e):.2f} "
                       f"p90 {100*np.percentile(e,90):.2f}")
        else:
            out.append(f"{f}F: count-ok {co:.2f}")
    return "  |  ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='domain_data_aug.pt')
    ap.add_argument('--d-model', type=int, default=256)
    ap.add_argument('--n-enc', type=int, default=4)
    ap.add_argument('--n-dec', type=int, default=4)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--warmup-frac', type=float, default=0.05)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--val-frac', type=float, default=0.1)
    ap.add_argument('--ve-every', type=int, default=5)
    ap.add_argument('--ve-n', type=int, default=240)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--save', default=None, help='Modell + Meta hier speichern (.pt)')
    ap.add_argument('--init', default=None,
                    help='Warmstart: Gewichte aus vertex_head-ckpt laden und '
                         'WEITER trainieren (frischer Kosinus-LR). Kein Optimizer-State.')
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = TwoStageTokenizer(repr_mode='cubic_bezier')
    Qr, Qa = tok.Qr, tok.Qa
    START = Qr + 2 * Qa; STOP = Qr + 2 * Qa + 1; PAD = Qr + 2 * Qa + 2
    vocab = Qr + 2 * Qa + 3

    print(f"Lade {args.data} ...")
    data = torch.load(args.data, weights_only=False)
    if args.limit:
        data = data[:args.limit]
    print("baue Vertex-Beispiele (alle Facecounts) ...")
    t0 = time.time()
    examples = build_vertex_examples(data, tok)
    from collections import Counter
    fc_dist = Counter(e['fc'] for e in examples)
    Ls = [e['seq'].numel() for e in examples]
    print(f"{len(examples)} Beispiele in {time.time()-t0:.0f}s  |  facecounts "
          f"{dict(sorted(fc_dist.items()))}  |  seq {min(Ls)}..{max(Ls)}  vocab {vocab}")
    max_len = max(Ls) + 1

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(examples))
    n_val = max(1, int(args.val_frac * len(examples)))
    val_ids = perm[:n_val]; train_ids = perm[n_val:]
    print(f"split: {len(train_ids)} train / {len(val_ids)} val")

    model = VertexGen(vocab, d=args.d_model, n_enc=args.n_enc, n_dec=args.n_dec,
                      max_len=max_len, start_id=START).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"VertexGen d={args.d_model} enc={args.n_enc} dec={args.n_dec}  "
          f"Params {n_par/1e6:.2f}M  device={device}")

    if args.init:                                    # Warmstart aus vorherigem ckpt
        ck = torch.load(args.init, weights_only=False, map_location=device)
        for k, want in (('vocab', vocab), ('max_len', max_len), ('d_model', args.d_model)):
            got = ck.get(k)
            assert got is None or got == want, \
                f"init-ckpt {k}={got} != aktuell {want} (Architektur/Daten muessen passen)"
        model.load_state_dict(ck['model'])
        print(f"== Warmstart aus {args.init} (war ep{ck.get('epoch')}, "
              f"val-loss {ck.get('val_loss')}) -> trainiere weiter ==")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    spe = math.ceil(len(train_ids) / args.batch)
    total = args.epochs * spe; warm = max(1, int(args.warmup_frac * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    def save_ckpt(tag_ep, tag_val):
        os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
        torch.save({'model': model.state_dict(), 'd_model': args.d_model,
                    'n_enc': args.n_enc, 'n_dec': args.n_dec, 'vocab': vocab,
                    'start': START, 'stop': STOP, 'pad': PAD, 'max_len': max_len,
                    'facecounts': FACECOUNTS, 'kind': 'vertex',
                    'epoch': tag_ep, 'val_loss': tag_val}, args.save)

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    ve_ids = val_ids[:min(args.ve_n, len(val_ids))]
    best_val = float('inf'); best_ep = 0
    tg = time.time()
    for ep in range(1, args.epochs + 1):
        trl, tra = run_epoch(model, rng.permutation(train_ids), examples, args.batch,
                             opt, sched, args.clip, device, START, PAD, train=True,
                             progress=True, epoch=ep)
        vll, vla = run_epoch(model, val_ids, examples, args.batch, None, None, 0,
                             device, START, PAD, train=False)
        vram = torch.cuda.max_memory_allocated() / 1e9 if device == 'cuda' else 0
        best = ""
        if args.save and vll < best_val:                 # bestes Modell speichern
            best_val = vll; best_ep = ep; save_ckpt(ep, vll); best = "  *best*"
        line = (f"ep {ep:3d}  tr-loss {trl:.4f} tr-acc {tra:.3f}  |  "
                f"val-loss {vll:.4f} val-acc {vla:.3f}  lr {sched.get_last_lr()[0]:.1e} "
                f"peakVRAM {vram:.2f}GB{best}")
        if ep % args.ve_every == 0 or ep == args.epochs:
            per = vertex_eval(model, examples, ve_ids, device, tok, START, STOP)
            line += "\n     " + fmt_eval(per)
        print(line)
    print(f"\nfertig in {time.time()-tg:.0f}s.")
    if args.save:
        print(f"bestes Modell (ep{best_ep}, val-loss {best_val:.4f}) -> {args.save}")


if __name__ == '__main__':
    main()
