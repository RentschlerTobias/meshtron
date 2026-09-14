"""
polytron_geom_model.py

STUFE 3: Geometrie-Kopf. Gegeben die Vertices (Stufe 1) + die Face-Topologie
(Stufe 2, Pointer-Faces), regressiere pro gerichteter Half-Edge die HO-Kanten-
geometrie als cubic_bezier-Kontrollpunkte (s1,h1,s2,h2 Chord-lokal, siehe
polytron_tokenizer.py). Multigraph-treu: (a,b) und (b,a) sind VERSCHIEDENE Kanten
(Blade Druck/Saug) -> geordnete Edge-Features, kein {a,b}-Dedup.

Nicht autoregressiv: die Topologie ist komplett bekannt, alle Kanten-Geometrien
werden PARALLEL vorhergesagt. Modell:
  vert_feats [B,M,3] --Encoder--> H [B,M,d]
  je Kante (a,b):  edge_tok = MLP([H[a], H[b]])   (geordnet -> richtungsabhaengig)
  edge_toks --Transformer (voll, kontextbewusst)--> --Head--> (s1,h1,s2,h2)
Loss = SmoothL1 auf die 4 Skalare. Metrik = Kurven-Rekonstruktionsfehler vs echte
Streamline (max-dist / Chord), vergleichbar mit dem Fit-Floor (Targets selbst).

  ~/Environments/meshtron/bin/python polytron_geom_model.py \
      [--data domain_data_smoke.pt] [--d-model 256] [--batch 32] [--epochs 30]
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

from polytron_tokenizer import PolytronTokenizer


# ------------------------------------------------------------------
# Daten: (vert_feats, edges_new, targets, refs) je Mesh
# ------------------------------------------------------------------
def build_geom_examples(data, tok):
    """Pro Mesh: vert_feats [M,3] (dim=2) oder [M,4] (dim=3), edges_new [E,2]
    (new-Idx, Face-Traversal ueber tok._face_edge_pairs() -- korrekte Kanten-
    topologie, NICHT range(corners_per_block), siehe polytron_tokenizer.py),
    targets [E,4] (dim=2: s1,h1,s2,h2) oder [E,6] (dim=3: s1,h1a,h1b,s2,h2a,h2b),
    edges_glob [E,2] (global, fuer Eval)."""
    examples = []
    edge_pairs = tok._face_edge_pairs()
    for d in data:
        _, meta = tok.tokenize(d)
        order = meta['order']; old2new = meta['old2new']
        vp = d['vertices_polar'].numpy()
        r = vp[order, 0]; th = vp[order, 1]
        if tok.dim == 3:
            z = vp[order, 2]
            vf = np.stack([r, np.sin(th), np.cos(th), z], axis=1).astype(np.float32)
        else:
            vf = np.stack([r, np.sin(th), np.cos(th)], axis=1).astype(np.float32)
        cart = d['vertices_cartesian'].numpy()
        faces_g = d['faces'].numpy()               # [cpb,F] global
        F_ = faces_g.shape[1]
        e_new = []; e_glob = []; tgt = []

        if tok.dim == 3:
            ei = d['edge_index'].numpy(); ec = d['edge_ctrl'].numpy()
            ctrl = {(int(ei[0, e]), int(ei[1, e])): ec[e] for e in range(ei.shape[1])}
            for fi in range(F_):
                for k0, k1 in edge_pairs:
                    p0 = int(faces_g[k0, fi]); p1 = int(faces_g[k1, fi])
                    a = int(old2new[p0]); b = int(old2new[p1])
                    P0, P1 = cart[p0], cart[p1]
                    info = ctrl.get((p0, p1))
                    if info is None:
                        B1 = P0 + (P1 - P0) / 3; B2 = P0 + 2 * (P1 - P0) / 3
                    else:
                        B1, B2 = info[0], info[1]
                    uh, nh1, nh2, L = PolytronTokenizer._chord_frame_3d(P0, P1)
                    s1 = float(np.dot(B1 - P0, uh) / L)
                    h1a = float(np.dot(B1 - P0, nh1) / L); h1b = float(np.dot(B1 - P0, nh2) / L)
                    s2 = float(np.dot(B2 - P0, uh) / L)
                    h2a = float(np.dot(B2 - P0, nh1) / L); h2b = float(np.dot(B2 - P0, nh2) / L)
                    e_new.append((a, b)); e_glob.append((p0, p1))
                    tgt.append((s1, h1a, h1b, s2, h2a, h2b))
            # edge_to_streamline is populated for dim=3 too (domain_extractor_3d.py
            # rebuilds it from edge_polyline/edge_polyline_offset) -- kept for
            # curve_eval's ground-truth comparison, same role as dim=2.
            e2s = d.get('edge_to_streamline')
        else:
            e2s = d['edge_to_streamline']
            for fi in range(F_):
                for k0, k1 in edge_pairs:
                    p0 = int(faces_g[k0, fi]); p1 = int(faces_g[k1, fi])
                    a = int(old2new[p0]); b = int(old2new[p1])
                    P0, P1 = cart[p0], cart[p1]
                    pts = e2s.get((p0, p1))
                    if pts is None:
                        s1, h1, s2, h2 = 1 / 3, 0.0, 2 / 3, 0.0
                    else:
                        B1, B2 = PolytronTokenizer._fit_cubic_bezier(P0, P1, pts)
                        uh, nh, L = PolytronTokenizer._chord_frame(P0, P1)
                        s1 = float(np.dot(B1 - P0, uh) / L); h1 = float(np.dot(B1 - P0, nh) / L)
                        s2 = float(np.dot(B2 - P0, uh) / L); h2 = float(np.dot(B2 - P0, nh) / L)
                    e_new.append((a, b)); e_glob.append((p0, p1)); tgt.append((s1, h1, s2, h2))

        examples.append({
            'vf': torch.from_numpy(vf),
            'e_new': torch.tensor(e_new, dtype=torch.long),
            'tgt': torch.tensor(tgt, dtype=torch.float32),
            'e_glob': np.array(e_glob, dtype=np.int64),
            'cart': cart, 'e2s': e2s})
    return examples


# ------------------------------------------------------------------
# Modell: Geometrie-Kopf
# ------------------------------------------------------------------
class GeomHeadModel(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_enc=4, n_edge=4, d_ff_mult=4,
                 vert_feat_dim=3, geom_out_dim=4):
        super().__init__()
        # dim=2: vert_feat_dim=3 (r,sin,cos), geom_out_dim=4 (s1,h1,s2,h2).
        # dim=3: vert_feat_dim=4 (r,sin,cos,z), geom_out_dim=6 (s1,h1a,h1b,s2,h2a,h2b).
        self.vert_proj = nn.Linear(vert_feat_dim, d_model)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, d_ff_mult * d_model,
                                         activation='gelu', batch_first=True,
                                         norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, n_enc)
        self.edge_mlp = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU())
        eec = nn.TransformerEncoderLayer(d_model, n_heads, d_ff_mult * d_model,
                                         activation='gelu', batch_first=True,
                                         norm_first=True)
        self.edge_enc = nn.TransformerEncoder(eec, n_edge)
        self.head = nn.Linear(d_model, geom_out_dim)

    def forward(self, vf, e_new, vpad=None, epad=None):
        """vf [B,M,vert_feat_dim], e_new [B,E,2] -> geom [B,E,geom_out_dim]."""
        H = self.encoder(self.vert_proj(vf), src_key_padding_mask=vpad)   # [B,M,d]
        d = H.size(-1)
        a = e_new[..., 0].clamp(min=0); b = e_new[..., 1].clamp(min=0)
        ha = torch.gather(H, 1, a.unsqueeze(-1).expand(-1, -1, d))
        hb = torch.gather(H, 1, b.unsqueeze(-1).expand(-1, -1, d))
        et = self.edge_mlp(torch.cat([ha, hb], dim=-1))                   # [B,E,d]
        et = self.edge_enc(et, src_key_padding_mask=epad)
        return self.head(et)                                             # [B,E,4]


# ------------------------------------------------------------------
# Batching
# ------------------------------------------------------------------
def collate(batch):
    """vert_feat_dim (3 or 4) and geom_out_dim (4 or 6) read from the data
    itself rather than hardcoded -- same reasoning as train_pointer.py:collate."""
    Mmax = max(e['vf'].shape[0] for e in batch)
    Emax = max(e['e_new'].shape[0] for e in batch)
    vert_feat_dim = batch[0]['vf'].shape[1]
    geom_out_dim = batch[0]['tgt'].shape[1]
    B = len(batch)
    vf = torch.zeros(B, Mmax, vert_feat_dim); vpad = torch.ones(B, Mmax, dtype=torch.bool)
    e_new = torch.zeros(B, Emax, 2, dtype=torch.long)
    tgt = torch.zeros(B, Emax, geom_out_dim); epad = torch.ones(B, Emax, dtype=torch.bool)
    for i, e in enumerate(batch):
        M, E = e['vf'].shape[0], e['e_new'].shape[0]
        vf[i, :M] = e['vf']; vpad[i, :M] = False
        e_new[i, :E] = e['e_new']; tgt[i, :E] = e['tgt']; epad[i, :E] = False
    return vf, e_new, tgt, vpad, epad


def run_epoch(model, order, examples, bs, opt, sched, clip, device, train=True,
              progress=False, epoch=0):
    model.train() if train else model.eval()
    tot = cnt = 0.0
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
            vf, e_new, tgt, vpad, epad = collate(batch)
            vf, e_new, tgt = vf.to(device), e_new.to(device), tgt.to(device)
            vpad, epad = vpad.to(device), epad.to(device)
            pred = model(vf, e_new, vpad, epad)                    # [B,E,4]
            m = ~epad
            loss = F.smooth_l1_loss(pred[m], tgt[m])
            if train:
                opt.zero_grad(); loss.backward()
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step(); sched.step()
            n = int(m.sum())
            tot += loss.item() * n; cnt += n
            if show_bar:
                bar.set_postfix(l1=f"{tot/max(cnt,1):.5f}")
            elif progress and (bi % log_every == 0 or bi == nsteps - 1):
                print(f"  ep{epoch} {'train' if train else 'val'} {bi+1:4d}/{nsteps} "
                      f"l1 {tot/max(cnt,1):.5f}", flush=True)
    return tot / max(cnt, 1)


@torch.no_grad()
def curve_eval(model, examples, ids, device, n=50):
    """Kurven-Rekonstruktionsfehler (max-dist/Chord) je Half-Edge: Modell vs Fit-Floor.
    dim inferred from the target width (4 = dim=2, 6 = dim=3) rather than a
    separate parameter -- self-contained, matches the data actually in hand."""
    model.eval()
    err_model = []; err_floor = []
    for i in ids:
        e = examples[i]
        vf = e['vf'].unsqueeze(0).to(device)
        en = e['e_new'].unsqueeze(0).to(device)
        pred = model(vf, en)[0].cpu().numpy()              # [E,4] or [E,6]
        tgt = e['tgt'].numpy()
        dim3 = tgt.shape[-1] == 6
        cart = e['cart']; e2s = e['e2s']; eg = e['e_glob']
        if e2s is None:
            continue
        for j in range(len(eg)):
            p0, p1 = int(eg[j, 0]), int(eg[j, 1])
            pts = e2s.get((p0, p1))
            if pts is None:
                continue
            pts = np.asarray(pts, float)
            P0, P1 = cart[p0], cart[p1]
            chord = np.linalg.norm(P1 - P0) + 1e-9
            if dim3:
                uh, nh1, nh2, L = PolytronTokenizer._chord_frame_3d(P0, P1)
            else:
                uh, nh, L = PolytronTokenizer._chord_frame(P0, P1)
            for scal, bucket in ((pred[j], err_model), (tgt[j], err_floor)):
                if dim3:
                    s1, h1a, h1b, s2, h2a, h2b = scal
                    B1 = P0 + s1 * L * uh + h1a * L * nh1 + h1b * L * nh2
                    B2 = P0 + s2 * L * uh + h2a * L * nh1 + h2b * L * nh2
                else:
                    s1, h1, s2, h2 = scal
                    B1 = P0 + s1 * L * uh + h1 * L * nh
                    B2 = P0 + s2 * L * uh + h2 * L * nh
                curve = _cubic_curve(P0, P1, B1, B2, n)
                # naechster-Punkt-Abstand Kurve<->GT (grob: pro GT-Punkt min dist)
                dd = np.linalg.norm(pts[:, None, :] - curve[None, :, :], axis=2).min(1)
                bucket.append(float(dd.max() / chord))
    return np.array(err_model), np.array(err_floor)


def _cubic_curve(P0, P1, B1, B2, n):
    t = np.linspace(0, 1, n)[:, None]
    return ((1 - t) ** 3 * P0 + 3 * (1 - t) ** 2 * t * B1 +
            3 * (1 - t) * t ** 2 * B2 + t ** 3 * P1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='domain_data_smoke.pt')
    ap.add_argument('--dim', type=int, default=2, choices=[2, 3])
    ap.add_argument('--corners-per-block', type=int, default=4)
    ap.add_argument('--d-model', type=int, default=256)
    ap.add_argument('--n-enc', type=int, default=4)
    ap.add_argument('--n-edge', type=int, default=4)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--warmup-frac', type=float, default=0.05)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--val-frac', type=float, default=0.1)
    ap.add_argument('--ce-every', type=int, default=5, help='Kurven-Eval alle N Ep.')
    ap.add_argument('--ce-n', type=int, default=60, help='Kurven-Eval Stichprobe')
    ap.add_argument('--limit', type=int, default=None)
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
    tok = PolytronTokenizer(max_vertices=max_v + 16, repr_mode='cubic_bezier',
                            dim=args.dim, corners_per_block=args.corners_per_block)
    print(f"baue Geom-Beispiele aus {len(data)} Meshes ...")
    t0 = time.time()
    examples = build_geom_examples(data, tok)
    Es = [e['e_new'].shape[0] for e in examples]
    print(f"{len(examples)} Beispiele in {time.time()-t0:.0f}s  |  Kanten/Mesh "
          f"{min(Es)}..{max(Es)}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(examples))
    n_val = max(1, int(args.val_frac * len(examples)))
    val_ids = perm[:n_val]; train_ids = perm[n_val:]
    print(f"split: {len(train_ids)} train / {len(val_ids)} val")

    vert_feat_dim = 4 if args.dim == 3 else 3
    geom_out_dim = 6 if args.dim == 3 else 4
    model = GeomHeadModel(d_model=args.d_model, n_enc=args.n_enc, n_edge=args.n_edge,
                          vert_feat_dim=vert_feat_dim, geom_out_dim=geom_out_dim).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"GeomHeadModel d={args.d_model} enc={args.n_enc} edge={args.n_edge}  "
          f"Params {n_par/1e6:.2f}M  device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    spe = math.ceil(len(train_ids) / args.batch)
    total = args.epochs * spe; warm = max(1, int(args.warmup_frac * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    ce_ids = val_ids[:min(args.ce_n, len(val_ids))]
    tg = time.time()
    best_val = float('inf'); best_ep = 0
    for ep in range(1, args.epochs + 1):
        tr = run_epoch(model, rng.permutation(train_ids), examples, args.batch,
                       opt, sched, args.clip, device, train=True, progress=True, epoch=ep)
        vl = run_epoch(model, val_ids, examples, args.batch, None, None, 0,
                       device, train=False)
        vram = torch.cuda.max_memory_allocated() / 1e9 if device == 'cuda' else 0
        best = ""
        if args.save and vl < best_val:                 # bestes Modell speichern
            best_val = vl; best_ep = ep
            os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
            torch.save({'model': model.state_dict(), 'd_model': args.d_model,
                        'n_enc': args.n_enc, 'n_edge': args.n_edge, 'kind': 'geom',
                        'epoch': ep, 'val_l1': vl}, args.save)
            best = "  *best*"
        if ep % args.ce_every == 0 or ep == args.epochs:
            em, ef = curve_eval(model, examples, ce_ids, device)
            ces = (f"curve-err%(model) med {100*np.median(em):.2f} p90 {100*np.percentile(em,90):.2f}"
                   f" | floor med {100*np.median(ef):.2f}")
        else:
            ces = "curve-err% -"
        print(f"ep {ep:3d}  tr-l1 {tr:.5f}  val-l1 {vl:.5f}  {ces}  "
              f"lr {sched.get_last_lr()[0]:.1e} peakVRAM {vram:.2f}GB{best}")
    print(f"\nfertig in {time.time()-tg:.0f}s.")
    if args.save:
        print(f"bestes Modell (ep{best_ep}, val-l1 {best_val:.5f}) -> {args.save}")


if __name__ == '__main__':
    main()
