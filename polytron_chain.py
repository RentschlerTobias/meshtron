"""
chain_e2e.py

END-TO-END-Verkettung von Plan B (nur 6-Face). Trainiert die drei Koepfe
(Stufe 1 Vertices, Stufe 2 Pointer-Faces, Stufe 3 HO-Geometrie) auf denselben
6F-Meshes (index-aligned) und fuehrt dann die volle Kette auf held-out aus:

    Punktwolke --S1--> Vertices --S2--> Faces --S3--> Geometrie --> Mesh

Kernfrage: FEHLER-FORTPFLANZUNG. S2/S3 laufen im Chain auf den GENERIERTEN
Vertices (nicht GT), S3 auf der generierten Topologie. Gemessen wird, wie stark
sich S1-Vertexfehler auf Face-Exaktheit (S2) und Kurvenfehler (S3) durchschlagen,
plus Loch-offen/Quads-distinct-Gueltigkeit. Galerie GT vs generiert -> figures/e2e/.

  ~/Environments/meshtron/bin/python chain_e2e.py \
      [--data domain_data_aug.pt] [--ep1 20 --ep2 25 --ep3 25] [--eval-n 150]
"""

import argparse
import math
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from prototype_twostage import TwoStageTokenizer
import vertex_head_prototype as vh
import pointer_head_prototype as ph
import geom_head_prototype as gh
import train_pointer as tp


def make_sched(opt, epochs, steps_per_ep, warmup_frac=0.05):
    total = max(1, epochs * steps_per_ep); warm = max(1, int(warmup_frac * total))
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))


def train_head(name, model, run_epoch, examples, train_ids, val_ids, epochs, bs,
               lr, device, rng, extra=()):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = make_sched(opt, epochs, math.ceil(len(train_ids) / bs))
    t0 = time.time()
    for ep in range(1, epochs + 1):
        tr = run_epoch(model, rng.permutation(train_ids), examples, bs, opt, sched,
                       1.0, device, *extra, train=True)
        if ep == epochs or ep % max(1, epochs // 4) == 0:
            vl = run_epoch(model, val_ids, examples, bs, None, None, 0, device,
                           *extra, train=False)
            trm = tr[0] if isinstance(tr, tuple) else tr
            vlm = vl[0] if isinstance(vl, tuple) else vl
            print(f"  [{name}] ep {ep:3d}  tr {trm:.4f}  val {vlm:.4f}")
    print(f"  [{name}] trainiert in {time.time()-t0:.0f}s")


# ---- Chain-Inferenz-Bausteine ----------------------------------------
def faces_to_edges(faces_new):
    """[F,4] -> [4F,2] gerichtete Half-Edges (Face-Traversal)."""
    e = []
    for f in faces_new:
        for k in range(4):
            e.append((int(f[k]), int(f[(k + 1) % 4])))
    return np.array(e, dtype=np.int64)


@torch.no_grad()
def chain_one(vmodel, pmodel, gmodel, pts, n, tok, START, STOP, device):
    """Volle Kette bei Aufloesung n (F = 6 n^2 explizit). Returns cart[Mgen,2],
    faces_new[F,4], edge_curves dict (a,b)->[.,2]. faces_new kann leer sein, falls
    S1 zu wenige Vertices erzeugt (Mgen < 4)."""
    polar = vh.s1_generate(vmodel, pts, n, tok, START, STOP, device)  # [Mgen,2]
    r, th = polar[:, 0], polar[:, 1]
    vert_feats = torch.tensor(np.stack([r, np.sin(th), np.cos(th)], 1),
                              dtype=torch.float32)             # [Mgen,3]
    cart = np.stack([r * np.cos(th), r * np.sin(th)], 1)       # [Mgen,2] (um center)

    Fexp = 6 * n * n                                          # Face-Zahl EXPLIZIT
    if polar.shape[0] < 4:                                    # zu wenig -> leer
        return cart, np.zeros((0, 4), dtype=np.int64), {}
    vf = vert_feats.unsqueeze(0).to(device)
    ptrs = pmodel.generate(vf, 4 * Fexp)                     # 4F Zeiger (explizit)
    faces_new = np.array(ptrs, dtype=np.int64).reshape(Fexp, 4)

    e_new = faces_to_edges(faces_new)
    en = torch.tensor(e_new, device=device).unsqueeze(0)
    geom = gmodel(vf, en)[0].cpu().numpy()                    # [24,4]

    curves = {}
    for j, (a, b) in enumerate(e_new):
        P0, P1 = cart[a], cart[b]
        uh, nh, L = TwoStageTokenizer._chord_frame(P0, P1)
        s1, h1, s2, h2 = geom[j]
        B1 = P0 + s1 * L * uh + h1 * L * nh
        B2 = P0 + s2 * L * uh + h2 * L * nh
        curves[(a, b)] = gh._cubic_curve(P0, P1, B1, B2, 24)
    return cart, faces_new, curves


def draw_gen(ax, cart, center, faces_new, curves):
    cmap = plt.get_cmap('tab20')
    for fi, f in enumerate(faces_new):
        loop = []
        for k in range(4):
            a, b = int(f[k]), int(f[(k + 1) % 4])
            loop.append(curves[(a, b)] + center[None])
        poly = np.vstack(loop)
        ax.fill(poly[:, 0], poly[:, 1], color=cmap(fi % 20), alpha=0.6, ec='k', lw=0.5)
    ax.set_aspect('equal'); ax.axis('off')


def draw_gt(ax, d):
    vc = d['vertices_cartesian'].numpy(); faces = d['faces'].numpy().T
    e2s = d['edge_to_streamline']; cmap = plt.get_cmap('tab20')
    for fi, q in enumerate(faces):
        loop = []
        for k in range(4):
            u, v = int(q[k]), int(q[(k + 1) % 4])
            c = e2s.get((u, v))
            if c is None:
                c = e2s.get((v, u))
                c = np.asarray(c, float)[::-1] if c is not None else np.stack([vc[u], vc[v]])
            else:
                c = np.asarray(c, float)
            loop.append(c)
        poly = np.vstack(loop)
        ax.fill(poly[:, 0], poly[:, 1], color=cmap(fi % 20), alpha=0.6, ec='k', lw=0.5)
    ax.set_aspect('equal'); ax.axis('off')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='domain_data_aug.pt')
    ap.add_argument('--d-model', type=int, default=256)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--ep1', type=int, default=20)
    ap.add_argument('--ep2', type=int, default=25)
    ap.add_argument('--ep3', type=int, default=25)
    ap.add_argument('--eval-n', type=int, default=150)
    ap.add_argument('--gallery', type=int, default=6)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--lr', type=float, default=5e-4, help='LR fuer alle 3 Koepfe')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--save-dir', default=None, help='3 Koepfe nach Training hier speichern')
    ap.add_argument('--load-dir', default=None, help='3 Koepfe laden, Training ueberspringen')
    ap.add_argument('--load-s1', default=None,
                    help='vortrainiertes S1 (vertex_head-Format) laden, nur S2+S3 trainieren')
    ap.add_argument('--out', default='figures/e2e/e2e_gallery.png')
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = TwoStageTokenizer(repr_mode='cubic_bezier')
    Qr, Qa = tok.Qr, tok.Qa
    START = Qr + 2 * Qa; STOP = Qr + 2 * Qa + 1; PAD = Qr + 2 * Qa + 2
    VOCAB = Qr + 2 * Qa + 3

    print(f"Lade {args.data} ...")
    data = torch.load(args.data, weights_only=False)
    six = [d for d in data if d['faces'].shape[1] in vh.FC2N]   # alle Facecounts
    if args.limit:
        six = six[:args.limit]
    from collections import Counter
    print(f"Meshes: {len(six)}  facecounts {dict(sorted(Counter(d['faces'].shape[1] for d in six).items()))}"
          f"  |  baue Beispiele (S1/S2/S3, index-aligned) ...")
    t0 = time.time()
    ex_v = vh.build_vertex_examples(six, tok)
    ex_p = ph.build_examples(six, tok)
    ex_g = gh.build_geom_examples(six, tok)
    max_len = max(e['seq'].numel() for e in ex_v) + 1
    print(f"  Beispiele in {time.time()-t0:.0f}s")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(six))
    n_val = max(1, len(six) // 10)
    val_ids = perm[:n_val]; train_ids = perm[n_val:]
    print(f"split: {len(train_ids)} train / {len(val_ids)} val")

    vmodel = vh.VertexGen(VOCAB, d=args.d_model, max_len=max_len, start_id=START).to(device)
    pmodel = ph.PointerFaceModel(d_model=args.d_model).to(device)
    gmodel = gh.GeomHeadModel(d_model=args.d_model).to(device)

    if args.load_dir:
        meta = torch.load(f"{args.load_dir}/meta.pt", weights_only=False)
        assert meta['d_model'] == args.d_model, \
            f"d_model mismatch: ckpt {meta['d_model']} != --d-model {args.d_model}"
        vmodel.load_state_dict(torch.load(f"{args.load_dir}/s1_vertex.pt", map_location=device))
        pmodel.load_state_dict(torch.load(f"{args.load_dir}/s2_pointer.pt", map_location=device))
        gmodel.load_state_dict(torch.load(f"{args.load_dir}/s3_geom.pt", map_location=device))
        print(f"== 3 Koepfe geladen aus {args.load_dir} (Training uebersprungen) ==")
    else:
        if args.save_dir:                        # meta frueh -> teilweise Ergebnisse nutzbar
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({'d_model': args.d_model, 'vocab': VOCAB, 'start': START,
                        'stop': STOP, 'pad': PAD, 'max_len': max_len,
                        'data': args.data, 'seed': args.seed}, f"{args.save_dir}/meta.pt")

        def _save(fname, m):                     # jede Stufe SOFORT nach Training sichern
            if args.save_dir:                    # (24h-Job-Crash behaelt fertige Stufen)
                torch.save(m.state_dict(), f"{args.save_dir}/{fname}")
                print(f"  == {fname} gespeichert -> {args.save_dir}/ ==", flush=True)

        if args.load_s1:
            ck = torch.load(args.load_s1, weights_only=False, map_location=device)
            assert ck['d_model'] == args.d_model, \
                f"S1 d_model {ck['d_model']} != --d-model {args.d_model}"
            vmodel.load_state_dict(ck['model'])
            print(f"== S1 geladen aus {args.load_s1} (ep{ck.get('epoch')}), trainiere nur S2+S3 ==")
            _save('s1_vertex.pt', vmodel)
        else:
            print("== Training S1 ==")
            train_head('S1', vmodel, vh.run_epoch, ex_v, train_ids, val_ids, args.ep1,
                       args.batch, args.lr, device, rng, extra=(START, PAD))
            _save('s1_vertex.pt', vmodel)
        print("== Training S2 ==")
        train_head('S2', pmodel, tp.run_epoch, ex_p, train_ids, val_ids, args.ep2,
                   args.batch, args.lr, device, rng)
        _save('s2_pointer.pt', pmodel)
        print("== Training S3 ==")
        train_head('S3', gmodel, gh.run_epoch, ex_g, train_ids, val_ids, args.ep3,
                   args.batch, args.lr, device, rng)
        _save('s3_geom.pt', gmodel)
        if args.save_dir:
            print(f"== 3 Koepfe gespeichert -> {args.save_dir}/ ==")

    print("== Chain-Inferenz auf held-out (je Facecount) ==")
    vmodel.eval(); pmodel.eval(); gmodel.eval()
    ev_ids = val_ids[:min(args.eval_n, len(val_ids))]
    B = {f: {'n': 0, 'count_ok': 0, 'verr': [], 'fex': 0, 'nf': 0, 'qd': 0, 'cerr': []}
         for f in vh.FACECOUNTS}
    for i in ev_ids:
        d = six[i]; fc = ex_v[i]['fc']; nlvl = ex_v[i]['n']; b = B[fc]
        cart, faces_new, curves = chain_one(vmodel, pmodel, gmodel, ex_v[i]['pts'],
                                            nlvl, tok, START, STOP, device)
        center = d['center'].numpy(); gt_vc = ex_v[i]['vc']
        b['n'] += 1
        if cart.shape[0] != gt_vc.shape[0]:      # falsche Vertexzahl -> Kette bricht
            continue
        b['count_ok'] += 1
        b['verr'].append(np.linalg.norm(cart + center - gt_vc, axis=1) / math.sqrt(2))
        gt_faces = ex_p[i][2]
        for gi, fi in zip(faces_new, gt_faces):
            b['nf'] += 1
            if np.array_equal(gi, fi):
                b['fex'] += 1
            if len(set(gi.tolist())) == 4:
                b['qd'] += 1
        if np.array_equal(faces_new, gt_faces):  # Topologie exakt -> Kurvenfehler
            e2s = d['edge_to_streamline']; e_glob = ex_g[i]['e_glob']; gcart = ex_g[i]['cart']
            for j, (a, bb) in enumerate(faces_to_edges(faces_new)):
                p0, p1 = int(e_glob[j, 0]), int(e_glob[j, 1])
                pts = e2s.get((p0, p1))
                if pts is None:
                    continue
                pts = np.asarray(pts, float); cur = curves[(a, bb)] + center[None]
                chord = np.linalg.norm(gcart[p1] - gcart[p0]) + 1e-9
                dd = np.linalg.norm(pts[:, None, :] - cur[None, :, :], axis=2).min(1)
                b['cerr'].append(float(dd.max() / chord))

    print(f"\n== E2E-Ergebnis ({len(ev_ids)} held-out) ==")
    for f in vh.FACECOUNTS:
        b = B[f]
        if b['n'] == 0:
            continue
        co = b['count_ok'] / b['n']
        vs = f"vert-err% med {100*np.median(np.concatenate(b['verr'])):.2f}" if b['verr'] else "vert-err -"
        fx = f"face-exact {b['fex']/b['nf']:.3f}" if b['nf'] else "face-exact -"
        qd = f"quads-dist {b['qd']/b['nf']:.3f}" if b['nf'] else ""
        cs = f"curve% med {100*np.median(b['cerr']):.2f}" if b['cerr'] else "curve -"
        print(f"  {f:3d}F (n={vh.FC2N[f]}): count-ok {co:.2f}  {vs}  {fx}  {qd}  {cs}  (N={b['n']})")

    # Galerie: je Facecount ein Beispiel
    print("== Galerie ==")
    gids = []
    for f in vh.FACECOUNTS:
        cand = [i for i in ev_ids if ex_v[i]['fc'] == f]
        gids += cand[:max(1, args.gallery // 4)]
    gids = gids[:args.gallery] or list(ev_ids[:args.gallery])
    fig, axes = plt.subplots(2, len(gids), figsize=(3 * len(gids), 6))
    axes = np.atleast_2d(axes)
    for c, i in enumerate(gids):
        d = six[i]; nlvl = ex_v[i]['n']
        cart, faces_new, curves = chain_one(vmodel, pmodel, gmodel, ex_v[i]['pts'],
                                            nlvl, tok, START, STOP, device)
        draw_gt(axes[0, c], d); axes[0, c].set_title(f"GT {ex_v[i]['fc']}F", fontsize=9)
        if faces_new.shape[0] > 0:
            draw_gen(axes[1, c], cart, d['center'].numpy(), faces_new, curves)
        axes[1, c].set_title(f"gen {faces_new.shape[0]}F", fontsize=9)
        axes[1, c].set_aspect('equal'); axes[1, c].axis('off')
    axes[0, 0].set_ylabel("GT", fontsize=11); axes[1, 0].set_ylabel("generiert", fontsize=11)
    fig.suptitle("Plan B end-to-end (variabler Facecount): Punktwolke + n -> Mesh",
                 fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches='tight')
    print("wrote", args.out)


if __name__ == '__main__':
    main()
