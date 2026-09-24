"""
polytron_chain.py

END-TO-END-Verkettung von Polytron (beliebige Face-/Block-Zahl, dim=2 oder
dim=3 -- kein 6-Block/F=6n^2-Vorlagenzwang mehr, siehe polytron_vertex_model.py
Modul-Docstring). Trainiert die drei Koepfe
(Stufe 1 Vertices, Stufe 2 Pointer-Faces, Stufe 3 HO-Geometrie) auf denselben
6F-Meshes (index-aligned) und fuehrt dann die volle Kette auf held-out aus:

    Punktwolke --S1--> Vertices --S2--> Faces --S3--> Geometrie --> Mesh

Kernfrage: FEHLER-FORTPFLANZUNG. S2/S3 laufen im Chain auf den GENERIERTEN
Vertices (nicht GT), S3 auf der generierten Topologie. Gemessen wird, wie stark
sich S1-Vertexfehler auf Face-Exaktheit (S2) und Kurvenfehler (S3) durchschlagen,
plus Loch-offen/Quads-distinct-Gueltigkeit. Galerie GT vs generiert -> figures/e2e/.

  ~/Environments/meshtron/bin/python polytron_chain.py \
      [--data domain_data_aug.pt] [--ep1 20 --ep2 25 --ep3 25] [--eval-n 150]
"""

import argparse
import math
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from meshtron.data.polytron_tokenizer import PolytronTokenizer
from meshtron.training.trainer import TrainingCancelled
import polytron_vertex_model as vh
import polytron_pointer_model as ph
import polytron_geom_model as gh
from meshtron.training import train_pointer as tp


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
def faces_to_edges(faces_new, edge_pairs):
    """[F,cpb] -> [len(edge_pairs)*F, 2] gerichtete Half-Edges (Face-Traversal),
    ueber die ECHTE Kantentopologie (edge_pairs = tok._face_edge_pairs()) --
    NICHT range(cpb)/(k+1)%cpb, das ist nur fuer ein planares Quad ein Ring
    (4 Ecken = 4 Kanten); ein Hex-Block hat 8 Ecken aber 12 Kanten, kein
    einzelner Ring (siehe der gleiche Fix in polytron_tokenizer.py)."""
    e = []
    for f in faces_new:
        for k0, k1 in edge_pairs:
            e.append((int(f[k0]), int(f[k1])))
    return np.array(e, dtype=np.int64)


@torch.no_grad()
def chain_one(vmodel, pmodel, gmodel, pts, n, tok, START, STOP, device):
    """Volle Kette, konditioniert auf die rohe Ziel-Face-/Block-Zahl n (siehe
    polytron_vertex_model.py -- kein 6n^2-Proxy mehr, n IST die erwartete Anzahl).
    Returns cart[Mgen,dim], faces_new[F,corners_per_block], edge_curves dict
    (a,b)->[.,dim]. faces_new kann leer sein, falls S1 zu wenige Vertices
    erzeugt (Mgen < corners_per_block)."""
    cpb = tok.corners_per_block
    edge_pairs = tok._face_edge_pairs()
    polar = vh.s1_generate(vmodel, pts, n, tok, START, STOP, device)  # [Mgen,dim]

    if tok.dim == 3:
        r, th, z = polar[:, 0], polar[:, 1], polar[:, 2]
        vert_feats = torch.tensor(np.stack([r, np.sin(th), np.cos(th), z], 1),
                                  dtype=torch.float32)          # [Mgen,4]
        cart = np.stack([r * np.cos(th), r * np.sin(th), z], 1)  # [Mgen,3] (um center)
    else:
        r, th = polar[:, 0], polar[:, 1]
        vert_feats = torch.tensor(np.stack([r, np.sin(th), np.cos(th)], 1),
                                  dtype=torch.float32)          # [Mgen,3]
        cart = np.stack([r * np.cos(th), r * np.sin(th)], 1)   # [Mgen,2] (um center)

    Fexp = n                                                  # Face-/Block-Zahl EXPLIZIT (rohe Zahl, kein Proxy)
    if polar.shape[0] < cpb:                                  # zu wenig -> leer
        return cart, np.zeros((0, cpb), dtype=np.int64), {}
    vf = vert_feats.unsqueeze(0).to(device)
    ptrs = pmodel.generate(vf, cpb * Fexp)                    # cpb*F Zeiger (explizit)
    faces_new = np.array(ptrs, dtype=np.int64).reshape(Fexp, cpb)

    e_new = faces_to_edges(faces_new, edge_pairs)
    en = torch.tensor(e_new, device=device).unsqueeze(0)
    geom = gmodel(vf, en)[0].cpu().numpy()                    # [len(edge_pairs)*F, 4 or 6]

    curves = {}
    for j, (a, b) in enumerate(e_new):
        P0, P1 = cart[a], cart[b]
        if tok.dim == 3:
            uh, nh1, nh2, L = PolytronTokenizer._chord_frame_3d(P0, P1)
            s1, h1a, h1b, s2, h2a, h2b = geom[j]
            B1 = P0 + s1 * L * uh + h1a * L * nh1 + h1b * L * nh2
            B2 = P0 + s2 * L * uh + h2a * L * nh1 + h2b * L * nh2
        else:
            uh, nh, L = PolytronTokenizer._chord_frame(P0, P1)
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


def run_polytron(cfg, ep1=None, ep2=None, ep3=None, eval_n=150, gallery=6,
                  limit=None, save_dir=None, load_dir=None, load_s1=None,
                  out='figures/e2e/e2e_gallery.png', stop_check=None):
    """Trains (or loads) the 3 Polytron heads end-to-end and runs the chain
    on held-out data. `cfg` supplies the shared hyperparameters
    (data_path, d_model, batch_size, learning_rate, seed); ep1/ep2/ep3 are
    per-head epoch counts (default to `cfg.num_epochs` for all three heads
    when not given -- pass them explicitly for the finer-grained schedule
    the original `chain_e2e.py --ep1/--ep2/--ep3` flags gave standalone
    runs). Called from `train.py --model-family polytron`, and from `main()`
    below for standalone CLI use.

    `stop_check`: optional `() -> bool`, polled between the coarse units of
    work this function has (before each of S1/S2/S3's full `train_head` call,
    and per held-out item in the chain-inference eval loop) -- raises
    `TrainingCancelled` if it returns True. This is coarser than `Trainer`'s
    per-epoch `on_epoch` hook (can't interrupt mid-epoch of a single head, S1
    on a big dataset could still run a while before the next check point),
    because `train_head`/the three heads' own `run_epoch` functions have no
    equivalent per-epoch callback of their own to hook into without changing
    all three independently -- this was the achievable granularity without
    a larger, separate change.
    """
    def _check():
        if stop_check is not None and stop_check():
            raise TrainingCancelled("run_polytron cancelled via stop_check")

    ep1 = cfg.num_epochs if ep1 is None else ep1
    ep2 = cfg.num_epochs if ep2 is None else ep2
    ep3 = cfg.num_epochs if ep3 is None else ep3

    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Lade {cfg.data_path} ...")
    data = torch.load(cfg.data_path, weights_only=False)
    six = list(data)   # alle Meshes, jede Face-/Block-Zahl (kein FACECOUNTS-Filter mehr)
    if limit:
        six = six[:limit]
    max_M = max(d['vertices_polar'].shape[0] for d in six)
    tok = PolytronTokenizer(repr_mode=cfg.repr_mode, dim=cfg.dim,
                            corners_per_block=cfg.corners_per_block,
                            max_vertices=max_M)
    Qr, Qa = tok.Qr, tok.Qa
    START = Qr + 2 * Qa; STOP = Qr + 2 * Qa + 1; PAD = Qr + 2 * Qa + 2
    VOCAB = Qr + 2 * Qa + 3
    from collections import Counter
    print(f"Meshes: {len(six)}  facecounts {dict(sorted(Counter(d['faces'].shape[1] for d in six).items()))}"
          f"  |  baue Beispiele (S1/S2/S3, index-aligned) ...")
    t0 = time.time()
    ex_v = vh.build_vertex_examples(six, tok)
    ex_p = ph.build_examples(six, tok)
    ex_g = gh.build_geom_examples(six, tok)
    max_len = max(e['seq'].numel() for e in ex_v) + 1
    print(f"  Beispiele in {time.time()-t0:.0f}s")

    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(len(six))
    n_val = max(1, len(six) // 10)
    val_ids = perm[:n_val]; train_ids = perm[n_val:]
    print(f"split: {len(train_ids)} train / {len(val_ids)} val")

    vert_feat_dim = 4 if cfg.dim == 3 else 3
    geom_out_dim = 6 if cfg.dim == 3 else 4
    res_max = max(d['faces'].shape[1] for d in six)  # rohe Konditionierungs-Obergrenze

    vmodel = vh.VertexGen(VOCAB, d=cfg.d_model, max_len=max_len, start_id=START,
                          res_max=res_max).to(device)
    pmodel = ph.PointerFaceModel(
        d_model=cfg.d_model, vert_feat_dim=vert_feat_dim,
        max_ptr=max(e[1].numel() for e in ex_p) + 8).to(device)
    gmodel = gh.GeomHeadModel(d_model=cfg.d_model, vert_feat_dim=vert_feat_dim,
                              geom_out_dim=geom_out_dim).to(device)

    if load_dir:
        meta = torch.load(f"{load_dir}/meta.pt", weights_only=False)
        assert meta['d_model'] == cfg.d_model, \
            f"d_model mismatch: ckpt {meta['d_model']} != cfg.d_model {cfg.d_model}"
        vmodel.load_state_dict(torch.load(f"{load_dir}/s1_vertex.pt", map_location=device))
        pmodel.load_state_dict(torch.load(f"{load_dir}/s2_pointer.pt", map_location=device))
        gmodel.load_state_dict(torch.load(f"{load_dir}/s3_geom.pt", map_location=device))
        print(f"== 3 Koepfe geladen aus {load_dir} (Training uebersprungen) ==")
    else:
        if save_dir:                              # meta frueh -> teilweise Ergebnisse nutzbar
            os.makedirs(save_dir, exist_ok=True)
            torch.save({'d_model': cfg.d_model, 'vocab': VOCAB, 'start': START,
                        'stop': STOP, 'pad': PAD, 'max_len': max_len,
                        'data': cfg.data_path, 'seed': cfg.seed}, f"{save_dir}/meta.pt")

        def _save(fname, m):                     # jede Stufe SOFORT nach Training sichern
            if save_dir:                          # (24h-Job-Crash behaelt fertige Stufen)
                torch.save(m.state_dict(), f"{save_dir}/{fname}")
                print(f"  == {fname} gespeichert -> {save_dir}/ ==", flush=True)

        if load_s1:
            ck = torch.load(load_s1, weights_only=False, map_location=device)
            assert ck['d_model'] == cfg.d_model, \
                f"S1 d_model {ck['d_model']} != cfg.d_model {cfg.d_model}"
            vmodel.load_state_dict(ck['model'])
            print(f"== S1 geladen aus {load_s1} (ep{ck.get('epoch')}), trainiere nur S2+S3 ==")
            _save('s1_vertex.pt', vmodel)
        else:
            _check()
            print("== Training S1 ==")
            train_head('S1', vmodel, vh.run_epoch, ex_v, train_ids, val_ids, ep1,
                       cfg.batch_size, cfg.learning_rate, device, rng, extra=(START, PAD))
            _save('s1_vertex.pt', vmodel)
        _check()
        print("== Training S2 ==")
        train_head('S2', pmodel, tp.run_epoch, ex_p, train_ids, val_ids, ep2,
                   cfg.batch_size, cfg.learning_rate, device, rng)
        _save('s2_pointer.pt', pmodel)
        _check()
        print("== Training S3 ==")
        train_head('S3', gmodel, gh.run_epoch, ex_g, train_ids, val_ids, ep3,
                   cfg.batch_size, cfg.learning_rate, device, rng)
        _save('s3_geom.pt', gmodel)
        if save_dir:
            print(f"== 3 Koepfe gespeichert -> {save_dir}/ ==")

    print("== Chain-Inferenz auf held-out (je Facecount) ==")
    vmodel.eval(); pmodel.eval(); gmodel.eval()
    ev_ids = val_ids[:min(eval_n, len(val_ids))]
    facecounts = sorted({ex_v[i]['fc'] for i in ev_ids})   # dynamisch, keine feste Vorlage mehr
    B = {f: {'n': 0, 'count_ok': 0, 'verr': [], 'fex': 0, 'nf': 0, 'qd': 0, 'cerr': []}
         for f in facecounts}
    edge_pairs = tok._face_edge_pairs()
    cpb = tok.corners_per_block
    for i in ev_ids:
        _check()
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
            if len(set(gi.tolist())) == cpb:
                b['qd'] += 1
        if np.array_equal(faces_new, gt_faces):  # Topologie exakt -> Kurvenfehler
            e2s = d['edge_to_streamline']; e_glob = ex_g[i]['e_glob']; gcart = ex_g[i]['cart']
            for j, (a, bb) in enumerate(faces_to_edges(faces_new, edge_pairs)):
                p0, p1 = int(e_glob[j, 0]), int(e_glob[j, 1])
                pts = e2s.get((p0, p1))
                if pts is None:
                    continue
                pts = np.asarray(pts, float); cur = curves[(a, bb)] + center[None]
                chord = np.linalg.norm(gcart[p1] - gcart[p0]) + 1e-9
                dd = np.linalg.norm(pts[:, None, :] - cur[None, :, :], axis=2).min(1)
                b['cerr'].append(float(dd.max() / chord))

    print(f"\n== E2E-Ergebnis ({len(ev_ids)} held-out) ==")
    for f in facecounts:
        b = B[f]
        if b['n'] == 0:
            continue
        co = b['count_ok'] / b['n']
        vs = f"vert-err% med {100*np.median(np.concatenate(b['verr'])):.2f}" if b['verr'] else "vert-err -"
        fx = f"face-exact {b['fex']/b['nf']:.3f}" if b['nf'] else "face-exact -"
        qd = f"quads-dist {b['qd']/b['nf']:.3f}" if b['nf'] else ""
        cs = f"curve% med {100*np.median(b['cerr']):.2f}" if b['cerr'] else "curve -"
        print(f"  {f:3d}F: count-ok {co:.2f}  {vs}  {fx}  {qd}  {cs}  (N={b['n']})")

    # Galerie: je Facecount ein Beispiel
    print("== Galerie ==")
    gids = []
    for f in facecounts:
        cand = [i for i in ev_ids if ex_v[i]['fc'] == f]
        gids += cand[:max(1, gallery // 4)]
    gids = gids[:gallery] or list(ev_ids[:gallery])
    fig, axes = plt.subplots(2, len(gids), figsize=(3 * len(gids), 6),
                             squeeze=False)
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
    fig.suptitle("Polytron end-to-end (variabler Facecount): Punktwolke + n -> Mesh",
                 fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print("wrote", out)


def main():
    """Standalone CLI, unchanged flags from the original chain_e2e.py."""
    from meshtron.training.config import PipelineConfig

    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='domain_data_aug.pt')
    ap.add_argument('--dim', type=int, default=2, choices=[2, 3])
    ap.add_argument('--corners-per-block', type=int, default=4,
                    help='4 = 2D-Quad, 8 = 3D-Hex-Block.')
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

    cfg = PipelineConfig(
        model_family='polytron', dim=args.dim, corners_per_block=args.corners_per_block,
        data_path=args.data, d_model=args.d_model,
        batch_size=args.batch, learning_rate=args.lr, seed=args.seed,
    )
    run_polytron(cfg, ep1=args.ep1, ep2=args.ep2, ep3=args.ep3, eval_n=args.eval_n,
                 gallery=args.gallery, limit=args.limit, save_dir=args.save_dir,
                 load_dir=args.load_dir, load_s1=args.load_s1, out=args.out)


if __name__ == '__main__':
    main()
