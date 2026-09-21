"""generate.py

Pfad 1 Inference-Glue: HexaRow-Modell-Ckpt -> Token-Generation -> Detokenize
-> Rekonstruiertes Hexa-Mesh (VTK + Plot).

Input-Mesh = rohe xyz Vertices (z.B. tistos-Mesh); Polar-Umwandlung hier:
    r = sqrt(x^2 + y^2), theta = atan2(y, x), z unveraendert, center = 0.
Conditioning: n gescannte vertices_polar (r/z normalisiert auf Checkpoint-
bounds, theta als sin/cos) + face_count (Blockzahl).

  uv run python generate.py --idx 0 --ckpt data/hexarow_full_model_3090.pt
  uv run python generate.py --mesh <tistos-mesh.pt> --blocks 20 ...
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from hexa_row_tokenizer import HexaRowTokenizer
from mesh_validation import validate_generated_mesh
from train_hexarow_full import GPTCond, sample_points


# ----------------------------------------------------------- KV-Cache forward
# Kopie der Block-Mathematik aus train_hexarow_full.Block mit K/V-Cache
# (Decode: nur 1 Token pro Schritt). FiLM-Conditioning passiert vor den
# Blocks und ist sequenzunabhaengig -> gecachte K/V (post-Conditioning) korrekt.
@torch.no_grad()
def forward_cached(model: GPTCond, xs, pc, fc, kv: list, pos0: int, slot=None):
    B, L = xs.shape
    dev = xs.device
    pos = torch.arange(pos0, pos0 + L, device=dev)
    h = model.tok(xs) + model.pos(pos)[None]
    if slot is not None:
        h = h + model.slot(slot)  # Slot-Embedding wie im Trainer (Paritaet!)
    D = h.shape[-1]
    c = model.condition(pc, fc)
    h = h * (1 + torch.tanh(c)[:, None]) + c[:, None]
    for i, blk in enumerate(model.blocks):
        heads = blk.heads
        dh = D // heads
        qkv = blk.qkv(blk.ln1(h)).reshape(B, L, 3, heads, dh)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # je [B, H, L, dh]
        if kv[i][0] is not None:
            k = torch.cat([kv[i][0], k], dim=2)
            v = torch.cat([kv[i][1], v], dim=2)
        kv[i] = [k, v]
        a = F.scaled_dot_product_attention(q, k, v)  # L==1: volle Attention
        a = a.transpose(1, 2).reshape(B, L, D)
        h = h + blk.drop(blk.proj(a))
        h = h + blk.mlp(blk.drop(blk.ln2(h)))
    return model.head(model.ln(h))


# ------------------------------------------------------------------- Helpers
def mesh_to_polar(xyz: np.ndarray) -> np.ndarray:
    """Rohe xyz [M,3] -> vertices_polar [M,3] (r, theta, z), center=0."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    return np.stack([np.hypot(x, y), np.arctan2(y, x), z], axis=-1)


def load_sample(obj, idx: int):
    """Sample aus .pt-Liste oder dict -> (xyz, blocks, name, faces_T|None)."""
    if isinstance(obj, (list, tuple)):
        s = obj[idx]
        xyz = np.asarray(s.get("vertices_cartesian", s.get("vertices")),
                         dtype=np.float64)
        faces = s.get("faces")
        blocks = int(faces.shape[1]) if faces is not None else -1
        return xyz, blocks, s.get("name", f"sample{idx}"), \
            (faces.T if faces is not None else None)
    if isinstance(obj, dict):
        xyz = np.asarray(obj.get("vertices", obj.get("vertices_cartesian")),
                         dtype=np.float64)
        faces = obj.get("faces")
        blocks = int(faces.shape[1]) if faces is not None else -1
        return xyz, blocks, obj.get("name", "mesh"), \
            (faces.T if faces is not None else None)
    raise ValueError(f"unbekanntes Mesh-Format in Sample {idx}")


def slot_mask(tok, seq: list, cnt: int, vocab: int, coords: str) -> torch.Tensor:
    """Constrained-Decoding-Maske: erlaubt je Schritt nur grammatikalisch
    legale Tokens (Slot-Range + sep/stop nur an Row-/Stream-Grenzen)."""
    core = tok.core
    npt = 3 if coords == "cart" else 4
    gsize = 4 * npt      # Vertex-Gruppe (1 Vert) = npt Tokens, Vert-Blockpaar = 2
    head_n = 8 * npt     # Head-Row enthaelt 8 Verts (Entry+Exit-Ring)
    if coords == "cart":
        lo, hi = core.off_r, core.off_idx
    else:
        slot = cnt % npt
        if slot == 1:
            lo, hi = core.off_ts, core.off_tc
        elif slot == 2:
            lo, hi = core.off_tc, core.off_idx
        else:  # r und z teilen denselben off_r-Quantizer
            lo, hi = core.off_r, core.Qr
    row_len, seen_sep = 0, False
    for t in seq[1:]:
        if t == core.sep_token:
            seen_sep, row_len = True, 0
        elif t == core.stop_token:
            break
        elif t in _SPECIAL_SET:
            continue
        else:
            row_len += 1
    m = torch.full((vocab,), float("-inf"))
    m[lo:hi] = 0.0
    at_group = row_len % gsize == 0 and row_len > 0
    # erste Row endet erst nach komplettem Head-Block (2 Gruppen)
    min_row = head_n if not seen_sep else gsize
    if at_group and row_len >= min_row:
        m[core.sep_token] = 0.0
        m[core.stop_token] = 0.0
    if seq[-1] == core.stop_token:
        m[:] = float("-inf")
        m[core.end_token] = 0.0
    return m


_SPECIAL_SET: set = set()


@torch.no_grad()
def generate(model, pc, fc, start_id, stop_id, sep_id, max_tokens, temperature,
             top_k, dev, dtype, specials=(), use_slot=True, tok=None,
             constrained=True, coords="polar"):
    """Autoregressives Sampling mit KV-Cache bis stop_token / Budget.
    slot: Position im Vertex-Quant (0..3) des INPUT-Tokens je Schritt —
    identische Semantik wie _slot_ids in train_hexarow_full (Parität).
    constrained=True: Slot-Maske erzwingt Row-Grammatik (32/16-To-Rows,
    sep/stop nur an legalen Stellen)."""
    global _SPECIAL_SET
    _SPECIAL_SET = set(specials)
    kv: list = [[None, None] for _ in model.blocks]
    seq = [start_id]
    pos0 = 0
    cnt = 0  # non-special Tokens seit letztem Special (Quant-Positionszähler)
    while len(seq) < max_tokens:
        x = torch.tensor([[seq[-1]]], dtype=torch.long, device=dev)
        s = torch.zeros((1, 1), dtype=torch.long, device=dev)
        if use_slot:
            s.fill_(cnt % (3 if coords == "cart" else 4))
        with torch.autocast(dev, dtype=dtype, enabled=dev == "cuda"):
            logits = forward_cached(model, x, pc, fc, kv, pos0,
                                    slot=s if use_slot else None)
        pos0 += 1
        nxt_l = (logits[:, -1, :] / max(1e-9, temperature)).float()
        if constrained and tok is not None:
            nxt_l = nxt_l + slot_mask(tok, seq, cnt, nxt_l.shape[-1],
                                      coords).to(nxt_l.device)
        if top_k and top_k > 0:
            kth = torch.topk(nxt_l, min(top_k, nxt_l.shape[-1]),
                             dim=-1).values.min(dim=-1, keepdim=True).values
            nxt_l = nxt_l.masked_fill(nxt_l < kth, float("-inf"))
        nxt = int(torch.multinomial(F.softmax(nxt_l, dim=-1), 1).item())
        if not (0 <= nxt < model.tok.weight.shape[0]):
            raise RuntimeError(f"Sampled token id {nxt} out of vocab "
                               f"{model.tok.weight.shape[0]} (logits finite: "
                               f"{bool(torch.isfinite(nxt_l).all())})")
        pos_rows = model.pos.weight.shape[0]
        if pos0 >= pos_rows:
            print(f"warn: positional limit {pos_rows} ohne stop erreicht")
            break
        seq.append(nxt)
        if nxt in specials:
            cnt = 0
        else:
            cnt += 1
        if nxt == stop_id:
            break
    _SPECIAL_SET = set()
    return seq


def detokenize_safe(toks: list, tok: HexaRowTokenizer, stop_id: int):
    """Detokenize; bei unvollstaendiger Schluss-Row: Zeilen validieren und an
    der ersten unvollstaendigen Zeile trimmen."""
    if toks[-1] != stop_id:
        toks = toks + [stop_id]
    try:
        return tok.detokenize(toks), None
    except AssertionError as e:
        rows, cur = [], []
        sep = tok.core.sep_token
        ignore = {tok.core.sep2_token, tok.core.pad_token, tok.core.end_token}
        for t in toks[1:-1]:
            if t in ignore:
                continue  # Sonder-Tokens counten die Quant-Grammatik nicht
            if t == sep:
                if cur:
                    rows.append(cur)
                cur = []
            else:
                cur.append(t)
        if cur and toks[-1] != stop_id:
            # abgebrochene Schlusszeile ohne sep: nur pruefen, nicht verlieren
            rows.append(cur)
        valid, seen_first = [], False
        for r in rows:
            if not seen_first:
                ok = len(r) >= 32 and (len(r) - 32) % 4 == 0
            else:
                ok = len(r) >= 16 and len(r) % 4 == 0
            if not ok:
                continue  # kaputte Row (v.a. fuehrendes Fragment) ueberspringen
            valid.append(r)
            seen_first = True
        if not valid:
            return None, str(e)
        trimmed = [toks[0]]
        for r in valid:
            trimmed += r + [sep]
        trimmed.append(stop_id)
        try:
            return tok.detokenize(trimmed), str(e)
        except AssertionError as e2:
            return None, f"trim-Neuaufbau fehlgeschlagen: {e2}"


# -------------------------------------------------------------------- Output
_BLOCK_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7))


def write_vtk(path, vpt_cart, blk, pc_cart=None):
    """Legacy-ASCII-VTK UNSTRUCTURED_GRID, Hexa-Zellen (Typ 12); optionale
    Conditioning-Punktwolke als 1-Punkt-Zellen (Typ 1)."""
    m, f = vpt_cart.shape[0], len(blk)
    np_pts = m + (len(pc_cart) if pc_cart is not None else 0)
    with open(path, "w") as fh:
        fh.write("# vtk DataFile Version 2.0\nmeshtron generate\nASCII\n")
        fh.write("DATASET UNSTRUCTURED_GRID\n")
        fh.write(f"POINTS {m + (len(pc_cart) if pc_cart is not None else 0)} double\n")
        for p in vpt_cart:
            fh.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        if pc_cart is not None:
            for p in pc_cart:
                fh.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        cells = [f"8 {' '.join(str(int(i)) for i in b)}" for b in blk]
        types = ["12"] * f
        if pc_cart is not None:
            cells += [f"1 {m + k}" for k in range(len(pc_cart))]
            types += ["1"] * len(pc_cart)
        fh.write(f"CELLS {len(cells)} {sum(9 if t == '12' else 2 for t in types)}\n")
        fh.write("\n".join(cells) + "\n")
        fh.write(f"CELL_TYPES {len(types)}\n" + "\n".join(types) + "\n")


def draw_mesh(ax, vpt_cart, blk, color, alpha):
    V = np.asarray(vpt_cart)
    for b in blk:
        ids = [int(i) for i in b]
        for a, z in _BLOCK_EDGES:
            ax.plot(*V[[ids[a], ids[z]]].T, color=color, lw=0.4, alpha=alpha)


def plot_result(path, vpt_cart, blk, gt, title, pc_cart=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    draw_mesh(ax, vpt_cart, blk, "tab:blue", 0.9)
    if pc_cart is not None:
        ax.scatter(pc_cart[:, 0], pc_cart[:, 1], pc_cart[:, 2],
                   s=2, color="tab:green", alpha=0.5)
    if gt is not None:
        draw_mesh(ax, gt[0], gt[1], "tab:red", 0.35)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(
        description="HexaRow-Generation aus Checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="data/hexarow_full_model_3090.pt")
    ap.add_argument("--mesh", default="", help=".pt (Sample-Liste/-dict, rohe xyz)")
    ap.add_argument("--src", default="data/polytron_data_3d_full_aug.pt")
    ap.add_argument("--idx", type=int, default=-1)
    ap.add_argument("--blocks", type=int, default=-1,
                    help="face_count-Conditioning; -1 = auto aus Sample")
    ap.add_argument("--n-points", type=int, default=1000)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=40000)
    ap.add_argument("--out", default="data/gen.vtk")
    ap.add_argument("--plot", default="data/gen.png")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump-seq", default="",
                    help="Token-Sequence als .pt dumpen (Detail-Analyse)")
    ap.add_argument("--unconstrained", action="store_true",
                    help="alte freie Sampling-Route (ohne Slot-Maske)")
    ap.add_argument("--coords", default="polar", choices=["polar", "cart"])
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32

    ck = torch.load(args.ckpt, weights_only=False)
    cfg = ck["cfg"]
    rb = tuple(float(v) for v in ck["r_bounds"])
    zb = tuple(float(v) for v in ck["z_bounds"])
    max_len = int(ck["model"]["pos.weight"].shape[0])
    if args.max_tokens > max_len - 2:
        print(f"note: --max-tokens {args.max_tokens} > pos-Matrix ({max_len}) "
              f"-> gekappt auf {max_len - 2} (sonst pos-Embedding-Overlauf)")
        args.max_tokens = max_len - 2
    print(f"ckpt d={cfg['d']} L={cfg['layers']} H={cfg['heads']} | bounds r={rb} z={zb}")

    model = GPTCond(ck["vocab"], cfg["d"], cfg["layers"], cfg["heads"], max_len,
                    ck["pad_id"], 0.0, cfg["n_points"], cfg["n_latent"]).to(dev)
    missing = model.load_state_dict(ck["model"], strict=False)
    # CKPTs vom Trainer-Stand < Slot-Upgrade enthalten slot.weight nicht
    # (Forward ohne slot-Arg nutzt es nicht) -> nur das ist veraendert ok.
    if missing.missing_keys:
        print(f"note: fehlende CKPT-Keys (altes Format, ok wenn slot): "
              f"{missing.missing_keys}")
    model.eval()
    tok = HexaRowTokenizer(r_bounds=rb, z_bounds=zb)
    core = tok.core

    # Quelle / Conditioning
    gt, xyz = None, None
    if args.mesh:
        obj = torch.load(args.mesh, weights_only=False)
        if isinstance(obj, dict) and 'samples' in obj:
            obj = obj['samples']
        xyz, blocks, name, faces_t = load_sample(obj, max(0, args.idx))
        if faces_t is not None:
            gt_faces_t = faces_t
            gt = (xyz, faces_t.tolist())
    elif args.idx >= 0:
        src = torch.load(args.src, weights_only=False)
        if isinstance(src, dict) and 'samples' in src:
            src = src['samples']
        xyz, blocks, name, faces_t = load_sample(src, args.idx)
        if faces_t is not None:
            gt = (xyz, faces_t.tolist())
    else:
        if args.blocks < 0:
            ap.error("--blocks oder --idx noetig")
        xyz, blocks, name = None, args.blocks, "unconditioned"
    if blocks < 0:
        if args.blocks < 0:
            ap.error("--blocks noetig (Blockzahl des Inputs unbekannt)")
        blocks = args.blocks
    print(f"mesh={name} blocks={blocks}")

    rng = np.random.default_rng(args.seed)
    if xyz is not None:
        pts = sample_points(mesh_to_polar(xyz), args.n_points, rb, zb, rng)
    else:
        pts = np.zeros((args.n_points, 4), dtype=np.float64)
    pc = torch.as_tensor(pts[None], dtype=torch.float32, device=dev)
    fc = torch.tensor([float(blocks)], device=dev)

    # Generation
    specials = {core.start_token, core.end_token, core.sep_token,
                core.sep2_token, core.stop_token, core.pad_token}
    use_slot = "slot.weight" in ck["model"]
    if not use_slot:
        print("note: CKPT ohne slot.weight (alter Trainer-Stand) -> decode ohne Slot-Embedding")
    seq = generate(model, pc, fc, core.start_token, core.stop_token,
                   core.sep_token, args.max_tokens, args.temperature,
                   args.top_k, dev, dtype, specials, use_slot, tok=tok,
                   constrained=not args.unconstrained, coords=args.coords)
    stopped = seq[-1] == core.stop_token
    print(f"generated tokens={len(seq)} (stop={'ja' if stopped else 'NEIN (cap)'}) "
          f"rows={seq.count(core.sep_token)}")

    res, trim = detokenize_safe(seq, tok, core.stop_token)
    if args.dump_seq:
        torch.save(torch.tensor(seq), args.dump_seq)
        print(f"seq dumped: {args.dump_seq}")
    if res is None:
        ignore = {core.sep2_token, core.pad_token, core.end_token}
        rows, cur = [], []
        for t in seq[1:]:
            if t == core.stop_token:
                break
            if t in ignore:
                continue
            if t == core.sep_token:
                rows.append(len(cur)); cur = []
            else:
                cur.append(t)
        if cur and seq[-1] != core.stop_token:
            rows.append(len(cur))
        print(f"DIAGNOSE: keine valide Row rekonstruierbar ({trim})")
        print(f"  rows gesamt={seq.count(core.sep_token)}, tokens={len(seq)}, "
              f"row-laengen={rows} -> fuer Detail-Analyse seq dumpen")
        return 2
    vpt, blk = res
    if trim:
        print(f"WARN: Generation instabil, getrimmt: {trim}")
    print(f"reconstructed: verts={vpt.shape[0]} blocks={blk.shape[0]}")

    v_np = vpt.numpy()
    vcart = np.stack([v_np[:, 0] * np.cos(v_np[:, 1]),
                      v_np[:, 0] * np.sin(v_np[:, 1]),
                      v_np[:, 2]], axis=-1)
    validation = validate_generated_mesh(vcart, blk.numpy(), expected_blocks=blocks)
    if not validation.valid:
        print("INVALID GENERATED MESH: " + "; ".join(validation.errors))
        print(f"  vertices={validation.n_vertices} blocks={validation.n_blocks} "
              f"expected_blocks={blocks}")
        return 2
    print("mesh validation: valid")
    pc_cart = None
    if xyz is not None:
        r01, s01, c01, z01 = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
        pc_th = np.arctan2(s01, c01)
        pc_cart = np.stack([rb[0] + r01 * (rb[1] - rb[0]) * np.cos(pc_th),
                            rb[0] + r01 * (rb[1] - rb[0]) * np.sin(pc_th),
                            zb[0] + z01 * (zb[1] - zb[0])], axis=-1)
    write_vtk(args.out, vcart, blk.tolist(), pc_cart)
    print(f"saved {args.out}")
    if args.plot:
        title = "generated (blau) vs source (rot) vs punktwolke (grün)" if gt else "generated (blau)"
        plot_result(args.plot, vcart, blk.tolist(), gt, title, pc_cart)
        print(f"saved {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
