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
from train_hexarow_full import GPTCond, sample_points


# ----------------------------------------------------------- KV-Cache forward
# Kopie der Block-Mathematik aus train_hexarow_full.Block mit K/V-Cache
# (Decode: nur 1 Token pro Schritt). FiLM-Conditioning passiert vor den
# Blocks und ist sequenzunabhaengig -> gecachte K/V (post-Conditioning) korrekt.
@torch.no_grad()
def forward_cached(model: GPTCond, xs, pc, fc, kv: list, pos0: int):
    B, L = xs.shape
    dev = xs.device
    pos = torch.arange(pos0, pos0 + L, device=dev)
    h = model.tok(xs) + model.pos(pos)[None]
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


@torch.no_grad()
def generate(model, pc, fc, start_id, stop_id, sep_id, max_tokens, temperature,
             top_k, dev, dtype):
    """Autoregressives Sampling mit KV-Cache bis stop_token / Budget."""
    kv: list = [[None, None] for _ in model.blocks]
    seq = [start_id]
    pos0 = 0
    while len(seq) < max_tokens:
        x = torch.tensor([[seq[-1]]], dtype=torch.long, device=dev)
        with torch.autocast(dev, dtype=dtype, enabled=dev == "cuda"):
            logits = forward_cached(model, x, pc, fc, kv, pos0)
        pos0 += 1
        nxt_l = (logits[:, -1, :] / max(1e-9, temperature)).float()
        if top_k and top_k > 0:
            kth = torch.topk(nxt_l, min(top_k, nxt_l.shape[-1]),
                             dim=-1).values.min(dim=-1, keepdim=True).values
            nxt_l = nxt_l.masked_fill(nxt_l < kth, float("-inf"))
        nxt = int(torch.multinomial(F.softmax(nxt_l, dim=-1), 1).item())
        seq.append(nxt)
        if nxt == stop_id:
            break
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
        for t in toks[1:-1]:
            if t == sep:
                if cur:
                    rows.append(cur)
                cur = []
            else:
                cur.append(t)
        valid, seen_first = [], False
        for r in rows:
            if not seen_first:
                ok = len(r) >= 32 and (len(r) - 32) % 4 == 0
            else:
                ok = len(r) >= 16 and len(r) % 4 == 0
            if not ok:
                break
            valid.append(r)
            seen_first = True
        if not valid:
            raise
        trimmed = [toks[0]]
        for r in valid:
            trimmed += r + [sep]
        trimmed.append(stop_id)
        return tok.detokenize(trimmed), str(e)


# -------------------------------------------------------------------- Output
_BLOCK_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7))


def write_vtk(path: str, vpt_cart: np.ndarray, blk) -> None:
    """Legacy-ASCII-VTK UNSTRUCTURED_GRID, Hexa-Zellen (Typ 12)."""
    m, f = vpt_cart.shape[0], len(blk)
    with open(path, "w") as fh:
        fh.write("# vtk DataFile Version 2.0\nmeshtron generate\nASCII\n")
        fh.write("DATASET UNSTRUCTURED_GRID\n")
        fh.write(f"POINTS {m} double\n")
        for p in vpt_cart:
            fh.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        fh.write(f"CELLS {f} {9 * f}\n")
        for b in blk:
            fh.write("8 " + " ".join(str(int(i)) for i in b) + "\n")
        fh.write(f"CELL_TYPES {f}\n" + "12\n" * f)


def draw_mesh(ax, vpt_cart, blk, color, alpha):
    V = np.asarray(vpt_cart)
    for b in blk:
        ids = [int(i) for i in b]
        for a, z in _BLOCK_EDGES:
            ax.plot(*V[[ids[a], ids[z]]].T, color=color, lw=0.4, alpha=alpha)


def plot_result(path, vpt_cart, blk, gt, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    draw_mesh(ax, vpt_cart, blk, "tab:blue", 0.9)
    if gt is not None:
        draw_mesh(ax, gt[0], gt[1], "tab:red", 0.35)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------- main
def main():
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
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32

    ck = torch.load(args.ckpt, weights_only=False)
    cfg = ck["cfg"]
    rb = tuple(float(v) for v in ck["r_bounds"])
    zb = tuple(float(v) for v in ck["z_bounds"])
    max_len = int(ck["model"]["pos.weight"].shape[0])
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
        xyz, blocks, name, faces_t = load_sample(obj, max(0, args.idx))
        if faces_t is not None:
            gt_faces_t = faces_t
            gt = (xyz, faces_t.tolist())
    elif args.idx >= 0:
        src = torch.load(args.src, weights_only=False)
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
    seq = generate(model, pc, fc, core.start_token, core.stop_token,
                   core.sep_token, args.max_tokens, args.temperature,
                   args.top_k, dev, dtype)
    stopped = seq[-1] == core.stop_token
    print(f"generated tokens={len(seq)} (stop={'ja' if stopped else 'NEIN (cap)'}) "
          f"rows={seq.count(core.sep_token)}")

    (vpt, blk), trim = detokenize_safe(seq, tok, core.stop_token)
    if trim:
        print(f"WARN: Generation instabil, getrimmt: {trim}")
    print(f"reconstructed: verts={vpt.shape[0]} blocks={blk.shape[0]}")

    v_np = vpt.numpy()
    vcart = np.stack([v_np[:, 0] * np.cos(v_np[:, 1]),
                      v_np[:, 0] * np.sin(v_np[:, 1]),
                      v_np[:, 2]], axis=-1)
    write_vtk(args.out, vcart, blk.tolist())
    print(f"saved {args.out}")
    if args.plot:
        title = "generated (blau)" + (" vs source (rot)" if gt else "")
        plot_result(args.plot, vcart, blk.tolist(), gt, title)
        print(f"saved {args.plot}")


if __name__ == "__main__":
    main()
