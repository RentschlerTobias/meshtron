"""
pointer_head_prototype.py

Machbarkeitsnachweis fuer den POINTER-KOPF der Stufe 2 (Faces als Zeiger),
siehe docs/ho_quad_transformer/05_face_block_generator.md und die Erklaerung in
prototype_twostage.py (Stufe 1 = eindeutige Vertices, Stufe 2 = Faces als 4 Zeiger).

Problem, das der Pointer-Kopf loest
------------------------------------
Ein fester Vocab-Slot pro Vertex-Index (Range [off_idx, off_idx+Vmax]) hat drei
Nachteile: (a) Vmax hart gedeckelt, (b) Index-Klasse traegt keine Geometrie, das
Modell weiss nicht *wo* Vertex i liegt, (c) kann auf nicht-existierenden Index zeigen
-> ungueltig.

Der Pointer-Kopf (PolyGen / Pointer-Network, Vinyals 2015) ersetzt das feste Index-
Vocab: er berechnet ein Dot-Product zwischen dem Decoder-State und den Hidden-States
der bereits generierten Vertices und macht softmax NUR ueber die tatsaechlich
vorhandenen M Vertices. Folgen:
  - Gueltig per Konstruktion: jeder Zeiger trifft einen existierenden Vertex.
  - Geometrie-bewusst: der Score nutzt das Vertex-Embedding (Position steckt drin).
  - Groessen-/Permutations-invariant: kein Vmax, generalisiert auf mehr Verts.

Aufbau (isoliert Stufe 2)
-------------------------
Gegeben die Stufe-1-Vertices (r, sin theta, cos theta), generiere die Face-Zeiger
autoregressiv. Das ist genau die Stufe-2-Konditionierung im Vollmodell (dort werden
die Vertices zuvor aus der Punktwolke erzeugt; hier als gegeben angenommen).

  vert_feats [M,3] --Encoder-->  H [M,d]                (Vertex-States, "die Ziele")
  Zeiger-Sequenz  --Decoder(kausal, cross-attn zu H)--> q_t [d]
  pointer_logits_t = H @ q_t   [M]   ->  softmax ueber M  ->  Zeiger

Loss = Cross-Entropy(pointer_logits, ziel_index). Batch=1 (variable M,F ohne Padding).

Der Test overfittet auf n Meshes und misst:
  - Teacher-Forcing Pointer-Accuracy (argmax == Ziel).
  - Free-Run exakte Face-Uebereinstimmung (autoregressiv vs GT).
  - Gueltigkeit: Anteil Quads mit 4 verschiedenen Vertices (Zeiger sind per
    Konstruktion immer gueltige Indizes; distinct ist die einzige echte Frage).
"""

import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prototype_twostage import TwoStageTokenizer


# ------------------------------------------------------------------
# Daten: aus dem Zwei-Stufen-Tokenizer die (Vertex-Features, Zeiger-Ziele) ziehen
# ------------------------------------------------------------------
def build_examples(data, tok, n_max=None):
    """Pro Mesh: vert_feats [M,3] (r, sin th, cos th, in SORTIERTER new-Reihenfolge)
    und ptr_targets [4F] (Face-Zeiger auf new-Indizes, Face-Traversal-Reihenfolge)."""
    n = len(data) if n_max is None else min(n_max, len(data))
    examples = []
    for idx in range(n):
        d = data[idx]
        _, meta = tok.tokenize(d)
        order = meta['order']                      # new_pos -> old_idx
        vp = d['vertices_polar'].numpy()
        r = vp[order, 0]
        th = vp[order, 1]
        vert_feats = np.stack([r, np.sin(th), np.cos(th)], axis=1).astype(np.float32)
        faces_new = meta['faces_new']              # [4,F] (new-Indizes)
        ptr = faces_new.T.reshape(-1).astype(np.int64)   # [4F] Face-Traversal
        examples.append((torch.from_numpy(vert_feats),
                         torch.from_numpy(ptr),
                         faces_new.T.copy()))       # [F,4] fuers Auswerten
    return examples


# ------------------------------------------------------------------
# Modell: Pointer-Face-Netz
# ------------------------------------------------------------------
class PointerFaceModel(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_enc=4, n_dec=4,
                 d_ff_mult=4, dropout=0.0, max_ptr=4096):
        super().__init__()
        self.d_model = d_model
        self.vert_proj = nn.Linear(3, d_model)     # (r, sin, cos) -> d
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff_mult * d_model, dropout,
            activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_enc)
        dec_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, d_ff_mult * d_model, dropout,
            activation='gelu', batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, n_dec)
        # Start-Query (vor dem ersten Zeiger) + lernbare Decoder-Positionen
        self.start = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, max_ptr, d_model) * 0.02)
        # Bilinearer Pointer: q' = q_proj(dec_out); score = q' . H
        self.q_proj = nn.Linear(d_model, d_model)

    def encode(self, vert_feats, vert_pad=None):
        """vert_feats [B,M,3] -> H [B,M,d]  (bidirektional; Vertices sind bekannt).
        vert_pad [B,M] bool (True=Padding, wird ignoriert)."""
        return self.encoder(self.vert_proj(vert_feats), src_key_padding_mask=vert_pad)

    def _decoder_states(self, H, dec_in, tgt_pad=None, mem_pad=None):
        """dec_in [B,L,d] -> Decoder-States [B,L,d] (kausal + cross-attn zu H).
        tgt_pad [B,L] / mem_pad [B,M] bool (True=Padding)."""
        L = dec_in.size(1)
        causal = torch.triu(torch.full((L, L), float('-inf'), device=dec_in.device),
                            diagonal=1)
        dec_in = dec_in + self.pos[:, :L]
        return self.decoder(dec_in, H, tgt_mask=causal,
                            tgt_key_padding_mask=tgt_pad,
                            memory_key_padding_mask=mem_pad)

    def forward(self, vert_feats, ptr_targets, vert_pad=None, tgt_pad=None):
        """Teacher-Forcing. Returns pointer_logits [B,L,M].
        vert_pad [B,M] / tgt_pad [B,L] bool (True=Padding). Pad-Vertex-Spalten der
        Logits werden auf -inf gesetzt -> Zeiger kann nie auf Padding zeigen."""
        H = self.encode(vert_feats, vert_pad)      # [B,M,d]
        B, M, d = H.shape
        L = ptr_targets.size(1)
        # Decoder-Eingang: Start-Query + H an den ZUVOR gewaehlten Zielen (shift right).
        # Pad-Ziele auf 0 klemmen (Gather braucht gueltigen Index; Loss ignoriert sie).
        safe_tgt = ptr_targets.clamp(min=0)
        gathered = torch.gather(
            H, 1, safe_tgt.unsqueeze(-1).expand(-1, -1, d))      # [B,L,d]
        dec_in = torch.cat([self.start.expand(B, -1, -1), gathered[:, :-1]], dim=1)
        dec_out = self._decoder_states(H, dec_in, tgt_pad, vert_pad)  # [B,L,d]
        q = self.q_proj(dec_out)                   # [B,L,d]
        logits = torch.bmm(q, H.transpose(1, 2)) / (d ** 0.5)  # [B,L,M] skal. Pointer
        if vert_pad is not None:
            logits = logits.masked_fill(vert_pad.unsqueeze(1), float('-inf'))
        return logits

    @torch.no_grad()
    def generate(self, vert_feats, n_ptr):
        """Free-Run: autoregressiv n_ptr Zeiger erzeugen. Returns idx-Liste [n_ptr]."""
        H = self.encode(vert_feats)                # [1,M,d]
        d = H.size(-1)
        chosen = []
        for _ in range(n_ptr):
            if chosen:
                gathered = H[:, chosen, :]         # [1,t,d]
                dec_in = torch.cat([self.start, gathered], dim=1)
            else:
                dec_in = self.start
            dec_out = self._decoder_states(H, dec_in)     # [1,t+1,d]
            q = self.q_proj(dec_out[:, -1:])              # [1,1,d]
            logits = torch.bmm(q, H.transpose(1, 2))[0, 0] / (d ** 0.5)  # [M]
            chosen.append(int(logits.argmax()))
        return chosen


# ------------------------------------------------------------------
# Overfit-Test
# ------------------------------------------------------------------
def evaluate(model, examples, device):
    model.eval()
    tf_correct = tf_total = 0
    face_exact = face_total = 0
    quads_distinct = quads_total = 0
    with torch.no_grad():
        for vert_feats, ptr, faces in examples:
            vf = vert_feats.unsqueeze(0).to(device)
            tgt = ptr.unsqueeze(0).to(device)
            # Teacher-Forcing-Accuracy
            logits = model(vf, tgt)
            pred = logits.argmax(-1)[0].cpu()
            tf_correct += int((pred == ptr).sum()); tf_total += ptr.numel()
            # Free-Run exakte Faces
            gen = model.generate(vf, ptr.numel())
            gen = np.array(gen).reshape(-1, 4)     # [F,4]
            for gi, fi in zip(gen, faces):
                face_total += 1
                if np.array_equal(gi, fi):
                    face_exact += 1
                quads_total += 1
                if len(set(gi.tolist())) == 4:
                    quads_distinct += 1
    return (tf_correct / max(tf_total, 1),
            face_exact / max(face_total, 1),
            quads_distinct / max(quads_total, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='domain_data.pt')
    ap.add_argument('--n', type=int, default=30, help='Anzahl Meshes (overfit)')
    ap.add_argument('--epochs', type=int, default=400)
    ap.add_argument('--d-model', type=int, default=256)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--warmup-frac', type=float, default=0.05,
                    help='Anteil der Schritte fuer linearen LR-Warmup')
    ap.add_argument('--clip', type=float, default=1.0, help='Grad-Norm-Clip (0=aus)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Lade {args.data} ...")
    data = torch.load(args.data, weights_only=False)
    max_v = max(d['vertices_polar'].shape[0] for d in data)
    tok = TwoStageTokenizer(max_vertices=max_v + 16)   # nur fuer meta (order/faces_new)

    examples = build_examples(data, tok, n_max=args.n)
    Ms = [e[0].shape[0] for e in examples]
    Ls = [e[1].numel() for e in examples]
    print(f"{len(examples)} Meshes  |  M (Verts) {min(Ms)}..{max(Ms)}  "
          f"|  Zeiger/Mesh {min(Ls)}..{max(Ls)}")

    model = PointerFaceModel(d_model=args.d_model).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"PointerFaceModel: d_model={args.d_model}  Params {n_par/1e6:.2f}M  device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # LR-Schedule: linearer Warmup -> Cosine-Decay (glaettet die batch=1-Spruenge)
    total_steps = args.epochs * len(examples)
    warmup = max(1, int(args.warmup_frac * total_steps))

    def lr_scale(step):
        if step < warmup:
            return (step + 1) / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)

    best = (0.0, 0.0, 0.0)
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(len(examples))
        tot = 0.0
        for i in perm:
            vert_feats, ptr, _ = examples[i]
            vf = vert_feats.unsqueeze(0).to(device)
            tgt = ptr.unsqueeze(0).to(device)
            logits = model(vf, tgt)                 # [1,L,M]
            loss = F.cross_entropy(logits[0], tgt[0])
            opt.zero_grad(); loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step(); sched.step()
            tot += loss.item()
        if ep % max(1, args.epochs // 15) == 0 or ep == 1:
            tf_acc, face_acc, distinct = evaluate(model, examples, device)
            if face_acc > best[1]:
                best = (tf_acc, face_acc, distinct)
            print(f"ep {ep:4d}  loss {tot/len(examples):.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                  f"TF-ptr-acc {tf_acc:.3f}  free-run face-exact {face_acc:.3f}  "
                  f"quads-distinct {distinct:.3f}")
    print(f"\nBEST free-run: TF-acc {best[0]:.3f}  face-exact {best[1]:.3f}  "
          f"quads-distinct {best[2]:.3f}")

    print("\nHinweis: Zeiger sind PER KONSTRUKTION immer gueltige Vertex-Indizes "
          "(softmax nur ueber M). 'quads-distinct' misst, ob die 4 Zeiger eines "
          "Quads verschieden sind.")


if __name__ == '__main__':
    main()
