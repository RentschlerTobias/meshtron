"""Hexa-row tokenizer: 3D hexa blocks als Row-Ketten emitieren.

Design (mit User festgelegt):
  - Pro Block werden die 8 Verts so geordnet, dass Entry-Face (min-y) = ersten 4
    und Exit-Face (max-y) = letzten 4 Emission-Positionen sind; Rows laufen
    aufsteigend in y, Layer werden von unten nach oben (z) aufgebaut.
  - Block-Stack: lex nach Zentroid (z, y, x); Row-Start = erster unbesuchter
    Block im Stack; walk +y ueber die max-y-Face (Exit-Face) zum Block, der
    diese Face teilt; Ende der Row, wenn die Exit-Face keinen unbesuchten
    Nachbar-Block mehr hat.
  - Emission row-encoded: Row-Start-Block emittiert Entry-Ring (4 Verts,
    kanonischer Ring) + Exit-Ring (4 Verts, axial zum Entry-Ring gepaart),
    jedes Folgende nur seine Exit-Ring-Tokens (Entry ist exakt die Exit-Face
    des Vorgaengers). EOR (sep_token) nach jeder Row, stop_token am Ende.
  - Winding/Pairing: Entry-Ring startet am lex-min-Vertex (x,z), Ringrichtung
    so, dass die Newell-Normale in den Block zeigt (Richtung Exit). Exit-Ring
    startet am axialen Pair des Entry-Ring-Starts, Richtung +1 im
    Min-Umfang-Ring (Ring zyklisch, Konvention fixiert das Training).

Token-Stream: [start] block1(8V=32tok) [block2(4V=16tok)]... [sep(EOR)]
               [block...] [sep] ... [stop]
Quantisierung identisch PolytronTokenizer dim=3: 4 Tokens/Vert
(r, sin, cos, z); Offsets off_r=0, off_ts=Qr, off_tc=Qr+Qa, off_idx=Qr+2Qa.
"""
from __future__ import annotations

import itertools

import numpy as np
import torch

from polytron_tokenizer import PolytronTokenizer


# ---------------------------------------------------------------- Ring-Logik
_HEX_FACE_Q = ([0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
               [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7])


def _faces_of(b):
    v = [int(x) for x in b]
    return [frozenset(v[i] for i in q) for q in _HEX_FACE_Q]


def _ring_min_perimeter(pts):
    """Positionen (Index in pts) eines 4-Punkt-Rings mit minimalem Umfang."""
    best, best_order = None, None
    for perm in itertools.permutations(range(4)):
        s = sum(np.linalg.norm(pts[perm[(i + 1) % 4]] - pts[perm[i]]) for i in range(4))
        if best is None or s < best - 1e-12:
            best, best_order = s, list(perm)
    return best_order


def _rotate_orient_ring(order, pts, target):
    """Ringordnung (Positionen) rotieren: Start am lex-min-Vertex (x,z), Richtung
    so, dass Newell-Normale grob in `target` zeigt."""
    start = min(order, key=lambda p: (round(float(pts[p][0]), 9), round(float(pts[p][2]), 9)))
    k = order.index(start)
    rot = order[k:] + order[:k]
    n = np.zeros(3)
    for i in range(4):
        p, q = pts[rot[i]], pts[rot[(i + 1) % 4]]
        n += np.cross(p, q)
    if float(np.dot(n, np.asarray(target))) < 0:
        rot = [rot[0]] + rot[:0:-1]
    return rot


def _axial_pairing(entry_ids, exit_ids, edge_index):
    """entry_id -> exit_id via Achsenkante (globaler Wireframe)."""
    nbr = {}
    for u, v in edge_index.T.tolist():
        nbr.setdefault(int(u), set()).add(int(v))
        nbr.setdefault(int(v), set()).add(int(u))
    exit_set = set(int(x) for x in exit_ids)
    pairs = {}
    for e in entry_ids:
        cands = (nbr.get(int(e), set()) & exit_set) - set(pairs.values())
        if len(cands) == 1:
            pairs[int(e)] = int(next(iter(cands)))
    return pairs


# ---------------------------------------------------------------- Row-Plan
def build_row_plan(blks, Vcart, edges=None, start_rule='min_theta', z_split=None):
    """Deterministischer Hexa-Row-Plan (azimutale Reihen, CCW).

    blks: list of blocks (je 8 global vertex ids, VTK order)
    Vcart: [M,3] cartesische Koordinaten
    edges: [2,E] Wireframe-Kanten (fuer Exit-Ring-Pairing; None -> Fallback)
    start_rule: 'min_theta' | 'max_theta' — Tie-Break der Startwahl im Quadrant
      theta in [-90deg, 0) unter den mit max r.
    z_split: z-Sprung-Schwelle; Reihe bricht, wenn |dz| zum naechsten Block sie
      uebersteigt (None -> 0.4 * mediane Block-z-Ausdehnung).
    Returns (rows, emit): rows = Listen von Block-Indizes;
      emit[b] = 8 global vertex ids [entry_ring(4), exit_ring(4)].
    """
    F = len(blks)
    V = Vcart if isinstance(Vcart, torch.Tensor) else torch.as_tensor(Vcart)
    edges = torch.zeros(2, 0, dtype=torch.long) if edges is None else edges

    fc = [_faces_of(b) for b in blks]
    shared = {}
    adj = [set() for _ in range(F)]
    for a, b in itertools.combinations(range(F), 2):
        c = set(fc[a]) & set(fc[b])
        if c:
            shared[(a, b)] = frozenset(c)  # saemtliche geteilten Faces
            adj[a].add(b); adj[b].add(a)
    cent = [torch.stack([V[int(v)] for v in b]).mean(0).tolist() for b in blks]

    def _theta(p):
        return float(np.arctan2(p[1], p[0]))

    def _dtheta(th_face, th_blk):
        d = th_face - th_blk
        return float((d + np.pi) % (2.0 * np.pi) - np.pi)

    cent_th = [_theta(c) for c in cent]
    cent_r = [float(np.hypot(c[0], c[1])) for c in cent]
    fc_th = [[_theta(np.mean([[V[int(v), 0].item(), V[int(v), 1].item()] for v in f], axis=0))
              for f in fb] for fb in fc]

    def pick(b, sgn, disjoint_from=()):
        """Face-Slot mit extremalem Azimut-Delta: sgn=+1 Exit (+theta), sgn=-1
        Entry (-theta).  Azimutale Rows (Walk um die z-Achse).
        disjoint_from: frozensets; Kandidaten muessen VERTEXX-DISJOINT dazu sein
        (Entry/Exit sind gegenueberliegende Faces des Hexaeders)."""
        best, bi = None, None
        for i, f in enumerate(fc[b]):
            if any(f & g for g in disjoint_from):
                continue
            m = sgn * _dtheta(fc_th[b][i], cent_th[b])
            if best is None or m > best:
                best, bi = m, i
        if bi is None:
            raise AssertionError(f"pick({b},{sgn}): kein disjunktes Face zu {disjoint_from}")
        return bi

    ord_z = sorted(range(F), key=lambda i: min(V[int(v), 2].item() for v in blks[i]))
    zs_all = [V[int(v), 2].item() for b in blks for v in b]
    blk_minz = [min(V[int(v), 2].item() for v in b) for b in blks]
    z_gap_layer = max(1e-3, 0.5 * (0.4 * float(np.median(
        [abs(max([V[int(v), 2].item() for v in b]) - min([V[int(v), 2].item() for v in b]))
         for b in blks])) if blks else 0.3))
    layers, cur_l = [], []
    for i in ord_z:
        if cur_l and blk_minz[i] - blk_minz[cur_l[-1]] > z_gap_layer:
            layers.append(cur_l)
            cur_l = []
        cur_l.append(i)
    layers.append(cur_l)

    visited = set()
    rows = []    # Block-Index-Listen (Walk-Reihenfolge)
    walk_faces = {}  # b -> (entry_ids, exit_face-frozenset); Emit nutzt exakt diese Faces
    blk_zspan = []
    for i, c in enumerate(cent):
        zs = [V[int(v), 2].item() for v in blks[i]]
        blk_zspan.append(abs(max(zs) - min(zs)))
    if z_split is None:
        z_split = 0.4 * float(np.median(blk_zspan)) if blk_zspan else 0.3

    def _lex_key(b):
        """Block-Key = Meshpunkte des Blocks nach yzx sortiert, kleinster Punkt (y prioritaet)."""
        pts = [(round(V[int(v), 1].item(), 6), round(V[int(v), 2].item(), 6),
                round(V[int(v), 0].item(), 6)) for v in blks[b]]
        return min(pts)

    def _choose_start(todo):
        return min(todo, key=_lex_key)

    def _walk_chain(start, todo, sgn):
        """Kette ab start in Richtung sgn (*+1: +delta-rheta). Liefert (row, faces-dict)."""
        chain_faces = {}
        oi = pick(start, -1 if sgn > 0 else +1)
        cur, cur_entry_face, cur_entry_ids = start, fc[start][oi], [int(v) for v in fc[start][oi]]
        row = [start]
        while True:
            ei = pick(cur, +sgn, disjoint_from=(cur_entry_face,))
            exit_f = fc[cur][ei]
            chain_faces[cur] = (cur_entry_ids, exit_f)
            nxt = []
            for b in adj[cur]:
                if b in todo and b not in chain_faces \
                        and exit_f in shared.get((min(cur, b), max(cur, b)), set()):
                    nxt.append(b)
            if not nxt:
                break
            cur_entry_face = exit_f
            cur_entry_ids = [int(v) for v in exit_f]
            # vorwaerts = sgn*delta-rheta (Reihen aufsteigend orientierbar)
            cand = min(nxt, key=lambda b: sgn * (_dtheta(cent_th[b], cent_th[cur]) % (2.0 * np.pi)))
            if abs(cent[cand][2] - cent[cur][2]) > z_split:
                break  # z-Sprung: Reihe endet hier (Reihen bleiben z-Naeh)
            cur = cand
            row.append(cur)
        return row, chain_faces

    for layer in layers:
        while True:
            todo = set(b for b in layer if b not in visited)
            if not todo:
                break
            start = _choose_start(sorted(todo, key=_lex_key))
            fwd_row, fwd_faces = _walk_chain(start, todo, +1)
            rest = todo - set(fwd_row)
            bk_row, bk_faces = _walk_chain(start, rest - {start}, -1)
            window = bk_row[1:][::-1] + fwd_row
            if len(window) > 1:
                row, chain_faces = _walk_chain(window[0], set(window), +1)
            else:
                row, chain_faces = window, fwd_faces
            rows.append(row)
            visited.update(row)
            walk_faces.update(chain_faces)

    # ---- Emission-Reihenfolge ( nutzt die beim Walk bestimmten Faces ) ----
    emit = [None] * F
    for row in rows:
        for b in row:
            entry_ids, exit_face = walk_faces[b]
            exit_ids_list = [int(v) for v in exit_face]
            if set(entry_ids) & set(exit_ids_list):
                raise AssertionError(
                    f"entry/exit faces ueberlappen: block {b} row={row} "
                    f"entry_ids={entry_ids} exit={exit_ids_list}")
            assert not (set(entry_ids) & set(exit_ids_list)), "entry/exit faces ueberlappen"
            e_pts = np.stack([V[i].tolist() for i in entry_ids])
            x_pts = np.stack([V[i].tolist() for i in exit_ids_list])

            # Entry-Ring: min-Umfang, lex-min Start, Normale in den Block
            perm_e = _ring_min_perimeter(e_pts)
            rot_e = _rotate_orient_ring(perm_e, e_pts, x_pts.mean(0) - e_pts.mean(0))
            entry_seq = [entry_ids[p] for p in rot_e]

            # Exit-Ring: min-Umfang; Anchor = axiales Pair des Entry-Starts
            exit_seq = None
            if edges.numel() > 0:
                pairs = _axial_pairing(entry_seq, exit_ids_list, edges)
                if len(pairs) == 4 and pairs[int(entry_seq[0])] in exit_ids_list:
                    anchor = pairs[int(entry_seq[0])]
                    perm_x = _ring_min_perimeter(x_pts)
                    k = perm_x.index(exit_ids_list.index(anchor))
                    exit_seq = [exit_ids_list[perm_x[(k + s) % 4]] for s in range(4)]
            if exit_seq is None:  # Fallback: lex (x,z) Reihenfolge der Exit-Verts
                exit_seq = [exit_ids_list[p] for p in
                            sorted(range(4), key=lambda p: (x_pts[p][0], x_pts[p][2]))]
            emit[b] = list(entry_seq) + list(exit_seq)
            entry_ids = list(exit_ids_list)  # naechster Block erbt Exit-Face als Entry
    return rows, emit


# ---------------------------------------------------------------- Tokenizer
class HexaRowTokenizer:
    def __init__(self, quantization_r=512, quantization_a=256, r_bounds=(0.0, 1.0),
                 z_bounds=(0.0, 1.0)):
        self.core = PolytronTokenizer(quantization_r=quantization_r,
                                      quantization_a=quantization_a,
                                      max_vertices=2048, repr_mode='cubic_bezier',
                                      r_bounds=r_bounds, dim=3, corners_per_block=8,
                                      z_bounds=z_bounds)
        self.Qr = self.core.Qr
        self.Qa = self.core.Qa

    def quant_vertex_tokens(self, vid, vp):
        """4 Token (r, sin, cos, z) eines Vertex in Polar-Koordinaten (Demo-Hilfer)."""
        c = self.core
        r = float(vp[vid, 0]); th = float(vp[vid, 1]); z = float(vp[vid, 2])
        ts, tc = c._q_angle(th)
        return [int(c._q_scalar(r, c.R_MIN, c.R_MAX) + c.off_r),
                int(ts + c.off_ts), int(tc + c.off_tc),
                int(c._q_scalar(z, c.Z_MIN, c.Z_MAX) + c.off_r)]

    # -- Tokenize -----------------------------------------------------------
    def tokenize(self, mesh_data, emit_override=None):
        """mesh_data: {vertices_polar [M,3], faces [8,F], vertices_cartesian [M,3],
        edge_index [2,E]} -> tokens list."""
        vp = mesh_data['vertices_polar']
        blks = mesh_data['faces'].T.tolist()
        Vc = mesh_data.get('vertices_cartesian')
        edges = mesh_data.get('edge_index')
        if emit_override is None:
            rows, emit = build_row_plan(blks, Vc, edges=edges)
        else:
            rows, emit = emit_override, None
        toks = [self.core.start_token]
        for row in rows:
            for bi, b in enumerate(row):
                for vid in (emit[b] if bi == 0 else emit[b][4:8]):
                    r = float(vp[vid, 0]); th = float(vp[vid, 1])
                    toks.append(self.core._q_scalar(r, self.core.R_MIN, self.core.R_MAX) + self.core.off_r)
                    ts, tc = self.core._q_angle(th)
                    toks += [ts + self.core.off_ts, tc + self.core.off_tc]
                    z = float(vp[vid, 2])
                    toks.append(self.core._q_scalar(z, self.core.Z_MIN, self.core.Z_MAX) + self.core.off_r)
            toks.append(self.core.sep_token)
        toks.append(self.core.stop_token)
        return toks

    # -- Detokenize -----------------------------------------------------------
    def detokenize(self, toks):
        """Token list -> (vertices_polar [M,3], blocks [F,8]) mit Token-Dedup."""
        off_r, off_ts, off_tc = self.core.off_r, self.core.off_ts, self.core.off_tc
        rmin, rmax = self.core.R_MIN, self.core.R_MAX
        i_end = toks.index(self.core.stop_token)
        toks = [int(t) for t in toks[1:i_end]]  # drop start + stop
        rows_t = []
        cur = []
        for t in toks:
            if t == self.core.sep_token:
                rows_t.append(cur); cur = []
            else:
                cur.append(t)
        if cur:
            rows_t.append(cur)

        def dqv(group4):
            r = self.core._dq_scalar(group4[0] - off_r, rmin, rmax)
            a = self.core._dq_angle(group4[1] - off_ts, group4[2] - off_tc)
            z = self.core._dq_scalar(group4[3] - off_r, self.core.Z_MIN, self.core.Z_MAX)
            return (r, a, z)

        verts = []    # (r,th,z) tuples
        vmap = {}     # token-tuple -> vertex index (Dedup)
        blocks = []
        for row in rows_t:
            j = 0
            pending = None  # Exit-Token-Quadrupel des Vorgaengers
            while j < len(row):
                need = 32 if pending is None else 16
                group = row[j:j + need]
                assert len(group) == need, \
                    f"zu wenige Tokens fuer Block-Emission: {len(group)} < {need}"
                if pending is None:
                    ids = []
                    for k in range(8):
                        tk = tuple(group[k * 4:(k + 1) * 4])
                        if tk in vmap:
                            ids.append(vmap[tk])
                        else:
                            vmap[tk] = len(verts)
                            verts.append(dqv(tk))
                            ids.append(len(verts) - 1)
                    blocks.append(ids)
                    pending = tuple(group[16:32])
                else:
                    # pending = 4 Token-Quadrupel der Exit-Ring-Flaeche des Vorgaengers;
                    # deren Vertex-Ids via vmap (bereits verausgabt) holen.
                    ids = [vmap[tuple(pending[k * 4:(k + 1) * 4])] for k in range(4)]
                    for k in range(4):
                        tk = tuple(group[k * 4:(k + 1) * 4])
                        if tk in vmap:
                            ids.append(vmap[tk])
                        else:
                            vmap[tk] = len(verts)
                            verts.append(dqv(tk))
                            ids.append(len(verts) - 1)
                    blocks.append(ids)
                    pending = tuple(group)
                j += need
        vpt = torch.tensor(verts, dtype=torch.float32)
        blk = torch.tensor(blocks, dtype=torch.long)
        return vpt, blk


# ------------------------------------------------------------- Roundtrip-Check
def roundtrip(data, tok, coord_tol=0.02):
    ok, maxerr, msgs = True, 0.0, []
    for si, mesh in enumerate(data):
        toks = tok.tokenize(mesh)
        vpt, blk = tok.detokenize(toks)
        ob = mesh['faces'].T.tolist()
        orig = mesh['vertices_polar']
        bm_o = [np.stack([orig[int(v)].numpy() for v in b]).mean(0) for b in ob]
        bm_n = [np.stack([vpt[int(i)].numpy() if torch.is_tensor(vpt[i]) else np.asarray(vpt[i])
                          for i in b]).mean(0) for b in blk.tolist()]
        bm_o_s = sorted(bm_o, key=lambda p: tuple(p))
        matched = np.zeros(len(bm_n), dtype=bool)
        for m in bm_n:
            dist = [np.linalg.norm(o - m) for o in bm_o_s]
            k = int(np.argmin(dist))
            if dist[k] < 0.01:
                matched[k] = True
        if not matched.all():
            ok = False
            msgs.append(f'sample {si}: block mismatch, unmatched={int((~matched).sum())}/{len(matched)}')
        if len(vpt):
            d = (orig.unsqueeze(0) - vpt.unsqueeze(1)).norm(dim=-1)
            nn = d.min(dim=-1).values.max().item()
            maxerr = max(maxerr, nn)
            if nn > coord_tol:
                ok = False
                msgs.append(f'sample {si}: coord err {nn:.4f} > {coord_tol}')
    return ok, maxerr, msgs


def main():
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/polytron_data_3d_smoke.pt'
    data = torch.load(path, weights_only=False)
    rpol = torch.cat([d['vertices_polar'][:, 0] for d in data])
    zpol = torch.cat([d['vertices_polar'][:, 2] for d in data])
    rp = (rpol.max() - rpol.min()) * 0.02
    zp = (zpol.max() - zpol.min()) * 0.02
    rb = (float(rpol.min()) - rp, float(rpol.max()) + rp)
    zb = (float(zpol.min()) - zp, float(zpol.max()) + zp)
    tok = HexaRowTokenizer(r_bounds=rb, z_bounds=zb)
    for si, mesh in enumerate(data):
        toks = tok.tokenize(mesh)
        nb = mesh['faces'].shape[1]
        print(f'sample {si}: tokens={len(toks)} blocks={nb} est_full={32 + 16 * (nb - 1)}')
    ok, maxerr, msgs = roundtrip(data, tok)
    print(f'roundtrip: ok={ok} maxerr={maxerr:.4f}')
    for m in msgs:
        print(' ', m)


if __name__ == '__main__':
    main()
