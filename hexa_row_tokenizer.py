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
  - Winding/Pairing: Entry- und Exit-Ring sind Ringe GEGENUEBERLIEGENDER
    Faces. Der Entry-Ring wird lex-min/Newell-nach-Exit orientiert, der
    Exit-Ring folgt dem axialen Pairing (exit_seq[i] = axialer Partner von
    entry_seq[i]) -> emit[b] ist per Konstruktion eine gueltige VTK-Relabelung.
    Folgebloecke uebernehmen den Exit-Ring des Vorgaengers EXAKT als Entry
    (Rotation ODER Reversal); ohne paarungskonsistente Fortsetzung bricht die
    Row und der Block startet als frischer Head (volle 8 Verts, SEP davor).

Token-Stream: [start] block1(8V=32tok) [block2(4V=16tok)]... [sep(EOR)]
               [block...] [sep] ... [stop]
Quantisierung identisch PolytronTokenizer dim=3: 4 Tokens/Vert
(r, sin, cos, z) oder carts-Modus 3 Tokens/Vert (x,y,z); Offsets off_r=0,
off_ts=Qr, off_tc=Qr+Qa, off_idx=Qr+2Qa (cart: x,y,z alle ueber off_r).
"""
from __future__ import annotations

import itertools

import numpy as np
import torch


class DegenerateBlockError(ValueError):
    """Block-Geometrie mit <4 eindeutigen Vert-Mengen auf einem Hexa-Face (weld-Fusion)."""
    pass

from polytron_tokenizer import PolytronTokenizer
from mesh_validation import hex_min_jacobian, hex_signed_volumes


# ---------------------------------------------------------------- Ring-Logik
_HEX_FACE_Q = ([0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
               [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7])


def _faces_of(b):
    v = [int(x) for x in b]
    return [frozenset(v[i] for i in q) for q in _HEX_FACE_Q]


def _ring_min_perimeter(pts):
    """Positionen (Index in pts) eines 4-Punkt-Rings mit minimalem Umfang."""
    # 6 Paar-Distanzen je einmal (np.linalg.norm bei 24 Perms x 4 Kanten = 173k Calls)
    d = {(a, b): np.linalg.norm(pts[b] - pts[a])
         for i, a in enumerate(range(4)) for b in range(i + 1, 4)}
    dist = lambda a, b: d[(a, b) if a < b else (b, a)]
    best, best_order = None, None
    for perm in itertools.permutations(range(4)):
        s = sum(dist(perm[i], perm[(i + 1) % 4]) for i in range(4))
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


def _is_rotation(a, b):
    """True wenn b eine zyklische Rotation von a ist (gleiche Ringordnung)."""
    return len(a) == 4 and len(b) == 4 and any(a == b[k:] + b[:k] for k in range(4))


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


# VTK-6 Hexaeder-Kanten als Positionspaare in der Block-Eckreihenfolge.
_HEX_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7))


def _axial_pairing_local(entry_ids, exit_ids, blk):
    """entry_id -> exit_id via Block-lokale VTK-Kanten (ohne Wireframe).

    blk: 8 globale Vertex-IDs in VTK-Reihenfolge. Fuer strukturell valide
    Hexa-Blocks exakt (gegenueberliegende Faces sind kantenseitig nur ueber
    Achsenkanten verbunden); leeres Dict bei unklarer Struktur.
    """
    exit_set = set(int(x) for x in exit_ids)
    nbrs = {int(v): set() for v in blk}
    for a, b in _HEX_EDGES:
        nbrs[int(blk[a])].add(int(blk[b]))
        nbrs[int(blk[b])].add(int(blk[a]))
    pairs = {}
    for e in entry_ids:
        cands = (nbrs.get(int(e), set()) & exit_set) - set(pairs.values())
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
    for bi, fb in enumerate(fc):
        if any(len(f) < 4 for f in fb):
            raise DegenerateBlockError(f"block {bi} hat Face mit <4 eindeutigen Verts (geweldete Zwillings-Verts)")
    shared_acc = {}
    adj = [set() for _ in range(F)]
    by_face = {}
    for b in range(F):
        for f in fc[b]:
            by_face.setdefault(f, []).append(b)
    for f, owners in by_face.items():
        for i in range(len(owners)):
            for j in range(i + 1, len(owners)):
                a, b = owners[i], owners[j]
                key = (a, b) if a < b else (b, a)
                sl = shared_acc.get(key)
                if sl is None:
                    shared_acc[key] = {f}
                else:
                    sl.add(f)
                adj[a].add(b); adj[b].add(a)
    shared = {k: frozenset(v) for k, v in shared_acc.items()}
    cent = [torch.stack([V[int(v)] for v in b]).mean(0).tolist() for b in blks]

    def _theta(p):
        return float(np.arctan2(p[1], p[0]))

    def _dtheta(th_face, th_blk):
        d = th_face - th_blk
        return float((d + np.pi) % (2.0 * np.pi) - np.pi)

    cent_th = [_theta(c) for c in cent]
    cent_r = [float(np.hypot(c[0], c[1])) for c in cent]
    Vn = V.numpy()
    xy = [(float(Vn[i, 0]), float(Vn[i, 1])) for i in range(V.shape[0])]
    fc_th = [[_theta(np.mean([[xy[v][0], xy[v][1]] for v in f], axis=0))
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

    # Lex-Keys einmal pro Block/Vertex vorberechnen (sonst ~O(N^2) Tensor.item())
    vkey = [(round(float(V[i, 1]), 6), round(float(V[i, 2]), 6), round(float(V[i, 0]), 6))
            for i in range(V.shape[0])]
    lex_keys = {b: min(vkey[int(v)] for v in blks[b]) for b in range(len(blks))}

    def _lex_key(b):
        """Block-Key = Meshpunkte des Blocks nach yzx sortiert, kleinster Punkt (y prioritaet)."""
        return lex_keys[b]

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
    # Entry-Ring lex-min/Newell nach Exit orientiert, Exit-Ring als axiales
    # Pairing-Follow des Entry-Rings -> emit[b] ist eine gueltige VTK-Relabelung.
    # Folgebloecke erben den Exit-Ring des Vorgaengers exakt als Entry-Ring
    # (Rotation ODER Reversal); ohne paarungskonsistente Fortsetzung bricht die
    # Row und der Block startet als frischer Head (volle 8 Verts, SEP davor).
    emit = [None] * F
    out_rows: list[list[int]] = []
    for row in rows:
        cur_row: list[int] = []
        prev_exit: list[int] | None = None
        for b in row:
            entry_ids, exit_face = walk_faces[b]
            exit_ids_list = [int(v) for v in exit_face]
            if set(entry_ids) & set(exit_ids_list):
                raise AssertionError(
                    f"entry/exit faces ueberlappen: block {b} row={row} "
                    f"entry_ids={entry_ids} exit={exit_ids_list}")
            e_pts = np.stack([V[i].tolist() for i in entry_ids])
            x_pts = np.stack([V[i].tolist() for i in exit_ids_list])

            # Referenz-Entry-Ring: min-Umfang, lex-min Start, Normale zu Exit
            perm_e = _ring_min_perimeter(e_pts)
            rot_e = _rotate_orient_ring(perm_e, e_pts, x_pts.mean(0) - e_pts.mean(0))
            entry_ref = [entry_ids[p] for p in rot_e]
            pairs = _axial_pairing_local(entry_ref, exit_ids_list, blks[b])
            if len(pairs) < 4 and edges.numel() > 0:
                pairs = _axial_pairing(entry_ref, exit_ids_list, edges)

            def _follow(seq, pairs=pairs):
                if len(pairs) == 4 and all(int(e) in pairs for e in seq):
                    return [int(pairs[int(e)]) for e in seq]
                return None

            entry_seq = None
            if prev_exit is not None:
                if _is_rotation(entry_ref, prev_exit):
                    entry_seq = list(prev_exit)
                elif _is_rotation(list(reversed(entry_ref)), prev_exit):
                    # Reversal: Ringrichtung dreht mit, Pairing-Follow bleibt
                    # orientierungstreu -> Positivitaet verifizieren, sonst Break.
                    seq = list(prev_exit)
                    ex = _follow(seq)
                    if ex is not None:
                        pts8 = np.stack([V[int(v)].tolist() for v in seq + ex])[None]
                        if (float(hex_signed_volumes(pts8)[0]) > 0.0
                                and float(hex_min_jacobian(pts8)[0]) > -1e-9):
                            entry_seq = seq
            if entry_seq is None:
                if prev_exit is not None and cur_row:  # Break: neue Row
                    out_rows.append(cur_row)
                    cur_row = []
                # Head-Ring verifizieren: min-Umfang kann auf Sattelflaechen
                # Bowties liefern. Erste zyklisch-propere Variante mit
                # vol>0 & minJ>-1e-9 gewinnt; sonst Fallback auf entry_ref.
                target = x_pts.mean(0) - e_pts.mean(0)
                seen_rings = set()
                for rep in (tuple(perm_e), (0, 1, 3, 2), (0, 2, 1, 3)):
                    cand = tuple(_rotate_orient_ring(list(rep), e_pts, target))
                    if cand in seen_rings:
                        continue
                    seen_rings.add(cand)
                    seq_c = [entry_ids[p] for p in cand]
                    ex_c = _follow(seq_c)
                    if ex_c is None:
                        continue
                    pts8 = np.stack([V[int(v)].tolist() for v in seq_c + ex_c])[None]
                    if (float(hex_signed_volumes(pts8)[0]) > 0.0
                            and float(hex_min_jacobian(pts8)[0]) > -1e-9):
                        entry_seq = seq_c
                        break
                if entry_seq is None:
                    entry_seq = entry_ref
            exit_seq = _follow(entry_seq)
            if exit_seq is None:  # Fallback ohne Wireframe: lex (x,z)
                exit_seq = [exit_ids_list[p] for p in
                            sorted(range(4), key=lambda p: (x_pts[p][0], x_pts[p][2]))]
            emit[b] = list(entry_seq) + list(exit_seq)
            prev_exit = emit[b][4:8]
            cur_row.append(b)
        if cur_row:
            out_rows.append(cur_row)
    return out_rows, emit


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

    def quant_vertex_tokens(self, vid, vp, vc=None, coords='polar'):
        """polar: 4 Token (r, sin, cos, z); cart: 3 Token (x,y,z) aus vc [M,3]."""
        c = self.core
        if coords == 'cart':
            assert vc is not None, "cart coords benoetigen vertices_cartesian"
            x, y, z = (float(v) for v in vc[int(vid)])
            rmax = float(c.R_MAX)
            return [int(c._q_scalar(x, -rmax, rmax) + c.off_r),
                    int(c._q_scalar(y, -rmax, rmax) + c.off_r),
                    int(c._q_scalar(z, c.Z_MIN, c.Z_MAX) + c.off_r)]
        r = float(vp[vid, 0]); th = float(vp[vid, 1]); z = float(vp[vid, 2])
        ts, tc = c._q_angle(th)
        return [int(c._q_scalar(r, c.R_MIN, c.R_MAX) + c.off_r),
                int(ts + c.off_ts), int(tc + c.off_tc),
                int(c._q_scalar(z, c.Z_MIN, c.Z_MAX) + c.off_r)]

    # -- Tokenize -----------------------------------------------------------
    def tokenize(self, mesh_data, emit_override=None, granularity='row', eoe=False,
                 coords='polar'):
        """mesh_data: {vertices_polar [M,3], faces [8,F], vertices_cartesian [M,3],
        edge_index [2,E]} -> tokens list.

        granularity:
          'row'   — row-encoded (Dedup: Head 8 Verts, Forts. Exit-Ring), EOR nach Row.
          'block' — wie row, aber EOE (sep2) nach JEDEM Block falls eoe.
          'face'  — Meshtron-artig ohne Dedup: alle 6 Quads je Block vollstaendig
                    (je 16 Tok polar / 12 Tok cart), EOE (sep2) je Quad falls eoe,
                    EOR nach Row.
        eoe: End-of-Element-Token (sep2) nach jedem Element der Granularitaet.
        coords: 'polar' (4 Tok/Vert: r,sin,cos,z) | 'cart' (3 Tok/Vert: x,y,z);
        alle Offsets bleiben (x,y ueber r-Range/Quant, z ueber z-Range/Quant).
        Face-Orientierung: Quad-Ringe mit Normale entlang der Row-Fortpflanzung
        werden in diese Richtung orientiert (n·row_dir > 0); Seitenquads
        (Normale senkrecht zur Row-Richtung) behalten lex-min Rotation +
        kanonische Zyklusrichtung."""
        vp = mesh_data['vertices_polar']
        blks = mesh_data['faces'].T.tolist()
        Vc = mesh_data.get('vertices_cartesian')
        edges = mesh_data.get('edge_index')
        if emit_override is None:
            rows, emit = build_row_plan(blks, Vc, edges=edges)
        else:
            rows, emit = emit_override
        c = self.core
        toks = [c.start_token]
        cart = (coords == 'cart')

        def vq(vid):
            if cart:
                x, y, z = (float(v) for v in Vc[int(vid)])
                rmax = float(c.R_MAX)
                return [int(c._q_scalar(x, -rmax, rmax) + c.off_r),
                        int(c._q_scalar(y, -rmax, rmax) + c.off_r),
                        int(c._q_scalar(z, c.Z_MIN, c.Z_MAX) + c.off_r)]
            r = float(vp[vid, 0]); th = float(vp[vid, 1])
            ts, tc = c._q_angle(th)
            z = float(vp[vid, 2])
            return [int(c._q_scalar(r, c.R_MIN, c.R_MAX) + c.off_r), ts + c.off_ts,
                    tc + c.off_tc, int(c._q_scalar(z, c.Z_MIN, c.Z_MAX) + c.off_r)]

        npt = 3 if cart else 4
        cent = None
        if granularity == 'face':
            cent = torch.stack([Vc[list(b)].mean(0) for b in blks])
            row_dir = {}
            for row in rows:
                for bi, b in enumerate(row):
                    if bi + 1 < len(row):
                        d = cent[row[bi + 1]] - cent[b]
                    elif bi > 0:
                        d = cent[b] - cent[row[bi - 1]]
                    else:
                        d = torch.zeros(3)
                    n = float(d.norm())
                    row_dir[b] = d / n if n > 1e-12 else torch.zeros(3)

        def _orient_face_ring(ids_ring, d):
            """lex-min Rotation; Zyklusrichtung: wenn Newell-Normale ~ parallel
            zur Row-Richtung d, so orientieren, dass n·d > 0 (Flach-Normalen-
            Konvention); senkrechte Seitenquads behalten kanonische Richtung."""
            pts = [Vc[v].tolist() for v in ids_ring]
            k0 = min(range(4), key=lambda k: (round(pts[k][0], 9), round(pts[k][2], 9)))
            ring = [ids_ring[(k0 + s) % 4] for s in range(4)]
            if float(d.norm()) == 0.0:
                return ring
            n = torch.zeros(3)
            for i in range(4):
                n += torch.cross(Vc[ring[i]], Vc[ring[(i + 1) % 4]])
            nn = float(n.norm())
            if nn < 1e-9:
                return ring
            nd = float(torch.dot(n, d.to(n.dtype)))
            if abs(nd) > 0.5 * nn * 1.0 and nd < 0:
                ring = [ring[0]] + ring[:0:-1]
            return ring

        if granularity == 'face':
            for row in rows:
                for b in row:
                    v8 = blks[b]
                    if len({tuple(vq(int(v))) for v in v8}) < 8:
                        raise DegenerateBlockError(
                            f"block {b}: Quantisierungs-Kollision, <8 eindeutige Vert-Tuples")
                    for quad in _HEX_FACE_Q:
                        ids = [int(v8[i]) for i in quad]
                        ring = _orient_face_ring(ids, row_dir[b])
                        for vid in ring:
                            toks += vq(vid)
                        if eoe:
                            toks.append(c.sep2_token)
                toks.append(c.sep_token)
        else:
            for row in rows:
                prev_set = None
                for bi, b in enumerate(row):
                    vids = emit[b] if bi == 0 else emit[b][4:8]
                    tq = [tuple(vq(int(v))) for v in vids]
                    # Quantisierungs-Guard (wie Face-Pfad): zwei verschiedene
                    # Ecken duerfen nicht auf dasselbe Token-Tupel fallen,
                    # sonst verliert der Decode eine Ecke (Index-Wiederverwendung).
                    if len(set(tq)) < len(tq) or (prev_set is not None and set(tq) & prev_set):
                        raise DegenerateBlockError(
                            f"block {b}: Quantisierungs-Kollision, "
                            f"Emissions-Tupels nicht eindeutig")
                    prev_set = set(tq)
                    for vid in vids:
                        toks += vq(vid)
                    if granularity == 'block' and eoe:
                        toks.append(c.sep2_token)
                toks.append(c.sep_token)
        toks.append(c.stop_token)
        return toks

    # -- Detokenize -----------------------------------------------------------
    def detokenize(self, toks, granularity='row', coords='polar'):
        """Token list -> (vertices [M,3], blocks [F,8]) mit Token-Dedup.

        coords='cart': Verts kartesisch (x,y,z) zurueck, sonst polar (r,th,z).
        granularity='face': Quads (4 Verts) mit EOE/ohne sep2 parsen und aus den
        6 Quad-Ringen die 8 Block-Verts ueber geteilte Kanten reassemblieren
        (richtungstolerant: Ring-Winding beliebig).
        Sonst: sep2 ignorieren und Row-Grammatik (Head 8 Verts, Forts. 4) wie bisher."""

        if granularity == 'face':
            return self._detokenize_face(toks, coords=coords)

        toks = [int(t) for t in toks if int(t) != self.core.sep2_token]
        cartesian = (coords == 'cart')
        npt = 3 if cartesian else 4
        off_r, off_ts, off_tc = self.core.off_r, self.core.off_ts, self.core.off_tc
        rmin, rmax_ = self.core.R_MIN, self.core.R_MAX

        def dqv(group):
            if cartesian:
                rmax = float(rmax_)
                x = self.core._dq_scalar(group[0] - off_r, -rmax, rmax)
                y = self.core._dq_scalar(group[1] - off_r, -rmax, rmax)
                z = self.core._dq_scalar(group[2] - off_r, self.core.Z_MIN, self.core.Z_MAX)
                return (x, y, z)
            r = self.core._dq_scalar(group[0] - off_r, rmin, rmax_)
            a = self.core._dq_angle(group[1] - off_ts, group[2] - off_tc)
            z = self.core._dq_scalar(group[3] - off_r, self.core.Z_MIN, self.core.Z_MAX)
            return (r, a, z)

        i_end = toks.index(self.core.stop_token)
        toks = [int(t) for t in toks[1:i_end]]
        rows_t = []
        cur = []
        for t in toks:
            if t == self.core.sep_token:
                rows_t.append(cur); cur = []
            else:
                cur.append(t)
        if cur:
            rows_t.append(cur)

        verts = []
        vmap = {}
        blocks = []
        head_n = 8 * npt
        cont_n = 4 * npt
        for row in rows_t:
            j = 0
            pending = None
            while j < len(row):
                need = head_n if pending is None else cont_n
                group = row[j:j + need]
                assert len(group) == need, \
                    f"zu wenige Tokens fuer Block-Emission: {len(group)} < {need}"
                if pending is None:
                    ids = []
                    for k in range(8):
                        tk = tuple(group[k * npt:(k + 1) * npt])
                        if tk in vmap:
                            ids.append(vmap[tk])
                        else:
                            vmap[tk] = len(verts)
                            verts.append(dqv(tk))
                            ids.append(len(verts) - 1)
                    blocks.append(ids)
                    pending = tuple(group[4 * npt:8 * npt])
                else:
                    ids = [vmap[tuple(pending[k * npt:(k + 1) * npt])] for k in range(4)]
                    for k in range(4):
                        tk = tuple(group[k * npt:(k + 1) * npt])
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

    def _detokenize_face(self, toks, coords='polar'):
        """EOE/sep2-tolerante Face-Grammatik: Block = 6 Quads à 4 Verts
        (16 Tok polar / 12 Tok cart). Quad-Ring-Winding ist beliebig
        (Row-Fortpflanzungs-Orientierung), Reassemblierung nutzt nur
        geteilte Kanten über die Ring-Nachbarschaften."""
        c = self.core
        npt = 3 if coords == 'cart' else 4
        qn = 4 * npt
        blk_n = 6 * qn
        i_end = toks.index(c.stop_token)
        core_toks = [int(t) for t in toks[1:i_end] if int(t) != c.sep2_token]
        rows_t = []
        cur = []
        for t in core_toks:
            if t == c.sep_token:
                rows_t.append(cur); cur = []
            else:
                cur.append(t)
        if cur:
            rows_t.append(cur)

        def dqv(group):
            if coords == 'cart':
                rmax = float(c.R_MAX)
                x = c._dq_scalar(group[0] - c.off_r, -rmax, rmax)
                y = c._dq_scalar(group[1] - c.off_r, -rmax, rmax)
                z = c._dq_scalar(group[2] - c.off_r, c.Z_MIN, c.Z_MAX)
                return (x, y, z)
            r = c._dq_scalar(group[0] - c.off_r, c.R_MIN, c.R_MAX)
            a = c._dq_angle(group[1] - c.off_ts, group[2] - c.off_tc)
            z = c._dq_scalar(group[3] - c.off_r, c.Z_MIN, c.Z_MAX)
            return (r, a, z)

        def nxt(ring, a, not_b):
            ks = [k for k in range(4) if ring[k] == a]
            assert len(ks) == 1, f"Vertex {a} mehrfach im Quad-Ring"
            k = ks[0]
            out = ring[(k + 1) % 4]
            if out == not_b:
                out = ring[(k - 1) % 4]
            assert out != not_b, f"Quad-Ring degeneriert an {a}"
            return out

        verts, vmap, blocks = [], {}, []
        for row in rows_t:
            if len(row) % blk_n != 0:
                raise ValueError(
                    f"Face-Row-Länge {len(row)} nicht teilbar durch {blk_n} Tok")
            for bj in range(0, len(row), blk_n):
                qu = []
                for qi in range(6):
                    ids = []
                    for k in range(4):
                        p = bj + qi * qn
                        tk = tuple(row[p + k * npt: p + (k + 1) * npt])
                        if tk not in vmap:
                            vmap[tk] = len(verts)
                            verts.append(dqv(tk))
                        ids.append(vmap[tk])
                    qu.append(ids)
                v = [None] * 8
                v[0], v[1], v[2], v[3] = qu[0]
                unused = list(range(2, 6))

                def side_with(*want):
                    hits = [q for q in unused if all(w in qu[q] for w in want)]
                    assert len(hits) == 1, \
                        f"Seitenquad-Suche {want} -> {len(hits)} Kandidaten"
                    unused.remove(hits[0])
                    return qu[hits[0]]

                r2 = side_with(v[0], v[1])   # kanonisch (0,1,5,4)
                v[4] = nxt(r2, v[0], v[1])
                v[5] = nxt(r2, v[1], v[0])
                r3 = side_with(v[1], v[5])   # kanonisch (1,2,6,5)
                v[2] = nxt(r3, v[1], v[5])
                v[6] = nxt(r3, v[5], v[1])
                r4 = side_with(v[2], v[6])   # kanonisch (2,3,7,6)
                v[3] = nxt(r4, v[2], v[6])
                v[7] = nxt(r4, v[6], v[2])
                r5 = side_with(v[3], v[7])   # kanonisch (3,0,4,7)
                assert set(r5) == {v[3], v[0], v[4], v[7]}
                assert len(set(v)) == 8, "dupe Verts im Block aus Face-Quads"
                blocks.append(v)
        return torch.tensor(verts, dtype=torch.float32), torch.tensor(blocks, dtype=torch.long)


# ------------------------------------------------------------- Roundtrip-Check
def polar_from_cart(v):
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    return torch.stack([torch.hypot(x, y), torch.atan2(y, x), z], dim=-1)


def roundtrip(data, tok, coord_tol=0.02, granularity='row', eoe=False, coords='polar'):
    ok, maxerr, msgs = True, 0.0, []
    for si, mesh in enumerate(data):
        try:
            toks = tok.tokenize(mesh, granularity=granularity, eoe=eoe, coords=coords)
            vpt, blk = tok.detokenize(toks, granularity=granularity, coords=coords)
        except DegenerateBlockError as e:
            ok = False
            msgs.append(f'sample {si}: DegenerateBlockError: {e}')
            continue
        if coords == 'cart':
            vpt = polar_from_cart(vpt)
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
    for coords in ('polar', 'cart'):
        for gran, eoe in [('row', False), ('block', True), ('face', False), ('face', True)]:
            ok, maxerr, msgs = roundtrip(data, tok, granularity=gran, eoe=eoe, coords=coords)
            try:
                toks = tok.tokenize(data[0], granularity=gran, eoe=eoe, coords=coords)
                ntok = f'tokens={len(toks)}'
            except DegenerateBlockError as e:
                ntok = 'tokens=DEGENERATE'
            nb = data[0]['faces'].shape[1]
            name = f'{gran}{"" if gran == "row" else ("+eoe" if eoe else "")}/{coords}'
            print(f'[{name}] sample0: {ntok} '
                  f'blocks={nb} roundtrip: ok={ok} maxerr={maxerr:.4f}')
            for m in msgs:
                print(' ', m)


if __name__ == '__main__':
    main()
