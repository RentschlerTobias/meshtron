"""
prototype_twostage.py

Prototyp fuer den Zwei-Stufen-Tokenizer (Plan B, siehe
docs/ho_quad_transformer/05_face_block_generator.md).

Idee: Topologie von Geometrie trennen.
  Stufe 1 (Vertices): jede *eindeutige* Blockecke einmal als (r, theta) quantisiert.
  Stufe 2 (Faces):    jedes Quad als 4 *Zeiger* (Indizes) in die Vertexliste.
                      -> Gueltigkeit per Konstruktion, keine driftenden Duplikate.

Dieses Skript ist bewusst standalone (die Produktions-Klasse DomainTokenizer bleibt
unangetastet), damit der Round-trip real ausgefuehrt und gemessen werden kann.

Token-Layout (embedding_mode 0 / split ranges):
    r / scalar  : [0,               Qr-1]            (r UND t_norm, per Sektion getrennt)
    angle_sin   : [Qr,              Qr+Qa-1]         (theta UND alpha)
    angle_cos   : [Qr+Qa,           Qr+2Qa-1]
    vertex_idx  : [Qr+2Qa,          Qr+2Qa+Vmax-1]   (Pointer-Ziele fuer Stufe 2)
    specials    : start, end, sep, sep2, stop, pad   (danach)

Sequenz:
    [start]  (r,ts,tc)*M  [sep]  (i0,i1,i2,i3)*F  [sep2]  geom*4F  [end]

Stufe 3 (geom): pro gerichteter Half-Edge einer Face-Seite p0->p1 (Traversal-
Reihenfolge, jede gerichtete Kante mesh-weit GENAU EINMAL -> Twin liegt im
Nachbarblock in Gegenrichtung). 6 Tokens je Half-Edge:
    alpha_start(sin,cos), t_norm_start,  alpha_end(sin,cos), t_norm_end
Hermite-Tangenten: T0 = tn_s*[cos,sin](a_s) am Start, T1 = tn_e*[cos,sin](a_e)
am Ende. KEIN {u,v}-Dedup -> Multigraph/Blade Druck/Saug bleiben getrennt.
"""

import numpy as np
import torch


class TwoStageTokenizer:
    def __init__(self, quantization_r=512, quantization_a=256, max_vertices=2048,
                 repr_mode='hermite', r_bounds=(0.0, 1.0), tn_bounds=(0.0, 1.0)):
        # repr_mode:
        #   'hermite' -> pro Half-Edge 6 Tokens (α_s sin/cos, tn_s, α_e sin/cos, tn_e).
        #                kubisch, kann Wendepunkte (S-Kurve), braucht Tangenten-Betrag.
        #   'bezier'  -> pro Half-Edge 4 Tokens (α_s sin/cos, α_e sin/cos).
        #                quadratisch (k=2), Kontrollpunkt = Tangentenschnitt, KEIN Betrag
        #                -> kein Overshoot, kuerzere Sequenz, aber KEIN Wendepunkt.
        #   'cubic_bezier' -> pro Half-Edge 4 Tokens (s1,h1,s2,h2 in Chord-lokalen
        #                Koords). Kubisch, best-fit an die Streamline (2 Kontrollpunkte
        #                per Least-Squares). Kann Wendepunkt (S) UND ist genauer als
        #                hermite, weil nicht an die Extractor-Tangenten gebunden.
        assert repr_mode in ('hermite', 'bezier', 'cubic_bezier')
        self.repr_mode = repr_mode
        self.geom_per_edge = 6 if repr_mode == 'hermite' else 4
        self.Qr = quantization_r
        self.Qa = quantization_a
        self.Vmax = max_vertices

        self.off_r = 0
        self.off_ts = self.Qr
        self.off_tc = self.Qr + self.Qa
        self.off_idx = self.Qr + 2 * self.Qa

        # feste Bounds fuer cubic_bezier Chord-lokale Kontrollpunkt-Koords
        # s = Position laengs der Sehne (0..1 normal), h = Auslenkung quer (Chord-Einheiten)
        self.S_MIN, self.S_MAX = -0.5, 1.5
        self.H_MIN, self.H_MAX = -1.2, 1.2

        # FIXE globale Bounds (NICHT per-Mesh!) fuer r (Stufe 1) und tn (Hermite).
        # Grund: das Modell kennt zur Inferenzzeit keine Ground-Truth-Bounds -> per-Mesh
        # min/max waeren nicht dequantisierbar. r (Abstand zum center) und Tangenten-
        # Magnitude leben in normalisierter Einheitsdomain -> [0,1] mit Puffer deckt den
        # 10k-Datensatz (r in [0.013,0.910], tn in [0.0012,0.950]). fit_bounds() misst
        # die Spanne exakt aus einem Datensatz.
        self.R_MIN, self.R_MAX = r_bounds
        self.TN_MIN, self.TN_MAX = tn_bounds

        base = self.off_idx + self.Vmax
        self.start_token = base + 0
        self.end_token = base + 1
        self.sep_token = base + 2           # Trenner Stufe1 -> Stufe2
        self.sep2_token = base + 3          # Trenner Stufe2 -> Stufe3 (geom)
        self.stop_token = base + 4          # optional: explizites Ende
        self.pad_token = base + 5
        self.vocab_size = base + 6

    # ---- Quantisierung -------------------------------------------------
    def _q_scalar(self, val, vmin, vmax):
        if vmax - vmin < 1e-12:
            return 0
        q = int(np.round((val - vmin) / (vmax - vmin) * (self.Qr - 1)))
        return int(np.clip(q, 0, self.Qr - 1))

    def _dq_scalar(self, tok, vmin, vmax):
        return vmin + (tok / max(self.Qr - 1, 1)) * (vmax - vmin)

    def _q_angle(self, angle):
        s = (np.sin(angle) + 1.0) / 2.0
        c = (np.cos(angle) + 1.0) / 2.0
        s_tok = int(np.clip(np.round(s * (self.Qa - 1)), 0, self.Qa - 1))
        c_tok = int(np.clip(np.round(c * (self.Qa - 1)), 0, self.Qa - 1))
        return s_tok, c_tok

    def _dq_angle(self, s_tok, c_tok):
        s = (s_tok / max(self.Qa - 1, 1)) * 2.0 - 1.0
        c = (c_tok / max(self.Qa - 1, 1)) * 2.0 - 1.0
        return float(np.arctan2(s, c))  # [-pi, pi]

    # ---- Vertex-Sortierung (deterministisch) ---------------------------
    @staticmethod
    def _sort_order(vertices_polar):
        """Lexikografisch nach (theta, r) -> stabile, eindeutige Reihenfolge."""
        r = vertices_polar[:, 0].numpy()
        th = vertices_polar[:, 1].numpy()
        # np.lexsort: letzter Key ist primaer
        return np.lexsort((r, th))

    # ---- Bounds aus Datensatz messen -----------------------------------
    @staticmethod
    def fit_bounds(data, pad=0.02):
        """Globale (r, tn)-Bounds aus einem Datensatz -> FIXE Bounds fuers Modell.

        Einmal auf dem Trainingssatz bestimmen, dann an __init__(r_bounds=, tn_bounds=)
        geben. pad = relativer Puffer auf die Spanne (Robustheit gegen neue Meshes).
        Returns (r_bounds, tn_bounds) je als (min, max).
        """
        rmin, rmax = 1e9, -1e9
        tnmin, tnmax = 1e9, -1e9
        for d in data:
            r = d['vertices_polar'][:, 0].numpy()
            rmin = min(rmin, float(r.min())); rmax = max(rmax, float(r.max()))
            et = d['edge_tangents'].numpy()[:, [1, 3]]
            tnmin = min(tnmin, float(et.min())); tnmax = max(tnmax, float(et.max()))
        rp = (rmax - rmin) * pad; tp = (tnmax - tnmin) * pad
        return (rmin - rp, rmax + rp), (tnmin - tp, tnmax + tp)

    # ---- Tokenize ------------------------------------------------------
    def tokenize(self, mesh_data):
        vp = mesh_data['vertices_polar']          # [M,2] (r, theta)
        faces = mesh_data['faces']                # [4, F] globale Indizes
        M = vp.shape[0]
        assert M <= self.Vmax, f"M={M} > Vmax={self.Vmax}"

        # 1) Vertices sortieren, old->new Mapping
        order = self._sort_order(vp)              # new_pos -> old_idx
        old2new = np.empty(M, dtype=np.int64)
        old2new[order] = np.arange(M)

        # FIXE Bounds fuer r (nicht per-Mesh -> modelltauglich); theta via sincos braucht keine
        r_min, r_max = self.R_MIN, self.R_MAX

        toks = [self.start_token]

        # Stufe 1: sortierte Vertices
        for new_i in range(M):
            old_i = order[new_i]
            r = float(vp[old_i, 0]); th = float(vp[old_i, 1])
            r_tok = self._q_scalar(r, r_min, r_max) + self.off_r
            ts, tc = self._q_angle(th)
            toks += [r_tok, ts + self.off_ts, tc + self.off_tc]

        toks.append(self.sep_token)

        # Stufe 2: Faces als Pointer (auf new-Indizes)
        F = faces.shape[1]
        faces_new = old2new[faces.numpy()]        # [4, F]
        for fi in range(F):
            for k in range(4):
                toks.append(int(faces_new[k, fi]) + self.off_idx)

        toks.append(self.sep2_token)

        # Stufe 3: HO-Kantengeometrie pro gerichteter Half-Edge (Face-Traversal)
        # Lookup gerichtete Kante (global) -> [a_start, tn_start, a_end, tn_end]
        ei = mesh_data['edge_index'].numpy()          # [2,E] global
        et = mesh_data['edge_tangents'].numpy()       # [E,4]
        tan = {(int(ei[0, e]), int(ei[1, e])): et[e] for e in range(ei.shape[1])}
        tn_min, tn_max = self.TN_MIN, self.TN_MAX   # FIXE Bounds (nicht per-Mesh)
        e2s = mesh_data['edge_to_streamline']         # (u,v) -> [N,2] global
        cart = mesh_data['vertices_cartesian'].numpy()

        faces_g = faces.numpy()                       # [4,F] global
        n_missing = 0
        for fi in range(F):
            for k in range(4):
                p0 = int(faces_g[k, fi]); p1 = int(faces_g[(k + 1) % 4, fi])

                if self.repr_mode == 'cubic_bezier':
                    # best-fit 2 Kontrollpunkte an die Streamline, Chord-lokal kodiert
                    P0, P1 = cart[p0], cart[p1]
                    pts = e2s.get((p0, p1))
                    if pts is None:
                        n_missing += 1
                        s1, h1, s2, h2 = 1 / 3, 0.0, 2 / 3, 0.0     # gerade
                    else:
                        B1, B2 = self._fit_cubic_bezier(P0, P1, pts)
                        uh, nh, L = self._chord_frame(P0, P1)
                        s1 = float(np.dot(B1 - P0, uh) / L); h1 = float(np.dot(B1 - P0, nh) / L)
                        s2 = float(np.dot(B2 - P0, uh) / L); h2 = float(np.dot(B2 - P0, nh) / L)
                    toks += [self._q_scalar(s1, self.S_MIN, self.S_MAX) + self.off_r,
                             self._q_scalar(h1, self.H_MIN, self.H_MAX) + self.off_r,
                             self._q_scalar(s2, self.S_MIN, self.S_MAX) + self.off_r,
                             self._q_scalar(h2, self.H_MIN, self.H_MAX) + self.off_r]
                    continue

                info = tan.get((p0, p1))
                if info is None:
                    n_missing += 1
                    info = np.array([0.0, tn_min, 0.0, tn_min])  # Fallback: gerade
                a_s, tn_s, a_e, tn_e = info
                as_s, as_c = self._q_angle(a_s)
                ae_s, ae_c = self._q_angle(a_e)
                if self.repr_mode == 'hermite':
                    toks += [as_s + self.off_ts, as_c + self.off_tc,
                             self._q_scalar(tn_s, tn_min, tn_max) + self.off_r,
                             ae_s + self.off_ts, ae_c + self.off_tc,
                             self._q_scalar(tn_e, tn_min, tn_max) + self.off_r]
                else:  # bezier (k=2): nur Winkel, Kontrollpunkt = Tangentenschnitt
                    toks += [as_s + self.off_ts, as_c + self.off_tc,
                             ae_s + self.off_ts, ae_c + self.off_tc]

        toks.append(self.end_token)

        meta = {'r_min': r_min, 'r_max': r_max, 'M': M, 'F': F,
                'order': order, 'old2new': old2new, 'faces_new': faces_new,
                'tn_min': tn_min, 'tn_max': tn_max, 'n_missing_edges': n_missing}
        return toks, meta

    # ---- Detokenize ----------------------------------------------------
    def detokenize(self, toks, r_min=None, r_max=None, tn_min=None, tn_max=None):
        """Rekonstruiert (vertices, faces_new, geom) aus der Sequenz.

        geom: Liste je Half-Edge [a_start, tn_start, a_end, tn_end] (rad/skaliert),
        Reihenfolge = Face-Traversal (fi, k). Leer, wenn keine Stufe-3-Sektion.

        Bounds sind default die FIXEN self-Bounds -> kein per-Mesh meta noetig
        (modelltauglich). Explizite Werte nur fuer Sonderfaelle/Tests.
        """
        if r_min is None: r_min = self.R_MIN
        if r_max is None: r_max = self.R_MAX
        if tn_min is None: tn_min = self.TN_MIN
        if tn_max is None: tn_max = self.TN_MAX
        try:
            i_start = toks.index(self.start_token)
        except ValueError:
            i_start = -1
        i_sep = toks.index(self.sep_token)
        try:
            i_end = toks.index(self.end_token)
        except ValueError:
            i_end = len(toks)
        i_sep2 = toks.index(self.sep2_token) if self.sep2_token in toks else i_end

        vtoks = toks[i_start + 1:i_sep]
        ftoks = toks[i_sep + 1:i_sep2]
        gtoks = toks[i_sep2 + 1:i_end] if i_sep2 < i_end else []

        # Stufe 1: je 3 Tokens = ein Vertex
        n_v = len(vtoks) // 3
        verts = []
        for i in range(n_v):
            r_tok = vtoks[3 * i] - self.off_r
            ts = vtoks[3 * i + 1] - self.off_ts
            tc = vtoks[3 * i + 2] - self.off_tc
            r = self._dq_scalar(r_tok, r_min, r_max)
            th = self._dq_angle(ts, tc)            # [-pi, pi]
            verts.append((r, th))

        # Stufe 2: je 4 Tokens = ein Face (Pointer)
        n_f = len(ftoks) // 4
        faces_new = []
        for i in range(n_f):
            quad = [ftoks[4 * i + k] - self.off_idx for k in range(4)]
            faces_new.append(quad)

        # Stufe 3: je geom_per_edge Tokens = eine gerichtete Half-Edge
        geom = []
        gpe = self.geom_per_edge
        n_g = len(gtoks) // gpe
        for i in range(n_g):
            g = gtoks[gpe * i:gpe * i + gpe]
            if self.repr_mode == 'hermite':
                a_s = self._dq_angle(g[0] - self.off_ts, g[1] - self.off_tc)
                a_e = self._dq_angle(g[3] - self.off_ts, g[4] - self.off_tc)
                tn_s = self._dq_scalar(g[2] - self.off_r, tn_min, tn_max)
                tn_e = self._dq_scalar(g[5] - self.off_r, tn_min, tn_max)
                geom.append([a_s, tn_s, a_e, tn_e])
            elif self.repr_mode == 'cubic_bezier':  # 4 Skalare s1,h1,s2,h2 (Chord-lokal)
                s1 = self._dq_scalar(g[0] - self.off_r, self.S_MIN, self.S_MAX)
                h1 = self._dq_scalar(g[1] - self.off_r, self.H_MIN, self.H_MAX)
                s2 = self._dq_scalar(g[2] - self.off_r, self.S_MIN, self.S_MAX)
                h2 = self._dq_scalar(g[3] - self.off_r, self.H_MIN, self.H_MAX)
                geom.append([s1, h1, s2, h2])
            else:  # bezier: nur 2 Winkel, kein Betrag
                a_s = self._dq_angle(g[0] - self.off_ts, g[1] - self.off_tc)
                a_e = self._dq_angle(g[2] - self.off_ts, g[3] - self.off_tc)
                geom.append([a_s, None, a_e, None])

        return verts, faces_new, geom

    # ---- Kanten-Rekonstruktion (Stufe 3) -------------------------------
    @staticmethod
    def _hermite(P0, P1, T0, T1, n=50):
        t = np.linspace(0, 1, n)
        h00 = 2 * t**3 - 3 * t**2 + 1; h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2;    h11 = t**3 - t**2
        return (h00[:, None] * P0 + h10[:, None] * T0 +
                h01[:, None] * P1 + h11[:, None] * T1)

    @staticmethod
    def _chord_frame(P0, P1):
        u = np.asarray(P1, float) - np.asarray(P0, float)
        L = float(np.linalg.norm(u))
        if L < 1e-12:
            return np.array([1.0, 0.0]), np.array([0.0, 1.0]), 1e-12
        uh = u / L
        nh = np.array([-uh[1], uh[0]])
        return uh, nh, L

    @staticmethod
    def _fit_cubic_bezier(P0, P1, pts):
        """Best-fit kubischer Bezier (Endpunkte fix): 2 innere Kontrollpunkte via
        Least-Squares an die Streamline. Returns (B1, B2)."""
        pts = np.asarray(pts, float)
        P0 = np.asarray(P0, float); P1 = np.asarray(P1, float)
        d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(d)])
        t = cum / cum[-1] if cum[-1] > 1e-12 else np.linspace(0, 1, len(pts))
        b1 = 3 * (1 - t) ** 2 * t
        b2 = 3 * (1 - t) * t ** 2
        rhs = pts - ((1 - t) ** 3)[:, None] * P0 - (t ** 3)[:, None] * P1
        A = np.stack([b1, b2], axis=1)                # [N,2]
        sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)  # [2,2]
        return sol[0], sol[1]

    def _cubic_bezier_curve(self, P0, P1, B1, B2, n=50):
        t = np.linspace(0, 1, n)[:, None]
        return ((1 - t) ** 3 * P0 + 3 * (1 - t) ** 2 * t * B1 +
                3 * (1 - t) * t ** 2 * B2 + t ** 3 * P1)

    @staticmethod
    def _quad_bezier(P0, P2, dir0, dir2, n=50, straight_deg=3.0):
        """Quadratischer Bezier (k=2) durch P0,P2; Kontrollpunkt = Tangentenschnitt.
        Portiert aus reconstruct_domain.quadratic_bezier_2d (kein Overshoot,
        kein Wendepunkt). dir0/dir2 = Tangenten-RICHTUNGEN (Betrag egal)."""
        t = np.linspace(0, 1, n)[:, None]
        P0 = np.asarray(P0, float); P2 = np.asarray(P2, float)
        d0 = np.asarray(dir0, float); d2 = np.asarray(dir2, float)
        n0 = np.linalg.norm(d0); n2 = np.linalg.norm(d2)
        if n0 < 1e-12 or n2 < 1e-12:
            return (1 - t) * P0 + t * P2
        d0 /= n0; d2 /= n2
        cosang = abs(float(np.clip(np.dot(d0, d2), -1.0, 1.0)))
        if np.degrees(np.arccos(cosang)) < straight_deg:
            return (1 - t) * P0 + t * P2
        A = np.array([[d0[0], -d2[0]], [d0[1], -d2[1]]])
        try:
            s, _u = np.linalg.solve(A, P2 - P0)
        except np.linalg.LinAlgError:
            return (1 - t) * P0 + t * P2
        C = P0 + s * d0
        chord = np.linalg.norm(P2 - P0)
        if np.linalg.norm(C - 0.5 * (P0 + P2)) > 1.0 * max(chord, 1e-9):
            return (1 - t) * P0 + t * P2
        return (1 - t) ** 2 * P0 + 2 * (1 - t) * t * C + t ** 2 * P2

    def reconstruct_edges(self, verts, faces_new, geom, center, n=50):
        """Baut gekruemmte Half-Edge-Kurven aus verts/faces_new/geom.

        Returns Liste von Dicts: {P0,P1,curve,(u,v)} in Face-Traversal-Reihenfolge.
        """
        cx, cy = float(center[0]), float(center[1])
        xy = np.array([[cx + r * np.cos(th), cy + r * np.sin(th)] for (r, th) in verts])
        edges = []
        j = 0
        for quad in faces_new:
            for k in range(4):
                u = quad[k]; v = quad[(k + 1) % 4]
                P0, P1 = xy[u], xy[v]
                g0, g1, g2, g3 = geom[j]; j += 1
                if self.repr_mode == 'cubic_bezier':
                    s1, h1, s2, h2 = g0, g1, g2, g3
                    uh, nh, L = self._chord_frame(P0, P1)
                    B1 = P0 + s1 * L * uh + h1 * L * nh
                    B2 = P0 + s2 * L * uh + h2 * L * nh
                    curve = self._cubic_bezier_curve(P0, P1, B1, B2, n)
                    edges.append({'u': u, 'v': v, 'P0': P0, 'P1': P1, 'curve': curve})
                    continue
                a_s, tn_s, a_e, tn_e = g0, g1, g2, g3
                if self.repr_mode == 'bezier':
                    dir0 = np.array([np.cos(a_s), np.sin(a_s)])
                    dir2 = np.array([np.cos(a_e), np.sin(a_e)])
                    curve = self._quad_bezier(P0, P1, dir0, dir2, n)
                elif tn_s is None:   # hermite ohne Magnitude -> gerade
                    curve = (1 - np.linspace(0, 1, n))[:, None] * P0 + \
                            np.linspace(0, 1, n)[:, None] * P1
                else:
                    T0 = tn_s * np.array([np.cos(a_s), np.sin(a_s)])
                    T1 = tn_e * np.array([np.cos(a_e), np.sin(a_e)])
                    curve = self._hermite(P0, P1, T0, T1, n)
                edges.append({'u': u, 'v': v, 'P0': P0, 'P1': P1, 'curve': curve})
        return edges


# ------------------------------------------------------------------
# Round-trip Test
# ------------------------------------------------------------------
def _ang_diff(a, b):
    """Kleinste Winkeldifferenz, wrap-sicher."""
    d = (a - b + np.pi) % (2 * np.pi) - np.pi
    return abs(d)


def round_trip_report(data, tok, n_max=None, verbose_fail=3):
    n = len(data) if n_max is None else min(n_max, len(data))
    topo_ok = 0
    topo_fail = 0
    r_err_max = 0.0
    ang_err_max = 0.0
    cart_err_max = 0.0
    geom_rel_errs = []
    seq_lens = []
    fails_shown = 0

    for idx in range(n):
        d = data[idx]
        vp = d['vertices_polar']
        toks, meta = tok.tokenize(d)
        seq_lens.append(len(toks))
        verts, faces_new, geom = tok.detokenize(toks)   # nutzt FIXE self-Bounds

        # --- Geometrie Stufe 3: rekonstruierte Kurven vs. edge_to_streamline ---
        center = d['center'].numpy()
        e2s = d['edge_to_streamline']
        order = meta['order']
        edges = tok.reconstruct_edges(verts, faces_new, geom, center, n=50)
        for e in edges:
            ug = int(order[e['u']]); vg = int(order[e['v']])
            gt = e2s.get((ug, vg))
            if gt is None:
                continue
            gt = np.asarray(gt, float)
            # resample rekonstruierte Kurve auf len(gt)
            c = e['curve']
            ts = np.linspace(0, 1, len(gt))
            src = np.linspace(0, 1, len(c))
            rec = np.column_stack([np.interp(ts, src, c[:, 0]),
                                   np.interp(ts, src, c[:, 1])])
            chord = np.linalg.norm(e['P1'] - e['P0']) + 1e-9
            geom_rel_errs.append(float(np.max(np.linalg.norm(rec - gt, axis=1)) / chord))

        # --- Topologie exakt? (Integer-Vergleich der Face-Indizes) ---
        gt_faces = meta['faces_new'].T.tolist()   # [F,4]
        topo_match = (len(verts) == meta['M'] and
                      len(faces_new) == meta['F'] and
                      faces_new == gt_faces)
        if topo_match:
            topo_ok += 1
        else:
            topo_fail += 1
            if fails_shown < verbose_fail:
                print(f"  [FAIL topo] mesh {idx}: M {len(verts)}/{meta['M']} "
                      f"F {len(faces_new)}/{meta['F']} faces_eq={faces_new == gt_faces}")
                fails_shown += 1

        # --- Geometrie: Quantisierungsfehler (in sortierter Reihenfolge) ---
        order = meta['order']
        for new_i, (r_rec, th_rec) in enumerate(verts):
            old_i = order[new_i]
            r_gt = float(vp[old_i, 0]); th_gt = float(vp[old_i, 1])
            r_err = abs(r_rec - r_gt)
            a_err = _ang_diff(th_rec, th_gt)
            r_err_max = max(r_err_max, r_err)
            ang_err_max = max(ang_err_max, a_err)
            # kartesischer Fehler (relativ zum Zentrum)
            x_gt = r_gt * np.cos(th_gt); y_gt = r_gt * np.sin(th_gt)
            x_rc = r_rec * np.cos(th_rec); y_rc = r_rec * np.sin(th_rec)
            cart_err_max = max(cart_err_max, float(np.hypot(x_rc - x_gt, y_rc - y_gt)))

    print(f"\n=== Round-trip ueber {n} Meshes ===")
    print(f"Topologie exakt : {topo_ok}/{n}  (fail: {topo_fail})")
    print(f"Seq-Len         : min {min(seq_lens)}  max {max(seq_lens)}  "
          f"mean {sum(seq_lens)/len(seq_lens):.0f}")
    print(f"max r-Fehler    : {r_err_max:.6e}")
    print(f"max Winkelfehler: {ang_err_max:.6e} rad")
    print(f"max kart. Fehler: {cart_err_max:.6e}  (Einheiten der Domain)")
    if geom_rel_errs:
        ge = np.array(geom_rel_errs)
        print(f"Stufe-3 Kanten  : {len(ge)} Half-Edges vs edge_to_streamline")
        print(f"  rel-Fehler/Chord: mean {ge.mean():.4f}  median {np.median(ge):.4f}  "
              f"max {ge.max():.4f}")
    print(f"vocab_size      : {tok.vocab_size}")
    return topo_ok == n


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='domain_data.pt')
    ap.add_argument('--n', type=int, default=None)
    ap.add_argument('--qr', type=int, default=512)
    ap.add_argument('--qa', type=int, default=256)
    ap.add_argument('--mode', default='hermite',
                    choices=['hermite', 'bezier', 'cubic_bezier'])
    ap.add_argument('--fit-bounds', action='store_true',
                    help='globale r/tn-Bounds aus dem Datensatz messen statt Default [0,1]')
    args = ap.parse_args()

    print(f"Lade {args.data} ...")
    data = torch.load(args.data, weights_only=False)
    print(f"{len(data)} Meshes geladen.")

    # Vmax dynamisch bestimmen
    max_v = max(d['vertices_polar'].shape[0] for d in data)
    print(f"max Vertices im Datensatz: {max_v}")

    # FIXE Bounds: Default [0,1]; --fit-bounds misst die Spanne aus dem Set
    r_bounds, tn_bounds = (0.0, 1.0), (0.0, 1.0)
    if args.fit_bounds:
        r_bounds, tn_bounds = TwoStageTokenizer.fit_bounds(data)

    tok = TwoStageTokenizer(quantization_r=args.qr, quantization_a=args.qa,
                            max_vertices=max_v + 16, repr_mode=args.mode,
                            r_bounds=r_bounds, tn_bounds=tn_bounds)
    print(f"repr_mode: {args.mode}  ({tok.geom_per_edge} geom-Tokens/Half-Edge)")
    print(f"Bounds  : r [{tok.R_MIN:.3f},{tok.R_MAX:.3f}]  tn [{tok.TN_MIN:.3f},{tok.TN_MAX:.3f}]  (FIX)")
    ok = round_trip_report(data, tok, n_max=args.n)
    print("\nRESULT:", "PASS (Topologie 100% exakt)" if ok else "TEILWEISE (siehe fails)")
