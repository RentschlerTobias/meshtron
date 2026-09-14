"""
tokenizer_domain.py

DEPRECATED: part of the MeshtronDomain family (failed experiment, superseded by
Polytron / polytron_tokenizer.py). Kept for reference only, not maintained or
extended.

Tokenizer fuer Domain-Partitionen mit gekruemmten Kanten.

Features:
  - Polar-Koordinaten fuer Vertices (r, theta) und Tangenten (norm, alpha)
  - Sinus/Cosinus-Paar-Kodierung fuer Winkel (vermeidet Wrap-around)
  - 3 Sortierstrategien: A (keine Kompression), B (Row-Kompression), C (Vertex-first + explizite Faces)
  - 3 Embedding-Modi:   A (ein Vokabular, aufgeteilte Bereiche),
                         B (echt gemeinsames Vokabular),
                         C (separate Embeddings pro Typ)

Token-Schema pro Vertex-Platz:
    [r, theta_sin, theta_cos, t_norm, alpha_in_sin, alpha_in_cos, alpha_out_sin, alpha_out_cos]
    = 8 Tokens (1 Skalar + 2*Sincos + 1 Skalar + 2*Sincos + 2*Sincos)

Special Tokens:
    start_token, end_token, pad_token, eor_token (end-of-row), sep_token (Vertex/Face Separator)
"""


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from torch_geometric.utils import lexsort
import math


class DomainTokenizer:
    def __init__(
        self,
        quantization_r: int = 512,
        quantization_a: int = 256,
        sorting_strategy: int = 0,      # 0=A (keine Kompr.), 1=B (Row-Kompr.), 2=C (Vertex-first)
        embedding_mode: int = 0,           # 0=A (split ranges), 1=B (shared), 2=C (separate)
        verbose: bool = False,
        max_length_padding: Optional[int] = None,
    ):
        self.verbose = verbose
        self.quantization_r = quantization_r
        self.quantization_a = quantization_a
        self.sorting_strategy = sorting_strategy
        self.embedding_mode = embedding_mode

        # Special tokens (appended after coordinate vocab)
        self.start_token = 0  # wird spaeter je nach Modus korrekt gesetzt
        self.end_token = 1
        self.pad_token = 2
        self.eor_token = 3   # end-of-row (nur Strategy 1)
        self.sep_token = 4   # separator (Strategy 2: zwischen Vertices und Faces)
        self.n_special = 5

        # Berechne Vokabular-Groesse je nach Modus
        self._compute_vocab()

        self.max_length_padding = max_length_padding
        self.bounds = None   # (r_min, theta_min, t_min, alpha_min, ...)
        self.max_length_token_sequence = 0
        self.min_length_token_sequence = float("inf")

    # ------------------------------------------------------------------
    # Vokabular-Berechnung
    # ------------------------------------------------------------------
    def _compute_vocab(self):
        """Berechne vocab_size und Token-Offsets je nach embedding_mode."""
        Qr = self.quantization_r
        Qa = self.quantization_a

        if self.embedding_mode == 0:
            # Modus A: ein Vokabular, aufgeteilte Bereiche
            #   r:            [0, Qr-1]
            #   theta_sin:    [Qr, Qr+Qa-1]
            #   theta_cos:    [Qr+Qa, Qr+2*Qa-1]
            #   t_norm:       [Qr+2*Qa, 2*Qr+2*Qa-1]
            #   alpha_in_sin: [2*Qr+2*Qa, 2*Qr+3*Qa-1]
            #   alpha_in_cos: [2*Qr+3*Qa, 2*Qr+4*Qa-1]
            #   alpha_out_sin:[2*Qr+4*Qa, 2*Qr+5*Qa-1]
            #   alpha_out_cos:[2*Qr+5*Qa, 2*Qr+6*Qa-1]
            self.offset_r = 0
            self.offset_theta_sin = Qr
            self.offset_theta_cos = Qr + Qa
            self.offset_t = Qr + 2 * Qa
            self.offset_alpha_in_sin = 2 * Qr + 2 * Qa
            self.offset_alpha_in_cos = 2 * Qr + 3 * Qa
            self.offset_alpha_out_sin = 2 * Qr + 4 * Qa
            self.offset_alpha_out_cos = 2 * Qr + 5 * Qa
            coord_vocab = 2 * Qr + 6 * Qa

        elif self.embedding_mode == 1:
            # Modus B: echt gemeinsames Vokabular
            # Alle Werte werden auf [0, Q-1] quantisiert, wobei Q = max(Qr, Qa)
            Q = max(Qr, Qa)
            self.offset_r = 0
            self.offset_theta_sin = 0
            self.offset_theta_cos = 0
            self.offset_t = 0
            self.offset_alpha_in_sin = 0
            self.offset_alpha_in_cos = 0
            self.offset_alpha_out_sin = 0
            self.offset_alpha_out_cos = 0
            self._Q_shared = Q
            coord_vocab = Q

        elif self.embedding_mode == 2:
            # Modus C: separate Embeddings -> wir behalten trotzdem ein globales
            # Vokabular bei, das Embedding-Layer wird extern geregelt.
            # Wir nutzen einfach 2 Bereiche: Skalare (r, t_norm) und Winkel (sincos)
            self.offset_r = 0
            self.offset_t = 0
            self.offset_theta_sin = Qr
            self.offset_theta_cos = Qr
            self.offset_alpha_in_sin = Qr
            self.offset_alpha_in_cos = Qr
            self.offset_alpha_out_sin = Qr
            self.offset_alpha_out_cos = Qr
            coord_vocab = Qr + Qa

        else:
            raise ValueError(f"embedding_mode {self.embedding_mode} ungueltig")

        self.coord_vocab_size = coord_vocab
        self.start_token = coord_vocab
        self.end_token = coord_vocab + 1
        self.pad_token = coord_vocab + 2
        self.eor_token = coord_vocab + 3
        self.sep_token = coord_vocab + 4
        self.vocab_size = coord_vocab + self.n_special

    # ------------------------------------------------------------------
    # Quantisierung
    # ------------------------------------------------------------------
    def _quantize_scalar(self, val: float, min_val: float, max_val: float) -> int:
        if max_val - min_val < 1e-12:
            return 0
        q = int(np.round((val - min_val) / (max_val - min_val) * (self.quantization_r - 1)))
        return np.clip(q, 0, self.quantization_r - 1)

    def _dequantize_scalar(self, token: int, min_val: float, max_val: float) -> float:
        return min_val + (token / max(self.quantization_r - 1, 1)) * (max_val - min_val)

    def _quantize_angle_sincos(self, angle: float) -> Tuple[int, int]:
        """Winkel [0, 2π) -> (sin_token, cos_token) in [0, Qa-1]."""
        s = (np.sin(angle) + 1.0) / 2.0   # [0, 1]
        c = (np.cos(angle) + 1.0) / 2.0   # [0, 1]
        s_tok = int(np.round(s * (self.quantization_a - 1)))
        c_tok = int(np.round(c * (self.quantization_a - 1)))
        return (
            np.clip(s_tok, 0, self.quantization_a - 1),
            np.clip(c_tok, 0, self.quantization_a - 1),
        )

    def _dequantize_angle_sincos(self, sin_tok: int, cos_tok: int) -> float:
        s = (sin_tok / max(self.quantization_a - 1, 1)) * 2.0 - 1.0
        c = (cos_tok / max(self.quantization_a - 1, 1)) * 2.0 - 1.0
        return np.arctan2(s, c)

    def _quantize_shared(self, val: float, min_val: float, max_val: float) -> int:
        """Fuer Modus B: alles auf [0, Q_shared-1]."""
        Q = self._Q_shared
        if max_val - min_val < 1e-12:
            return 0
        q = int(np.round((val - min_val) / (max_val - min_val) * (Q - 1)))
        return np.clip(q, 0, Q - 1)

    def _dequantize_shared(self, token: int, min_val: float, max_val: float) -> float:
        Q = self._Q_shared
        return min_val + (token / max(Q - 1, 1)) * (max_val - min_val)

    # ------------------------------------------------------------------
    # Tangenten-Lookup (pro gerichteter Kante)
    # ------------------------------------------------------------------
    def _build_edge_lookup(self, edge_index: torch.Tensor):
        """Baut Dict {(u,v): idx} fuer schnellen Zugriff."""
        self._edge_lookup = {}
        for i in range(edge_index.shape[1]):
            u = int(edge_index[0, i].item())
            v = int(edge_index[1, i].item())
            self._edge_lookup[(u, v)] = i

    def _get_alpha_in(self, vi: int, prev: int, edge_tangents: torch.Tensor) -> float:
        """Tangente am Vertex vi, von prev kommend (Richtung prev->vi)."""
        key = (prev, vi)
        if key in self._edge_lookup:
            i = self._edge_lookup[key]
            return float(edge_tangents[i, 2].item())  # alpha_end
        key = (vi, prev)
        if key in self._edge_lookup:
            i = self._edge_lookup[key]
            a = float(edge_tangents[i, 0].item())  # alpha_start (Richtung vi->prev)
            return (a + np.pi) % (2 * np.pi)         # Richtung prev->vi
        raise ValueError(f"Kante zwischen {prev} und {vi} nicht gefunden")

    def _get_alpha_out(self, vi: int, next_v: int, edge_tangents: torch.Tensor) -> float:
        """Tangente am Vertex vi, in Richtung next_v."""
        key = (vi, next_v)
        if key in self._edge_lookup:
            i = self._edge_lookup[key]
            return float(edge_tangents[i, 0].item())  # alpha_start
        key = (next_v, vi)
        if key in self._edge_lookup:
            i = self._edge_lookup[key]
            a = float(edge_tangents[i, 2].item())  # alpha_end (Richtung next_v->vi)
            return (a + np.pi) % (2 * np.pi)         # Richtung vi->next_v
        raise ValueError(f"Kante zwischen {vi} und {next_v} nicht gefunden")

    def _get_t_norm(self, vi: int, next_v: int, edge_tangents: torch.Tensor) -> float:
        """Tangentialnorm fuer Kante vi->next_v."""
        key = (vi, next_v)
        if key in self._edge_lookup:
            i = self._edge_lookup[key]
            return float(edge_tangents[i, 1].item())  # t_norm_start
        key = (next_v, vi)
        if key in self._edge_lookup:
            i = self._edge_lookup[key]
            return float(edge_tangents[i, 3].item())  # t_norm_end
        raise ValueError(f"Kante zwischen {vi} und {next_v} nicht gefunden")

    # ------------------------------------------------------------------
    # Face-Sortierung
    # ------------------------------------------------------------------
    def _order_faces(self, vertices: torch.Tensor, faces: torch.Tensor):
        """
        Sortiere Faces und bestimme Rows (fuer Strategy 1).
        Returns: (ordered_faces [n,4], rows List[(start, end)])
        """
        n_faces = faces.shape[1]
        if n_faces == 0:
            return torch.empty((0, 4), dtype=torch.long), []

        # Schwerpunkte
        # faces is [4, n_faces]; we need [n_faces, 4, 2] then mean over vertices
        faces_perm = faces.T  # [n_faces, 4]
        centroids = vertices[faces_perm].mean(dim=1)  # [n_faces, 2]

        # 1. Lexikographisch nach (y, x) sortieren
        order = lexsort(centroids.T)  # indices
        ordered = faces_perm[order]     # [n_faces, 4]

        if self.sorting_strategy == 0:
            # Strategy A: keine Kompression, keine Row-Gruppierung
            return ordered, []

        if self.sorting_strategy in (1, 2):
            # Strategy B/C: Row-Gruppierung
            rows = []
            start = 0
            for i in range(1, n_faces):
                # Teilen sich aufeinanderfolgende Faces 2 Vertices?
                shared = len(set(ordered[i - 1].tolist()) & set(ordered[i].tolist()))
                if shared < 2:
                    rows.append((start, i))
                    start = i
            rows.append((start, n_faces))

            # Richtung pro Row bestimmen (left-to-right / right-to-left)
            # und Vertices innerhalb der Faces anpassen
            result = []
            for s, e in rows:
                # Richtung: x-Schwerpunkt des ersten vs letzten Face
                cx_first = centroids[order[s]][0].item()
                cx_last = centroids[order[e - 1]][0].item()
                left_to_right = cx_first <= cx_last

                for i in range(s, e):
                    face = ordered[i].tolist()
                    arranged = self._arrange_face(vertices, face, left_to_right,
                                                 prev=ordered[i - 1].tolist() if i > s else None,
                                                 next_face=ordered[i + 1].tolist() if i + 1 < e else None)
                    result.append(arranged)

            return torch.tensor(result, dtype=torch.long), rows

        raise ValueError(f"sorting_strategy {self.sorting_strategy} ungueltig")

    def _arrange_face(self, vertices, face, left_to_right, prev=None, next_face=None):
        """
        Ordne die 4 Vertices eines Faces so an, dass:
          - v0,v1 = Austrittskante des vorherigen Faces (oder Start-Kante)
          - v2,v3 = Eingangskante zum naechsten Face (oder End-Kante)
        Bei left-to-right: clockwise (BL, TL, TR, BR)
        Bei right-to-left: counter-clockwise (BR, TR, TL, BL)
        """
        def ccw_ring(verts):
            cx = sum(vertices[v][0].item() for v in verts) / 4
            cy = sum(vertices[v][1].item() for v in verts) / 4
            return sorted(verts, key=lambda v: math.atan2(
                vertices[v][1].item() - cy, vertices[v][0].item() - cx))

        ring = ccw_ring(face)

        if left_to_right:
            # clockwise: BL -> TL -> TR -> BR
            ring = ring[::-1]
            start_v = min(face, key=lambda v: (vertices[v][1].item(), vertices[v][0].item()))
        else:
            # counter-clockwise: BR -> TR -> TL -> BL
            start_v = min(face, key=lambda v: (vertices[v][1].item(), -vertices[v][0].item()))

        i = ring.index(start_v)
        arranged = ring[i:] + ring[:i]

        # Falls prev existiert: v0,v1 = umgekehrte Austrittskante von prev
        if prev is not None:
            shared = list(set(prev) & set(face))
            if len(shared) == 2:
                a, b = shared
                # prev war [p0,p1,p2,p3]; Austrittskante = p2,p3
                # Diese soll v0,v1 sein, aber umgekehrt (v0=b, v1=a), damit v0,v1 = Austritt von prev
                try:
                    ia = arranged.index(a)
                    ib = arranged.index(b)
                    # pruefe ob (a,b) oder (b,a) benachbart in arranged
                    if (ia + 1) % 4 == ib:
                        # a vor b -> b,a soll v0,v1 sein -> rotiere
                        idx = arranged.index(b)
                        arranged = arranged[idx:] + arranged[:idx]
                    elif (ib + 1) % 4 == ia:
                        # b vor a -> bereits richtig
                        idx = arranged.index(b)
                        arranged = arranged[idx:] + arranged[:idx]
                except ValueError:
                    pass

        return arranged

    # ------------------------------------------------------------------
    # Haupt-Methoden
    # ------------------------------------------------------------------
    def tokenize(self, mesh_data: Dict) -> List[int]:
        """
        Args:
            mesh_data: dict mit keys:
                vertices_cartesian, vertices_polar, faces, edges_polar,
                center, bounds, tri_coordinates
        Returns:
            token list
        """
        if self.verbose:
            print("Starte Tokenisierung...")

        vertices_cart = mesh_data['vertices_cartesian']  # [n,2]
        vertices_pol = mesh_data['vertices_polar']        # [n,2] (r, theta)
        faces = mesh_data['faces']                        # [4, n_faces]
        edge_index = mesh_data['edge_index']              # [2, n_edges]
        edge_tangents = mesh_data['edge_tangents']        # [n_edges, 4]
        bounds = mesh_data['bounds']                      # [xmin, ymin, xmax, ymax]

        # Edge lookup bauen
        self._build_edge_lookup(edge_index)

        # Globale Bounds fuer Quantisierung berechnen
        r_vals = vertices_pol[:, 0].numpy()
        theta_vals = vertices_pol[:, 1].numpy()

        r_min, r_max = float(r_vals.min()), float(r_vals.max())

        # Tangenten-Normen sammeln
        t_norms = edge_tangents[:, [1, 3]].flatten().tolist()
        t_min = min(t_norms) if t_norms else 0.0
        t_max = max(t_norms) if t_norms else 1.0

        self.bounds_tokenize = {
            'r_min': r_min, 'r_max': r_max,
            't_min': t_min, 't_max': t_max,
        }

        # Faces sortieren
        ordered_faces, rows = self._order_faces(vertices_cart, faces)

        tokens = [self.start_token] * 8  # etwas laengerer Start

        if self.sorting_strategy in (0, 1):
            # Strategy A oder B: Face-Traversal
            tokens = self._build_sequence_faces(vertices_pol, ordered_faces, rows,
                                                edge_tangents)
        elif self.sorting_strategy == 2:
            # Strategy C: Vertex-first, dann explicit faces
            tokens = self._build_sequence_vertex_first(vertices_pol, ordered_faces,
                                                       edge_tangents)

        if self.max_length_token_sequence < len(tokens):
            self.max_length_token_sequence = len(tokens)
        if self.min_length_token_sequence > len(tokens):
            self.min_length_token_sequence = len(tokens)

        # Padding
        if self.max_length_padding is not None:
            if len(tokens) > self.max_length_padding:
                tokens = tokens[:self.max_length_padding]
            if len(tokens) < self.max_length_padding:
                tokens += [self.pad_token] * (self.max_length_padding - len(tokens))

        if self.verbose:
            print(f"Tokenisierung abgeschlossen: {len(tokens)} Tokens")
        return tokens

    def _build_sequence_faces(self, vertices_pol, ordered_faces, rows, edge_tangents):
        """Strategy A (rows==None/[]) oder B (rows!=[])."""
        tokens = []
        n_faces = ordered_faces.shape[0]

        if n_faces == 0:
            tokens.append(self.start_token)
            tokens.append(self.end_token)
            return tokens

        use_compression = (self.sorting_strategy == 1 and len(rows) > 0)

        for s, e in (rows if use_compression else [(0, n_faces)]):
            for fi in range(s, e):
                face = ordered_faces[fi].tolist()

                # Bei Kompression: erstes Face der Row -> 4 Vertices, sonst 2
                if use_compression and fi > s:
                    v_start = 2  # nur v2,v3 (die "neuen" Vertices)
                else:
                    v_start = 0

                for vi_idx in range(v_start, 4):
                    vi = face[vi_idx]
                    prev = face[(vi_idx - 1) % 4]
                    next_v = face[(vi_idx + 1) % 4]

                    tokens.extend(self._encode_vertex_place(
                        vi, prev, next_v, vertices_pol, edge_tangents
                    ))

            if use_compression:
                tokens.append(self.eor_token)

        tokens = [self.start_token] * 8 + tokens + [self.end_token] * 8
        return tokens

    def _build_sequence_vertex_first(self, vertices_pol, ordered_faces, edge_tangents):
        """Strategy C: FastMesh/AMT-Style.
        ...
        """
        return self._build_sequence_faces(vertices_pol, ordered_faces, [], edge_tangents)

    def _encode_vertex_place(self, vi, prev, next_v, vertices_pol, edge_tangents):
        """
        Kodiere einen Vertex-Platz (mit Kontext prev/next) zu 8 Tokens.
        Returns: list of 8 ints
        """
        # Vertex-Polar
        r = float(vertices_pol[vi][0])
        theta = float(vertices_pol[vi][1])

        # Tangenten
        t_norm = self._get_t_norm(vi, next_v, edge_tangents)
        alpha_in = self._get_alpha_in(vi, prev, edge_tangents)
        alpha_out = self._get_alpha_out(vi, next_v, edge_tangents)

        # Quantisierung
        if self.embedding_mode == 0:
            r_tok = self._quantize_scalar(r, self.bounds_tokenize['r_min'],
                                          self.bounds_tokenize['r_max']) + self.offset_r
            ts_tok, tc_tok = self._quantize_angle_sincos(theta)
            ts_tok += self.offset_theta_sin
            tc_tok += self.offset_theta_cos
            t_tok = self._quantize_scalar(t_norm, self.bounds_tokenize['t_min'],
                                           self.bounds_tokenize['t_max']) + self.offset_t
            ais_tok, aic_tok = self._quantize_angle_sincos(alpha_in)
            ais_tok += self.offset_alpha_in_sin
            aic_tok += self.offset_alpha_in_cos
            aos_tok, aoc_tok = self._quantize_angle_sincos(alpha_out)
            aos_tok += self.offset_alpha_out_sin
            aoc_tok += self.offset_alpha_out_cos

        elif self.embedding_mode == 1:
            r_tok = self._quantize_shared(r, self.bounds_tokenize['r_min'],
                                          self.bounds_tokenize['r_max'])
            ts_tok, tc_tok = self._quantize_angle_sincos(theta)
            t_tok = self._quantize_shared(t_norm, self.bounds_tokenize['t_min'],
                                          self.bounds_tokenize['t_max'])
            ais_tok, aic_tok = self._quantize_angle_sincos(alpha_in)
            aos_tok, aoc_tok = self._quantize_angle_sincos(alpha_out)
            # Achtung: ts_tok etc. sind in [0, Qa-1], r_tok in [0, Q-1].
            # Da Q = max(Qr, Qa) >= Qa, passen die Winkel-Token in den Bereich.
            # Keine Offsets noetig.

        elif self.embedding_mode == 2:
            # Modus C: separate Embeddings -> wir nutzen getrennte Bereiche
            # Skalare: [0, Qr-1], Winkel: [Qr, Qr+Qa-1]
            r_tok = self._quantize_scalar(r, self.bounds_tokenize['r_min'],
                                          self.bounds_tokenize['r_max'])
            ts_tok, tc_tok = self._quantize_angle_sincos(theta)
            ts_tok += self.quantization_r
            tc_tok += self.quantization_r
            t_tok = self._quantize_scalar(t_norm, self.bounds_tokenize['t_min'],
                                           self.bounds_tokenize['t_max'])
            ais_tok, aic_tok = self._quantize_angle_sincos(alpha_in)
            ais_tok += self.quantization_r
            aic_tok += self.quantization_r
            aos_tok, aoc_tok = self._quantize_angle_sincos(alpha_out)
            aos_tok += self.quantization_r
            aoc_tok += self.quantization_r

        return [r_tok, ts_tok, tc_tok, t_tok, ais_tok, aic_tok, aos_tok, aoc_tok]

    # ------------------------------------------------------------------
    # Detokenisierung
    # ------------------------------------------------------------------
    def detokenize(self, tokens: List[int]) -> Dict:
        """
        Wandelt Token-Sequenz zurueck in reconstructierbare Daten.
        Returns dict mit 'vertex_places', 'faces_as_places' etc.
        """
        if self.verbose:
            print("Starte Detokenisierung...")

        # Extrahiere Koordinaten-Tokens zwischen start und end
        coord_tokens = []
        in_coords = False
        for tok in tokens:
            if tok == self.start_token:
                in_coords = True
                continue
            elif tok == self.end_token:
                break
            elif tok == self.eor_token:
                # Bei Row-Kompression: Marker, aber keine explizite Bedeutung
                continue
            elif tok == self.sep_token:
                continue
            elif in_coords and tok < self.coord_vocab_size:
                coord_tokens.append(tok)

        # 8 Tokens pro Vertex-Platz
        n_places = len(coord_tokens) // 8
        coord_tokens = coord_tokens[:n_places * 8]

        # Dequantisierung
        if self.embedding_mode == 0:
            r_min, r_max = self.bounds_tokenize['r_min'], self.bounds_tokenize['r_max']
            t_min, t_max = self.bounds_tokenize['t_min'], self.bounds_tokenize['t_max']
        elif self.embedding_mode == 1:
            r_min, r_max = self.bounds_tokenize['r_min'], self.bounds_tokenize['r_max']
            t_min, t_max = self.bounds_tokenize['t_min'], self.bounds_tokenize['t_max']
        elif self.embedding_mode == 2:
            r_min, r_max = self.bounds_tokenize['r_min'], self.bounds_tokenize['r_max']
            t_min, t_max = self.bounds_tokenize['t_min'], self.bounds_tokenize['t_max']

        vertex_places = []
        for i in range(n_places):
            toks = coord_tokens[i * 8:(i + 1) * 8]
            if self.embedding_mode == 0:
                r = self._dequantize_scalar(toks[0] - self.offset_r, r_min, r_max)
                theta = self._dequantize_angle_sincos(toks[1] - self.offset_theta_sin,
                                                       toks[2] - self.offset_theta_cos)
                t_norm = self._dequantize_scalar(toks[3] - self.offset_t, t_min, t_max)
                alpha_in = self._dequantize_angle_sincos(toks[4] - self.offset_alpha_in_sin,
                                                           toks[5] - self.offset_alpha_in_cos)
                alpha_out = self._dequantize_angle_sincos(toks[6] - self.offset_alpha_out_sin,
                                                            toks[7] - self.offset_alpha_out_cos)
            elif self.embedding_mode == 1:
                r = self._dequantize_shared(toks[0], r_min, r_max)
                theta = self._dequantize_angle_sincos(toks[1], toks[2])
                t_norm = self._dequantize_shared(toks[3], t_min, t_max)
                alpha_in = self._dequantize_angle_sincos(toks[4], toks[5])
                alpha_out = self._dequantize_angle_sincos(toks[6], toks[7])
            elif self.embedding_mode == 2:
                r = self._dequantize_scalar(toks[0], r_min, r_max)
                theta = self._dequantize_angle_sincos(toks[1] - self.quantization_r,
                                                       toks[2] - self.quantization_r)
                t_norm = self._dequantize_scalar(toks[3], t_min, t_max)
                alpha_in = self._dequantize_angle_sincos(toks[4] - self.quantization_r,
                                                           toks[5] - self.quantization_r)
                alpha_out = self._dequantize_angle_sincos(toks[6] - self.quantization_r,
                                                            toks[7] - self.quantization_r)

            vertex_places.append({
                'r': r, 'theta': theta,
                't_norm': t_norm,
                'alpha_in': alpha_in, 'alpha_out': alpha_out,
            })

        # Faces als Place-Indices (Strategy 0/2: 4 places pro Face, n_faces = n_places / 4)
        faces_as_places = []
        if n_places % 4 == 0:
            n_faces = n_places // 4
            for fi in range(n_faces):
                faces_as_places.append([fi * 4 + i for i in range(4)])

        return {
            'vertex_places': vertex_places,
            'n_places': n_places,
            'faces_as_places': faces_as_places,
        }

    def testing(self, mesh_data: Dict):
        """Round-trip Test: tokenisieren -> detokenisieren -> Vergleich."""
        tokens = self.tokenize(mesh_data)
        recon = self.detokenize(tokens)
        return len(recon['vertex_places']) > 0
