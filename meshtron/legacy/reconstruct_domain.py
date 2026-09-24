"""
reconstruct_domain.py

Rekonstruktion einer Domain-Partition aus detokenisierten Daten.
Schritte:
  1. Polar -> Cartesian fuer alle Vertex-Plaetze
  2. Merge duplicate vertices (Proximity-Clustering)
  3. Baue Faces mit unique Vertex-Indices
  4. Generiere Hermite-Spline Punkte pro Kante
  5. Erstelle blocked_mesh Data-Objekt
  6. Rufe Transfinite_Interpolation auf
"""

import numpy as np
import torch
from torch_geometric.data import Data
from typing import List, Dict, Tuple
import os
import sys
_TOOLS = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'domain_partition', 'domain_partition_2D', 'tools'))
sys.path.insert(0, _TOOLS)
from transfinite_interpolation import Transfinite_Interpolation


def polar_to_cartesian(places: List[Dict], center: np.ndarray) -> np.ndarray:
    """Konvertiere polar places zu cartesian [n_places, 2]."""
    pts = []
    for p in places:
        r = p['r']
        theta = p['theta']
        x = r * np.cos(theta) + center[0]
        y = r * np.sin(theta) + center[1]
        pts.append([x, y])
    return np.array(pts, dtype=np.float32)


def merge_duplicate_vertices(cartesian: np.ndarray, threshold: float = 1e-3) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Merge nahe beieinander liegende Vertex-Plaetze.
    
    Returns:
        unique_verts: [n_unique, 2]
        place_to_unique: dict {place_idx -> unique_idx}
    """
    n = cartesian.shape[0]
    unique_verts = []
    place_to_unique = {}
    
    for i in range(n):
        found = False
        for j, uv in enumerate(unique_verts):
            if np.linalg.norm(cartesian[i] - uv) < threshold:
                place_to_unique[i] = j
                found = True
                break
        if not found:
            place_to_unique[i] = len(unique_verts)
            unique_verts.append(cartesian[i].copy())
    
    return np.array(unique_verts, dtype=np.float32), place_to_unique


def hermite_spline_2d(P0: np.ndarray, P1: np.ndarray,
                      T0: np.ndarray, T1: np.ndarray,
                      n_points: int = 50) -> np.ndarray:
    """
    Hermite-Spline in 2D.
    P0, P1: Start/End Punkt [2]
    T0, T1: Tangentenvektoren [2]
    Returns: [n_points, 2]
    """
    t = np.linspace(0, 1, n_points)
    h00 = 2*t**3 - 3*t**2 + 1
    h10 = t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 = t**3 - t**2
    
    points = (h00[:, None] * P0 +
              h10[:, None] * T0 +
              h01[:, None] * P1 +
              h11[:, None] * T1)
    return points.astype(np.float32)


def quadratic_bezier_2d(P0: np.ndarray, P2: np.ndarray,
                        dir0: np.ndarray, dir2: np.ndarray,
                        n_points: int = 50,
                        straight_deg: float = 3.0) -> np.ndarray:
    """
    Quadratischer Bezier durch P0, P2 mit Tangenten-RICHTUNGEN dir0 (bei P0),
    dir2 (bei P2). Kontrollpunkt = Schnittpunkt der beiden Tangentenlinien.

    Vorteil ggü. kubischem Hermite: keine Tangenten-MAGNITUDE noetig -> kein
    Overshoot/Bulge. Eine quadratische Kurve hat zudem keinen Wendepunkt,
    passt zu monoton gekruemmten Block-/Profilkanten.

    dir0, dir2 muessen nicht normiert sein; nur die Richtung zaehlt.
    Bei nahezu parallelen Tangenten (Turn < straight_deg) -> Gerade.
    """
    t = np.linspace(0, 1, n_points)[:, None]
    P0 = np.asarray(P0, float); P2 = np.asarray(P2, float)
    d0 = np.asarray(dir0, float); d2 = np.asarray(dir2, float)
    n0 = np.linalg.norm(d0); n2 = np.linalg.norm(d2)
    if n0 < 1e-12 or n2 < 1e-12:
        return ((1 - t) * P0 + t * P2).astype(np.float32)
    d0 = d0 / n0; d2 = d2 / n2

    # Turn-Winkel zwischen den Tangentenlinien (Richtung egal -> auf [0,90] falten)
    cosang = abs(float(np.clip(np.dot(d0, d2), -1.0, 1.0)))
    turn = np.degrees(np.arccos(cosang))  # 0 = parallel/anti-parallel
    if turn < straight_deg:
        return ((1 - t) * P0 + t * P2).astype(np.float32)

    # Schnittpunkt: P0 + s*d0 = P2 + u*d2  ->  [d0, -d2] [s,u]^T = P2 - P0
    A = np.array([[d0[0], -d2[0]], [d0[1], -d2[1]]])
    b = P2 - P0
    try:
        s, u = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return ((1 - t) * P0 + t * P2).astype(np.float32)
    C = P0 + s * d0

    # Sicherung gegen entartete Schnittpunkte (Tangenten zeigen "weg" oder
    # schneiden weit hinter den Endpunkten): Kontrollpunkt muss nahe der Sehne
    # liegen. Ein sinnvoller Bogen hat Offset < ~1 Sehnenlaenge; alles darueber
    # ist ein Artefakt (falsche/mehrdeutige Tangente) -> Gerade.
    chord = np.linalg.norm(P2 - P0)
    if np.linalg.norm(C - 0.5 * (P0 + P2)) > 1.0 * max(chord, 1e-9):
        return ((1 - t) * P0 + t * P2).astype(np.float32)

    B = (1 - t) ** 2 * P0 + 2 * (1 - t) * t * C + t ** 2 * P2
    return B.astype(np.float32)


def build_hermite_edges(places: List[Dict], cartesian: np.ndarray,
                        faces_as_places: List[List[int]],
                        place_to_unique: Dict[int, int]) -> Tuple[torch.Tensor, List[np.ndarray]]:
    """
    Baue unique edges und Hermite-Spline Punkte.
    
    Returns:
        edge_index: [2, n_unique_edges]
        edge_points: Liste von [n_points, 2] Arrays
    """
    # Sammle alle directed edges aus den Faces
    directed_edges = []
    directed_edge_info = []  # (place0, place1, alpha_out, alpha_in, t_norm)
    
    for face in faces_as_places:
        for i in range(4):
            p0 = face[i]
            p1 = face[(i + 1) % 4]
            
            alpha_out = places[p0]['alpha_out']
            alpha_in = places[p1]['alpha_in']
            t_norm = places[p0]['t_norm']
            
            u = place_to_unique[p0]
            v = place_to_unique[p1]
            
            directed_edges.append((u, v))
            directed_edge_info.append({
                'P0': cartesian[p0],
                'P1': cartesian[p1],
                'alpha_out': alpha_out,
                'alpha_in': alpha_in,
                't_norm': t_norm,
            })
    
    # Geometrie-basiertes Dedup:
    #   Zwei gerichtete Half-Edges werden nur gemerged, wenn sie DIESELBE
    #   physikalische Kurve beschreiben. Kriterium: gleiches Endpunkt-Paar UND
    #   Kurven-Mittelpunkt (Hermite t=0.5) innerhalb einer Chord-relativen Toleranz.
    #   -> Innenkanten (von 2 Bloecken geteilt) verschmelzen wie bisher.
    #   -> Parallele Kanten (z.B. Druck-/Saugseite LE<->TE) bleiben GETRENNT,
    #      da ihre Mittelpunkte auf gegenueberliegenden Seiten der Sehne liegen.
    MERGE_TOL_FRAC = 0.05  # 5% der Sehnenlaenge

    def _hermite_mid(info):
        P0, P1 = info['P0'], info['P1']
        T0 = info['t_norm'] * np.array([np.cos(info['alpha_out']), np.sin(info['alpha_out'])])
        T1 = info['t_norm'] * np.array([np.cos(info['alpha_in']), np.sin(info['alpha_in'])])
        # Hermite bei t=0.5: h00=.5 h10=.125 h01=.5 h11=-.125
        return 0.5 * P0 + 0.125 * T0 + 0.5 * P1 - 0.125 * T1

    unique_list = []  # jede Gruppe = ein Dict wie zuvor + 'pair', 'mid'
    for (u, v), info in zip(directed_edges, directed_edge_info):
        pair = tuple(sorted([u, v]))
        mid = _hermite_mid(info)
        chord = float(np.linalg.norm(info['P1'] - info['P0']))
        tol = MERGE_TOL_FRAC * max(chord, 1e-6)

        matched = None
        for g in unique_list:
            if g['pair'] == pair and np.linalg.norm(mid - g['mid']) < tol:
                matched = g
                break

        if matched is None:
            unique_list.append({
                'pair': pair, 'mid': mid,
                'u': u, 'v': v,
                'P0': info['P0'], 'P1': info['P1'],
                'alpha_out': info['alpha_out'],
                'alpha_in': info['alpha_in'],
                't_norm': info['t_norm'],
                'count': 1,
            })
        else:
            # Gleiche physische Kante, aber diese Half-Edge ist die Gegenrichtung.
            # NICHT mitteln: alpha_out/alpha_in der beiden Richtungen sitzen an
            # verschiedenen Endpunkten und zeigen entgegengesetzt -> Mittelung
            # begradigt die Kurve faelschlich. Erste Half-Edge behalten; fuer den
            # Nachbarblock wird die Kurve in face_edge_curve nur umgedreht (konform).
            matched['count'] += 1

    n_edges = len(unique_list)
    edge_index = torch.zeros((2, n_edges), dtype=torch.long)
    edge_points = []

    for i, info in enumerate(unique_list):
        u, v = info['u'], info['v']
        edge_index[0, i] = u
        edge_index[1, i] = v
        
        P0 = info['P0']
        P1 = info['P1']
        
        # Quadratischer Bezier: nur Tangenten-RICHTUNGEN, keine Magnitude.
        # dir0 = Tangente die P0 in Richtung P1 verlaesst (alpha_out).
        # dir2 = Tangentenrichtung bei P1 (alpha_in, Reiserichtung ankommend);
        #        fuer den Schnittpunkt zaehlt nur die Linienrichtung.
        dir0 = np.array([np.cos(info['alpha_out']), np.sin(info['alpha_out'])])
        dir2 = np.array([np.cos(info['alpha_in']), np.sin(info['alpha_in'])])

        pts = quadratic_bezier_2d(P0, P1, dir0, dir2, n_points=50)
        edge_points.append(pts)
    
    return edge_index, edge_points


def _resample_curve(c: np.ndarray, n: int) -> np.ndarray:
    """Kurve auf n Punkte nach Bogenlaenge resamplen (numpy-only, linear)."""
    c = np.asarray(c, float)
    if len(c) == n:
        return c
    if len(c) < 2:
        return np.tile(c[0], (n, 1))
    seg = np.linalg.norm(np.diff(c, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-12:
        return np.tile(c[0], (n, 1))
    s /= s[-1]
    snew = np.linspace(0, 1, n)
    return np.stack([np.interp(snew, s, c[:, 0]),
                     np.interp(snew, s, c[:, 1])], axis=1)


def coons_patch_2d(e01: np.ndarray, e32: np.ndarray,
                   e03: np.ndarray, e12: np.ndarray, n: int) -> np.ndarray:
    """
    Bilineare Coons/Transfinite-Interpolation eines Quad-Blocks aus 4 Randkurven.
    Konvention (2D-Port von interpolate_quad_surface, domain_partition_3D):
        e01: Ecke0->Ecke1 (u-Richtung, v=0)
        e32: Ecke3->Ecke2 (u-Richtung, v=1)  (gegenueber e01, gleiche Richtung)
        e03: Ecke0->Ecke3 (v-Richtung, u=0)
        e12: Ecke1->Ecke2 (v-Richtung, u=1)
    Returns: Gitter [n, n, 2]  (v-Index, u-Index).
    """
    C0 = _resample_curve(e01, n); C1 = _resample_curve(e32, n)
    D0 = _resample_curve(e03, n); D1 = _resample_curve(e12, n)
    u = np.linspace(0, 1, n); v = np.linspace(0, 1, n)
    uu, vv = np.meshgrid(u, v)
    P00, P10, P01, P11 = C0[0], C0[-1], C1[0], C1[-1]
    surf = (
        (1 - vv[:, :, None]) * C0[None, :, :]
        + vv[:, :, None] * C1[None, :, :]
        + (1 - uu[:, :, None]) * D0[:, None, :]
        + uu[:, :, None] * D1[:, None, :]
        - (1 - uu[:, :, None]) * (1 - vv[:, :, None]) * P00
        - uu[:, :, None] * (1 - vv[:, :, None]) * P10
        - (1 - uu[:, :, None]) * vv[:, :, None] * P01
        - uu[:, :, None] * vv[:, :, None] * P11
    )
    return surf


def reconstruct_domain_coons(tokenizer_output: Dict, center: np.ndarray,
                             n: int = 11, merge_tol: float = 1e-4) -> Data:
    """
    Rekonstruktion via per-Block Coons-TFI (Methode aus domain_partition_3D).

    Jeder Block wird UNABHAENGIG aus seinen 4 (rekonstruierten) Randkurven zu
    einem n x n Gitter interpoliert. Die Airfoil-Linse ist kein Block -> wird
    nie gemeshed -> Loch bleibt automatisch hohl (kein gmsh, kein addPlaneSurface).

    Geteilte Kanten nutzen DIESELBE deduped Kurve (Konformitaet).
    """
    from collections import defaultdict
    places = tokenizer_output['vertex_places']
    faces_as_places = tokenizer_output['faces_as_places']
    cart = polar_to_cartesian(places, center)
    _, p2u = merge_duplicate_vertices(cart, threshold=1e-3)

    # Deduped Kantenkurven (quadratische Beziers, geometrie-basiertes Dedup)
    ei, ep = build_hermite_edges(places, cart, faces_as_places, p2u)
    ded = defaultdict(list)
    for i in range(ei.shape[1]):
        u, v = int(ei[0, i]), int(ei[1, i])
        c = np.asarray(ep[i], float)
        ded[frozenset((u, v))].append(c)

    def face_edge_curve(p0, p1):
        """Deduped Kurve fuer Face-Kante p0->p1, orientiert p0->p1."""
        a, b = p2u[p0], p2u[p1]
        # eigene Quadratik als Referenz-Mittelpunkt (disambiguiert Parallelkanten)
        d0 = np.array([np.cos(places[p0]['alpha_out']), np.sin(places[p0]['alpha_out'])])
        d2 = np.array([np.cos(places[p1]['alpha_in']), np.sin(places[p1]['alpha_in'])])
        own = quadratic_bezier_2d(cart[p0], cart[p1], d0, d2, n)
        ownmid = own[len(own) // 2]
        cands = ded.get(frozenset((a, b)), [own])
        c = min(cands, key=lambda cc: np.linalg.norm(cc[len(cc) // 2] - ownmid))
        c = np.asarray(c, float)
        if np.linalg.norm(c[0] - cart[p0]) > np.linalg.norm(c[-1] - cart[p0]):
            c = c[::-1]
        return _resample_curve(c, n)

    # Node-Merge ueber gerundete Koordinaten
    node_map = {}
    nodes = []

    def node_id(pt):
        key = (round(float(pt[0]) / merge_tol), round(float(pt[1]) / merge_tol))
        if key not in node_map:
            node_map[key] = len(nodes)
            nodes.append([float(pt[0]), float(pt[1])])
        return node_map[key]

    quads = []
    for face in faces_as_places:
        # 4 Ecken-Places in Reihenfolge; Kanten: e01,e12,e23,e30
        e01 = face_edge_curve(face[0], face[1])
        e12 = face_edge_curve(face[1], face[2])
        e23 = face_edge_curve(face[2], face[3])
        e30 = face_edge_curve(face[3], face[0])
        # Coons-Konvention: e32 = reverse(e23) (3->2), e03 = reverse(e30) (0->3)
        grid = coons_patch_2d(e01, e23[::-1], e30[::-1], e12, n)  # [n,n,2]
        ids = np.empty((n, n), dtype=np.int64)
        for iv in range(n):
            for iu in range(n):
                ids[iv, iu] = node_id(grid[iv, iu])
        for iv in range(n - 1):
            for iu in range(n - 1):
                quads.append([ids[iv, iu], ids[iv, iu + 1],
                              ids[iv + 1, iu + 1], ids[iv + 1, iu]])

    x = torch.tensor(nodes, dtype=torch.float32)
    faces_t = torch.tensor(quads, dtype=torch.long).T  # [4, n_quads]
    return Data(x=x, faces=faces_t)


def reconstruct_blocked_mesh(tokenizer_output: Dict, center: np.ndarray) -> Data:
    """
    Erstelle blocked_mesh Data-Objekt aus detokenisierten Daten.
    
    Args:
        tokenizer_output: dict mit 'vertex_places', 'faces_as_places', 'n_places'
        center: [2] array (Mesh-Zentrum)
    
    Returns:
        blocked_mesh: Data mit x, faces, edge_subdomain_index, edge_subdomain_points
    """
    places = tokenizer_output['vertex_places']
    faces_as_places = tokenizer_output['faces_as_places']
    
    # 1. Polar -> Cartesian
    cartesian = polar_to_cartesian(places, center)
    
    # 2. Merge duplicates
    unique_verts, place_to_unique = merge_duplicate_vertices(cartesian, threshold=1e-3)
    
    # 3. Baue Faces mit unique indices
    n_faces = len(faces_as_places)
    faces = torch.zeros((4, n_faces), dtype=torch.long)
    for fi, face in enumerate(faces_as_places):
        for i, p in enumerate(face):
            faces[i, fi] = place_to_unique[p]
    
    # 4. Hermite-Splines fuer Kanten
    edge_index, edge_points = build_hermite_edges(places, cartesian, faces_as_places, place_to_unique)
    
    # 5. Erstelle Data-Objekt
    # x muss [n_verts, 3] sein? Transfinite_Interpolation nimmt pt[0], pt[1], 0
    # Also [n_verts, 2] reicht, weil es nur pt[0], pt[1] liest
    x = torch.from_numpy(unique_verts).float()
    
    blocked_mesh = Data(x=x, faces=faces,
                        edge_subdomain_index=edge_index,
                        edge_subdomain_points=edge_points)
    
    return blocked_mesh


def reconstruct_domain(tokenizer_output: Dict, center: np.ndarray,
                         transfinite_divisions: int = 10) -> Data:
    """
    Vollstaendige Rekonstruktion: blocked_mesh -> transfinite interpolation.
    
    Returns:
        quad_mesh: Data mit x, faces, edge_index
    """
    blocked_mesh = reconstruct_blocked_mesh(tokenizer_output, center)
    
    # Transfinite Interpolation
    # Hinweis: Transfinite_Interpolation.__init__ ruft bereits generate() und get_mesh() auf.
    # Das Ergebnis ist in interpolator.quad_mesh gespeichert.
    interpolator = Transfinite_Interpolation(blocked_mesh, mesh_size=0.4)
    quad_mesh = interpolator.quad_mesh
    
    return quad_mesh


if __name__ == '__main__':
    _HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _HERE)
    from tokenizer_domain import DomainTokenizer

    # Test round-trip
    data = torch.load(os.path.join(_HERE, 'domain_data.pt'), weights_only=False)
    mesh = data[0]
    
    tok = DomainTokenizer(quantization_r=64, quantization_a=32,
                           sorting_strategy=0, embedding_mode=0, verbose=False)
    tokens = tok.tokenize(mesh)
    output = tok.detokenize(tokens)
    
    center = mesh['center'].numpy()
    print(f"Center: {center}")
    print(f"Places: {output['n_places']}")
    print(f"Faces: {len(output['faces_as_places'])}")
    
    blocked = reconstruct_blocked_mesh(output, center)
    print(f"Blocked mesh: n_verts={blocked.x.shape[0]}, n_faces={blocked.faces.shape[1]}, n_edges={blocked.edge_subdomain_index.shape[1]}")
    
    # Transfinite interpolation
    try:
        quad = reconstruct_domain(output, center, transfinite_divisions=5)
        print(f"Quad mesh: n_verts={quad.x.shape[0]}, n_faces={quad.faces.shape[1]}")
    except Exception as e:
        print(f"Transfinite interpolation failed: {e}")
        import traceback
        traceback.print_exc()
