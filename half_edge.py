try:
    import openmesh as om
except Exception:  # pragma: no cover - optional build dependency may be missing
    om = None

import numpy as np
import torch

import plotting_tools


def ensure_counter_clockwise(coords, indices):
    c = coords.mean(dim=0)
    angles = sorted(
        [(torch.atan2(coords[i][1]-c[1], coords[i][0]-c[0]).item(), i) for i in range(4)])
    return indices[[a[1] for a in angles]]


def order_quads_yx(vertices: torch.Tensor, quads: torch.Tensor) -> torch.Tensor:
    """
    Directed neighbor-first traversal with shared-edge vertex ordering.

    Traversal priority (unchanged):
      1. Face on the opposite edge (continue straight along the row).
      2. Lex-min edge-sharing neighbor (new row start).
      3. Global lex-min fallback (disconnected region).

    Vertex ordering per face (applied later in Tokenizer2D._shared_edge_vertex_order):
      - First 2 = entrance edge (vertices shared with previous face), reversed
        relative to that face's exit edge so all faces in a row share the same
        rotation direction (no CCW/CW alternation).
      - Last  2 = exit edge    (vertices shared with next face).
      → last 2 of face N reversed == first 2 of face N+1 within a row.

    Row-start: exit edge last, other 2 lex-sorted first.
    Row-end:   entrance edge first, other 2 lex-sorted last.
    Fallback:  CCW.
    """
    if om is None:
        raise RuntimeError(
            "openmesh is not installed; order_quads_yx is only available when "
            "openmesh is built. Install the 'mesh' extra or use a 3D data path."
        )
    n = quads.shape[1]
    quads_np = quads.numpy()

    # Deduplicate vertices by rounded coordinate before building the mesh,
    # so that faces sharing a geometric edge but different vertex indices
    # are correctly connected in the half-edge structure.
    v_np = np.round(vertices.numpy(), decimals=8)
    coord_to_id: dict = {}
    unique_verts = []
    remap = np.empty(len(v_np), dtype=int)
    for i, (x, y) in enumerate(v_np):
        k = (x, y)
        if k not in coord_to_id:
            coord_to_id[k] = len(unique_verts)
            unique_verts.append([x, y])
        remap[i] = coord_to_id[k]

    mesh = om.PolyMesh()
    vhs = [mesh.add_vertex(np.array([x, y, 0.0])) for x, y in unique_verts]
    for fi in range(n):
        mesh.add_face([vhs[remap[v]] for v in quads_np[:, fi]])

    def key(i):
        v = vertices[quads[:, i]]
        return tuple(sorted(zip(v[:, 1].tolist(), v[:, 0].tolist())))

    keys = [key(i) for i in range(n)]

    def opposite_neighbor(current_idx, prev_idx):
        fh = mesh.face_handle(current_idx)
        for heh in mesh.fh(fh):
            twin = mesh.opposite_halfedge_handle(heh)
            if mesh.face_handle(twin).idx() == prev_idx:
                opp = mesh.face_handle(
                    mesh.opposite_halfedge_handle(
                        mesh.next_halfedge_handle(
                            mesh.next_halfedge_handle(heh))))
                return opp.idx() if opp.is_valid() else None
        return None

    visited = [False] * n
    result = []
    prev_idx = None

    def place(idx, came_from):
        nonlocal prev_idx
        visited[idx] = True
        result.append(idx)
        prev_idx = came_from

    place(min(range(n), key=lambda i: keys[i]), None)

    while len(result) < n:
        current = result[-1]

        if prev_idx is not None:
            opp = opposite_neighbor(current, prev_idx)
            if opp is not None and not visited[opp]:
                place(opp, current)
                continue
        else:
            nbrs = [nb.idx() for nb in mesh.ff(mesh.face_handle(current))
                    if not visited[nb.idx()]]
            if nbrs:
                place(min(nbrs, key=lambda i: keys[i]), current)
                continue

        unvisited = [i for i in range(n) if not visited[i]]
        if not unvisited:
            break
        place(min(unvisited, key=lambda i: keys[i]), None)

    ordered = [ensure_counter_clockwise(vertices[quads[:, i]], quads[:, i])
               for i in result]
    return torch.stack(ordered).T
