"""Topology-canonical row plan for HexaRowTokenizer.

`hexa_row_tokenizer.build_row_plan` cuts and orders rows with geometric
thresholds (z-layer gaps, z-jump, azimuth of the next block). Measured on the
machine_0004 perturbation family (domain_partition_3D
experimentell/hex3d_algohex/analysis/row_plan_diag2.py): the
same 12-block topology gives 5-8 rows depending on sub-percent geometry
changes, and the emission-level break never fires -- all of the variation
comes from those thresholds.

This module replaces them with a plan that depends only on the labelled
block complex:

* Every block gets a *frame*: one of the 24 proper rotations of the VTK hex,
  i.e. which pair of opposite faces is the local x, y, z axis and which side
  is minus/plus. A frame on one block fixes the frame of a face neighbour
  uniquely (`_transport`): the shared face is x+ on one side and x- on the
  other, with the in-face corner correspondence preserved.
* Canonical anchor: for every (block, rotation) candidate, a BFS over face
  neighbours in fixed slot order (x-, x+, y-, y+, z-, z+) assigns indices and
  frames and writes a code: per block and slot, the boundary label (negative)
  or the BFS index of the neighbour. Rows are then built from that BFS
  (below). The candidate with the smallest (row count, code) wins. The code
  is a complete invariant of the labelled complex, so equal topologies give
  equal plans regardless of vertex positions.
* Rows are dual chords along the local x axis: take the unvisited block
  with the smallest BFS index, walk x+ through opposite faces while blocks
  are unvisited, walk x- likewise, row = backward part reversed + forward
  part. A closed chord (O-ring) starts at that block. Frames are transported
  along the row, so the exit ring of one block IS the entry ring of the next
  in the same corner order -- exactly what `HexaRowTokenizer.detokenize`
  assumes for continuation blocks.
* Only a genuine automorphism of the labelled complex (several candidates
  with the identical key) leaves a choice; it is broken geometrically and
  counted in the returned info.

Returns (rows, emit, info) where rows/emit are the `emit_override` contract
of `HexaRowTokenizer.tokenize`: emit[b] = [entry ring (4), exit ring (4)],
a valid positively oriented VTK relabelling of block b.
"""
from __future__ import annotations

import itertools

import numpy as np

# VTK hex corner -> unit-cube coordinate
_C = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
               (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)])
_IDX = {tuple(c): i for i, c in enumerate(_C)}
# slot order x-, x+, y-, y+, z-, z+  -> (axis, side)
_SLOTS = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
_SLOT_CORNERS = [[i for i in range(8) if _C[i][a] == s] for a, s in _SLOTS]
# emission: x is the row axis; ring (y,z) = (0,0),(1,0),(1,1),(0,1) is
# right-handed with x, so [entry, exit] is a positive VTK hex.
_ENTRY = [_IDX[(0, 0, 0)], _IDX[(0, 1, 0)], _IDX[(0, 1, 1)], _IDX[(0, 0, 1)]]
_EXIT = [_IDX[(1, 0, 0)], _IDX[(1, 1, 0)], _IDX[(1, 1, 1)], _IDX[(1, 0, 1)]]


def _rotations():
    """The 24 proper cube rotations as corner permutations p: new[i] = old[p[i]]."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            R = np.zeros((3, 3), dtype=int)
            for r, (c, s) in enumerate(zip(perm, signs)):
                R[r, c] = s
            if round(np.linalg.det(R)) != 1:
                continue
            p = []
            for i in range(8):
                # new corner i sits at coordinate _C[i]; it is the old corner
                # whose coordinate maps onto it under R (about the cube centre)
                x = R.T @ (_C[i] * 2 - 1)
                p.append(_IDX[tuple(((x + 1) // 2).astype(int))])
            out.append(tuple(p))
    assert len(set(out)) == 24
    return out


ROT = _rotations()


def _apply(blk, p):
    return tuple(blk[i] for i in p)


class _Complex:
    def __init__(self, blks, face_label=None):
        self.blks = [tuple(int(v) for v in b) for b in blks]
        self.by_face = {}
        for b, blk in enumerate(self.blks):
            for corners in _SLOT_CORNERS:
                self.by_face.setdefault(frozenset(blk[i] for i in corners), []).append(b)
        bad = [f for f, o in self.by_face.items() if len(o) > 2]
        if bad:
            raise ValueError(f"non-manifold block complex: {len(bad)} faces with >2 blocks")
        self.face_label = face_label or {}

    def across(self, b, face):
        o = self.by_face[face]
        return o[0] if o[1:] and o[1] == b else (o[1] if len(o) == 2 else None)

    def label(self, face):
        return int(self.face_label.get(face, 0))


def _transport(framed_a, slot, blk_b):
    """Frame of neighbour b across slot `slot` of a (framed_a = a's corners in
    its frame). Returns b's corners in the matching frame."""
    axis, side = _SLOTS[slot]
    want = {}
    for i in _SLOT_CORNERS[slot]:
        c = _C[i].copy(); c[axis] = 1 - side
        want[_IDX[tuple(c)]] = framed_a[i]
    for p in ROT:
        fb = _apply(blk_b, p)
        if all(fb[j] == v for j, v in want.items()):
            return fb
    raise ValueError("no proper frame transport (inconsistent block orientation)")


def _plan_from(cx, start, rot):
    """BFS code + rows + emit for one anchor candidate."""
    F = len(cx.blks)
    frame = {start: _apply(cx.blks[start], rot)}
    order, index, code = [start], {start: 0}, []
    q = 0
    while q < len(order):
        b = order[q]; q += 1
        fb = frame[b]
        for s, corners in enumerate(_SLOT_CORNERS):
            face = frozenset(fb[i] for i in corners)
            nb = cx.across(b, face)
            if nb is None:
                code.append(-1 - cx.label(face))
                continue
            if nb not in index:
                index[nb] = len(order); order.append(nb)
                frame[nb] = _transport(fb, s, cx.blks[nb])
            code.append(index[nb])
    if len(order) != F:
        raise ValueError(f"block complex not face-connected ({len(order)}/{F})")

    visited, rows, emit = set(), [], [None] * F

    def walk(b, fb, slot):
        out = []
        while True:
            face = frozenset(fb[i] for i in _SLOT_CORNERS[slot])
            nb = cx.across(b, face)
            if nb is None or nb in visited or nb in [x for x, _ in out]:
                return out, nb
            fb = _transport(fb, slot, cx.blks[nb])
            # walking x-: the transported frame keeps the row direction
            b = nb
            out.append((b, fb))

    for b in order:
        if b in visited:
            continue
        visited.add(b)
        fwd, stop = walk(b, frame[b], 1)
        if stop == b:                      # closed chord
            row = [(b, frame[b])] + fwd
        else:
            visited.update(x for x, _ in fwd)
            bwd, _ = walk(b, frame[b], 0)
            row = bwd[::-1] + [(b, frame[b])] + fwd
        visited.update(x for x, _ in row)
        rows.append([x for x, _ in row])
        for x, fx in row:
            emit[x] = [fx[i] for i in _ENTRY] + [fx[i] for i in _EXIT]
    return (len(rows), code), rows, emit


def build_row_plan_topo(blks, Vcart, face_label=None, return_info=True):
    """blks: [F][8] VTK-ordered, positively oriented blocks. Vcart: [M,3].
    face_label: {frozenset(4 vertex ids): int} for boundary faces (optional;
    without labels the plan is canonical for the bare complex, with more
    automorphisms). Returns (rows, emit, info)."""
    cx = _Complex(blks, face_label)
    V = np.asarray(Vcart, dtype=np.float64)
    best, cands = None, []
    for b in range(len(cx.blks)):
        for p in ROT:
            key, rows, emit = _plan_from(cx, b, p)
            if best is None or key < best:
                best, cands = key, [(rows, emit)]
            elif key == best:
                cands.append((rows, emit))

    def geo_key(c):
        rows, emit = c
        seq = [emit[r[0]] for r in rows]
        return tuple(round(float(x), 4) for blk in seq for v in blk for x in V[v])
    rows, emit = min(cands, key=geo_key)
    for e in emit:
        _check_positive(V[e])
    info = {"n_rows": best[0], "row_lens": [len(r) for r in rows],
            "n_ties": len(cands), "code": best[1]}
    return (rows, emit, info) if return_info else (rows, emit)


def _check_positive(P):
    tets = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]
    vol = sum(np.dot(np.cross(P[b] - P[a], P[c] - P[a]), P[d] - P[a]) / 6.0 for a, b, c, d in tets)
    if vol <= 0:
        raise ValueError(f"emitted block not positively oriented (vol {vol:.3g})")


def face_labels_from_npz(npz, blks, V, merge=None):
    """Boundary-face labels from sample.npz's labelled surface triangulation:
    majority label of the nearest surface triangles at the face centre and at
    four points between centre and corners. Boundary = face owned by one block.
    merge: {label: label} applied afterwards (e.g. fold patches whose boundary
    the block faces straddle)."""
    from scipy.spatial import cKDTree
    P = np.asarray(npz["surface_points"]); T = np.asarray(npz["surface_tris"])
    L = np.asarray(npz["surface_tri_label"])
    tree = cKDTree(P[T].mean(1))
    cx = _Complex(blks)
    out = {}
    for face, owners in cx.by_face.items():
        if len(owners) != 1:
            continue
        Q = V[sorted(face)]
        c = Q.mean(0)
        # centre plus points pulled 40 % towards each corner: stays inside the
        # face, away from the patch boundaries the face edges may lie on
        pts = np.vstack([c, 0.6 * c + 0.4 * Q])
        _, k = tree.query(pts)
        vals, cnt = np.unique(L[k], return_counts=True)
        lab = int(vals[np.argmax(cnt)])
        out[face] = (merge or {}).get(lab, lab)
    return out


_TETS = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]


def signed_vol(P):
    """Signed volume of a VTK hex (6-tet split)."""
    return sum(np.dot(np.cross(P[b] - P[a], P[c] - P[a]), P[d] - P[a]) / 6.0 for a, b, c, d in _TETS)


def load_npz_sample(path, labels=True, merge=None):
    """sample.npz -> dict in the format HexaRowTokenizer.tokenize expects, plus
    boundary-face labels (merge: e.g. {6: 5, 7: 5} folds the three cut interfaces)."""
    import os

    import torch

    from meshtron.data.domain_extractor_3d import to_cylindrical
    s = np.load(path, allow_pickle=True)
    V = np.asarray(s["vertices"], dtype=np.float64)
    blks = np.asarray(s["blocks"], dtype=np.int64).copy()
    for i in range(len(blks)):          # same orientation fix as clean_base_npz.py
        if signed_vol(V[blks[i]]) < 0:
            blks[i] = blks[i][::-1]
    name = os.path.basename(os.path.dirname(os.path.abspath(path)))
    fl = face_labels_from_npz(s, blks.tolist(), V, merge) if labels else None
    return {"name": name, "blocks": len(blks), "vertices_cartesian": torch.tensor(V),
            "vertices_polar": torch.tensor(to_cylindrical(V), dtype=torch.float32),
            "faces": torch.tensor(blks.T), "face_label": fl}
