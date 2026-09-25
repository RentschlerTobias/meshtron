"""polytron_blocks.py -- curvilinear hex block structures as PolyGen sequences.

The Polytron (after PolyGen, arXiv 2002.10880) splits a block structure into
three things a model can each learn on its own:

  vertices  every distinct block corner once, quantised xyz, sorted (z, y, x)
  blocks    every hex as 8 POINTERS into that vertex list, canonically rotated
            and sorted -- topology is exact by construction, no corner can drift
            away from the corner it shares with the neighbouring block
  curves    every undirected block edge (a < b) carries a cubic Bezier; its two
            inner control points are stored as offsets from the straight-line
            third points, divided by the chord length, and companded before
            quantisation because most edges are nearly straight

That third part is what the row-token Quadtron cannot express: its blocks have
straight edges, and the edge shape is the lever that decides whether the
transfinite fill folds (the blocking's own polylines give 9 folded cells where
routed geodesics give 139, same corners). `edge_ctrl` in every sample.npz
already carries the Bezier; its median relative Hausdorff error against the
exported polylines is 0.3 %, the chord's is 2.6 %.

Everything here is numpy; the torch side lives in `meshtron.model.polytron`.
"""
from __future__ import annotations

import itertools
import json
import os
from dataclasses import dataclass

import numpy as np

# VTK hex corner order, identical to tfi.CORNER and edge_curves.CORNERS.
CORNERS = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
           (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))
_CIDX = {c: i for i, c in enumerate(CORNERS)}
HEX_EDGES = tuple((i, j) for i in range(8) for j in range(i + 1, 8)
                  if sum(a != b for a, b in zip(CORNERS[i], CORNERS[j])) == 1)

# Patch labels of the npz surfaces (checked on the geometry: 5 sits at the hub
# radius, 6 at the shroud, 7 spans hub to shroud around the blade row):
# 1 inlet, 2 outlet, 3/4 periodic, 5 hub, 6 shroud, 7 O-grid band.
# conditioning.BLADE_LABEL = 5 is the HUB here; the first Polytron checkpoints
# oversampled it by mistake (spec.weight_label=5 reproduces them).
OGRID_LABEL = 7
N_LABELS = 7


def _hex_rotations() -> np.ndarray:
    """The 24 orientation-preserving symmetries of the cube as corner
    permutations: `block[perm]` is the same hex, relabelled."""
    out = []
    for axes in itertools.permutations(range(3)):
        for flips in itertools.product((0, 1), repeat=3):
            M = np.zeros((3, 3))
            for r, (a, f) in enumerate(zip(axes, flips)):
                M[r, a] = -1 if f else 1
            if np.linalg.det(M) < 0:
                continue
            perm = []
            for c in CORNERS:
                v = np.array(c) * 2 - 1
                w = M @ v
                perm.append(_CIDX[tuple(int(x) for x in (w + 1) // 2)])
            out.append(perm)
    P = np.asarray(out, dtype=np.int64)
    assert len(P) == 24 and len({tuple(p) for p in P}) == 24
    return P


HEX_ROT = _hex_rotations()


def canonical_block(b: np.ndarray) -> np.ndarray:
    """Lexicographically smallest of the 24 rotations of one hex. Corner 0 is
    then the block's smallest vertex id -- the PolyGen rule "start each face
    at its lowest index", lifted to hexahedra."""
    cands = np.asarray(b)[HEX_ROT]
    order = np.lexsort(cands.T[::-1])
    return cands[order[0]]


def block_edges(blocks: np.ndarray) -> np.ndarray:
    """Unique undirected block edges (a < b), sorted."""
    e = set()
    for b in blocks:
        for i, j in HEX_EDGES:
            u, v = int(b[i]), int(b[j])
            e.add((u, v) if u < v else (v, u))
    return np.asarray(sorted(e), dtype=np.int64).reshape(-1, 2)


# --------------------------------------------------------------------------
# quantisation
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PolytronSpec:
    """Everything needed to go from real coordinates to tokens and back.
    Stored in every dataset and every checkpoint, so a checkpoint can never be
    decoded with bounds it was not trained with."""
    lo: tuple                       # xyz lower bound
    hi: tuple                       # xyz upper bound
    q_vert: int = 512               # bins per coordinate
    q_curve: int = 256              # bins per control-point offset component
    curve_max: float = 2.0          # |offset| clip, in chord lengths
    curve_mu: float = 50.0          # mu-law companding strength
    seam_cloud: bool = False        # cloud always carries every seam point
    weight_label: int = 5           # patch drawn `blade_weight` x as often

    def to_json(self) -> dict:
        return dict(lo=list(self.lo), hi=list(self.hi), q_vert=self.q_vert,
                    q_curve=self.q_curve, curve_max=self.curve_max,
                    curve_mu=self.curve_mu, seam_cloud=self.seam_cloud,
                    weight_label=self.weight_label)

    @staticmethod
    def from_json(d: dict) -> "PolytronSpec":
        return PolytronSpec(lo=tuple(d["lo"]), hi=tuple(d["hi"]),
                            q_vert=int(d["q_vert"]), q_curve=int(d["q_curve"]),
                            curve_max=float(d["curve_max"]),
                            curve_mu=float(d["curve_mu"]),
                            seam_cloud=bool(d.get("seam_cloud", False)),
                            weight_label=int(d.get("weight_label", 5)))

    # vertices ---------------------------------------------------------------
    def quant_xyz(self, X: np.ndarray) -> np.ndarray:
        lo, hi = np.asarray(self.lo), np.asarray(self.hi)
        u = (np.asarray(X, float) - lo) / (hi - lo)
        return np.clip(np.round(u * (self.q_vert - 1)), 0,
                       self.q_vert - 1).astype(np.int64)

    def dequant_xyz(self, Q: np.ndarray) -> np.ndarray:
        lo, hi = np.asarray(self.lo), np.asarray(self.hi)
        return lo + np.asarray(Q, float) / (self.q_vert - 1) * (hi - lo)

    def norm_xyz(self, X: np.ndarray) -> np.ndarray:
        """Real xyz -> [-1, 1] per axis (point-cloud features)."""
        lo, hi = np.asarray(self.lo), np.asarray(self.hi)
        return 2.0 * (np.asarray(X, float) - lo) / (hi - lo) - 1.0

    # curves -----------------------------------------------------------------
    def _compand(self, x):
        m, mu = self.curve_max, self.curve_mu
        x = np.clip(np.asarray(x, float), -m, m)
        return np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu * m)

    def _expand(self, y):
        m, mu = self.curve_max, self.curve_mu
        y = np.asarray(y, float)
        return np.sign(y) * np.expm1(np.abs(y) * np.log1p(mu * m)) / mu

    def quant_curve(self, off: np.ndarray) -> np.ndarray:
        y = self._compand(off)
        return np.clip(np.round((y + 1) / 2 * (self.q_curve - 1)), 0,
                       self.q_curve - 1).astype(np.int64)

    def dequant_curve(self, q: np.ndarray) -> np.ndarray:
        y = np.asarray(q, float) / (self.q_curve - 1) * 2 - 1
        return self._expand(y)


def ctrl_to_offsets(P0, P1, B1, B2) -> np.ndarray:
    """Bezier inner control points -> [.., 6] chord-relative offsets."""
    P0, P1, B1, B2 = (np.asarray(a, float) for a in (P0, P1, B1, B2))
    L = np.linalg.norm(P1 - P0, axis=-1, keepdims=True)
    L = np.where(L < 1e-12, 1.0, L)
    d1 = (B1 - (2 * P0 + P1) / 3) / L
    d2 = (B2 - (P0 + 2 * P1) / 3) / L
    return np.concatenate([d1, d2], axis=-1)


def offsets_to_ctrl(P0, P1, off):
    P0, P1, off = (np.asarray(a, float) for a in (P0, P1, off))
    L = np.linalg.norm(P1 - P0, axis=-1, keepdims=True)
    B1 = (2 * P0 + P1) / 3 + off[..., :3] * L
    B2 = (P0 + 2 * P1) / 3 + off[..., 3:] * L
    return B1, B2


def bezier(P0, P1, B1, B2, n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[:, None]
    return ((1 - t) ** 3 * P0 + 3 * (1 - t) ** 2 * t * B1
            + 3 * (1 - t) * t ** 2 * B2 + t ** 3 * P1)


# --------------------------------------------------------------------------
# sample <-> sequences
# --------------------------------------------------------------------------

@dataclass
class PolySeq:
    """One block structure in Polytron form (all integer arrays)."""
    vq: np.ndarray        # [M,3]  quantised xyz, sorted (z, y, x)
    blocks: np.ndarray    # [B,8]  pointers into vq, canonical, sorted
    edges: np.ndarray     # [E,2]  a < b, sorted
    cq: np.ndarray        # [E,6]  quantised curve offsets for edge a -> b
    src: np.ndarray | None = None   # [M] raw vertex id of each sorted vertex


def encode_sample(V: np.ndarray, B: np.ndarray, E: np.ndarray,
                  ctrl: np.ndarray, spec: PolytronSpec) -> PolySeq:
    """Raw npz arrays -> PolySeq. `E`/`ctrl` are the directed edge records of
    export_sample (both directions present)."""
    V = np.asarray(V, float)
    vq = spec.quant_xyz(V)
    # sort (z, y, x); ties keep the original order -- distinct corners that
    # share a quantisation cell stay distinct vertices
    order = np.lexsort((vq[:, 0], vq[:, 1], vq[:, 2]))
    new_of = np.empty(len(V), np.int64)
    new_of[order] = np.arange(len(V))
    blocks = np.stack([canonical_block(new_of[b]) for b in np.asarray(B)])
    blocks = blocks[np.lexsort(blocks.T[::-1])]
    edges = block_edges(blocks)
    ctrl_of = {}
    for (a, b), (c1, c2) in zip(np.asarray(E), np.asarray(ctrl)):
        ctrl_of[(int(new_of[a]), int(new_of[b]))] = (c1, c2)
    Vs = V[order]
    off = np.zeros((len(edges), 6))
    for k, (a, b) in enumerate(edges):
        c = ctrl_of.get((int(a), int(b)))
        if c is None:                     # no record: straight edge
            continue
        off[k] = ctrl_to_offsets(Vs[a], Vs[b], c[0], c[1])
    return PolySeq(vq=vq[order], blocks=blocks, edges=edges,
                   cq=spec.quant_curve(off), src=order)


def decode_seq(seq: PolySeq, spec: PolytronSpec, n_edge_pts: int = 17):
    """PolySeq -> (V [M,3] float, blocks [B,8], curves {(a,b): [n,3]})."""
    V = spec.dequant_xyz(seq.vq)
    curves = {}
    off = spec.dequant_curve(seq.cq)
    for (a, b), o in zip(seq.edges, off):
        B1, B2 = offsets_to_ctrl(V[a], V[b], o)
        curves[(int(a), int(b))] = bezier(V[a], V[b], B1, B2, n_edge_pts)
    return V, np.asarray(seq.blocks, np.int64), curves


# --------------------------------------------------------------------------
# conditioning cloud: xyz + patch labels, from the labelled surface
# --------------------------------------------------------------------------

def point_labels(n_points: int, tris: np.ndarray, tri_label: np.ndarray) -> np.ndarray:
    """[P, N_LABELS] multi-hot: which patches each surface point touches. A
    point on a seam carries both labels -- exactly where corners sit."""
    M = np.zeros((int(n_points), N_LABELS), dtype=np.uint8)
    tris = np.asarray(tris, np.int64)
    lab = np.asarray(tri_label, np.int64) - 1
    for c in range(3):
        M[tris[:, c], lab] = 1
    return M


def build_cloud(surface_points: np.ndarray, labels: np.ndarray, n: int,
                spec: PolytronSpec, rng, blade_weight: float = 3.0) -> np.ndarray:
    """[n, 3 + N_LABELS] float32: normalised xyz + multi-hot labels. Points
    of patch `spec.weight_label` (7, the O-grid band around the blade, in
    current checkpoints) are drawn `blade_weight` times as often: that band is
    where the block structure is decided."""
    P = np.asarray(surface_points, float)
    w = np.where(labels[:, spec.weight_label - 1] > 0, blade_weight, 1.0)
    seam = np.zeros(0, np.int64)
    if spec.seam_cloud:
        # Block corners sit where patches meet (median 0.015 from a seam
        # point), but seams are 4 % of the surface points and all but absent
        # from a random draw. Take every seam point (at most half the cloud),
        # the rest as before.
        seam = np.where(labels.sum(1) >= 2)[0]
        if len(seam) > n // 2:
            seam = seam[np.linspace(0, len(seam) - 1, n // 2).astype(np.int64)]
    idx = rng.choice(len(P), size=n - len(seam), replace=True, p=w / w.sum())
    idx = np.concatenate([seam, idx])
    return np.concatenate([spec.norm_xyz(P[idx]), labels[idx].astype(float)],
                          axis=1).astype(np.float32)


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------

def load_npz(path: str) -> dict:
    with np.load(path, allow_pickle=True) as z:
        return {k: np.asarray(z[k]) for k in
                ("vertices", "blocks", "edges", "edge_ctrl", "edge_polyline",
                 "edge_polyline_offset", "surface_points", "surface_tris",
                 "surface_tri_label")}


def fit_spec(vertex_sets, pad: float = 0.02, **kw) -> PolytronSpec:
    V = np.vstack(vertex_sets)
    lo, hi = V.min(0), V.max(0)
    span = hi - lo
    return PolytronSpec(lo=tuple((lo - pad * span).tolist()),
                        hi=tuple((hi + pad * span).tolist()), **kw)


def make_item(raw: dict, spec: PolytronSpec, meta: dict) -> dict:
    seq = encode_sample(raw["vertices"], raw["blocks"], raw["edges"],
                        raw["edge_ctrl"], spec)
    lab = point_labels(len(raw["surface_points"]), raw["surface_tris"],
                       raw["surface_tri_label"])
    return dict(meta, vq=seq.vq, blocks=seq.blocks, edges=seq.edges,
                cq=seq.cq, src=seq.src, n_blocks=int(len(seq.blocks)),
                surface_points=raw["surface_points"].astype(np.float32),
                point_labels=lab)


def item_seq(it: dict) -> PolySeq:
    return PolySeq(vq=it["vq"], blocks=it["blocks"], edges=it["edges"],
                   cq=it["cq"])


def selected_runs(root: str) -> list[dict]:
    sel = json.load(open(os.path.join(root, "data", "family_selection.json")))
    runs = [r for r in sel["runs"] if r["select"]]
    runs.sort(key=lambda r: r["dir"])
    return runs
