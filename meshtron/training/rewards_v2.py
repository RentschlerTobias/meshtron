"""rewards_v2.py -- prototype of a revised GRPO reward for block-structure generation.

Motivation (see scripts/reward_probes/, reports/grpo_reward_analysis.md):
  * r_conform (symmetric Chamfer / bbox diagonal) is saturated: GT 0.964, a mesh of a
    DIFFERENT geometry 0.962. It cannot rank rollouts inside a group.
  * r_quality uses the unscaled min det(J): it scales with size^3 and rewards big blocks.
  * The strict validity gate requires the GT block count; with the relabelled data
    (canonical 12-block topology) and the block-count-free v2 model this is the wrong gate.

Structure:  total = gate * (w_fid * R_fid + w_lab * R_lab + w_q * R_q + w_mesh * R_mesh) / sum(w)
  gate    0 for garbage (handled by the caller), else 1 if validate_generated_mesh passes
          WITHOUT the GT block count, else soft_gate * share of positive-volume blocks
          (halved if non-manifold). Corner scaled Jacobians are NOT a gate: coarse GT
          blocks on curved surfaces often have a negative corner value.
  R_fid   geometry fidelity in LOCAL spacing h (median incident edge length):
            corners  : boundary corners -> surface,       exp(-(d / (tau h))^2)
            coverage : surface samples -> distance to the boundary quads, within cov_tol * h
                       (labels weighted; the blade label counts blade_factor times)
  R_lab   label correctness: a boundary face should lie on ONE surface patch; share of
          boundary faces whose corners and centre all have the same nearest-surface label
          (blade = label 7 in the v2 data; is_blade in the older family data). Corner labels
          are label SETS within reach, so corners on patch edges do not count as wrong.
          Face-centre distances are only reported (detail['mid']): at block level they are
          dominated by chord sag on curved patches -- that belongs to the post-curve term.
  R_q     block quality: p5 of the scaled Jacobian over all block corners, mapped
          clip(p5 / sj_target, 0, 1) -- scale-invariant, tail-sensitive
  R_mesh  optional post-TFI mesh quality (checkMesh-style, meshtron.geometry.tfi_mesh_quality),
          passed in as a callable;
          weight is dropped if None

All terms are in [0, 1]. Pure numpy + scipy (cKDTree).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.spatial import cKDTree

from meshtron.geometry.mesh_validation import hex_signed_volumes, validate_generated_mesh
from meshtron.geometry.tfi_mesh_quality import CORNER_NB, HEX_FACES, scaled_jacobian_corners

# Same local vertex order as meshtron.geometry.mesh_validation (VTK hexahedron).
HEX_FACES = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
# For every corner: its three neighbours along the local u, v, w directions.
CORNER_NB = ((1, 3, 4), (2, 0, 5), (3, 1, 6), (0, 2, 7),
             (7, 5, 0), (4, 6, 1), (5, 7, 2), (6, 4, 3))


@dataclass(frozen=True, slots=True)
class RewardV2Config:
    w_fid: float = 0.45
    w_lab: float = 0.15
    w_q: float = 0.25
    w_mesh: float = 0.15
    tau: float = 0.1          # fidelity kernel width, in units of local spacing h
    cov_tol: float = 0.15     # coverage tolerance, in units of local spacing h
    blade_label: int = 7
    blade_factor: float = 2.0
    sj_target: float = 0.4    # p5 scaled Jacobian that earns full R_q (GT median ~0.3)
    soft_gate: float = 0.3
    n_cov: int = 4000         # surface samples for the coverage term


@dataclass(frozen=True, slots=True)
class RewardV2Terms:
    total: float
    gate: float
    r_fid: float
    r_lab: float
    r_q: float
    r_mesh: float | None
    detail: dict


def scaled_jacobian_corners(cells: np.ndarray) -> np.ndarray:
    """[N,8,3] -> [N,8] scaled Jacobian at the corners (det of the unit edge triad)."""
    nb = np.asarray(CORNER_NB)
    e = cells[:, nb, :] - cells[:, :, None, :]                      # [N,8,3,3]
    n = np.linalg.norm(e, axis=-1, keepdims=True)
    e = e / np.maximum(n, 1e-30)
    return np.linalg.det(e)


def boundary_faces(blocks: np.ndarray) -> np.ndarray:
    """Faces used by exactly one block -> [F,4] vertex ids. Also returns manifoldness."""
    count: dict = {}
    for b in blocks:
        for f in HEX_FACES:
            q = tuple(int(b[i]) for i in f)
            k = tuple(sorted(q))
            if k in count:
                count[k][1] += 1
            else:
                count[k] = [q, 1]
    bnd = np.array([q for q, c in count.values() if c == 1], dtype=np.int64).reshape(-1, 4)
    manifold = all(c <= 2 for _, c in count.values())
    return bnd, manifold


def local_spacing(v: np.ndarray, blocks: np.ndarray) -> np.ndarray:
    """Median incident block-edge length per vertex."""
    nb = np.asarray(CORNER_NB)
    a = blocks[:, :, None].repeat(3, 2).ravel()
    b = blocks[:, nb].ravel()
    ln = np.linalg.norm(v[a] - v[b], axis=1)
    order = np.argsort(a, kind="stable")
    a, ln = a[order], ln[order]
    h = np.full(len(v), np.nan)
    starts = np.r_[0, np.flatnonzero(np.diff(a)) + 1]
    ends = np.r_[starts[1:], len(a)]
    for s, e in zip(starts, ends):
        h[a[s]] = np.median(ln[s:e])
    return h


def _point_tri(p, a, b, c):
    """Distance of points p [N,3] to triangles (a, b, c) [N,3] (Ericson, closest point)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    bp = p - b; d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    cp = p - c; d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    va = d3 * d6 - d5 * d4; vb = d5 * d2 - d1 * d6; vc = d1 * d4 - d3 * d2
    den = np.where(np.abs(va + vb + vc) < 1e-300, 1e-300, va + vb + vc)
    v_, w_ = vb / den, vc / den
    q = a + ab * v_[:, None] + ac * w_[:, None]                      # interior
    t_ab = np.clip(d1 / np.maximum(d1 - d3, 1e-300), 0, 1)
    t_ac = np.clip(d2 / np.maximum(d2 - d6, 1e-300), 0, 1)
    t_bc = np.clip((d4 - d3) / np.maximum((d4 - d3) + (d5 - d6), 1e-300), 0, 1)
    cand = np.stack([q, a + ab * t_ab[:, None], a + ac * t_ac[:, None],
                     b + (c - b) * t_bc[:, None], a, b, c], 1)
    inside = (va >= 0) & (vb >= 0) & (vc >= 0)
    d = np.linalg.norm(cand - p[:, None], axis=-1)
    d[:, 0] = np.where(inside, d[:, 0], np.inf)
    return d.min(1)


def point_quad_distance(p: np.ndarray, quads: np.ndarray, centres: np.ndarray, k: int = 8):
    """Distance of points to the nearest of the k quads with the closest centres.
    Quads are split into two triangles. Returns (distance, quad index)."""
    k = min(k, len(quads))
    _, cand = cKDTree(centres).query(p, k=k)
    cand = cand.reshape(len(p), k)
    best = np.full(len(p), np.inf); arg = np.zeros(len(p), np.int64)
    for j in range(k):
        q = quads[cand[:, j]]
        d = np.minimum(_point_tri(p, q[:, 0], q[:, 1], q[:, 2]), _point_tri(p, q[:, 0], q[:, 2], q[:, 3]))
        better = d < best
        best[better] = d[better]; arg[better] = cand[better, j]
    return best, arg


def score(v: np.ndarray, blocks: np.ndarray, surf: np.ndarray, labels: np.ndarray,
          cfg: RewardV2Config = RewardV2Config(),
          mesh_quality: Callable[[np.ndarray, np.ndarray], float] | None = None,
          rng: np.random.Generator | None = None) -> RewardV2Terms:
    """v: [V,3] corners, blocks: [B,8], surf: [S,3] surface points, labels: [S] int."""
    v = np.asarray(v, np.float64); blocks = np.asarray(blocks, np.int64)
    surf = np.asarray(surf, np.float64); labels = np.asarray(labels)
    rng = rng or np.random.default_rng(0)
    cells = v[blocks]

    # --- gate ----------------------------------------------------------------
    sj = scaled_jacobian_corners(cells)
    bnd, manifold = boundary_faces(blocks)
    val = validate_generated_mesh(v, blocks, expected_blocks=None)
    vol = hex_signed_volumes(cells)
    gate = 1.0 if val.valid else cfg.soft_gate * float((vol > 0).mean()) * (1.0 if manifold else 0.5)

    # --- R_q: tail of the scaled Jacobian ---------------------------------------
    p5 = float(np.percentile(sj, 5))
    r_q = float(np.clip(p5 / cfg.sj_target, 0.0, 1.0))

    # --- R_fid -------------------------------------------------------------------
    tree = cKDTree(surf)
    h = local_spacing(v, blocks)
    bv = np.unique(bnd)
    d_c, i_c = tree.query(v[bv])
    s_corner = np.exp(-(d_c / (cfg.tau * h[bv])) ** 2)
    mid = v[bnd].mean(axis=1)
    h_f = np.median(h[bnd], axis=1)
    d_m, i_m = tree.query(mid)
    s_mid = np.exp(-(d_m / (cfg.tau * h_f)) ** 2)
    # coverage: surface samples -> distance to the boundary quads (2 triangles each)
    sel = rng.choice(len(surf), size=min(cfg.n_cov, len(surf)), replace=False)
    d_s, j_s = point_quad_distance(surf[sel], v[bnd], mid, k=8)
    covered = d_s <= cfg.cov_tol * h_f[j_s]
    w_s = np.where(labels[sel] == cfg.blade_label, cfg.blade_factor, 1.0)
    cov = float((covered * w_s).sum() / w_s.sum())
    blade = labels[sel] == cfg.blade_label
    cov_blade = float(covered[blade].mean()) if blade.any() else float("nan")
    r_fid = float(0.5 * (s_corner.mean() + cov))

    # --- R_lab: one patch per boundary face ---------------------------------------
    # Only faces lying on the surface count (corners within tau*h); a face is label-
    # consistent if its corners share at least one label within reach.
    near = tree.query_ball_point(v[bv], d_c + cfg.tau * h[bv])
    lab_sets = {int(x): set(labels[n].tolist()) for x, n in zip(bv, near)}
    on_srf = dict(zip(bv.tolist(), (d_c <= cfg.tau * h[bv]).tolist()))
    good = tot = 0
    for f in bnd:
        if not all(on_srf[int(x)] for x in f):
            continue
        tot += 1
        good += bool(set.intersection(*(lab_sets[int(x)] for x in f)))
    r_lab = good / tot if tot else 0.0

    # --- R_mesh (optional, post-TFI) -----------------------------------------------
    r_mesh = None if mesh_quality is None else float(np.clip(mesh_quality(v, blocks), 0, 1))
    parts = [(cfg.w_fid, r_fid), (cfg.w_lab, r_lab), (cfg.w_q, r_q)]
    if r_mesh is not None:
        parts.append((cfg.w_mesh, r_mesh))
    body = sum(w * r for w, r in parts) / sum(w for w, _ in parts)
    detail = dict(corner=float(s_corner.mean()), mid=float(s_mid.mean()), cov=cov, n_lab_faces=tot,
                  cov_blade=cov_blade, sj_p5=p5, sj_min=float(sj.min()),
                  n_sj_neg=int((sj <= 0).sum()), manifold=bool(manifold), n_bnd_faces=len(bnd))
    return RewardV2Terms(float(gate * body), float(gate), r_fid, r_lab, r_q, r_mesh, detail)
