"""rewards_hexarow.py — GRPO-Belohnungen fuer HexaRow (P2-1).

Factory-Stil wie rewards.py (Closure + Config-Dataclass). Drei Terme, alle
CONFIG-gesteuert:

  r_valid   binär: gestoppt UND detokenisierbar ohne Trim UND
            validate_generated_mesh (inkl. expected_blocks) -> 1.0 sonst 0.0.
  r_quality mean hex_min_jacobian ueber Zellen mit GT-kompatibler Orientierung
            (signed volume > 0) MINUS lambda * mean(max(0, -detJ_cell)); 0 wenn
            invalid. Bestraft also isolierte gefaltete Zellen direkt.
  r_conform symmetrischer Chamfer(gen-Verts, GT-surface_points), Blade-Punkte
            der GT-Seite x blade_factor gewichtet, bbox-diagonal-normiert,
            als Reward 1 - min(1, d/scale); 0 wenn invalid.

total = w_valid*r_valid + w_quality*r_quality + w_conform*r_conform.

KEIN Vertex-/Token-Reward: die Grammatik ist durch slot_mask-constrained Decoding
bereits garantiert (rewards.py-Vertexreward waere hier vakuum). Das Fehlerprofil
der SFT-Rollouts (reports/sft_family_eval.md) sind isolierte gefaltete Zellen /
Dup-Artifakte — genau was r_quality adressiert.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from generate import detokenize_safe
from mesh_validation import hex_min_jacobian, hex_signed_volumes, validate_generated_mesh


@dataclass(frozen=True, slots=True)
class HexaRowRewardConfig:
    w_valid: float = 1.0
    w_quality: float = 0.5
    w_conform: float = 0.1
    lam_fold: float = 1.0
    blade_factor: float = 2.0
    conform_scale: float = 1.0


@dataclass(frozen=True, slots=True)
class RewardTerms:
    r_valid: float
    r_quality: float
    r_conform: float
    total: float
    valid: bool


_INVALID = RewardTerms(0.0, 0.0, 0.0, 0.0, False)


def _to_cartesian(vpt, coords: str) -> np.ndarray:
    v = np.asarray(vpt, dtype=np.float64)
    if coords == "cart":
        return v
    return np.stack([v[:, 0] * np.cos(v[:, 1]), v[:, 0] * np.sin(v[:, 1]),
                     v[:, 2]], axis=-1)


def _nn_distances(a: np.ndarray, b: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Fuer jeden Punkt in a: min. euklidischer Abstand zu b (float32, chunked)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    out = np.empty(a.shape[0], dtype=np.float32)
    for i in range(0, a.shape[0], chunk):
        ac = a[i:i + chunk]
        d2 = ((ac[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        out[i:i + chunk] = np.sqrt(d2.min(axis=1))
    return out


def chamfer_symmetric(gen: np.ndarray, gt: np.ndarray,
                      weights: np.ndarray | None = None) -> float:
    """Symmetrischer Chamfer; weights gewichten die GT->gen-Richtung."""
    if len(gen) == 0 or len(gt) == 0:
        return float("inf")
    d_ab = _nn_distances(gen, gt)
    d_ba = _nn_distances(gt, gen)
    if weights is None:
        return float(0.5 * (d_ab.mean() + d_ba.mean()))
    w = np.asarray(weights, dtype=np.float64)
    return float(0.5 * (d_ab.mean() + (d_ba * w).sum() / w.sum()))


def _score(token_ids: list, item: dict, tokenizer, cfg: HexaRowRewardConfig,
           coords: str, stop_id: int) -> RewardTerms:
    if not token_ids or token_ids[-1] != stop_id:
        return _INVALID
    res, trim = detokenize_safe(list(token_ids), tokenizer, stop_id, coords=coords)
    if res is None or trim is not None:
        return _INVALID
    vpt, blk = res
    vcart = _to_cartesian(vpt.numpy(), coords)
    blocks = blk.numpy()
    val = validate_generated_mesh(vcart, blocks,
                                  expected_blocks=int(item["blocks"]))
    if not val.valid:
        return _INVALID

    cells = vcart[blocks]
    jac = hex_min_jacobian(cells)
    vol = hex_signed_volumes(cells)
    oriented = jac[vol > 0.0]
    base = float(oriented.mean()) if oriented.size else 0.0
    fold = float(np.maximum(0.0, -jac).mean())
    r_quality = base - cfg.lam_fold * fold

    r_conform = 0.0
    gt_pts = item.get("surface_points")
    if gt_pts is not None:
        gt = np.asarray(gt_pts.numpy() if hasattr(gt_pts, "numpy") else gt_pts,
                        dtype=np.float64)
        blade = item.get("is_blade")
        if blade is not None and cfg.blade_factor != 1.0:
            blade = np.asarray(blade.numpy() if hasattr(blade, "numpy") else blade,
                               dtype=bool)
            w = np.where(blade, cfg.blade_factor, 1.0)
        else:
            w = None
        d = chamfer_symmetric(vcart, gt, w)
        diag = float(np.linalg.norm(gt.max(0) - gt.min(0)))
        if diag > 1e-12:
            r_conform = max(0.0, 1.0 - min(1.0, d / (cfg.conform_scale * diag)))

    total = (cfg.w_valid * 1.0 + cfg.w_quality * r_quality
             + cfg.w_conform * r_conform)
    return RewardTerms(1.0, r_quality, r_conform, float(total), True)


def make_hexarow_reward(tokenizer, config: HexaRowRewardConfig | None = None,
                        coords: str = "cart", stop_id: int | None = None):
    """Factory wie rewards.py: gibt reward_fn(token_ids, item) -> RewardTerms."""
    cfg = config or HexaRowRewardConfig()
    sid = tokenizer.core.stop_token if stop_id is None else int(stop_id)

    def reward_fn(token_ids: list, item: dict) -> RewardTerms:
        return _score(token_ids, item, tokenizer, cfg, coords, sid)

    return reward_fn
