"""rewards_hexarow.py — dense GRPO reward for HexaRow (P2-1).

Factory style like rewards.py (closure + config dataclass). The reward is now
DENSE on the current policy instead of all-or-nothing: a rollout that fails the
strict ``validate_generated_mesh`` gate (e.g. 56 blocks vs 60 expected, or a few
inverted cells) still receives a partial gradient, because full validity had
``valid_rate = 0.000`` on the RL start checkpoint and therefore produced
``advantage = 0`` for every group.

Hard-invalid only for garbage (all terms exactly 0.0, ``valid=False``):
  * empty sequence, or last token != stop_token;
  * no sep_token anywhere (not a row-grammar program — e.g. random tokens);
  * detokenize_safe returns ``res is None`` (no valid row reconstructible);
  * non-finite Cartesian vertices.
A trailing incomplete row (``trim is not None``) is NOT invalid: it only takes
the mild ``w_trim`` penalty.

Dense terms (computed ALWAYS, not behind ``val.valid``):

  r_count   max(0, 1 - |gen_blocks - gt_blocks| / max(1, gt_blocks)); replaces
            the binary block-count check.
  r_quality mean(hex_min_jacobian over cells with signed volume > 0) MINUS
            lam_fold * mean(max(0, -minJ)); if no cell has positive signed
            volume, base is 0.0 and an extra penalty_all_folded is subtracted.
            Directly punishes isolated folded / inverted cells.
  r_conform symmetric Chamfer(gen verts, GT surface_points), GT blade points
            weighted by blade_factor, bbox-diagonal-normalised, as
            1 - min(1, d / (conform_scale * diag)); only if the item carries
            surface_points.

Validity bonus: ``valid``/``r_valid`` keep the strict validate_generated_mesh
semantics (the ``r_valid_share`` CSV column uses them); when valid, the full
``w_full_valid`` bonus is added to ``total``.

  total = w_count*r_count + w_quality*r_quality + w_conform*r_conform
          + w_full_valid*(1 if valid else 0) - w_trim*(1 if trim else 0)

``w_valid`` is retained only for positional-config backwards compatibility and
is UNUSED (folded into ``w_full_valid``).

NO vertex/token reward: the grammar is already guaranteed by slot_mask
constrained decoding (rewards.py's vertex reward would be vacuous here). The
SFT rollout failure profile (reports/sft_family_eval.md) is isolated folded
cells / dup artefacts — exactly what r_quality addresses.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from meshtron.training.generate import detokenize_safe
from meshtron.geometry.mesh_validation import hex_min_jacobian, hex_signed_volumes, validate_generated_mesh


@dataclass(frozen=True, slots=True)
class HexaRowRewardConfig:
    # Legacy fields kept (same order) so positional construction still works.
    w_valid: float = 1.0          # deprecated: unused, folded into w_full_valid
    w_quality: float = 0.5
    w_conform: float = 0.1
    lam_fold: float = 1.0
    blade_factor: float = 2.0
    conform_scale: float = 1.0
    # Dense-scheme fields.
    w_count: float = 0.5
    w_full_valid: float = 1.0
    w_trim: float = 0.25
    penalty_all_folded: float = 0.5


@dataclass(frozen=True, slots=True)
class RewardTerms:
    r_valid: float
    r_quality: float
    r_conform: float
    total: float
    valid: bool


_INVALID = RewardTerms(0.0, 0.0, 0.0, 0.0, False)


def block_count_reward(gen_blocks: int, gt_blocks: int) -> float:
    """Dense block-count term: max(0, 1 - |gen - gt| / max(1, gt))."""
    return max(0.0, 1.0 - abs(int(gen_blocks) - int(gt_blocks)) / max(1, int(gt_blocks)))


def _to_cartesian(vpt, coords: str) -> np.ndarray:
    v = np.asarray(vpt, dtype=np.float64)
    if coords == "cart":
        return v
    return np.stack([v[:, 0] * np.cos(v[:, 1]), v[:, 0] * np.sin(v[:, 1]),
                     v[:, 2]], axis=-1)


def _nn_distances(a: np.ndarray, b: np.ndarray, chunk: int = 256) -> np.ndarray:
    """For each point in a: min. Euclidean distance to b (float32, chunked)."""
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
    """Symmetric Chamfer; weights weight the GT->gen direction."""
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
    # --- hard-invalid garbage ------------------------------------------------
    if not token_ids or token_ids[-1] != stop_id:
        return _INVALID
    # A legitimate row-grammar program always emits sep_token at a row boundary;
    # a stop-terminated stream without any SEP (e.g. 300 random coordinate
    # tokens) is garbage -> exactly 0.0 / invalid.
    if tokenizer.core.sep_token not in token_ids[:-1]:
        return _INVALID
    res, trim = detokenize_safe(list(token_ids), tokenizer, stop_id, coords=coords)
    if res is None:
        return _INVALID
    vpt, blk = res
    vcart = _to_cartesian(vpt.numpy(), coords)
    if not np.isfinite(vcart).all():
        return _INVALID
    blocks = blk.numpy()
    trim_pen = 1.0 if trim is not None else 0.0

    # --- dense block-count term ---------------------------------------------
    gt_blocks = max(1, int(item["blocks"]))
    gen_blocks = int(blocks.shape[0])
    r_count = block_count_reward(gen_blocks, gt_blocks)

    # --- dense quality term (always) ----------------------------------------
    if blocks.size:
        cells = vcart[blocks]
        jac = hex_min_jacobian(cells)
        vol = hex_signed_volumes(cells)
        oriented = jac[vol > 0.0]
        fold = float(np.maximum(0.0, -jac).mean())
    else:
        oriented = np.empty(0, dtype=np.float64)
        fold = 0.0
    base = float(oriented.mean()) if oriented.size else 0.0
    r_quality = base - cfg.lam_fold * fold
    if oriented.size == 0:
        r_quality -= cfg.penalty_all_folded

    # --- dense conform term (always, same chamfer path) ---------------------
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

    # --- validity bonus (kept strict semantics for the CSV) -----------------
    val = validate_generated_mesh(vcart, blocks,
                                  expected_blocks=int(item["blocks"]))
    valid = bool(val.valid)
    r_valid = 1.0 if valid else 0.0

    total = (cfg.w_count * r_count + cfg.w_quality * r_quality
             + cfg.w_conform * r_conform + cfg.w_full_valid * r_valid
             - cfg.w_trim * trim_pen)
    return RewardTerms(r_valid, r_quality, r_conform, float(total), valid)


def make_hexarow_reward(tokenizer, config: HexaRowRewardConfig | None = None,
                        coords: str = "cart", stop_id: int | None = None):
    """Factory like rewards.py: returns reward_fn(token_ids, item) -> RewardTerms."""
    cfg = config or HexaRowRewardConfig()
    sid = tokenizer.core.stop_token if stop_id is None else int(stop_id)

    def reward_fn(token_ids: list, item: dict) -> RewardTerms:
        return _score(token_ids, item, tokenizer, cfg, coords, sid)

    return reward_fn
