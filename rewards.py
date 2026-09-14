"""
rewards.py

Reward functions for the RL curriculum (Part B). Each `make_*_reward` factory
closes over a tokenizer (and per-stage weights) and returns a plain
`reward_fn(token_ids: list[int], face_count: int) -> float` matching
`RLObjective`'s expected interface (objectives.py) -- kept as closures rather
than a class hierarchy since there is exactly one thing every reward needs
(the tokenizer to decode with) plus a config-driven weight dict, not enough
structure to justify more machinery.

Every reward here is computed from properties of the GENERATED sequence
(decodability, structural validity, face count match) -- it has to vary with
what the policy actually produces to carry any learning signal. This is a
deliberate correction to one line of the original plan text ("incorporate the
tistos dataset's own hex_hex_metrics_*.json quality metrics... as a reward
term"): those files (`hex_hex_metrics_<name>.json`, one per tistos machine
directory) hold the *ground-truth* mesh's own AlgoHex solve quality
(`LocalMeshability.percentage_meshable_vertices`, `Parametrization.
n_invalid_param_tets`, etc.) -- constant with respect to whatever the model
generates, so using it directly as a per-generation reward would produce zero
gradient signal (it doesn't depend on the model's output at all). They are
useful for a different purpose -- weighting/filtering *which* training
examples are worth learning from (a machine with poor `percentage_meshable_
vertices` is a harder target) -- exposed here as `load_ground_truth_quality`
for that optional use, not wired into any reward by default.
"""

import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch


# ------------------------------------------------------------------
# Quadtron (tokenizer_v2.Tokenizer2D) -- validity is the real problem here,
# there is no structural guarantee a generated sequence decodes to anything.
# ------------------------------------------------------------------
def _safe_detokenize(tokenizer, token_ids: list) -> Optional[tuple]:
    """detokenize() can throw on garbage generation (e.g. a truncated/empty
    coordinate run) -- exactly the failure mode RL is meant to reduce, so it
    must be caught and scored as invalid (reward 0 for that term), not raised."""
    try:
        vertices, quads = tokenizer.detokenize(token_ids)
        if quads.numel() == 0:
            return None
        return vertices, quads
    except Exception:
        return None


def _quad_nondegenerate_fraction(vertices: torch.Tensor, quads: torch.Tensor) -> float:
    """Fraction of generated quads with 4 distinct corners and nonzero area
    (shoelace formula on the first 2 coords -- works for dim=2 and dim=3
    quads that are at least planar-ish, which is what a valid mesh face is)."""
    if quads.numel() == 0:
        return 0.0
    n_faces = quads.size(1)
    ok = 0
    for fi in range(n_faces):
        idx = quads[:, fi].tolist()
        if len(set(idx)) != quads.size(0):
            continue  # duplicate corner -> degenerate
        pts = vertices[quads[:, fi]][:, :2].numpy()
        area = 0.5 * abs(
            sum(pts[i, 0] * pts[(i + 1) % 4, 1] - pts[(i + 1) % 4, 0] * pts[i, 1]
                for i in range(4))
        )
        if area > 1e-8:
            ok += 1
    return ok / n_faces


def make_vertex_reward(tokenizer, weights: Optional[dict] = None) -> Callable:
    """'vertex' stage: reward whether coordinate tokens at their expected
    position in the stream actually fall in the coordinate sub-vocabulary
    (not a stray pad/special token) -- a free-sampling policy CAN put a
    special token where a coordinate is expected, teacher-forcing alone never
    penalizes that off-distribution mistake the way a rollout-based reward can."""
    w = {"in_range": 1.0, **(weights or {})}

    def reward_fn(token_ids: list, face_count: int) -> float:
        in_coords = False
        n_coord_slots = n_valid = 0
        for tok in token_ids:
            if tok == tokenizer.start_token:
                in_coords = True
                continue
            if tok == tokenizer.end_token:
                break
            if not in_coords:
                continue
            if tok == getattr(tokenizer, "eor_token", -1):
                continue
            n_coord_slots += 1
            if tok < tokenizer.quantization_levels:
                n_valid += 1
        frac = n_valid / n_coord_slots if n_coord_slots else 0.0
        return w["in_range"] * frac

    return reward_fn


def make_face_reward(tokenizer, weights: Optional[dict] = None) -> Callable:
    """'face' stage: decodability + fraction of non-degenerate quads."""
    w = {"decodes": 0.4, "nondegenerate": 0.6, **(weights or {})}

    def reward_fn(token_ids: list, face_count: int) -> float:
        decoded = _safe_detokenize(tokenizer, token_ids)
        if decoded is None:
            return 0.0
        vertices, quads = decoded
        return (w["decodes"]
                + w["nondegenerate"] * _quad_nondegenerate_fraction(vertices, quads))

    return reward_fn


def make_row_reward(tokenizer, weights: Optional[dict] = None) -> Callable:
    """'row' stage: as 'face', plus a bonus for eor-delimited chunks that each
    decode as an internally-consistent row (row-compressed strategies 2/3
    only -- strategy 0/1 have no eor tokens, term is a no-op there)."""
    w = {"decodes": 0.3, "nondegenerate": 0.4, "row_consistency": 0.3, **(weights or {})}

    def reward_fn(token_ids: list, face_count: int) -> float:
        decoded = _safe_detokenize(tokenizer, token_ids)
        base = 0.0 if decoded is None else _quad_nondegenerate_fraction(*decoded)
        eor = getattr(tokenizer, "eor_token", None)
        if eor is None or eor not in token_ids:
            # Strategy without row-compression (0/1): no eor tokens to check,
            # so this term degrades to "did it decode at all" -- same
            # weakness as `decodes` above, i.e. it can't distinguish
            # sensible-but-uncompressed from noisy-but-technically-decodable
            # sequences. Acceptable for now (`nondegenerate` still carries
            # real signal); tighten once strategies 2/3 get 3D support and
            # this path is exercised for real (see decision log).
            row_term = 1.0 if decoded is not None else 0.0
        else:
            n_rows = token_ids.count(eor)
            row_term = min(1.0, n_rows / max(1, face_count // 4))
        return (w["decodes"] * (1.0 if decoded is not None else 0.0)
                + w["nondegenerate"] * base + w["row_consistency"] * row_term)

    return reward_fn


def make_mesh_reward(tokenizer, weights: Optional[dict] = None) -> Callable:
    """'mesh' stage: full-sequence validity, matching the existing
    `Tokenizer2D.testing()` round-trip check's spirit -- decodes, right face
    count, all faces non-degenerate."""
    w = {"decodes": 0.2, "face_count_match": 0.3, "nondegenerate": 0.5, **(weights or {})}

    def reward_fn(token_ids: list, face_count: int) -> float:
        decoded = _safe_detokenize(tokenizer, token_ids)
        if decoded is None:
            return 0.0
        vertices, quads = decoded
        n_gen = quads.size(1)
        count_term = 1.0 - min(1.0, abs(n_gen - face_count) / max(1, face_count))
        return (w["decodes"] + w["face_count_match"] * count_term
                + w["nondegenerate"] * _quad_nondegenerate_fraction(vertices, quads))

    return reward_fn


_QUADTRON_STAGE_FACTORIES = {
    "vertex": make_vertex_reward,
    "face": make_face_reward,
    "row": make_row_reward,
    "mesh": make_mesh_reward,
}


def make_quadtron_reward(tokenizer, stage: str, weights: Optional[dict] = None) -> Callable:
    """Top-level factory, dispatches on `PipelineConfig.rl_curriculum_stage`."""
    if stage not in _QUADTRON_STAGE_FACTORIES:
        raise ValueError(f"Unknown curriculum stage {stage!r}, expected one of "
                         f"{list(_QUADTRON_STAGE_FACTORIES)}")
    return _QUADTRON_STAGE_FACTORIES[stage](tokenizer, weights)


# ------------------------------------------------------------------
# Polytron Stage 1 (VertexGen) -- reuses the count_ok / vertex-error logic
# pattern from polytron_vertex_model.py:vertex_eval, as a standalone reward
# function. NOTE: usable once the FACECOUNTS/FC2N block-count gap documented
# in the decision log is resolved (VertexGen can't currently train on
# non-templated 3D data at all) -- the reward itself doesn't depend on that
# fix, only on having a trained-enough VertexGen checkpoint to roll out from.
# ------------------------------------------------------------------
def make_polytron_vertex_reward(tokenizer, expected_vertex_count: int,
                                weights: Optional[dict] = None) -> Callable:
    w = {"count_ok": 0.5, "position": 0.5, **(weights or {})}

    def reward_fn(token_ids: list, face_count: int) -> float:
        try:
            verts, _faces, _geom = tokenizer.detokenize(token_ids)
        except Exception:
            return 0.0
        if not verts:
            return 0.0
        count_term = 1.0 if len(verts) == expected_vertex_count else 0.0
        # Position term needs a reference to compare against; without one
        # (this function only sees token_ids/face_count, no ground truth),
        # falls back to a bounds-sanity check: r within the tokenizer's own
        # fixed R_MIN/R_MAX (rewards staying in-distribution).
        r_vals = np.array([v[0] for v in verts])
        in_bounds = np.mean((r_vals >= tokenizer.R_MIN) & (r_vals <= tokenizer.R_MAX))
        return w["count_ok"] * count_term + w["position"] * float(in_bounds)

    return reward_fn


# ------------------------------------------------------------------
# Optional: ground-truth mesh quality, for weighting/filtering training
# examples by how meshable the source machine is -- NOT a per-generation
# reward (see module docstring for why).
# ------------------------------------------------------------------
def load_ground_truth_quality(hex_hex_metrics_path: Path) -> dict:
    d = json.loads(Path(hex_hex_metrics_path).read_text())
    lm = d.get("LocalMeshability", {})
    pm = d.get("Parametrization", {})
    return {
        "percentage_meshable_vertices": lm.get("percentage_meshable_vertices"),
        "n_invalid_param_tets": pm.get("n_invalid_param_tets"),
        "n_invalid_valencies": pm.get("n_invalid_valencies"),
    }
