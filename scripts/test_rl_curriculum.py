#!/usr/bin/env python3
"""test_rl_curriculum.py -- the Quadtron RL curriculum, end to end.

Nothing covered `make_quadtron_reward` or `PipelineConfig.rl_curriculum_stage`
before this. Three things are checked, in the order a defect would surface:

  A  discrimination -- per stage, ground-truth tokens must score strictly
     higher than a corruption of the kind that stage exists to catch, and
     garbage must score lowest. Run for sorting_strategy 1 (no eor tokens)
     and 2 (eor tokens), so the row-compressed path is exercised for real.
  B  wiring -- `Trainer._build_objective` builds an `RLObjective` for each
     stage with `cfg.rl_enabled=True`, one real step over the 2D corpus
     produces a finite loss, and the gradient matches the reward spread:
     a spread means a policy update, no spread means exactly none (the same
     invariant scripts/test_training_e2e.py phase 3 asserts for GRPO).
  C  signal -- what the reward actually measures at the point RL starts,
     i.e. against the untrained policy the curriculum is meant to improve.
     Reported as a measurement; `--strict` turns a saturated stage into a
     failure.

  uv run python scripts/test_rl_curriculum.py
  uv run python scripts/test_rl_curriculum.py --strict

Exit 0 green, 1 red, 2 the 2D corpus is not reachable (a different problem
from a broken one -- see scripts/verify_pipeline.py's MISSING).
"""
from __future__ import annotations

import argparse
import os
import shutil
import statistics
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

import meshtron  # noqa: E402,F401  (puts the subpackage dirs on sys.path)
from meshtron.data.quad_domain import load_quad_domain  # noqa: E402
from meshtron.data.tokenizer_v2 import Tokenizer2D  # noqa: E402
from meshtron.training.config import PipelineConfig  # noqa: E402
from meshtron.training.rewards import make_quadtron_reward  # noqa: E402

STAGES = ("vertex", "face", "row", "mesh")
N_MESHES = 4
QUANT = 128


def _expect(failures: list, cond: bool, msg: str) -> None:
    if not cond:
        failures.append(msg)


# --------------------------------------------------------------------------
# corruptions -- each one breaks exactly the property a stage claims to score
# --------------------------------------------------------------------------

def corrupt_coords(seq: list, tok: Tokenizer2D, frac: float = 0.5) -> list:
    """Half the coordinate tokens replaced by pad -- a special token where a
    coordinate belongs, which is what the vertex stage is about."""
    out = list(seq)
    idx = [i for i, t in enumerate(out) if t < tok.quantization_levels]
    for i in idx[: int(len(idx) * frac)]:
        out[i] = tok.pad_token
    return out


def flatten_quads(seq: list, tok: Tokenizer2D) -> list:
    """Every face collapsed onto one of its corners -- decodes, but every quad
    has duplicate corners and zero area, which is what the face stage scores."""
    out: list = []
    buf: list = []
    started = False
    for t in seq:
        if t == tok.start_token:
            out.append(t)
            started = True
            continue
        if t in (tok.end_token, tok.eor_token):
            out.extend(buf)
            buf = []
            out.append(t)
            continue
        if not started or t >= tok.quantization_levels:
            out.append(t)
            continue
        buf.append(t)
        if len(buf) == 8:          # one 2D face = 4 corners x 2 coords
            out.extend(buf[:2] * 4)
            buf = []
    out.extend(buf)
    return out


def truncate(seq: list, tok: Tokenizer2D, keep: float = 0.5) -> list:
    """Half the faces missing -- the face count no longer matches, which is
    what the mesh stage scores."""
    coords = [i for i, t in enumerate(seq) if t < tok.quantization_levels]
    cut = coords[int(len(coords) * keep)]
    return seq[:cut] + [tok.end_token] * tok.n_start_end_tokens_repeat


def garbage(tok: Tokenizer2D, n: int = 40) -> list:
    """No coordinate token at all -- nothing can decode from this."""
    specials = [tok.pad_token, tok.eor_token]
    return ([tok.start_token] * tok.n_start_end_tokens_repeat
            + [specials[i % 2] for i in range(n)]
            + [tok.end_token] * tok.n_start_end_tokens_repeat)


# --------------------------------------------------------------------------
# A: discrimination
# --------------------------------------------------------------------------

def part_a(meshes, failures: list) -> list:
    """The corruption each stage must catch, and the ones it need not."""
    must_catch = {"vertex": "coord_corrupt", "face": "flat",
                  "row": "flat", "mesh": "trunc"}
    rows = []
    for strategy in (1, 2):
        for mi, mesh in enumerate(meshes):
            tok = Tokenizer2D(quantization_levels=QUANT, dim=2, verbose=False,
                              sorting_strategy=strategy)
            good = tok.tokenize(mesh.x[:, :2], mesh.faces)
            n_faces = int(mesh.faces.size(1))
            variants = {
                "good": good,
                "coord_corrupt": corrupt_coords(good, tok),
                "flat": flatten_quads(good, tok),
                "trunc": truncate(good, tok),
                "garbage": garbage(tok),
            }
            for stage in STAGES:
                reward = make_quadtron_reward(tok, stage)
                score = {k: reward(v, n_faces) for k, v in variants.items()}
                tag = f"strategy {strategy}, mesh {mi}, stage {stage}"
                bad = must_catch[stage]
                _expect(failures, score["good"] > score[bad],
                        f"{tag}: ground truth ({score['good']:.3f}) does not beat "
                        f"{bad} ({score[bad]:.3f})")
                _expect(failures, score["good"] > score["garbage"],
                        f"{tag}: ground truth ({score['good']:.3f}) does not beat "
                        f"garbage ({score['garbage']:.3f})")
                _expect(failures, score["garbage"] <= score[bad],
                        f"{tag}: garbage ({score['garbage']:.3f}) outscores "
                        f"{bad} ({score[bad]:.3f})")
                _expect(failures, all(0.0 <= v <= 1.0 for v in score.values()),
                        f"{tag}: a reward left [0, 1]: {score}")
                if mi == 0:
                    rows.append((strategy, stage, score))
    return rows


def part_a_factory(failures: list) -> None:
    tok = Tokenizer2D(quantization_levels=QUANT, dim=2, verbose=False)
    try:
        make_quadtron_reward(tok, "no_such_stage")
        failures.append("make_quadtron_reward accepted an unknown stage")
    except ValueError:
        pass
    doc_stages = tuple(
        PipelineConfig.__dataclass_fields__["rl_curriculum_stage"].default
        and s for s in STAGES)
    for stage in doc_stages:
        try:
            make_quadtron_reward(tok, stage)
        except Exception as e:  # noqa: BLE001
            failures.append(f"stage {stage!r} from the config docstring does "
                            f"not build: {type(e).__name__}: {e}")


# --------------------------------------------------------------------------
# B + C: one real RLObjective step per stage, through Trainer
# --------------------------------------------------------------------------

def _cfg(stage: str, data_path: str, log_dir: str) -> PipelineConfig:
    """Small enough to run on the CPU in seconds; every RL field set so the
    step goes through the same `_build_objective` a real run would."""
    return PipelineConfig(
        model_family="quadtron", dim=2, data_path=data_path,
        sorting_strategy=2, quantization=QUANT, n_sample_points=64,
        train_val_ratio=0.5, d_model=64, n_heads=2, stage_layers=(1, 1, 1),
        n_latents=8, batch_size=1, num_epochs=1, warmup_steps=1,
        precision="fp32", rl_enabled=True, rl_curriculum_stage=stage,
        rl_rollouts_per_condition=8, rl_max_length=48, rl_temperature=1.0,
        log_dir=log_dir, seed=0,
    )


def part_bc(data_path: str, log_dir: str, failures: list) -> list:
    from meshtron.training.trainer import Trainer

    rows = []
    for stage in STAGES:
        trainer = Trainer(_cfg(stage, data_path, log_dir))
        # By name, not isinstance: trainer.py reaches objectives.py through a
        # bare `from objectives import ...`, so the class it instantiates is a
        # different object from `meshtron.training.objectives.RLObjective`
        # (the sys.path bridge's known limitation, see meshtron/__init__.py).
        _expect(failures, type(trainer.objective).__name__ == "RLObjective",
                f"stage {stage}: rl_enabled=True built a "
                f"{type(trainer.objective).__name__}, not an RLObjective")
        batch = next(iter(trainer.train_loader))

        # The spread the objective will see, measured on the same rollouts the
        # same seed produces -- without it a zero gradient cannot be told from
        # a broken backward pass.
        spread, mean_r = _reward_spread(trainer, batch, stage)

        out = trainer.objective.compute(batch, trainer.policy)
        out.loss.backward()
        grads = [p.grad for p in trainer.model.parameters() if p.grad is not None]
        grad_norm = float(sum(float(g.norm()) ** 2 for g in grads) ** 0.5)

        loss_val = float(out.loss.detach())
        _expect(failures, torch.isfinite(out.loss.detach()).all(),
                f"stage {stage}: loss is not finite ({loss_val})")
        _expect(failures, len(grads) > 0,
                f"stage {stage}: backward reached no parameter at all")
        if spread > 0:
            _expect(failures, grad_norm > 0,
                    f"stage {stage}: rewards spread by {spread:.4f} but the "
                    f"gradient is {grad_norm:.3e} -- no policy update from a "
                    f"live reward signal")
        else:
            _expect(failures, grad_norm == 0.0,
                    f"stage {stage}: every rollout scored the same and the "
                    f"gradient is {grad_norm:.3e}, not 0 -- an update without "
                    f"a reward signal")
        rows.append((stage, loss_val, grad_norm, mean_r, spread))
    return rows


def _reward_spread(trainer, batch, stage: str):
    """Roll out the untrained policy once and score it -- mean and population
    standard deviation of the reward over the group."""
    device = trainer.device
    pc = batch["point_cloud"].to(device)
    fc = batch["face_count"].to(device)
    cfg = trainer.cfg
    G = cfg.rl_rollouts_per_condition
    start = batch["input_tokens"][:, :8].to(device)
    with torch.no_grad():
        rollouts = trainer.policy.sample(
            point_cloud=pc.expand(G, *pc.shape[1:]), face_count=fc.expand(G),
            start_tokens=start.expand(G, -1), max_length=cfg.rl_max_length,
            temperature=cfg.rl_temperature, eos_token=trainer.tokenizer.end_token,
        )
    reward = make_quadtron_reward(trainer.tokenizer, stage, cfg.reward_weights)
    n_faces = int(fc[0].item())
    vals = [reward(rollouts[g].tolist(), n_faces) for g in range(G)]
    return statistics.pstdev(vals), statistics.mean(vals)


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None,
                    help="directory of the 2D quad corpus (default: the one "
                         "meshtron.data.quad_domain points at)")
    ap.add_argument("--strict", action="store_true",
                    help="fail when a stage gives the untrained policy no "
                         "reward spread, i.e. nothing for RL to learn from")
    args = ap.parse_args()

    try:
        meshes = (load_quad_domain(args.corpus, limit=N_MESHES) if args.corpus
                  else load_quad_domain(limit=N_MESHES))
    except (FileNotFoundError, RuntimeError) as e:
        print(f"MISSING — the 2D corpus is not reachable: {e}")
        return 2

    failures: list = []
    scores = part_a(meshes, failures)
    part_a_factory(failures)

    work = tempfile.mkdtemp(prefix="rl_curriculum_")
    try:
        data_path = os.path.join(work, "quad2d.pt")
        torch.save(meshes, data_path)
        steps = part_bc(data_path, os.path.join(work, "runs"), failures)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print("\nA  reward against corrupted ground truth (first mesh)")
    print(f"   {'strat':>5} {'stage':<7} {'good':>6} {'coord':>6} {'flat':>6} "
          f"{'trunc':>6} {'garbage':>8}")
    for strategy, stage, s in scores:
        print(f"   {strategy:>5} {stage:<7} {s['good']:>6.3f} "
              f"{s['coord_corrupt']:>6.3f} {s['flat']:>6.3f} "
              f"{s['trunc']:>6.3f} {s['garbage']:>8.3f}")

    print("\nB  one RLObjective step per stage, through Trainer")
    print(f"   {'stage':<7} {'loss':>10} {'grad':>10} {'mean R':>8} {'std R':>8}"
          f"  signal")
    saturated = []
    for stage, loss, grad, mean_r, spread in steps:
        live = spread > 0
        if not live:
            saturated.append(stage)
        print(f"   {stage:<7} {loss:>10.4f} {grad:>10.4f} {mean_r:>8.4f} "
              f"{spread:>8.4f}  {'LIVE' if live else 'SATURATED'}")

    if saturated:
        print(f"\nC  no reward spread against the untrained policy for: "
              f"{', '.join(saturated)}.")
        print("   A random token stream decodes into non-degenerate quads by "
              "construction, so these stages score it perfectly and leave the")
        print("   policy gradient exactly zero -- which is where the curriculum "
              "is supposed to start.")
        if args.strict:
            failures.append(f"--strict: no reward spread for "
                            f"{', '.join(saturated)}")

    if failures:
        print(f"\nRED — RL curriculum, {len(failures)} failure(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"\nGREEN — RL curriculum: {len(STAGES)} stages discriminate on "
          f"{N_MESHES} meshes x 2 sorting strategies, each builds an "
          f"RLObjective\n        and takes a finite step whose gradient "
          f"matches its reward spread.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
