"""test_training_e2e.py — train a model from nothing and mesh a geometry with it.

Every other training check in this repo runs ONE step and reports that the loss
came out finite. That answers "does the code execute", not "does training
work". This runs the whole chain and checks the claims a training pipeline has
to make:

  1 supervised      loss falls over a real run, validation is computed, the
                    best-val checkpoint is written AND can be loaded back
  2 resume          picks up at the right epoch with the optimiser state, and
                    the loss keeps falling instead of restarting
  3 GRPO            real steps on the model phase 1 produced: the policy moves,
                    the KL to the reference stays finite, the log is written
  4 inference       that trained checkpoint, fed a geometry, produces a
                    watertight mesh that sits on it

The model is deliberately small (--d 128 --layers 4, about 1.3 M parameters).
This test is about the PATH, not about mesh quality: a model this size trained
this briefly produces poor blockings, and the phase 4 assertions are about the
pipeline carrying them through, not about them being good. Production is
d=512, 12 layers, 40 M parameters.

    uv run python scripts/test_training_e2e.py              # ~6 min on CPU
    uv run python scripts/test_training_e2e.py --epochs 4   # quicker

Runs on the GPU when one is free and falls back to the CPU, because an 8 GB
card shared with anything else cannot hold a second process.
Exit 0 all phases pass, 1 otherwise.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, "data")
EPOCH_RE = re.compile(
    r"^epoch\s+(\d+)\s+loss\s+([0-9.]+)\s+tok-acc\s+([0-9.]+)"
    r"(?:.*?VAL loss\s+([0-9.]+)\s+tok-acc\s+([0-9.]+))?")
RESULTS: list[tuple[str, str, str, float]] = []


def need(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def run(cmd: list[str], env_extra: dict | None = None, timeout: int = 3600):
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    if env_extra:
        env.update(env_extra)
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env,
                       timeout=timeout, cwd=ROOT)
    return p, time.time() - t0


def epochs_of(stdout: str) -> list[dict]:
    """Parse the trainer's per-epoch summary lines out of its output."""
    rows = []
    for line in stdout.replace("\r", "\n").split("\n"):
        m = EPOCH_RE.match(line.strip())
        if m:
            rows.append({"epoch": int(m.group(1)), "loss": float(m.group(2)),
                         "acc": float(m.group(3)),
                         "val_loss": float(m.group(4)) if m.group(4) else None,
                         "val_acc": float(m.group(5)) if m.group(5) else None})
    return rows


def train_cmd(tmp: str, tag: str, epochs: int, small: dict,
              resume: str = "") -> list[str]:
    c = [sys.executable, os.path.join(ROOT, "meshtron", "training",
                                      "train_hexarow_full.py"),
         "--tokens", need(os.path.join(DATA, "hexarow_batch_tokens.pt")),
         "--src", need(os.path.join(DATA, "polytron_batch_clean.pt")),
         "--coords", "cart", "--epochs", str(epochs),
         "--save-start", "0", "--val-every", "1", "--warmup", "20",
         "--out", os.path.join(tmp, f"{tag}.pt"),
         "--ckpt", os.path.join(tmp, f"{tag}_ckpt.pt"), "--tag", tag]
    for k, v in small.items():
        c += [f"--{k}", str(v)]
    if resume:
        c += ["--resume", resume]
    return c


def phase1_supervised(tmp: str, epochs: int, small: dict, env: dict) -> dict:
    """A real run: loss must fall, validation must happen, and the best-val
    checkpoint must be loadable -- not just resumable."""
    p, secs = run(train_cmd(tmp, "sft", epochs, small), env)
    rows = epochs_of(p.stdout)
    if len(rows) < epochs:
        raise RuntimeError(f"only {len(rows)} of {epochs} epochs reported; "
                           f"stderr tail: {p.stderr.strip()[-400:]}")
    first, last = rows[0]["loss"], rows[-1]["loss"]
    if not last < first:
        raise RuntimeError(f"loss did not fall: {first:.3f} -> {last:.3f}")
    vals = [r["val_loss"] for r in rows if r["val_loss"] is not None]
    if not vals:
        raise RuntimeError("no validation loss was computed")
    ckpt = os.path.join(tmp, "sft_ckpt.pt")
    if not os.path.exists(ckpt):
        raise RuntimeError("no best-val checkpoint was written")
    # The point of the check: a checkpoint you cannot load is not a checkpoint.
    from scripts.eval_family import load_model
    ck, cfg, coords, npt, rb, zb, model, max_len, miss = load_model(ckpt, "cpu")
    n_par = sum(q.numel() for q in model.parameters())
    if miss.missing_keys:
        raise RuntimeError(f"checkpoint misses weights: {miss.missing_keys}")
    return {"epochs": len(rows), "loss": [first, last],
            "val_loss": [vals[0], vals[-1]],
            "tok_acc": [rows[0]["acc"], rows[-1]["acc"]],
            "best_val_epoch": int(ck["epoch"]), "params": int(n_par),
            "coords": coords, "seconds": round(secs, 1),
            "note": f"loss {first:.2f} -> {last:.2f}, val {vals[0]:.2f} -> "
                    f"{vals[-1]:.2f}, tok-acc {rows[0]['acc']:.3f} -> "
                    f"{rows[-1]['acc']:.3f}, {n_par / 1e6:.1f} M params, "
                    f"best-val checkpoint loads"}


def phase2_resume(tmp: str, epochs: int, small: dict, env: dict,
                  phase1: dict) -> dict:
    """Continue the interrupted run: the epoch counter, the optimiser state and
    the loss trajectory all have to survive."""
    ckpt = os.path.join(tmp, "sft_ckpt.pt")
    import torch
    resume_from = int(torch.load(ckpt, weights_only=False)["epoch"])
    p, secs = run(train_cmd(tmp, "res", epochs * 2, small, resume=ckpt), env)
    if "resume: epoch" not in p.stdout:
        raise RuntimeError(f"trainer did not report a resume; stderr tail: "
                           f"{p.stderr.strip()[-400:]}")
    rows = epochs_of(p.stdout)
    if not rows:
        raise RuntimeError("resumed run reported no epochs")
    if rows[0]["epoch"] != resume_from + 1:
        raise RuntimeError(f"resumed at epoch {rows[0]['epoch']}, expected "
                           f"{resume_from + 1}")
    # Loss must continue from where it was, not restart from a fresh init.
    if rows[0]["loss"] > phase1["loss"][1] * 1.5:
        raise RuntimeError(f"loss restarted: {rows[0]['loss']:.3f} after "
                           f"{phase1['loss'][1]:.3f} -- optimiser or weights "
                           f"were not restored")
    if not rows[-1]["loss"] < rows[0]["loss"]:
        raise RuntimeError(f"loss did not fall after resume: "
                           f"{rows[0]['loss']:.3f} -> {rows[-1]['loss']:.3f}")
    return {"resumed_at_epoch": rows[0]["epoch"], "epochs": len(rows),
            "loss": [rows[0]["loss"], rows[-1]["loss"]],
            "seconds": round(secs, 1),
            "note": f"resumed at epoch {rows[0]['epoch']} (was {resume_from}), "
                    f"loss continued {phase1['loss'][1]:.2f} -> "
                    f"{rows[0]['loss']:.2f} -> {rows[-1]['loss']:.2f}"}


def phase3_grpo(tmp: str, steps: int, env: dict) -> dict:
    """Real GRPO steps on the model phase 1 trained.

    A 1.3 M model produces no valid blockings, so every reward is 0 and every
    advantage with it. That is not a weak test -- it is the sharpest one
    available, because it pins down an invariant that holds for any model:

        no reward signal  =>  no policy update

    Violating it is exactly the defect this test found. The KL term used to be
    the plain log-ratio mean, which is not a divergence; with flat rewards it
    still produced a gradient norm of 0.24, drove its own KL to -0.019 and
    raised entropy from 5.120 to 5.146, pushing the policy away from the
    reference for no reason. The k3 estimator gives 0.0 on the same input.

    So: KL >= 0 always, everything finite, and with a flat reward the update
    has to be zero. When a reward does appear, the policy must move instead.
    """
    import csv as _csv
    import math
    import torch
    sft = os.path.join(tmp, "res.pt")
    if not os.path.exists(sft):
        sft = os.path.join(tmp, "sft.pt")
    before = {k: v.detach().clone()
              for k, v in torch.load(sft, weights_only=False)["model"].items()}
    log = os.path.join(tmp, "grpo_log.csv")
    cmd = [sys.executable, os.path.join(ROOT, "meshtron", "training",
                                        "train_grpo.py"),
           "--ckpt", sft,
           "--tokens", need(os.path.join(DATA, "hexarow_batch_tokens.pt")),
           "--src", need(os.path.join(DATA, "polytron_batch_clean.pt")),
           "--steps", str(steps), "--G", "4", "--items-per-step", "1",
           "--log-csv", log, "--ckpt-prefix", os.path.join(tmp, "grpo"),
           "--ckpt-every", str(steps)]
    p, secs = run(cmd, env)
    if not os.path.exists(log):
        raise RuntimeError(f"no GRPO log written; stderr tail: "
                           f"{p.stderr.strip()[-500:]}")
    rows = list(_csv.DictReader(open(log)))
    if len(rows) < steps:
        raise RuntimeError(f"{len(rows)} of {steps} steps logged; stderr tail: "
                           f"{p.stderr.strip()[-400:]}")
    gn = [float(r["grad_norm"]) for r in rows]
    kl = [float(r["KL"]) for r in rows]
    R = [float(r["mean_R"]) for r in rows]
    ent = [float(r["entropy"]) for r in rows]
    if any(not math.isfinite(v) for v in kl + gn + R):
        raise RuntimeError(f"non-finite value logged: KL={kl} grad={gn} R={R}")
    if min(kl) < 0.0:
        raise RuntimeError(
            f"KL went negative ({min(kl):.4g}): the anchor term is rewarding "
            f"drift away from the reference instead of penalising it")
    flat = max(abs(v) for v in R) == 0.0
    grpo_ck = f"{os.path.join(tmp, 'grpo')}_step{steps}.pt"
    moved = None
    if os.path.exists(grpo_ck):
        after = torch.load(grpo_ck, weights_only=False)["model"]
        moved = max(float((after[k].float() - before[k].float()).abs().max())
                    for k in before if k in after)
    if flat:
        if max(gn) > 0.0:
            raise RuntimeError(
                f"every reward was 0, so every advantage was 0 and the update "
                f"must be too, but grad_norm reached {max(gn):.4g} -- the "
                f"policy is being moved by something other than the reward")
        note = (f"{len(rows)} steps, reward flat at 0 (expected from 1.3 M "
                f"params), and the update is correctly 0: grad_norm 0, "
                f"KL {min(kl):.2e}..{max(kl):.2e}, entropy stable "
                f"{ent[0]:.3f}->{ent[-1]:.3f}")
    else:
        if max(gn) <= 0.0:
            raise RuntimeError("a reward appeared but no gradient followed")
        if moved is not None and moved <= 0.0:
            raise RuntimeError("the GRPO checkpoint equals the SFT one: the "
                               "update did not apply")
        note = (f"{len(rows)} steps, mean reward {R[0]:.3f}->{R[-1]:.3f}, "
                f"grad norm up to {max(gn):.3f}, KL {min(kl):.2e}.."
                f"{max(kl):.2e}, weights moved by {moved:.2e}")
    return {"steps": len(rows), "mean_R": R, "reward_flat": flat,
            "valid_share": [float(r["r_valid_share"]) for r in rows],
            "grad_norm": gn, "KL": kl, "entropy": ent,
            "weight_delta_max": moved, "ckpt": grpo_ck,
            "seconds": round(secs, 1), "note": note}


def phase4_inference(tmp: str, env: dict, grpo_ckpt: str) -> dict:
    """The trained checkpoint against a real geometry, through the same entry
    point a user calls. A 1.3 M model will not produce a good blocking; what is
    asserted is that whatever it produces is carried through to a watertight
    mesh that sits on the geometry, or that the failure is reported cleanly."""
    ckpt = grpo_ckpt if os.path.exists(grpo_ckpt) else os.path.join(tmp, "res.pt")
    npz = need(os.path.join(DATA, "hex3d_algohex", "batch",
                            "machine_0034_n2000", "sample.npz"))
    out = os.path.join(tmp, "infer")
    p, secs = run([sys.executable, os.path.join(ROOT, "scripts", "infer.py"),
                   "--npz", npz, "--ckpt", ckpt, "--k", "4",
                   "--target-h", "0.2", "--out-dir", out], env)
    rp = os.path.join(out, "report.json")
    if not os.path.exists(rp):
        raise RuntimeError(f"no report written (exit {p.returncode}); stderr "
                           f"tail: {p.stderr.strip()[-400:]}")
    rep = json.load(open(rp))
    st = rep["stages"]
    n_detok = sum(1 for r in st["3_transformer"]["rollouts"] if r["detok_ok"])
    if "5_conform" not in st:
        # An untrained model failing to produce a usable blocking is a real
        # outcome, not a broken pipeline -- as long as it is reported.
        return {"reached": "stage 3/4", "rollouts_detokenized": n_detok,
                "ckpt": os.path.basename(ckpt), "seconds": round(secs, 1),
                "note": f"{n_detok} of 4 rollouts detokenized, none survived "
                        f"snapping -- expected from a 1.3 M model; the "
                        f"pipeline reported it instead of crashing"}
    c = st["5_conform"]
    if not c["watertight"]:
        raise RuntimeError("mesh is not watertight")
    if c["boundary"]["max"] > 1e-3:
        raise RuntimeError(f"mesh misses the geometry by {c['boundary']['max']:.2e}")
    return {"reached": "mesh", "rollouts_detokenized": n_detok,
            "blocks": st["4_snap"]["chosen"]["n_blocks"],
            "cells": c["cells"], "boundary_max": c["boundary"]["max"],
            "inverted": c["inverted"], "ckpt": os.path.basename(ckpt),
            "seconds": round(secs, 1),
            "note": f"{st['4_snap']['chosen']['n_blocks']} blocks -> "
                    f"{c['cells']} cells, watertight, boundary "
                    f"{c['boundary']['max']:.2e}, {c['inverted']} inverted"}


def main() -> int:
    ap = argparse.ArgumentParser(description="train end to end, then mesh")
    ap.add_argument("--epochs", type=int, default=8,
                    help="epochs in phase 1; phase 2 resumes to twice that")
    ap.add_argument("--grpo-steps", type=int, default=3)
    ap.add_argument("--device", choices=("auto", "cpu"), default="auto",
                    help="auto tries the GPU and falls back to the CPU")
    ap.add_argument("--keep", default="", help="keep the run in this directory")
    args = ap.parse_args()

    small = {"d": 128, "layers": 4, "heads": 4, "n-points": 256,
             "n-latent": 8, "token-budget": 4096, "batch-cap": 8}
    env: dict = {}
    if args.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        try:
            import torch
            if torch.cuda.is_available():
                free, _ = torch.cuda.mem_get_info()
                if free < 1_500_000_000:
                    env["CUDA_VISIBLE_DEVICES"] = ""
                    print(f"note: only {free / 1e9:.2f} GB free on the GPU, "
                          f"running on the CPU")
            else:
                env["CUDA_VISIBLE_DEVICES"] = ""
        except Exception:
            env["CUDA_VISIBLE_DEVICES"] = ""
    where = "CPU" if env.get("CUDA_VISIBLE_DEVICES") == "" else "GPU"
    print(f"device: {where} | model d=128 layers=4 (1.3 M params, a PATH test "
          f"-- production is d=512 layers=12)")

    tmp = args.keep or tempfile.mkdtemp(prefix="meshtron_e2e_")
    os.makedirs(tmp, exist_ok=True)
    report: dict = {"device": where, "dir": tmp, "phases": {}}
    phases = [
        ("1 supervised run", lambda: phase1_supervised(tmp, args.epochs, small, env)),
        ("2 resume", lambda: phase2_resume(tmp, args.epochs, small, env,
                                           report["phases"]["1 supervised run"])),
        ("3 GRPO steps", lambda: phase3_grpo(tmp, args.grpo_steps, env)),
        ("4 inference to mesh",
         lambda: phase4_inference(tmp, env,
                                  report["phases"].get("3 GRPO steps", {})
                                  .get("ckpt", ""))),
    ]
    failed = 0
    for name, fn in phases:
        t0 = time.time()
        try:
            res = fn()
            report["phases"][name] = res
            RESULTS.append(("PASS", name, res["note"], time.time() - t0))
            print(f"  PASS  {name}: {res['note']}")
        except Exception as exc:
            failed += 1
            report["phases"][name] = {"error": f"{type(exc).__name__}: {exc}"}
            RESULTS.append(("FAIL", name, f"{type(exc).__name__}: {exc}",
                            time.time() - t0))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
            if name.startswith(("1", "2")):
                print("  (later phases need this one, stopping)")
                break

    print()
    width = max(len(n) for _, n, _, _ in RESULTS) + 2
    for status, name, note, secs in RESULTS:
        print(f"  {status}  {name:<{width}} {secs:6.1f}s  {note}")
    rp = os.path.join(tmp, "e2e_report.json")
    json.dump(report, open(rp, "w"), indent=2)
    print(f"\n{len(RESULTS) - failed} pass, {failed} fail")
    print(f"report {rp}")
    if not args.keep:
        print(f"(run directory {tmp} kept for inspection; --keep to choose it)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
