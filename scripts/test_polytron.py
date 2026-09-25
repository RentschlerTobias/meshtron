#!/usr/bin/env python3
"""test_polytron.py -- smoke gates for the Polytron path, data to mesh.

Each phase asserts the claim its stage makes, and runs entry points as a user
would (subprocess, clean path), not by importing them:

  1 representation  24 hex rotations preserve orientation; the canonical form
                    is rotation invariant; encode -> decode returns the block
                    topology exactly and every corner within half a bin
  2 oracle fill     ground-truth tokens -> curved TFI: watertight, boundary
                    points ON the surface (< 1e-6), surface fully covered.
                    This is the gate the weld-id bug of the face projector
                    failed (0.08 off the surface) before it was fixed.
  3 training        every stage's entry point trains on 8 items: the loss
                    falls by half, and the checkpoint loads back
  4 memorisation    each stage overfits ONE structure; the chain then
                    (beam search, production flags: seam cloud, point memory)
                    regenerates it EXACTLY (vertices and blocks) from the
                    geometry alone, and the eval entry point fills it into a
                    watertight mesh on the surface -- the architecture carries
                    a structure end to end

    uv run python scripts/test_polytron.py           # ~3 min on the GPU
Exit 0 all phases pass, 1 otherwise.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

DATA = os.path.join(ROOT, "data", "polytron_blocks_clean.pt")
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)
    return ok


def run(cmd, timeout=1800):
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env,
                       timeout=timeout, cwd=ROOT)
    if p.returncode != 0:
        print(p.stdout[-2000:], p.stderr[-3000:])
    return p, time.time() - t0


def phase_representation():
    import torch
    from meshtron.data.polytron_blocks import (CORNERS, HEX_ROT, PolytronSpec,
                                               canonical_block, decode_seq,
                                               item_seq)
    X = np.array(CORNERS, float)
    ok = True
    for p in HEX_ROT:
        Y = X[p]
        vol = np.linalg.det(np.stack([Y[1] - Y[0], Y[3] - Y[0], Y[4] - Y[0]]))
        ok &= vol > 0
    check("24 rotations preserve orientation", ok and len(HEX_ROT) == 24)
    rng = np.random.default_rng(0)
    b = rng.permutation(100)[:8]
    c0 = canonical_block(b)
    check("canonical block rotation invariant",
          all((canonical_block(b[p]) == c0).all() for p in HEX_ROT)
          and c0[0] == b.min())
    d = torch.load(DATA, weights_only=False)
    spec = PolytronSpec.from_json(d["spec"])
    step = (np.asarray(spec.hi) - np.asarray(spec.lo)) / (spec.q_vert - 1)
    bad = 0
    from meshtron.data.polytron_blocks import load_npz
    for it in d["train"][:40]:
        raw = load_npz(os.path.join(ROOT, "data", "hex3d_algohex", it["dir"],
                                    "sample.npz"))
        V, B, curves = decode_seq(item_seq(it), spec)
        src = it["src"]
        bad += int((np.abs(V - raw["vertices"][src]) > 0.5 * step + 1e-9).any())
        bad += int({frozenset(x) for x in src[B].tolist()}
                   != {frozenset(x) for x in raw["blocks"].tolist()})
        bad += int(2 * len(curves) != len(raw["edges"]))
    check("encode/decode round trip (40 items)", bad == 0, f"{bad} mismatches")
    off = rng.normal(0, 0.2, (1000, 6))
    back = spec.dequant_curve(spec.quant_curve(off))
    err = np.abs(back - np.clip(off, -spec.curve_max, spec.curve_max))
    check("curve offset quantisation", float(np.percentile(err, 99)) < 0.02,
          f"p99 {np.percentile(err, 99):.4f} chord lengths")
    return d, spec


def phase_oracle(d, spec, out):
    from meshtron.data.polytron_blocks import decode_seq, item_seq, load_npz
    from meshtron.geometry.polytron_tfi import (read_hex_vtk, refill_polytron,
                                                surface_fit)
    for it in d["val"][2:5]:
        raw = load_npz(os.path.join(ROOT, "data", "hex3d_algohex", it["dir"],
                                    "sample.npz"))
        V, B, C = decode_seq(item_seq(it), spec, n_edge_pts=64)
        f = os.path.join(out, f"oracle_{it['name']}.vtk")
        r = refill_polytron(V, B, C, 0.1, f, write_edges=False,
                            surface=(raw["surface_points"], raw["surface_tris"],
                                     raw["surface_tri_label"]))
        P, H = read_hex_vtk(f)
        s = surface_fit(P, r["boundary_point_ids"], r["boundary_quads"],
                        raw["surface_points"], raw["surface_tris"])
        inv = r["inverted_curved"] / r["cells_after"]
        check(f"oracle fill {it['name']}",
              r["watertight"] and s["on_surface_max"] < 1e-6
              and s["uncovered_share"] < 0.02 and inv < 0.01,
              f"cells {r['cells_after']} inverted {inv:.2%} "
              f"on-surface max {s['on_surface_max']:.1e} "
              f"uncovered {s['uncovered_share']:.3f}")


def phase_training(out):
    for st in ("vertex", "block", "curve"):
        pre = os.path.join(out, f"tr_{st}")
        p, dt = run([sys.executable, "-m", "meshtron.training.train_polytron",
                     "--stage", st, "--limit", "8", "--val-is-train",
                     "--epochs", "120", "--bs", "8", "--d", "128", "--layers", "2",
                     "--lr", "1e-3", "--dropout", "0", "--eval-every", "20",
                     "--out", pre])
        if not check(f"train entry point {st}", p.returncode == 0, f"{dt:.0f}s"):
            continue
        res = json.loads(p.stdout.strip().splitlines()[-1])
        check(f"{st} loss falls by half",
              res["final_train_loss"] < 0.5 * res["first_train_loss"],
              f"{res['first_train_loss']:.3f} -> {res['final_train_loss']:.3f}")
        from meshtron.training.train_polytron import load_stage
        try:
            m, _spec, ck = load_stage(pre + "_best.pt")
            check(f"{st} best checkpoint loads", ck["stage"] == st,
                  f"epoch {ck['epoch']}")
        except Exception as e:  # noqa: BLE001
            check(f"{st} best checkpoint loads", False, repr(e))


def phase_memorisation(out):
    pre = os.path.join(out, "mem")
    epochs = {"vertex": 500, "block": 300, "curve": 400}
    for st in ("vertex", "block", "curve"):
        p, dt = run([sys.executable, "-m", "meshtron.training.train_polytron",
                     "--stage", st, "--limit", "1", "--val-is-train",
                     "--epochs", str(epochs[st]), "--bs", "1", "--d", "128",
                     "--layers", "3", "--lr", "1e-3", "--dropout", "0",
                     "--eval-every", "50", "--out", f"{pre}_{st}"]
                    + (["--seam-cloud", "--point-memory"] if st == "vertex" else []))
        check(f"memorise one structure: {st}", p.returncode == 0,
              p.stdout.strip().splitlines()[-2] if p.returncode == 0 else "")
    ev = os.path.join(out, "mem_eval")
    p, dt = run([sys.executable, "scripts/eval_polytron.py", "--prefix", pre,
                 "--which", "last", "--split", "train", "--n", "1", "--k", "1",
                 "--beam", "4", "--target-h", "0.1", "--out-dir", ev])
    if not check("eval entry point", p.returncode == 0, f"{dt:.0f}s"):
        return
    s = json.load(open(os.path.join(ev, "summary.json")))
    row, sm = s["rows"][0], s["summary"]
    check("chain regenerates vertices exactly", sm["vertices_exact"] == 1.0)
    check("chain regenerates blocks exactly", sm["blocks_exact"] == 1.0)
    best = row.get("best", {})
    t, f = best.get("tfi") or {}, best.get("surface") or {}
    check("regenerated structure fills into a watertight mesh",
          sm["mesh_generated"] == 1.0 and t.get("watertight"),
          f"cells {t.get('cells_after')} inverted {t.get('inverted_curved')}")
    check("mesh sits on and covers the geometry",
          f.get("on_surface_max", 1) < 1e-6 and f.get("uncovered_share", 1) < 0.02,
          f"on-surface max {f.get('on_surface_max', 1):.1e} "
          f"uncovered {f.get('uncovered_share', 1):.3f}")
    print(f"artifacts: {ev}")


def main() -> int:
    t0 = time.time()
    out = tempfile.mkdtemp(prefix="polytron_smoke_")
    d, spec = phase_representation()
    phase_oracle(d, spec, out)
    phase_training(out)
    phase_memorisation(out)
    n_fail = sum(not ok for _n, ok, _d in RESULTS)
    print(f"\n{len(RESULTS) - n_fail} pass, {n_fail} fail  "
          f"({time.time() - t0:.0f}s)  artifacts in {out}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
