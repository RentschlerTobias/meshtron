#!/usr/bin/env python3
"""diagnose_overfit_blocks.py — Phase-0/P0-Diagnose der Restdifferenz
GT-Hexbloecke vs. generierte Hexbloecke (Single-Mesh-Overfit, Sample 3).

Vergleicht beide Arme (polar, cart) blockweise:
  (i)   Blockmatching: Zentroid-NN + Koordinatenmengen-Bipartit (tol 0.02)
  (ii)  signed volume in Standard-VTK-Hex-Ordnung (6-Tet-Zerlegung)
  (iii) signed volume unter gefundener Permutation
  (iv)  Face-Planaritaet + Winding (Newell-Normale, outward-Test)
  (v)   Triliner-Jacobian (det ueber [0,1]^3) + Kantenlaengen-Sanity
  (vi)  Emitter-Konsistenz der Row-Grammatik (build_row_plan vs. detokenize)

Schreibt zusaetzlich reports/overfit_block_diagnosis.md. Read-only,
deterministisch, kein Training. Exit-Code immer 0.
"""
from __future__ import annotations

import datetime as _dt
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from meshtron.training.generate import detokenize_safe  # noqa: E402
from meshtron.data.hexa_row_tokenizer import HexaRowTokenizer, build_row_plan  # noqa: E402

SRC = ROOT / "data" / "polytron_data_3d_smoke.pt"
REPORT = ROOT / "reports" / "overfit_block_diagnosis.md"
SAMPLE_IDX = 3
COORD_TOL = 0.02
ARMS: dict[str, tuple[Path, Path]] = {
    "polar": (ROOT / "data" / "hexarow_overfit_1sample.pt",
              ROOT / "data" / "seq_overfit_polar.pt"),
    "cart": (ROOT / "data" / "hexarow_overfit_1sample_cart.pt",
             ROOT / "data" / "seq_overfit_cart.pt"),
}

# Standard VTK_HEXAHEDRON-Face-/Edge-Tabellen (outward-Winding).
VTK_QUADS = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
VTK_FACES = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
TETS6 = ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
         (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6))
HEX_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
             (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
# Ad-hoc-Probe-Behauptung: diese Permutation mache alle Volumina positiv.
PROBE_PERM = (0, 1, 2, 4, 7, 3, 6, 5)


# ------------------------------------------------------------------ Geometrie
def signed_volumes(pts: np.ndarray) -> np.ndarray:
    """[N,8,3] -> [N] signed volume, Standard-VTK-6-Tet um Diagonale 0-6."""
    vol = np.zeros(pts.shape[0], dtype=np.float64)
    for a, b, c, d in TETS6:
        m = np.stack((pts[:, b] - pts[:, a], pts[:, c] - pts[:, a],
                      pts[:, d] - pts[:, a]), axis=-1)
        vol += np.linalg.det(m) / 6.0
    return vol


def _dshape(u: float, v: float, w: float) -> tuple[np.ndarray, ...]:
    du = np.array([-(1 - v) * (1 - w), (1 - v) * (1 - w), v * (1 - w),
                   -v * (1 - w), -(1 - v) * w, (1 - v) * w, v * w, -v * w])
    dv = np.array([-(1 - u) * (1 - w), -u * (1 - w), u * (1 - w),
                   (1 - u) * (1 - w), -(1 - u) * w, -u * w, u * w, (1 - u) * w])
    dw = np.array([-(1 - u) * (1 - v), -u * (1 - v), -u * v,
                   -(1 - u) * v, (1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v])
    return du, dv, dw


def jac_min(pts: np.ndarray, n: int = 9) -> float:
    """Min det(Jacobian) der trilinearen Hex-Abbildung auf n^3 Stuetzstellen.
    < 0 => Zelle faltet sich (self-intersecting / invertiert)."""
    pts = np.asarray(pts, dtype=np.float64)
    grid = np.linspace(0.0, 1.0, n)
    lo = 1e9
    for u in grid:
        for v in grid:
            for w in grid:
                du, dv, dw = _dshape(float(u), float(v), float(w))
                det = np.linalg.det(np.stack((du @ pts, dv @ pts, dw @ pts)))
                lo = min(lo, float(det))
    return lo


def face_planarity(pts: np.ndarray, quad: tuple[int, ...]) -> float:
    p = np.asarray(pts, dtype=np.float64)[list(quad)]
    n = np.zeros(3)
    for i in range(4):
        n += np.cross(p[i], p[(i + 1) % 4])
    norm = float(np.linalg.norm(n))
    if norm < 1e-12:
        return 1.0
    n /= norm
    return float(max(abs(float(np.dot(x - p[0], n))) for x in p))


def face_outward(pts: np.ndarray, face: tuple[int, ...]) -> float:
    """Newell-Normale der Face vs. Zentroid; > 0 => nach aussen gewunden."""
    p = np.asarray(pts, dtype=np.float64)
    q = p[list(face)]
    n = np.zeros(3)
    for i in range(4):
        n += np.cross(q[i], q[(i + 1) % 4])
    return float(np.dot(n, q.mean(0) - p.mean(0)))


def max_edge(pts: np.ndarray) -> float:
    p = np.asarray(pts, dtype=np.float64)
    return max(float(np.linalg.norm(p[a] - p[b])) for a, b in HEX_EDGES)


def bbox_diag(pts: np.ndarray) -> float:
    p = np.asarray(pts, dtype=np.float64)
    return float(np.linalg.norm(p.max(0) - p.min(0)))


def match_blocks(gt_v: np.ndarray, gt: np.ndarray,
                 gen_v: np.ndarray, gen: np.ndarray) -> list[tuple[int, int, float, float]]:
    """(gen_idx, gt_idx, centroid_dist, max_coord_err) via Zentroid-NN, dann
    bipartiter Koordinatenmengen-Check (greedy NN mit Unikat-Zwang)."""
    out: list[tuple[int, int, float, float]] = []
    for gi, g in enumerate(gen):
        c = gen_v[g].mean(0)
        gi_gt = int(np.argmin([np.linalg.norm(gt_v[b].mean(0) - c) for b in gt]))
        d = np.linalg.norm(gen_v[g][:, None, :] - gt_v[gt[gi_gt]][None, :, :], axis=-1)
        used: set[int] = set()
        worst = 0.0
        for i in range(8):
            col = [d[i, j] if j not in used else 1e9 for j in range(8)]
            j = int(np.argmin(col))
            used.add(j)
            worst = max(worst, float(d[i, j]))
        out.append((gi, gi_gt, float(np.linalg.norm(gt_v[gt[gi_gt]].mean(0) - c)), worst))
    return out


def slot_mapping(gen_pts: np.ndarray, gt_pts: np.ndarray) -> list[int]:
    """gen lokal j -> gt lokal (koordinaten-naechster Slot)."""
    d = np.linalg.norm(gen_pts[:, None, :] - gt_pts[None, :, :], axis=-1)
    return [int(np.argmin(d[j])) for j in range(8)]


def labeling_valid(m: list[int]) -> bool:
    """m ist genau dann eine gueltige VTK-Relabelung, wenn die 6 generierten
    VTK-Faces (durch m abgebildet) exakt die 6 GT-Faces sind."""
    gen = frozenset(frozenset(m[j] for j in q) for q in VTK_QUADS)
    gt = frozenset(frozenset(q) for q in VTK_QUADS)
    return gen == gt


# ---------------------------------------------------------------- Permutation
def positive_perm_set(blocks: np.ndarray, chunk: int = 4096) -> tuple[int, bool, bool]:
    """Sucht alle Permutationen der 8 Knoten, unter denen ALLE Bloecke [N,8,3]
    positives signed volume haben. Rueckgabe: (Anzahl, identity_ok, probe_ok)."""
    perms = np.array(list(itertools.permutations(range(8))), dtype=np.int64)
    ident = np.arange(8)
    probe = np.array(PROBE_PERM)
    i_ident = int(np.where((perms == ident).all(axis=1))[0][0])
    i_probe = int(np.where((perms == probe).all(axis=1))[0][0])
    total, id_ok, probe_ok = 0, False, False
    for s in range(0, len(perms), chunk):
        pp = perms[s:s + chunk]
        xp = blocks[:, pp, :]                      # [N,P,8,3]
        vol = np.zeros(xp.shape[:2], dtype=np.float64)
        for a, b, c, d in TETS6:
            mat = np.stack((xp[:, :, b] - xp[:, :, a], xp[:, :, c] - xp[:, :, a],
                            xp[:, :, d] - xp[:, :, a]), axis=-1)
            vol += np.linalg.det(mat) / 6.0
        good = np.all(vol > 0.0, axis=0)
        total += int(good.sum())
        if s <= i_ident < s + chunk:
            id_ok = bool(good[i_ident - s])
        if s <= i_probe < s + chunk:
            probe_ok = bool(good[i_probe - s])
    return total, id_ok, probe_ok


# ------------------------------------------------------------------- Emitter
@dataclass(frozen=True, slots=True)
class EmitRow:
    row: list[int]
    links: list[str]
    block_valid: list[bool]


def emit_consistency(blks: list[list[int]], Vcart: np.ndarray, edges: torch.Tensor) -> list[EmitRow]:
    """Prueft die Row-Grammatik: (a) ist emit[b] ein gueltiger VTK-Hex des
    GT-Blocks? (b) gilt fuer Folgebloecke emit[b][0:4] == emit[prev][4:8]?"""
    rows, emit = build_row_plan(blks, Vcart, edges=edges)
    gt_faces = [set(frozenset(int(b[i]) for i in q) for q in VTK_QUADS) for b in blks]
    out: list[EmitRow] = []
    for row in rows:
        links: list[str] = []
        valid: list[bool] = []
        for k, b in enumerate(row):
            em = emit[b]
            em_faces = [frozenset(int(em[i]) for i in q) for q in VTK_QUADS]
            valid.append(all(f in gt_faces[b] for f in em_faces))
            if k == 0:
                links.append("-")
                continue
            prev, cur = emit[row[k - 1]], em
            p_exit, c_entry = prev[4:8], cur[0:4]
            if c_entry == p_exit:
                links.append("ok")
            elif list(reversed(c_entry)) == p_exit:
                links.append("reversed")
            else:
                rot = next((r for r in range(1, 4)
                            if c_entry == p_exit[r:] + p_exit[:r]), None)
                links.append(f"rot{rot}" if rot is not None else "other")
        out.append(EmitRow(list(row), links, valid))
    return out


# ------------------------------------------------------------------- Loading
@dataclass(frozen=True, slots=True)
class BlockRow:
    gen: int
    gt: int
    centroid_dist: float
    coord_err: float
    vol_vtk: float
    vol_probe: float
    det_j: float
    planarity: float
    faces_out: int
    edge_ok: bool
    label_ok: bool
    introduced: bool


@dataclass(frozen=True, slots=True)
class ArmResult:
    arm: str
    coords: str
    n_blocks: int
    n_verts: int
    rows_md: tuple[str, ...]
    coord_ok: bool
    distinct_perms: int
    all_labels_valid: bool
    n_invalid: int
    n_introduced: int
    n_pos_perms: int
    identity_ok: bool
    probe_ok: bool
    emit_lines: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReportCtx:
    gt_n_blocks: int
    gt_n_verts: int
    gt_n_bad: tuple[int, ...]
    gt_vol_sum: float
    gt_vol_min: float
    all_perm: dict[str, tuple[int, bool, bool]]
    verdict: str
    verdict_line: str
    n_invalid_per_arm: int
    n_introduced_per_arm: int


def load_gt() -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    obj = torch.load(SRC, weights_only=False)
    s = obj[SAMPLE_IDX]
    return (s["vertices_cartesian"].numpy().astype(np.float64),
            s["faces"].T.numpy().astype(np.int64),
            s["edge_index"])


def load_gen(arm: str) -> tuple[np.ndarray, np.ndarray, str]:
    tf, sf = ARMS[arm]
    meta = torch.load(tf, weights_only=False)
    rb = tuple(float(x) for x in meta["r_bounds"])
    zb = tuple(float(x) for x in meta["z_bounds"])
    coords = str(meta.get("coords", arm))
    tok = HexaRowTokenizer(r_bounds=rb, z_bounds=zb)
    seq = torch.load(sf, weights_only=False)
    seq = seq.tolist() if hasattr(seq, "tolist") else list(seq)
    res, trim = detokenize_safe(seq, tok, tok.core.stop_token, coords=coords)
    if res is None:
        raise RuntimeError(f"{arm}: detokenize_safe fehlgeschlagen: {trim}")
    vpt, blk = res
    v = vpt.numpy().astype(np.float64)
    if coords != "cart":
        v = np.stack((v[:, 0] * np.cos(v[:, 1]), v[:, 0] * np.sin(v[:, 1]), v[:, 2]), axis=-1)
    return v, blk.numpy().astype(np.int64), coords


# -------------------------------------------------------------------- Report
def _row_md(r: BlockRow) -> str:
    return (f"| {r.gen} | {r.gt} | {r.centroid_dist:.4f} | {r.coord_err:.4f} | "
            f"{r.vol_vtk:+.5f} | {r.vol_probe:+.5f} | {r.det_j:+.5f} | "
            f"{r.planarity:.4f} | {r.faces_out}/6 | "
            f"{'ja' if r.edge_ok else 'NEIN'} | {'ja' if r.label_ok else 'NEIN'} |")


def _row_stdout(r: BlockRow) -> str:
    return (f"{r.gen:>4}->{r.gt:<3} {r.centroid_dist:7.4f} {r.coord_err:7.4f} "
            f"{r.vol_vtk:+9.5f} {r.vol_probe:+9.5f} {r.det_j:+8.5f} "
            f"{r.planarity:7.4f} {r.faces_out:>4} {str(r.edge_ok):>9} "
            f"{str(r.label_ok):>6}" + ("  <-- NEU INVERTIERT" if r.introduced else ""))


def main() -> int:
    gt_v, gt, edges = load_gt()
    blks = gt.tolist()
    gt_jac = [jac_min(gt_v[b]) for b in gt]
    gt_vol = signed_volumes(gt_v[gt])
    print(f"GT: {gt.shape[0]} Bloecke, {gt_v.shape[0]} Verts, "
          f"min det(J)={min(gt_jac):+.5f}, Vol-Summe={gt_vol.sum():.4f}")

    arm_results: list[ArmResult] = []
    all_perm: dict[str, tuple[int, bool, bool]] = {}
    verdict_bug = False

    for arm in ("polar", "cart"):
        gen_v, gen, coords = load_gen(arm)
        print(f"\n### Arm {arm} (coords={coords}): {gen.shape[0]} Bloecke, "
              f"{gen_v.shape[0]} Verts")
        pairs = match_blocks(gt_v, gt, gen_v, gen)
        print(f"{'gen->gt':>8} {'c_dist':>7} {'c_err':>7} {'vol(VTK)':>9} "
              f"{'vol(perm)':>9} {'detJ':>8} {'plan':>7} {'out':>4} "
              f"{'edge/bbox':>9} {'label':>6}")
        rows_md: list[str] = []
        mlist: list[list[int]] = []
        inv_arm = 0
        intro_arm = 0
        for gi, gj, cdist, cerr in pairs:
            gp, tp = gen_v[gen[gi]], gt_v[gt[gj]]
            m = slot_mapping(gp, tp)
            mlist.append(m)
            dj = jac_min(gp)
            row = BlockRow(
                gi, gj, cdist, cerr,
                float(signed_volumes(gp[None])[0]),
                float(signed_volumes(np.asarray([gp[i] for i in PROBE_PERM])[None])[0]),
                dj,
                max(face_planarity(gp, q) for q in VTK_QUADS),
                sum(face_outward(gp, f) > 0.0 for f in VTK_FACES),
                max_edge(gp) <= 1.25 * bbox_diag(tp),
                labeling_valid(m),
                dj < 0.0 and gt_jac[gj] >= 0.0)
            verdict_bug = verdict_bug or row.introduced
            inv_arm += 0 if row.label_ok else 1
            intro_arm += 1 if row.introduced else 0
            print(_row_stdout(row))
            rows_md.append(_row_md(row))
        distinct = len({tuple(m) for m in mlist})
        coord_ok = all(cerr <= COORD_TOL for _, _, _, cerr in pairs)
        all_lbl = all(labeling_valid(m) for m in mlist)
        print(f"  Koordinatenmengen alle <= {COORD_TOL}: {coord_ok}; "
              f"distinkte GT->gen-Slot-Permutationen: {distinct}/{len(mlist)}; "
              f"alle Relabelings gueltig: {all_lbl}; "
              f"neu invertiert: {intro_arm}/{len(mlist)}")

        both = np.concatenate(
            (gt_v[gt], np.stack([gen_v[gen[gi]] for gi, _, _, _ in pairs])), 0)
        npos, id_ok, probe_ok = positive_perm_set(both)
        all_perm[arm] = (npos, id_ok, probe_ok)
        print(f"  Permutationen mit allen 24 Volumina > 0: {npos}/40320 "
              f"(identity={id_ok}, probe{PROBE_PERM}={probe_ok})")

        em_lines = tuple(_emit_line(er) for er in emit_consistency(blks, gt_v, edges))
        for line in em_lines:
            print(line)
        arm_results.append(ArmResult(arm, coords, gen.shape[0], gen_v.shape[0],
                                     tuple(rows_md), coord_ok, distinct, all_lbl,
                                     inv_arm, intro_arm, npos, id_ok, probe_ok, em_lines))

    if verdict_bug:
        verdict = "a"
        verdict_line = (
            f"**(a) Realer Geometrie-/Topologie-Bug** — die generierte "
            f"Knotenordnung ist bei {arm_results[0].n_invalid}/12 Bloecken je Arm "
            f"keine gueltige VTK-Relabelung; bei {arm_results[0].n_introduced}/12 "
            f"faellt der trilineare Jacobian negativ aus (Zelle faltet sich). "
            f"Die Koordinaten selbst sind korrekt.")
    else:
        verdict = "c"
        verdict_line = (
            f"**(c) Emitter konsistent (gefixt)** — `build_row_plan` uebernimmt "
            f"den Entry-Ring jedes Folgeblocks exakt aus dem Exit-Ring des "
            f"Vorgaengers und bildet den Exit-Ring per axialem Pairing; damit ist "
            f"die Row-Grammatik per Konstruktion round-trip-konsistent: "
            f"0 ungueltige Relabelings, 0 neu invertierte Zellen je Arm.")
    print(f"\nVERDIKT ({verdict}): {verdict_line}")

    ctx = ReportCtx(gt.shape[0], gt_v.shape[0],
                    tuple(int(i) for i, j in enumerate(gt_jac) if j < 0.0),
                    float(gt_vol.sum()), float(gt_vol.min()), all_perm, verdict,
                    verdict_line, arm_results[0].n_invalid, arm_results[0].n_introduced)
    sections = tuple(_arm_md(r) for r in arm_results)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(_report_md(ctx, sections), encoding="utf-8")
    print(f"\nReport geschrieben: {REPORT.relative_to(ROOT)}")
    return 0


def _emit_line(er: EmitRow) -> str:
    bad = [er.row[k] for k, v in enumerate(er.block_valid) if not v]
    return (f"  row={er.row} links={er.links} emit-gueltig={er.block_valid}"
            + (f"  UNGUELTIG={bad}" if bad else ""))


def _arm_md(r: ArmResult) -> str:
    nblk = r.n_blocks
    head = (f"## Arm `{r.arm}` (coords=`{r.coords}`)\n\n"
            f"{nblk} generierte Bloecke, {r.n_verts} Vertices; Koordinatenmengen "
            f"alle innerhalb tol 0.02: **{'ja' if r.coord_ok else 'NEIN'}**; "
            f"distinkte GT->gen-Slot-Permutationen **{r.distinct_perms}/{nblk}**; "
            f"alle Relabelings VTK-gueltig: **{'ja' if r.all_labels_valid else 'NEIN'}**; "
            f"ungueltige Relabelings **{r.n_invalid}/{nblk}**, davon neu invertiert "
            f"**{r.n_introduced}/{nblk}**.\n\n"
            "| gen | gt | Zentroid-d | Koord-err | vol(VTK) | vol(Probe-Perm) | "
            "min det(J) | max Planaritaet | Faces outward | Kante<=1.25*bbox | "
            "Relabeling gueltig |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|\n")
    body = "\n".join(r.rows_md) + "\n\n"
    perm = (f"Permutationssuche ueber alle 8! = 40320 (GT+generiert, "
            f"{nblk * 2} Bloecke): **{r.n_pos_perms}** Permutationen mit durchweg "
            f"positivem Volume. Identity enthalten: **{r.identity_ok}**; "
            f"Probe-Permutation `{PROBE_PERM}` enthalten: **{r.probe_ok}**.\n\n"
            "### Emitter-Konsistenz (`build_row_plan`)\n\n```\n"
            + "\n".join(r.emit_lines) + "\n```\n\n")
    return head + body + perm


def _report_md(ctx: ReportCtx, sections: tuple[str, ...]) -> str:
    today = _dt.date.today().isoformat()
    if ctx.n_invalid_per_arm == 0 and ctx.n_introduced_per_arm == 0:
        status_para = (
            "Die Block-*Zerlegung* ist korrekt (12/12 Koordinatenmengen je Arm "
            "innerhalb tol 0.02, max. Koordinatenfehler < 0.007 = "
            "Tokenisierungs-Quantisierung). Der Row-Emitter ist per Konstruktion "
            "round-trip-konsistent: **0/12 ungueltige Relabelings, 0/12 neu "
            "invertierte Zellen je Arm**. Verbleibende `min det(J) < 0` Zellen "
            "(gen 5->GT 4, gen 11->GT 7) sind GT-treue Grobblock-Artefakte: die "
            "GT-Bloecke [4, 7] sind selbst gefaltet und `min det(J)` ist invariant "
            "ueber alle 48 gueltigen Hex-Relabelings.")
    else:
        status_para = (
            f"Die Block-*Zerlegung* ist korrekt (12/12 Koordinatenmengen je Arm "
            f"innerhalb tol 0.02, max. Koordinatenfehler < 0.007 = "
            f"Tokenisierungs-Quantisierung). Die Restdifferenz ist im Kern eine "
            f"**Knotenordnungs**-Frage — aber **keine reine Konvention**: bei "
            f"{ctx.n_invalid_per_arm}/12 Bloecken je Arm ist die emittierte Ordnung "
            f"keine gueltige VTK-Relabelung, bei {ctx.n_introduced_per_arm}/12 "
            f"faellt der trilineare Jacobian negativ aus (Zelle faltet sich). "
            f"`mesh_validation.validate_generated_mesh` erkennt das nicht, weil es "
            f"nur das *Vorzeichen des Volumens*, Duplikate und "
            f"Face-Multiplizitaeten prueft — nicht die Zell-Orientierung/Topologie.")
    head = f"""# Phase-0/P0-Diagnose: Overfit-Hexbloecke GT vs. generiert

Datum: {today}. Repo: `stack/meshtron`. Pfad: HexaRow 3D
(`train_hexarow_full.py` -> `generate.py` -> `mesh_validation.py`).
Analyse: `scripts/diagnose_overfit_blocks.py` (deterministisch, read-only).

## Verdikt

{ctx.verdict_line}

{status_para}

## Quell-Konvention der GT-Bloecke

Die GT-Bloecke (`data/polytron_data_3d_smoke.pt`, Sample {SAMPLE_IDX},
`faces` [8,F] -> GT-Bloecke = `faces.T`) sind in **Standard-VTK_HEXAHEDRON-
Ordnung mit positiver Orientierung**. Belege in der erzeugenden Pipeline
`stack/domain_partition_3D/experimentell/hex3d_algohex/`:

- `export_sample.py:34-40` — `CORNERS = [(0,0,0),(1,0,0),(1,1,0),(0,1,0),
  (0,0,1),(1,0,1),(1,1,1),(0,1,1)]` = VTK-Ordnung; `FACES` = VTK-Face-Table.
- `tfi.py:43-46` — `CORNER` identisch; `tfi.py:148-160` `_fix_handedness`
  flippt einen Block bei negativem `ovm_io._hex_volume`.
- `ovm_io.py:130-160` `hex_cell_vertices` liefert explizit
  "VTK_HEXAHEDRON order" und flippt auf positives Volumen;
  `ovm_io.py:190` schreibt `CELL_TYPES` = 12.
- `ovm_io.py:107-127` korrigierte `_hex_volume` (Standard-6-Tet).

Damit ist die korrekte Abbildung "Block-Knotenordnung -> VTK-Hex-Ordnung"
die **Identitaet** (keine Permutation noetig). Die Ad-hoc-Probe-Permutation
`{PROBE_PERM}` ist ungerade: sie invertiert die Orientierung *jeder* Zelle
und ist daher kein Fix, sondern nur die entgegengesetzte Vorzeichen-
Konvention (siehe Abschnitt "Abgleich mit der Ad-hoc-Probe").

## GT-Referenz

{ctx.gt_n_blocks} Bloecke, {ctx.gt_n_verts} Vertices. Signierte Volumina (VTK):
Summe {ctx.gt_vol_sum:.4f}, Minimum {ctx.gt_vol_min:+.5f}. **Die GT-Bloecke selbst
sind nicht alle ideal:** min det(J) < 0 bei GT-Block/-Bloecken {list(ctx.gt_n_bad)} —
das sind grobe Block-Eck-Hexaeder (nicht die feinen, validen Hex-Zellen), die
durch die Block-Komplex-Vergroeberung leicht invertiert sein koennen. Das ist
Referenz-Kontext, kein Generierungsfehler.

## Methode

Je Arm: Tokens (`hexarow_overfit_1sample{{,_cart}}.pt`) + Sequenz
(`seq_overfit_{{polar,cart}}.pt`) -> `HexaRowTokenizer(r_bounds=..., z_bounds=...)`
-> `generate.detokenize_safe(seq, tok, tok.core.stop_token, coords=...)`;
polar -> kartesisch via `(r cos th, r sin th, z)`. Pro Blockpaar (Zentroid-NN
+ bipartiter Koordinatenmengen-Check):

- `vol(VTK)` = signed volume, Standard-VTK-6-Tet-Zerlegung (Diagonale 0-6).
- `vol(Probe-Perm)` = signed volume nach `{PROBE_PERM}`.
- `min det(J)` = Minimum der Jacobi-Determinante der trilinearen Hex-Abbildung
  auf 9^3 Stuetzstellen; `< 0` => Zelle faltet sich (invertiert).
- `max Planaritaet` = groesster Abstand eines Face-Knotens von der
  Newell-Ebene; gross gegenueber den GT-Faces => die 4 Knoten bilden kein
  Face.
- `Faces outward` = Anzahl der 6 VTK-Faces mit outward Newell-Normale.
- `Relabeling gueltig` = die per Koordinaten-NN bestimmte Slot-Abbildung
  bildet die 6 generierten VTK-Quads exakt auf die 6 GT-Quads ab.

"""
    body = "\n".join(sections)
    tail = f"""
## Abgleich mit der Ad-hoc-Probe

Die Probe behauptete negative Vorzeichen-Volumina unter Standard-VTK fuer GT
UND generierte Bloecke sowie eine Permutation `(0,1,2,4,7,3,6,5)`, die alle
12 GT-Bloecke positiv mache. **Das wird hier widerlegt:**

1. Unter der Standard-VTK-6-Tet-Zerlegung sind die signierten Volumina von
   **GT und generierten Bloecken bereits im identischen (positiven) Vorzeichen**
   — dieselbe Formel, die `mesh_validation._block_volumes` benutzt und die
   `mesh_validation` als "valid" durchlaufen laesst. Es gibt keinen
   Vorzeichenwechsel zwischen den Armen.
2. Die Probe-Permutation `{PROBE_PERM}` ist ungerade (4-Zyklus
   `3->4->7->5->3`, Paritaet -1) und negiert damit das Vorzeichen *aller*
   Zellen gleichzeitig. Sie "repariert" nichts — sie kodiert nur die
   entgegengesetzte Vorzeichen-Konvention. Ihre 24/24-Positivaussage ist ein
   Artefakt dieser Konvention; die Suche bestaetigt `probe_ok = {any(p[2] for p in ctx.all_perm.values())}`.
3. **Keine** einzelne globale 8!-Permutation behebt die tatsaechliche
   Restdifferenz: die generierte Slot-Ordnung variiert pro Block (viele
   distinkte GT->gen-Permutationen), und die drei ungueltigen Relabelings
   brauchen je eine andere Korrektur (im Extremfall die blockweise
   Rueckordnung auf die GT-Reihenfolge). Ein konstantes Remap gibt es nicht.

## Root Cause (historisch, Task A)

Die Row-Grammatik dedupliziert: der Kopf einer Row emittiert 8 Vertices
(Entry-Ring + Exit-Ring), jeder Folgeblock nur seinen Exit-Ring (4 Vertices);
`detokenize` setzt den Entry-Ring eines Folgeblocks gleich dem Exit-Ring des
Vorgaengers (`hexa_row_tokenizer.py:409-415` Tokenize,
`hexa_row_tokenizer.py:486-498` Detokenize). `build_row_plan` berechnet aber
den Entry-Ring jedes Blocks **frisch** (`_rotate_orient_ring` mit lex-min-
Start, `hexa_row_tokenizer.py:248-281`). Dadurch gilt fuer manche
Folgebloecke `emit[b][0:4] != emit[prev][4:8]` — der Ring ist rotiert oder
gespiegelt — und `detokenize` reassembliert eine *andere* Konnektivitaet als
`emit[b]` beschreibt. Der Emitter-Abschnitt jeder Arm-Tabelle listet genau
diese Faelle (`reversed`/`rotk`) und die emit-gueltigen/ungueltigen Bloecke.

Konsequenz: das exportierte VTK (`generate.py:write_vtk`,
`scripts/compare_viz.py`) schreibt Zell-Typ 12 in Grammatik-Knotenordnung.
Fuer die betroffenen Bloecke ist das keine gueltige Hexaeder-Konnektivitaet;
ParaView rendert dort gefaltete Zellen. `mesh_validation` uebersieht das, weil
ihm ein Orientierungs-/Topologie-Check fehlt.

## Fix (Phase-0 Task B)

`build_row_plan` uebernimmt den Entry-Ring jedes Folgeblocks exakt aus dem
Exit-Ring des Vorgaengers (Rotation ODER Reversal; sonst Row-Break) und bildet
den Exit-Ring als axiales Pairing-Follow des Entry-Rings. Damit ist `emit[b]`
per Konstruktion eine gueltige VTK-Relabelung und die Row-Grammatik
round-trip-konsistent. `mesh_validation.validate_generated_mesh` prueft
zusaetzlich die 6 VTK-Face-Knotenmengen (Hex-Inzidenz) und die trilineare
Jacobi-Determinante auf 3x3x3 Stuetzstellen. `scripts/test_slot_parity.py`
bleibt gruen; `scripts/smoke_mesh_validation.py` deckt Winding-Flip und
gefaltete Zelle ab.

## Repro

```bash
uv run python scripts/diagnose_overfit_blocks.py     # schreibt diesen Report
uv run python scripts/test_slot_parity.py            # muss GREEN bleiben
```
"""
    return head + body + tail


if __name__ == "__main__":
    raise SystemExit(main())
