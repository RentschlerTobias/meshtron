#!/usr/bin/env python3
"""test_slot_parity.py — TDD-Regressionslock fuer die Slot-Embedding-Konvention.

BUG (verifiziert gegen Live-Code, .omo/runs/20260921-overfit/NOTEPAD.md):
  TRAIN `batchify` (train_hexarow_full.py:159) setzt slot[i, 1:len] auf
  _slot_ids(tk[:-1]) -> Input-Token an Pos j traegt SLOT DES VORHERIGEN Tokens
  s_id(tk[j-1]) (LAG).  INFER `generate` (generate.py:148) fuettert cnt%npt mit
  cnt = non-special inkl. seq[-1] -> SLOT DES NAECHSTEN Tokens (LEAD +1 mod npt).
  Free-running Decode sieht also nie ein Slot-Embedding, das es im Training sah.

ZIEL-KONVENTION (D1 im Notepad): slot[j] = EIGENES s_id(tk[j]) des Input-Tokens,
  train + infer identisch.  s_id_own(t) = 0 fuer specials (cnt-Reset), sonst
  cnt%npt mit cnt+=1.  Inferenz-Feed: feed_slot = 0 wenn cnt==0 sonst (cnt-1)%npt.

Dieser Test LAEUFT ROT gegen den aktuellen Code (Assertions (a) batchify-lag
und (d) _unit_weights-lag schlagen fehl) und wird nach dem Fix gruen.  Er
definiert NUR die Konvention; er aendert keinen Code.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from meshtron.data.hexa_row_tokenizer import HexaRowTokenizer
from meshtron.training.train_hexarow_full import _slot_ids, _unit_weights, batchify


def _specials() -> set[int]:
    """Spiegelt train_hexarow_full.main() (L293-297): die 6 Sonder-Ids aus dem
    Tokenizer-Core."""
    core = HexaRowTokenizer().core
    return {core.start_token, core.end_token, core.sep_token,
            core.sep2_token, core.stop_token, core.pad_token}


def _own_slots(tokens: list[int], specials: set[int], npt: int) -> list[int]:
    """Ziel-Konvention: eigenes s_id jedes Tokens (specials -> 0 + cnt-Reset)."""
    s: list[int] = []
    cnt = 0
    for t in tokens:
        if t in specials:
            s.append(0)
            cnt = 0
        else:
            s.append(cnt % npt)
            cnt += 1
    return s


def _simulate_feed(tk: list[int], specials: set[int], npt: int) -> tuple[list[int], list[int]]:
    """Simuliert generate.py decode-Schleife (L143-174) ueber bekannter Sequenz tk.

    Beim Fuettern von tk[j] zaehlt cnt tk[j] selbst mit (cnt wird VOR dem
    naechsten Sampling aktualisiert: special->0 sonst +1).  Liefert pro
    gefuetterter Position j (0..len-2, stop wird nur gesampelt, nie gefuettert):
      cur[j]   = cnt % npt        (HEUTIGE Formel, generate.py:148)
      fixed[j] = 0 wenn cnt==0 sonst (cnt-1) % npt  (Ziel-Formel)
    """
    cur: list[int] = []
    fixed: list[int] = []
    cnt = 0
    for j in range(len(tk) - 1):
        cur.append(cnt % npt)
        fixed.append(0 if cnt == 0 else (cnt - 1) % npt)
        nxt = tk[j + 1]
        cnt = 0 if nxt in specials else cnt + 1
    return cur, fixed


def _expect(failures: list[str], cond: bool, msg: str) -> None:
    """Plain assert in try/except: sammelt ALLE RED-Meldungen statt beim ersten
    Fehler abzubrechen."""
    try:
        assert cond, msg
    except AssertionError as e:
        failures.append(str(e))


def main() -> int:
    failures: list[str] = []
    core = HexaRowTokenizer().core
    specials = _specials()
    pad_id = core.pad_token
    start, sep, stop = core.start_token, core.sep_token, core.stop_token

    # Vocab-Ranges (polytron_tokenizer.py: off_r=0, off_ts=512, off_tc=768,
    # Qr=512, Qa=256): r:[0,512) sin:[512,768) cos:[768,1024) z:[0,512).
    # Polar: 4 Tokens/Vertex (r,sin,cos,z); Cart: 3 Tokens/Vertex (x,y,z) in [0,512).
    V0, V1, V2 = [10, 522, 778, 30], [20, 532, 788, 40], [50, 542, 798, 60]
    tk = [start, *V0, *V1, sep, *V2, stop]          # 15 Tokens (2 Vertices + 1)
    tk_short = [start, *V0, stop]                   # 6 Tokens -> erzwingt Padding
    C0, C1, C2 = [11, 21, 31], [12, 22, 32], [13, 23, 33]
    tk3 = [start, *C0, *C1, sep, *C2, stop]         # cart, 12 Tokens, npt=3

    E = _own_slots(tk, specials, 4)                 # own-slot Ziel (npt=4)
    E3 = _own_slots(tk3, specials, 3)               # own-slot Ziel (npt=3)
    # Sanity: lokale Konvention deckt sich mit der echten _slot_ids (npt=4).
    _expect(failures, E == _slot_ids(tk, specials),
            f"_own_slots(4) != _slot_ids: {E} != {_slot_ids(tk, specials)}")

    n = len(tk)
    pts = np.zeros((4, 4), dtype=np.float64)        # Dummy-Punktwolke (batchify braucht sie)
    items = [{"tokens": tk, "points": pts, "blocks": 3},
             {"tokens": tk_short, "points": pts, "blocks": 1}]

    # --- (a) batchify: slot muss OWN-slot sein, nicht LAG (tk[:-1]) ----------
    x, slot, w, _, _ = batchify(items, pad_id, "cpu", specials, 1.0)
    _expect(failures, bool((slot[0, :n] == torch.as_tensor(E, dtype=torch.long)).all()),
            "batchify slot ist LAGGED: slot[j]=s_id(tk[j-1]) statt OWN s_id(tk[j]); "
            f"slot[0]={slot[0, :n].tolist()}  own={E}")

    # --- (b) generate-Feed: Ziel-Formel == own-slot, HEUTE-Formel nicht -------
    cur, fixed = _simulate_feed(tk, specials, 4)
    fed = range(n - 1)                              # stop wird nie gefuettert
    _expect(failures, all(fixed[j] == E[j] for j in fed),
            "FIXED Feed-Formel (0 if cnt==0 else (cnt-1)%4) muss own-slot ergeben; "
            f"fixed={fixed}  E[:{n - 1}]={E[:n - 1]}")
    _expect(failures, any(cur[j] != E[j] for j in fed),
            "HEUTIGE Feed-Formel cnt%4 muss sich von own-slot unterscheiden "
            "(Bug-Doku, bleibt nach Fix gruen — betrifft die Formel, nicht den Code): "
            f"cur={cur}  E[:{n - 1}]={E[:n - 1]}")

    # --- (c) cart npt=3: gleiche zwei Feed-Checks, batchify fehlt npt-Param ---
    cur3, fixed3 = _simulate_feed(tk3, specials, 3)
    fed3 = range(len(tk3) - 1)
    _expect(failures, all(fixed3[j] == E3[j] for j in fed3),
            "FIXED Feed-Formel (npt=3) muss own-slot ergeben; "
            f"fixed3={fixed3}  E3[:{len(tk3) - 1}]={E3[:len(tk3) - 1]}")
    _expect(failures, any(cur3[j] != E3[j] for j in fed3),
            "HEUTIGE Feed-Formel cnt%3 muss sich von own-slot unterscheiden (Bug-Doku): "
            f"cur3={cur3}  E3[:{len(tk3) - 1}]={E3[:len(tk3) - 1]}")
    items3 = [{"tokens": tk3, "points": pts, "blocks": 3}]
    try:
        _, slot3, _, _, _ = batchify(items3, pad_id, "cpu", specials, 1.0, npt=3)
    except TypeError:
        failures.append("batchify hat keinen npt-Parameter (cart-Pfad nicht pruefbar)")
    else:
        _expect(failures, bool((slot3[0, :len(tk3)] == torch.as_tensor(E3, dtype=torch.long)).all()),
                "batchify(npt=3) slot muss OWN-slot cnt%3 sein; "
                f"slot3[0]={slot3[0, :len(tk3)].tolist()}  own={E3}")

    # --- (d) _unit_weights: Gewicht keyt auf OWN quant-end, nicht Vor-Token ---
    _, _, w2, _, _ = batchify(items, pad_id, "cpu", specials, 0.5)
    W_own = [0.5 if e == 3 else 1.0 for e in E]     # high iff s_id_own==npt-1
    _expect(failures, bool((w2[0, :n] == torch.as_tensor(W_own)).all()),
            "_unit_weights LAG: w[j] keyt auf s_id(tk[j-1]) statt OWN quant-end; "
            f"w[0]={w2[0, :n].tolist()}  own={W_own}")

    # ---------------------------------------------------------------------
    if failures:
        print("RED — Slot-Parity-Konvention verletzt:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("GREEN — Slot-Parity-Konvention erfuellt (eigener s_id je Input-Token).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
