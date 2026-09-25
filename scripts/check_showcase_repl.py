"""check_showcase_repl.py -- are the showcase scripts sendable to a REPL?

The showcase is meant to be stepped through from neovim (iron.nvim, vim-slime),
which sends source to a REPL LINE BY LINE. A REPL is not a file: it executes
each statement as soon as it is complete, and a blank line closes an open
block. So two things that are perfectly legal in a file break there:

  1. a top-level block followed directly by a dedented line

         with torch.no_grad():
             logits = model(x, pc, fc)
         C.show("x", x, 1)            <- REPL: IndentationError

     because the REPL is still inside the `with` when the dedented line
     arrives. A blank line before it closes the block.

  2. a blank line INSIDE a block -- it ends the statement, and the rest of the
     body then arrives at top level.

This walks each file through codeop.compile_command with the same buffer logic
as code.InteractiveConsole, without executing anything, and fails on any line
a REPL would reject. Run it after editing a showcase script.

    uv run python scripts/check_showcase_repl.py
"""
from __future__ import annotations

import codeop
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def repl_errors(path: str) -> list[tuple[int, int, str, str]]:
    """Lines a line-fed REPL would reject, as (first, last, text, message)."""
    bad: list[tuple[int, int, str, str]] = []
    buf: list[str] = []
    start = 0
    for ln, raw in enumerate(open(path).read().splitlines(), 1):
        if not buf:
            start = ln
        buf.append(raw)
        try:
            complete = codeop.compile_command("\n".join(buf), path, "single")
        except SyntaxError as e:
            bad.append((start, ln, raw, f"{type(e).__name__}: {e.msg}"))
            buf = []
            continue
        if complete is not None:
            buf = []
    if buf and any(l.strip() for l in buf):
        bad.append((start, start + len(buf) - 1, "<end of file>",
                    "block never closed -- the REPL would still be waiting"))
    return bad


def main() -> int:
    paths = sys.argv[1:] or sorted(
        glob.glob(os.path.join(ROOT, "showcase", "*.py")))
    fails = 0
    for p in paths:
        bad = repl_errors(p)
        rel = os.path.relpath(p, ROOT)
        print(f"{rel}: {'OK' if not bad else f'{len(bad)} REPL errors'}")
        fails += bool(bad)
        for s, e, raw, msg in bad:
            print(f"  L{s}-{e}  {msg}\n          | {raw[:72]}")
    print(f"{len(paths) - fails}/{len(paths)} files are REPL-safe")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
