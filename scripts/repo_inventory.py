"""repo_inventory.py -- which modules still hang off the active paths.

Static import graph from every module that carries a __main__ guard. Prints
what is reachable, what is orphaned, and which module names look like parallel
generations of the same idea -- the actual source of clutter here, since only
8 of 60 top-level modules are genuinely unused.

  uv run python scripts/repo_inventory.py
  uv run python scripts/repo_inventory.py --graph patch_paths
"""
from __future__ import annotations

import argparse
import ast
import os
from collections import defaultdict, deque

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRS = {"": ".", "scripts.": "scripts", "showcase.": "showcase"}
FAMILIES = ("tokenizer", "tokenization", "train", "half_edge", "polytron",
            "augment", "domain_extractor", "test")


def collect():
    mods = {}
    for pre, d in DIRS.items():
        p = os.path.join(ROOT, d)
        if not os.path.isdir(p):
            continue
        for f in sorted(os.listdir(p)):
            if f.endswith(".py"):
                mods[pre + f[:-3]] = os.path.join(p, f)
    return mods


def imports_of(path):
    try:
        tree = ast.parse(open(path).read())
    except Exception:
        return set()
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                out.add(a.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            out.add(n.module)
            out.add(n.module.split(".")[0])
    return out


def build(mods):
    graph = {m: set() for m in mods}
    for m, p in mods.items():
        for i in imports_of(p):
            if i in mods:
                graph[m].add(i)
            elif "scripts." + i in mods:
                graph[m].add("scripts." + i)
    rev = defaultdict(set)
    for m, ds in graph.items():
        for d in ds:
            rev[d].add(m)
    return graph, rev


def main() -> int:
    ap = argparse.ArgumentParser(description="module reachability")
    ap.add_argument("--graph", default="", help="show importers of one module")
    args = ap.parse_args()
    mods = collect()
    graph, rev = build(mods)
    lines = {m: sum(1 for _ in open(p)) for m, p in mods.items()}
    entries = [m for m, p in mods.items() if "__main__" in open(p).read()]

    if args.graph:
        m = args.graph
        print(f"{m}: {lines.get(m, 0)} lines")
        print(f"  imports  {sorted(graph.get(m, ()))}")
        print(f"  imported by {sorted(rev.get(m, ()))}")
        return 0

    seen, q = set(), deque(entries)
    while q:
        m = q.popleft()
        if m in seen:
            continue
        seen.add(m)
        q.extend(graph.get(m, ()))
    top = [m for m in mods if "." not in m]
    dead = sorted(m for m in top if m not in seen)

    print(f"{len(mods)} modules, {len(entries)} with a __main__ guard")
    print(f"top level: {len(top)}, reachable {len(top) - len(dead)}, "
          f"orphaned {len(dead)}")
    print("\norphaned (no __main__, nobody imports them):")
    for m in sorted(dead, key=lambda m: -lines[m]):
        print(f"   {m:30s} {lines[m]:5d} lines")
    print(f"   {'':30s} {sum(lines[m] for m in dead):5d} total")

    print("\nparallel generations of one idea:")
    for fam in FAMILIES:
        group = sorted(m for m in top if m.startswith(fam))
        if len(group) > 1:
            print(f"   {fam}:")
            for m in group:
                mark = "orphan" if m in dead else f"{len(rev[m])} importers"
                print(f"      {m:28s} {lines[m]:5d} lines  {mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
