# Showcase — walking through meshtron

Five scripts that take the repo apart in the order it actually runs. They are
written as `# %%` cells so they can be sent line by line or cell by cell to a
Python REPL from neovim (iron.nvim, vim-slime, …), or run straight through:

```
uv run python showcase/01_overview.py          # run it
uv run python -i showcase/03_model.py          # run it and stay in the REPL
```

## From neovim, cell by cell

**Start the REPL in the repo root**, so the scripts can find `showcase/`:

```
cd /home/t1dde/hydrostack_pipeline/stack/meshtron
uv run python            # or :IronRepl with python set to `uv run python`
```

Then send cell `[0]` first — it puts `showcase/` on the path and imports
`_common as C`, which every later cell uses — and after that any cell, in
order. Send whole cells, not fragments: a cell is the unit that leaves the
REPL in a usable state.

A REPL is not a file, and two differences matter:

- **There is no `__file__`.** Pasted code is stdin, so `__file__` is undefined
  and `os.path.dirname(os.path.abspath(__file__))` raises `NameError`. Cell
  `[0]` catches that and locates `showcase/` from the working directory
  instead; start the REPL somewhere else and it says so, naming the directory,
  instead of failing later with `ModuleNotFoundError: No module named
  '_common'`.
- **A blank line closes an open block.** So a top-level `with`/`for`/`if` needs
  a blank line after its body before the next dedented statement, and must
  contain none inside it. The scripts are written that way, and

  ```
  uv run python scripts/check_showcase_repl.py
  ```

  checks it — run it after editing a showcase script. It walks every line
  through the same buffer logic as `code.InteractiveConsole` without executing
  anything, so it names the line a REPL would choke on.

Every cell prints shapes, ranges and a few rows of whatever it produced, and
the ones that make geometry write a VTK into `data/showcase/`.

## The two levels

**`01_overview.py` — high level.** The whole chain in eight steps: geometry,
conditioning cloud, ground-truth blocking, tokens, one forward pass, generated
blocking, mapping onto the geometry, quality. Nothing is opened up; it is the
map you hold while reading the others.

**`02` to `05` — low level.** One box each, opened.

| script | what it opens |
|---|---|
| `02_data.py` | `sample.npz` field by field, the block and edge arrays, the cleaning gates, the conditioning cloud, tokenisation and the round trip back to blocks, how the corpus is split |
| `03_model.py` | the module tree, the conditioning path from cloud to one FiLM vector, a forward pass with the residual stream per layer, causality checked by experiment, the structural mask that constrains sampling, one sampling step, the KV cache |
| `04_training.py` | batching by token budget, what `batchify` returns, why the slot id exists, the weighted loss, the hardest positions, one optimiser step with gradient norms per module, the learning-rate schedule |
| `05_mapping.py` | the feature model and its seam curves, snapping corners, routing one edge with its candidates, all 84 edges by kind, and the three different quality questions a mapped mesh has to answer |

## Things worth knowing before you start

**The checkpoint in `_common.py` predates the slot embedding.** Its `cfg` has
no `npt` and no `coords` key, which is how you can tell. It has to be fed with
`slot=None`; feeding it the slot ids it never saw costs about 30 points of
next-token accuracy. `04_training.py` measures both so the difference is
visible rather than asserted. `data/hexarow_h05_model.pt` is the newer model,
trained on the h=0.5 subdivided blockings (68 blocks and 1689 tokens per
sample instead of 12 and 313).

**"On the geometry" and "covers the geometry" are different questions.** The
gate measures the first: every boundary point sits on the npz surface, to
1e-10. That says nothing about whether the geometry is covered — a blocking
that fails to wrap the blade scores perfectly on the first and badly on the
second. `05_mapping.py` asks all three questions, including cell validity,
separately.

**The folded cells are introduced by the mapping, not inherited.** The same
corners refilled with the blocking's own edge polylines give 9 inverted cells;
routed onto the geometry they give 218, all in the first cell layer at the
blade. Cell 6 of `05_mapping.py` writes both meshes so they can be put side by
side in ParaView.

## What gets written

```
data/showcase/02_cloud.vtk            conditioning cloud (01)
data/showcase/03_blocks_gt.vtk        ground-truth blocking (01)
data/showcase/06_blocks_generated.vtk generated blocking (01)
data/showcase/07_cfd_mesh.vtk         mapped mesh, colour by `inverted` (01)
data/showcase/12_cloud.vtk            the cloud back in xyz (02)
data/showcase/15_mapped.vtk           mapped mesh (05)
data/showcase/16_edges.vtk            routed edges, colour by `route_kind` (05)
data/showcase/17_mapped_gtcurve.vtk   same corners, own edge curves (05)
```

## Generating and comparing, from the command line

Two commands that used to live in a stray `showcase.py` at the repo root -- a
file of shell notes with a `.py` extension, which collided with this package
directory:

```
# greedy, writes data/compare_<item>.vtk
uv run python scripts/compare_viz.py \
    --tokens data/hexarow_tokens_family_cart.pt \
    --ckpt data/grpo_cart_step300.pt --idx 687

# stochastic, k rollouts -> <stem>_r<i>.vtk per file
uv run python scripts/compare_viz.py \
    --tokens data/hexarow_tokens_family_cart.pt \
    --ckpt data/grpo_cart_step300.pt --idx 683 \
    --temperature 0.7 --k 4 --out data/gen_test.vtk
```

For a presentation rather than a walkthrough there is
`scripts/make_pipeline_demo.py`, which writes one numbered VTK per pipeline
stage plus a README, and `scripts/make_inverted_debug.py`, which isolates the
folded cells.
