# Handoff: Apptainer sandbox setup (institute GPU machine)

Session interrupted mid-tutorial at user's request ("mach ein handoff, neue session").
Continue as a step-by-step tutorial (see "How to continue" below) -- do not just
implement it for the user.

## Goal

User wants to run coding agents on an institute GPU machine (AMD MI300A APUs,
SLURM-managed) without giving the agent access to the shared institute
filesystem (other employees' data lives there). Needs GPU/SLURM access from
inside the sandbox. Also wants the whole pipeline installable by colleagues
via a single `install.sh`.

## Decisions made

- **Apptainer over Enroot**: Enroot's automatic GPU passthrough is built on
  NVIDIA Container Toolkit (NVIDIA-only). Apptainer has a native `--rocm`
  flag for AMD GPUs -- direct match for MI300A hardware. Enroot also needs
  the Pyxis SLURM plugin (admin-installed, not default) for its ergonomic
  `srun --container-*` syntax; Apptainer needs no SLURM-side setup.
- **Isolation approach**: `apptainer exec --rocm --containall --no-home
  --bind <workspace>:/workspace --bind <agent-home>:/home/agent --env
  HOME=/home/agent ...` -- `--no-home` stops Apptainer's default (!) behavior
  of auto-binding the real `$HOME`; `--containall` blocks other implicit
  host binds/env. Agent gets a dedicated fake `$HOME` inside the workspace
  for its own config/credentials (`~/.claude/`, a **separate** SSH deploy
  key -- not the user's personal key) so nothing leaks back into the real
  home directory.
- **Workspace, not `$HOME`, as the bind target**: institute home dirs
  typically have small quotas; scratch space is meant for this kind of
  working data.
- **Own repos vs. third-party C++ repos -- different treatment**:
  - The user's 3 own repos (actively developed, change constantly) are
    bind-mounted live from the workspace, cloned by `install.sh` on the
    host side. Baking them into the image would force a rebuild on every
    code change.
  - Two third-party C++ dependencies, **dtoo** and **AlgoHex**, are stable
    and need a compiler toolchain -- these get cloned + compiled inside the
    Apptainer `.def` file's `%post` section, baked into the image at build
    time (`apptainer build`). End users never need a working C++ toolchain
    themselves. Not yet determined: dtoo/AlgoHex's build system (presumably
    CMake) and system library dependencies (Eigen/CGAL/Boost/etc. are the
    typical suspects for hex-meshing C++ libraries, unconfirmed) -- still
    need to read their READMEs/CMakeLists.txt to fill in `%post` for real.

## Concrete facts gathered this session

- Workspace: `~/ws_freq_aware_cfd` (symlink) -> `/mnt/tscratch/trentschler-ws_freq_aware_cfd`
  (scratch dir). Already set up; an earlier self-referencing symlink loop
  inside it was diagnosed and fixed.
- The 3 repos and their SSH remotes (read directly from the sibling
  checkouts under `/root/repos/duty/quadmesh/` and `/root/repos/duty/` in
  this dev environment):
  - `quadtron` -- `git@github.com:RentschlerTobias/quadtron.git` (this repo,
    aka "meshtron")
  - `domain_partition_3D` -- `git@github.com:RentschlerTobias/domain_partition_3D.git`
  - `eigenfrequencies` -- `git@github.com:RentschlerTobias/eigenfrequencies.git`
- Apptainer is available via `zypper` on the target machine (openSUSE/SLE)
  but install status unconfirmed -- `sudo zypper install apptainer` needs
  to be run (needs root/sudo; if the user doesn't have it, that's an admin
  ask).
- SLURM `--gres` label for the MI300A nodes is unconfirmed -- NVIDIA's
  `gpu:1` convention may not apply; check `sinfo`/cluster docs before the
  first real sbatch submission.
- Separately noted, not yet acted on: this repo's `origin` remote is HTTPS
  (`https://github.com/RentschlerTobias/quadtron.git`), which is why `git
  push` prompts for a username/password instead of using the user's
  existing SSH key (`~/.ssh/id_ed25519`). Offered to switch via `git remote
  set-url origin git@github.com:RentschlerTobias/quadtron.git` but not
  confirmed/done -- separate from the sandbox work, resolve whenever it
  comes up again.

## What was explained already (tutorial progress)

1. Workspace directory layout + idempotent repo cloning (`mkdir -p`,
   `[ -d name ] || git clone ...`) -- **user confirmed understanding**.
2. Why the 3 own repos are bind-mounted (host-side `git clone` in
   `install.sh`) while dtoo/AlgoHex are baked into the image -- **user
   confirmed understanding** (paraphrased correctly: their own repos change
   constantly, baking them in would force a rebuild every time).
3. Started explaining `.def` file `%post` semantics (build-time vs.
   run-time) -- **interrupted before the check-in question was answered**,
   this is where to resume.

A draft `install.sh` (workspace dirs + clone step only) was written directly
into this repo's root at one point, then **deleted at the user's explicit
request** -- the user wants to type/implement this themselves, with Claude
as tutor only. See project memory `feedback_tutorial_not_implementation.md`
in the memory store for this instruction; it should already be picked up
automatically in the new session.

## How to continue (new session)

Resume the tutorial as a step-by-step walkthrough, one small concept per
turn, ending each turn with a single check-in question -- do not write
`install.sh` or the `.def` file for the user. Suggested next steps in order:

1. Finish the `.def` file `%post` explanation (build-time vs. run-time
   commands) that was interrupted.
2. Ask the user for dtoo/AlgoHex's build system and system dependencies
   (or have them paste the READMEs/CMakeLists.txt) before writing any
   concrete `%post` compile commands.
3. Walk through the rest of `install.sh`: Apptainer-availability check,
   `agent-home` directory + dedicated SSH deploy key generation, triggering
   `apptainer build`.
4. Walk through the `--bind`/`--containall`/`--no-home`/`--env HOME=...`
   `apptainer exec` invocation and how to verify isolation actually works
   (e.g. confirm the agent process cannot see anything outside the bound
   paths).
5. SLURM wrapper script, once the `--gres` label for MI300A is confirmed.

## Unrelated, still-open item in this repo

Separately from the sandbox work: this repo (`quadtron`/"meshtron") has 14
local commits on `main` (renaming Meshtron/Plan-B to Quadtron/Polytron,
adding RL curriculum infra, 3D dataset support, the `tui.py` TUI, and a
repo-root cleanup into `deprecated/`, `analysis/`, `viz/`, `data/`,
`slurm/`, `docs/notes/`, `literatur/`) that are **not yet pushed** to
`origin/main`. Push was not yet confirmed by the user -- ask before running
`git push`.
