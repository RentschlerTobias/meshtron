"""meshtron -- block structures for turbine passages, from geometry to CFD mesh.

Subpackages:

    geometry   feature model, seam curves, edge routing, transfinite refill
    model      the transformer and its encoders
    data       tokenizers, conditioning, datasets, augmentation
    training   supervised and RL training, rewards, generation, the 2D stack
    viz        plotting, tokenisation animations, the terminal UI
    legacy     earlier generations, kept for reference, not on the active path

Importing this package puts every subpackage directory on sys.path. Several
modules -- among them files that may not be edited, such as trainer.py -- use
bare imports like `from config import PipelineConfig`, and this keeps those
working wherever the module now lives. New code should use the full path,
`from meshtron.geometry import patch_paths`.

Known limitation of the bridge: a module reached both ways -- once as
`quadtron` and once as `meshtron.model.quadtron` -- becomes two distinct module
objects, so classes defined in it fail `isinstance` across the two. Nothing on
the active path does that today; it will go away once the frozen files
(trainer.py, scripts/eval_stop_rate.py, train_hexarow_smoke.py) may be updated
to full paths.
"""
import os as _os
import sys as _sys

_here = _os.path.dirname(_os.path.abspath(__file__))
for _sub in ("geometry", "model", "data", "training", "viz",
             "legacy"):
    _p = _os.path.join(_here, _sub)
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
del _os, _sys, _here, _sub, _p
