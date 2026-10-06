"""Default locations for the block generator v2 scripts; override with environment variables.

BLOCKGEN_SAMPLES  directory with <name>/sample.npz of the beam-relabelled samples
                  (domain_partition_3D experimentell/hex3d_algohex/relabel/, out/)
BLOCKGEN_AUDIT    audit.csv of those samples (relabel/audit.py); default: <SAMPLES>/../audit.csv
BLOCKGEN_WORK     datasets, runs and chain outputs (default runs/blockgen_v2, git-ignored)
BLOCKGEN_FEATURE_CACHE  FeatureModelV2 cache
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SAMPLES = os.environ.get("BLOCKGEN_SAMPLES", "")
AUDIT = os.environ.get("BLOCKGEN_AUDIT", os.path.join(SAMPLES, os.pardir, "audit.csv") if SAMPLES else "")
WORK = os.environ.get("BLOCKGEN_WORK", os.path.join(ROOT, "runs", "blockgen_v2"))
FEATURE_CACHE = os.environ.get("BLOCKGEN_FEATURE_CACHE", os.path.join(ROOT, "data", "feature_cache_v2"))


def need_samples() -> str:
    if not SAMPLES:
        raise SystemExit("set BLOCKGEN_SAMPLES to the directory with <name>/sample.npz of the relabelled samples")
    return SAMPLES
