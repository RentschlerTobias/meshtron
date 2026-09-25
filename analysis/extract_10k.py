"""
extract_10k.py

Batch-Extraktion: laeuft domain_extractor.extract_mesh_data ueber ALLE
quad_domain_data/*.pt (102 Dateien, ~10014 Meshes) und schreibt einen
einzigen vorverarbeiteten Datensatz domain_data_10k.pt (Tokenizer-Eingabe).

Nutzung:
    python extract_10k.py                 # voll
    python extract_10k.py --limit-files 1 # Smoke-Test (1 Datei)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import glob
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import torch
from tqdm import tqdm

from meshtron.data.domain_extractor import extract_mesh_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=os.path.join(_HERE, "quad_domain_data"))
    ap.add_argument("--out", default=os.path.join(_HERE, "domain_data_10k.pt"))
    ap.add_argument("--limit-files", type=int, default=0, help="0 = alle")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.in_dir, "*.pt")))
    if args.limit_files > 0:
        files = files[: args.limit_files]
    print(f"{len(files)} Dateien in {args.in_dir}")

    processed = []
    n_ok = n_fail = 0
    for f in tqdm(files, desc="files"):
        meshes = torch.load(f, weights_only=False)
        if not isinstance(meshes, list):
            meshes = [meshes]
        for m in meshes:
            try:
                processed.append(extract_mesh_data(m))
                n_ok += 1
            except Exception as e:  # degenerierte Kante / fehlendes Feld
                n_fail += 1
                if n_fail <= 5:
                    print(f"  skip mesh in {os.path.basename(f)}: {type(e).__name__}: {e}")

    print(f"OK={n_ok}  FAIL={n_fail}  total_out={len(processed)}")
    torch.save(processed, args.out)
    sz = os.path.getsize(args.out) / 1e6
    print(f"Saved {args.out} ({sz:.1f} MB)")

    d = processed[0]
    print("sample:",
          "n_verts", d['vertices_polar'].shape[0],
          "n_faces", d['faces'].shape[1],
          "n_edges", d['edge_index'].shape[1],
          "n_points", d['tri_coordinates'].shape[0])


if __name__ == "__main__":
    main()
