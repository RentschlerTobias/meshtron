"""Inverted cells / quality per part and block of a closed-domain TFI mesh (out dir of blade_insert.py)."""
import sys
import meshio
import numpy as np

NB = {0: (1, 3, 4), 1: (2, 0, 5), 2: (3, 1, 6), 3: (0, 2, 7), 4: (7, 5, 0), 5: (4, 6, 1), 6: (5, 7, 2), 7: (6, 4, 3)}


def cell_sj(X):
    out = []
    for c, (a, b, d) in NB.items():
        e1, e2, e3 = X[:, a] - X[:, c], X[:, b] - X[:, c], X[:, d] - X[:, c]
        out.append(np.einsum("ij,ij->i", np.cross(e1, e2), e3)
                   / (np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) * np.linalg.norm(e3, axis=1)))
    return np.min(out, 0)


def report(d, vtk="mesh_full.vtk"):
    z = np.load(f"{d}/blocks_full.npz"); part = z["part"]
    m = meshio.read(f"{d}/{vtk}"); H = m.cells[0].data; bid = m.cell_data["block_id"][0].ravel()
    s = cell_sj(m.points[H])
    print(f"{d}: cells {len(H)} inverted {(s <= 0).sum()} sj p1 {np.percentile(s, 1):.2f} min {s.min():.2f}")
    for p in np.unique(part):
        c = part[bid] == p
        print(f"   part {p}: cells {c.sum():6d} inverted {(s[c] <= 0).sum():4d} sj p1 {np.percentile(s[c], 1):.2f}")
    bad = bid[s <= 0]
    u, n = np.unique(bad, return_counts=True)
    print("   inverted per block:", dict(zip(u.tolist(), n.tolist())))
    return s, bid


if __name__ == "__main__":
    for d in sys.argv[1:]:
        report(d)
