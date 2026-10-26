"""ParaView multiblock (.vtm) of a blocking result: core / O-ring block edges (cell data edge = "a-b" index,
on_wall), crossing points of edges on hub and shroud, wall surfaces, and the TFI mesh.
    python make_vtm.py <blocking out dir> <msh>"""
import os
import sys
import meshio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import msh_walls as MW

d, msh = sys.argv[1], sys.argv[2]
z = np.load(f"{d}/blocks_full.npz"); V, B, part = z["V"], z["B"], z["part"]
keys = [tuple(int(x) for x in k) for k in z["curve_keys"]]; off = z["curve_off"]; pts = z["curve_pts"]
cur = {k: pts[off[i]:off[i + 1]] for i, k in enumerate(keys)}
HE = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
epart = {}
for bi, b in enumerate(B):
    for i, j in HE:
        k = (min(int(b[i]), int(b[j])), max(int(b[i]), int(b[j])))
        epart[k] = max(epart.get(k, 0), int(part[bi]))
sub = os.path.join(d, "vtm"); os.makedirs(sub, exist_ok=True)
blocks = []


def lines(name, items):
    P, L, eid, a_, b_, wall = [], [], [], [], [], []
    base = 0
    for k, Q in items:
        n = len(Q); P.append(Q); L.append(np.stack([base + np.arange(n - 1), base + np.arange(1, n)], 1))
        r = np.hypot(Q[:, 0], Q[:, 1])
        w = 1 if np.all(np.abs(r - 0.5) < 3e-3) else (2 if np.all(np.abs(r - 1.9) < 3e-3) else 0)
        eid += [len(eid)] * 0
        a_.append(np.full(n - 1, k[0])); b_.append(np.full(n - 1, k[1])); wall.append(np.full(n - 1, w))
        base += n
    meshio.Mesh(np.concatenate(P), [("line", np.concatenate(L))],
                cell_data={"corner_a": [np.concatenate(a_).astype(np.int32)],
                           "corner_b": [np.concatenate(b_).astype(np.int32)],
                           "on_wall": [np.concatenate(wall).astype(np.int32)]}).write(f"{sub}/{name}.vtu")
    blocks.append(name)


lines("edges_core", [(k, Q) for k, Q in cur.items() if epart.get(k) == 0])
lines("edges_oring", [(k, Q) for k, Q in cur.items() if epart.get(k, -1) >= 1])
cid = np.arange(len(V))
meshio.Mesh(V, [("vertex", cid[:, None])], point_data={"corner": cid.astype(np.int32)}).write(f"{sub}/corners.vtu")
blocks.append("corners")
# crossings on hub / shroud (same test as plot_wall.py)
X3 = []
for rw in (0.5, 1.9):
    on = {k: Q for k, Q in cur.items() if k in epart and np.all(np.abs(np.hypot(Q[:, 0], Q[:, 1]) - rw) < 3e-3)}
    segs = [(k, Q[i], Q[i + 1]) for k, Q in on.items() for i in range(len(Q) - 1)]
    uv = lambda X: np.array([np.arctan2(X[1], X[0]) * rw, X[2]])
    A = np.array([uv(s[1]) for s in segs]); Bp = np.array([uv(s[2]) for s in segs]); kk = [s[0] for s in segs]
    A3 = np.array([s[1] for s in segs]); B3 = np.array([s[2] for s in segs])
    for i in range(len(segs)):
        p, r = A[i], Bp[i] - A[i]; s_ = Bp - A
        den = r[0] * s_[:, 1] - r[1] * s_[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((A[:, 0] - p[0]) * s_[:, 1] - (A[:, 1] - p[1]) * s_[:, 0]) / den
            u = ((A[:, 0] - p[0]) * r[1] - (A[:, 1] - p[1]) * r[0]) / den
        m = (np.abs(den) > 1e-14) & (t > 1e-6) & (t < 1 - 1e-6) & (u > 1e-6) & (u < 1 - 1e-6)
        for j in np.flatnonzero(m):
            if j > i and not (set(kk[i]) & set(kk[j])):
                X3.append(A3[i] + t[j] * (B3[i] - A3[i]))
if X3:
    X3 = np.array(X3)
    meshio.Mesh(X3, [("vertex", np.arange(len(X3))[:, None])]).write(f"{sub}/crossings.vtu"); blocks.append("crossings")
P, T, L, info, rt = MW.load_regions(msh)
names = {}
for i, dd in enumerate(info):
    if dd["kind"] == "phys":
        names[i] = {4: "hub", 5: "shroud", 6: "blade", 0: "inlet", 1: "outlet", 2: "periodic_a", 3: "periodic_b"}.get(dd["phys"], f"phys{dd['phys']}") + f"_r{dd['regions'][0]}"
for i, nm in names.items():
    if not any(w in nm for w in ("hub", "shroud", "blade")):
        continue
    Ti = T[L == i]; u = np.unique(Ti); ix = {v: j for j, v in enumerate(u)}
    meshio.Mesh(P[u], [("triangle", np.vectorize(ix.get)(Ti))]).write(f"{sub}/wall_{nm}.vtu"); blocks.append(f"wall_{nm}")
m = meshio.read(f"{d}/mesh_full.vtk")
meshio.Mesh(m.points, [("hexahedron", m.cells[0].data)], cell_data={k: [np.asarray(v[0])] for k, v in m.cell_data.items()}
            ).write(f"{sub}/mesh.vtu"); blocks.append("mesh")
with open(f"{d}/result.vtm", "w") as f:
    f.write('<?xml version="1.0"?>\n<VTKFile type="vtkMultiBlockDataSet" version="1.0">\n<vtkMultiBlockDataSet>\n')
    for i, b in enumerate(blocks):
        f.write(f'  <DataSet index="{i}" name="{b}" file="vtm/{b}.vtu"/>\n')
    f.write('</vtkMultiBlockDataSet>\n</VTKFile>\n')
print("wrote", f"{d}/result.vtm", blocks)
