"""2D plot (unrolled: x = theta * r_wall, y = z) of the block edges on the hub (or shroud): source edges on the
layer cut, their images on the wall, the O-grid footprint outline from mesh.msh; image edges that intersect
each other (except at shared corners) are marked."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/opt/stack/meshtron/experiments/flow_topology/blade")
import msh_walls as MW

d, part_id, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
name = sys.argv[4] if len(sys.argv) > 4 else "tistos_s1_0131"
z = np.load(f"{d}/blocks_full.npz"); V, B, part = z["V"], z["B"], z["part"]
keys = [tuple(k) for k in z["curve_keys"]]; off = z["curve_off"]; pts = z["curve_pts"]
cur = {k: pts[off[i]:off[i + 1]] for i, k in enumerate(keys)}
rw = 0.5 if part_id == 1 else 1.9


def uv(Q):
    Q = np.asarray(Q); return np.c_[np.unwrap(np.arctan2(Q[:, 1], Q[:, 0])) * rw, Q[:, 2]]


top, bot = {}, {}
for bi in np.flatnonzero(part == part_id):
    b = B[bi]
    for i, j in ((0, 1), (1, 2), (2, 3), (3, 0)):
        k = tuple(sorted((int(b[i]), int(b[j])))); kb = tuple(sorted((int(b[i + 4]), int(b[j + 4]))))
        top[k] = cur[k]; bot[kb] = cur[kb]
# ring blocks' wall-side edges on this wall: ring part = max part; their faces on the wall
ring_p = part.max(); ring_on = {}
for bi in np.flatnonzero(part == ring_p):
    b = B[bi]
    for i, j in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)):
        k = tuple(sorted((int(b[i]), int(b[j]))))
        Q = cur.get(k)
        if Q is not None and np.all(np.abs(np.hypot(Q[:, 0], Q[:, 1]) - rw) < 2e-3):
            ring_on[k] = Q
P, T, L, info, rt = MW.load_regions(f"/opt/stack/meshtron/experiments/flow_topology/datagen/runs/{name}/case/mesh.msh")
fig, ax = plt.subplots(1, 2, figsize=(16, 9))
for a, (title, E) in zip(ax, (("source edges on the layer cut", top), ("images on the wall", bot))):
    for k, Q in E.items():
        X = uv(Q); a.plot(X[:, 0], X[:, 1], "b-", lw=0.8)
    for k, Q in ring_on.items():
        if title.startswith("images"):
            X = uv(Q); a.plot(X[:, 0], X[:, 1], "r-", lw=1.2)
    a.set_title(title); a.set_aspect("equal"); a.set_xlabel("theta * r"); a.set_ylabel("z")
# intersections between image edges
segs = []
for k, Q in list(bot.items()) + list(ring_on.items()):
    X = uv(Q); segs += [(k, X[i], X[i + 1]) for i in range(len(X) - 1)]
A = np.array([s[1] for s in segs]); Bp = np.array([s[2] for s in segs]); kk = [s[0] for s in segs]
hits = []
for i in range(len(segs)):
    p, r = A[i], Bp[i] - A[i]
    q, s_ = A, Bp - A
    den = r[0] * s_[:, 1] - r[1] * s_[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((q[:, 0] - p[0]) * s_[:, 1] - (q[:, 1] - p[1]) * s_[:, 0]) / den
        u = ((q[:, 0] - p[0]) * r[1] - (q[:, 1] - p[1]) * r[0]) / den
    m = (np.abs(den) > 1e-14) & (t > 1e-6) & (t < 1 - 1e-6) & (u > 1e-6) & (u < 1 - 1e-6)
    for j in np.flatnonzero(m):
        if j > i and not (set(kk[i]) & set(kk[j])):
            hits.append((kk[i], kk[j], p + t[j] * r))
for k1, k2, x in hits:
    ax[1].plot(*x, "kx", ms=10, mew=2)
print("intersecting edge pairs:", sorted({(k1, k2) for k1, k2, _ in hits}))
plt.tight_layout(); plt.savefig(out, dpi=110)
print("wrote", out)
