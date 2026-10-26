"""Unrolled plot (x = theta * r_wall, y = z) of all block edges lying on a wall of revolution (hub r=0.5 / shroud
r=1.9 for tistos), coloured by part (0 core = blue, others = red), with crossing detection between edges that do
not share a corner.   python plot_wall.py <blocking out dir> <r_wall> <png>"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d, rw, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
z = np.load(f"{d}/blocks_full.npz"); V, B, part = z["V"], z["B"], z["part"]
keys = [tuple(int(x) for x in k) for k in z["curve_keys"]]; off = z["curve_off"]; pts = z["curve_pts"]
cur = {k: pts[off[i]:off[i + 1]] for i, k in enumerate(keys)}
HE = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
epart = {}
for bi, b in enumerate(B):
    for i, j in HE:
        k = (min(int(b[i]), int(b[j])), max(int(b[i]), int(b[j])))
        epart[k] = max(epart.get(k, 0), int(part[bi]))
on = {k: Q for k, Q in cur.items() if k in epart and np.all(np.abs(np.hypot(Q[:, 0], Q[:, 1]) - rw) < 3e-3)}
uv = lambda Q: np.c_[np.unwrap(np.arctan2(Q[:, 1], Q[:, 0])) * rw, Q[:, 2]]
fig, ax = plt.subplots(figsize=(9, 12))
for k, Q in on.items():
    X = uv(Q); ax.plot(X[:, 0], X[:, 1], "b-" if epart[k] == 0 else "r-", lw=1.0)
segs = [(k, X[i], X[i + 1]) for k, Q in on.items() for X in [uv(Q)] for i in range(len(X) - 1)]
A = np.array([s[1] for s in segs]); Bp = np.array([s[2] for s in segs]); kk = [s[0] for s in segs]
hits = set()
for i in range(len(segs)):
    p, r = A[i], Bp[i] - A[i]; s_ = Bp - A
    den = r[0] * s_[:, 1] - r[1] * s_[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((A[:, 0] - p[0]) * s_[:, 1] - (A[:, 1] - p[1]) * s_[:, 0]) / den
        u = ((A[:, 0] - p[0]) * r[1] - (A[:, 1] - p[1]) * r[0]) / den
    m = (np.abs(den) > 1e-14) & (t > 1e-6) & (t < 1 - 1e-6) & (u > 1e-6) & (u < 1 - 1e-6)
    for j in np.flatnonzero(m):
        if j > i and not (set(kk[i]) & set(kk[j])):
            hits.add((kk[i], kk[j])); ax.plot(*(p + t[j] * r), "kx", ms=10, mew=2)
ax.set_aspect("equal"); ax.set_title(f"block edges on r = {rw} (blue core, red O-ring), crossings x: {len(hits)}")
plt.tight_layout(); plt.savefig(out, dpi=110)
print("edges on wall", len(on), "| crossing pairs", sorted(hits), "| wrote", out)
