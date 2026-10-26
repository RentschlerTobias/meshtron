"""Block edges on the blade in its own tube parametrisation (tubeparam.TubeParam): x = s (normalised arc length
around the profile), y = u (0 root profile, 1 tip profile).   python plot_blade.py <out dir> <msh> <png>"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, "/work/meshtron_v2/blade"); sys.path.insert(0, "/opt/stack/meshtron")
import msh_walls as MW
import tubeparam as TPm
from meshtron.geometry.polytron_tfi import SurfaceProjector

d, msh, out = sys.argv[1:4]
P, T, L, info, rt = MW.load_regions(msh)
bl = [i for i, x in enumerate(info) if x["kind"] == "phys" and x["phys"] == 6][0]
tp = TPm.TubeParam(P, T[L == bl]); sp = SurfaceProjector(P, T[L == bl], L[L == bl])
z = np.load(f"{d}/blocks_full.npz")
keys = [tuple(int(x) for x in k) for k in z["curve_keys"]]; off = z["curve_off"]; pts = z["curve_pts"]
fig, ax = plt.subplots(figsize=(14, 7)); n = 0
for i, k in enumerate(keys):
    Q = pts[off[i]:off[i + 1]]
    if len(Q) > 2 and sp.patches[bl].project(Q)[0].max() < 1e-3:
        us = tp.to_param(Q); s = np.unwrap(us[:, 1] * 2 * np.pi) / (2 * np.pi); n += 1
        for sh in (-1, 0, 1):
            ax.plot(s + sh, us[:, 0], "-", lw=1.3)
        m = len(Q) // 2; ax.text(s[m] % 1, us[m, 0], str(k), fontsize=8)
ax.set_xlim(0, 1); ax.set_ylim(-0.03, 1.03); ax.set_xlabel("s (around the profile)"); ax.set_ylabel("u (root 0 -> tip 1)")
ax.set_title(f"block edges on the blade, {n} edges"); plt.tight_layout(); plt.savefig(out, dpi=110)
print("edges on blade", n, "wrote", out)
