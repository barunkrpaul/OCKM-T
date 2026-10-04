import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "ockm"), os.path.join(ROOT, "heldout"), HERE]
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
DEV_RUNS = os.path.join(ROOT, "results", "development", "runs.csv")
DEV_S_RUNS = os.path.join(ROOT, "results", "development", "runs_ockms.csv")
HELDOUT_RUNS = os.path.join(ROOT, "results", "heldout", "results_heldout_all.csv")
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from cckm_original import cckm_original, kmeans_cost
import ockmt, ockm_tc
from ockmt import ockm_t, project
from ockm_tc import circle_tree_seeding, _cs_tree
k, sig = 6, 0.6
rng = np.random.default_rng(3); ang = 2*np.pi*np.arange(k)/k
M = np.c_[np.cos(ang), np.sin(ang)]*10
X = np.vstack([m + sig*rng.standard_normal((200, 2)) for m in M])
opt = KMeans(k, init=M, n_init=1).fit(X).inertia_
Z, G, mu = project(X, 64, 0); m = 8*k
labT = ockmt._orbit_tree(Z, m, k, 4, 64, 2, 2, 4096, False, 4, 6, 0)[0]
labS, *_ , used = _cs_tree(Z, m, k, 32, True, True)
CT, iT = ockm_t(X, k, seed=0, return_info=True); CS, iS = circle_tree_seeding(X, k, seed=0, return_info=True)
CC = cckm_original(X, k)[0]
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.6))
cmap = plt.get_cmap("tab20")
r = np.linalg.norm(X - X.mean(0), axis=1); rmin, rmax = r.min(), r.max()
ax[0].scatter(X[:, 0], X[:, 1], s=3, c="0.6")
for i in range(k + 1):
    ax[0].add_patch(plt.Circle(X.mean(0), rmin + i*(rmax - rmin)/k, fill=False, lw=0.6, color="C0"))
ax[0].scatter(CC[:, 0], CC[:, 1], marker="x", s=60, c="k", lw=2)
ax[0].set_title(f"(a) CCKM: annuli, seed cost {kmeans_cost(X, CC)/opt:.0f}$\\times$OPT", fontsize=9)
for a, lab, C, name, phi in ((ax[1], labT, CT, "OCKM-T", iT["phi"]), (ax[2], labS, CS, "OCKM-S", iS["phi"])):
    a.scatter(X[:, 0], X[:, 1], s=3, c=[cmap(l % 20) for l in lab])
    a.scatter(C[:, 0], C[:, 1], marker="x", s=60, c="k", lw=2)
    a.set_title(f"({'b' if name=='OCKM-T' else 'c'}) {name}: {m} cells, seed cost {kmeans_cost(X, C)/opt:.2f}$\\times$OPT", fontsize=9)
for a in ax: a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([]); a.set_xlim(-13, 13); a.set_ylim(-13, 13)
plt.tight_layout(pad=1.5); plt.savefig(os.path.join(OUT, "fig_trees.pdf"))
print(kmeans_cost(X, CC)/opt, kmeans_cost(X, CT)/opt, kmeans_cost(X, CS)/opt, iT["phi"]/opt, iS["phi"]/opt, used)
