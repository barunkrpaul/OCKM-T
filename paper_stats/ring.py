"""Ring instance: k tight clusters evenly spaced on a circle about the global centroid."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "ockm"), os.path.join(ROOT, "heldout"), HERE]
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
DEV_RUNS = os.path.join(ROOT, "results", "development", "runs.csv")
DEV_S_RUNS = os.path.join(ROOT, "results", "development", "runs_ockms.csv")
HELDOUT_RUNS = os.path.join(ROOT, "results", "heldout", "results_heldout_all.csv")

import numpy as np
from sklearn.cluster import KMeans, kmeans_plusplus
from cckm_original import cckm_original, kmeans_cost
from ockmt import ockm_t
from ockm_tc import circle_tree_seeding
rows = []
for k in (4, 8):
    for sigma in (0.05, 0.01, 0.002):
        rng = np.random.default_rng(1)
        ang = 2 * np.pi * np.arange(k) / k
        M = np.c_[np.cos(ang), np.sin(ang)] * 10
        X = np.vstack([m + sigma * rng.standard_normal((200, 2)) for m in M])
        opt = KMeans(k, init=M, n_init=1).fit(X).inertia_
        res = {}
        C = cckm_original(X, k)[0]; res["CCKM"] = [kmeans_cost(X, C)]
        seeds = [kmeans_plusplus(X, k, random_state=s)[0] for s in range(10)]
        res["KM++"] = [np.mean([kmeans_cost(X, c) for c in seeds])]
        CT, iT = ockm_t(X, k, seed=0, return_info=True); res["OCKM-T"] = [kmeans_cost(X, CT), iT["phi"]]
        CS, iS = circle_tree_seeding(X, k, seed=0, return_info=True); res["OCKM-S"] = [kmeans_cost(X, CS), iS["phi"], iS["shell_cuts"], iS["sector_cuts"]]
        fin = {nm: KMeans(k, init=c, n_init=1).fit(X).inertia_ / opt for nm, c in (("CCKM", C), ("OCKM-T", CT), ("OCKM-S", CS))}
        fin["KM++"] = np.mean([KMeans(k, init=c, n_init=1).fit(X).inertia_ for c in seeds]) / opt
        kmw = np.mean([KMeans(k, init=c, n_init=1).fit(X).inertia_ for c in seeds] ) / opt
        kmmax = max(KMeans(k, init=c, n_init=1).fit(X).inertia_ for c in seeds) / opt
        print(f"k={k} s={sigma}: " + " | ".join(f"{nm} seed/OPT={v[0]/opt:.4g} final/OPT={fin[nm]:.4g}" for nm, v in res.items()),
              f"| phi_T/OPT={res['OCKM-T'][1]/opt:.3f} phi_S/OPT={res['OCKM-S'][1]/opt:.3f} cutsS={res['OCKM-S'][2:]} KM++ worst final/OPT={kmmax:.4g}")
