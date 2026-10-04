"""Dev benchmark on 5 random 90% subsamples of each of the 40 small sets (reduces basin noise;
not the test protocol). Gap is measured to the best WCSS found on that subsample by any variant."""
import numpy as np, pandas as pd, time, sys, json, warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans, kmeans_plusplus
import ockmt, ockm_tree
from bench_data import load_any, bench_names
ockmt.warmup()
V = json.loads(sys.argv[1]); out = sys.argv[2]
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 5
rows = []
for nm in bench_names() + ["digits"]:
    X0, y, k = load_any(nm); k = int(min(k, len(X0) - 1))
    for r in range(reps):
        idx = np.sort(np.random.default_rng(100 + r).choice(len(X0), int(0.9 * len(X0)), replace=False))
        X = X0[idx]
        for v, kw in V.items():
            t = time.perf_counter()
            if v == "old":
                C = ockm_tree.ockm_t(X, k)
            elif v == "KM++":
                C = kmeans_plusplus(X, k, random_state=r)[0]
            else:
                C = ockmt.ockm_t(X, k, **kw)
            ts = time.perf_counter() - t
            t = time.perf_counter()
            km = KMeans(k, init=C, n_init=1, max_iter=300, tol=1e-4, algorithm="lloyd", random_state=0).fit(X)
            rows.append(dict(dataset=nm, rep=r, variant=v, seed_time=ts, WCSS=km.inertia_, lloyd_time=time.perf_counter() - t))
    print(nm, flush=True)
pd.DataFrame(rows).to_csv(out, index=False)
