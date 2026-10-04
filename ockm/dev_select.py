import numpy as np, pandas as pd, warnings, sys
warnings.filterwarnings("ignore")
import ockmt
from bench_data import load_any, bench_names
from sklearn.cluster import KMeans
from scipy import sparse
from scipy.stats import spearmanr
ockmt.warmup()
S = 16
rows = []
names = bench_names() + ["digits"]
for nm in names:
    X0, y, k = load_any(nm); k = int(min(k, len(X0) - 1))
    for r in range(3):
        idx = np.sort(np.random.default_rng(100 + r).choice(len(X0), int(0.9 * len(X0)), replace=False)); X = X0[idx]
        n, d = X.shape
        Z, G, mu = ockmt.project(X)
        m = int(min(n, max(k, round(8 * k))))
        lab, csum, cnt, csse, snapC, L, used = ockmt._orbit_tree(Z, m, k, 8, 64, 2, 2, 4096, False, 4, 6, 0)
        w = cnt.astype(float); R = np.ascontiguousarray(csum / w[:, None]); kk = min(k, L)
        Sfull = np.asarray(sparse.csr_matrix((np.ones(n), (lab, np.arange(n))), shape=(L, n)) @ X)
        res = []
        for si in range(S):
            C0 = snapC[:kk].copy() if si == 0 else ockmt.d2_seeding(R, w, kk, 1 + 7919 * si)
            C0, _, _ = ockmt.local_search_pp(R, w, C0, 2 * kk, seed=2 + 7919 * si)
            Cs, ll, sc = ockmt.weighted_lloyd(R, w, C0)
            num = np.asarray(sparse.csr_matrix((np.ones(L), (ll, np.arange(L))), shape=(kk, L)) @ Sfull)
            den = np.bincount(ll, weights=w, minlength=kk); ok = den > 0
            C = np.array([num[i] / den[i] if ok[i] else Sfull[np.argmax(w)] / w.max() for i in range(kk)])
            lifted = ((X[:, None, :] - C[None]) ** 2).sum(-1).min(1).sum() if n * kk * d < 5e7 else np.nan
            fin = KMeans(kk, init=C, n_init=1, max_iter=300, tol=1e-4, algorithm="lloyd", random_state=0).fit(X).inertia_
            res.append((sc, lifted, fin))
        res = np.array(res)
        best = res[:, 2].min()
        g = lambda i: 100 * (res[i, 2] / best - 1)
        rows.append(dict(dataset=nm, rep=r, A2=g(int(np.argmin(res[:2, 0]))), sum4=g(int(np.argmin(res[:4, 0]))),
                         sum8=g(int(np.argmin(res[:8, 0]))), sum16=g(int(np.argmin(res[:, 0]))),
                         lift16=g(int(np.nanargmin(res[:, 1]))) if not np.isnan(res[:, 1]).all() else np.nan,
                         top2lift=g([0, 1][int(np.argmin(res[:2, 1]))]) if not np.isnan(res[:, 1]).all() else np.nan,
                         rho=spearmanr(res[:, 0], res[:, 2]).correlation, avg=np.mean([g(i) for i in range(S)])))
    print(nm, flush=True)
df = pd.DataFrame(rows); df.to_csv("../results/development/t2/dev_select.csv", index=False)
print(df.drop(columns=["rep"]).groupby("dataset").mean().round(3).to_string())
print(df.drop(columns=["dataset", "rep"]).mean().round(3))
