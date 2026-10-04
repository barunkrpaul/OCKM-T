"""Final benchmark (OCKM-T, compiled implementation ockmt.py) on 42 datasets (39 clustering-benchmark sets, digits, MNIST,
Fashion-MNIST) at k = number of classes (given k for the 4 unlabelled sets).

Each unit: seeding (timed), cost of the seeds, one Lloyd run to convergence
(scikit-learn, n_init=1, max_iter=300, tol=1e-4, algorithm='lloyd'; timed),
final WCSS, iterations, ARI and NMI. Randomised methods: seeds 0..9.
Published OCKM keeps its min3 rule (three Lloyd runs, best kept).
Rows are appended to the CSV at once, and finished units are skipped on restart.
"""
from __future__ import annotations
import os, sys, time, json, argparse, warnings
import numpy as np, pandas as pd
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from bench_data import load_any, bench_names
from qkmeans import qkmeans
from baselines2 import afkmc2, kmeans_par
from det_baselines import pca_part, var_part
from ockm_fast import ockm_fast
import ockmt
from ockmt import ockm_t
import sacckm, sac_fast
sac_fast.install()
warnings.filterwarnings("ignore")

RANDOM_BASE = {"KM++", "KM++v", "Qk", "AFK-MC2", "k-means||"}
OCKMT_VARIANTS = {"OCKM-T": {}, "T-pca": dict(pca_only=True), "T-noLS": dict(ls_steps=0.0), "T-S2": dict(n_starts=2),
                  "T-tree": dict(n_starts=1, ls_steps=0.0, max_iter=0), "T-D2": dict(use_tree=False),
                  "T-g4": dict(gamma=4), "T-g16": dict(gamma=16), "T-noPolish": dict(polish_budget=0)}
RANDOM = RANDOM_BASE | set(OCKMT_VARIANTS)     # OCKM-T is randomised (projection, D2 starts): 10 seeds as well
ALL = ["KM++", "KM++v", "Qk", "AFK-MC2", "k-means||", "PCA-Part", "Var-Part", "CCKM", "OCKM", "OCKM-fast"] + list(OCKMT_VARIANTS)


REPEAT_BELOW = 1.0     # seconds: faster seedings and Lloyd runs are timed three times (median)


def seed_fn(meth, X, k, s):
    if meth == "KM++":
        return kmeans_plusplus(X, k, random_state=s)[0], {}
    if meth == "KM++v":
        return kmeans_plusplus(X, k, random_state=s, n_local_trials=1)[0], {}
    if meth == "Qk":
        c, info = qkmeans(X, k, m=10.0, seed=s, return_info=True)
        return c, dict(fallbacks=info["fallbacks"])
    if meth == "AFK-MC2":
        return afkmc2(X, k, seed=s), {}
    if meth == "k-means||":
        return kmeans_par(X, k, seed=s), {}
    if meth == "PCA-Part":
        return pca_part(X, k), {}
    if meth == "Var-Part":
        return var_part(X, k), {}
    if meth == "CCKM":
        return sacckm.cckm_init(X, k)["centres"], {}
    if meth == "OCKM-fast":
        return ockm_fast(X, k), {}
    if meth in OCKMT_VARIANTS:
        c, info = ockm_t(X, k, return_info=True, seed=s, **OCKMT_VARIANTS[meth])
        return c, {kk: info[kk] for kk in ("winner", "m", "phi", "phi_full", "summary_cost", "polish_it", "cuts", "n_starts", "times")}
    raise ValueError(meth)


def lloyd(X, k, C):
    return KMeans(k, init=C, n_init=1, max_iter=300, tol=1e-4, algorithm="lloyd", random_state=0).fit(X)


def done(path):
    if not os.path.exists(path):
        return set()
    d = pd.read_csv(path, usecols=["dataset", "method", "seed"])
    return set(zip(d.dataset, d.method, d.seed))


def run(name, methods, out, runs=10):
    X, y, k = load_any(name)
    k = int(min(k, len(X) - 1))
    fin = done(out)
    for meth in methods:
        for s in (range(runs) if meth in RANDOM else [0]):
            if (name, meth, s) in fin:
                continue
            if meth == "OCKM":                                   # published OCKM: min3 rule
                def run_ockm():
                    t0 = time.perf_counter(); plane = sacckm.pc_plane(X); ts = time.perf_counter() - t0
                    tl = 0.0; best = None
                    for q in sacckm.REGIMES:
                        t0 = time.perf_counter()
                        r = sacckm.sac_init_regime(X, k, q, complement=True, plane=plane, compute_sse=False)
                        ts += time.perf_counter() - t0
                        t0 = time.perf_counter(); km = lloyd(X, k, r.centres); tl += time.perf_counter() - t0
                        if best is None or km.inertia_ < best[1].inertia_:
                            best = (r.centres, km)
                    return best, ts, tl
                (C, km), ts, tl = run_ockm()
                if ts + tl < REPEAT_BELOW:                       # median of three timings for fast units
                    reps = [(ts, tl)] + [run_ockm()[1:] for _ in range(2)]
                    ts = float(np.median([r[0] for r in reps])); tl = float(np.median([r[1] for r in reps]))
                ex = {}
            else:
                t0 = time.perf_counter(); C, ex = seed_fn(meth, X, k, s); ts = time.perf_counter() - t0
                if ts < REPEAT_BELOW:
                    tss = [ts]
                    for _ in range(2):
                        t0 = time.perf_counter(); C2, _ = seed_fn(meth, X, k, s); tss.append(time.perf_counter() - t0)
                        assert np.allclose(C2, C), (name, meth, s, "seeding not deterministic")
                    ts = float(np.median(tss))
                t0 = time.perf_counter(); km = lloyd(X, k, C); tl = time.perf_counter() - t0
                if tl < REPEAT_BELOW:
                    tls = [tl]
                    for _ in range(2):
                        t0 = time.perf_counter(); km2 = lloyd(X, k, C); tls.append(time.perf_counter() - t0)
                        assert np.isclose(km2.inertia_, km.inertia_, rtol=1e-9)
                    tl = float(np.median(tls))
            row = dict(dataset=name, n=len(X), d=X.shape[1], k=k, method=meth, seed=s, seed_time=ts,
                       seed_cost=sacckm._sse(X, C), WCSS=km.inertia_, IT=km.n_iter_, lloyd_time=tl, total_time=ts + tl,
                       ARI=adjusted_rand_score(y, km.labels_) if y is not None else np.nan,
                       NMI=normalized_mutual_info_score(y, km.labels_) if y is not None else np.nan,
                       extra=json.dumps(ex))
            pd.DataFrame([row]).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
        print(f"{name:14s} {meth:10s} done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="bench")
    ap.add_argument("--methods", default=",".join(ALL))
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    names = bench_names() + ["digits"] if a.datasets == "bench" else a.datasets.split(",")
    ockmt.warmup()                                     # JIT compilation is not timed (kernels are cached on disk)
    for nm in names:
        run(nm, a.methods.split(","), a.out, a.runs)
