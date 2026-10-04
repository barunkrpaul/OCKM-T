"""Seeding at large k (the regime of the Qkmeans paper) and anytime Lloyd curves.

Part 1: seeding time and seeding cost for k in {10, 100, 1000} on MNIST, Fashion-MNIST,
letter and birch-rg1. Randomised methods: 3 runs.
Part 2 (--trace): for k = 1000 on MNIST and Fashion-MNIST, the cost after 0..10 Lloyd
iterations and the cumulative time (seeding + iterations), with one common NumPy Lloyd
implementation for all methods.
"""
import os, sys, time, json, argparse, warnings
import numpy as np, pandas as pd
from sklearn.cluster import kmeans_plusplus
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from bench_data import load_any
from qkmeans import qkmeans
from det_baselines import pca_part
import ockmt
from ockmt import ockm_t
from baselines2 import afkmc2, kmeans_par
from sacckm import _sse
warnings.filterwarnings("ignore")

METHODS = {
    "OCKM-T": lambda X, k, s: ockm_t(X, k, seed=s),
    "T-pca": lambda X, k, s: ockm_t(X, k, pca_only=True, seed=s),
    "PCA-Part": lambda X, k, s: pca_part(X, k),
    "KM++": lambda X, k, s: kmeans_plusplus(X, k, random_state=s)[0],
    "KM++v": lambda X, k, s: kmeans_plusplus(X, k, random_state=s, n_local_trials=1)[0],
    "Qk": lambda X, k, s: qkmeans(X, k, seed=s),
    "AFK-MC2": lambda X, k, s: afkmc2(X, k, seed=s),
    "k-means||": lambda X, k, s: kmeans_par(X, k, seed=s),
}
RANDOM = {"KM++", "KM++v", "Qk", "AFK-MC2", "k-means||", "OCKM-T", "T-pca"}


def lloyd_trace(X, C, iters=10, block=20000):
    """Plain Lloyd with per-iteration cost and time (same code for every method)."""
    C = np.array(C, dtype=float, copy=True)
    sqX = np.einsum("ij,ij->i", X, X)
    out = []
    t0 = time.perf_counter()
    for it in range(iters + 1):
        lab = np.empty(len(X), dtype=np.int64); cost = 0.0
        sqC = np.einsum("ij,ij->i", C, C)
        for i in range(0, len(X), block):
            D = sqX[i:i + block, None] + sqC[None, :] - 2.0 * X[i:i + block] @ C.T
            lab[i:i + block] = D.argmin(1)
            cost += float(np.maximum(D[np.arange(len(D)), lab[i:i + block]], 0).sum())
        out.append((it, time.perf_counter() - t0, cost))
        if it == iters:
            break
        cnt = np.bincount(lab, minlength=len(C))
        S = np.zeros_like(C); np.add.at(S, lab, X)
        nz = cnt > 0
        C[nz] = S[nz] / cnt[nz, None]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--methods", default=None)
    a = ap.parse_args()
    ockmt.warmup()
    rows = []
    if not a.trace:
        for name in ("MNIST", "FMNIST", "letter", "birch-rg1"):
            X, y, _ = load_any(name)
            for k in (10, 100, 1000):
                for m, fn in METHODS.items():
                    if a.methods and m not in a.methods.split(","):
                        continue
                    for s in (range(3) if m in RANDOM else [0]):
                        t = time.perf_counter(); C = fn(X, k, s); ts = time.perf_counter() - t
                        rows.append(dict(dataset=name, k=k, method=m, seed=s, seed_time=ts, seed_cost=_sse(X, C)))
                        pd.DataFrame([rows[-1]]).to_csv(a.out, mode="a", header=not os.path.exists(a.out), index=False)
                    print(name, k, m, "done", flush=True)
    else:
        for name in ("MNIST", "FMNIST"):
            X, y, _ = load_any(name)
            k = 1000
            for m, fn in METHODS.items():
                if m == "KM++v" or (a.methods and m not in a.methods.split(",")):
                    continue
                t = time.perf_counter(); C = fn(X, k, 0); ts = time.perf_counter() - t
                for it, tt, cost in lloyd_trace(X, C, iters=10):
                    r = dict(dataset=name, k=k, method=m, iteration=it, time=ts + tt, seed_time=ts, cost=cost)
                    pd.DataFrame([r]).to_csv(a.out, mode="a", header=not os.path.exists(a.out), index=False)
                print(name, m, "trace done", flush=True)


if __name__ == "__main__":
    main()
