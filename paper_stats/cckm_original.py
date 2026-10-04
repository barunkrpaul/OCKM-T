"""Original CCKM seeding (equal-width annuli about the global centroid, densest
coordinate-sign region per annulus), used for the ring example."""
import numpy as np


def sq_dists(X, C):
    """Squared Euclidean distances, shape (n, k)."""
    xx = np.einsum("ij,ij->i", X, X)[:, None]
    cc = np.einsum("ij,ij->i", C, C)[None, :]
    D = xx + cc - 2.0 * X @ C.T
    np.maximum(D, 0, out=D)
    return D


def kmeans_cost(X, C, w=None):
    d = sq_dists(X, C).min(1)
    return float(d.sum() if w is None else (w * d).sum())


# ------------------------------------------------------------------- original CCKM seeding
def cckm_original(X, k):
    n, d = X.shape
    C0 = X.mean(0)
    Y = X - C0
    r = np.sqrt((Y ** 2).sum(1))
    rmin, rmax = r.min(), r.max()
    h = (rmax - rmin) / k
    centres, empty = [], 0
    if h == 0:
        sign = (Y > 0).astype(np.int64)
        keys = sign @ (1 << np.arange(d, dtype=np.int64)[::-1] % 63)
        uk, cnt = np.unique(keys, return_counts=True)
        order = np.argsort(-cnt, kind="stable")
        for i in range(k):
            centres.append(X[keys == uk[order[i]]].mean(0) if i < len(order) else C0)
        return np.array(centres), dict(empty_annuli=0)
    idx = np.clip(np.ceil((r - rmin) / h).astype(int), 0, k)   # 0 = on inner circle
    for i in range(1, k + 1):
        A = Y[idx == i]
        if len(A) == 0:
            centres.append(C0.copy()); empty += 1; continue
        sign = (A > 0).astype(np.int64)
        # lexicographic order: non-positive (0) before positive (1), first coordinate primary
        keys = [tuple(s) for s in sign]
        uniq = {}
        for t, key in enumerate(keys):
            uniq.setdefault(key, []).append(t)
        best = max(sorted(uniq.keys()), key=lambda kk: len(uniq[kk]))  # max() keeps first on ties
        centres.append(X[idx == i][uniq[best]].mean(0))
    return np.array(centres), dict(empty_annuli=empty)


