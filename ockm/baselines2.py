"""Randomised seeding baselines reimplemented from their papers.

afkmc2        Bachem, Lucic, Hassani and Krause (NeurIPS 2016), "Fast and provably
              good seedings for k-means": proposal q(x) = 1/2 d(x,c1)^2 / sum + 1/(2n),
              Metropolis-Hastings chain of length m per centre (m = 200 as in the paper).
kmeans_par    Bahmani et al. (VLDB 2012), "Scalable k-means++": r = 5 rounds of
              oversampling with l = 2k, candidates weighted by cluster size, reduced
              to k by weighted k-means++ and a few weighted Lloyd steps.
"""
import numpy as np


def _mind2(A, C):
    d = np.einsum("ij,ij->i", A, A)[:, None] + np.einsum("ij,ij->i", C, C)[None, :] - 2.0 * A @ C.T
    return np.maximum(d.min(1), 0.0)


def _mind2_chunked(A, C, chunk=4096):
    out = np.empty(len(A))
    if len(C) == 0:
        out[:] = np.inf
        return out
    sqC = np.einsum("ij,ij->i", C, C)
    for i in range(0, len(A), chunk):
        B = A[i:i + chunk]
        out[i:i + chunk] = np.maximum((np.einsum("ij,ij->i", B, B)[:, None] + sqC[None, :] - 2.0 * B @ C.T).min(1), 0.0)
    return out


def _argmin_chunked(A, C, chunk=4096):
    out = np.empty(len(A), dtype=np.int64)
    sqC = np.einsum("ij,ij->i", C, C)
    for i in range(0, len(A), chunk):
        B = A[i:i + chunk]
        out[i:i + chunk] = (np.einsum("ij,ij->i", B, B)[:, None] + sqC[None, :] - 2.0 * B @ C.T).argmin(1)
    return out


def afkmc2(X, k, chain=200, seed=0):
    rng = np.random.default_rng(seed)
    n = len(X)
    c1 = int(rng.integers(n))
    d1 = np.maximum(np.einsum("ij,ij->i", X, X) - 2.0 * X @ X[c1] + X[c1] @ X[c1], 0.0)
    q = 0.5 * d1 / d1.sum() + 0.5 / n
    C = [X[c1]]
    for _ in range(1, k):
        cand = rng.choice(n, size=chain, p=q)
        dy = _mind2(X[cand], np.array(C))
        u = rng.random(chain)
        x, dx, qx = cand[0], dy[0], q[cand[0]]
        for j in range(1, chain):
            qy = q[cand[j]]
            if dx * qy == 0 or u[j] < (dy[j] * qx) / (dx * qy):
                x, dx, qx = cand[j], dy[j], qy
        C.append(X[x])
    return np.array(C)


def kmeans_par(X, k, seed=0, l=None, rounds=5):
    rng = np.random.default_rng(seed)
    n = len(X); l = 2 * k if l is None else l
    C = X[[int(rng.integers(n))]]
    d = _mind2_chunked(X, C)
    for r in range(rounds):
        if r > 0:
            d = np.minimum(d, _mind2_chunked(X, new))       # incremental update (same values)
        p = np.minimum(1.0, l * d / d.sum())
        new = X[rng.random(n) < p]
        C = np.vstack([C, new])
    C = np.unique(C, axis=0)
    while len(C) < k:
        d = _mind2_chunked(X, C); C = np.vstack([C, X[[int(np.argmax(d))]]])
    w = np.bincount(_argmin_chunked(X, C), minlength=len(C)).astype(float)
    keep = w > 0
    C, w = C[keep], w[keep]
    if len(C) <= k:
        return np.vstack([C, X[rng.choice(n, k - len(C), replace=False)]]) if len(C) < k else C
    idx = [int(rng.choice(len(C), p=w / w.sum()))]
    dmin = _mind2(C, C[idx])
    while len(idx) < k:
        p = w * dmin; j = int(rng.choice(len(C), p=p / p.sum())); idx.append(j)
        dmin = np.minimum(dmin, _mind2(C, C[[j]]))
    cen = C[idx].copy()
    for _ in range(10):
        a = (np.einsum("ij,ij->i", C, C)[:, None] + np.einsum("ij,ij->i", cen, cen)[None, :] - 2.0 * C @ cen.T).argmin(1)
        for j in range(k):
            msk = a == j
            if msk.any():
                cen[j] = (C[msk] * w[msk, None]).sum(0) / w[msk].sum()
    return cen
