"""OCKM-fast: a fast, deterministic version of orbit-chart (SAC-CKM / OCKM) seeding.

It keeps the OCKM idea (orbit charts E, H, P on the principal plane, equal-count
bands of the invariant I and sectors of the orbit coordinate s, seeds chosen by
density-filtered farthest-first, DFF) and changes only how each step is computed:

1. principal plane by randomised subspace iteration (3 steps) on a fixed
   subsample of at most 10,000 points, instead of the d x d covariance of all points;
2. one pass over the data gives the chart coordinates and a fixed Gaussian
   projection of dimension p = min(d, 128); cell centroids are formed in the
   projection with one sparse product, and full-dimensional centroids are
   computed only for the k chosen cells;
3. DFF on squared distances in the projection, one matrix-vector product per
   step;
4. the chart is chosen by the seeding cost on a fixed 5,000-point subsample,
   measured in the same projection (instead of three full cost passes, or
   three Lloyd runs as in OCKM's min3 rule), so one Lloyd run follows.

Variant 'split': the complement sign split of OCKM, but on r extra principal
coordinates only, with r = floor(log2(n / (8 k t))) clipped to [0, 8] and
t = 20 the target mean occupancy. This keeps cells populated (the full split
makes one cell per point when d is large). For k >= n / (8 t) it equals the
plain version.

Everything uses a fixed random seed, so the output is deterministic.
"""
from __future__ import annotations
import math
import numpy as np
from scipy import sparse
import sacckm

P_DIM = 128
N_SUB = 5000
T_OCC = 20


def rand_pcs(X, mu, r, seed=0, oversample=8, iters=4):
    """Top-r principal directions (r x d) of X - mu by randomised subspace iteration."""
    n, d = X.shape
    l = min(d, r + oversample)
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((d, l))
    Y = X @ G - mu @ G
    for _ in range(iters):
        Q, _ = np.linalg.qr(Y)
        Z = X.T @ Q - np.outer(mu, Q.sum(0))          # (X - mu)^T Q
        Q2, _ = np.linalg.qr(Z)
        Y = X @ Q2 - mu @ Q2
    Q, _ = np.linalg.qr(Y)
    Bt = X.T @ Q - np.outer(mu, Q.sum(0))              # d x l = (Q^T (X - mu))^T
    _, _, Vt = np.linalg.svd(Bt.T, full_matrices=False)
    W = Vt[:r].copy()
    for j in range(r):                                 # sign convention of sacckm.pc_plane
        i = np.argmax(np.abs(W[j]))
        if W[j, i] < 0:
            W[j] *= -1
    return W


def _dff(Cp, counts, k, expected):
    """Density-filtered farthest-first on the rows of Cp (squared distances)."""
    m = len(Cp)
    chosen = [int(np.argmax(counts))]
    if m == 1:
        return chosen
    sq = np.einsum("ij,ij->i", Cp, Cp)
    sep = np.maximum(sq - 2.0 * Cp @ Cp[chosen[0]] + sq[chosen[0]], 0.0)
    pool = counts >= expected
    taken = np.zeros(m, bool); taken[chosen[0]] = True
    while len(chosen) < min(k, m):
        cand = pool & ~taken
        if not cand.any():
            cand = ~taken
        score = np.where(cand, sep, -np.inf)
        j = int(np.argmax(score)); chosen.append(j); taken[j] = True
        np.minimum(sep, np.maximum(sq - 2.0 * Cp @ Cp[j] + sq[j], 0.0), out=sep)
    return chosen


def _cost_proj(Sp, Cp):
    d2 = np.einsum("ij,ij->i", Sp, Sp)[:, None] + np.einsum("ij,ij->i", Cp, Cp)[None, :] - 2.0 * Sp @ Cp.T
    return float(np.maximum(d2.min(1), 0).sum())


def ockm_fast(X, k, variant="plain", Q=4, seed=0, return_info=False, n_pca=10000, pca_iters=3):
    n, d = X.shape
    mu = X.mean(0)
    B = k
    r_extra = 0
    if variant == "split":
        r_extra = int(np.clip(math.floor(math.log2(max(n / (8.0 * k * T_OCC), 1.0))), 0, 8))
    rng = np.random.default_rng(seed + 1)
    # principal directions from a fixed subsample (PCA only needs the plane, not every point)
    ps = rng.choice(n, n_pca, replace=False) if n > n_pca else np.arange(n)
    Xs = X[ps]
    W = rand_pcs(Xs, Xs.mean(0), 2 + r_extra, seed=seed, iters=pca_iters)
    p = min(d, P_DIM)
    G = rng.standard_normal((d, p)) / math.sqrt(p) if p < d else np.eye(d)
    # one pass over X: chart coordinates and the random projection together
    A = np.hstack([W.T, G])
    XA = X @ A - mu @ A
    UV, Xp = XA[:, :2 + r_extra], XA[:, 2 + r_extra:]
    u, v = UV[:, 0], UV[:, 1]
    sub = rng.choice(n, min(n, N_SUB), replace=False)
    Sp = Xp[sub]
    extra_key = None
    if r_extra:
        bits = (UV[:, 2:] > 0).astype(np.int64)
        extra_key = (bits * (1 << np.arange(r_extra))).sum(1)
    best = None
    for regime in sacckm.REGIMES:
        I, s, lab = sacckm.chart(u, v, regime)
        key = (sacckm.quantile_bins(I, B) * (Q * 2) + sacckm.quantile_bins(s, Q) * 2 + lab).astype(np.int64)
        if extra_key is not None:
            key = key * (1 << r_extra) + extra_key
        uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
        inv = inv.ravel(); m = len(uniq)
        M = sparse.csr_matrix((np.ones(n), (inv, np.arange(n))), shape=(m, n))
        Cp = np.asarray(M @ Xp) / counts[:, None]        # centred projected cell centroids
        expected = counts.mean() if r_extra else n / (B * Q)
        ch = _dff(Cp, counts, k, expected)
        cost = _cost_proj(Sp, Cp[ch])
        if best is None or cost < best[0]:
            best = (cost, regime, M[ch], counts[ch], m)
    cost, regime, Mch, cch, m = best
    centres = np.asarray(Mch @ X) / cch[:, None]          # full-dimensional centroids of the chosen cells only
    if len(centres) < k:
        centres = np.vstack([centres, np.repeat(mu[None], k - len(centres), 0)])
    if return_info:
        return centres, dict(regime=regime, n_cells=m, r_extra=r_extra)
    return centres
