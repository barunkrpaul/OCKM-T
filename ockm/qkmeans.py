"""Qkmeans seeding (Shah, Agrawal & Jaiswal, "Fast k-means Seeding Under The
Manifold Hypothesis", arXiv:2602.01104), reimplemented from Algorithms 1-3 of
the paper. No official code was found, so this is our own implementation.

Algorithm 1 PREPROCESS
    JL-type random projection to D' = ceil(ln(k/eps)/eps^2) dimensions
    (eps = eps_jl in (0, 1/4); the paper gives D' only up to a constant,
    we use constant 1 and skip the projection when D' >= D), centre the data,
    store ||x||^2 and ||X||_F^2.
Algorithm 2 SAMPLE(L, C, m)
    s ~ Uniform(X)                                   (fallback)
    repeat at most ceil(m ln k) times:
        x ~ kappa(.|C) = (||x||^2 + ||c1||^2) / (||X||_F^2 + n ||c1||^2)
        R ~ U(0,1)
        r = cost(x, QUERY(L, x)) / (2 rho^-1 (||x||^2 + ||c1||^2))
        if R <= r: return x
    return s
Algorithm 3 QKMEANS
    c1 ~ Uniform(X); then k-1 calls of SAMPLE.

QUERY(L, x) is the approximate nearest-centre search. The paper uses an LSH
structure that returns a centre within 1/sqrt(rho) of the true nearest
distance. Here QUERY is exact (rho = 1). With rho = 1 every accepted point is
drawn exactly from the D^2 (k-means++) distribution in the projected space, so
this variant gives Qkmeans its best possible seeding quality; its runtime is
that of our NumPy code, not of the authors' LSH implementation.

Proposals are drawn in blocks. The centre set does not change until a proposal
is accepted, so taking the first accepted proposal of a block gives exactly the
same distribution as drawing proposals one at a time.
"""
from __future__ import annotations
import math
import numpy as np


def qkmeans(X: np.ndarray, k: int, m: float = 10.0, rho: float = 1.0,
            eps_jl: float = 0.2, project: bool = True, seed=None,
            return_info: bool = False):
    rng = np.random.default_rng(seed)
    n, D = X.shape
    # ---------------- Algorithm 1: PREPROCESS ----------------
    Dp = int(math.ceil(math.log(k / eps_jl) / eps_jl ** 2))
    if project and Dp < D:
        G = rng.standard_normal((D, Dp)) / math.sqrt(Dp)
        Y = X @ G
    else:
        Dp = D
        Y = np.array(X, dtype=float, copy=True)
    Y -= Y.mean(axis=0)
    sq = np.einsum("ij,ij->i", Y, Y)
    F = float(sq.sum())
    cdf_sq = np.cumsum(sq); cdf_sq /= cdf_sq[-1]
    # ---------------- Algorithm 3: QKMEANS ----------------
    idx = [int(rng.integers(n))]
    c1sq = float(sq[idx[0]])
    p_norm = F / (F + n * c1sq)          # kappa is a mixture: ||x||^2-weighted w.p. p_norm, uniform otherwise
    Cmat = np.empty((k, Dp)); Cmat[0] = Y[idx[0]]
    Csq = np.empty(k); Csq[0] = c1sq
    T = (math.inf if math.isinf(m) else int(math.ceil(m * math.log(k)))) if k > 1 else 0
    proposals = 0; fallbacks = 0
    for t in range(1, k):
        s = int(rng.integers(n))           # Algorithm 2, step 1
        done = 0; chosen = None; B = 16
        while done < T:
            b = int(min(B, T - done, 1 << 16))
            u = rng.random(b)
            from_norm = rng.random(b) < p_norm
            x = np.where(from_norm, np.searchsorted(cdf_sq, u, side="right"), rng.integers(n, size=b))
            x = np.minimum(x, n - 1)
            R = rng.random(b)
            Yx = Y[x]
            d2 = sq[x][:, None] + Csq[None, :t] - 2.0 * Yx @ Cmat[:t].T
            cost = np.maximum(d2.min(axis=1), 0.0)
            r = cost / (2.0 / rho * (sq[x] + c1sq))
            acc = np.flatnonzero(R <= r)
            if acc.size:
                j = int(acc[0]); chosen = int(x[j]); proposals += j + 1; break
            proposals += b; done += b; B = min(B * 2, 1 << 16)
        if chosen is None:
            chosen = s; fallbacks += 1
        idx.append(chosen)
        Cmat[t] = Y[chosen]; Csq[t] = sq[chosen]
    idx = np.asarray(idx)
    if return_info:
        return X[idx].copy(), dict(idx=idx, Dp=Dp, proposals=proposals, fallbacks=fallbacks, T=T)
    return X[idx].copy()
