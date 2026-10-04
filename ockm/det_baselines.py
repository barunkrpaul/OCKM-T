"""Deterministic divisive seedings of Su and Dy (2007), "In search of deterministic
methods for initializing K-means and Gaussian mixture clustering", Intelligent
Data Analysis 11(4):319-338, as usually described (e.g. Celebi, Kingravi and
Vela 2013): start with one cluster; repeatedly split the cluster with the
largest SSE by a hyperplane through its centroid,
  PCA-Part: orthogonal to the cluster's principal direction,
  Var-Part: orthogonal to the coordinate axis of largest variance;
stop at k clusters and return their centroids.

Implementation: each split gathers the cluster once (O(n_c d)); SSEs come from
precomputed squared norms. The principal direction is found by randomised
subspace iteration (3 steps, fixed seed) on at most 10,000 points of the
cluster, the same routine OCKM-fast uses, so the result is deterministic.
"""
import numpy as np
from ockm_fast import rand_pcs


def divisive_part(X, k, mode="pca", n_pca=10000, seed=0):
    n = len(X)
    sq = np.einsum("ij,ij->i", X, X)
    rng = np.random.default_rng(seed)
    mu0 = X.mean(0)
    clusters, mus, sse = [np.arange(n)], [mu0], [float(sq.sum() - n * mu0 @ mu0)]
    while len(clusters) < k:
        j = int(np.argmax(sse))
        if sse[j] <= 0:
            break
        idx = clusters[j]; Xi = X[idx]; mu = mus[j]
        if mode == "pca":
            if len(idx) > n_pca:
                Xs = Xi[rng.choice(len(idx), n_pca, replace=False)]
            else:
                Xs = Xi
            v = rand_pcs(Xs, mu, 1, seed=seed, iters=3)[0]
            proj = Xi @ v - mu @ v
        else:
            var = np.einsum("ij,ij->j", Xi, Xi) / len(idx) - mu * mu
            dim = int(np.argmax(var))
            proj = Xi[:, dim] - mu[dim]
        left = proj <= 0
        nl = int(left.sum())
        if nl == 0 or nl == len(idx):
            sse[j] = -1.0            # cannot be split; never chosen again
            continue
        parts = []
        for mask in (left, ~left):
            sub = idx[mask]; m_ = Xi[mask].mean(0)
            parts.append((sub, m_, float(sq[sub].sum() - len(sub) * m_ @ m_)))
        clusters[j], mus[j], sse[j] = parts[0]
        clusters.append(parts[1][0]); mus.append(parts[1][1]); sse.append(parts[1][2])
    C = np.array(mus)
    if len(C) < k:
        C = np.vstack([C, np.repeat(mu0[None], k - len(C), 0)])
    return C


def pca_part(X, k):
    return divisive_part(X, k, "pca")


def var_part(X, k):
    return divisive_part(X, k, "var")
