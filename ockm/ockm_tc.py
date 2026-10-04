"""OCKM-S (shell-and-sector tree) and OCKM-TC (OCKM-T or OCKM-S, whichever seeds cheaper).

Two partitions of the data are built:
  * the orbit-cut tree of OCKM-T (straight cuts: rotated principal cuts + PCA-Part, 2-means refinement);
  * the shell-and-sector tree of OCKM-S (each split is either a shell cut, i.e. a sphere about
    the cell centroid, or a sector cut, i.e. a threshold on the first principal coordinate,
    whichever gives the larger SSE reduction; no refinement, so shell cuts stay spheres).
The same summary search (tree start + D^2 starts, LocalSearch++, weighted Lloyd) and the same lift
are run on each summary, and the centres with the lower cost on X are returned.

Guarantee: each branch satisfies E[cost] <= (1+2a) Phi_b + 2a OPT_k (in the working space);
for d <= p both trees are deterministic, so the returned centres satisfy
E[cost] <= (1+2a) min(Phi_1, Phi_2) + 2a OPT_k.
"""
from __future__ import annotations
import time
import numpy as np
import numba as nb
from scipy import sparse
import ockmt
from ockmt import ockm_t, project, summary_search, weighted_lloyd, _n_starts, _one_blas_thread, P_DEFAULT


# ------------------------------------------------------------------ concentric-sector tree
@nb.njit(cache=True)
def _best_split(keys, B, sq, ncand):
    """Best threshold split of the rows of B (centred) in the order of keys.
    Gain of a split = n ||S_L||^2 / (n_L n_R), S_L the sum of the left centred rows.
    Returns (gain, number of rows on the left, order)."""
    nc, p = B.shape
    order = np.argsort(keys, kind="mergesort")
    run = np.zeros(p)
    best_g = -1.0; best_c = 0
    nxt = 0
    # candidate positions: ncand evenly spaced split positions in 1..nc-1
    cands = np.empty(ncand, np.int64)
    for t in range(ncand):
        cands[t] = 1 + (t * (nc - 2)) // max(1, ncand - 1) if ncand > 1 else nc // 2
    c_idx = 0
    for i in range(nc - 1):
        r = order[i]
        for j in range(p):
            run[j] += B[r, j]
        pos = i + 1
        while c_idx < ncand and cands[c_idx] < pos:
            c_idx += 1
        if c_idx < ncand and cands[c_idx] == pos:
            # skip ties in the key: the cut must separate different key values
            if keys[order[i]] == keys[order[i + 1]]:
                continue
            s2 = 0.0
            for j in range(p):
                s2 += run[j] * run[j]
            g = nc * s2 / (pos * (nc - pos))
            if g > best_g:
                best_g = g; best_c = pos
    return best_g, best_c, order


@nb.njit(cache=True)
def _lead_dir(B):
    """First principal direction of the centred rows B (power iteration, 60 steps)."""
    nc, p = B.shape
    if p == 1:
        return np.ones(1)
    C = B.T @ B
    v = np.ones(p) / np.sqrt(p)
    for j in range(p):                      # deterministic start: largest-variance axis
        if C[j, j] > C[0, 0]:
            pass
    jmax = 0
    for j in range(p):
        if C[j, j] > C[jmax, jmax]:
            jmax = j
    v[:] = 0.0; v[jmax] = 1.0
    for _ in range(60):
        w = C @ v
        nrm = np.sqrt((w * w).sum())
        if nrm == 0.0:
            break
        v = w / nrm
    return v


@nb.njit(cache=True)
def _cs_tree(Z, m, k_snap, ncand, use_shell, use_sector):
    n, p = Z.shape
    Zs = Z.copy()
    start = np.zeros(m, np.int64); end = np.zeros(m, np.int64)
    csum = np.zeros((m, p)); cnt = np.zeros(m, np.int64); csse = np.zeros(m)
    labels = np.zeros(n, np.int64)
    perm = np.arange(n)
    end[0] = n; cnt[0] = n
    for i in range(n):
        for j in range(p):
            csum[0, j] += Zs[i, j]
    mu0 = csum[0] / n
    s0 = 0.0
    for i in range(n):
        for j in range(p):
            t = Zs[i, j] - mu0[j]; s0 += t * t
    csse[0] = s0
    L = 1
    snapC = np.zeros((k_snap, p))
    if k_snap == 1:
        snapC[0] = mu0
    used = np.zeros(2, np.int64)                      # [shell, sector]
    frozen = np.zeros(m, np.bool_)
    tol = 1e-12 * max(s0, 1e-300)
    while L < m:
        # cell with the largest SSE (linear scan)
        jb = -1; kb = -1.0
        for c in range(L):
            if (not frozen[c]) and csse[c] > kb:
                kb = csse[c]; jb = c
        if jb < 0 or kb <= tol:
            break
        s = start[jb]; e = end[jb]; nc = e - s
        if nc < 2:
            frozen[jb] = True; continue
        mu = csum[jb] / nc
        B = np.empty((nc, p)); sq = np.empty(nc)
        for i in range(nc):
            a = 0.0
            for j in range(p):
                t = Zs[s + i, j] - mu[j]; B[i, j] = t; a += t * t
            sq[i] = a
        bg = -1.0; bc = 0; border = np.arange(nc); kind = -1
        if use_shell:
            g, c, o = _best_split(sq, B, sq, ncand)
            if g > bg:
                bg = g; bc = c; border = o; kind = 0
        if use_sector:
            u = _lead_dir(B)
            g, c, o = _best_split(B @ u, B, sq, ncand)
            if g > bg:
                bg = g; bc = c; border = o; kind = 1
        if bg <= 0.0 or bc <= 0 or bc >= nc:
            frozen[jb] = True; continue
        used[kind] += 1
        # reorder the cell: first bc rows (in key order) form the left child
        tmpZ = Zs[s:e].copy(); tmpP = perm[s:e].copy()
        for i in range(nc):
            Zs[s + i] = tmpZ[border[i]]; perm[s + i] = tmpP[border[i]]
        new = L
        start[new] = s + bc; end[new] = e; end[jb] = s + bc
        for c2 in (jb, new):
            a = start[c2]; b = end[c2]
            sm = np.zeros(p)
            for i in range(a, b):
                for j in range(p):
                    sm[j] += Zs[i, j]
            ncc = b - a
            mc = sm / ncc
            ss = 0.0
            for i in range(a, b):
                for j in range(p):
                    t = Zs[i, j] - mc[j]; ss += t * t
            csum[c2] = sm; cnt[c2] = ncc; csse[c2] = ss
        L += 1
        if L == k_snap:
            for c2 in range(L):
                snapC[c2] = csum[c2] / cnt[c2]
    for c in range(L):
        for i in range(start[c], end[c]):
            labels[perm[i]] = c
    return labels, csum[:L].copy(), cnt[:L].copy(), csse[:L].copy(), snapC, L, used


def _branch(X, k, labels, csum, cnt, csse, snapC, L, G, mu, seed, ls_steps, n_starts, max_starts, max_iter, polish_budget):
    """Summary search + lift for one tree (same code path as OCKM-T)."""
    n, d = X.shape
    w = cnt.astype(np.float64)
    R = csum / w[:, None]
    kk = min(k, L)
    S = _n_starts(n, kk, L, n_starts, max_starts)
    steps = int(round(ls_steps * kk)) if kk > 1 else 0
    Cs, lab_leaf, scost, costs = summary_search(R, w, snapC[:kk], S, steps, max_iter, seed, True)
    if G is None:
        Sfull = csum + w[:, None] * mu[None, :]
    else:
        Sfull = np.asarray(sparse.csr_matrix((np.ones(n), (labels, np.arange(n))), shape=(L, n)) @ X)
    num = np.asarray(sparse.csr_matrix((np.ones(L), (lab_leaf, np.arange(L))), shape=(kk, L)) @ Sfull)
    den = np.bincount(lab_leaf, weights=w, minlength=kk)
    C = np.empty((k, d)); okc = den > 0
    C[:kk][okc] = num[okc] / den[okc, None]
    if (~okc).any() or kk < k:
        order = np.argsort(-w); fill = [i for i in range(k) if i >= kk or not okc[i]]
        C[fill] = (Sfull / w[:, None])[order[:len(fill)]]
    if polish_budget > 0 and G is not None:
        it = int(max(1, min(50, polish_budget // max(1, L * k * d))))
        C, _, _ = weighted_lloyd(Sfull / w[:, None], w, C, max_iter=it)
    return C, float(csse.sum()), float(scost)


def data_cost(X, C, chunk=8192):
    cc = (C * C).sum(1); tot = 0.0
    for i in range(0, len(X), chunk):
        A = X[i:i + chunk]
        D = (A * A).sum(1)[:, None] + cc[None, :] - 2.0 * A @ C.T
        tot += float(np.maximum(D.min(1), 0).sum())
    return tot


def circle_tree_seeding(X, k, gamma=8, p=P_DEFAULT, ncand=32, use_shell=True, use_sector=True, ls_steps=2.0,
                        n_starts="auto", max_starts=16, max_iter=100, polish_budget=5e9, seed=0, return_info=False):
    """Seeding with the concentric-sector (circle) tree and the OCKM-T summary search."""
    X = np.asarray(X, dtype=np.float64); n, d = X.shape; k = int(min(k, n))
    t0 = time.perf_counter()
    Z, G, mu = project(X, p, seed)
    m = int(min(n, max(k, round(gamma * k))))
    with _one_blas_thread():
        labels, csum, cnt, csse, snapC, L, used = _cs_tree(Z, m, k, ncand, use_shell, use_sector)
    t_tree = time.perf_counter() - t0
    C, phi, scost = _branch(X, k, labels, csum, cnt, csse, snapC, L, G, mu, seed, ls_steps, n_starts, max_starts, max_iter, polish_budget)
    if return_info:
        return C, dict(phi=phi, summary_cost=scost, m=L, shell_cuts=int(used[0]), sector_cuts=int(used[1]), t_tree=t_tree)
    return C


def ockm_tc(X, k, seed=0, return_info=False, **kw):
    """OCKM-TC: run OCKM-T and the circle-tree seeding, return the centres with the lower cost on X."""
    X = np.asarray(X, dtype=np.float64)
    t0 = time.perf_counter()
    C1, i1 = ockm_t(X, k, seed=seed, return_info=True, **kw)
    t1 = time.perf_counter() - t0; t0 = time.perf_counter()
    C2, i2 = circle_tree_seeding(X, k, seed=seed, return_info=True)
    t2 = time.perf_counter() - t0
    c1, c2 = data_cost(X, C1), data_cost(X, C2)
    win = "circle" if c2 < c1 else "line"
    C = C2 if c2 < c1 else C1
    if return_info:
        return C, dict(winner=win, cost_line=c1, cost_circle=c2, phi_line=i1["phi"], phi_circle=i2["phi"],
                       t_line=t1, t_circle=t2, shell_cuts=i2["shell_cuts"], sector_cuts=i2["sector_cuts"])
    return C


def warmup():
    ockmt.warmup()
    rng = np.random.default_rng(0)
    for X, k in ((rng.standard_normal((300, 2)), 4), (rng.standard_normal((300, 5)), 4), (rng.standard_normal((300, 80)), 4),
                 (rng.standard_normal((3000, 20)), 60)):
        ockm_tc(X, k)
