"""OCKM-T: orbit-chart tree summaries for fast k-means seeding with a guarantee.

Three stages.

1. Orbit-chart tree (deterministic). Work in a fixed Gaussian projection of
   dimension p = min(d, P) of the centred data. Starting from one cell, the cell
   with the largest SSE is split, until m = gamma * k cells exist. A split of a
   cell uses the orbit charts of the cell's OWN principal plane (u, v):
     * 8 hyperplanes u cos t + v sin t = 0, t = j pi / 8 (t = 0 is PCA-Part),
     * hyperplanes u = q-quantile of u, q in {1/4, 3/8, 5/8, 3/4} (off-centre cuts),
     * elliptic cut (ring r > median r), hyperbolic cuts (u v > 0, |u| > |v|),
       parabolic cut (u > v^2 / (2 s_v)),
   and keeps the cut with the largest SSE reduction, computed exactly as
   n ||S_L||^2 / (n_L n_R) from the side sums S_L of the centred cell. The chosen
   cut can be refined by a few 2-means steps inside the cell. Every split
   therefore reduces the SSE at least as much as the PCA-Part cut of that cell.
   The partition after k - 1 splits is kept as well (the k-leaf tree seeding).

2. Summary search. The m leaves form a weighted summary (centroid, count).
   Weighted Lloyd is run on the summary from several starts: the k-leaf tree
   partition, `restarts` greedy k-means++ starts and one vanilla (D^2) k-means++
   start. The start with the lowest summary cost wins. m << n, so this is cheap.

3. Lift. Each winning summary cluster is mapped back to full dimension as the
   mean of the points in its leaves (one sparse pass over X).

Guarantee (Theorem 1 of the note). With Phi = within-leaf cost of the summary
and alpha = 8 (ln k + 2),
    E[cost(X, C)] <= Phi + alpha (2 Phi + 2 OPT_k(X)),
because the winner costs no more on the summary than the vanilla D^2 start after
Lloyd, the summary cost bounds the lifted cost (bias-variance identity), and
the k-means++ bound of Arthur and Vassilvitskii (2007) holds on the weighted
summary (a multiset). Computed in a JL projection, the bound holds up to
(1 +/- eps) factors (Makarychev, Makarychev and Razenshteyn, 2019).

All random choices use a fixed seed, so the output is reproducible.
"""
from __future__ import annotations
import heapq, math
import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans, kmeans_plusplus

P_DEFAULT = 64
OFFSETS = (0.25, 0.375, 0.625, 0.75)
CUT_NAMES = [f"plane{j}" for j in range(8)] + [f"offset{q}" for q in OFFSETS] + ["E_ring", "H_quad", "H_sect", "P_parab"]


def project(X, p=P_DEFAULT, seed=0):
    n, d = X.shape
    mu = X.mean(0)
    if d <= p:
        return X - mu, None
    G = np.random.default_rng(seed).standard_normal((d, p)) / math.sqrt(p)
    return X @ G - mu @ G, G


def _top2_dirs(Zc, Q0, iters=8):
    """Top-2 principal directions of the centred rows Zc (n x p): exact (SVD when n <= p,
    eigh when n >= 1000), otherwise block power iteration (8 steps) from a fixed start
    Q0 followed by Rayleigh-Ritz."""
    n, p = Zc.shape
    if n <= p or p <= 8:
        _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
        return Vt[:2].T
    if n >= 1000:                                   # large cells: exact (cheap next to the n p^2 product)
        _, V = np.linalg.eigh(Zc.T @ Zc)
        return V[:, [-1, -2]]
    Q = Q0
    for _ in range(iters):
        Q, _ = np.linalg.qr(Zc.T @ (Zc @ Q))
    B = Zc @ Q
    _, _, Vt = np.linalg.svd(B, full_matrices=False)
    return Q @ Vt.T


def _qsel(x, qs):
    """Quantiles by order statistics (lower index), faster than np.quantile for a few q."""
    n = len(x)
    ks = sorted({min(n - 1, int(q * (n - 1))) for q in qs})
    part = np.partition(x, ks)
    return [part[min(n - 1, int(q * (n - 1)))] for q in qs]


def _masks(u, v, family):
    cols = []
    if family == "pca":
        cols.append(u > 0)
    else:
        for j in range(8):
            t = j * math.pi / 8
            cols.append(u * math.cos(t) + v * math.sin(t) > 0)
        if family in ("offsets", "orbit"):
            qs = _qsel(u, OFFSETS)
            for q in qs:
                cols.append(u > q)
        if family == "orbit":
            r2 = u * u + v * v
            cols.append(r2 > _qsel(r2, (0.5,))[0])
            cols.append(u * v > 0)
            cols.append(np.abs(u) > np.abs(v))
            sv = math.sqrt(float(np.mean(v * v))) + 1e-12
            cols.append(u - v * v / (2.0 * sv) > 0)
    return np.stack(cols, 1)


def _names(family):
    if family == "pca":
        return ["plane0"]
    if family == "angles":
        return CUT_NAMES[:8]
    if family == "offsets":
        return CUT_NAMES[:12]
    return CUT_NAMES


def orbit_tree(Z, m, k_snap=None, family="orbit", refine=2):
    """Greedy divisive tree on the rows of Z (n x p, any origin). Returns leaf
    labels for m leaves, the labels of the k_snap-leaf partition, and cut counts."""
    n, p = Z.shape
    m = int(min(m, n))
    sq = np.einsum("ij,ij->i", Z, Z)
    labels = np.zeros(n, dtype=np.int64)
    members = [np.arange(n)]
    sums = [Z.sum(0)]
    cnt = [n]
    sse0 = float(sq.sum() - sums[0] @ sums[0] / n)
    heap = [(-sse0, 0)]
    snap = labels.copy() if k_snap == 1 else None
    names = _names(family); used = {}
    Q0 = np.linalg.qr(np.random.default_rng(12345).standard_normal((p, 2)))[0] if p >= 2 else None
    n_leaves = 1
    while n_leaves < m and heap:
        negs, j = heapq.heappop(heap)
        idx = members[j]
        nc = len(idx)
        if nc < 2 or -negs <= 1e-12 * max(sse0, 1e-300):
            continue
        mu = sums[j] / nc
        Zc = Z[idx] - mu
        if p >= 2 and nc > 2:
            W2 = _top2_dirs(Zc, Q0)
            u, v = Zc @ W2[:, 0], Zc @ W2[:, 1]
        else:
            u = Zc[:, 0].copy(); v = np.zeros(nc)
        M = _masks(u, v, family)
        SL = M.T.astype(float) @ Zc
        nL = M.sum(0).astype(float); nR = nc - nL
        ok = (nL > 0) & (nR > 0)
        if not ok.any():
            continue
        red = np.full(M.shape[1], -np.inf)
        red[ok] = nc * np.einsum("ij,ij->i", SL[ok], SL[ok]) / (nL[ok] * nR[ok])
        c = int(np.argmax(red))
        left = M[:, c].copy()
        used[names[c]] = used.get(names[c], 0) + 1
        for _ in range(refine):                      # 2-means steps inside the cell
            aL = Zc[left].mean(0); aR = Zc[~left].mean(0)
            new = ((Zc - aL) ** 2).sum(1) < ((Zc - aR) ** 2).sum(1)
            if new.all() or (~new).all() or np.array_equal(new, left):
                break
            left = new
        iL, iR = idx[left], idx[~left]
        sL, sR = Z[iL].sum(0), Z[iR].sum(0)
        new_id = len(members)
        members[j], sums[j], cnt[j] = iL, sL, len(iL)
        members.append(iR); sums.append(sR); cnt.append(len(iR))
        labels[iR] = new_id
        for jj, ii, ss in ((j, iL, sL), (new_id, iR, sR)):
            e = float(sq[ii].sum() - ss @ ss / len(ii))
            heapq.heappush(heap, (-e, jj))
        n_leaves += 1
        if k_snap is not None and n_leaves == k_snap:
            snap = labels.copy()
    if k_snap is not None and snap is None:
        snap = labels.copy()
    return labels, snap, used


def _sqd(A, B):
    return np.maximum(np.einsum("ij,ij->i", A, A)[:, None] + np.einsum("ij,ij->i", B, B)[None, :] - 2.0 * A @ B.T, 0.0)


def _top2(D):
    k = D.shape[1]
    if k == 1:
        a = np.zeros(len(D), dtype=np.int64)
        return a, a.copy(), D[:, 0].copy(), np.full(len(D), np.inf)
    part = np.argpartition(D, 1, axis=1)[:, :2]
    r = np.arange(len(D))
    v0, v1 = D[r, part[:, 0]], D[r, part[:, 1]]
    swap = v1 < v0
    a = np.where(swap, part[:, 1], part[:, 0]); b = np.where(swap, part[:, 0], part[:, 1])
    return a, b, np.minimum(v0, v1), np.maximum(v0, v1)


def local_search_pp(R, w, C, steps, seed=0):
    """LocalSearch++ (Lattanzi and Sohler, ICML 2019) on the weighted points R:
    draw a candidate by D^2 sampling, swap it with the centre whose removal gives
    the lowest cost, keep the swap only if the cost drops. The cost change of all
    k possible swaps is computed in O(m) from nearest / second-nearest bookkeeping."""
    rng = np.random.default_rng(seed)
    C = np.array(C, dtype=float, copy=True)
    m, k = len(R), len(C)
    D = _sqd(R, C)
    a, b, d1, d2 = _top2(D)
    cost = float(w @ d1)
    sqR = np.einsum("ij,ij->i", R, R)
    accepted = 0
    for _ in range(int(steps)):
        pr = w * d1
        tot = pr.sum()
        if tot <= 0:
            break
        x = int(rng.choice(m, p=pr / tot))
        dx = np.maximum(sqR - 2.0 * R @ R[x] + sqR[x], 0.0)
        base = np.minimum(dx, d1)
        keep = float(w @ base)
        delta = np.bincount(a, weights=w * (np.minimum(dx, d2) - base), minlength=k)
        new = keep + delta
        c = int(np.argmin(new))
        if new[c] >= cost * (1.0 - 1e-12):
            continue
        accepted += 1
        C[c] = R[x]; D[:, c] = dx
        redo = (a == c) | (b == c)
        oth = ~redo
        closer = oth & (dx < d1)
        mid = oth & ~closer & (dx < d2)
        b[closer] = a[closer]; d2[closer] = d1[closer]; a[closer] = c; d1[closer] = dx[closer]
        b[mid] = c; d2[mid] = dx[mid]
        if redo.any():
            ra, rb, r1, r2 = _top2(D[redo])
            a[redo], b[redo], d1[redo], d2[redo] = ra, rb, r1, r2
        cost = float(w @ d1)
    return C, cost, accepted


def _leaf_stats(labels, Z):
    uniq, inv, cnt = np.unique(labels, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    Msp = sparse.csr_matrix((np.ones(len(labels)), (inv, np.arange(len(labels)))), shape=(len(uniq), len(labels)))
    return inv, cnt, Msp, np.asarray(Msp @ Z) / cnt[:, None]


def ockm_t(X, k, gamma=8, p=P_DEFAULT, family="orbit", refine=2, restarts=0, ls_steps=2.0, max_iter=100,
           polish_budget=5e9, seed=0, return_info=False):
    """ls_steps: LocalSearch++ swaps per start on the summary, as a multiple of k (0 disables).
    polish_budget: flops allowed for the final weighted Lloyd on the summary in the ORIGINAL
    space (m * k * d per iteration), which removes the projection distortion; 0 disables."""
    n, d = X.shape
    Z, G = project(X, p, seed)
    m = int(min(n, max(k, round(gamma * k))))
    labels, snap, used = orbit_tree(Z, m, k_snap=k, family=family, refine=refine)
    inv, w, Msp, R = _leaf_stats(labels, Z)            # summary: m leaf centroids in the working space
    w = w.astype(float)
    mleaf = len(w)
    sqZ = np.einsum("ij,ij->i", Z, Z)
    phi = float(sqZ.sum() - np.einsum("ij,ij->i", R, R) @ w)   # within-leaf cost in the working space
    _, sinv, scnt = np.unique(snap, return_inverse=True, return_counts=True)
    sinv = sinv.ravel()
    Ssp = sparse.csr_matrix((np.ones(n), (sinv, np.arange(n))), shape=(len(scnt), n))
    tree_init = np.asarray(Ssp @ Z) / scnt[:, None]
    kk = min(k, mleaf)
    starts = []
    if len(tree_init) >= kk:
        starts.append(("tree", tree_init[:kk]))
    c0, _ = kmeans_plusplus(R, kk, sample_weight=w, n_local_trials=1, random_state=seed + 1)
    starts.append(("D2", c0))
    cands = []
    steps = int(round(ls_steps * kk))
    for name, C0 in starts:
        if steps > 0:
            C0, _, acc = local_search_pp(R, w, C0, steps, seed=seed + 2)
        km = KMeans(kk, init=C0, n_init=1, max_iter=max_iter, algorithm="lloyd", random_state=seed).fit(R, sample_weight=w)
        cands.append((name, km))
    if restarts > 0:
        cands.append(("greedy++", KMeans(kk, init="k-means++", n_init=restarts, max_iter=max_iter, algorithm="lloyd",
                                         random_state=seed).fit(R, sample_weight=w)))
    name, best = min(cands, key=lambda t: t[1].inertia_)
    lab_leaf = best.labels_
    Sfull = np.asarray(Msp @ X)                         # m x d leaf sums
    num = np.asarray(sparse.csr_matrix((np.ones(mleaf), (lab_leaf, np.arange(mleaf))), shape=(kk, mleaf)) @ Sfull)
    den = np.bincount(lab_leaf, weights=w, minlength=kk)
    C = np.empty((k, d))
    okc = den > 0
    C[:kk][okc] = num[okc] / den[okc, None]
    if (~okc).any() or kk < k:
        order = np.argsort(-w)
        full_leaf = Sfull / w[:, None]
        fill = [i for i in range(k) if i >= kk or not okc[i]]
        C[fill] = full_leaf[order[:len(fill)]]
    polish_it = 0
    if polish_budget > 0 and G is not None:
        polish_it = int(max(1, min(50, polish_budget // max(1, mleaf * k * d))))
        if polish_it >= 1:
            Rfull = Sfull / w[:, None]
            C = KMeans(k, init=C, n_init=1, max_iter=polish_it, algorithm="lloyd",
                       random_state=seed).fit(Rfull, sample_weight=w).cluster_centers_
    if return_info:
        return C, dict(m=mleaf, phi=phi, summary_cost=float(best.inertia_), winner=name, polish_it=polish_it,
                       cand_costs={nm: float(km.inertia_) for nm, km in cands}, cuts=used)
    return C


# ----------------------------------------------------------------------------
# Local orbit-chart summary: charts of each leaf's own principal plane
# ----------------------------------------------------------------------------
def _group_rank_bins(key_group, value, nbins, gsize, gstart):
    """Equal-count bins of `value` inside each group; nbins and sizes given per point."""
    order = np.lexsort((value, key_group))
    rank = np.empty(len(value), dtype=np.int64)
    rank[order] = np.arange(len(value))
    within = rank - gstart[key_group]
    return (within * nbins) // np.maximum(gsize[key_group], 1)


def local_chart_summary(Z, leaf, gamma, Q=4):
    """Split every leaf of the tree into about gamma * k * SSE_leaf / SSE_total cells
    (at least 1) using the best of the three orbit charts (E, H, P) of the leaf's
    own principal plane: equal-count bands of the invariant I times equal-count
    sectors of the orbit coordinate s. Returns cell labels and the chosen chart
    per leaf."""
    n, p = Z.shape
    uniq, L, size = np.unique(leaf, return_inverse=True, return_counts=True)
    L = L.ravel(); k = len(uniq)
    S = np.zeros((k, p)); np.add.at(S, L, Z)
    MU = S / size[:, None]
    sq = np.einsum("ij,ij->i", Z, Z)
    sse = np.bincount(L, weights=sq, minlength=k) - np.einsum("ij,ij->i", S, S) / size
    g = np.maximum(1, np.round(gamma * k * sse / max(sse.sum(), 1e-300))).astype(np.int64)
    g = np.minimum(g, size)
    order = np.argsort(L, kind="stable")
    start = np.zeros(k, dtype=np.int64); start[1:] = np.cumsum(size)[:-1]
    W1 = np.zeros((k, p)); W2 = np.zeros((k, p))
    for l in range(k):
        idx = order[start[l]:start[l] + size[l]]
        if size[l] > 2 and p >= 2:
            Zc = Z[idx] - MU[l]
            _, V = np.linalg.eigh(Zc.T @ Zc)
            W1[l], W2[l] = V[:, -1], V[:, -2]
        else:
            W1[l, 0] = 1.0
    Zc = Z - MU[L]
    u = np.einsum("ij,ij->i", Zc, W1[L]); v = np.einsum("ij,ij->i", Zc, W2[L])
    Qg = np.minimum(Q, g); Bg = np.maximum(1, np.round(g / Qg)).astype(np.int64)
    Qp, Bp = Qg[L], Bg[L]
    best_cost, best_lab, charts = None, None, None
    eps = 1e-12
    per_leaf = []
    labs = []
    for regime in ("E", "H", "P"):
        if regime == "E":
            I = np.sqrt(u * u + v * v); s = np.arctan2(v, u); lab = np.zeros(n, dtype=np.int64)
        elif regime == "H":
            I = u * v; s = 0.5 * np.log((np.abs(u) + eps) / (np.abs(v) + eps)); lab = (u > 0).astype(np.int64)
        else:
            I = v; s = u / np.where(np.abs(v) < eps, np.where(v < 0, -eps, eps), v); lab = np.zeros(n, dtype=np.int64)
        bI = _group_rank_bins(L, I, Bp, size, start)
        bS = _group_rank_bins(L, s, Qp, size, start)
        key = ((L * (Bg.max() + 1) + bI) * (Qg.max() + 1) + bS) * 2 + lab
        _, cell, ccnt = np.unique(key, return_inverse=True, return_counts=True)
        cell = cell.ravel()
        CS = np.zeros((len(ccnt), p)); np.add.at(CS, cell, Z)
        csse = np.bincount(cell, weights=sq, minlength=len(ccnt)) - np.einsum("ij,ij->i", CS, CS) / ccnt
        cleaf = np.zeros(len(ccnt), dtype=np.int64); cleaf[cell] = L
        per_leaf.append(np.bincount(cleaf, weights=csse, minlength=k))
        labs.append(cell)
    per_leaf = np.stack(per_leaf, 1)                    # k x 3 within-cell SSE per leaf and chart
    choice = np.argmin(per_leaf, 1)
    pick = choice[L]
    # combine: cell id = (leaf, chart cell) for the chosen chart of each leaf
    raw = np.where(pick == 0, labs[0], np.where(pick == 1, labs[1], labs[2]))
    key = raw * 3 + pick
    _, cells = np.unique(key, return_inverse=True)
    return cells.ravel(), np.array(["E", "H", "P"])[choice]


def ockm_t2(X, k, gamma=8, p=P_DEFAULT, family="orbit", refine=2, ls_steps=1.0, max_iter=100, summary="charts",
            seed=0, return_info=False):
    """OCKM-T with a greedy orbit-cut tree to k leaves and a local orbit-chart summary
    (summary='charts'), or a deeper greedy tree to gamma*k leaves (summary='tree')."""
    n, d = X.shape
    Z, G = project(X, p, seed)
    if summary == "tree":
        return ockm_t(X, k, gamma=gamma, p=p, family=family, refine=refine, ls_steps=ls_steps,
                      max_iter=max_iter, seed=seed, return_info=return_info)
    leaf, _, used = orbit_tree(Z, k, k_snap=None, family=family, refine=refine)
    cells, charts = local_chart_summary(Z, leaf, gamma)
    inv, w, Msp, R = _leaf_stats(cells, Z)
    w = w.astype(float); mleaf = len(w)
    sqZ = np.einsum("ij,ij->i", Z, Z)
    phi = float(sqZ.sum() - np.einsum("ij,ij->i", R, R) @ w)
    # tree start on the summary: centroid of each tree leaf
    _, Lr, lsz = np.unique(leaf, return_inverse=True, return_counts=True)
    Lsp = sparse.csr_matrix((np.ones(n), (Lr.ravel(), np.arange(n))), shape=(len(lsz), n))
    tree_init = np.asarray(Lsp @ Z) / lsz[:, None]
    kk = min(k, mleaf)
    starts = [("tree", tree_init[:kk])] if len(tree_init) >= kk else []
    c0, _ = kmeans_plusplus(R, kk, sample_weight=w, n_local_trials=1, random_state=seed + 1)
    starts.append(("D2", c0))
    steps = int(round(ls_steps * kk))
    cands = []
    for name, C0 in starts:
        if steps > 0:
            C0, _, _ = local_search_pp(R, w, C0, steps, seed=seed + 2)
        km = KMeans(kk, init=C0, n_init=1, max_iter=max_iter, algorithm="lloyd", random_state=seed).fit(R, sample_weight=w)
        cands.append((name, km))
    name, best = min(cands, key=lambda t: t[1].inertia_)
    lab_leaf = best.labels_
    Sfull = np.asarray(Msp @ X)
    num = np.asarray(sparse.csr_matrix((np.ones(mleaf), (lab_leaf, np.arange(mleaf))), shape=(kk, mleaf)) @ Sfull)
    den = np.bincount(lab_leaf, weights=w, minlength=kk)
    C = np.empty((k, d)); okc = den > 0
    C[:kk][okc] = num[okc] / den[okc, None]
    if (~okc).any() or kk < k:
        order = np.argsort(-w); full_leaf = Sfull / w[:, None]
        fill = [i for i in range(k) if i >= kk or not okc[i]]
        C[fill] = full_leaf[order[:len(fill)]]
    if return_info:
        from collections import Counter
        return C, dict(m=mleaf, phi=phi, summary_cost=float(best.inertia_), winner=name,
                       cand_costs={nm: float(km.inertia_) for nm, km in cands}, cuts=used,
                       charts=dict(Counter(charts.tolist())))
    return C
