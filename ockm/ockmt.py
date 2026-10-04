"""OCKM-T: orbit-cut tree summaries for k-means seeding (compiled implementation).

Pipeline
1. Projection. Centre X and, when d > p, multiply by a fixed Gaussian matrix G (d x p).
2. Orbit-cut tree (deterministic). Starting from one cell, the cell with the largest SSE
   is split until m = gamma * k cells exist. For a cell with principal plane (u, v), the
   candidate cuts are hyperplanes whose normals form the orbit of the first principal
   direction under rotations of the principal plane, n_t = u cos t + v sin t,
   t = j pi / n_dirs, j = 0..n_dirs-1. Along each normal the offset is chosen by a
   binned scan of the SSE reduction in the leading r principal coordinates. The PCA-Part
   cut (t = 0, offset 0) is always a candidate. Among the candidates the cut with the
   largest EXACT SSE reduction n ||S_L||^2 / (n_L n_R) is chosen, and it is refined by a
   few 2-means steps inside the cell (each step moves to the bisector of the two means,
   so the SSE never increases).
3. Summary search. The m leaves form a weighted summary (centroid, count). Two starts on
   the summary: the k-leaf partition of the tree and one D^2 (k-means++) sample. Each
   start is improved by LocalSearch++ swaps and weighted Lloyd; the lower summary cost wins.
4. Lift and polish. Each summary cluster is mapped to the mean of the original points in
   its leaves, and a few weighted Lloyd steps on the full-dimensional leaf centroids remove
   the projection distortion.

Guarantee: E[cost(X, C)] <= (1 + 2 alpha) Phi + 2 alpha OPT_k(X), where Phi is the
within-leaf cost (returned in info) and alpha is the approximation factor of the
summary algorithm (see the paper, Section 4).
"""
from __future__ import annotations
import math, time
import numpy as np
import numba as nb
from scipy import sparse
from sklearn.cluster import kmeans_plusplus
from threadpoolctl import ThreadpoolController

_TPC = None


def _one_blas_thread():
    """Context that limits BLAS to one thread. The tree calls BLAS only on small matrices
    (p x p covariances, n_c x 4 products), where multi-threaded BLAS is slower and very slow
    when another process competes for the cores."""
    global _TPC
    if _TPC is None:
        _TPC = ThreadpoolController()
    return _TPC.limit(limits=1, user_api="blas")

P_DEFAULT = 64


def project(X, p=P_DEFAULT, seed=0):
    """Centred data in the working space: X - mean (d <= p) or (X - mean) G (d > p)."""
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    mu = X.mean(0)
    if d <= p:
        return np.ascontiguousarray(X - mu), None, mu
    G = np.random.default_rng(seed).standard_normal((d, p)) / math.sqrt(p)
    return np.ascontiguousarray(X @ G - mu @ G), G, mu


# ----------------------------------------------------------------------------- heap
@nb.njit(cache=True)
def _hpush(hk, hv, size, key, val):
    i = size
    hk[i] = key; hv[i] = val
    while i > 0:
        par = (i - 1) >> 1
        if hk[par] >= hk[i]:
            break
        hk[par], hk[i] = hk[i], hk[par]
        hv[par], hv[i] = hv[i], hv[par]
        i = par
    return size + 1


@nb.njit(cache=True)
def _hpop(hk, hv, size):
    key, val = hk[0], hv[0]
    size -= 1
    hk[0] = hk[size]; hv[0] = hv[size]
    i = 0
    while True:
        l = 2 * i + 1; r = l + 1; b = i
        if l < size and hk[l] > hk[b]:
            b = l
        if r < size and hk[r] > hk[b]:
            b = r
        if b == i:
            break
        hk[b], hk[i] = hk[i], hk[b]
        hv[b], hv[i] = hv[i], hv[b]
        i = b
    return key, val, size


# ----------------------------------------------------------------------------- tree
@nb.njit(cache=True)
def _lead_dirs(Cv, Q0, rr, iters):
    """Leading rr eigenvectors of the p x p PSD matrix Cv (columns, decreasing order).
    Exact (eigh) for p <= 16; otherwise block subspace iteration with block size
    b = Q0.shape[1] >= rr from the start Q0, followed by Rayleigh-Ritz."""
    p = Cv.shape[0]
    if p <= 16:
        ev, V = np.linalg.eigh(Cv)
        out = np.empty((p, rr))
        for q in range(rr):
            out[:, q] = V[:, p - 1 - q]
        return out, out
    Q = Q0.copy()
    for it in range(iters):
        Qn, _ = np.linalg.qr(np.dot(Cv, Q))
        Q = np.ascontiguousarray(Qn)
    H = np.dot(Q.T, np.dot(Cv, Q))
    ev, W = np.linalg.eigh(H)
    b = Q.shape[1]
    Wr = np.empty((b, b))
    for q in range(b):
        Wr[:, q] = W[:, b - 1 - q]
    full = np.dot(Q, Wr)
    out = np.ascontiguousarray(full[:, :rr])
    return out, full


@nb.njit(cache=True)
def _orbit_tree(Z, m, k_snap, n_dirs, nbins, r_scan, refine, cov_cap, pca_only, block=4, iters=6, fam=0):
    """Greedy divisive tree on the rows of Z (n x p). Returns leaf labels, leaf sums,
    counts and SSEs, the centroids of the k_snap-leaf partition, the number of leaves,
    and how often each candidate direction was chosen (last entry: PCA-Part cut)."""
    n, p = Z.shape
    Zs = Z.copy()
    perm = np.arange(n)
    start = np.zeros(m, np.int64); end = np.zeros(m, np.int64)
    csum = np.zeros((m, p)); cnt = np.zeros(m, np.int64); csse = np.zeros(m)
    bb = min(block, p)
    basis = np.zeros((m, p, bb))
    np.random.seed(12345)
    Q0, _ = np.linalg.qr(np.random.standard_normal((p, bb)))
    basis[0] = Q0
    used = np.zeros(n_dirs + 1, np.int64)
    end[0] = n; cnt[0] = n
    for i in range(n):
        for j in range(p):
            csum[0, j] += Zs[i, j]
    mu0 = csum[0] / n
    s0 = 0.0
    for i in range(n):
        for j in range(p):
            t = Zs[i, j] - mu0[j]
            s0 += t * t
    csse[0] = s0
    hk = np.zeros(2 * m + 2); hv = np.zeros(2 * m + 2, np.int64)
    hsize = _hpush(hk, hv, 0, s0, 0)
    tol = 1e-12 * max(s0, 1e-300)
    nleaves = 1
    snapC = np.zeros((k_snap, p))
    snap_done = False
    if k_snap == 1:
        snapC[0] = mu0
        snap_done = True
    r = min(r_scan, p)
    angs = np.zeros(n_dirs)
    for a in range(n_dirs):
        angs[a] = a * math.pi / n_dirs
    nfix = n_dirs + 4 if fam >= 1 else 0
    nscan = n_dirs if fam != 1 else 0
    ncand = 1 if pca_only else nscan + nfix + 1
    while nleaves < m and hsize > 0:
        key, j, hsize = _hpop(hk, hv, hsize)
        if key <= tol:
            break
        s = start[j]; e = end[j]; nc = e - s
        if nc < 2:
            continue
        mu = csum[j] / nc
        B = np.empty((nc, p))                              # centred rows of the cell
        sqB = np.empty(nc)
        for i in range(nc):
            acc = 0.0
            for jj in range(p):
                t = Zs[s + i, jj] - mu[jj]
                B[i, jj] = t
                acc += t * t
            sqB[i] = acc
        # principal directions (covariance of at most cov_cap rows, strided)
        if p == 1:
            Vr = np.ones((1, 1))
            rr = 1
        else:
            if nc > cov_cap:
                step = nc // cov_cap
                Bs = np.ascontiguousarray(B[::step])
            else:
                Bs = B
            if p <= 4:
                nb_ = Bs.shape[0]
                Cv = np.zeros((p, p))
                for i in range(nb_):
                    for j1 in range(p):
                        t = Bs[i, j1]
                        for j2 in range(j1, p):
                            Cv[j1, j2] += t * Bs[i, j2]
                for j1 in range(p):
                    for j2 in range(j1):
                        Cv[j1, j2] = Cv[j2, j1]
            else:
                Cv = np.dot(Bs.T, Bs)
            rr = r if not pca_only else 1
            rr = min(rr, bb)
            Vr, full = _lead_dirs(Cv, np.ascontiguousarray(basis[j]), rr, iters if nleaves > 1 else 5 * iters)
            if full.shape[1] == bb:
                basis[j] = full
        Y = np.dot(B, Vr)                                  # nc x rr leading principal coordinates
        # candidate cuts: (direction index, threshold); last one is the PCA-Part cut
        cdir = np.zeros(ncand, np.int64); cthr = np.zeros(ncand)
        cdir[ncand - 1] = 0; cthr[ncand - 1] = 0.0
        if (not pca_only) and nfix > 0:
            for a in range(n_dirs):
                cdir[nscan + a] = a; cthr[nscan + a] = 0.0
            u0 = np.sort(Y[:, 0])
            qs = (0.25, 0.375, 0.625, 0.75)
            for qi in range(4):
                cdir[nscan + n_dirs + qi] = 0
                cthr[nscan + n_dirs + qi] = u0[min(nc - 1, int(qs[qi] * (nc - 1)))]
        if (not pca_only) and nscan > 0:
            proj = np.empty(nc)
            bcnt = np.zeros(nbins); bsum = np.zeros((nbins, rr))
            for a in range(n_dirs):
                ca = math.cos(angs[a]); sa = math.sin(angs[a])
                lo = 1e300; hi = -1e300
                for i in range(nc):
                    val = Y[i, 0] * ca + (Y[i, 1] * sa if rr > 1 else 0.0)
                    proj[i] = val
                    if val < lo:
                        lo = val
                    if val > hi:
                        hi = val
                cdir[a] = a
                if hi - lo <= 0.0:
                    cthr[a] = 0.0
                    continue
                width = (hi - lo) / nbins
                bcnt[:] = 0.0; bsum[:, :] = 0.0
                for i in range(nc):
                    bi = int((proj[i] - lo) / width)
                    if bi >= nbins:
                        bi = nbins - 1
                    bcnt[bi] += 1.0
                    for q in range(rr):
                        bsum[bi, q] += Y[i, q]
                best = -1.0; bq = -1
                PL = np.zeros(rr); nL = 0.0
                for bi in range(nbins - 1):
                    nL += bcnt[bi]
                    for q in range(rr):
                        PL[q] += bsum[bi, q]
                    nR = nc - nL
                    if nL <= 0.0 or nR <= 0.0 or bcnt[bi] == 0.0:
                        continue
                    nn = 0.0
                    for q in range(rr):
                        nn += PL[q] * PL[q]
                    red = nc * nn / (nL * nR)
                    if red > best:
                        best = red; bq = bi
                cthr[a] = lo + (bq + 1) * width if bq >= 0 else 0.0
        # exact SSE reduction of every candidate
        nLc = np.zeros(ncand)
        ccos = np.empty(ncand); csin = np.empty(ncand)
        for c in range(ncand):
            ccos[c] = math.cos(angs[cdir[c]]); csin[c] = math.sin(angs[cdir[c]])
        if p <= 16:
            SL = np.zeros((ncand, p))
            M = np.zeros((1, ncand))
            for i in range(nc):
                y0 = Y[i, 0]; y1 = Y[i, 1] if rr > 1 else 0.0
                for c in range(ncand):
                    if y0 * ccos[c] + y1 * csin[c] > cthr[c]:
                        nLc[c] += 1.0
                        for jj in range(p):
                            SL[c, jj] += B[i, jj]
        else:
            M = np.zeros((nc, ncand))
            for i in range(nc):
                y0 = Y[i, 0]; y1 = Y[i, 1] if rr > 1 else 0.0
                for c in range(ncand):
                    if y0 * ccos[c] + y1 * csin[c] > cthr[c]:
                        M[i, c] = 1.0
                        nLc[c] += 1.0
            SL = np.dot(M.T, B)                            # ncand x p side sums
        bestc = -1; bestred = -1.0
        for c in range(ncand):
            nR = nc - nLc[c]
            if nLc[c] <= 0.0 or nR <= 0.0:
                continue
            nn = 0.0
            for jj in range(p):
                nn += SL[c, jj] * SL[c, jj]
            red = nc * nn / (nLc[c] * nR)
            if red > bestred + 1e-15 * abs(bestred):
                bestred = red; bestc = c
        if bestc < 0:
            continue                                       # cell cannot be split
        used[cdir[bestc] if bestc < ncand - 1 else n_dirs] += 1
        left = np.empty(nc, np.bool_)
        for i in range(nc):
            y0 = Y[i, 0]; y1 = Y[i, 1] if rr > 1 else 0.0
            left[i] = y0 * ccos[bestc] + y1 * csin[bestc] > cthr[bestc]
        sL = SL[bestc].copy(); nl = nLc[bestc]
        # 2-means refinement inside the cell
        for it in range(refine):
            nr = nc - nl
            dv = np.empty(p); thr = 0.0
            for jj in range(p):
                aL = sL[jj] / nl; aR = -sL[jj] / nr
                dv[jj] = aR - aL
                thr += 0.5 * (aR * aR - aL * aL)
            pr = np.dot(B, dv)
            nl2 = 0.0; changed = 0
            for i in range(nc):
                lf = pr[i] < thr
                if lf != left[i]:
                    changed += 1
                if lf:
                    nl2 += 1.0
            if changed == 0 or nl2 == 0.0 or nl2 == nc:
                break
            sL[:] = 0.0
            for i in range(nc):
                left[i] = pr[i] < thr
                if left[i]:
                    for jj in range(p):
                        sL[jj] += B[i, jj]
            nl = nl2
        # child statistics
        nlI = int(nl)
        sqL = 0.0; sqAll = 0.0
        for i in range(nc):
            sqAll += sqB[i]
            if left[i]:
                sqL += sqB[i]
        nn = 0.0
        for jj in range(p):
            nn += sL[jj] * sL[jj]
        sseL = max(sqL - nn / nl, 0.0)
        sseR = max((sqAll - sqL) - nn / (nc - nl), 0.0)
        # reorder the segment: left rows first
        tmpZ = Zs[s:e].copy(); tmpP = perm[s:e].copy()
        a0 = s; b0 = s + nlI
        for i in range(nc):
            if left[i]:
                Zs[a0] = tmpZ[i]; perm[a0] = tmpP[i]; a0 += 1
            else:
                Zs[b0] = tmpZ[i]; perm[b0] = tmpP[i]; b0 += 1
        nid = nleaves
        start[nid] = s + nlI; end[nid] = e; cnt[nid] = nc - nlI
        end[j] = s + nlI; cnt[j] = nlI
        for jj in range(p):
            full = csum[j, jj]
            csum[j, jj] = sL[jj] + nl * mu[jj]
            csum[nid, jj] = full - csum[j, jj]
        csse[j] = sseL; csse[nid] = sseR
        basis[nid] = basis[j]
        hsize = _hpush(hk, hv, hsize, sseL, j)
        hsize = _hpush(hk, hv, hsize, sseR, nid)
        nleaves += 1
        if (not snap_done) and nleaves == k_snap:
            for c in range(k_snap):
                snapC[c] = csum[c] / cnt[c]
            snap_done = True
    if not snap_done:
        for c in range(min(k_snap, nleaves)):
            snapC[c] = csum[c] / cnt[c]
    labels = np.empty(n, np.int64)
    for c in range(nleaves):
        for i in range(start[c], end[c]):
            labels[perm[i]] = c
    return labels, csum[:nleaves].copy(), cnt[:nleaves].copy(), csse[:nleaves].copy(), snapC, nleaves, used


# ----------------------------------------------------------------------------- distances
def _top2_blas(R, C, chunk=4096):
    """Nearest and second-nearest centre of every row of R (squared distances)."""
    m, k = len(R), len(C)
    a = np.zeros(m, np.int64); b = np.zeros(m, np.int64)
    d1 = np.zeros(m); d2 = np.full(m, np.inf)
    sqC = np.einsum("ij,ij->i", C, C)
    for i in range(0, m, chunk):
        Rb = R[i:i + chunk]
        D = np.maximum(np.einsum("ij,ij->i", Rb, Rb)[:, None] + sqC[None, :] - 2.0 * Rb @ C.T, 0.0)
        rr = np.arange(len(Rb))
        if k == 1:
            a[i:i + chunk] = 0; b[i:i + chunk] = 0; d1[i:i + chunk] = D[:, 0]
            continue
        part = np.argpartition(D, 1, axis=1)[:, :2]
        v0, v1 = D[rr, part[:, 0]], D[rr, part[:, 1]]
        sw = v1 < v0
        a[i:i + chunk] = np.where(sw, part[:, 1], part[:, 0]); b[i:i + chunk] = np.where(sw, part[:, 0], part[:, 1])
        d1[i:i + chunk] = np.minimum(v0, v1); d2[i:i + chunk] = np.maximum(v0, v1)
    return a, b, d1, d2


@nb.njit(cache=True)
def _ls_core(R, w, C, a, b, d1, d2, steps, seed):
    """LocalSearch++ (Lattanzi and Sohler, 2019) on weighted points. Swap costs for all k
    centres in O(m) per step from nearest / second-nearest bookkeeping. A point whose
    triangle-inequality lower bound |d(x, c_a) - d(r, c_a)| on its distance to the
    candidate x is at least sqrt(d2) is unaffected by the swap and is not evaluated."""
    np.random.seed(seed)
    m, p = R.shape; k = C.shape[0]
    cost = 0.0
    for i in range(m):
        cost += w[i] * d1[i]
    rc = np.zeros(k)                                   # removal cost of every centre
    for i in range(m):
        rc[a[i]] += w[i] * (d2[i] - d1[i])
    cum = np.empty(m); dx = np.empty(m); delta = np.empty(k); dcx = np.empty(k)
    sq1 = np.sqrt(d1)
    acc = 0
    for st in range(steps):
        tot = 0.0
        for i in range(m):
            tot += w[i] * d1[i]
            cum[i] = tot
        if tot <= 0.0:
            break
        x = np.searchsorted(cum, np.random.random() * tot, side="right")
        if x >= m:
            x = m - 1
        for q in range(k):
            t = 0.0
            for jj in range(p):
                df = R[x, jj] - C[q, jj]
                t += df * df
            dcx[q] = math.sqrt(t)
        keep = cost
        for q in range(k):
            delta[q] = rc[q]
        for i in range(m):
            lb = dcx[a[i]] - sq1[i]
            if lb < 0.0:
                lb = -lb
            if lb * lb >= d2[i]:
                dx[i] = np.inf                             # dx >= d2: contribution already in rc
                continue
            t = 0.0
            for jj in range(p):
                df = R[i, jj] - R[x, jj]
                t += df * df
            dx[i] = t
            if t < d1[i]:
                keep -= w[i] * (d1[i] - t)
                base = t
            else:
                base = d1[i]
            alt = t if t < d2[i] else d2[i]
            delta[a[i]] += w[i] * ((alt - base) - (d2[i] - d1[i]))
        c = 0
        for q in range(1, k):
            if delta[q] < delta[c]:
                c = q
        new = keep + delta[c]
        if new >= cost * (1.0 - 1e-12):
            continue
        acc += 1
        for jj in range(p):
            C[c, jj] = R[x, jj]
        for i in range(m):
            if a[i] == c or b[i] == c:
                f1 = np.inf; f2 = np.inf; i1 = 0; i2 = 0
                for q in range(k):
                    t = 0.0
                    for jj in range(p):
                        df = R[i, jj] - C[q, jj]
                        t += df * df
                    if t < f1:
                        f2 = f1; i2 = i1; f1 = t; i1 = q
                    elif t < f2:
                        f2 = t; i2 = q
                a[i] = i1; b[i] = i2; d1[i] = f1; d2[i] = f2
            elif dx[i] < d1[i]:
                b[i] = a[i]; d2[i] = d1[i]; a[i] = c; d1[i] = dx[i]
            elif dx[i] < d2[i]:
                b[i] = c; d2[i] = dx[i]
        cost = 0.0
        rc[:] = 0.0
        for i in range(m):
            cost += w[i] * d1[i]
            rc[a[i]] += w[i] * (d2[i] - d1[i])
            sq1[i] = math.sqrt(d1[i])
    return cost, acc


def local_search_pp(R, w, C, steps, seed=0):
    C = np.array(C, dtype=np.float64, copy=True)
    if steps <= 0 or len(C) < 2:
        a, b, d1, d2 = _top2_blas(R, C)
        return C, float(w @ d1), 0
    a, b, d1, d2 = _top2_blas(R, C)
    cost, acc = _ls_core(np.ascontiguousarray(R), w, C, a, b, d1, d2, int(steps), int(seed))
    return C, float(cost), int(acc)


@nb.njit(cache=True)
def _assign_small(R, C):
    m, p = R.shape; k = C.shape[0]
    lab = np.zeros(m, np.int64); dmin = np.zeros(m)
    for i in range(m):
        best = np.inf; bi = 0
        for q in range(k):
            t = 0.0
            for jj in range(p):
                df = R[i, jj] - C[q, jj]
                t += df * df
            if t < best:
                best = t; bi = q
        lab[i] = bi; dmin[i] = best
    return lab, dmin


def _assign(R, C, chunk=4096):
    m, p = R.shape; k = len(C)
    if m * k * p <= 2_000_000:
        return _assign_small(R, C)
    lab = np.empty(m, np.int64); dmin = np.empty(m)
    sqC = np.einsum("ij,ij->i", C, C)
    for i in range(0, m, chunk):
        Rb = R[i:i + chunk]
        D = sqC[None, :] - 2.0 * Rb @ C.T
        l = D.argmin(1)
        lab[i:i + chunk] = l
        dmin[i:i + chunk] = np.maximum(D[np.arange(len(Rb)), l] + np.einsum("ij,ij->i", Rb, Rb), 0.0)
    return lab, dmin


@nb.njit(cache=True)
def _wlloyd_small(R, w, C, max_iter):
    m, p = R.shape; k = C.shape[0]
    lab, dmin = _assign_small(R, C)
    S = np.zeros((k, p)); W = np.zeros(k)
    for it in range(max_iter):
        S[:, :] = 0.0; W[:] = 0.0
        for i in range(m):
            W[lab[i]] += w[i]
            for jj in range(p):
                S[lab[i], jj] += w[i] * R[i, jj]
        for q in range(k):
            if W[q] > 0:
                for jj in range(p):
                    C[q, jj] = S[q, jj] / W[q]
        lab2, dmin = _assign_small(R, C)
        same = True
        for i in range(m):
            if lab2[i] != lab[i]:
                same = False
                break
        lab = lab2
        if same:
            break
    cost = 0.0
    for i in range(m):
        cost += w[i] * dmin[i]
    return C, lab, cost


@nb.njit(cache=True)
def d2_seeding(R, w, k, seed):
    """Vanilla D^2 sampling (k-means++ without greedy trials) on weighted points. A point is
    skipped when the triangle inequality shows that the new centre cannot be closer."""
    np.random.seed(seed)
    m, p = R.shape
    C = np.empty((k, p))
    cum = np.cumsum(w)
    x = np.searchsorted(cum, np.random.random() * cum[-1], side="right")
    x = min(x, m - 1)
    C[0] = R[x]
    d = np.empty(m); a = np.zeros(m, np.int64); dcc = np.empty(k)
    for i in range(m):
        t = 0.0
        for jj in range(p):
            df = R[i, jj] - C[0, jj]
            t += df * df
        d[i] = t
    for c in range(1, k):
        tot = 0.0
        for i in range(m):
            tot += w[i] * d[i]
            cum[i] = tot
        if tot <= 0.0:
            x = c % m
        else:
            x = np.searchsorted(cum, np.random.random() * tot, side="right")
            x = min(x, m - 1)
        C[c] = R[x]
        for q in range(c):
            t = 0.0
            for jj in range(p):
                df = C[c, jj] - C[q, jj]
                t += df * df
            dcc[q] = t
        for i in range(m):
            if dcc[a[i]] >= 4.0 * d[i]:
                continue
            t = 0.0
            for jj in range(p):
                df = R[i, jj] - C[c, jj]
                t += df * df
            if t < d[i]:
                d[i] = t; a[i] = c
    return C


@nb.njit(cache=True)
def _top2_small(R, C):
    m, p = R.shape; k = C.shape[0]
    a = np.zeros(m, np.int64); b = np.zeros(m, np.int64); d1 = np.empty(m); d2 = np.empty(m)
    for i in range(m):
        f1 = np.inf; f2 = np.inf; i1 = 0; i2 = 0
        for q in range(k):
            t = 0.0
            for jj in range(p):
                df = R[i, jj] - C[q, jj]
                t += df * df
            if t < f1:
                f2 = f1; i2 = i1; f1 = t; i1 = q
            elif t < f2:
                f2 = t; i2 = q
        a[i] = i1; b[i] = i2; d1[i] = f1; d2[i] = f2
    return a, b, d1, d2


@nb.njit(cache=True)
def _summary_search_small(R, w, Ctree, S, steps, max_iter, seed, ls_d2):
    """All summary starts in compiled code (small problems): start 0 is the tree start,
    starts 1..S-1 are D^2 samples; each gets LocalSearch++ (the D^2 starts only if ls_d2)
    and weighted Lloyd. Returns the best centres, their leaf labels, and all costs."""
    k = Ctree.shape[0]
    costs = np.empty(S)
    bestC = Ctree.copy(); bestlab = np.zeros(R.shape[0], np.int64); bestcost = np.inf
    for si in range(S):
        if si == 0:
            C0 = Ctree.copy()
        else:
            C0 = d2_seeding(R, w, k, seed + 1 + 7919 * si)
        if steps > 0 and k > 1 and (si == 0 or ls_d2):
            a, b, d1, d2 = _top2_small(R, C0)
            _ls_core(R, w, C0, a, b, d1, d2, steps, seed + 2 + 7919 * si)
        C1, lab, cst = _wlloyd_small(R, w, C0, max_iter)
        costs[si] = cst
        if cst < bestcost:
            bestcost = cst; bestC = C1.copy(); bestlab = lab.copy()
    return bestC, bestlab, bestcost, costs


def weighted_lloyd(R, w, C, max_iter=100):
    """Weighted Lloyd on (R, w) from C until the assignment is stable or max_iter updates.
    Empty clusters keep their centre. Returns centres, labels and cost of the returned
    centres (the labels are the assignment to the returned centres)."""
    R = np.ascontiguousarray(R); C = np.array(C, dtype=np.float64, copy=True)
    k = len(C)
    if R.shape[0] * k * R.shape[1] <= 2_000_000:
        return _wlloyd_small(R, np.ascontiguousarray(w, dtype=np.float64), C, int(max_iter))
    lab, dmin = _assign(R, C)
    for it in range(max_iter):
        W = np.bincount(lab, weights=w, minlength=k)
        S = sparse.csr_matrix((w, (lab, np.arange(len(R)))), shape=(k, len(R))) @ R
        nz = W > 0
        C[nz] = S[nz] / W[nz, None]
        lab2, dmin = _assign(R, C)
        if np.array_equal(lab2, lab):
            lab = lab2
            break
        lab = lab2
    return C, lab, float(w @ dmin)


# ----------------------------------------------------------------------------- main
def _n_starts(n, k, m, n_starts, max_starts):
    """Number of summary starts. 'auto': the summary search gets about the work of the
    tree, S * k * m <= 64 n, with 2 <= S <= max_starts."""
    if n_starts == "auto":
        return int(min(max_starts, max(2, (64 * n) // max(1, k * m))))
    return int(n_starts)


def summary_search(R, w, Ctree, S, steps, max_iter=100, seed=0, use_tree=True):
    """Weighted k-means on the summary (R, w). Start 0 is the tree start (if use_tree),
    the other starts are D^2 samples; every start gets `steps` LocalSearch++ swaps and
    weighted Lloyd. Returns the centres, leaf labels and cost of the best start, and the
    costs of all starts."""
    R = np.ascontiguousarray(R, dtype=np.float64); w = np.ascontiguousarray(w, dtype=np.float64)
    k = len(Ctree)
    if use_tree and R.shape[0] * k * R.shape[1] <= 2_000_000:
        return _summary_search_small(R, w, np.ascontiguousarray(Ctree), S, steps, int(max_iter), int(seed), True)
    costs = np.empty(S); best = None
    for si in range(S):
        C0 = np.array(Ctree, dtype=np.float64) if (si == 0 and use_tree) else d2_seeding(R, w, k, seed + 1 + 7919 * si)
        if steps > 0 and k > 1:
            C0, _, _ = local_search_pp(R, w, C0, steps, seed=seed + 2 + 7919 * si)
        Cs, lab, cst = weighted_lloyd(R, w, C0, max_iter=max_iter)
        costs[si] = cst
        if best is None or cst < best[2]:
            best = (Cs, lab, cst)
    return best[0], best[1], best[2], costs


def ockm_t(X, k, **kw):
    """OCKM-T seeding; see _ockm_t_impl for the parameters. Small problems run with one BLAS
    thread throughout (their matrix products are too small to gain from threads)."""
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] * X.shape[1] * min(X.shape[1], kw.get("p", P_DEFAULT)) < 5e7:
        with _one_blas_thread():
            return _ockm_t_impl(X, k, **kw)
    return _ockm_t_impl(X, k, **kw)


def _ockm_t_impl(X, k, gamma=8, p=P_DEFAULT, n_dirs=4, nbins=64, r_scan=2, refine=2, cov_cap=4096, block=4, iters=6,
           pca_only=False, ls_steps=2.0, n_starts="auto", max_starts=16, use_tree=True, max_iter=100,
           polish_budget=5e9, seed=0, return_info=False):
    """OCKM-T seeding. Returns a k x d array of centres (and an info dict).

    gamma: leaves per centre (m = gamma * k); p: projection dimension (used when d > p);
    n_dirs: rotations of the principal direction searched per split; nbins: bins of the
    offset scan; refine: 2-means steps per split; pca_only: PCA-Part cuts only (ablation);
    ls_steps: LocalSearch++ swaps per start, as a multiple of k; n_starts: summary starts
    ('auto' or an integer); use_tree: include the tree start; polish_budget: flops allowed
    for the weighted Lloyd polish on the full-dimensional leaf centroids (0 disables)."""
    tt = {}
    t0 = time.perf_counter()
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    k = int(min(k, n))
    Z, G, mu = project(X, p, seed)
    tt["project"] = time.perf_counter() - t0; t0 = time.perf_counter()
    m = int(min(n, max(k, round(gamma * k))))
    with _one_blas_thread():
        labels, csum, cnt, csse, snapC, L, used = _orbit_tree(Z, m, k, n_dirs, nbins, r_scan, 0 if pca_only else refine,
                                                              cov_cap, pca_only, block, iters, 0)
    tt["tree"] = time.perf_counter() - t0; t0 = time.perf_counter()
    w = cnt.astype(np.float64)
    R = csum / w[:, None]
    phi = float(csse.sum())
    kk = min(k, L)
    S = _n_starts(n, kk, L, n_starts, max_starts)
    steps = int(round(ls_steps * kk)) if kk > 1 else 0
    Cs, lab_leaf, scost, costs = summary_search(R, w, snapC[:kk], S, steps, max_iter, seed, use_tree)
    tt["summary"] = time.perf_counter() - t0; t0 = time.perf_counter()
    # lift: mean of the original points in the leaves of each summary cluster
    if G is None:
        Sfull = csum + w[:, None] * mu[None, :]           # leaf sums in the original space
    else:
        Sfull = np.asarray(sparse.csr_matrix((np.ones(n), (labels, np.arange(n))), shape=(L, n)) @ X)
    num = np.asarray(sparse.csr_matrix((np.ones(L), (lab_leaf, np.arange(L))), shape=(kk, L)) @ Sfull)
    den = np.bincount(lab_leaf, weights=w, minlength=kk)
    C = np.empty((k, d))
    okc = den > 0
    C[:kk][okc] = num[okc] / den[okc, None]
    if (~okc).any() or kk < k:
        order = np.argsort(-w)
        fill = [i for i in range(k) if i >= kk or not okc[i]]
        C[fill] = (Sfull / w[:, None])[order[:len(fill)]]
    polish_it = 0
    if polish_budget > 0 and G is not None:
        polish_it = int(max(1, min(50, polish_budget // max(1, L * k * d))))
        C, _, _ = weighted_lloyd(Sfull / w[:, None], w, C, max_iter=polish_it)
    tt["lift"] = time.perf_counter() - t0
    if return_info:
        if G is None:
            phi_full = phi
        else:                                            # within-leaf cost in the original space (one pass)
            tot = float(np.einsum("ij,ij->", X, X) - n * mu @ mu)
            D = Sfull - w[:, None] * mu[None, :]
            phi_full = max(tot - float(np.einsum("ij,ij->i", D, D) @ (1.0 / w)), 0.0)
        wi = int(np.argmin(costs))
        names = [("tree" if (i == 0 and use_tree) else f"D2_{i}") for i in range(S)]
        return C, dict(m=L, phi=phi, phi_full=phi_full, summary_cost=float(scost), winner=names[wi], n_starts=S, polish_it=polish_it,
                       cand_costs={nm: float(c) for nm, c in zip(names, costs)},
                       cuts={f"dir{j}": int(used[j]) for j in range(n_dirs)} | {"pca": int(used[n_dirs])},
                       times=tt)
    return C


def warmup():
    """Compile the numba kernels once (they are cached on disk afterwards)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 5))
    ockm_t(X, 4)
    ockm_t(X, 4, pca_only=True)
    ockm_t(X, 4, n_starts=1, ls_steps=0, max_iter=0)
    ockm_t(rng.standard_normal((300, 80)), 4)
    ockm_t(rng.standard_normal((3000, 20)), 60)          # large-summary path
