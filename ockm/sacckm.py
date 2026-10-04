"""
SAC-CKM prototype: Symmetry-Adaptive Concentric-region K-Means seeding
with corrected one-parameter-subgroup charts.

Everything (densities, centroids, Lloyd, metrics) is computed in the
standardised d-dimensional feature space. The one-parameter subgroup acts on
the plane spanned by the first two principal directions of the standardised
data and trivially on the orthogonal complement. That plane is used ONLY to
define the invariant coordinate I(x) and the orbit coordinate s(x).

Charts (u, v) = coordinates in the PC plane, centred at the global centroid:

  regime      generator A        flow T(t)            I(x)            s(x)
  elliptic    [[0,-1],[1,0]]     rotation by t        sqrt(u²+v²)     atan2(v,u)
  hyperbolic  [[1,0],[0,-1]]     (e^t u, e^-t v)      u*v             ½ ln(|u|/|v|)  (+ sign(u) label)
  parabolic   [[0,1],[0,0]]      (u + t v, v)         v               u / v

Along the flow, I is constant and s shifts by exactly t. Bands are
equal-count (quantile) intervals of I, sectors are equal-count intervals of s,
so the whole B x Q partition is shifted rigidly by the flow: the seeding is
equivariant under the chosen subgroup.

CCKM (Algorithm 1 of the manuscript) is implemented verbatim for comparison.

[Implementation of the earlier OCKM work.
 Functional code unchanged. The only addition is the optional argument
 `hash_signs` of sac_init_regime / sac_min3 / sac_init_subsample: when True the
 complement sign pattern is hashed with np.unique on packed bits instead of the
 integer shift `key * 2**d`, which overflows int64 for d > 62 (the original
 code assumes d <= 62). For d <= 62 both give the same partition.]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

EPS = 1e-12


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_dataset(path: str):
    df = pd.read_csv(path)
    df.columns = [str(c).strip().strip('"') for c in df.columns]
    label_col = [c for c in df.columns if c.lower() == "class"][0]
    y = pd.factorize(df[label_col].astype(str))[0]
    X = df.drop(columns=[label_col]).to_numpy(dtype=float)
    return X, y


def standardise(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


# ----------------------------------------------------------------------------
# CCKM (Algorithm 1 of the manuscript, d-dimensional sign regions)
# ----------------------------------------------------------------------------
def _sign_key(v: np.ndarray) -> tuple:
    # non-positive coordinate (0) precedes positive (1); first coordinate primary
    return tuple((v > 0).astype(int).tolist())


def cckm_init(D: np.ndarray, k: int) -> dict:
    n, d = D.shape
    C = D.mean(axis=0)
    dist = np.linalg.norm(D - C, axis=1)
    r_min, r_max = dist.min(), dist.max()
    h = (r_max - r_min) / k
    centres = np.empty((k, d))
    info = {"empty_annuli": 0}
    if h == 0:
        # degenerate radial range: rank sign regions of D by cardinality
        keys = [_sign_key(x - C) for x in D]
        uniq, counts = np.unique(np.array(keys), axis=0, return_counts=True)
        order = np.lexsort((*[uniq[:, j] for j in range(d - 1, -1, -1)], -counts))
        regs = uniq[order]
        for i in range(k):
            if i < len(regs):
                m = np.all(np.array(keys) == regs[i], axis=1)
                centres[i] = D[m].mean(axis=0)
            else:
                centres[i] = C
        return {"centres": centres, **info}
    r = r_min + h * np.arange(k + 1)
    for i in range(1, k + 1):
        m = (dist > r[i - 1]) & (dist <= r[i])
        A = D[m]
        if len(A) == 0:
            centres[i - 1] = C
            info["empty_annuli"] += 1
            continue
        keys = (A - C > 0).astype(int)
        uniq, counts = np.unique(keys, axis=0, return_counts=True)
        best = counts.max()
        cand = uniq[counts == best]
        # ties: lexicographic order of sign pattern, non-positive first, first coord primary
        cand = cand[np.lexsort([cand[:, j] for j in range(d - 1, -1, -1)])]
        tq = cand[0]
        mm = np.all(keys == tq, axis=1)
        centres[i - 1] = A[mm].mean(axis=0)
    return {"centres": centres, **info}


# ----------------------------------------------------------------------------
# SAC-CKM
# ----------------------------------------------------------------------------
REGIMES = ("E", "H", "P")


def pc_plane(D: np.ndarray):
    """Return (u, v) coordinates in the first two principal directions of the
    centred standardised data, and the 2 x d basis."""
    C = D.mean(axis=0)
    Z = D - C
    if D.shape[1] == 1:
        Z = np.hstack([Z, np.zeros_like(Z)])
    # principal directions from the d x d covariance (O(n d^2), no n x d factor);
    # deterministic sign convention (largest |loading| positive)
    evals, evecs = np.linalg.eigh(Z.T @ Z)
    W = evecs[:, ::-1][:, :2].T.copy()
    for j in range(2):
        i = np.argmax(np.abs(W[j]))
        if W[j, i] < 0:
            W[j] *= -1
    uv = Z @ W.T
    return uv[:, 0], uv[:, 1], W, C


def chart(u: np.ndarray, v: np.ndarray, regime: str):
    """Invariant coordinate I, orbit coordinate s, and discrete sign label
    (for the hyperbolic regime, where a level set of I=uv has two branches)."""
    if regime == "E":
        I = np.sqrt(u * u + v * v)
        s = np.arctan2(v, u)
        lab = np.zeros_like(u, dtype=int)
    elif regime == "H":
        I = u * v
        s = 0.5 * np.log((np.abs(u) + EPS) / (np.abs(v) + EPS))
        lab = (u > 0).astype(int)
    elif regime == "P":
        I = v
        denom = np.where(np.abs(v) < EPS, np.where(v < 0, -EPS, EPS), v)
        s = u / denom
        lab = np.zeros_like(u, dtype=int)
    else:
        raise ValueError(regime)
    return I, s, lab


def quantile_bins(x: np.ndarray, nb: int) -> np.ndarray:
    """Equal-count bins by rank (ties broken by index, deterministic)."""
    order = np.argsort(x, kind="stable")
    b = np.empty(len(x), dtype=int)
    b[order] = (np.arange(len(x)) * nb) // len(x)
    return b


@dataclass
class SACResult:
    centres: np.ndarray
    regime: str
    init_sse: float
    n_cells: int
    cell_counts: dict = field(default_factory=dict)
    seeds_cells: list = field(default_factory=list)
    sse_by_regime: dict = field(default_factory=dict)
    chart_data: Optional[dict] = None


def _sse(D: np.ndarray, centres: np.ndarray) -> float:
    # memory-safe: ||x||^2 + ||c||^2 - 2 x.c, in blocks of 100k points
    tot = 0.0; cn = (centres ** 2).sum(1)
    for i in range(0, len(D), 100_000):
        X = D[i:i + 100_000]
        d2 = (X ** 2).sum(1)[:, None] + cn[None, :] - 2.0 * X @ centres.T
        tot += float(np.maximum(d2.min(axis=1), 0).sum())
    return tot


def _select_seeds(cands: np.ndarray, counts: np.ndarray, k: int, expected: float,
                  mode: str = "dff", alpha: float = 0.5):
    """Deterministic seed selection from candidate cell centroids.

    mode = 'dff'  : density-filtered farthest-first (parameter-free).
                    Start from the densest cell; then repeatedly take, among cells
                    with count >= expected (the mean cell occupancy), the one
                    farthest from the already selected seeds. If the dense pool is
                    exhausted, continue farthest-first over all remaining cells.
    mode = 'score': alpha * density~ + (1-alpha) * separation~ (the weighted rule
                    from the original SAC-CKM proposal), kept for ablation.
    """
    m = len(cands)
    chosen = [int(np.argmax(counts))]
    if m == 1:
        return chosen
    dens = counts / counts.max()
    diam = (np.max(np.linalg.norm(cands[:, None] - cands[None, :], axis=-1)) + EPS) if mode == "score" else 1.0
    sep = np.linalg.norm(cands - cands[chosen[0]], axis=1)  # running min distance to chosen seeds
    while len(chosen) < min(k, m):
        sep = sep.copy()
        sep[chosen] = -np.inf
        if mode == "dff":
            pool = (counts >= expected)
            pool[chosen] = False
            if not pool.any():
                pool = np.ones(m, bool); pool[chosen] = False
            score = np.where(pool, sep, -np.inf)
        else:
            score = alpha * dens + (1 - alpha) * sep / diam
            score[chosen] = -np.inf
        j = int(np.argmax(score)); chosen.append(j)
        sep = np.minimum(sep, np.linalg.norm(cands - cands[j], axis=1))
    return chosen


def _complement_key(key: np.ndarray, sign: np.ndarray, hash_signs: bool) -> np.ndarray:
    n, d = sign.shape
    if not hash_signs:
        sk = np.zeros(n, dtype=np.int64)
        for j in range(d):
            sk = sk * 2 + sign[:, j]
        return key * (2 ** d) + sk
    # overflow-free: label each distinct packed sign row, then combine with key
    packed = np.packbits(sign.astype(np.uint8), axis=1)
    _, sid = np.unique(packed, axis=0, return_inverse=True)
    sid = sid.ravel().astype(np.int64)
    return key * (int(sid.max()) + 1) + sid


def sac_init_regime(D: np.ndarray, k: int, regime: str, B: Optional[int] = None,
                    Q: int = 4, mode: str = "dff", alpha: float = 0.5,
                    keep_chart: bool = False, complement: bool = False,
                    density_ref: str = "auto", plane=None, compute_sse: bool = True,
                    hash_signs: bool = False) -> SACResult:
    """complement=True additionally splits every (band, sector) cell by the sign
    pattern of the coordinates orthogonal to the PC plane (in the original
    standardised axes, as CCKM does). The subgroup acts trivially there, so the
    partition stays equivariant while regaining d-dimensional resolution."""
    n, d = D.shape
    B = k if B is None else B
    u, v, W, C = pc_plane(D) if plane is None else plane
    I, s, lab = chart(u, v, regime)
    bI = quantile_bins(I, B)
    bS = quantile_bins(s, Q)
    key = (bI * (Q * 2) + bS * 2 + lab).astype(np.int64)
    if complement and d > 2:
        Z = D - C
        Zc = Z - (Z @ W.T) @ W  # component orthogonal to the PC plane
        sign = (Zc > 1e-12).astype(np.int64)
        key = _complement_key(key, sign, hash_signs or d > 62)
    uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    cands = np.stack([np.bincount(inv, weights=D[:, j], minlength=len(uniq)) for j in range(d)], axis=1) / counts[:, None]
    # density reference: expected occupancy under independence of the I and s
    # bins for the plain quantile grid; mean occupancy of populated cells when the
    # complement split makes the number of possible cells data-dependent.
    # density_ref: 'indep' = n/(B*Q); 'popmean' = mean occupancy of populated cells;
    # 'auto' = popmean when the complement split is active (d > 2), else indep.
    if density_ref == "auto":
        density_ref = "popmean" if (complement and d > 2) else "indep"
    expected = counts.mean() if density_ref == "popmean" else n / (B * Q)
    chosen = _select_seeds(cands, counts, k, expected, mode, alpha)
    centres = cands[chosen]
    if len(centres) < k:  # fewer populated cells than k: pad with global centroid
        centres = np.vstack([centres, np.repeat(C[None], k - len(centres), 0)])
    res = SACResult(centres=centres, regime=regime, init_sse=(_sse(D, centres) if compute_sse else np.nan),
                    n_cells=len(uniq), seeds_cells=[int(uniq[j]) for j in chosen],
                    cell_counts={int(uniq[j]): int(counts[j]) for j in range(len(uniq))})
    if keep_chart:
        res.chart_data = dict(u=u, v=v, I=I, s=s, lab=lab, bI=bI, bS=bS, W=W, C=C,
                              cell_key=key, cand=cands, chosen=chosen)
    return res


def cckm_cells_init(D: np.ndarray, k: int, bands: str = "equal", mode: str = "dff") -> SACResult:
    """Ablation: CCKM's own d-dimensional cells (annulus x sign region about the
    global centroid) but with the SAC selection rule instead of one-seed-per-annulus.
    bands = 'equal'    : k equal-width annuli exactly as CCKM (empty annuli possible)
    bands = 'quantile' : k equal-count radial bands
    No PCA plane, no subgroup chart: isolates the effect of the selection step."""
    n, d = D.shape
    C = D.mean(axis=0)
    Z = D - C
    dist = np.linalg.norm(Z, axis=1)
    if bands == "equal":
        r_min, r_max = dist.min(), dist.max()
        h = (r_max - r_min) / k
        if h == 0:
            b = np.zeros(n, int)
        else:
            b = np.clip(np.ceil((dist - r_min) / h).astype(int) - 1, 0, k - 1)
            b[dist <= r_min] = 0  # innermost point joins band 1
    else:
        b = quantile_bins(dist, k)
    sign = (Z > 0).astype(np.int64)
    # hash the sign pattern per point
    sk = np.zeros(n, dtype=np.int64)
    for j in range(d):
        sk = sk * 2 + sign[:, j]  # d <= 62 assumed
    key = b.astype(np.int64) * (2 ** d) + sk
    uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    cands = np.stack([np.bincount(inv, weights=D[:, j], minlength=len(uniq)) for j in range(d)], axis=1) / counts[:, None]
    expected = counts.mean()  # mean occupancy of the populated cells
    chosen = _select_seeds(cands, counts, k, expected, mode)
    centres = cands[chosen]
    if len(centres) < k:
        centres = np.vstack([centres, np.repeat(C[None], k - len(centres), 0)])
    return SACResult(centres=centres, regime="cckm-" + bands, init_sse=_sse(D, centres),
                     n_cells=len(uniq), seeds_cells=[int(uniq[j]) for j in chosen])


def sac_init(D: np.ndarray, k: int, regime: str = "auto", **kw) -> SACResult:
    """regime in {'auto','E','H','P'}. 'auto' evaluates the three charts and keeps
    the seeding with the lowest initial SSE (no Lloyd, no threshold)."""
    if regime != "auto":
        return sac_init_regime(D, k, regime, **kw)
    kw.setdefault("plane", pc_plane(D))
    results = {q: sac_init_regime(D, k, q, **kw) for q in REGIMES}
    best = min(REGIMES, key=lambda q: results[q].init_sse)  # ties -> E, H, P order
    out = results[best]
    out.sse_by_regime = {q: results[q].init_sse for q in REGIMES}
    return out


def sac_init_subsample(D: np.ndarray, k: int, n_sub: int = 5000, seed: int = 0, complement: bool = True, **kw):
    """Scalable regime selection: run Lloyd from the three charts on a fixed random
    subsample of n_sub points, keep the chart with the lowest subsample WCSS, then
    return the full-data seeding of that chart (one full Lloyd run follows)."""
    from sklearn.cluster import KMeans
    n = len(D)
    if n <= n_sub:
        return None  # caller should use the plain min3 rule
    idx = np.random.default_rng(seed).choice(n, n_sub, replace=False)
    Ds = D[idx]
    best = None
    for q in REGIMES:
        r = sac_init_regime(Ds, k, q, complement=complement, compute_sse=False, **kw)
        km = KMeans(n_clusters=k, init=r.centres, n_init=1, max_iter=300, tol=1e-4, algorithm="lloyd").fit(Ds)
        if best is None or km.inertia_ < best[1]:
            best = (q, km.inertia_)
    return sac_init_regime(D, k, best[0], complement=complement, compute_sse=False, **kw)


def sac_min3(D: np.ndarray, k: int, lloyd_fn, complement: bool = True, **kw):
    """Run Lloyd from the three charts (PC plane computed once) and return
    (result, fitted_model, init_time_seconds) of the lowest final WCSS."""
    import time as _t
    t0 = _t.perf_counter(); plane = pc_plane(D); ti = _t.perf_counter() - t0
    best = None
    for q in REGIMES:
        ta = _t.perf_counter(); r = sac_init_regime(D, k, q, complement=complement, plane=plane, compute_sse=False, **kw); ti += _t.perf_counter() - ta
        km = lloyd_fn(D, k, r.centres)
        if best is None or km.inertia_ < best[1].inertia_: best = (r, km)
    return best[0], best[1], ti
