"""Drop-in replacement for sacckm._select_seeds with identical output.

The original evaluates np.linalg.norm(cands - cands[j], axis=1) on the full
m x d candidate matrix at every step. With the complement split on image data
m is close to n (60,000 x 784), so each step allocates a 376 MB temporary.
Here the same expression is evaluated on row blocks. Each row's norm is
computed by exactly the same NumPy operations, so the selected cells are
identical (checked in test_sac_fast.py)."""
import numpy as np
import sacckm

EPS = sacckm.EPS
_BLOCK = 4096


def _rownorm_to(cands, c):
    out = np.empty(len(cands))
    for i in range(0, len(cands), _BLOCK):
        out[i:i + _BLOCK] = np.linalg.norm(cands[i:i + _BLOCK] - c, axis=1)
    return out


def _select_seeds(cands, counts, k, expected, mode="dff", alpha=0.5):
    m = len(cands)
    chosen = [int(np.argmax(counts))]
    if m == 1:
        return chosen
    dens = counts / counts.max()
    diam = (np.max(np.linalg.norm(cands[:, None] - cands[None, :], axis=-1)) + EPS) if mode == "score" else 1.0
    sep = _rownorm_to(cands, cands[chosen[0]])
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
        sep = np.minimum(sep, _rownorm_to(cands, cands[j]))
    return chosen


_original = sacckm._select_seeds


def install():
    sacckm._select_seeds = _select_seeds
