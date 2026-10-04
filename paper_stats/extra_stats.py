"""Extra macros: low/high-dimensional split of the development benchmark and two-run combinations."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "ockm"), os.path.join(ROOT, "heldout"), HERE]
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
DEV_RUNS = os.path.join(ROOT, "results", "development", "runs.csv")
DEV_S_RUNS = os.path.join(ROOT, "results", "development", "runs_ockms.csv")
HELDOUT_RUNS = os.path.join(ROOT, "results", "heldout", "results_heldout_all.csv")

import sys, numpy as np, pandas as pd
from scipy import stats
import heldout_bench as hb
A = pd.read_csv(DEV_RUNS)
S = pd.read_csv(DEV_S_RUNS).drop_duplicates(["dataset", "method", "seed"])
best = pd.concat([A[["dataset", "WCSS"]], S[["dataset", "WCSS"]]]).groupby("dataset").WCSS.min()
S["method"] = S.method.replace({"OCKM-S": "Circle tree"}); S["hit_max_iter"] = 0
D = hb.add_portfolios(S); D["gap"] = 100 * (D.WCSS / D.dataset.map(best) - 1)
P = D.groupby(["dataset", "method"]).gap.mean().unstack()
dd = A.groupby("dataset").d.first(); low = dd[dd <= 3].index; high = dd.index.difference(low)
out = {"NlowDim": len(low), "GapKMppLow": f"{P.loc[low, 'KM++'].mean():.2f}", "GapKMppHigh": f"{P.loc[high, 'KM++'].mean():.2f}",
       "GapTHigh": f"{P.loc[high, 'OCKM-T'].mean():.2f}"}
x = P.loc[high, "OCKM-T"] - P.loc[high, "KM++"]
out["WTLHigh"] = f"{(x < -0.01).sum()}/{(x.abs() <= 0.01).sum()}/{(x > 0.01).sum()}"
def w(a, b):
    x = P[a] - P[b]; return f"{(x < -0.01).sum()}/{(x.abs() <= 0.01).sum()}/{(x > 0.01).sum()}", stats.wilcoxon(x).pvalue
out["DvTCWTL"], p = w("OCKM-TC", "OCKM-T"); out["DvTCP"] = f"{p:.3f}"
out["DvBtwoWTL"], p = w("Line+Line (2 runs)", "OCKM-T"); e = int(np.floor(np.log10(p))); out["DvBtwoP"] = f"{p/10**e:.0f}\\cdot10^{{{e}}}"
with open(os.path.join(OUT, "numbers_extra.tex"), "w") as f:
    for k, v in out.items(): f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(out)
