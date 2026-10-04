"""OCKM-S on the 42 development data sets: macros and a table."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "ockm"), os.path.join(ROOT, "heldout"), HERE]
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
DEV_RUNS = os.path.join(ROOT, "results", "development", "runs.csv")
DEV_S_RUNS = os.path.join(ROOT, "results", "development", "runs_ockms.csv")
HELDOUT_RUNS = os.path.join(ROOT, "results", "heldout", "results_heldout_all.csv")

import json, numpy as np, pandas as pd
from scipy import stats
A = pd.read_csv(DEV_RUNS)
S = pd.read_csv(DEV_S_RUNS).drop_duplicates(["dataset", "method", "seed"])
# reproducibility check of the rerun
m = S[S.method.isin(["OCKM-T", "KM++"])].merge(A, on=["dataset", "method", "seed"], suffixes=("", "_old"))
rel = (m.WCSS / m.WCSS_old - 1).abs()
print("rerun vs ICML runs: n =", len(m), "max rel diff", rel.max(), "share identical(1e-9)", (rel < 1e-9).mean())
best_old = A.groupby("dataset").WCSS.min()
best = pd.concat([A[["dataset", "WCSS"]], S[["dataset", "WCSS"]]]).groupby("dataset").WCSS.min()
newbest = (best < best_old * (1 - 1e-12))
print("datasets where OCKM-S run lowers best known:", list(best.index[newbest]), ((best_old / best - 1) * 100)[newbest].round(4).to_dict())
S["gap"] = 100 * (S.WCSS / S.dataset.map(best) - 1)
S["gap_old"] = 100 * (S.WCSS / S.dataset.map(best_old) - 1)
g = S.groupby(["dataset", "method"]).gap.mean().unstack()
worst = S.groupby(["dataset", "method"]).gap.max().unstack()
tm = S.groupby(["dataset", "method"]).total_time.median().unstack()
stime = S.groupby(["dataset", "method"]).seed_time.median().unstack()
sc = S.groupby(["dataset", "method"]).seed_cost.mean().unstack()
ex = S[S.method == "OCKM-S"].extra.apply(json.loads)
Sx = S[S.method == "OCKM-S"].assign(phi=[e["phi"] for e in ex], sh=[e["shell_cuts"] for e in ex], se=[e["sector_cuts"] for e in ex])
ext = S[S.method == "OCKM-T"].extra.apply(json.loads)
Tx = S[S.method == "OCKM-T"].assign(phi=[e["phi"] for e in ext])
phiS = Sx.groupby("dataset").phi.mean(); phiT = Tx.groupby("dataset").phi.mean()
out = {}
def wtl(a, b, sub=None):
    x = g[a] - g[b]
    if sub is not None: x = x[sub]
    return int((x < -0.01).sum()), int((x.abs() <= 0.01).sum()), int((x > 0.01).sum()), stats.wilcoxon(x).pvalue
r = g[["OCKM-T", "OCKM-S", "KM++"]].rank(axis=1).mean()
for nm, key in (("OCKM-T", "T"), ("OCKM-S", "S"), ("KM++", "KMpp")):
    out[f"DvGap{key}"] = f"{g[nm].mean():.2f}"; out[f"DvWorst{key}"] = f"{worst[nm].max():.2f}"
    out[f"DvRank{key}"] = f"{r[nm]:.2f}"
    out[f"DvTime{key}"] = f"{(tm[nm] / tm['KM++']).median():.2f}"
    out[f"DvSeedTime{key}"] = f"{(stime[nm] / stime['KM++']).median():.2f}"
w = wtl("OCKM-T", "OCKM-S"); out["DvTvsS"] = f"{w[0]}/{w[1]}/{w[2]}"; out["DvTvsSP"] = f"{w[3]:.2f}"
w = wtl("OCKM-S", "KM++"); out["DvSvsKMpp"] = f"{w[0]}/{w[1]}/{w[2]}"; e_ = int(np.floor(np.log10(w[3]))); out["DvSvsKMppP"] = f"{w[3]/10**e_:.0f}{{\\cdot}}10^{{{e_}}}"
w = wtl("OCKM-T", "KM++"); out["DvTvsKMpp"] = f"{w[0]}/{w[1]}/{w[2]}"
out["DvPhiSoverT"] = f"{np.exp(np.log(phiS / phiT).mean()):.2f}"
out["DvPhiSlarger"] = int((phiS > phiT * (1 + 1e-9)).sum())
out["DvShellShare"] = f"{100 * Sx.sh.sum() / (Sx.sh.sum() + Sx.se.sum()):.0f}"
shd = (Sx.groupby("dataset").sh.sum() / (Sx.groupby("dataset").sh.sum() + Sx.groupby("dataset").se.sum()))
out["DvShellMax"] = f"{100 * shd.max():.0f}"; out["DvShellMaxDs"] = shd.idxmax()
seedw = (sc["OCKM-T"] < sc["OCKM-S"] * (1 - 1e-9)).sum(); out["DvSeedTbetter"] = int(seedw)
out["DvSeedSoverT"] = f"{np.exp(np.log(sc['OCKM-S'] / sc['OCKM-T']).mean()):.3f}"
out["DvNewBest"] = int(newbest.sum())
# portfolios
rows = []
for ds, D in S.groupby("dataset"):
    L = D[D.method == "OCKM-T"].set_index("seed").sort_index(); C = D[D.method == "OCKM-S"].set_index("seed")
    seeds = list(L.index)
    for i, s in enumerate(seeds):
        rows.append(dict(dataset=ds, method="LC", gap=min(L.loc[s, "gap"], C.loc[s, "gap"])))
        rows.append(dict(dataset=ds, method="LL", gap=min(L.loc[s, "gap"], L.loc[seeds[(i + 1) % len(seeds)], "gap"])))
        tc = L.loc[s] if L.loc[s, "seed_cost"] <= C.loc[s, "seed_cost"] else C.loc[s]
        rows.append(dict(dataset=ds, method="TC", gap=tc.gap))
P = pd.DataFrame(rows).groupby(["dataset", "method"]).gap.mean().unstack()
x = P.LC - P.LL
out["DvPortWTL"] = f"{(x < -0.01).sum()}/{(x.abs() <= 0.01).sum()}/{(x > 0.01).sum()}"; out["DvPortP"] = f"{stats.wilcoxon(x).pvalue:.2f}"
out["DvGapLC"] = f"{P.LC.mean():.3f}"; out["DvGapLL"] = f"{P.LL.mean():.3f}"; out["DvGapTC"] = f"{P.TC.mean():.3f}"
# data sets where OCKM-S is clearly better / worse
d = (g["OCKM-S"] - g["OCKM-T"])
out["DvSbetterList"] = ", ".join(f"{i} ({-v:.2f})" for i, v in d[d < -0.1].sort_values().items())
out["DvSworseList"] = ", ".join(f"{i} ({v:.2f})" for i, v in d[d > 0.1].sort_values(ascending=False).items())
with open(os.path.join(OUT, "numbers_dev_s.tex"), "w") as f:
    for k_, v in out.items(): f.write(f"\\newcommand{{\\{k_}}}{{{v}}}\n")
for k_, v in out.items(): print(k_, v)
print(g.round(2).to_string())
g.to_csv(os.path.join(OUT, "dev_s_gaps.csv"))
