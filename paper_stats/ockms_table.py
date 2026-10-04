import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "ockm"), os.path.join(ROOT, "heldout"), HERE]
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
DEV_RUNS = os.path.join(ROOT, "results", "development", "runs.csv")
DEV_S_RUNS = os.path.join(ROOT, "results", "development", "runs_ockms.csv")
HELDOUT_RUNS = os.path.join(ROOT, "results", "heldout", "results_heldout_all.csv")
import sys, numpy as np, pandas as pd
import heldout_bench as hb
A = pd.read_csv(DEV_RUNS)
S = pd.read_csv(DEV_S_RUNS).drop_duplicates(["dataset", "method", "seed"])
best = pd.concat([A[["dataset", "WCSS"]], S[["dataset", "WCSS"]]]).groupby("dataset").WCSS.min()
S = S.rename(columns={"dataset": "ds"}); S["k"] = 0; S = S.rename(columns={"ds": "dataset"})
S["method"] = S.method.replace({"OCKM-S": "Circle tree"}); S["hit_max_iter"] = 0
D = hb.add_portfolios(S); D["gap"] = 100 * (D.WCSS / D.dataset.map(best) - 1)
H = hb.add_portfolios(pd.read_csv(HELDOUT_RUNS).drop_duplicates(["dataset","k","method","seed"]))
H["setting"] = H.dataset + H.k.astype(str); hb_ = H.groupby("setting").WCSS.min(); H["gap"] = 100 * (H.WCSS / H.setting.map(hb_) - 1)
H["dataset"] = H.setting
rows = [("OCKM-T", r"\ockmt{}", "1"), ("Circle tree", r"\ockms{}", "1"), ("OCKM-TC", r"\ockmt{} or \ockms{}, lower seeding cost", "1"),
        ("Line+Line (2 runs)", r"best of two \ockmt{} runs", "2"), ("Line+Circle (2 runs)", r"best of \ockmt{} and \ockms{}", "2"), ("KM++", r"$k$-means++ (greedy)", "1")]
with open(os.path.join(OUT, "table_ockms.tex"), "w") as f:
    f.write(r"""\begin{table}[t]
\centering
\caption{\ockms{} and combinations of the two trees. Gap: mean gap of the final cost (\%) over data sets
or settings; worst: largest gap of a single run; Lloyd: number of Lloyd runs. The combinations are formed from
the recorded runs with the same seed (two runs of \ockmt{}: seeds $s$ and $s+1$).}
\label{tab:ockms}
\small
\begin{tabular}{lrrrrr}
\toprule
& & \multicolumn{2}{c}{Development (\Nds{} sets)} & \multicolumn{2}{c}{Held-out (\HoNset{} settings)} \\
Method & Lloyd & Gap & Worst & Gap & Worst \\
\midrule
""")
    for m, nm, ll in rows:
        dg = D[D.method == m].groupby("dataset").gap.mean().mean(); dw = D[D.method == m].gap.max()
        hg = H[H.method == m].groupby("dataset").gap.mean().mean(); hw = H[H.method == m].gap.max()
        f.write(f"{nm} & {ll} & {dg:.3f} & {dw:.2f} & {hg:.3f} & {hw:.2f} \\\\\n")
        print(nm, round(dg,3), round(dw,2), round(hg,3), round(hw,2))
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
from scipy import stats
P = D.groupby(["dataset", "method"]).gap.mean().unstack()
for a, b in (("OCKM-TC", "OCKM-T"), ("Line+Line (2 runs)", "OCKM-T")):
    x = P[a] - P[b]; print(a, b, (x < -0.01).sum(), (x.abs() <= 0.01).sum(), (x > 0.01).sum(), stats.wilcoxon(x).pvalue)
