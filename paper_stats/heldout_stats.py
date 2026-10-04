"""Held-out study: tables and macros for the TMLR paper, from results_heldout_all.csv."""
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
d = pd.read_csv(HELDOUT_RUNS).drop_duplicates(["dataset","k","method","seed"])
d = hb.add_portfolios(d)
REN = {"Circle tree": "OCKM-S", "Line+Circle (2 runs)": "best of OCKM-T, OCKM-S", "Line+Line (2 runs)": "best of 2 OCKM-T"}
d["method"] = d.method.replace(REN)
d["setting"] = d.dataset + " k=" + d.k.astype(str)
best = d.groupby("setting").WCSS.min()
d["gap"] = 100 * (d.WCSS / d.setting.map(best) - 1)
d["sgap"] = 100 * (d.seed_cost / d.setting.map(best) - 1)
MAIN = ["OCKM-T", "OCKM-S", "KM++", "KM++v", "Qk", "AFK-MC2", "k-means||", "PCA-Part", "Var-Part", "T-pca"]
NAME = {"OCKM-T": r"\ockmt{}", "OCKM-S": r"\ockms{}", "KM++": r"$k$-means++ (greedy)", "KM++v": r"$k$-means++ (vanilla)",
        "Qk": "Qkmeans", "AFK-MC2": r"AFK-MC$^2$", "k-means||": r"$k$-means$\|$", "PCA-Part": "PCA-Part", "Var-Part": "Var-Part",
        "T-pca": r"\ockmt{}, PCA-Part cuts"}
D = d[d.method.isin(MAIN)]
g = D.groupby(["setting", "method"]).gap.mean().unstack()[MAIN]
worst = D.groupby(["setting", "method"]).gap.max().unstack()[MAIN]
sg = D.groupby(["setting", "method"]).sgap.mean().unstack()[MAIN]
tm = D.groupby(["setting", "method"]).total_time.median().unstack()[MAIN]
st = D.groupby(["setting", "method"]).seed_time.median().unstack()[MAIN]
ranks = g.rank(axis=1).mean()
fp = stats.friedmanchisquare(*g.T.values).pvalue
rows = []; pv = []
for m in MAIN:
    if m == "OCKM-T":
        rows.append((m, None, None)); continue
    x = g["OCKM-T"] - g[m]
    w, t, l = int((x < -0.01).sum()), int((x.abs() <= 0.01).sum()), int((x > 0.01).sum())
    p = stats.wilcoxon(x).pvalue
    rows.append((m, f"{w}/{t}/{l}", p)); pv.append(p)
holm = hb._holm(np.array(pv)); hi = iter(holm)
M = {}
def fmt_p(p):
    if p is None: return "--"
    if p >= 0.01: return f"{p:.2f}"
    e = int(np.floor(np.log10(p))); return f"${p/10**e:.0f}{{\\cdot}}10^{{{e}}}$"
lines = []
for m, wtl, p in rows:
    ph = next(hi) if p is not None else None
    tr = (tm[m] / tm["OCKM-T"]).median()
    rk = ranks[m]
    cells = [NAME[m], f"{rk:.2f}", f"{g[m].mean():.2f}", f"{worst[m].max():.2f}", f"{sg[m].mean():.1f}", wtl or "--", fmt_p(ph), f"{tr:.2f}"]
    if m == "OCKM-T":
        cells = [r"\textbf{" + c + "}" if i in (0,) else c for i, c in enumerate(cells)]
    lines.append((rk, " & ".join(cells) + r" \\"))
    M[m] = dict(rank=rk, gap=g[m].mean(), worst=worst[m].max(), sgap=sg[m].mean(), wtl=wtl, p=p, ph=ph, tr=tr)
lines.sort()
with open(os.path.join(OUT, "table_heldout.tex"), "w") as f:
    f.write(r"""\begin{table}[t]
\centering
\caption{Held-out study: ten settings on five data sets that were not used during development
(Table~\ref{tab:heldoutdata}). Rank: mean Friedman rank of the mean gap per setting; gap: mean gap of the final
cost to the best known cost in \%; worst: largest gap of any single run; seed gap: mean gap of the seeding cost
before Lloyd; W/T/L: settings on which \ockmt{} is better / tied / worse; Holm: Holm-corrected Wilcoxon $p$
against \ockmt{}; time: median ratio of seeding plus Lloyd time to \ockmt{}. Friedman $p=\HoFriedP$.}
\label{tab:heldout}
\small
\begin{tabular}{lrrrrrrr}
\toprule
Method & Rank & Gap & Worst & Seed gap & W/T/L & Holm & Time \\
\midrule
""")
    for _, l in lines: f.write(l + "\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
# per-setting table
COLS = ["OCKM-T", "OCKM-S", "KM++", "KM++v", "Qk", "AFK-MC2", "k-means||", "PCA-Part"]
SH = {"OCKM-T": r"\ockmt{}", "OCKM-S": r"\ockms{}", "KM++": "KM++", "KM++v": "KM++v", "Qk": "Qk", "AFK-MC2": "AFK", "k-means||": r"KM$\|$", "PCA-Part": "PCA"}
order = ["20ng k=20", "20ng k=200", "agnews k=4", "agnews k=100", "cifar10 k=10", "cifar10 k=100", "cifar10 k=1000", "cifar100 k=100", "covtype k=7", "covtype k=100"]
DSN = {"20ng": "20 Newsgroups", "agnews": "AG News", "cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "covtype": "Covertype"}
with open(os.path.join(OUT, "table_heldout_setting.tex"), "w") as f:
    f.write(r"""\begin{table}[t]
\centering
\caption{Held-out study: mean gap of the final cost (\%) per setting, with the largest gap of a single run in
parentheses for \ockmt{}, \ockms{} and greedy $k$-means++. Randomised methods: \ockmt{}, \ockms{} and greedy
$k$-means++ with ten runs (five at $k\ge100$ on AG News and Covertype and at $k=1000$), the other randomised methods with three runs.
The lowest mean per row is in bold.}
\label{tab:heldoutsetting}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lr""" + "r" * len(COLS) + r"""}
\toprule
Data set & $k$ & """ + " & ".join(SH[c] for c in COLS) + r""" \\
\midrule
""")
    for s in order:
        ds, kk = s.split(" k=")
        mn = g.loc[s, COLS].min()
        cells = []
        for c in COLS:
            v = g.loc[s, c]
            txt = f"{v:.2f}"
            if abs(v - mn) < 5e-3: txt = r"\textbf{" + txt + "}"
            if c in ("OCKM-T", "OCKM-S", "KM++"): txt += f" ({worst.loc[s, c]:.2f})"
            cells.append(txt)
        f.write(f"{DSN[ds]} & {kk} & " + " & ".join(cells) + r" \\" + "\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
# portfolios
P = d.groupby(["setting", "method"]).gap.mean().unstack()
def wtl(a, b, sub=None):
    x = (P[a] - P[b]); x = x if sub is None else x[sub]
    return int((x < -0.01).sum()), int((x.abs() <= 0.01).sum()), int((x > 0.01).sum()), stats.wilcoxon(x).pvalue
big = [s for s in P.index if int(s.split("k=")[1]) >= 100]
pf = wtl("best of OCKM-T, OCKM-S", "best of 2 OCKM-T"); pfb = wtl("best of OCKM-T, OCKM-S", "best of 2 OCKM-T", big)
st_ = wtl("OCKM-T", "OCKM-S"); b2 = wtl("best of 2 OCKM-T", "OCKM-T"); tc = wtl("OCKM-TC", "OCKM-T")
# equal time
eq = hb.equal_time(HELDOUT_RUNS)
# ARI at k = classes
cls = {"20ng": 20, "agnews": 4, "cifar10": 10, "cifar100": 100, "covtype": 7}
A = d[(d.k == d.dataset.map(cls)) & d.method.isin(["OCKM-T", "OCKM-S", "KM++"])].groupby(["dataset", "method"]).ARI.mean().unstack()
out = {}
def put(k, v): out[k] = v
put("HoNset", len(g)); put("HoNruns", len(d[d.method.isin(MAIN)])); put("HoFriedP", fmt_p(fp).strip("$"))
for key, m in (("T", "OCKM-T"), ("S", "OCKM-S"), ("KMpp", "KM++"), ("KMppv", "KM++v"), ("Qk", "Qk"), ("AFK", "AFK-MC2"), ("KMpar", "k-means||"), ("PCA", "PCA-Part"), ("Var", "Var-Part"), ("Tpca", "T-pca")):
    put(f"HoGap{key}", f"{M[m]['gap']:.2f}"); put(f"HoWorst{key}", f"{M[m]['worst']:.2f}"); put(f"HoRank{key}", f"{M[m]['rank']:.2f}")
    put(f"HoSeedGap{key}", f"{M[m]['sgap']:.1f}"); put(f"HoTime{key}", f"{M[m]['tr']:.2f}")
    if M[m]["wtl"]: put(f"HoWTL{key}", M[m]["wtl"]); put(f"HoP{key}", f"{M[m]['p']:.2f}"); put(f"HoHolm{key}", f"{M[m]['ph']:.2f}" if M[m]['ph']>=0.01 else fmt_p(M[m]['ph']))
put("HoPortWTL", f"{pf[0]}/{pf[1]}/{pf[2]}"); put("HoPortP", f"{pf[3]:.2f}")
put("HoPortBigWTL", f"{pfb[0]}/{pfb[1]}/{pfb[2]}")
put("HoTSWTL", f"{st_[0]}/{st_[1]}/{st_[2]}"); put("HoTSP", f"{st_[3]:.2f}")
put("HoBtwoWTL", f"{b2[0]}/{b2[1]}/{b2[2]}"); put("HoBtwoP", f"{b2[3]:.3f}")
put("HoTCWTL", f"{tc[0]}/{tc[1]}/{tc[2]}"); put("HoTCP", f"{tc[3]:.2f}")
put("HoGapPortLC", f"{P['best of OCKM-T, OCKM-S'].mean():.3f}"); put("HoGapPortLL", f"{P['best of 2 OCKM-T'].mean():.3f}"); put("HoGapTC", f"{P['OCKM-TC'].mean():.3f}")
# selected settings
for s, tag in (("covtype k=100", "CovH"), ("cifar10 k=1000", "CifK"), ("covtype k=7", "CovS"), ("agnews k=100", "AgH")):
    for m, mm in (("OCKM-T", "T"), ("KM++", "KMpp"), ("OCKM-S", "S")):
        put(f"Ho{tag}Gap{mm}", f"{g.loc[s, m]:.2f}"); put(f"Ho{tag}Worst{mm}", f"{worst.loc[s, m]:.2f}")
        put(f"Ho{tag}Time{mm}", f"{tm.loc[s, m]:.2f}")
if len(eq):
    for m, key in (("KM++", "KMpp"), ("KM++v", "KMppv"), ("Qk", "Qk"), ("AFK-MC2", "AFK"), ("k-means||", "KMpar")):
        if m in eq.index: put(f"HoEq{key}", eq.loc[m, "wtl"])
for ds in A.index:
    for m in A.columns:
        put(f"HoARI{ {'20ng':'ng','agnews':'ag','cifar10':'ciften','cifar100':'cifhund','covtype':'cov'}[ds]}{m.replace('-','').replace('+','p')}", f"{A.loc[ds, m]:.3f}")
with open(os.path.join(OUT, "numbers_heldout.tex"), "w") as f:
    for k_, v in out.items(): f.write(f"\\newcommand{{\\{k_}}}{{{v}}}\n")
print(open(os.path.join(OUT, "table_heldout.tex")).read()); print(open(os.path.join(OUT, "table_heldout_setting.tex")).read())
for k_, v in out.items(): print(k_, v)
print(eq); print(A); print(tm.round(2).to_string()); print(st.round(2).to_string())
