"""Generate numbers.tex and table files for the manuscript from the result CSVs.
Every number quoted in the paper text comes from here (macros), so it can be re-checked."""
import os, json, numpy as np, pandas as pd
from scipy import stats
import summarise_t3 as S

RES = "../results/development"; OUT = "../paper_stats/out"; os.makedirs(OUT, exist_ok=True)
d = pd.read_csv(os.path.join(RES, "runs.csv"))
d, s = S.per_dataset(d)
et = pd.read_csv(os.path.join(RES, "t_equal_time.csv"))
M = {}
def put(k, v): M[k] = v
def pfmt(p):
    """p-value for use inside math mode in the text."""
    if p >= 0.001: return f"{p:.3f}"
    m, e = f"{p:.0e}".split("e"); return f"{m}\\times10^{{{int(e)}}}"
def sci(p):
    """p-value for LaTeX tables: 0.53, 0.068, 2e-6 -> $2{\cdot}10^{-6}$."""
    if p >= 0.01: return f"{p:.2f}"
    m, e = f"{p:.0e}".split("e"); return f"${m}{{\\cdot}}10^{{{int(e)}}}$"

MAIN = S.MAIN; ABL = S.ABL
put("Nds", s.dataset.nunique()); put("Nmeth", len(MAIN)); put("Nbase", len(MAIN) - 1)
put("Nsyn", int(s.dataset.isin(S.SYNTH).sum() / s.method.nunique())); 

def ranks(metric, methods, higher=False, sub=None):
    t = s.pivot_table(index="dataset", columns="method", values=metric)[methods].dropna()
    if sub is not None: t = t.loc[t.index.isin(sub)]
    return t, t.rank(axis=1, ascending=not higher).mean()

def wtl(t, a, b, tol=0.01):
    diff = t[a] - t[b]
    return int((diff < -tol).sum()), int((diff.abs() <= tol).sum()), int((diff > tol).sum())

tg, rg = ranks("gap", MAIN)
put("RankT", f"{rg['OCKM-T']:.2f}"); put("RankKMpp", f"{rg['KM++']:.2f}"); put("RankOCKM", f"{rg['OCKM']:.2f}")
put("RankPCA", f"{rg['PCA-Part']:.2f}")
chi2, pf = stats.friedmanchisquare(*[tg[m].values for m in MAIN]); put("FriedP", f"{pf:.0e}".replace("e-", "\\times10^{-") + "}")
mg = s.groupby("method").gap.mean(); hit = s.groupby("method").hit01.mean(); medg = s.groupby("method").gap.median()
put("GapT", f"{mg['OCKM-T']:.2f}"); put("GapKMpp", f"{mg['KM++']:.1f}")
fast = [mg[m] for m in ("Qk", "AFK-MC2", "k-means||")]
put("GapFastRange", f"{min(fast):.0f} to {max(fast):.0f}\\%"); put("GapKMppv", f"{mg['KM++v']:.0f}")
put("HitT", f"{100*hit['OCKM-T']:.0f}"); put("HitKMpp", f"{100*hit['KM++']:.0f}")
# pairwise tests on gap
pv = {}; W = {}
for m in MAIN:
    if m == "OCKM-T": continue
    x = tg["OCKM-T"] - tg[m]
    pv[m] = stats.wilcoxon(tg["OCKM-T"], tg[m]).pvalue if (x != 0).any() else 1.0
    W[m] = wtl(tg, "OCKM-T", m)
keys = list(pv); hp = dict(zip(keys, S.holm([pv[k] for k in keys])))
put("HolmMax", pfmt(max(hp.values())))
w, t_, l = W["KM++"]; put("WTLKMpp", f"{w}/{t_}/{l}")
w, t_, l = W["OCKM"]; put("WTLOCKM", f"{w}/{t_}/{l}")
# seeding cost: datasets where OCKM-T has the lowest seed cost among MAIN
sc = s.pivot_table(index="dataset", columns="method", values="seed_ratio")[MAIN]
put("SeedBestT", int((sc["OCKM-T"] <= sc.min(axis=1) * (1 + 1e-12)).sum()))
ts, rs = ranks("seed_ratio", MAIN); put("SeedRankT", f"{rs['OCKM-T']:.2f}")
geo = np.exp(np.log(sc.div(sc["OCKM-T"], axis=0)).mean()); put("SeedKMppOverT", f"{geo['KM++']:.2f}"); put("SeedQkOverT", f"{geo['Qk']:.2f}")
# time ratios against one greedy k-means++ run
tk = s[s.method == "KM++"].set_index("dataset")
def tratio(m, col="total_time"):
    g = s[s.method == m].set_index("dataset"); return (g[col] / tk[col].reindex(g.index))
put("TimeRatioT", f"{tratio('OCKM-T').median():.2f}"); put("SeedTimeRatioT", f"{tratio('OCKM-T','seed_time').median():.2f}")
put("TimeRatioPca", f"{tratio('T-pca').median():.2f}")
# equal time
eqp = {}
for ref, g in et.groupby("ref"):
    b = int((g.gap < -0.01).sum()); w_ = int((g.gap > 0.01).sum()); t2 = len(g) - b - w_
    p = stats.wilcoxon(g.gap.values).pvalue
    eqp[ref] = (b, t2, w_, p, g.runs.mean(), g.gap.median())
put("EqTimePmax", pfmt(max(v[3] for v in eqp.values())))
put("EqMinWins", str(min(v[0] for v in eqp.values())))
put("EqRunsMedKMpp", f"{et[et.ref == 'KM++'].runs.median():.1f}")
b, t2, w_, p, r, mgap = eqp["KM++"]; put("EqKMpp", f"{b}/{t2}/{w_}"); put("EqKMppP", pfmt(p)); put("EqKMppRuns", f"{r:.1f}")
b, t2, w_, p, r, mgap = eqp["Qk"]; put("EqQk", f"{b}/{t2}/{w_}")
# phi / final cost
o = d[d.method == "OCKM-T"].copy(); ex = o.extra.apply(json.loads)
o["phi"] = ex.apply(lambda e: e.get("phi_full", e["phi"]))       # within-leaf cost in the ORIGINAL space
oq = o.groupby("dataset").agg(phi=("phi", "mean"), W=("WCSS", "mean"), k=("k", "first"))
q = oq.phi / oq.W
put("PhiRatioMin", f"{q.min():.3f}"); put("PhiRatioMax", f"{q.max():.2f}"); put("PhiRatioMed", f"{q.median():.2f}")
alpha = 8 * (np.log(oq.k) + 2)
put("BoundRatioMed", f"{((1 + 2 * alpha) * q).median():.0f}")
o["ns"] = ex.apply(lambda e: e["n_starts"]); o["tw"] = ex.apply(lambda e: e["winner"] == "tree")
put("NstartsSixteen", int((o.groupby("dataset").ns.first() == 16).sum()))
put("TreeWinShare", f"{100 * o.tw.mean():.0f}")
cuts = pd.DataFrame(list(ex.apply(lambda e: e["cuts"]))).sum(); cuts = cuts / cuts.sum()
put("CutDirZero", f"{100*cuts['dir0']:.0f}"); put("CutRot", f"{100*(cuts['dir1']+cuts['dir2']+cuts['dir3']):.0f}"); put("CutPca", f"{100*cuts['pca']:.1f}")
# phi of PCA-cut variant
t = d[d.method.isin(["OCKM-T", "T-pca"])].copy(); t["phi"] = t.extra.apply(lambda e: json.loads(e)["phi"])
P = t.groupby(["dataset", "method"]).phi.mean().unstack(); r = P["T-pca"] / P["OCKM-T"]
put("PhiPcaGain", f"{100*(1-1/np.exp(np.log(r).mean())):.0f}"); put("PhiPcaAll", int((r > 1).sum()))
SC2 = d[d.method.isin(["OCKM-T", "T-pca"])].groupby(["dataset", "method"]).seed_cost.mean().unstack()
put("SeedPcaWorse", int((SC2["T-pca"] > SC2["OCKM-T"]).sum()))
put("SeedPcaGeo", f"{100 * (np.exp(np.log(SC2['T-pca'] / SC2['OCKM-T']).mean()) - 1):.1f}")
# first version (results/t) quadric share and speed-up of the compiled version
v1 = pd.read_csv("../results/development/t_v1/runs.csv"); x1 = v1[v1.method == "OCKM-T"]
cu = {}
for e in x1.extra.apply(json.loads):
    for kk, v in e["cuts"].items(): cu[kk] = cu.get(kk, 0) + v
tot = sum(cu.values()); put("QuadricShare", f"{100*sum(cu.get(c,0) for c in ('E_ring','H_quad','H_sect','P_parab'))/tot:.1f}")
v2 = d[(d.method == "OCKM-T") & (d.seed == 0)].set_index("dataset"); x1 = x1.set_index("dataset")
put("VoneSpeed", f"{(x1.seed_time / v2.seed_time).median():.1f}")
bestv = pd.concat([pd.read_csv("../results/development/t_v1/runs.csv"), d]).groupby("dataset").WCSS.min()
g1 = 100 * (x1.WCSS / bestv.reindex(x1.index) - 1); g2 = 100 * (v2.WCSS / bestv.reindex(v2.index) - 1)
dd = (g2 - g1).dropna()
put("VoneWTL", f"{int((dd < -0.01).sum())}/{int((dd.abs() <= 0.01).sum())}/{int((dd > 0.01).sum())}")
put("VoneP", f"{stats.wilcoxon(g2.reindex(dd.index), g1.reindex(dd.index)).pvalue:.2f}")

# ---------------------------------------------------------------- tables
LAB = dict(S.LABEL); LAB.update({"OCKM-T": "\\textbf{\\ockmt{}}", "k-means||": "$k$-means$\\|$", "AFK-MC2": "AFK-MC$^2$",
          "KM++": "$k$-means++ (greedy)", "KM++v": "$k$-means++ (vanilla)", "OCKM": "OCKM"})
rows = []
for m in MAIN:
    g = s[s.method == m]
    tr = tratio(m).median()
    if m == "OCKM-T":
        rows.append(f"{LAB[m]} & \\textbf{{{rg[m]:.2f}}} & \\textbf{{{mg[m]:.2f}}} & \\textbf{{{100*hit[m]:.0f}}} & -- & -- & {tr:.2f} & -- \\\\")
        continue
    w, t_, l = W[m]
    eq = f"{eqp[m][0]}/{eqp[m][1]}/{eqp[m][2]}" if m in eqp else "--"
    rows.append(f"{LAB[m]} & {rg[m]:.2f} & {mg[m]:.2f} & {100*hit[m]:.0f} & {w}/{t_}/{l} & {sci(hp[m])} & {tr:.2f} & {eq} \\\\")
tab = ["\\begin{table*}[t]", "\\centering", "\\caption{Results on the \\Nds{} data sets at $k$ = number of classes. Rank: mean Friedman rank of the gap of the final cost (after one Lloyd run) to the best known cost (lower is better); gap: mean gap in \\%; $\\le0.1$\\%: share of runs within 0.1\\% of the best known; W/T/L: data sets on which \\ockmt{} is better / tied (within 0.01 points) / worse; Holm: Holm-corrected Wilcoxon $p$ against \\ockmt{}; time: median ratio of seeding plus Lloyd time to one greedy $k$-means++ run; equal time: \\ockmt{} against restarts of the method within the time of \\ockmt{} (W/T/L). Randomised methods: 10 runs.}", "\\label{tab:main}", "\\small",
       "\\begin{tabular}{lrrrrrrr}", "\\toprule",
       "Method & Rank & Gap & $\\le0.1$\\% & W/T/L & Holm & Time & Equal time \\\\", "\\midrule"] + rows + ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
open(os.path.join(OUT, "table_main.tex"), "w").write("\n".join(tab))

# ablation table
ta, ra = ranks("gap", ABL)
pa = {m: stats.wilcoxon(ta["OCKM-T"], ta[m]).pvalue for m in ABL if m != "OCKM-T"}
ha = dict(zip(pa, S.holm(list(pa.values()))))
tt = d[d.method.isin(ABL)].copy(); tt["phi"] = tt.extra.apply(lambda e: json.loads(e)["phi"])
PH = tt.groupby(["dataset", "method"]).phi.mean().unstack()
AL = {"OCKM-T": "\\ockmt{} (default)", "T-pca": "PCA-Part cuts only", "T-noLS": "no LocalSearch++", "T-S2": "2 summary starts",
      "T-tree": "tree only, no search", "T-D2": "no tree start", "T-g4": "$\\gamma=4$", "T-g16": "$\\gamma=16$",
      "T-noPolish": "no polish"}
rows = []
for m in ABL:
    phr = (PH[m] / PH["OCKM-T"]).median()
    tr = tratio(m).median()
    if m == "OCKM-T":
        rows.append(f"{AL[m]} & {ra[m]:.2f} & {mg[m]:.3f} & -- & -- & 1.00 & {tr:.2f} \\\\"); continue
    w, t_, l = wtl(ta, "OCKM-T", m)
    rows.append(f"{AL[m]} & {ra[m]:.2f} & {mg[m]:.3f} & {w}/{t_}/{l} & {sci(ha[m])} & {phr:.2f} & {tr:.2f} \\\\")
tab = ["\\begin{table}[t]", "\\centering", "\\caption{Ablation on the \\Nds{} data sets. W/T/L: the default is better / tied / worse than the variant; Holm: corrected Wilcoxon $p$; $\\Phi$: median ratio of $\\Phi$ to that of the default; the variant without tree start runs its search in uncompiled code; time: median time ratio to one greedy $k$-means++ run.}", "\\label{tab:ablation}", "\\small", "\\setlength{\\tabcolsep}{3pt}",
       "\\resizebox{\\columnwidth}{!}{%", "\\begin{tabular}{lrrrrrr}", "\\toprule", "Variant & Rank & Gap & W/T/L & Holm & $\\Phi$ & Time \\\\", "\\midrule"] + rows + ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
open(os.path.join(OUT, "table_ablation.tex"), "w").write("\n".join(tab))
ABLNAME = {"T-pca": "Pca", "T-noLS": "NoLS", "T-S2": "Stwo", "T-tree": "Tree", "T-D2": "Dtwo", "T-g4": "Gfour",
           "T-g16": "Gsixteen", "T-noPolish": "NoPolish"}
for m in ABL:
    if m != "OCKM-T": put("HolmAbl" + ABLNAME[m], f"{ha[m]:.2g}")
put("WTLnoLS", "/".join(map(str, wtl(ta, "OCKM-T", "T-noLS")))); put("WTLtree", "/".join(map(str, wtl(ta, "OCKM-T", "T-tree"))))
put("WTLpca", "/".join(map(str, wtl(ta, "OCKM-T", "T-pca")))); put("WTLStwo", "/".join(map(str, wtl(ta, "OCKM-T", "T-S2"))))



from scipy.stats import studentized_range
K = len(MAIN); N = tg.shape[0]
put("NemenyiCD", f"{studentized_range.ppf(0.95, K, np.inf) / np.sqrt(2) * np.sqrt(K * (K + 1) / (6.0 * N)):.2f}")
put("RankDiffOCKM", f"{rg['OCKM'] - rg['OCKM-T']:.2f}"); put("RankDiffPCA", f"{rg['PCA-Part'] - rg['OCKM-T']:.2f}")
cdv = studentized_range.ppf(0.95, K, np.inf) / np.sqrt(2) * np.sqrt(K * (K + 1) / (6.0 * N))
put("NemOCKM", "significant" if rg['OCKM'] - rg['OCKM-T'] > cdv else "not significant")
put("NemPCA", "significant" if rg['PCA-Part'] - rg['OCKM-T'] > cdv else "not significant")
qk = d[d.method == "Qk"].copy(); qk["fb"] = qk.extra.apply(lambda e: json.loads(e).get("fallbacks", 0))
qd = qk.groupby("dataset").apply(lambda g: g.fb.sum() / g.k.sum())
put("QkFallback", f"{100 * qd.median():.0f}")
put("QkFallbackPooled", f"{100 * qk.fb.sum() / qk.k.sum():.0f}"); put("QkFallbackNten", int((qd >= 0.10).sum()))
put("QkFallbackMax", f"{100 * qd.max():.0f}"); put("QkFallbackMaxDs", str(qd.idxmax()))
behind = (tg["OCKM-T"] - tg.min(axis=1))
bh = behind[behind > 0.1].sort_values(ascending=False)
put("NBehind", len(bh)); put("BehindList", ", ".join(f"{ds} ({v:.2f})" for ds, v in bh.items()))
rkM = tg.rank(axis=1).loc["MNIST"]; put("RankMNIST", f"{rkM['OCKM-T']:.0f}"); put("GapMNIST", f"{tg.loc['MNIST','OCKM-T']:.2f}")
rkF = tg.rank(axis=1).loc["FMNIST"]; put("RankFMNIST", f"{rkF['OCKM-T']:.0f}")

# extra macros for the experiments text
put("PhiPcaGainInv", f"{100*(np.exp(np.log(r).mean())-1):.0f}")
put("TimeGsixteen", f"{tratio('T-g16').median():.2f}"); put("TimeRatioOCKM", f"{tratio('OCKM').median():.1f}")
put("WTLgsixteen", "/".join(map(str, wtl(ta, "OCKM-T", "T-g16"))))
put("TenRunsBudget", f"{10/tratio('OCKM-T').median():.0f}")
g_ = et[et.ref == "KM++"]
put("EqKMppAll", f"{int((g_.gap_all < -0.01).sum())}/{int((g_.gap_all.abs() <= 0.01).sum())}/{int((g_.gap_all > 0.01).sum())}")
ta_, ra_ = ranks("ARI", MAIN, higher=True)
pa_ = {m: stats.wilcoxon(ta_["OCKM-T"], ta_[m]).pvalue for m in MAIN if m != "OCKM-T"}
put("ARIHolmMin", f"{min(S.holm(list(pa_.values()))):.2f}"); put("ARIRankT", f"{ra_['OCKM-T']:.2f}")
put("ARIRankBest", f"{ra_.min():.2f}"); put("ARIBest", str(ra_.idxmin()))

# ---------------------------------------------------------------- appendix tables
order = sorted(s.dataset.unique(), key=lambda x: (x not in S.SYNTH, x.lower()))
cols = MAIN
short = {"OCKM-T": "\\ockmt{}", "KM++": "KM++", "KM++v": "KM++v", "Qk": "Qk", "AFK-MC2": "AFK", "k-means||": "$k$m$\\|$",
         "PCA-Part": "PCA", "Var-Part": "Var", "CCKM": "CCKM", "OCKM": "OCKM", "OCKM-fast": "OCKMf"}
rows = []
for ds in order:
    g = s[s.dataset == ds].set_index("method"); r0 = g.iloc[0]
    bestv = min(g.gap[m] for m in cols)
    cells = []
    for m in cols:
        v = g.gap[m]; txt = f"{v:.2f}" if v < 100 else f"{v:.0f}"
        cells.append(f"\\textbf{{{txt}}}" if v - bestv <= 0.01 else txt)
    rows.append(f"{ds} & {int(r0.n)} & {int(r0.d)} & {int(r0.k)} & " + " & ".join(cells) + " \\\\")
    if ds == [x for x in order if x in S.SYNTH][-1]:
        rows.append("\\midrule")
tab = ["\\begin{table}[h]", "\\centering", "\\caption{Gap (\\%) of the final cost to the best known cost per data set (randomised methods, including \\ockmt{}: mean of ten runs). Synthetic sets first. Bold: within 0.01 points of the best mean.}", "\\label{tab:perdataset}", "\\scriptsize", "\\setlength{\\tabcolsep}{2.5pt}",
       "\\begin{tabular}{lrrr" + "r" * len(cols) + "}", "\\toprule", "Data set & $n$ & $d$ & $k$ & " + " & ".join(short[m] for m in cols) + " \\\\", "\\midrule"] + rows + ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
# ARI and best-of-ten table
ta_, ra_ = ranks("ARI", MAIN, higher=True)
hpa = dict(zip(pa_, S.holm(list(pa_.values()))))
ari_mean = s.groupby("method").ARI.mean()
rows2 = []
for m in MAIN:
    ref = et[et.ref == m]
    b10 = f"{int((ref.gap_all < -0.01).sum())}/{int((ref.gap_all.abs() <= 0.01).sum())}/{int((ref.gap_all > 0.01).sum())}" if len(ref) else "--"
    rows2.append(f"{LAB[m]} & {ra_[m]:.2f} & {ari_mean[m]:.3f} & " + ("--" if m == "OCKM-T" else sci(hpa[m])) + f" & {b10} \\\\")
tab += ["", "\\begin{table}[h]", "\\centering", "\\caption{Adjusted Rand index on the 38 labelled data sets (mean Friedman rank, higher ARI is better; mean ARI; Holm-corrected Wilcoxon $p$ against \\ockmt{}). Last column: final $k$-means cost of \\ockmt{} against the best of all ten runs of each randomised method (better/tied/worse).}", "\\label{tab:ari}", "\\small",
        "\\begin{tabular}{lrrrr}", "\\toprule", "Method & ARI rank & mean ARI & Holm & vs best of 10 \\\\", "\\midrule"] + rows2 + ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
# first version vs compiled version
open(os.path.join(OUT, "table_perdataset.tex"), "w").write("\n".join(tab))

# OCKM-T against the best of r runs of greedy k-means++ and Qkmeans (budget-free view)
rows_b = []
for ref in ("KM++", "Qk"):
    for rr in (1, 2, 3, 5, 10):
        res_ = []; tr_ = []
        for ds, g in d.groupby("dataset"):
            T_ = g[g.method == "OCKM-T"]; q_ = g[g.method == ref].sort_values("seed"); w_ = q_.WCSS.to_numpy(); nq = len(w_)
            bestr = np.mean([min(w_[(s0 + j) % nq] for j in range(rr)) for s0 in range(nq)])
            bk = g.WCSS.min()
            res_.append(100 * (T_.WCSS.mean() / bk - 1) - 100 * (bestr / bk - 1))
            tr_.append(rr * q_.total_time.mean() / T_.total_time.mean())
        res_ = np.array(res_)
        rows_b.append(f"{LAB[ref]} & {rr} & {np.median(tr_):.1f} & {int((res_ < -0.01).sum())}/{int((np.abs(res_) <= 0.01).sum())}/{int((res_ > 0.01).sum())} \\\\")
        if ref == "KM++":
            put("BestOf" + {1: "One", 2: "Two", 3: "Three", 5: "Five", 10: "Ten"}[rr], f"{int((res_ < -0.01).sum())}/{int((np.abs(res_) <= 0.01).sum())}/{int((res_ > 0.01).sum())}")
            if rr == 2:
                put("BestOfTwoWins", int((res_ < -0.01).sum())); put("BestOfTwoLosses", int((res_ > 0.01).sum()))
tabb = ["\\begin{table}[h]", "\\centering", "\\caption{\\ockmt{} (mean of ten runs) against the expected best of $r$ runs of greedy $k$-means++ and of Qkmeans, without a time budget. Time: median ratio of the time of $r$ runs to that of \\ockmt{}. W/T/L: \\ockmt{} better/tied/worse.}", "\\label{tab:bestof}", "\\small",
        "\\begin{tabular}{lrrr}", "\\toprule", "Method & $r$ & Time & W/T/L \\\\", "\\midrule"] + rows_b + ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
open(os.path.join(OUT, "table_bestof.tex"), "w").write("\n".join(tabb))

# large-k grid (optional, when present)
g = os.path.join(RES, "seed_grid.csv")
if os.path.exists(g):
    G = pd.read_csv(g).groupby(["dataset", "k", "method"])[["seed_time", "seed_cost"]].mean().reset_index()
    if ((G.dataset == "MNIST") & (G.k == 1000) & (G.method == "KM++")).any():
        a = G[(G.dataset == "MNIST") & (G.k == 1000)].set_index("method")
        put("SpeedupK", f"{a.seed_time['KM++'] / a.seed_time['OCKM-T']:.1f}")
        put("TimeKT", f"{a.seed_time['OCKM-T']:.1f}"); put("TimeKKM", f"{a.seed_time['KM++']:.0f}"); put("TimeKQk", f"{a.seed_time['Qk']:.2f}")
        put("CostKQkOverT", f"{a.seed_cost['Qk'] / a.seed_cost['OCKM-T']:.2f}"); put("CostKKMOverT", f"{a.seed_cost['KM++'] / a.seed_cost['OCKM-T']:.2f}")
        put("CostKPcaOverT", f"{a.seed_cost['PCA-Part'] / a.seed_cost['OCKM-T']:.3f}")
        GT = G.pivot_table(index=["dataset", "k"], columns="method", values="seed_time")
        GC = G.pivot_table(index=["dataset", "k"], columns="method", values="seed_cost")
        mets = [m for m in ["OCKM-T", "T-pca", "PCA-Part", "KM++", "KM++v", "Qk", "AFK-MC2", "k-means||"] if m in GT.columns]
        rows = []
        for (ds, kk), r in GT.iterrows():
            ref = GC.loc[(ds, kk), "OCKM-T"]
            cells = [f"{r[m]:.2f} / {GC.loc[(ds, kk), m] / ref:.3f}" if not np.isnan(r[m]) else "--" for m in mets]
            rows.append(f"{ds} & {kk} & " + " & ".join(cells) + " \\\\")
        LK = {"OCKM-T": "\\ockmt{}", "T-pca": "PCA cuts", "PCA-Part": "PCA-Part", "KM++": "KM++", "KM++v": "KM++v", "Qk": "Qkmeans", "AFK-MC2": "AFK-MC$^2$", "k-means||": "$k$-means$\\|$"}
        tab = ["\\begin{table}[h]", "\\centering", "\\caption{Seeding time (s) / seeding cost relative to \\ockmt{} at large $k$ (randomised methods: mean of three runs).}", "\\label{tab:largek}", "\\scriptsize", "\\setlength{\\tabcolsep}{2pt}",
               "\\begin{tabular}{lr" + "r" * len(mets) + "}", "\\toprule", "Data & $k$ & " + " & ".join(LK[m] for m in mets) + " \\\\", "\\midrule"] + rows + ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
        open(os.path.join(OUT, "table_largek.tex"), "w").write("\n".join(tab))
        lt = G[(G.dataset == "letter") & (G.k == 1000)].set_index("method")
        put("LetterPcaK", f"{lt.seed_cost['PCA-Part'] / lt.seed_cost['OCKM-T']:.2f}")
        put("PcaLargeKGain", f"{100*(1 - a.seed_cost['PCA-Part'] / a.seed_cost['OCKM-T']):.1f}")

assert all(k.isalpha() for k in M), [k for k in M if not k.isalpha()]
# anytime traces at k = 1000
tp = os.path.join(RES, "trace.csv")
if os.path.exists(tp):
    T = pd.read_csv(tp)
    for ds, tag in (("MNIST", "MNIST"), ("FMNIST", "FMNIST")):
        g = T[T.dataset == ds]; cur = {m: x.sort_values("iteration") for m, x in g.groupby("method")}
        o, q = cur["OCKM-T"], cur["Qk"]
        put("TraceQk" + tag, f"{100 * (1 - o.cost.iloc[-1] / q.cost.iloc[-1]):.1f}")
        # first time from which OCKM-T's best-so-far cost stays below that of Qkmeans (both step curves)
        def best_at(c, t):
            v = c[c.time <= t].cost; return v.min() if len(v) else np.inf
        times = np.sort(np.concatenate([o.time.to_numpy(), q.time.to_numpy()]))
        below = [(t, best_at(o, t) < best_at(q, t)) for t in times]
        cross = None
        for i, (t, b) in enumerate(below):
            if all(bb for _, bb in below[i:]):
                cross = t; break
        put("TraceCross" + tag, f"{cross:.0f}" if cross is not None else "never")
        put("TraceTpca" + tag, f"{100 * (1 - cur['T-pca'].cost.iloc[-1] / o.cost.iloc[-1]):.2f}")
        put("TraceKMTime" + tag, f"{cur['KM++'].time.iloc[-1] / o.time.iloc[-1]:.1f}")
        put("TraceKMBelow" + tag, "yes" if cur['KM++'].cost.iloc[-1] < o.cost.iloc[-1] else "no")

lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()]
open(os.path.join(OUT, "numbers.tex"), "w").write("% generated by ockm/make_numbers.py; do not edit\n" + "\n".join(lines) + "\n")
print("\n".join(lines))
