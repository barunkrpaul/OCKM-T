"""Analysis of bench_t.py output: quality against the best-known solution, ranks,
statistical tests, equal-time comparison with restarts, seeding cost/time, ablations.

usage: python summarise_t.py RESULTS_DIR   (reads runs.csv, writes t_tables.md, t_stats.md,
       t_summary.csv, t_equal_time.csv, fig_t_gap.(png|pdf), fig_t_pareto.(png|pdf))
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import studentized_range
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAIN = ["OCKM-T", "KM++", "KM++v", "Qk", "AFK-MC2", "k-means||", "PCA-Part", "Var-Part", "CCKM", "OCKM", "OCKM-fast"]
ABL = ["OCKM-T", "T-pca", "T-noLS", "T-S2", "T-tree", "T-D2", "T-g4", "T-g16", "T-noPolish"]
RANDOM = {"KM++", "KM++v", "Qk", "AFK-MC2", "k-means||"}
LABEL = {"OCKM-T": "OCKM-T", "KM++": "k-means++ (greedy)", "KM++v": "k-means++ (vanilla)", "Qk": "Qkmeans",
         "AFK-MC2": "AFK-MC²", "k-means||": "k-means‖", "PCA-Part": "PCA-Part", "Var-Part": "Var-Part", "CCKM": "CCKM",
         "OCKM": "OCKM (published)", "OCKM-fast": "OCKM-fast", "T-pca": "T: PCA cuts only", "T-noLS": "T: no LocalSearch++",
         "T-g4": "T: gamma = 4", "T-g16": "T: gamma = 16", "T-noPolish": "T: no full-space polish",
         "T-S2": "T: 2 summary starts", "T-tree": "T: tree partition only (no summary search)",
         "T-D2": "T: summary search without tree start"}
SYNTH = {"2d-10c", "D31", "R15", "aggregation", "atom", "birch-rg1", "birch-rg3", "chainlink", "compound", "engytime", "flame",
         "fourty", "hepta", "jain", "pathbased", "s-set1", "s-set2", "s-set3", "s-set4", "tetra", "twenty"}
TIE = 1e-4          # relative WCSS tolerance for ties


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def per_dataset(d):
    best = d.groupby("dataset").WCSS.min()
    d = d.assign(gap=100.0 * (d.WCSS / d.dataset.map(best) - 1.0))
    kv = d[d.method == "KM++v"].groupby("dataset").seed_cost.mean()
    d = d.assign(seed_ratio=d.seed_cost / d.dataset.map(kv))
    g = d.groupby(["dataset", "method"])
    s = g.agg(gap=("gap", "mean"), gap_best=("gap", "min"), hit01=("gap", lambda x: float((x <= 0.1).mean())),
              hit1=("gap", lambda x: float((x <= 1.0).mean())), ARI=("ARI", "mean"), NMI=("NMI", "mean"),
              seed_ratio=("seed_ratio", "mean"), seed_time=("seed_time", "mean"), total_time=("total_time", "mean"),
              IT=("IT", "mean"), WCSS=("WCSS", "mean"), runs=("seed", "count"), n=("n", "first"), d=("d", "first"),
              k=("k", "first")).reset_index()
    return d, s


def friedman_table(s, methods, metric, higher, target, out, title):
    t = s.pivot_table(index="dataset", columns="method", values=metric)[methods].dropna()
    N, K = len(t), len(methods)
    ranks = t.rank(axis=1, ascending=not higher, method="average"); mr = ranks.mean()
    chi2, p = stats.friedmanchisquare(*[t[m].values for m in methods])
    q = studentized_range.ppf(0.95, K, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(K * (K + 1) / (6.0 * N))
    out.append(f"\n### {title}\n")
    out.append(f"Friedman chi2 = {chi2:.1f}, p = {p:.1e}; N = {N} datasets, {K} methods; Nemenyi CD (alpha = 0.05) = {cd:.2f}.\n")
    out.append(f"| method | mean rank | Wilcoxon p vs {LABEL[target]} | Holm | {LABEL[target]} wins / ties / losses |")
    out.append("|---|---|---|---|---|")
    rows, pv = [], []
    for m in methods:
        if m == target:
            continue
        x, y = t[target].values, t[m].values
        diff = (x - y) if higher else (y - x)
        tol = TIE * 100 if metric == "gap" else TIE * np.abs(y).mean()
        wi, lo = int((diff > tol).sum()), int((diff < -tol).sum())
        try:
            w = stats.wilcoxon(x, y, zero_method="wilcox").pvalue if np.any(x != y) else 1.0
        except ValueError:
            w = 1.0
        pv.append(w); rows.append((m, mr[m], w, wi, N - wi - lo, lo))
    for (m, r, w, wi, ti, lo), h in zip(rows, holm(pv)):
        out.append(f"| {LABEL[m]} | {r:.2f} | {w:.1e} | {h:.1e} | {wi} / {ti} / {lo} |")
    out.append(f"| **{LABEL[target]}** | **{mr[target]:.2f}** | | | |")
    return mr


def equal_time(d, target="OCKM-T", refs=("KM++", "Qk", "k-means||", "KM++v", "AFK-MC2")):
    """For every run of the target (seed s), its total time is the budget; the reference is restarted in
    seed order starting at seed s (cyclically) while its cumulative total time fits (at least one run),
    and its best final cost is kept. Gaps are averaged over the target seeds."""
    rows = []
    for ds, g in d.groupby("dataset"):
        T = g[g.method == target].sort_values("seed")
        if not len(T):
            continue
        for ref in refs:
            q = g[g.method == ref].sort_values("seed")
            if not len(q):
                continue
            nq = len(q); gaps, gaps_all, runs, ari = [], [], [], []
            for t in T.itertuples():
                qq = q.iloc[[(int(t.seed) + j) % nq for j in range(nq)]]
                ct = qq.total_time.cumsum().to_numpy(); w = qq.WCSS.to_numpy()
                nr = max(1, int((ct <= t.total_time).sum())); b = int(np.argmin(w[:nr]))
                gaps.append(100.0 * (t.WCSS / w[b] - 1.0)); gaps_all.append(100.0 * (t.WCSS / w.min() - 1.0))
                runs.append(nr); ari.append(qq.ARI.to_numpy()[b])
            rows.append(dict(dataset=ds, ref=ref, budget=T.total_time.mean(), runs=float(np.mean(runs)),
                             target_WCSS=T.WCSS.mean(), gap=float(np.mean(gaps)), gap_all=float(np.mean(gaps_all)),
                             target_ARI=T.ARI.mean(), ref_ARI=float(np.mean(ari))))
    return pd.DataFrame(rows)


def main(res):
    d = pd.read_csv(os.path.join(res, "runs.csv"))
    d, s = per_dataset(d)
    s.to_csv(os.path.join(res, "t_summary.csv"), index=False)
    out = ["# OCKM-T benchmark: statistics", "",
           f"{s.dataset.nunique()} datasets. Quality = gap of the final WCSS (after one Lloyd run) to the best WCSS found by any",
           "run of any method on that dataset, in per cent (0 = best known). Randomised methods (including OCKM-T) enter with their mean over 10 runs",
           "(the expected single-run gap). Ties: |difference| <= 0.01 percentage points of gap.", ""]
    meths = [m for m in MAIN if m in set(s.method)]
    friedman_table(s, meths, "gap", False, "OCKM-T", out, "Final WCSS gap to best known, all datasets")
    for label, sel in (("synthetic (2-D and 3-D benchmark shapes)", s.dataset.isin(SYNTH)), ("real (tabular and image)", ~s.dataset.isin(SYNTH))):
        ss = s[sel]
        if ss.dataset.nunique() >= 5:
            friedman_table(ss, meths, "gap", False, "OCKM-T", out, f"Final WCSS gap, {label}")
    lab = s[s.ARI.notna()]
    friedman_table(lab, meths, "ARI", True, "OCKM-T", out, "ARI at k = number of classes (labelled datasets)")
    friedman_table(s, meths, "seed_ratio", False, "OCKM-T", out, "Seeding cost relative to vanilla k-means++")
    friedman_table(s, meths, "total_time", False, "OCKM-T", out, "Total time (seeding + Lloyd)")
    abl = [m for m in ABL if m in set(s.method)]
    if len(abl) > 2:
        friedman_table(s, abl, "gap", False, "OCKM-T", out, "Ablation of OCKM-T, final WCSS gap")
    et = equal_time(d)
    et.to_csv(os.path.join(res, "t_equal_time.csv"), index=False)
    out.append("\n## Equal-time comparison\n")
    out.append("Budget = total time of an OCKM-T run (seed s). Each randomised method is restarted in seed order from seed s while its")
    out.append("cumulative total time fits the budget (at least one run) and its best WCSS is kept; gaps averaged over s. Gap < 0: OCKM-T better.\n")
    out.append("| reference | datasets | OCKM-T better / tie / worse | median gap % | mean runs in budget | Wilcoxon p (gap vs 0) | also vs best of all 10 runs: better / tie / worse |")
    out.append("|---|---|---|---|---|---|---|")
    for ref, g in et.groupby("ref", sort=False):
        b = int((g.gap < -0.01).sum()); w_ = int((g.gap > 0.01).sum()); t_ = len(g) - b - w_
        b2 = int((g.gap_all < -0.01).sum()); w2 = int((g.gap_all > 0.01).sum()); t2 = len(g) - b2 - w2
        try:
            pw = stats.wilcoxon(g.gap.values).pvalue if np.any(g.gap.values != 0) else 1.0
        except ValueError:
            pw = 1.0
        out.append(f"| {LABEL[ref]} | {len(g)} | {b} / {t_} / {w_} | {g.gap.median():+.3f} | {g.runs.mean():.1f} | {pw:.1e} | {b2} / {t2} / {w2} |")
    # time ratios against one k-means++ run (seeding + Lloyd)
    tk = s[s.method == "KM++"].set_index("dataset")
    out.append("\n## Time relative to one k-means++ (greedy) run\n")
    out.append("| method | median total-time ratio | median seeding-time ratio | datasets with total ratio <= 1 |")
    out.append("|---|---|---|---|")
    for m in meths + [a for a in abl if a != "OCKM-T"]:
        g = s[s.method == m].set_index("dataset")
        rt = (g.total_time / tk.total_time.reindex(g.index)); rs = (g.seed_time / tk.seed_time.reindex(g.index))
        out.append(f"| {LABEL[m]} | {rt.median():.2f} | {rs.median():.2f} | {int((rt <= 1).sum())} / {len(rt)} |")
    open(os.path.join(res, "t_stats.md"), "w").write("\n".join(out))

    # per-dataset table
    tab = ["# OCKM-T benchmark: per-dataset results", "",
           "Gap of the final WCSS to the best known (per cent; randomised methods: mean over 10 runs, best run in brackets).", ""]
    cols = meths
    tab.append("| dataset | n | d | k | " + " | ".join(LABEL[m] for m in cols) + " |")
    tab.append("|---|---|---|---|" + "---|" * len(cols))
    for ds, g in s.groupby("dataset"):
        g = g.set_index("method")
        cells = []
        best_mean = min(g.gap.get(m, np.inf) for m in cols)
        for m in cols:
            if m not in g.index:
                cells.append(""); continue
            v = f"{g.gap[m]:.2f}"
            if m in RANDOM:
                v += f" ({g.gap_best[m]:.2f})"
            if abs(g.gap[m] - best_mean) < 1e-9:
                v = f"**{v}**"
            cells.append(v)
        r = g.iloc[0]
        tab.append(f"| {ds} | {int(r.n)} | {int(r.d)} | {int(r.k)} | " + " | ".join(cells) + " |")
    tab.append("\nMean over datasets:\n")
    tab.append("| method | mean gap % | median gap % | datasets within 0.1% of best known (share of runs) | mean ARI | mean seed cost / KM++v | mean seed time s | mean total time s |")
    tab.append("|---|---|---|---|---|---|---|---|")
    for m in meths + [a for a in abl if a != "OCKM-T"]:
        g = s[s.method == m]
        tab.append(f"| {LABEL[m]} | {g.gap.mean():.3f} | {g.gap.median():.3f} | {g.hit01.mean():.2f} | {g.ARI.mean():.3f} | "
                   f"{g.seed_ratio.mean():.3f} | {g.seed_time.mean():.3f} | {g.total_time.mean():.2f} |")
    open(os.path.join(res, "t_tables.md"), "w").write("\n".join(tab))
    figures(s, res, meths)
    print("written", res)


def figures(s, res, meths):
    order = s[s.method == "OCKM-T"].sort_values("gap").dataset.tolist()
    show = ["OCKM-T", "KM++", "Qk", "PCA-Part", "OCKM"]
    col = {"OCKM-T": "#1baf7a", "KM++": "#e87ba4", "Qk": "#2a78d6", "PCA-Part": "#008300", "OCKM": "#eb6834"}
    mk = {"OCKM-T": "^", "KM++": "v", "Qk": "o", "PCA-Part": "P", "OCKM": "s"}
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = np.arange(len(order))
    for m in show:
        g = s[s.method == m].set_index("dataset").reindex(order)
        ax.plot(x, np.maximum(g.gap.values, 1e-3), ls="none", marker=mk[m], ms=6, color=col[m], label=LABEL[m],
                markeredgecolor="white", markeredgewidth=0.6)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(order, rotation=70, fontsize=7)
    ax.set_ylabel("gap to best-known WCSS (%)", color="#52514e")
    ax.grid(True, axis="y", color="#e4e3df", lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper left")
    ax.set_title("Final WCSS after one Lloyd run (randomised methods: mean of 10 runs; gaps below 0.001% drawn at 0.001%)", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(res, f"fig_t_gap.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    agg = s.groupby("method").agg(gap=("gap", "mean"), t=("total_time", "mean")).reindex(meths)
    for m, r in agg.iterrows():
        c = col.get(m, "#8a8985")
        ax.plot(r.t, r.gap, marker=mk.get(m, "o"), ms=8, color=c, ls="none", markeredgecolor="white")
        ax.annotate(LABEL[m], (r.t, r.gap), textcoords="offset points", xytext=(5, 3), fontsize=7, color="#52514e")
    ax.set_xscale("log"); ax.set_xlabel("mean total time per run (s)", color="#52514e"); ax.set_ylabel("mean gap to best known (%)", color="#52514e")
    ax.grid(True, color="#e4e3df", lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(res, f"fig_t_pareto.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "development"))
