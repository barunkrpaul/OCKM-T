"""Independent re-computation of the numbers quoted in the manuscript (numbers.tex).
Uses only pandas/scipy on results/development/runs.csv, t_equal_time is recomputed from runs.csv."""
import re, json, numpy as np, pandas as pd
from scipy import stats

M = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", open("../paper_stats/out/numbers.tex").read()))
d = pd.read_csv("../results/development/runs.csv")
MAIN = ["OCKM-T", "KM++", "KM++v", "Qk", "AFK-MC2", "k-means||", "PCA-Part", "Var-Part", "CCKM", "OCKM", "OCKM-fast"]
best = d.groupby("dataset").WCSS.min()
d["gap"] = 100 * (d.WCSS / d.dataset.map(best) - 1)
g = d.groupby(["dataset", "method"]).gap.mean().unstack()
checks = []
def chk(name, got, tol=0.006):
    exp = M[name]
    try:
        ok = abs(float(exp) - float(got)) <= tol * max(1, abs(float(got)))
    except ValueError:
        ok = str(exp) == str(got)
    checks.append((name, exp, got, ok))

rk = g[MAIN].rank(axis=1).mean()
chk("RankT", round(rk["OCKM-T"], 2)); chk("RankKMpp", round(rk["KM++"], 2)); chk("RankOCKM", round(rk["OCKM"], 2))
chk("GapT", round(g["OCKM-T"].mean(), 2), 0.01); chk("GapKMpp", round(g["KM++"].mean(), 1), 0.05)
hit = d.assign(h=d.gap <= 0.1).groupby(["dataset", "method"]).h.mean().unstack().mean()
chk("HitT", round(100 * hit["OCKM-T"]), 0.5); chk("HitKMpp", round(100 * hit["KM++"]), 0.5)
def wtl(a, b, t=g):
    x = t[a] - t[b]; return f"{int((x < -0.01).sum())}/{int((x.abs() <= 0.01).sum())}/{int((x > 0.01).sum())}"
chk("WTLKMpp", wtl("OCKM-T", "KM++")); chk("WTLOCKM", wtl("OCKM-T", "OCKM"))
# seeding cost best
kv = d[d.method == "KM++v"].groupby("dataset").seed_cost.mean()
sc = d.assign(r=d.seed_cost / d.dataset.map(kv)).groupby(["dataset", "method"]).r.mean().unstack()[MAIN]
chk("SeedBestT", int((sc["OCKM-T"] <= sc.min(axis=1) * (1 + 1e-12)).sum()))
# time ratios
tt = d.groupby(["dataset", "method"]).total_time.mean().unstack()
chk("TimeRatioT", round((tt["OCKM-T"] / tt["KM++"]).median(), 2), 0.01)
st = d.groupby(["dataset", "method"]).seed_time.mean().unstack()
chk("SeedTimeRatioT", round((st["OCKM-T"] / st["KM++"]).median(), 2), 0.01)
# equal time vs KM++ (recomputed; per OCKM-T seed, reference restarts from the same seed cyclically)
res = []
for ds, gg in d.groupby("dataset"):
    T = gg[gg.method == "OCKM-T"].sort_values("seed"); q = gg[gg.method == "KM++"].sort_values("seed")
    w = q.WCSS.to_numpy(); tt_ = q.total_time.to_numpy(); n = len(w); gaps = []
    for t in T.itertuples():
        order = [(int(t.seed) + j) % n for j in range(n)]
        cum = np.cumsum(tt_[order]); nr = max(1, int((cum <= t.total_time).sum()))
        gaps.append(100 * (t.WCSS / w[order][:nr].min() - 1))
    res.append(np.mean(gaps))
res = np.array(res)
chk("EqKMpp", f"{int((res < -0.01).sum())}/{int((np.abs(res) <= 0.01).sum())}/{int((res > 0.01).sum())}")
# phi ratio
o = d[d.method == "OCKM-T"].copy(); o["phi"] = o.extra.apply(lambda e: json.loads(e)["phi_full"])
oo = o.groupby("dataset")[["phi", "WCSS"]].mean(); q = oo.phi / oo.WCSS
chk("PhiRatioMed", round(q.median(), 2), 0.01); chk("PhiRatioMax", round(q.max(), 2), 0.01)
# ablations
A = d[d.method.str.startswith("T-") | (d.method == "OCKM-T")].groupby(["dataset", "method"]).gap.mean().unstack()
chk("WTLtree", wtl("OCKM-T", "T-tree", A)); chk("WTLnoLS", wtl("OCKM-T", "T-noLS", A)); chk("WTLpca", wtl("OCKM-T", "T-pca", A))
for name, exp, got, ok in checks:
    print(f"{'OK ' if ok else 'BAD'} {name:16s} paper={exp:>10s} recomputed={got}")
print("all ok" if all(c[3] for c in checks) else "MISMATCH")
