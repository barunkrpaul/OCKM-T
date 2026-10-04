import pandas as pd, numpy as np, sys
from scipy.stats import wilcoxon
d = pd.read_csv(sys.argv[1]); ref = sys.argv[2] if len(sys.argv) > 2 else "A"
best = d.groupby(["dataset", "rep"]).WCSS.transform("min")
d["gap"] = 100 * (d.WCSS / best - 1); d["tt"] = d.seed_time + d.lloyd_time
G = d.groupby(["dataset", "variant"]).gap.mean().unstack()
T = d.groupby(["dataset", "variant"]).tt.mean().unstack()
print("mean gap %:", G.mean().round(3).to_dict())
print("mean rank :", G.round(4).rank(axis=1).mean().round(2).to_dict())
for c in G.columns:
    if c == ref: continue
    diff = (G[ref] - G[c]).round(4); b, l = (diff < -0.01).sum(), (diff > 0.01).sum()
    p = wilcoxon(G[ref], G[c]).pvalue if (diff != 0).any() else 1
    print(f"{ref} vs {c}: better {b} tie {len(diff)-b-l} worse {l} p={p:.3g}")
if "KM++" in T: print("median time ratio vs KM++:", T.div(T["KM++"], axis=0).median().round(2).to_dict())
