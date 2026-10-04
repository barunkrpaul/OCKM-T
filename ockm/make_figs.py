"""Figures for the manuscript (written to paper/ockmt_icml/figs)."""
import os, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, kmeans_plusplus
import ockmt, summarise_t3 as S
from bench_data import load_any

OUT = "../paper_stats/out/figs"; RES = "../results/development"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
COL = {"OCKM-T": "#1b9e77", "T-pca": "#66a61e", "KM++": "#d95f02", "KM++v": "#e6ab02", "Qk": "#7570b3",
       "AFK-MC2": "#e7298a", "k-means||": "#a6761d", "PCA-Part": "#666666", "Var-Part": "#999999", "CCKM": "#bbbbbb",
       "OCKM": "#1f78b4", "OCKM-fast": "#a6cee3"}
LAB = {"OCKM-T": "OCKM-T", "T-pca": "OCKM-T (PCA cuts)", "KM++": "k-means++ (greedy)", "KM++v": "k-means++ (vanilla)",
       "Qk": "Qkmeans", "AFK-MC2": "AFK-MC²", "k-means||": "k-means‖", "PCA-Part": "PCA-Part", "Var-Part": "Var-Part",
       "CCKM": "CCKM", "OCKM": "OCKM", "OCKM-fast": "OCKM-fast"}


def fig_pareto():
    d = pd.read_csv(os.path.join(RES, "runs.csv")); d, s = S.per_dataset(d)
    t = s.pivot_table(index="dataset", columns="method", values="gap")[S.MAIN]
    rk = t.rank(axis=1).mean()
    tk = s[s.method == "KM++"].set_index("dataset").total_time
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    for m in S.MAIN:
        g = s[s.method == m].set_index("dataset")
        tr = (g.total_time / tk.reindex(g.index)).median()
        ax.scatter(tr, rk[m], s=28 if m == "OCKM-T" else 16, color=COL[m], zorder=3, marker="^" if m == "OCKM-T" else "o")
        off = {"KM++v": (0, 6, "center"), "AFK-MC2": (5, -1, "left"), "CCKM": (5, -9, "left"), "Var-Part": (4, -9, "left"),
               "Qk": (0, -11, "center"), "KM++": (-5, 3, "right")}.get(m, (4, 2, "left"))
        ax.annotate(LAB[m], (tr, rk[m]), textcoords="offset points", xytext=off[:2], ha=off[2], fontsize=6.5, color="#333333")
    ax.set_xscale("log"); ax.set_xlabel("median time / one greedy k-means++ run"); ax.set_ylabel("mean rank of final cost")
    ax.set_xticks([0.7, 1, 1.5, 2, 3]); ax.set_xticklabels(["0.7", "1", "1.5", "2", "3"]); ax.minorticks_off()
    ax.set_xlim(0.6, 3.6); ax.set_ylim(8.4, 2.1); ax.grid(alpha=0.3, lw=0.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_pareto.pdf")); fig.savefig(os.path.join(OUT, "fig_pareto.png"), dpi=200)


def fig_largek():
    G = pd.read_csv(os.path.join(RES, "seed_grid.csv"))
    T = pd.read_csv(os.path.join(RES, "trace.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.35))
    ax = axes[0]
    g = G[G.dataset == "MNIST"].groupby(["k", "method"])[["seed_time", "seed_cost"]].mean().reset_index()
    ref = g[g.method == "OCKM-T"].set_index("k").seed_cost
    handles = []
    for m in ["OCKM-T", "T-pca", "PCA-Part", "KM++", "KM++v", "Qk", "AFK-MC2", "k-means||"]:
        x = g[g.method == m].sort_values("k")
        h, = ax.plot(x.seed_time, x.seed_cost.values / ref.reindex(x.k).values, "-o", ms=2.5, lw=1, color=COL[m],
                     ls="--" if m == "T-pca" else "-", label=LAB[m])
        handles.append(h)
    for _, r in g[g.method == "KM++"].iterrows():
        ax.annotate(f"k={int(r.k)}", (r.seed_time, r.seed_cost / ref[r.k]), textcoords="offset points", xytext=(0, 5), fontsize=6, ha="center")
    ax.set_xscale("log"); ax.set_xlabel("seeding time (s), MNIST"); ax.set_ylabel("seeding cost / OCKM-T")
    ax.set_ylim(0.95, 1.95); ax.grid(alpha=0.3, lw=0.5)
    for ax, ds in zip(axes[1:], ["MNIST", "FMNIST"]):
        t = T[T.dataset == ds]; best = t.cost.min()
        for m in ["OCKM-T", "T-pca", "PCA-Part", "KM++", "Qk", "AFK-MC2", "k-means||"]:
            x = t[t.method == m].sort_values("iteration")
            ax.plot(x.time, 100 * (x.cost / best - 1), marker=".", ms=3, lw=1, color=COL[m], ls="--" if m == "T-pca" else "-")
        ax.set_ylim(-0.2, 8); ax.set_xlabel(f"seeding + Lloyd time (s), {ds}")
        ax.set_ylabel("excess over best reached (%)"); ax.set_title(f"{ds}, k = 1000", fontsize=8)
        ax.set_xlim(0, 66); ax.grid(alpha=0.3, lw=0.5, which="major")
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(os.path.join(OUT, "fig_largek.pdf")); fig.savefig(os.path.join(OUT, "fig_largek.png"), dpi=200)


def fig_example(name="2d-10c", seed_bad=None):
    X, y, k = load_any(name)
    C, info = ockmt.ockm_t(X, k, return_info=True)
    Z, G, mu = ockmt.project(X)
    m = 8 * k
    lab, csum, cnt, csse, snapC, L, used = ockmt._orbit_tree(Z, m, k, 4, 64, 2, 2, 4096, False, 4, 6, 0)
    R = csum / cnt[:, None] + mu
    # a k-means++ run that ends in a worse local optimum
    runs = [KMeans(k, init=kmeans_plusplus(X, k, random_state=s_)[0], n_init=1, max_iter=300, tol=1e-4,
                   algorithm="lloyd").fit(X) for s_ in range(31)]
    runs.sort(key=lambda km: km.inertia_)
    ref = KMeans(k, init=C, n_init=1, max_iter=300, tol=1e-4, algorithm="lloyd").fit(X).inertia_
    bad = [km for km in runs if km.inertia_ > 1.01 * ref]
    best = (len(bad), bad[len(bad) // 2] if bad else runs[15])   # median of the runs that end > 1% above OCKM-T
    print("k-means++ runs more than 1% above OCKM-T:", len(bad), "of", len(runs), [round(km.inertia_, 2) for km in bad])
    kmT = KMeans(k, init=C, n_init=1, max_iter=300, tol=1e-4, algorithm="lloyd").fit(X)
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.3))
    rng = np.random.default_rng(3); pal = rng.permutation(L)
    axes[0].scatter(X[:, 0], X[:, 1], c=pal[lab] % 20, cmap="tab20", s=0.6, lw=0)
    axes[0].scatter(R[:, 0], R[:, 1], s=3, c="k")
    axes[0].set_title(f"(a) tree: m = {L} cells, Φ = {info['phi']:.1f}", fontsize=8)
    axes[1].scatter(R[:, 0], R[:, 1], s=cnt / cnt.max() * 40, c="#7f7f7f", alpha=0.7, lw=0)
    axes[1].scatter(C[:, 0], C[:, 1], marker="^", s=22, c=COL["OCKM-T"], edgecolors="k", linewidths=0.4)
    axes[1].set_title(f"(b) summary and seeds (summary cost {info['summary_cost']:.1f})", fontsize=8)
    axes[2].scatter(X[:, 0], X[:, 1], s=0.4, c="#cccccc", lw=0)
    Cb = best[1].cluster_centers_
    axes[2].scatter(Cb[:, 0], Cb[:, 1], marker="x", s=20, c=COL["KM++"], linewidths=1, label=f"k-means++ (a run ending >1% higher): {best[1].inertia_:.1f}")
    axes[2].scatter(kmT.cluster_centers_[:, 0], kmT.cluster_centers_[:, 1], marker="^", s=18, c=COL["OCKM-T"], edgecolors="k", linewidths=0.4, label=f"OCKM-T: {kmT.inertia_:.1f}")
    axes[2].legend(fontsize=5.5, frameon=False, loc="lower left"); axes[2].set_title("(c) after Lloyd", fontsize=8)
    for a in axes:
        a.set_xticks([]); a.set_yticks([]); a.set_aspect("equal")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_example.pdf")); fig.savefig(os.path.join(OUT, "fig_example.png"), dpi=200)
    return info, kmT.inertia_, best[1].inertia_


if __name__ == "__main__":
    import sys
    what = sys.argv[1:] or ["pareto", "largek", "example"]
    if "pareto" in what: fig_pareto()
    if "largek" in what: fig_largek()
    if "example" in what: print(fig_example())
