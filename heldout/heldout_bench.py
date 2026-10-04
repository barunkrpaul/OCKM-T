"""Held-out large-scale benchmark for OCKM-T (none of these data sets was used for tuning).

Protocol (same as the paper): every seeding is followed by one Lloyd run with scikit-learn
(n_init=1, algorithm='lloyd', tol=1e-4); randomised methods use seeds 0..R-1; the gap of a run is
its final cost above the lowest final cost reached by any run on that setting.
Rows are appended to a CSV as soon as they finish, so the run can be resumed in a new session.
"""
from __future__ import annotations
import os, sys, time, json, warnings
import numpy as np, pandas as pd
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------- data sets
def _standardise(X):
    X = np.asarray(X, dtype=np.float64)
    X = X[:, X.std(0) > 0]
    return np.ascontiguousarray((X - X.mean(0)) / X.std(0))


def _pca(X, dim, seed=0):
    from sklearn.decomposition import PCA
    return np.ascontiguousarray(PCA(dim, random_state=seed).fit_transform(X).astype(np.float64))


def _image_features(name, device):
    """ResNet-50 (ImageNet weights) penultimate features of the CIFAR training set, reduced to 128 dims by PCA."""
    import torch, torchvision
    from torchvision import transforms
    ds_cls = {"cifar10": torchvision.datasets.CIFAR10, "cifar100": torchvision.datasets.CIFAR100,
              "svhn": torchvision.datasets.SVHN}[name]
    tf = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    ds = ds_cls("/content/tv_data", split="train", download=True, transform=tf) if name == "svhn" else \
        ds_cls("/content/tv_data", train=True, download=True, transform=tf)
    y = np.array(ds.labels if name == "svhn" else ds.targets)
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Identity(); model.eval().to(device)
    dl = torch.utils.data.DataLoader(ds, batch_size=256, num_workers=2)
    feats = []
    with torch.no_grad():
        for xb, _ in dl:
            feats.append(model(xb.to(device)).float().cpu().numpy())
    return _pca(np.concatenate(feats), 128), y


def _text_features(name, device):
    """all-MiniLM-L6-v2 sentence embeddings (384 dims, unit length)."""
    from sentence_transformers import SentenceTransformer
    if name == "20ng":
        from sklearn.datasets import fetch_20newsgroups
        d = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
        texts, y = d.data, np.array(d.target)
    else:  # ag_news
        from datasets import load_dataset
        d = None
        for repo in ("fancyzhx/ag_news", "ag_news"):          # the Hub moved 'ag_news' to 'fancyzhx/ag_news'
            try:
                d = load_dataset(repo, split="train"); break
            except Exception as e:
                print("could not load", repo, "->", type(e).__name__)
        if d is None:
            raise RuntimeError("AG News could not be downloaded from the Hugging Face Hub")
        texts, y = d["text"], np.array(d["label"])
    m = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    E = m.encode(texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    return np.ascontiguousarray(E.astype(np.float64)), y


def load_dataset(name, cache_dir, device="cpu"):
    """Returns (X, y). Features are computed once and cached as .npz in cache_dir."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{name}.npz")
    if os.path.exists(path):
        z = np.load(path); return z["X"], z["y"]
    if name == "covtype":
        from sklearn.datasets import fetch_covtype
        d = fetch_covtype(); X, y = _standardise(d.data), d.target
    elif name in ("cifar10", "cifar100", "svhn"):
        X, y = _image_features(name, device)
    elif name in ("20ng", "agnews"):
        X, y = _text_features(name, device)
    elif name == "letter":   # only for a quick test; letter was already in the paper's large-k study
        from sklearn.datasets import fetch_openml
        d = fetch_openml("letter", version=1, as_frame=False); X, y = _standardise(d.data), d.target
    else:
        raise ValueError(name)
    np.savez_compressed(path, X=X, y=np.asarray(y))
    return X, np.asarray(y)


# --------------------------------------------------------------------------------- methods
def get_methods(code_dir):
    sys.path.insert(0, code_dir)
    import ockmt
    from ockmt import ockm_t
    from qkmeans import qkmeans
    from baselines2 import afkmc2, kmeans_par
    from det_baselines import pca_part, var_part
    import ockm_tc

    def _t(X, k, s, **kw):
        c, info = ockm_t(X, k, return_info=True, seed=s, **kw)
        return c, {kk: info[kk] for kk in ("phi", "phi_full", "n_starts", "m") if kk in info}

    methods = {
        "OCKM-T":    (lambda X, k, s: _t(X, k, s), True),
        "Circle tree": (lambda X, k, s: (ockm_tc.circle_tree_seeding(X, k, seed=s), {}), True),
        "T-pca":     (lambda X, k, s: _t(X, k, s, pca_only=True), True),
        "KM++":      (lambda X, k, s: (kmeans_plusplus(X, k, random_state=s)[0], {}), True),
        "KM++v":     (lambda X, k, s: (kmeans_plusplus(X, k, random_state=s, n_local_trials=1)[0], {}), True),
        "Qk":        (lambda X, k, s: (qkmeans(X, k, m=10.0, seed=s), {}), True),
        "AFK-MC2":   (lambda X, k, s: (afkmc2(X, k, seed=s), {}), True),
        "k-means||": (lambda X, k, s: (kmeans_par(X, k, seed=s), {}), True),
        "PCA-Part":  (lambda X, k, s: (pca_part(X, k), {}), False),
        "Var-Part":  (lambda X, k, s: (var_part(X, k), {}), False),
    }
    return methods, ockm_tc.warmup


def cost(X, C, chunk=8192):
    cc = (C * C).sum(1)
    tot = 0.0
    for i in range(0, len(X), chunk):
        A = X[i:i + chunk]
        D = (A * A).sum(1)[:, None] + cc[None, :] - 2.0 * A @ C.T
        tot += float(np.maximum(D.min(1), 0).sum())
    return tot


CORE = ("OCKM-T", "Circle tree", "KM++")      # methods that answer the main question: always all seeds


# ---------------------------------------------------------------------------- GPU Lloyd
def lloyd_torch(X, C, max_iter=300, tol=1e-4, device="cuda", chunk=65536):
    """Lloyd's algorithm on the GPU, same rules as scikit-learn's 'lloyd': stop when the
    assignment is unchanged or the total squared centre shift is at most tol * mean feature
    variance; an empty cluster is moved to the point farthest from its centre. Iterations run
    in float32; the final cost and labels are recomputed in float64 on the CPU."""
    import torch
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    Ct = torch.as_tensor(C, dtype=torch.float32, device=device).clone()
    n, k = Xt.shape[0], Ct.shape[0]
    tol_abs = tol * float(np.mean(np.var(X, axis=0)))
    xx = (Xt * Xt).sum(1)
    prev = None; it = 0
    for it in range(1, max_iter + 1):
        cc = (Ct * Ct).sum(1)
        lab = torch.empty(n, dtype=torch.long, device=device); dmin = torch.empty(n, device=device)
        for i in range(0, n, chunk):
            D = xx[i:i + chunk, None] + cc[None, :] - 2.0 * Xt[i:i + chunk] @ Ct.T
            dmin[i:i + chunk], lab[i:i + chunk] = D.min(1)
        if prev is not None and torch.equal(lab, prev):
            it -= 1; break
        cnt = torch.bincount(lab, minlength=k).float()
        S = torch.zeros_like(Ct).index_add_(0, lab, Xt)
        newC = Ct.clone(); nz = cnt > 0
        newC[nz] = S[nz] / cnt[nz, None]
        if (~nz).any():                                    # relocate empty clusters
            far = torch.argsort(dmin, descending=True)[: int((~nz).sum())]
            newC[~nz] = Xt[far]
        shift = float(((newC - Ct) ** 2).sum())
        Ct = newC; prev = lab
        if shift <= tol_abs:
            break
    Cf = Ct.double().cpu().numpy()
    labels = np.empty(n, dtype=np.int64); inertia = 0.0; cf2 = (Cf * Cf).sum(1)
    for i in range(0, n, 8192):
        A = X[i:i + 8192]
        D = (A * A).sum(1)[:, None] + cf2[None, :] - 2.0 * A @ Cf.T
        labels[i:i + 8192] = D.argmin(1); inertia += float(np.maximum(D.min(1), 0).sum())
    return Cf, labels, inertia, it


# ------------------------------------------------------------------------------ benchmark
def run_setting(name, k, runs, X, y, methods, out_csv, max_iter=300, only=None, other_runs=3, gpu_lloyd=False):
    """runs: seeds for the core methods; other_runs: seeds for the other randomised baselines."""
    done = set()
    if os.path.exists(out_csv):
        d = pd.read_csv(out_csv, usecols=["dataset", "k", "method", "seed"])
        done = set(zip(d.dataset, d.k, d.method, d.seed))
    for meth, (fn, randomised) in methods.items():
        if only and meth not in only:
            continue
        nr = runs if meth in CORE else min(runs, other_runs)
        for s in (range(nr) if randomised else [0]):
            if (name, k, meth, s) in done:
                continue
            t0 = time.perf_counter(); C, ex = fn(X, k, s); ts = time.perf_counter() - t0
            C = np.ascontiguousarray(C, dtype=np.float64)
            t0 = time.perf_counter()
            if gpu_lloyd:
                _, labs, wcss, n_it = lloyd_torch(X, C, max_iter=max_iter, device=gpu_lloyd if isinstance(gpu_lloyd, str) else "cuda")
            else:
                km = KMeans(k, init=C, n_init=1, max_iter=max_iter, tol=1e-4, algorithm="lloyd", random_state=0).fit(X)
                labs, wcss, n_it = km.labels_, float(km.inertia_), int(km.n_iter_)
            tl = time.perf_counter() - t0
            row = dict(dataset=name, n=len(X), d=X.shape[1], k=k, method=meth, seed=s, seed_time=ts,
                       seed_cost=cost(X, C), WCSS=float(wcss), IT=int(n_it), lloyd_time=tl, total_time=ts + tl,
                       hit_max_iter=int(n_it >= max_iter), lloyd="gpu" if gpu_lloyd else "sklearn",
                       ARI=adjusted_rand_score(y, labs), NMI=normalized_mutual_info_score(y, labs),
                       extra=json.dumps({kk: float(v) for kk, v in ex.items()}))
            pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
        print(f"{name:10s} k={k:5d} {meth:10s} done", flush=True)


# -------------------------------------------------------------------------------- summary
def _holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def add_portfolios(d):
    """Two-run portfolios built from recorded runs, keeping the lower FINAL cost (two Lloyd runs):
    'Line+Circle (2 runs)' = OCKM-T seed s and circle tree seed s;
    'Line+Line (2 runs)'   = OCKM-T seeds s and s+1 (mod R), the fair comparison with the same work."""
    rows = []
    for (ds, k), D in d.groupby(["dataset", "k"]):
        L = D[D.method == "OCKM-T"].set_index("seed"); Cc = D[D.method == "Circle tree"].set_index("seed")
        seeds = sorted(set(L.index) & set(Cc.index)); R = len(L)
        for s in seeds:
            for name, a, b in (("Line+Circle (2 runs)", L.loc[s], Cc.loc[s]),
                               ("Line+Line (2 runs)", L.loc[s], L.loc[sorted(L.index)[(sorted(L.index).index(s) + 1) % R]])):
                win = a if a.WCSS <= b.WCSS else b
                r = win.copy(); r["method"] = name; r["seed"] = s
                r["total_time"] = a.total_time + b.total_time; r["seed_time"] = a.seed_time + b.seed_time
                rows.append(r)
            a, b = L.loc[s], Cc.loc[s]                      # OCKM-TC: pick by seeding cost, one Lloyd run
            win = a if a.seed_cost <= b.seed_cost else b
            r = win.copy(); r["method"] = "OCKM-TC"; r["seed"] = s
            r["seed_time"] = a.seed_time + b.seed_time; r["total_time"] = r["seed_time"] + win.lloyd_time
            rows.append(r)
    return pd.concat([d, pd.DataFrame(rows)], ignore_index=True) if rows else d


def read_results(out_csv):
    """out_csv may be one file or a glob pattern such as '.../results_part*.csv'."""
    import glob
    files = sorted(glob.glob(out_csv))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True).drop_duplicates(["dataset", "k", "method", "seed"])


def summarise(out_csv, ref="OCKM-T", tie=0.01, portfolios=True):
    from scipy import stats
    d = read_results(out_csv)
    if portfolios:
        d = add_portfolios(d)
    d["setting"] = d.dataset + " k=" + d.k.astype(str)
    best = d.groupby("setting").WCSS.min()
    d["gap"] = 100 * (d.WCSS / d.setting.map(best) - 1)
    g = d.groupby(["setting", "method"]).gap.mean().unstack()
    tm = d.groupby(["setting", "method"]).total_time.median().unstack()
    ranks = g.rank(axis=1).mean().sort_values()
    rows, pv = [], []
    for m in g.columns:
        if m == ref:
            continue
        x = (g[ref] - g[m]).dropna()
        w, t, l = int((x < -tie).sum()), int((x.abs() <= tie).sum()), int((x > tie).sum())
        p = stats.wilcoxon(x).pvalue if (x.abs() > 0).sum() >= 1 and len(x) >= 5 else np.nan
        rows.append(dict(method=m, mean_gap=g[m].mean(), mean_rank=ranks[m], time_vs_ref=(tm[m] / tm[ref]).median(),
                         **{f"{ref} better/tied/worse": f"{w}/{t}/{l}"}, p=p)); pv.append(p)
    tab = pd.DataFrame(rows)
    ok = ~tab.p.isna()
    tab.loc[ok, "p_holm"] = _holm(tab.p[ok].values)
    head = pd.DataFrame([dict(method=ref, mean_gap=g[ref].mean(), mean_rank=ranks[ref], time_vs_ref=1.0)])
    tab = pd.concat([head, tab], ignore_index=True).sort_values("mean_rank")
    try:
        from scipy.stats import friedmanchisquare
        cols = g.dropna().T.values
        fp = friedmanchisquare(*cols).pvalue if len(g.dropna()) >= 3 else np.nan
    except Exception:
        fp = np.nan
    return tab, g, fp


def equal_time(out_csv, ref="OCKM-T"):
    """For each run of ref, every randomised method is given the same total time: it is restarted with its
    recorded runs (in seed order) while the summed time stays within the budget (at least one run), and the
    best final cost is kept. Limited by the number of recorded runs."""
    d = add_portfolios(read_results(out_csv)); d = d[~d.method.isin(["Line+Circle (2 runs)", "Line+Line (2 runs)"])]
    d["setting"] = d.dataset + " k=" + d.k.astype(str)
    best = d.groupby("setting").WCSS.min(); res = []
    for st, D in d.groupby("setting"):
        R = D[D.method == ref]
        for m in D.method.unique():
            if m == ref or D[D.method == m].seed.nunique() <= 1:
                continue
            M = D[D.method == m].sort_values("seed")
            for _, r in R.iterrows():
                budget, acc, bestc = r.total_time, 0.0, np.inf
                order = np.roll(M.index.values, -int(r.seed) % len(M))
                for i in order:
                    if acc > 0 and acc + M.loc[i, "total_time"] > budget:
                        break
                    acc += M.loc[i, "total_time"]; bestc = min(bestc, M.loc[i, "WCSS"])
                res.append(dict(setting=st, method=m, ref_gap=100 * (r.WCSS / best[st] - 1), other_gap=100 * (bestc / best[st] - 1)))
    e = pd.DataFrame(res)
    if e.empty:
        return e
    e = e.groupby(["setting", "method"])[["ref_gap", "other_gap"]].mean().reset_index()
    e["diff"] = e.ref_gap - e.other_gap
    return e.groupby("method").apply(lambda x: pd.Series(dict(
        wtl=f"{int((x['diff'] < -0.01).sum())}/{int((x['diff'].abs() <= 0.01).sum())}/{int((x['diff'] > 0.01).sum())}",
        mean_ref_gap=x.ref_gap.mean(), mean_other_gap=x.other_gap.mean())))
