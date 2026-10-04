"""Benchmark datasets for the OCKM-T study.

* 39 sets (35 labelled, 4 unlabelled) from the clustering-benchmark collection
  (github.com/deric/clustering-benchmark, ARFF files downloaded to data/bench/).
* sklearn's digits (1797 x 64).
* MNIST and Fashion-MNIST training sets (idx files in data/).

Preprocessing: tabular and 2-D sets are z-scored per feature (the OCKM protocol);
image data (digits, MNIST, FMNIST) are scaled to [0, 1]. Nominal features with
two levels are mapped to 0/1; other nominal or string features are dropped;
missing values are replaced by the column mean.
"""
import os, io, re, glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
BENCH = os.path.join(DATA, "bench")

UNLABELLED_K = {"s-set3": 15, "s-set4": 15, "birch-rg1": 100, "birch-rg3": 100}
SKIP = {"german", "cpu", "golfball"}   # categorical features / regression target / one class


def _read_arff(path):
    """Minimal ARFF reader: numeric, nominal and string attributes, '?' as missing."""
    names, types, rows, in_data = [], [], [], False
    with open(path, "r", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            if not in_data:
                low = s.lower()
                if low.startswith("@attribute"):
                    m = re.match(r"@attribute\s+('[^']*'|\"[^\"]*\"|\S+)\s+(.*)$", s, re.I)
                    nm, tp = m.group(1).strip("'\""), m.group(2).strip()
                    names.append(nm)
                    if tp.startswith("{"):
                        levels = [v.strip().strip("'\"") for v in tp.strip("{}").split(",")]
                        try:
                            [float(v) for v in levels]
                            types.append("numeric")      # ordinal levels written as numbers
                        except ValueError:
                            types.append("nominal")
                    else:
                        types.append(tp.split()[0].lower())
                elif low.startswith("@data"):
                    in_data = True
            else:
                parts = s.split(",") if "," in s else s.split()
                rows.append([c.strip().strip("'\"") for c in parts])
    df = pd.DataFrame(rows, columns=names)
    return df, dict(zip(names, types))


def load_bench(name):
    df, types = _read_arff(os.path.join(BENCH, name + ".arff"))
    cls = [c for c in df.columns if c.lower() in ("class", "label", "target")]
    y = None
    if cls and name not in UNLABELLED_K:
        y = pd.factorize(df[cls[0]].astype(str))[0]
        df = df.drop(columns=cls)
    cols = []
    for c in df.columns:
        t = types.get(c, "numeric")
        if t in ("numeric", "real", "integer"):
            cols.append(pd.to_numeric(df[c].replace("?", np.nan), errors="coerce"))
        elif t == "nominal":
            vals = df[c][df[c] != "?"].unique()
            if len(vals) == 2:
                cols.append((df[c] == vals[0]).astype(float).where(df[c] != "?"))
    X = pd.concat(cols, axis=1).to_numpy(dtype=float)
    X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
    keep = X.std(0) > 0
    X = X[:, keep]
    X = (X - X.mean(0)) / X.std(0)
    k = len(np.unique(y)) if y is not None else UNLABELLED_K[name]
    return np.ascontiguousarray(X), y, k


def load_any(name):
    if name in ("MNIST", "FMNIST"):
        from bench_manifold import load
        X, y = load(name, "paper")
        return X, y, 10
    if name == "digits":
        from sklearn.datasets import load_digits
        X, y = load_digits(return_X_y=True)
        return np.ascontiguousarray(X / 16.0), y, 10
    return load_bench(name)


def bench_names():
    names = sorted(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(BENCH, "*.arff")))
    return [n for n in names if n not in SKIP]
