"""Build the 12 datasets of Shah, Agrawal & Jaiswal (arXiv:2602.01104, Table 2)
as data/<NAME>.npz (X float32/float64, y int) for bench_manifold.py.

    MNIST, FMNIST          60,000 x 784   train split, pixels / 255 (idx files)
    CIFAR10, CIFAR100      50,000 x 3072  train split, pixels / 255 (torchvision)
    MNIST_CLIP ...         60,000/50,000 x 512  CLIP ViT-B/32 image features
    REDDIT                 100,000 x 384  all-MiniLM-L6-v2 sentence embeddings
    STACKEXCHANGE          20,000 x 384   all-MiniLM-L6-v2 sentence embeddings
    SUSY                   5,000,000 x 18 UCI SUSY
    HAR                    10,299 x 561   UCI HAR, train + test

The paper does not say which Reddit/StackExchange corpora it embedded, nor
whether the CLIP features were L2-normalised. We use the MTEB clustering test
sets (mteb/reddit-clustering, mteb/stackexchange-clustering), flattened and
subsampled to the Table 2 sizes with a fixed seed, and raw (unnormalised) CLIP
image features. Set CLIP_L2NORM = True to normalise instead.

usage: python make_datasets.py NAME [NAME ...]   (or: all)
"""
import os, io, sys, gzip, zipfile, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
os.makedirs(DATA, exist_ok=True)
CLIP_L2NORM = False
SUBSAMPLE_SEED = 0

IDX = {"MNIST": ("mnist_", "https://raw.githubusercontent.com/fgnt/mnist/master/"),
       "FMNIST": ("fmnist_", "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/")}
SUSY_URLS = ["https://archive.ics.uci.edu/static/public/279/susy.zip",
             "https://archive.ics.uci.edu/ml/machine-learning-databases/00279/SUSY.csv.gz"]
HAR_URLS = ["https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"]
TEXT = {"REDDIT": ("mteb/reddit-clustering", 100_000), "STACKEXCHANGE": ("mteb/stackexchange-clustering", 20_000)}


def fetch(url, path):
    if not os.path.exists(path):
        print("downloading", url, flush=True)
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    return path


def fetch_any(urls, path):
    err = None
    for u in urls:
        try:
            return fetch(u, path)
        except Exception as e:  # try the next mirror
            err = e; print("  failed:", e)
    raise RuntimeError(f"all mirrors failed for {path}: {err}")


def save(name, X, y):
    np.savez(os.path.join(DATA, name + ".npz"), X=X, y=np.asarray(y))
    print(f"{name}: X {X.shape} {X.dtype}, {len(np.unique(y))} labels", flush=True)


# ---------------------------------------------------------------- images
def _idx(path):
    with gzip.open(path, "rb") as f:
        raw = f.read()
    nd = raw[3]
    dims = [int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(nd)]
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * nd).reshape(dims)


def idx_images(name):
    """uint8 images (n, 28, 28) and labels, from the same idx files bench_manifold.py reads."""
    p, base = IDX[name]
    out = []
    for f in ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"):
        out.append(_idx(fetch(base + f, os.path.join(DATA, p + f))))
    return out[0], out[1].astype(int)


def cifar_images(name):
    import torchvision
    cls = torchvision.datasets.CIFAR10 if name == "CIFAR10" else torchvision.datasets.CIFAR100
    ds = cls(root=os.path.join(DATA, "torchvision"), train=True, download=True)
    return np.asarray(ds.data), np.asarray(ds.targets, dtype=int)     # (50000, 32, 32, 3) uint8


def build_raw(name):
    if name in IDX:
        idx_images(name)                       # bench_manifold.py reads the idx files directly
        print(f"{name}: idx files ready", flush=True)
    else:
        imgs, y = cifar_images(name)
        save(name, imgs.reshape(len(imgs), -1).astype(np.float32) / 255.0, y)


def build_clip(name):
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    base = name.replace("_CLIP", "")
    imgs, y = idx_images(base) if base in IDX else cifar_images(base)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    feats = []
    with torch.no_grad():
        for i in range(0, len(imgs), 500):
            batch = [Image.fromarray(a).convert("RGB") for a in imgs[i:i + 500]]
            inp = proc(images=batch, return_tensors="pt").to(dev)
            f = model.get_image_features(**inp).float()
            if CLIP_L2NORM:
                f = torch.nn.functional.normalize(f, dim=1)
            feats.append(f.cpu().numpy())
            if i % 10000 == 0:
                print(f"  {name}: {i}/{len(imgs)}", flush=True)
    save(name, np.concatenate(feats).astype(np.float32), y)


# ---------------------------------------------------------------- text
def flatten_mteb(rows):
    """MTEB clustering sets store one list of sentences (and labels) per row."""
    sents, labs = [], []
    for r in rows:
        s, l = r["sentences"], r["labels"]
        if isinstance(s, str):
            s, l = [s], [l]
        sents += list(s); labs += list(l)
    return sents, labs


def build_text(name):
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    repo, n = TEXT[name]
    sents, labs = flatten_mteb(load_dataset(repo, split="test"))
    print(f"  {name}: {len(sents)} sentences in {repo}", flush=True)
    idx = np.sort(np.random.default_rng(SUBSAMPLE_SEED).choice(len(sents), size=min(n, len(sents)), replace=False))
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    X = model.encode([sents[i] for i in idx], batch_size=256, show_progress_bar=True, convert_to_numpy=True)
    y = np.unique(np.asarray([str(labs[i]) for i in idx]), return_inverse=True)[1]
    save(name, X.astype(np.float32), y)


# ---------------------------------------------------------------- tabular
def read_susy(path):
    import pandas as pd
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            member = [m for m in z.namelist() if m.lower().endswith((".csv", ".csv.gz"))][0]
            raw = z.read(member)
        buf = gzip.GzipFile(fileobj=io.BytesIO(raw)) if member.endswith(".gz") else io.BytesIO(raw)
        df = pd.read_csv(buf, header=None, dtype=np.float32)
    else:
        df = pd.read_csv(path, header=None, dtype=np.float32)
    return df.iloc[:, 1:].to_numpy(), df.iloc[:, 0].to_numpy().astype(int)


def build_susy():
    last = None
    for u in SUSY_URLS:
        try:
            p = fetch(u, os.path.join(DATA, os.path.basename(u)))
            X, y = read_susy(p); break
        except Exception as e:
            last = e; print("  failed:", e)
    else:
        raise RuntimeError(f"SUSY download failed: {last}")
    save("SUSY", X, y)


def read_har(path):
    """Handles the UCI zip that wraps 'UCI HAR Dataset.zip' and the direct zip."""
    z = zipfile.ZipFile(path)
    inner = [m for m in z.namelist() if m.endswith("UCI HAR Dataset.zip")]
    if inner:
        z = zipfile.ZipFile(io.BytesIO(z.read(inner[0])))
    names = z.namelist()

    def arr(suffix):
        m = [x for x in names if x.endswith(suffix)][0]
        return np.loadtxt(io.BytesIO(z.read(m)))
    X = np.vstack([arr("train/X_train.txt"), arr("test/X_test.txt")])
    y = np.concatenate([arr("train/y_train.txt"), arr("test/y_test.txt")]).astype(int)
    return X, y


def build_har():
    p = fetch_any(HAR_URLS, os.path.join(DATA, "har.zip"))
    X, y = read_har(p)
    save("HAR", X.astype(np.float32), y)


BUILDERS = {"MNIST": lambda: build_raw("MNIST"), "FMNIST": lambda: build_raw("FMNIST"),
            "CIFAR10": lambda: build_raw("CIFAR10"), "CIFAR100": lambda: build_raw("CIFAR100"),
            "MNIST_CLIP": lambda: build_clip("MNIST_CLIP"), "FMNIST_CLIP": lambda: build_clip("FMNIST_CLIP"),
            "CIFAR10_CLIP": lambda: build_clip("CIFAR10_CLIP"), "CIFAR100_CLIP": lambda: build_clip("CIFAR100_CLIP"),
            "REDDIT": lambda: build_text("REDDIT"), "STACKEXCHANGE": lambda: build_text("STACKEXCHANGE"),
            "SUSY": build_susy, "HAR": build_har}

if __name__ == "__main__":
    names = list(BUILDERS) if sys.argv[1:] == ["all"] else sys.argv[1:]
    for nm in names:
        if nm not in ("MNIST", "FMNIST") and os.path.exists(os.path.join(DATA, nm + ".npz")):
            print(nm, "already built"); continue
        BUILDERS[nm]()
