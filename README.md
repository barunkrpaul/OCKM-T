# OCKM-T: k-means seeding by search on tree summaries

This repository contains the code, the raw results and the analysis scripts of the paper
*k-Means Seeding by Search on Tree Summaries* by Barun Kr. Paul and Kinsuk Giri (submitted to
Transactions on Machine Learning Research).

**Authors**

- Barun Kr. Paul, Department of Computer Science and Engineering, NITTTR Kolkata, and Department of
  Computer Science, Srikrishna College, India
- Kinsuk Giri (corresponding author), Department of Computer Science and Engineering, NITTTR Kolkata,
  India

OCKM-T seeds Lloyd's algorithm in four steps:

1. a divisive tree (best hyperplane among rotations of the principal cut of a cell) splits the data into
   `m = 8k` cells, and the within-cell cost `Phi` is computed exactly;
2. the cell centroids, weighted by the cell sizes, form a small summary;
3. a weighted k-means problem is solved on the summary from several starts (D² sampling, `2k`
   LocalSearch++ steps and weighted Lloyd);
4. the best solution is lifted back to the data (each centre becomes the mean of the points of its cells).

The returned centres satisfy `E[cost] <= (1 + 2a) Phi + 2a OPT_k`, where `a` is the approximation factor
of the search on the summary (Theorem 4.3 of the paper). When `d > 64` the tree is built on a Gaussian
projection to 64 dimensions.

**OCKM-S** is the same method with a shell-and-sector tree: each split is a sphere about the cell
centroid or a threshold on the first principal coordinate, in the manner of concentric-circle (CCKM)
seeding. The guarantee holds for it without change.

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10 or later. The code was run with NumPy 2.4, SciPy 1.17, scikit-learn 1.8, Numba 0.67 and
pandas 3.0.

## Use

```python
import sys; sys.path.insert(0, "ockm")
from ockmt import ockm_t, warmup
from ockm_s import ockm_s
from sklearn.cluster import KMeans

warmup()                                      # compiles the Numba kernels once (about 20 s, cached on disk)
C, info = ockm_t(X, k, seed=0, return_info=True)   # X: n x d array, C: k x d seeds
km = KMeans(k, init=C, n_init=1).fit(X)       # Lloyd from the seeds
print(info["phi"])                            # within-cell cost in the working space (info["phi_full"]: original space)

C_s, info_s = ockm_s(X, k, seed=0, return_info=True)  # OCKM-S
```

Main parameters of `ockm_t` (defaults): `gamma=8` cells per centre, `p=64` projection dimension
(used when `d > p`), `n_dirs=4` rotations of the principal cut, `nbins=64`, `refine=2` two-means steps
per split, `ls_steps=2.0` LocalSearch++ swaps per start (times `k`), `n_starts="auto"` (2 to 16 starts),
`pca_only=False` (`True` uses PCA-Part cuts only, which is faster), `seed=0`.

A complete example is in `examples/quickstart.py`, and `python tests/test_basic.py` checks the outputs,
`Phi` and the lifting inequality.

## Repository layout

| path | content |
|---|---|
| `ockm/ockmt.py` | OCKM-T (orbit-cut tree, D² sampling, LocalSearch++, weighted Lloyd, lift, polish) |
| `ockm/ockm_tc.py`, `ockm/ockm_s.py` | OCKM-S (shell-and-sector tree) and OCKM-TC (the seeding of lower cost among OCKM-T and OCKM-S) |
| `ockm/baselines2.py`, `qkmeans.py`, `det_baselines.py` | AFK-MC², k-means‖, Qkmeans, PCA-Part, Var-Part |
| `ockm/sacckm.py`, `sac_fast.py`, `ockm_fast.py` | CCKM, OCKM and OCKM-fast (earlier orbit-chart seedings) |
| `ockm/ockm_tree.py` | first (interpreted) version of the tree, used for one comparison in the paper |
| `ockm/bench_data.py`, `bench_t3.py`, `run_t3.sh` | development benchmark (42 data sets) |
| `ockm/bench_s.py` | OCKM-S on the development benchmark |
| `ockm/seed_grid2.py`, `seed_grid3.py` | large-k study and Lloyd traces |
| `ockm/dev_bench2.py`, `dev_eval2.py`, `dev_select.py` | hyperparameter choice on 90% subsamples |
| `ockm/summarise_t3.py`, `make_numbers.py`, `make_figs.py`, `verify_t3.py` | statistics, tables and figures of the development benchmark |
| `heldout/heldout_bench.py`, `heldout/OCKM-T_heldout_benchmark.ipynb` | held-out study (Covertype, CIFAR-10/100 features, 20 Newsgroups and AG News embeddings), runs on Colab or Kaggle with a GPU |
| `paper_stats/` | scripts for the held-out tables, the OCKM-S results, the ring example and Figure 2 |
| `results/development/` | raw runs of the development benchmark (`runs.csv`), OCKM-S runs (`runs_ockms.csv`), large-k grid and traces, development runs (`t2/`), first version of the tree (`t_v1/`) |
| `results/heldout/` | raw runs of the held-out study |
| `data/bench/` | ARFF files of the clustering-benchmark collection (39 are used; `bench_data.py` leaves out cpu, german and golfball) |

In `heldout_bench.py`, the method called "Circle tree" is OCKM-S.

## Reproducing the paper

All numbers, tables and figures can be regenerated from the raw results without rerunning the
benchmarks:

```bash
cd ockm
python make_numbers.py && python make_figs.py   # development benchmark -> paper_stats/out/
python verify_t3.py                             # independent recomputation of the quoted numbers
cd ../paper_stats
python heldout_stats.py && python dev_s_stats.py && python extra_stats.py && python ockms_table.py
python ring.py && python fig_trees.py           # ring example (Example 3.1) and Figure 2
```

To rerun the benchmarks:

```bash
cd ockm
python make_datasets.py MNIST FMNIST            # downloads the MNIST and Fashion-MNIST training sets into data/
./run_t3.sh                                     # development benchmark, about 1.5 h on two cores
python bench_s.py ../results/development/runs_ockms.csv
python seed_grid2.py --out ../results/development/seed_grid.csv
python seed_grid2.py --trace --out ../results/development/trace.csv
python summarise_t3.py ../results/development
```

The held-out study is run with `heldout/OCKM-T_heldout_benchmark.ipynb` (all code is embedded in the
notebook; set `PART` to 1, 2 or 3 to split the work across sessions). The notebook is generated from
the code in this repository by `python heldout/build_nb.py`.

Timings depend on the machine. The compiled kernels run single-threaded for small problems, and the
one-time Numba compilation is not timed.

## Data

`data/bench/` holds data sets of the clustering-benchmark collection
(<https://github.com/deric/clustering-benchmark>); see that repository for the original sources and
terms of each data set. MNIST, Fashion-MNIST and the held-out data sets are downloaded by the scripts
from their public sources.

## Citation

If you use this code, please cite:

```bibtex
@misc{paul2026ockmt,
  author = {Barun Kr. Paul and Kinsuk Giri},
  title  = {$k$-Means Seeding by Search on Tree Summaries},
  note   = {Submitted to Transactions on Machine Learning Research},
  year   = {2026}
}
```

The entry will be updated with the publication details after review.

## License

MIT (see `LICENSE`). The data sets keep their own licenses.
