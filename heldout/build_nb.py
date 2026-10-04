import os
import nbformat as nbf
HERE = os.path.dirname(os.path.abspath(__file__)); CODE = os.path.join(HERE, '..', 'ockm')
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s)); code = lambda s: C.append(nbf.v4.new_code_cell(s))
md("""# OCKM-T: held-out large-scale benchmark (Google Colab, fast version)

This notebook runs OCKM-T, the circle (concentric-sector) tree, their combination OCKM-TC and the paper's baselines on **large data sets that were never used for tuning**: Covertype (581,012 points), CIFAR-10 and CIFAR-100 image features (50,000 points), and 20 Newsgroups and AG News sentence embeddings (18,846 and 120,000 points), with k from 4 to 1,000.

**Main question.** Does combining the line tree (OCKM-T) and the circle tree give better centres than spending the same work on OCKM-T alone? Compare *Line+Circle (2 runs)* with *Line+Line (2 runs)*, and *Circle tree* with *OCKM-T*, in cell 7 ('Circle tree' is OCKM-S in the paper), especially at large k.

**How to use**
1. Colab: `Runtime → Change runtime type → T4 GPU`. Kaggle: Settings → Accelerator → GPU T4, and Internet on.
2. Run the cells from top to bottom. On Colab, sign in to Google Drive; results are written to `MyDrive/OCKM-T_heldout`.
3. Every finished run is saved at once to Drive. If Colab disconnects, reconnect and run all cells again: finished runs are skipped.

**Why this version is faster**
* **Lloyd runs on the GPU** (`GPU_LLOYD = True`). It follows scikit-learn's Lloyd rules exactly; on test data it gave the same final cost as scikit-learn to within 1e-9. Lloyd was most of the running time.
* **OCKM-TC is not run separately.** It is built from the OCKM-T and circle-tree runs with the same seed.
* **Weaker baselines use fewer seeds** (`OTHER_RUNS = 3`). OCKM-T, the circle tree and greedy k-means++ keep all their seeds.
* **The work is split into 3 parts** (`PART` in cell 3) that can run at the same time in different sessions (Colab, Kaggle). Each part writes its own CSV file, and the results cells read them all.
* The same notebook runs on **Kaggle** without changes (cell 1 detects it). **No file needs to be uploaded or attached**: all the code is inside the notebook.""")
code("""# 1. Paths. Works on Google Colab and on Kaggle. No upload is needed: all code is inside this notebook.
import os
ON_KAGGLE = os.path.exists('/kaggle')
if ON_KAGGLE:
    WORK_DIR = '/kaggle/working/OCKM-T_heldout'          # download results from the Output tab when the run ends
else:
    from google.colab import drive
    drive.mount('/content/drive')
    WORK_DIR = '/content/drive/MyDrive/OCKM-T_heldout'
os.makedirs(WORK_DIR, exist_ok=True)
# cached feature matrices (.npz). On Kaggle they are kept outside /kaggle/working so the saved Output stays small
CACHE_DIR = '/tmp/ockm_features' if ON_KAGGLE else os.path.join(WORK_DIR, 'features')
CODE_DIR = '/content/ockm_code'
os.makedirs(CODE_DIR, exist_ok=True)
print('running on', 'Kaggle' if ON_KAGGLE else 'Colab'); print('working folder:', WORK_DIR)""")
code("""# 2. Install packages
!pip -q install numba threadpoolctl sentence-transformers datasets""")
md("The next cells write the OCKM-T code files (unchanged from the ICML code, plus `ockm_tc.py`). Just run them.")
code("%%writefile /content/ockm_code/ockmt.py\n" + open(os.path.join(CODE, "ockmt.py")).read())
code("%%writefile /content/ockm_code/ockm_tc.py\n" + open(os.path.join(CODE, "ockm_tc.py")).read())
code("%%writefile /content/ockm_code/qkmeans.py\n" + open(os.path.join(CODE, "qkmeans.py")).read())
code("%%writefile /content/ockm_code/baselines2.py\n" + open(os.path.join(CODE, "baselines2.py")).read())
code("%%writefile /content/ockm_code/det_baselines.py\n" + open(os.path.join(CODE, "det_baselines.py")).read())
code("%%writefile /content/ockm_code/ockm_fast.py\n" + open(os.path.join(CODE, "ockm_fast.py")).read())
code("%%writefile /content/ockm_code/sacckm.py\n" + open(os.path.join(CODE, "sacckm.py")).read())
code("%%writefile /content/heldout_bench.py\n" + open(os.path.join(HERE, "heldout_bench.py")).read())
code("""# 3. Settings: (data set, k, seeds for OCKM-T / circle tree / greedy k-means++), grouped into 3 parts.
#    Each part uses its own data sets, so two sessions never compute the same features.
PARTS = {
    1: [('cifar10', 10, 10), ('cifar10', 100, 10), ('cifar10', 1000, 5)],                  # 50,000 x 128 image features
    2: [('covtype', 7, 10), ('covtype', 100, 5), ('20ng', 20, 10), ('20ng', 200, 10)],        # tabular + text
    3: [('agnews', 4, 10), ('agnews', 100, 5), ('cifar100', 100, 10)],                        # text + image features
}
PART = 1              # 1, 2 or 3. Use 0 to run all three parts one after another in this session.
OTHER_RUNS = 3        # seeds for the other randomised baselines (KM++v, Qk, AFK-MC2, k-means||, T-pca)
GPU_LLOYD  = True     # Lloyd on the GPU (same rules as scikit-learn); False = scikit-learn on the CPU
QUICK_TEST = False    # True: one small setting with 2 seeds, to check that everything works

MY_SETTINGS = [s for p in sorted(PARTS) for s in PARTS[p]] if PART == 0 else PARTS[PART]
if QUICK_TEST:
    MY_SETTINGS = [('20ng', 20, 2)]
OUT_CSV = os.path.join(WORK_DIR, f'results_part{PART}.csv' if not QUICK_TEST else 'results_quicktest.csv')
print('this session runs:', MY_SETTINGS); print('results go to', OUT_CSV)""")
code("""# 4. Compute and cache the features once (uses the GPU)
import sys, torch, importlib
sys.path.insert(0, '/content')
import heldout_bench as hb; importlib.reload(hb)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)
if GPU_LLOYD and device != 'cuda':
    print('No GPU found: Lloyd will use scikit-learn on the CPU'); GPU_LLOYD = False
for name in sorted({s[0] for s in MY_SETTINGS}):
    X, y = hb.load_dataset(name, CACHE_DIR, device)
    print(f'{name:10s} n={X.shape[0]:7d} d={X.shape[1]:4d} classes={len(set(y))}')""")
code("""# 5. Run the benchmark (resumable). Numba compilation happens once and is not timed.
#    On Kaggle, download results_part*.csv from the Output tab when the run ends.
methods, warmup = hb.get_methods(CODE_DIR)
warmup()
if GPU_LLOYD:                                          # warm up the GPU so the first timing is fair
    import numpy as np; hb.lloyd_torch(np.random.rand(2000, 16), np.random.rand(10, 16))
import time
for name, k, runs in MY_SETTINGS:
    X, y = hb.load_dataset(name, CACHE_DIR, device)
    t0 = time.time()
    hb.run_setting(name, k, runs, X, y, methods, OUT_CSV, other_runs=OTHER_RUNS, gpu_lloyd=GPU_LLOYD)
    print(f'--- {name} k={k} finished in {(time.time()-t0)/60:.1f} min')""")
md("""## Results
Run the cells below in any session once all parts have finished; they read every `results_part*.csv` file in the working folder.""")
code("""# 6. Combine the parts
import pandas as pd, glob
pd.set_option('display.width', 200)
ALL = OUT_CSV if QUICK_TEST else os.path.join(WORK_DIR, 'results_part*.csv')   # on Colab, first copy the Kaggle result files into this folder
d = hb.read_results(ALL)
print(len(d), 'runs from', len(glob.glob(ALL)), 'files'); display(d.groupby(['dataset', 'k']).method.nunique())""")
code("""# 7. Mean gap, mean rank, time relative to OCKM-T, OCKM-T better/tied/worse, Wilcoxon with Holm correction.
#    'Line+Circle (2 runs)' and 'Line+Line (2 runs)' keep the lower final cost of two runs (same work);
#    'OCKM-TC' picks between OCKM-T and the circle tree by seeding cost (one Lloyd run).
tab, per_setting, friedman_p = hb.summarise(ALL)
print('Friedman p =', friedman_p)
display(tab.round(3))
print('Mean gap (%) per setting:')
display(per_setting.round(3))""")
code("""# 8. Equal time: each randomised method gets OCKM-T's time and keeps its best restart
eq = hb.equal_time(ALL)
display(eq.round(3))""")
code("""# 9. Checks a reviewer will ask about
print('Runs that stopped at the 300-iteration cap:')
display(d.groupby(['dataset', 'k', 'method']).hit_max_iter.sum().loc[lambda s: s > 0])
print('Median seeding time (s):')
display(d.groupby(['dataset', 'k', 'method']).seed_time.median().unstack().round(2))
print('Mean ARI at k = number of classes:')
cls = {'20ng': 20, 'agnews': 4, 'cifar10': 10, 'cifar100': 100, 'covtype': 7}
display(d[d.k == d.dataset.map(cls)].groupby(['dataset', 'method']).ARI.mean().unstack().round(3))""")
code("""# 10. Save LaTeX tables and the combined CSV for the paper
d.to_csv(os.path.join(WORK_DIR, 'results_heldout_all.csv'), index=False)
tab.round(3).to_latex(os.path.join(WORK_DIR, 'table_heldout.tex'), index=False)
per_setting.round(2).to_latex(os.path.join(WORK_DIR, 'table_heldout_per_setting.tex'))
if len(eq):
    eq.round(3).to_latex(os.path.join(WORK_DIR, 'table_heldout_equal_time.tex'))
print('saved in', WORK_DIR)""")
nb.cells = C
nb.metadata["accelerator"] = "GPU"
nb.metadata["colab"] = {"provenance": [], "gpuType": "T4"}
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
nbf.write(nb, os.path.join(HERE, "OCKM-T_heldout_benchmark.ipynb"))
