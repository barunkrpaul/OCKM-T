"""OCKM-S (circle/shell-cut tree) next to OCKM-T and greedy k-means++ on the 42 benchmark sets,
same protocol as bench_t3.py, same machine for all three methods."""
import sys, os
import bench_t3 as B
import ockm_tc
_orig = B.seed_fn
def seed_fn(meth, X, k, s):
    if meth == "OCKM-S":
        c, info = ockm_tc.circle_tree_seeding(X, k, seed=s, return_info=True)
        return c, {kk: info[kk] for kk in ("phi", "m", "shell_cuts", "sector_cuts")}
    return _orig(meth, X, k, s)
B.seed_fn = seed_fn
B.RANDOM.add("OCKM-S")
if __name__ == "__main__":
    out = sys.argv[1]
    ockm_tc.warmup()
    for nm in B.bench_names() + ["digits", "MNIST", "FMNIST"]:
        B.run(nm, ["OCKM-T", "OCKM-S", "KM++"], out, 10)
