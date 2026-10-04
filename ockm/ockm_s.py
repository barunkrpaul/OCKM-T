"""OCKM-S: OCKM-T with the shell-and-sector tree in place of the orbit-cut tree.

    from ockm_s import ockm_s
    C, info = ockm_s(X, k, seed=0, return_info=True)

info: phi (within-cell cost), summary_cost, m (cells), shell_cuts, sector_cuts, t_tree.
"""
from ockm_tc import circle_tree_seeding as ockm_s, ockm_tc, warmup  # noqa: F401
