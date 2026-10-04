# OCKM-T benchmark: statistics

42 datasets. Quality = gap of the final WCSS (after one Lloyd run) to the best WCSS found by any
run of any method on that dataset, in per cent (0 = best known). Randomised methods (including OCKM-T) enter with their mean over 10 runs
(the expected single-run gap). Ties: |difference| <= 0.01 percentage points of gap.


### Final WCSS gap to best known, all datasets

Friedman chi2 = 133.8, p = 7.7e-24; N = 42 datasets, 11 methods; Nemenyi CD (alpha = 0.05) = 2.33.

| method | mean rank | Wilcoxon p vs OCKM-T | Holm | OCKM-T wins / ties / losses |
|---|---|---|---|---|
| k-means++ (greedy) | 5.07 | 1.9e-07 | 1.1e-06 | 34 / 7 / 1 |
| k-means++ (vanilla) | 7.39 | 2.6e-08 | 2.6e-07 | 39 / 3 / 0 |
| Qkmeans | 7.61 | 4.2e-08 | 3.7e-07 | 38 / 4 / 0 |
| AFK-MC² | 7.68 | 4.2e-08 | 3.7e-07 | 38 / 4 / 0 |
| k-means‖ | 7.58 | 4.2e-08 | 3.7e-07 | 38 / 4 / 0 |
| PCA-Part | 4.92 | 1.8e-04 | 3.6e-04 | 26 / 9 / 7 |
| Var-Part | 6.18 | 1.5e-06 | 6.1e-06 | 31 / 7 / 4 |
| CCKM | 7.86 | 1.2e-06 | 6.1e-06 | 30 / 7 / 5 |
| OCKM (published) | 3.58 | 3.9e-03 | 3.9e-03 | 19 / 15 / 8 |
| OCKM-fast | 5.57 | 4.9e-06 | 1.5e-05 | 26 / 12 / 4 |
| **OCKM-T** | **2.56** | | | |

### Final WCSS gap, synthetic (2-D and 3-D benchmark shapes)

Friedman chi2 = 74.7, p = 5.4e-12; N = 21 datasets, 11 methods; Nemenyi CD (alpha = 0.05) = 3.29.

| method | mean rank | Wilcoxon p vs OCKM-T | Holm | OCKM-T wins / ties / losses |
|---|---|---|---|---|
| k-means++ (greedy) | 4.57 | 2.9e-04 | 1.8e-03 | 16 / 5 / 0 |
| k-means++ (vanilla) | 7.64 | 8.9e-05 | 8.9e-04 | 19 / 2 / 0 |
| Qkmeans | 8.17 | 1.3e-04 | 1.2e-03 | 18 / 3 / 0 |
| AFK-MC² | 7.26 | 1.3e-04 | 1.2e-03 | 18 / 3 / 0 |
| k-means‖ | 7.31 | 1.3e-04 | 1.2e-03 | 18 / 3 / 0 |
| PCA-Part | 5.31 | 5.7e-03 | 1.1e-02 | 12 / 7 / 2 |
| Var-Part | 6.26 | 3.9e-04 | 1.9e-03 | 15 / 6 / 0 |
| CCKM | 8.10 | 7.1e-04 | 2.9e-03 | 14 / 5 / 2 |
| OCKM (published) | 3.98 | 2.3e-02 | 2.3e-02 | 8 / 12 / 1 |
| OCKM-fast | 4.90 | 1.8e-03 | 5.4e-03 | 11 / 9 / 1 |
| **OCKM-T** | **2.50** | | | |

### Final WCSS gap, real (tabular and image)

Friedman chi2 = 66.4, p = 2.2e-10; N = 21 datasets, 11 methods; Nemenyi CD (alpha = 0.05) = 3.29.

| method | mean rank | Wilcoxon p vs OCKM-T | Holm | OCKM-T wins / ties / losses |
|---|---|---|---|---|
| k-means++ (greedy) | 5.57 | 2.4e-05 | 1.4e-04 | 18 / 2 / 1 |
| k-means++ (vanilla) | 7.14 | 1.9e-06 | 1.9e-05 | 20 / 1 / 0 |
| Qkmeans | 7.05 | 1.9e-06 | 1.9e-05 | 20 / 1 / 0 |
| AFK-MC² | 8.10 | 1.9e-06 | 1.9e-05 | 20 / 1 / 0 |
| k-means‖ | 7.86 | 1.9e-06 | 1.9e-05 | 20 / 1 / 0 |
| PCA-Part | 4.52 | 6.4e-03 | 1.3e-02 | 14 / 2 / 5 |
| Var-Part | 6.10 | 1.9e-03 | 5.8e-03 | 16 / 1 / 4 |
| CCKM | 7.62 | 1.6e-04 | 8.1e-04 | 16 / 2 / 3 |
| OCKM (published) | 3.19 | 8.6e-02 | 8.6e-02 | 11 / 3 / 7 |
| OCKM-fast | 6.24 | 5.9e-04 | 2.4e-03 | 15 / 3 / 3 |
| **OCKM-T** | **2.62** | | | |

### ARI at k = number of classes (labelled datasets)

Friedman chi2 = 20.9, p = 2.2e-02; N = 38 datasets, 11 methods; Nemenyi CD (alpha = 0.05) = 2.45.

| method | mean rank | Wilcoxon p vs OCKM-T | Holm | OCKM-T wins / ties / losses |
|---|---|---|---|---|
| k-means++ (greedy) | 4.91 | 3.6e-01 | 7.2e-01 | 20 / 4 / 14 |
| k-means++ (vanilla) | 6.13 | 5.1e-02 | 3.0e-01 | 24 / 1 / 13 |
| Qkmeans | 6.34 | 8.4e-02 | 4.2e-01 | 20 / 2 / 16 |
| AFK-MC² | 6.24 | 1.4e-02 | 1.2e-01 | 24 / 2 / 12 |
| k-means‖ | 6.53 | 9.1e-03 | 9.1e-02 | 23 / 2 / 13 |
| PCA-Part | 6.88 | 1.4e-02 | 1.2e-01 | 19 / 6 / 13 |
| Var-Part | 6.14 | 1.2e-01 | 4.6e-01 | 22 / 4 / 12 |
| CCKM | 6.88 | 1.9e-02 | 1.3e-01 | 20 / 4 / 14 |
| OCKM (published) | 4.80 | 9.4e-01 | 9.4e-01 | 14 / 11 / 13 |
| OCKM-fast | 6.20 | 1.6e-01 | 4.8e-01 | 19 / 8 / 11 |
| **OCKM-T** | **4.95** | | | |

### Seeding cost relative to vanilla k-means++

Friedman chi2 = 308.4, p = 2.7e-60; N = 42 datasets, 11 methods; Nemenyi CD (alpha = 0.05) = 2.33.

| method | mean rank | Wilcoxon p vs OCKM-T | Holm | OCKM-T wins / ties / losses |
|---|---|---|---|---|
| k-means++ (greedy) | 6.10 | 4.5e-13 | 4.5e-12 | 42 / 0 / 0 |
| k-means++ (vanilla) | 8.45 | 4.5e-13 | 4.5e-12 | 42 / 0 / 0 |
| Qkmeans | 8.83 | 4.5e-13 | 4.5e-12 | 42 / 0 / 0 |
| AFK-MC² | 8.60 | 4.5e-13 | 4.5e-12 | 42 / 0 / 0 |
| k-means‖ | 3.43 | 4.5e-13 | 4.5e-12 | 42 / 0 / 0 |
| PCA-Part | 2.95 | 4.5e-12 | 4.5e-12 | 41 / 0 / 1 |
| Var-Part | 3.86 | 9.1e-13 | 4.5e-12 | 41 / 0 / 1 |
| CCKM | 9.07 | 4.5e-13 | 4.5e-12 | 42 / 0 / 0 |
| OCKM (published) | 8.26 | 4.5e-13 | 4.5e-12 | 42 / 0 / 0 |
| OCKM-fast | 5.40 | 4.5e-13 | 4.5e-12 | 42 / 0 / 0 |
| **OCKM-T** | **1.05** | | | |

### Total time (seeding + Lloyd)

Friedman chi2 = 311.5, p = 5.8e-61; N = 42 datasets, 11 methods; Nemenyi CD (alpha = 0.05) = 2.33.

| method | mean rank | Wilcoxon p vs OCKM-T | Holm | OCKM-T wins / ties / losses |
|---|---|---|---|---|
| k-means++ (greedy) | 5.10 | 1.4e-03 | 2.9e-03 | 10 / 1 / 31 |
| k-means++ (vanilla) | 4.14 | 7.4e-05 | 3.7e-04 | 8 / 1 / 33 |
| Qkmeans | 1.90 | 1.1e-05 | 8.7e-05 | 3 / 0 / 39 |
| AFK-MC² | 4.79 | 1.7e-04 | 6.9e-04 | 7 / 1 / 34 |
| k-means‖ | 8.67 | 2.9e-05 | 2.1e-04 | 37 / 0 / 5 |
| PCA-Part | 8.10 | 8.9e-04 | 2.7e-03 | 29 / 1 / 12 |
| Var-Part | 1.71 | 3.1e-05 | 2.1e-04 | 4 / 0 / 38 |
| CCKM | 5.79 | 3.5e-01 | 3.5e-01 | 13 / 3 / 26 |
| OCKM (published) | 10.62 | 4.5e-13 | 4.5e-12 | 42 / 0 / 0 |
| OCKM-fast | 8.76 | 2.7e-06 | 2.4e-05 | 35 / 0 / 7 |
| **OCKM-T** | **6.43** | | | |

### Ablation of OCKM-T, final WCSS gap

Friedman chi2 = 45.7, p = 2.7e-07; N = 42 datasets, 9 methods; Nemenyi CD (alpha = 0.05) = 1.85.

| method | mean rank | Wilcoxon p vs OCKM-T | Holm | OCKM-T wins / ties / losses |
|---|---|---|---|---|
| T: PCA cuts only | 5.06 | 3.5e-01 | 1.0e+00 | 16 / 18 / 8 |
| T: no LocalSearch++ | 5.88 | 9.7e-04 | 6.8e-03 | 14 / 26 / 2 |
| T: 2 summary starts | 5.37 | 8.9e-03 | 5.3e-02 | 12 / 27 / 3 |
| T: tree partition only (no summary search) | 6.54 | 5.2e-04 | 4.1e-03 | 15 / 25 / 2 |
| T: summary search without tree start | 4.46 | 4.7e-01 | 1.0e+00 | 4 / 33 / 5 |
| T: gamma = 4 | 4.45 | 6.0e-01 | 1.0e+00 | 5 / 30 / 7 |
| T: gamma = 16 | 4.07 | 1.5e-01 | 5.8e-01 | 5 / 26 / 11 |
| T: no full-space polish | 4.55 | 1.1e-01 | 5.4e-01 | 0 / 42 / 0 |
| **OCKM-T** | **4.62** | | | |

## Equal-time comparison

Budget = total time of an OCKM-T run (seed s). Each randomised method is restarted in seed order from seed s while its
cumulative total time fits the budget (at least one run) and its best WCSS is kept; gaps averaged over s. Gap < 0: OCKM-T better.

| reference | datasets | OCKM-T better / tie / worse | median gap % | mean runs in budget | Wilcoxon p (gap vs 0) | also vs best of all 10 runs: better / tie / worse |
|---|---|---|---|---|---|---|
| k-means++ (greedy) | 42 | 34 / 7 / 1 | -1.743 | 1.1 | 1.9e-07 | 3 / 19 / 20 |
| Qkmeans | 42 | 37 / 4 / 1 | -2.279 | 1.4 | 8.2e-08 | 14 / 14 / 14 |
| k-means‖ | 42 | 38 / 4 / 0 | -5.133 | 1.0 | 4.2e-08 | 9 / 21 / 12 |
| k-means++ (vanilla) | 42 | 39 / 3 / 0 | -3.880 | 1.1 | 2.6e-08 | 10 / 18 / 14 |
| AFK-MC² | 42 | 38 / 4 / 0 | -3.189 | 1.1 | 4.2e-08 | 9 / 18 / 15 |

## Time relative to one k-means++ (greedy) run

| method | median total-time ratio | median seeding-time ratio | datasets with total ratio <= 1 |
|---|---|---|---|
| OCKM-T | 1.30 | 1.88 | 10 / 42 |
| k-means++ (greedy) | 1.00 | 1.00 | 42 / 42 |
| k-means++ (vanilla) | 0.94 | 0.83 | 36 / 42 |
| Qkmeans | 0.76 | 0.34 | 41 / 42 |
| AFK-MC² | 1.00 | 0.98 | 21 / 42 |
| k-means‖ | 1.59 | 2.49 | 2 / 42 |
| PCA-Part | 1.73 | 2.87 | 5 / 42 |
| Var-Part | 0.70 | 0.43 | 40 / 42 |
| CCKM | 1.15 | 1.22 | 16 / 42 |
| OCKM (published) | 2.70 | 1.74 | 0 / 42 |
| OCKM-fast | 1.80 | 2.87 | 1 / 42 |
| T: PCA cuts only | 0.98 | 1.12 | 23 / 42 |
| T: no LocalSearch++ | 1.10 | 1.54 | 13 / 42 |
| T: 2 summary starts | 1.04 | 1.42 | 19 / 42 |
| T: tree partition only (no summary search) | 1.06 | 1.37 | 19 / 42 |
| T: summary search without tree start | 1.69 | 2.84 | 4 / 42 |
| T: gamma = 4 | 1.04 | 1.32 | 18 / 42 |
| T: gamma = 16 | 1.68 | 2.46 | 5 / 42 |
| T: no full-space polish | 1.31 | 1.87 | 10 / 42 |