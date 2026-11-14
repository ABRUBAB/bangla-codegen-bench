"""Statistics used in the paper: task-cluster bootstrap, paired tests, Holm correction, effect sizes.

Design notes
* The unit of independence is the TASK (all variants/samples/models of a task share its difficulty), so every
  confidence interval resamples tasks with replacement (cluster bootstrap) and every paired test is paired by task.
* Per-task outcome for a (model, variant, condition) cell is the mean over its samples (3 per prompt).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy import stats as sps


def task_matrix(df, value: str, rows: str = "task_id", cols: Sequence[str] = ("model_id", "variant", "condition")):
    """Pivot to a (n_tasks x n_cells) matrix of per-task means. Returns (matrix, task_ids, cell_keys)."""
    g = df.groupby([rows, *cols])[value].mean().unstack(list(cols))
    return g.values.astype(float), list(g.index), list(g.columns)


def cluster_bootstrap_mean(per_task: np.ndarray, n_boot: int = 2000, seed: int = 42, alpha: float = 0.05):
    """per_task: 1-D array of per-task means (NaNs dropped). Returns (mean, lo, hi)."""
    x = per_task[~np.isnan(per_task)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def cluster_bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 42, alpha: float = 0.05):
    """Paired per-task arrays (same tasks, same order). Mean difference a-b with percentile CI."""
    m = ~(np.isnan(a) | np.isnan(b))
    d = (a - b)[m]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    bm = d[idx].mean(1)
    return float(d.mean()), float(np.quantile(bm, alpha / 2)), float(np.quantile(bm, 1 - alpha / 2))


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Two-sided Wilcoxon signed-rank on per-task differences; returns (statistic, p). p=1 if all differences are 0."""
    m = ~(np.isnan(a) | np.isnan(b))
    d = (a - b)[m]
    if np.allclose(d, 0):
        return 0.0, 1.0
    try:
        res = sps.wilcoxon(d, zero_method="wilcox", alternative="two-sided", method="auto")
        return float(res.statistic), float(res.pvalue)
    except ValueError:
        return 0.0, 1.0


def mcnemar_exact(a_bin: np.ndarray, b_bin: np.ndarray) -> Tuple[int, int, float]:
    """Exact McNemar on paired binary task outcomes. Returns (n10, n01, p)."""
    m = ~(np.isnan(a_bin) | np.isnan(b_bin))
    a, b = a_bin[m] > 0.5, b_bin[m] > 0.5
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    n = n10 + n01
    if n == 0:
        return n10, n01, 1.0
    return n10, n01, float(sps.binomtest(min(n10, n01), n, 0.5).pvalue)


def holm(pvals: Sequence[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values (monotone), same order as input."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def odds_ratio_paired(n10: int, n01: int) -> float:
    """Conditional (Mantel-Haenszel-style) paired odds ratio with 0.5 continuity correction."""
    return (n10 + 0.5) / (n01 + 0.5)


def cohens_kappa(x: Sequence, y: Sequence) -> float:
    x, y = np.asarray(x), np.asarray(y)
    cats = np.unique(np.concatenate([x, y]))
    po = float((x == y).mean())
    pe = float(sum((x == c).mean() * (y == c).mean() for c in cats))
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h
