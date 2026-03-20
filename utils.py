"""
spatial_efficiency/utils.py

Statistical utilities for spatial efficiency analysis.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.stats import wilcoxon, spearmanr
from sklearn.metrics import roc_auc_score


def cohens_d_wilcoxon(
    a: np.ndarray,
    b: np.ndarray,
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """
    Cohen's d and Wilcoxon signed-rank p-value for paired data.

    Parameters
    ----------
    a, b : ndarray
        Paired observations (subject-level means).
    alternative : str
        'two-sided', 'greater', or 'less'.

    Returns
    -------
    d : float — Cohen's d from paired differences
    p : float — Wilcoxon p-value
    """
    a, b = np.array(a), np.array(b)
    diff = a - b
    try:
        _, p = wilcoxon(a, b, alternative=alternative)
    except Exception:
        p = 1.0
    d = diff.mean() / (diff.std() + 1e-10)
    return float(d), float(p)


def compute_auc(
    group1: np.ndarray,
    group2: np.ndarray,
    group2_is_higher: bool = True,
) -> float:
    """
    Compute AUC for binary classification of two groups.

    Parameters
    ----------
    group1 : ndarray — scores for class 0
    group2 : ndarray — scores for class 1
    group2_is_higher : bool
        If True, higher scores indicate group2 (e.g. sedated).
        If False, lower scores indicate group2 (e.g. η falls with sedation).

    Returns
    -------
    auc : float in [0, 1]
    """
    y_true  = np.array([0] * len(group1) + [1] * len(group2))
    scores  = np.concatenate([group1, group2])
    if not group2_is_higher:
        scores = -scores
    return float(roc_auc_score(y_true, scores))


def monotonicity_r(
    values_by_level: Dict[int, np.ndarray],
) -> Tuple[float, float]:
    """
    Compute mean Spearman r between metric and ordinal level.

    Parameters
    ----------
    values_by_level : dict {level_int: ndarray of subject values}

    Returns
    -------
    mean_r : float
    std_r  : float
    """
    levels  = sorted(values_by_level.keys())
    n_subs  = len(values_by_level[levels[0]])
    r_vals  = []

    for sub_i in range(n_subs):
        sub_vals   = [values_by_level[l][sub_i] for l in levels]
        r, _       = spearmanr(levels, sub_vals)
        r_vals.append(r)

    return float(np.mean(r_vals)), float(np.std(r_vals))


def subject_level_means(
    epoch_results: List[Dict],
    stage_labels: List[str],
    metric: str = "eta",
) -> Dict[str, np.ndarray]:
    """
    Compute per-stage means from epoch-level results.

    Parameters
    ----------
    epoch_results : list of dict — output of compute_epochs()
    stage_labels  : list of str — stage label per epoch
    metric        : str — key to extract from each dict

    Returns
    -------
    means_by_stage : dict {stage: ndarray of epoch values}
    """
    from collections import defaultdict
    by_stage = defaultdict(list)
    for r, stage in zip(epoch_results, stage_labels):
        by_stage[stage].append(r[metric])
    return {s: np.array(v) for s, v in by_stage.items()}
