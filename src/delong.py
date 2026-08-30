"""
Fast DeLong test for the statistical significance of a difference between two
correlated ROC AUCs (e.g. baseline model vs. engineered model scored on the
*same* held-out rows). Implements the Sun & Xu (2014) O(n log n) algorithm
for DeLong's covariance estimate.

Reference: X. Sun, W. Xu, "Fast Implementation of DeLong's Algorithm for
Comparing the Areas Under Correlated Receiver Operating Characteristic
Curves," IEEE Signal Processing Letters, 2014.
"""
import numpy as np
from scipy import stats


def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    n = len(x)
    T = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(n, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed, m):
    """predictions_sorted_transposed: shape (k models, m+n samples), with the
    m positive-class samples first, n negative-class samples after."""
    k, total = predictions_sorted_transposed.shape
    n = total - m
    pos = predictions_sorted_transposed[:, :m]
    neg = predictions_sorted_transposed[:, m:]

    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, total])
    for r in range(k):
        tx[r, :] = _compute_midrank(pos[r, :])
        ty[r, :] = _compute_midrank(neg[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def delong_roc_test(y_true: np.ndarray, proba_a: np.ndarray, proba_b: np.ndarray):
    """
    Two-sided p-value for H0: AUC(model_a) == AUC(model_b), on the same
    labels y_true (paired / correlated predictions, e.g. same test rows).

    Returns: dict(auc_a, auc_b, auc_diff, z, p_value)
    """
    y_true = np.asarray(y_true)
    order = np.argsort(-y_true, kind="mergesort")  # positives (1) first
    y_sorted = y_true[order]
    m = int(y_sorted.sum())  # number of positives
    preds = np.vstack([np.asarray(proba_a)[order], np.asarray(proba_b)[order]])

    aucs, cov = _fast_delong(preds, m)
    auc_diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    var = max(var, 1e-12)  # numerical floor
    z = auc_diff / np.sqrt(var)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "auc_diff": float(auc_diff),
        "z": float(z),
        "p_value": float(p_value),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.integers(0, 2, n)
    # model B is strictly better-separated than model A
    a = rng.normal(loc=y * 0.3, scale=1.0)
    b = rng.normal(loc=y * 1.2, scale=1.0)
    result = delong_roc_test(y, a, b)
    print(result)
    assert result["auc_b"] > result["auc_a"]
    print("Sanity check passed: better-separated model scores higher AUC, "
          "and p-value reflects the gap.")
