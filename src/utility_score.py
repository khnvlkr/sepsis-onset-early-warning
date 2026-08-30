"""
Reimplementation of the official PhysioNet/CinC 2019 Challenge utility score.

Verified against the reference implementation at
https://github.com/physionetchallenges/evaluation-2019/blob/master/evaluate_sepsis_score.py
(compute_prediction_utility, lines ~412-472 as of the referenced commit).

This version is vectorized with numpy per-patient (the original loops hour by
hour in pure Python), which matters once you're scoring ~40k patients.

Usage:
    from utility_score import normalized_utility_score
    score = normalized_utility_score(
        df, patient_col="patient_id", label_col="SepsisLabel",
        proba_col="pred_proba", threshold=0.5,
    )
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class UtilityParams:
    dt_early: float = -12
    dt_optimal: float = -6
    dt_late: float = 3
    max_u_tp: float = 1
    min_u_fn: float = -2
    u_fp: float = -0.05
    u_tn: float = 0


def compute_prediction_utility(labels: np.ndarray, predictions: np.ndarray,
                                p: UtilityParams = UtilityParams()) -> float:
    """Utility for ONE patient's full hourly sequence. labels/predictions are
    same-length 0/1 arrays for that patient, in chronological hour order."""
    labels = np.asarray(labels)
    predictions = np.asarray(predictions).astype(bool)
    n = len(labels)

    if np.any(labels):
        is_septic = True
        t_sepsis = np.argmax(labels) - p.dt_optimal
    else:
        is_septic = False
        t_sepsis = np.inf

    m_1 = float(p.max_u_tp) / float(p.dt_optimal - p.dt_early)
    b_1 = -m_1 * p.dt_early
    m_2 = float(-p.max_u_tp) / float(p.dt_late - p.dt_optimal)
    b_2 = -m_2 * p.dt_late
    m_3 = float(p.min_u_fn) / float(p.dt_late - p.dt_optimal)
    b_3 = -m_3 * p.dt_optimal

    t = np.arange(n, dtype=float)
    u = np.zeros(n)

    if is_septic:
        within = t <= t_sepsis + p.dt_late
        tp_mask = within & predictions
        fn_mask = within & ~predictions

        tp_early = tp_mask & (t <= t_sepsis + p.dt_optimal)
        tp_late = tp_mask & ~tp_early
        u[tp_early] = np.maximum(m_1 * (t[tp_early] - t_sepsis) + b_1, p.u_fp)
        u[tp_late] = m_2 * (t[tp_late] - t_sepsis) + b_2

        fn_early = fn_mask & (t <= t_sepsis + p.dt_optimal)
        fn_late = fn_mask & ~fn_early
        u[fn_early] = 0.0
        u[fn_late] = m_3 * (t[fn_late] - t_sepsis) + b_3
        # t > t_sepsis + dt_late already 0 (np.zeros default)
    else:
        u[predictions] = p.u_fp
        u[~predictions] = p.u_tn

    return float(u.sum())


def best_worst_inaction_predictions(labels: np.ndarray, p: UtilityParams = UtilityParams()):
    n = len(labels)
    labels = np.asarray(labels)
    best = np.zeros(n, dtype=int)
    if np.any(labels):
        t_sepsis = int(np.argmax(labels) - p.dt_optimal)
        lo = max(0, t_sepsis + int(p.dt_early))
        hi = min(t_sepsis + int(p.dt_late) + 1, n)
        if hi > lo:
            best[lo:hi] = 1
    worst = 1 - best
    inaction = np.zeros(n, dtype=int)
    return best, worst, inaction


def normalized_utility_score(df: pd.DataFrame, patient_col: str, label_col: str,
                              proba_col: str = None, pred_col: str = None,
                              threshold: float = 0.5,
                              params: UtilityParams = UtilityParams()) -> float:
    """
    Normalized so a perfect ('best possible') set of predictions scores 1.0
    and doing nothing (all-negative) scores 0.0. Can go negative if the
    classifier is worse than doing nothing.

    Pass either `proba_col` (+ `threshold`) or a pre-binarized `pred_col`.
    """
    assert (proba_col is not None) or (pred_col is not None), \
        "provide proba_col+threshold or pred_col"

    obs_sum = best_sum = worst_sum = inaction_sum = 0.0

    for _, g in df.sort_values([patient_col, "hour"]).groupby(patient_col, sort=False):
        labels = g[label_col].to_numpy()
        if pred_col is not None:
            preds = g[pred_col].to_numpy().astype(int)
        else:
            preds = (g[proba_col].to_numpy() >= threshold).astype(int)

        obs_sum += compute_prediction_utility(labels, preds, params)

        best, worst, inaction = best_worst_inaction_predictions(labels, params)
        best_sum += compute_prediction_utility(labels, best, params)
        worst_sum += compute_prediction_utility(labels, worst, params)
        inaction_sum += compute_prediction_utility(labels, inaction, params)

    denom = (best_sum - inaction_sum)
    if denom == 0:
        return float("nan")
    return (obs_sum - inaction_sum) / denom


def sweep_thresholds_for_utility(df: pd.DataFrame, patient_col: str, label_col: str,
                                  proba_col: str, thresholds=None,
                                  params: UtilityParams = UtilityParams()) -> pd.DataFrame:
    """Grid-search the decision threshold that maximizes normalized utility.
    Unlike AUC, utility DOES depend on where you binarize, so this matters."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 25)
    rows = []
    for thr in thresholds:
        score = normalized_utility_score(
            df, patient_col, label_col, proba_col=proba_col, threshold=thr, params=params
        )
        rows.append({"threshold": thr, "normalized_utility": score})
    return pd.DataFrame(rows).sort_values("normalized_utility", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    # Tiny sanity check against a hand-worked example.
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1])  # onset (label first=1) at t=5
    perfect_preds = np.array([0, 0, 0, 0, 0, 1, 1, 1])
    all_zero_preds = np.zeros(8, dtype=int)

    p = UtilityParams()
    u_perfect = compute_prediction_utility(labels, perfect_preds, p)
    u_inaction = compute_prediction_utility(labels, all_zero_preds, p)
    print(f"utility(predict=labels)   = {u_perfect:.3f}")
    print(f"utility(never predict)    = {u_inaction:.3f}")
    assert u_perfect > u_inaction, "predicting on time should beat doing nothing"
    print("Sanity check passed.")
