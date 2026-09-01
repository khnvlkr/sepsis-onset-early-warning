# %% [markdown]
# # Phase 9 — Uncertainty-aware predictions via split conformal prediction
#
# Every model so far outputs a bare probability, forced onto every patient-
# hour regardless of whether that hour looks like anything the model has
# seen before. This script wraps the engineered XGBoost model in a
# split-conformal classifier (via `mapie`) so that, instead of just a risk
# score, each patient-hour gets a *prediction set* at a chosen confidence
# level (default 90%):
#
#   {no-sepsis}        model is confident: not septic
#   {sepsis}            model is confident: septic
#   {no-sepsis, sepsis}  model is genuinely uncertain — flag for review
#   {}                   (rare/pathological — no label meets the threshold)
#
# The middle case is the clinically interesting one: it's the model saying
# "I don't know" instead of quietly forcing a number, which is the same
# spirit as COMPOSER (Chang et al., PMC8429719) — a published sepsis early-
# warning model that explicitly abstains on out-of-distribution patients
# rather than scoring them with false confidence.
#
# Split conformal only guarantees marginal coverage over *exchangeable*
# data. Hourly ICU rows within one patient are correlated, not exchangeable,
# so coverage is evaluated per-patient-hour as an empirical check rather
# than treated as a proof — but the patient-disjoint train/calibrate/test
# split below is what keeps that approximation honest (no patient's hours
# leak across the three sets).

# %%
import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier
from mapie.classification import SplitConformalClassifier
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utility_score import normalized_utility_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "warehouse" / "sepsis.duckdb"
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
CONFIDENCE_LEVEL = 0.90  # i.e. target ~90% marginal coverage
NON_FEATURE_COLS = {"patient_id", "hospital_id", "hour", "ICULOS", "SepsisLabel"}

XGB_PARAMS = dict(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="aucpr", tree_method="hist",
    random_state=RANDOM_STATE, n_jobs=-1,
)


# %%
def load_engineered_frame():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    schema = con.execute("SELECT * FROM fact_features LIMIT 0").df()
    all_cols = list(schema.columns)
    numeric_cols = [c for c in all_cols if c not in NON_FEATURE_COLS]

    select_parts = []
    for c in all_cols:
        if c in numeric_cols:
            select_parts.append(f'CAST("{c}" AS FLOAT) AS "{c}"')
        else:
            select_parts.append(f'"{c}"')

    query = f"SELECT {', '.join(select_parts)} FROM fact_features"
    df = con.execute(query).df()
    con.close()

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    return df, feature_cols


# %%
def three_way_patient_split(df, random_state=RANDOM_STATE):
    """Patient-disjoint train / calibration / test split (60/20/20).

    A 3-way split rather than reusing GroupKFold OOF predictions: the
    calibration set needs to be genuinely held out from *training*, and the
    test set needs to be held out from both training and calibration, or
    the coverage guarantee is meaningless.
    """
    patient_ids = df["patient_id"].to_numpy()

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.40, random_state=random_state)
    train_idx, rest_idx = next(gss1.split(df, groups=patient_ids))

    rest_df = df.iloc[rest_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=random_state)
    cal_rel_idx, test_rel_idx = next(
        gss2.split(rest_df, groups=rest_df["patient_id"].to_numpy())
    )
    cal_idx = rest_df.index.to_numpy()[cal_rel_idx]
    test_idx = rest_df.index.to_numpy()[test_rel_idx]

    train_p = set(df.loc[train_idx, "patient_id"])
    cal_p = set(df.loc[cal_idx, "patient_id"])
    test_p = set(df.loc[test_idx, "patient_id"])
    assert train_p.isdisjoint(cal_p) and train_p.isdisjoint(test_p) and cal_p.isdisjoint(test_p), \
        "PATIENT LEAKAGE DETECTED across train/calibration/test split"

    return df.loc[train_idx], df.loc[cal_idx], df.loc[test_idx]


# %%
def run_conformal():
    df, feature_cols = load_engineered_frame()
    print(f"Engineered frame: {df.shape[0]:,} rows, {len(feature_cols)} features")

    train_df, cal_df, test_df = three_way_patient_split(df)
    print(f"Split: train={len(set(train_df.patient_id)):,} patients "
          f"({len(train_df):,} rows), "
          f"calibration={len(set(cal_df.patient_id)):,} patients "
          f"({len(cal_df):,} rows), "
          f"test={len(set(test_df.patient_id)):,} patients ({len(test_df):,} rows)")

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    y_train = train_df["SepsisLabel"].astype(int).to_numpy()
    X_cal = cal_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    y_cal = cal_df["SepsisLabel"].astype(int).to_numpy()
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    y_test = test_df["SepsisLabel"].astype(int).to_numpy()

    # --- fit the base XGBoost model on the training split only ---
    print("\nFitting base XGBoost model...")
    base_model = XGBClassifier(
        **XGB_PARAMS,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
    )
    base_model.fit(X_train, y_train)

    # --- wrap it with split-conformal calibration ---
    # 'lac' (least ambiguous set-valued classifier) conformity score is the
    # standard choice; it tends to produce tight singleton sets on the
    # dominant class (no-sepsis, ~98% of hours here) and only opens up to
    # {no-sepsis, sepsis} on genuinely ambiguous hours, which is the
    # behavior we want given the class imbalance.
    print(f"Conformalizing at confidence_level={CONFIDENCE_LEVEL}...")
    conformal_clf = SplitConformalClassifier(
        estimator=base_model,
        confidence_level=CONFIDENCE_LEVEL,
        conformity_score="lac",
        prefit=True,
        random_state=RANDOM_STATE,
    )
    conformal_clf.conformalize(X_cal, y_cal)

    pred_labels, pred_sets = conformal_clf.predict_set(X_test)
    # pred_sets shape: (n_samples, n_classes, n_confidence_levels) -> squeeze
    # to (n_samples, n_classes) since we only requested one confidence level
    pred_sets = pred_sets[:, :, 0].astype(bool)
    classes_ = list(base_model.classes_)
    sepsis_col = classes_.index(1)
    no_sepsis_col = classes_.index(0)

    in_set_sepsis = pred_sets[:, sepsis_col]
    in_set_no_sepsis = pred_sets[:, no_sepsis_col]
    set_size = pred_sets.sum(axis=1)

    is_confident_negative = in_set_no_sepsis & ~in_set_sepsis
    is_confident_positive = in_set_sepsis & ~in_set_no_sepsis
    is_uncertain = in_set_sepsis & in_set_no_sepsis
    is_empty = set_size == 0

    proba_test = base_model.predict_proba(X_test)[:, sepsis_col]

    # --- coverage check: does the true label fall in the predicted set at
    # roughly the target rate? this is the empirical stand-in for the
    # theoretical marginal-coverage guarantee, given non-exchangeable rows ---
    true_in_set = pred_sets[np.arange(len(y_test)), np.where(np.array(classes_) == 1, sepsis_col, no_sepsis_col)[y_test]]
    empirical_coverage = true_in_set.mean()

    print(f"\n--- Coverage & set-size summary (target confidence={CONFIDENCE_LEVEL}) ---")
    print(f"  Empirical coverage (true label in predicted set): {empirical_coverage:.4f}")
    print(f"  Confident no-sepsis:  {is_confident_negative.mean()*100:.2f}% of hours")
    print(f"  Confident sepsis:     {is_confident_positive.mean()*100:.2f}% of hours")
    print(f"  Uncertain ({{both}}):   {is_uncertain.mean()*100:.2f}% of hours")
    print(f"  Empty set:            {is_empty.mean()*100:.2f}% of hours")

    # --- does the model actually earn the right to be confident? accuracy
    # should be meaningfully higher on confident predictions than on the
    # hours it flags as uncertain, or the abstention mechanism isn't adding
    # anything ---
    def accuracy_on(mask, thresholded_pred):
        if mask.sum() == 0:
            return np.nan
        return (thresholded_pred[mask] == y_test[mask]).mean()

    default_pred = (proba_test >= 0.5).astype(int)
    acc_confident = accuracy_on(is_confident_negative | is_confident_positive, default_pred)
    acc_uncertain = accuracy_on(is_uncertain, default_pred)
    print(f"\n  Accuracy on confident hours: {acc_confident:.4f}")
    print(f"  Accuracy on uncertain hours: {acc_uncertain:.4f}  "
          f"(should be lower — that's the point of flagging them)")

    # --- clinical utility restricted to confident predictions only, vs. the
    # full-cohort utility, to show what's gained by deferring on the
    # uncertain slice instead of forcing a threshold call on everyone ---
    test_df = test_df.copy()
    test_df["conformal_proba"] = proba_test
    test_df["is_confident"] = (is_confident_negative | is_confident_positive)
    test_df["is_uncertain"] = is_uncertain
    test_df["set_size"] = set_size

    full_utility = normalized_utility_score(
        test_df, patient_col="patient_id", label_col="SepsisLabel",
        proba_col="conformal_proba", threshold=0.5,
    )
    confident_only = test_df[test_df["is_confident"]]
    if len(confident_only) > 0 and confident_only["SepsisLabel"].nunique() > 1:
        confident_utility = normalized_utility_score(
            confident_only, patient_col="patient_id", label_col="SepsisLabel",
            proba_col="conformal_proba", threshold=0.5,
        )
    else:
        confident_utility = float("nan")
    print(f"\n  Normalized utility, full test cohort:      {full_utility:.4f}")
    print(f"  Normalized utility, confident hours only:  {confident_utility:.4f}")

    # --- save everything ---
    summary = pd.DataFrame([{
        "confidence_level": CONFIDENCE_LEVEL,
        "n_test_rows": len(test_df),
        "n_test_patients": test_df["patient_id"].nunique(),
        "empirical_coverage": empirical_coverage,
        "pct_confident_no_sepsis": is_confident_negative.mean(),
        "pct_confident_sepsis": is_confident_positive.mean(),
        "pct_uncertain_both_labels": is_uncertain.mean(),
        "pct_empty_set": is_empty.mean(),
        "accuracy_confident_hours": acc_confident,
        "accuracy_uncertain_hours": acc_uncertain,
        "utility_full_cohort": full_utility,
        "utility_confident_only": confident_utility,
    }])
    summary.to_csv(OUT_DIR / "conformal_prediction_summary.csv", index=False)

    test_df[["patient_id", "hour", "SepsisLabel", "conformal_proba",
              "is_confident", "is_uncertain", "set_size"]].to_parquet(
        OUT_DIR / "conformal_prediction_predictions.parquet"
    )

    print("\nSaved: conformal_prediction_summary.csv, "
          "conformal_prediction_predictions.parquet")
    return summary


# %%
if __name__ == "__main__":
    run_conformal()
