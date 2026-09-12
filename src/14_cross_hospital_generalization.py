# %% [markdown]
# # Phase 8 — Cross-hospital generalization
#
# Every result so far (baseline, engineered, ablation) uses `GroupKFold`
# splits that are patient-disjoint but hospital-*mixed* — both folds contain
# a blend of hospital_system_1 and hospital_system_2 patients. That answers
# "does this generalize to new patients?" but not "does this generalize to a
# new hospital?", which is the question that actually matters for deployment
# (different charting habits, different lab-ordering thresholds, different
# case mix) and the first thing a clinical-ML reviewer will ask about.
#
# This script trains once on hospital_id=1 (all patients) and evaluates on
# hospital_id=2 (all patients) with zero patient or hospital overlap, then
# repeats it in the other direction for a complete picture. Same feature
# set, same XGBoost hyperparameters, and the same utility-score framing as
# `04_engineered_model.py`, so the numbers are directly comparable to the
# in-distribution GroupKFold result already in `engineered_results.csv`.

# %%
import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utility_score import sweep_thresholds_for_utility

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "warehouse" / "sepsis.duckdb"
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42  # kept consistent with 03/04 for comparability

NON_FEATURE_COLS = {"patient_id", "hospital_id", "hour", "ICULOS", "SepsisLabel"}

# XGBoost hyperparameters copied verbatim from 04_engineered_model.py so any
# gap vs. the in-distribution result is attributable to the train/test split,
# not to a different model configuration.
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
def train_on_a_test_on_b(df, feature_cols, train_hospital, test_hospital, tag):
    train_df = df[df["hospital_id"] == train_hospital]
    test_df = df[df["hospital_id"] == test_hospital]

    train_patients = set(train_df["patient_id"])
    test_patients = set(test_df["patient_id"])
    assert train_patients.isdisjoint(test_patients), (
        "PATIENT LEAKAGE DETECTED — a patient_id appears in both hospitals; "
        "check the hospital_id assignment in dim_patient."
    )

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    y_train = train_df["SepsisLabel"].astype(int).to_numpy()
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    y_test = test_df["SepsisLabel"].astype(int).to_numpy()

    model = XGBClassifier(
        **XGB_PARAMS,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
    )
    model.fit(X_train, y_train)
    proba_test = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, proba_test)
    ap = average_precision_score(y_test, proba_test)
    print(f"  [{tag}] train n={len(train_patients):,} patients / "
          f"test n={len(test_patients):,} patients  "
          f"AUROC={auc:.4f}  AUPRC={ap:.4f}")

    result_df = test_df[["patient_id", "hour", "SepsisLabel"]].copy()
    result_df["cross_hospital_proba"] = proba_test

    thr_table = sweep_thresholds_for_utility(
        result_df, patient_col="patient_id", label_col="SepsisLabel",
        proba_col="cross_hospital_proba",
    )
    best_row = thr_table.iloc[0]
    print(f"  [{tag}] best-threshold normalized utility = "
          f"{best_row.normalized_utility:.4f} at threshold={best_row.threshold:.2f}")

    del model
    return {
        "direction": tag,
        "train_hospital": train_hospital,
        "test_hospital": test_hospital,
        "n_train_patients": len(train_patients),
        "n_test_patients": len(test_patients),
        "auroc": auc,
        "auprc": ap,
        "best_threshold": best_row.threshold,
        "normalized_utility": best_row.normalized_utility,
        "n_features": len(feature_cols),
    }, result_df


# %%
def run_cross_hospital():
    df, feature_cols = load_engineered_frame()
    hospitals = sorted(df["hospital_id"].unique().tolist())
    print(f"Engineered frame: {df.shape[0]:,} rows, {len(feature_cols)} features, "
          f"hospitals={hospitals}")
    if len(hospitals) != 2:
        print(f"WARNING: expected exactly 2 hospital systems, found {hospitals}. "
              f"Proceeding with the first two found.")
    h1, h2 = hospitals[0], hospitals[1]

    print(f"\n--- Direction A: train on hospital {h1}, test on hospital {h2} ---")
    result_a, preds_a = train_on_a_test_on_b(
        df, feature_cols, train_hospital=h1, test_hospital=h2,
        tag=f"train_h{h1}_test_h{h2}",
    )

    print(f"\n--- Direction B: train on hospital {h2}, test on hospital {h1} ---")
    result_b, preds_b = train_on_a_test_on_b(
        df, feature_cols, train_hospital=h2, test_hospital=h1,
        tag=f"train_h{h2}_test_h{h1}",
    )

    results = pd.DataFrame([result_a, result_b])

    # Pull in the in-distribution (mixed-hospital GroupKFold) number for a
    # direct side-by-side comparison, if it's already been produced.
    engineered_path = OUT_DIR / "engineered_results.csv"
    if engineered_path.exists():
        in_dist = pd.read_csv(engineered_path).iloc[0]
        print(f"\nFor comparison — in-distribution (mixed-hospital, patient-grouped "
              f"5-fold) result already on file:")
        print(f"  AUROC={in_dist['auroc']:.4f}  AUPRC={in_dist['auprc']:.4f}  "
              f"utility={in_dist['normalized_utility']:.4f}")
        results["in_dist_auroc"] = in_dist["auroc"]
        results["in_dist_auprc"] = in_dist["auprc"]
        results["in_dist_utility"] = in_dist["normalized_utility"]
        results["auroc_drop_vs_in_dist"] = results["in_dist_auroc"] - results["auroc"]
    else:
        print("\n(No engineered_results.csv found — run 04_engineered_model.py first "
              "for the in-distribution comparison column.)")

    results.to_csv(OUT_DIR / "cross_hospital_results.csv", index=False)
    preds_a.to_parquet(OUT_DIR / f"cross_hospital_preds_train_h{h1}_test_h{h2}.parquet")
    preds_b.to_parquet(OUT_DIR / f"cross_hospital_preds_train_h{h2}_test_h{h1}.parquet")

    print("\nSaved: cross_hospital_results.csv, "
          "cross_hospital_preds_train_h{}_test_h{}.parquet, "
          "cross_hospital_preds_train_h{}_test_h{}.parquet".format(h1, h2, h2, h1))
    return results


# %%
if __name__ == "__main__":
    run_cross_hospital()
