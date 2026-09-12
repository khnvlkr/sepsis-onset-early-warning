# %% [markdown]
# # Phase 5b — Classical classifiers: Decision Tree & Naive Bayes
#
# DWM Module 6 names Decision Tree Induction and Bayesian Classification as
# distinct classification techniques, alongside association rule mining and
# clustering. Scripts 03/04 only cover XGBoost (a boosted-tree ensemble), so
# this script adds the two named classical classifiers as extra rows in the
# same baseline-vs-engineered comparison, reusing the identical `fact_features`
# table, patient-grouped folds, and metrics as 04_engineered_model.py — so the
# numbers are directly comparable to the rest of the results table.
#
# Two variants of each classifier are run: on the raw ffill features only
# (control, matches 03_baseline_model.py's feature set) and on the full
# engineered feature set (matches 04_engineered_model.py's feature set).

# %%
import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, average_precision_score
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utility_score import sweep_thresholds_for_utility

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "warehouse" / "sepsis.duckdb"
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

N_FOLDS = 5
RANDOM_STATE = 42  # matches 03/04 so folds line up for comparison

NON_FEATURE_COLS = {"patient_id", "hospital_id", "hour", "ICULOS", "SepsisLabel"}


# %%
def load_engineered_frame():
    """Same loader as 04_engineered_model.py — full fact_features table,
    cast to float32 at the SQL level to keep memory sane (see README §4)."""
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

    engineered_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    raw_cols = [c for c in engineered_cols if c.endswith("_ffill")]
    return df, raw_cols, engineered_cols


# %%
def train_with_group_folds(df, feature_cols, model_fn, tag):
    """Same fold logic as 03/04, but model-agnostic (takes a zero-arg
    constructor) so it works for DecisionTreeClassifier and GaussianNB too.

    Naive Bayes needs no class-imbalance handling built in (it isn't
    cost-sensitive like XGBoost's scale_pos_weight), so imbalance is left
    as-is for both classifiers here — the comparison is meant to show how
    the *named module-6 techniques* perform out of the box on this task,
    not to out-tune them against the XGBoost model.
    """
    X = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    y = df["SepsisLabel"].astype(int).to_numpy()
    groups = df["patient_id"].to_numpy()

    # Both DecisionTreeClassifier and GaussianNB choke on NaNs; fact_features
    # can contain NaNs (e.g. a vital never measured in a short stay), so
    # impute with a simple per-column median just for these two models.
    # XGBoost handles NaNs natively, which is why 03/04 don't need this step.
    col_medians = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        X = np.where(nan_mask, col_medians, X)

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_proba = np.zeros(len(df))

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        train_patients = set(groups[train_idx])
        test_patients = set(groups[test_idx])
        assert train_patients.isdisjoint(test_patients), "PATIENT LEAKAGE DETECTED"

        model = model_fn()
        model.fit(X[train_idx], y[train_idx])
        oof_proba[test_idx] = model.predict_proba(X[test_idx])[:, 1]
        del model

    auc = roc_auc_score(y, oof_proba)
    ap = average_precision_score(y, oof_proba)
    print(f"  [{tag}] AUROC={auc:.4f}  AUPRC={ap:.4f}  (n_features={len(feature_cols)})")
    return oof_proba, auc, ap


# %%
def run_classical_classifiers():
    df, raw_cols, engineered_cols = load_engineered_frame()
    print(f"Frame: {df.shape[0]:,} rows | raw_ffill features={len(raw_cols)} | "
          f"engineered features={len(engineered_cols)}")

    configs = [
        ("decision_tree_raw", raw_cols,
         lambda: DecisionTreeClassifier(max_depth=6, min_samples_leaf=50,
                                         class_weight="balanced", random_state=RANDOM_STATE)),
        ("decision_tree_engineered", engineered_cols,
         lambda: DecisionTreeClassifier(max_depth=6, min_samples_leaf=50,
                                         class_weight="balanced", random_state=RANDOM_STATE)),
        ("naive_bayes_raw", raw_cols, lambda: GaussianNB()),
        ("naive_bayes_engineered", engineered_cols, lambda: GaussianNB()),
    ]

    results = []
    oof_frames = {}
    for tag, cols, model_fn in configs:
        proba, auc, ap = train_with_group_folds(df, cols, model_fn, tag)
        proba_col = f"{tag}_proba"
        df[proba_col] = proba

        thr_table = sweep_thresholds_for_utility(
            df, patient_col="patient_id", label_col="SepsisLabel", proba_col=proba_col
        )
        best_row = thr_table.iloc[0]
        print(f"    best-threshold normalized utility = {best_row.normalized_utility:.4f} "
              f"at threshold={best_row.threshold:.2f}")

        results.append({
            "model": tag,
            "auroc": auc,
            "auprc": ap,
            "best_threshold": best_row.threshold,
            "normalized_utility": best_row.normalized_utility,
            "n_features": len(cols),
        })
        oof_frames[tag] = df[["patient_id", "hour", "SepsisLabel", proba_col]].copy()

    results_df = pd.DataFrame(results).sort_values("auroc", ascending=False)
    results_df.to_csv(OUT_DIR / "classical_classifier_results.csv", index=False)

    for tag, frame in oof_frames.items():
        frame.to_parquet(OUT_DIR / f"{tag}_oof_predictions.parquet")

    # Fold this comparison into the same table format as engineered_results.csv
    # / baseline_results.csv, so the report's results table can just concat
    # all four CSVs (baseline, engineered, ablation, classical) into one view.
    print("\nClassical classifier results:")
    print(results_df.to_string(index=False))
    print(f"\nSaved: classical_classifier_results.csv, "
          f"{len(configs)} x *_oof_predictions.parquet")
    return results_df


# %%
if __name__ == "__main__":
    run_classical_classifiers()
