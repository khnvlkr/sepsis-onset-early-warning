# %% [markdown]
# # Phase 3 — Baseline model: raw snapshot values only, 6h-ahead label
#
# No engineered features here on purpose — this is the number the engineered
# model (Phase 5) has to beat. Uses raw vitals+labs *forward-filled* only
# (no rolling stats, no ratios, no slopes), predicting `SepsisLabel` shifted
# so a positive at hour t means "sepsis onset within the next 6 hours" — which
# is already how the Challenge's SepsisLabel is defined (flips to 1 six hours
# before clinical onset), so no extra shifting is actually needed; we use it
# as-is and call this out explicitly in the report.

# %%
import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utility_score import normalized_utility_score, sweep_thresholds_for_utility

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "warehouse" / "sepsis.duckdb"
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

N_FOLDS = 5
RANDOM_STATE = 42

# %%
def load_baseline_frame():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    # Raw + forward-filled snapshot only (from fact_ffill built in script 02)
    df = con.execute("SELECT * FROM fact_ffill").df()
    con.close()
    ffill_cols = [c for c in df.columns if c.endswith("_ffill")]
    feature_cols = ffill_cols
    return df, feature_cols

# %%
def run_baseline():
    df, feature_cols = load_baseline_frame()
    print(f"Baseline frame: {df.shape[0]:,} rows, {len(feature_cols)} raw features")
    print("Positive rate:", df["SepsisLabel"].mean())

    X = df[feature_cols]
    y = df["SepsisLabel"].astype(int)
    groups = df["patient_id"]

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_proba = np.zeros(len(df))
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        # Guard: assert no patient leakage across the split
        train_patients = set(groups.iloc[train_idx])
        test_patients = set(groups.iloc[test_idx])
        assert train_patients.isdisjoint(test_patients), "PATIENT LEAKAGE DETECTED"

        model = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="aucpr", tree_method="hist",
            scale_pos_weight=(y.iloc[train_idx] == 0).sum() / max((y.iloc[train_idx] == 1).sum(), 1),
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = model.predict_proba(X.iloc[test_idx])[:, 1]
        oof_proba[test_idx] = proba

        auc = roc_auc_score(y.iloc[test_idx], proba)
        ap = average_precision_score(y.iloc[test_idx], proba)
        fold_metrics.append({"fold": fold, "auc": auc, "ap": ap})
        print(f"  fold {fold}: AUROC={auc:.4f}  AUPRC={ap:.4f}  "
              f"(train={len(train_patients):,} pts / test={len(test_patients):,} pts)")

    df["baseline_proba"] = oof_proba

    overall_auc = roc_auc_score(y, oof_proba)
    overall_ap = average_precision_score(y, oof_proba)
    print(f"\nOverall OOF AUROC={overall_auc:.4f}  AUPRC={overall_ap:.4f}")

    thr_table = sweep_thresholds_for_utility(
        df, patient_col="patient_id", label_col="SepsisLabel", proba_col="baseline_proba"
    )
    best_row = thr_table.iloc[0]
    print(f"Best-threshold normalized utility = {best_row.normalized_utility:.4f} "
          f"at threshold={best_row.threshold:.2f}")

    results = pd.DataFrame([{
        "model": "baseline_raw_snapshot",
        "auroc": overall_auc,
        "auprc": overall_ap,
        "best_threshold": best_row.threshold,
        "normalized_utility": best_row.normalized_utility,
        "n_features": len(feature_cols),
    }])
    results.to_csv(OUT_DIR / "baseline_results.csv", index=False)
    df[["patient_id", "hour", "SepsisLabel", "baseline_proba"]].to_parquet(
        OUT_DIR / "baseline_oof_predictions.parquet"
    )
    print(f"\nSaved: {OUT_DIR/'baseline_results.csv'}, {OUT_DIR/'baseline_oof_predictions.parquet'}")
    return results

# %%
if __name__ == "__main__":
    run_baseline()
