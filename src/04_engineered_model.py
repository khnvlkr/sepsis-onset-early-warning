# %% [markdown]
# # Phase 5 — Engineered-feature model vs. baseline
#
# Same patient-grouped folds as the baseline (rebuilt identically via the
# same GroupKFold call + random_state, so the comparison is apples-to-apples)
# but trained on the full `fact_features` table: rolling stats, slopes,
# missingness-as-signal, clinical ratios.

# %%
import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import sys
import gc 

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utility_score import normalized_utility_score, sweep_thresholds_for_utility
from delong import delong_roc_test

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "warehouse" / "sepsis.duckdb"
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

N_FOLDS = 5
RANDOM_STATE = 42  # must match 03_baseline_model.py so folds line up

NON_FEATURE_COLS = {"patient_id", "hospital_id", "hour", "ICULOS", "SepsisLabel"}

# %%
def load_engineered_frame():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    
    # Get column names/types first, without loading data
    schema = con.execute("SELECT * FROM fact_features LIMIT 0").df()
    all_cols = list(schema.columns)
    
    non_feature = NON_FEATURE_COLS
    numeric_cols = [c for c in all_cols if c not in non_feature]
    
    # Build a query that casts every numeric feature column to FLOAT (32-bit)
    # at the DuckDB level, so pandas never materializes float64 for these.
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
# %% [markdown]
# Feature "families" for the ablation study in Phase 6 — kept here since the
# grouping logic depends on the exact column-naming convention from script 02.

# %%
def feature_families(feature_cols):
    families = {
        "raw_ffill": [c for c in feature_cols if c.endswith("_ffill")],
        "rolling_stats": [c for c in feature_cols if any(
            tag in c for tag in ("_mean_", "_std_", "_min_", "_max_"))],
        "slopes_velocity": [c for c in feature_cols if "_slope_" in c or "_velocity_" in c],
        "missingness": [c for c in feature_cols if c.endswith("_missing") or c.endswith("_hours_since_last")],
        "clinical_ratios": [c for c in feature_cols if c in (
            "shock_index", "pulse_pressure", "partial_sirs_score", "partial_qsofa_score")],
    }
    covered = set().union(*families.values())
    leftover = [c for c in feature_cols if c not in covered]
    if leftover:
        families["other"] = leftover
    return families

# %%
def train_with_group_folds(df, feature_cols, tag):
    X = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    y = df["SepsisLabel"].astype(int).to_numpy()
    groups = df["patient_id"].to_numpy()

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_proba = np.zeros(len(df))

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        train_patients = set(groups[train_idx])
        test_patients = set(groups[test_idx])
        assert train_patients.isdisjoint(test_patients), "PATIENT LEAKAGE DETECTED"

        model = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="aucpr", tree_method="hist",
            scale_pos_weight=(y[train_idx] == 0).sum() / max((y[train_idx] == 1).sum(), 1),
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        model.fit(X[train_idx], y[train_idx])
        oof_proba[test_idx] = model.predict_proba(X[test_idx])[:, 1]
        del model  # free booster memory before next fold

    auc = roc_auc_score(y, oof_proba)
    ap = average_precision_score(y, oof_proba)
    print(f"  [{tag}] AUROC={auc:.4f}  AUPRC={ap:.4f}  (n_features={len(feature_cols)})")
    return oof_proba, auc, ap

# %%
def run_engineered():
    df, feature_cols = load_engineered_frame()
    print(f"Engineered frame: {df.shape[0]:,} rows, {len(feature_cols)} features")

    # --- full engineered model ---
    proba_full, auc_full, ap_full = train_with_group_folds(df, feature_cols, "full_engineered")
    df["engineered_proba"] = proba_full

    thr_table = sweep_thresholds_for_utility(
        df, patient_col="patient_id", label_col="SepsisLabel", proba_col="engineered_proba"
    )
    best_row = thr_table.iloc[0]
    print(f"Best-threshold normalized utility = {best_row.normalized_utility:.4f} "
          f"at threshold={best_row.threshold:.2f}")

    # --- compare against baseline via DeLong's test ---
    baseline_path = OUT_DIR / "baseline_oof_predictions.parquet"
    comparison_row = {}
    if baseline_path.exists():
        base = pd.read_parquet(baseline_path)
        merged = df[["patient_id", "hour", "SepsisLabel", "engineered_proba"]].merge(
            base[["patient_id", "hour", "baseline_proba"]], on=["patient_id", "hour"], how="inner"
        )
        test_result = delong_roc_test(
            merged["SepsisLabel"].to_numpy(),
            merged["baseline_proba"].to_numpy(),
            merged["engineered_proba"].to_numpy(),
        )
        print(f"\nDeLong's test (baseline vs engineered): "
              f"AUC {test_result['auc_a']:.4f} -> {test_result['auc_b']:.4f}, "
              f"z={test_result['z']:.2f}, p={test_result['p_value']:.2e}")
        comparison_row = test_result
    else:
        print("\n(No baseline_oof_predictions.parquet found — run 03_baseline_model.py "
              "first to get the DeLong comparison.)")

    # --- ablation: train on one feature family at a time ---
    print("\nAblation study (single feature family + demographics each time):")
    families = feature_families(feature_cols)
    ablation_rows = []
    for name, cols in families.items():
        if not cols:
            continue
        _, fam_auc, fam_ap = train_with_group_folds(df, cols, f"ablation:{name}")
        ablation_rows.append({"feature_family": name, "n_features": len(cols),
                               "auroc": fam_auc, "auprc": fam_ap})
        gc.collect()
    ablation_df = pd.DataFrame(ablation_rows).sort_values("auroc", ascending=False)

    # --- save everything ---
    results = pd.DataFrame([{
        "model": "engineered_full",
        "auroc": auc_full,
        "auprc": ap_full,
        "best_threshold": best_row.threshold,
        "normalized_utility": best_row.normalized_utility,
        "n_features": len(feature_cols),
        **comparison_row,
    }])
    results.to_csv(OUT_DIR / "engineered_results.csv", index=False)
    ablation_df.to_csv(OUT_DIR / "ablation_results.csv", index=False)
    df[["patient_id", "hour", "SepsisLabel", "engineered_proba"]].to_parquet(
        OUT_DIR / "engineered_oof_predictions.parquet"
    )
    print(f"\nSaved: engineered_results.csv, ablation_results.csv, engineered_oof_predictions.parquet")
    return results, ablation_df

# %%
if __name__ == "__main__":
    run_engineered()
