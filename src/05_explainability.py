# %% [markdown]
# # Phase 6 — Explainability: SHAP on the full engineered model
#
# Trains one final model on ALL engineered rows (no held-out fold needed here
# — this is for interpretation, not evaluation, so use everything) and
# produces: a summary plot, dependence plots for the two features you should
# sanity-check clinically (Shock Index, HR rolling slope), and a couple of
# individual-patient waterfall case studies.

# %%
import duckdb
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "warehouse" / "sepsis.duckdb"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

NON_FEATURE_COLS = {"patient_id", "hospital_id", "hour", "ICULOS", "SepsisLabel"}
RANDOM_STATE = 42

# %%
def load_data():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("SELECT * FROM fact_features").df()
    con.close()
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    return df, feature_cols

# %%
def fit_final_model(df, feature_cols):
    X = df[feature_cols]
    y = df["SepsisLabel"].astype(int)
    model = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="aucpr", tree_method="hist",
        scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X, y)
    return model

# %%
def run_shap_analysis(sample_size=20000):
    df, feature_cols = load_data()
    model = fit_final_model(df, feature_cols)

    # SHAP on a random sample -- full 1M+ rows is unnecessary and slow.
    # Force plain float64/np.nan (not pandas nullable dtypes / pd.NA), which
    # some shap plotting internals choke on.
    sample = df.sample(n=min(sample_size, len(df)), random_state=RANDOM_STATE)
    X_sample = sample[feature_cols].astype("float64")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # --- summary plot ---
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_summary.png", dpi=150)
    plt.close()

    # --- dependence plots: sanity-check clinically meaningful directions ---
    for feat in ["shock_index", "HR_slope_6h", "partial_sirs_score"]:
        if feat in X_sample.columns:
            plt.figure()
            shap.dependence_plot(feat, shap_values, X_sample, show=False)
            plt.tight_layout()
            plt.savefig(FIG_DIR / f"shap_dependence_{feat}.png", dpi=150)
            plt.close()

    # --- global importance table ---
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs_shap}) \
        .sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(PROJECT_ROOT / "outputs" / "shap_feature_importance.csv", index=False)
    print("Top 15 features by mean |SHAP|:")
    print(importance.head(15).to_string(index=False))

    # --- individual case studies: one true positive, one true negative near onset ---
    septic_rows = sample[(sample["SepsisLabel"] == 1)]
    if len(septic_rows) > 0:
        case_idx = septic_rows.index[0]
        pos_in_sample = X_sample.index.get_loc(case_idx)
        plt.figure()
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[pos_in_sample],
                base_values=explainer.expected_value,
                data=X_sample.iloc[pos_in_sample],
                feature_names=feature_cols,
            ),
            show=False,
        )
        plt.tight_layout()
        plt.savefig(FIG_DIR / "shap_waterfall_case_positive.png", dpi=150)
        plt.close()
        print(f"\nSaved case study for patient {df.loc[case_idx, 'patient_id']} "
              f"at hour {df.loc[case_idx, 'hour']} (SepsisLabel=1)")

    print(f"\nFigures saved to {FIG_DIR}")

# %%
if __name__ == "__main__":
    run_shap_analysis()
