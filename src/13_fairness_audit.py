# %% [markdown]
# # Phase 8 — Fairness / bias audit
#
# Not a syllabus requirement — added because it's the single most important
# question a clinical ML model needs to be able to answer honestly before
# anyone would consider deploying it: does it work equally well across
# patient subgroups, or does the pooled AUROC hide meaningfully worse
# performance for some slice of the population?
#
# This reuses the out-of-fold predictions already saved by
# 04_engineered_model.py (engineered_oof_predictions.parquet) and
# 03_baseline_model.py (baseline_oof_predictions.parquet) — no retraining
# needed, since OOF predictions already represent "this patient's hours,
# scored by a model that never saw this patient during training" for every
# patient in the dataset. Those predictions are joined against dim_patient
# (age, gender, hospital_id) and AUROC / AUPRC / normalized utility are
# recomputed per subgroup, using the exact same metric functions as every
# other results table in this project (utility_score.py), so the numbers
# are directly comparable to the pooled results in engineered_results.csv.
#
# Three subgroup axes, matching what's actually in dim_patient:
#   - Age bracket   (PhysioNet Age is a static per-patient field; ages 90+
#                     are capped at ~90 in the source data for de-identification,
#                     which is disclosed here since it affects the oldest bracket)
#   - Gender        (0/1 in the raw PhysioNet files, per the Challenge's own
#                     data dictionary)
#   - hospital_id   (the two "hospital systems" already used throughout this
#                     project, e.g. in the OLAP drill-down in 07_olap_and_export.py)
#
# Reported honestly either way: if gaps are small, that's a genuinely good
# finding worth stating plainly; if they're large, that's the more important
# finding to surface rather than bury.

# %%
import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utility_score import normalized_utility_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "warehouse" / "sepsis.duckdb"
OUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MIN_GROUP_PATIENTS = 200  # below this, AUROC is too noisy to report per-group
AGE_BINS = [0, 40, 60, 75, 200]
AGE_LABELS = ["<40", "40-59", "60-74", "75+"]


# %%
def load_predictions_with_demographics(model_tag):
    """model_tag is 'baseline' or 'engineered' -- picks the matching
    *_oof_predictions.parquet and proba column name."""
    proba_col = f"{model_tag}_proba"
    pred_path = OUT_DIR / f"{model_tag}_oof_predictions.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"{pred_path} not found -- run 0{'3' if model_tag=='baseline' else '4'}_"
            f"{'baseline' if model_tag=='baseline' else 'engineered'}_model.py first."
        )
    preds = pd.read_parquet(pred_path)

    con = duckdb.connect(str(DB_PATH), read_only=True)
    demo = con.execute("""
        SELECT patient_id, hospital_id, age, gender
        FROM dim_patient
    """).df()
    con.close()

    df = preds.merge(demo, on="patient_id", how="left")
    missing_demo = df["age"].isna().sum()
    if missing_demo:
        print(f"  Note: {missing_demo:,} rows had no matching dim_patient demographics "
              f"(dropped from the fairness breakdown, kept in the pooled results elsewhere).")
    df = df.dropna(subset=["age", "gender", "hospital_id"])

    df["age_bracket"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    df["gender_label"] = df["gender"].map({0: "Female", 1: "Male"}).fillna("Unknown")
    df["hospital_label"] = "hospital_system_" + df["hospital_id"].astype(int).astype(str)

    return df, proba_col


# %%
def metrics_for_group(g, proba_col):
    y = g["SepsisLabel"].to_numpy()
    p = g[proba_col].to_numpy()
    n_patients = g["patient_id"].nunique()
    n_positive_patients = g.loc[g["SepsisLabel"] == 1, "patient_id"].nunique()

    if n_patients < MIN_GROUP_PATIENTS or y.sum() == 0:
        return {
            "n_patients": n_patients, "n_positive_patients": n_positive_patients,
            "n_hours": len(g), "auroc": np.nan, "auprc": np.nan,
            "normalized_utility": np.nan,
            "note": f"skipped (n_patients={n_patients} < {MIN_GROUP_PATIENTS} or no positives)",
        }

    auc = roc_auc_score(y, p)
    ap = average_precision_score(y, p)
    utility = normalized_utility_score(
        g, patient_col="patient_id", label_col="SepsisLabel",
        proba_col=proba_col, threshold=0.5,
    )
    return {
        "n_patients": n_patients, "n_positive_patients": n_positive_patients,
        "n_hours": len(g), "auroc": auc, "auprc": ap,
        "normalized_utility": utility, "note": "",
    }


# %%
def audit_by_column(df, proba_col, group_col, tag):
    rows = []
    for group_val, g in df.groupby(group_col, observed=True):
        row = metrics_for_group(g, proba_col)
        row[group_col] = group_val
        rows.append(row)
    result = pd.DataFrame(rows).set_index(group_col)
    cols = ["n_patients", "n_positive_patients", "n_hours", "auroc", "auprc",
            "normalized_utility", "note"]
    result = result[cols]

    print(f"\n{tag} by {group_col}:")
    print(result.to_string())

    valid = result.dropna(subset=["auroc"])
    if len(valid) >= 2:
        gap = valid["auroc"].max() - valid["auroc"].min()
        print(f"  AUROC gap across groups: {gap:.4f} "
              f"({valid['auroc'].idxmax()}={valid['auroc'].max():.4f} vs "
              f"{valid['auroc'].idxmin()}={valid['auroc'].min():.4f})")

    result.to_csv(OUT_DIR / f"fairness_{tag}_by_{group_col}.csv")
    return result


# %%
def plot_group_auroc(results_dict, tag, out_path):
    fig, axes = plt.subplots(1, len(results_dict), figsize=(5 * len(results_dict), 4.5))
    if len(results_dict) == 1:
        axes = [axes]
    pooled_auroc = None  # filled in by caller if desired

    for ax, (group_col, result) in zip(axes, results_dict.items()):
        valid = result.dropna(subset=["auroc"])
        ax.bar(valid.index.astype(str), valid["auroc"])
        ax.set_title(f"AUROC by {group_col}")
        ax.set_ylabel("AUROC")
        ax.set_ylim(0.5, max(0.85, valid["auroc"].max() + 0.03))
        ax.tick_params(axis="x", rotation=30)

    plt.suptitle(f"Fairness audit — {tag} model AUROC by subgroup")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# %%
def run_fairness_audit(model_tag="engineered"):
    print(f"Fairness audit for model_tag='{model_tag}'")
    df, proba_col = load_predictions_with_demographics(model_tag)
    print(f"Loaded {len(df):,} patient-hours, {df['patient_id'].nunique():,} patients "
          f"with demographics for the audit.")

    results = {}
    results["age_bracket"] = audit_by_column(df, proba_col, "age_bracket", model_tag)
    results["gender_label"] = audit_by_column(df, proba_col, "gender_label", model_tag)
    results["hospital_label"] = audit_by_column(df, proba_col, "hospital_label", model_tag)

    plot_group_auroc(results, model_tag, FIG_DIR / f"fairness_audit_{model_tag}.png")

    # Combined summary row: worst-case AUROC gap across all three axes, for a
    # single headline number to quote in the report.
    all_gaps = []
    for group_col, result in results.items():
        valid = result.dropna(subset=["auroc"])
        if len(valid) >= 2:
            all_gaps.append({
                "axis": group_col,
                "max_auroc_group": valid["auroc"].idxmax(),
                "max_auroc": valid["auroc"].max(),
                "min_auroc_group": valid["auroc"].idxmin(),
                "min_auroc": valid["auroc"].min(),
                "gap": valid["auroc"].max() - valid["auroc"].min(),
            })
    gap_summary = pd.DataFrame(all_gaps).sort_values("gap", ascending=False)
    gap_summary.to_csv(OUT_DIR / f"fairness_{model_tag}_gap_summary.csv", index=False)
    print(f"\nGap summary (largest AUROC spread first):")
    print(gap_summary.to_string(index=False))

    print(f"\nSaved: fairness_{model_tag}_by_age_bracket.csv, "
          f"fairness_{model_tag}_by_gender_label.csv, "
          f"fairness_{model_tag}_by_hospital_label.csv, "
          f"fairness_{model_tag}_gap_summary.csv, "
          f"figures/fairness_audit_{model_tag}.png")
    return results, gap_summary


# %%
if __name__ == "__main__":
    run_fairness_audit("engineered")
    # Uncomment to also audit the baseline model for a fairness comparison:
    # run_fairness_audit("baseline")
