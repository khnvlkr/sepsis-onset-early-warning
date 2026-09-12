# %% [markdown]
# # Phase 7d — Outlier analysis
#
# DWM Module 6 names outlier analysis as its own technique, alongside
# association rules and clustering. It's also a natural fit for this dataset:
# extreme lactate, extreme temperature, extreme heart rate etc. are often the
# clinical signal itself in sepsis, so this isn't a bolted-on afterthought —
# it should corroborate the SHAP (05) and association-rule (08) findings,
# where Lactate, Temp, and Resp already show up as top predictors/flags.
#
# Two standard outlier-detection approaches, deliberately kept simple and
# interpretable rather than exotic, since the point is to demonstrate the
# *technique* (module 6 requirement) with a clinically sensible result, not
# to out-perform the supervised model:
#
#   1. Univariate IQR fencing per vital, at the patient-hour grain — the
#      classic, easily explained outlier rule.
#   2. Multivariate Isolation Forest across several vitals at once, to catch
#      combinations that look normal one-vital-at-a-time but are jointly
#      unusual (e.g. moderately high HR + moderately low SBP together).
#
# Both are checked against SepsisLabel: if outlier-flagged hours are
# disproportionately septic, that's a meaningful validation, not just a
# statistical curiosity.

# %%
import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "warehouse" / "sepsis.duckdb"
OUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
IQR_MULTIPLIER = 1.5  # standard Tukey fence

# Same headline vitals the SHAP (05) and association-rule (08) scripts
# already flagged as most informative, so this analysis lines up with the
# rest of the report rather than introducing a new, disconnected vital list.
KEY_VITALS = ["HR", "Temp", "Resp", "SBP", "Lactate", "WBC"]


# %%
def load_vitals_frame():
    """Patient-hour grain, forward-filled vitals + SepsisLabel from
    fact_ffill — same grain and table as 03_baseline_model.py, so results
    here are directly comparable to the rest of the pipeline."""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    cols = ", ".join(f'"{v}_ffill"' for v in KEY_VITALS)
    df = con.execute(f"""
        SELECT patient_id, hour, SepsisLabel, {cols}
        FROM fact_ffill
    """).df()
    con.close()
    df = df.rename(columns={f"{v}_ffill": v for v in KEY_VITALS})
    return df


# %%
def iqr_outlier_flags(df, vitals):
    """Per-vital Tukey IQR fencing. Returns the input df with one boolean
    '<vital>_outlier' column per vital, plus 'iqr_any_outlier' (flagged on
    at least one vital) and 'iqr_outlier_count' (how many vitals at once —
    more simultaneous outliers is a stronger signal, mirroring the
    compound-rule finding in 08_association_rules.py)."""
    bounds = {}
    for v in vitals:
        col = df[v].dropna()
        q1, q3 = col.quantile(0.25), col.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - IQR_MULTIPLIER * iqr, q3 + IQR_MULTIPLIER * iqr
        bounds[v] = (lo, hi)
        df[f"{v}_outlier"] = ((df[v] < lo) | (df[v] > hi)).fillna(False)

    outlier_cols = [f"{v}_outlier" for v in vitals]
    df["iqr_outlier_count"] = df[outlier_cols].sum(axis=1)
    df["iqr_any_outlier"] = df["iqr_outlier_count"] > 0

    bounds_df = pd.DataFrame(
        [{"vital": v, "lower_fence": lo, "upper_fence": hi} for v, (lo, hi) in bounds.items()]
    )
    return df, bounds_df


# %%
def isolation_forest_flags(df, vitals, contamination=0.02):
    """Multivariate outlier detection across all vitals jointly. contamination
    is set close to the dataset's own positive rate (1.8%, see README) so the
    flagged fraction is a comparable order of magnitude to the actual sepsis
    rate, rather than an arbitrary round number."""
    sub = df[vitals].copy()
    med = sub.median()
    sub = sub.fillna(med)  # IsolationForest can't handle NaN

    iso = IsolationForest(
        n_estimators=200, contamination=contamination,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    pred = iso.fit_predict(sub)  # -1 = outlier, 1 = inlier
    df["iforest_outlier"] = pred == -1
    return df


# %%
def summarize_against_sepsis(df, flag_col, label="flag"):
    overall_rate = df["SepsisLabel"].mean()
    flagged_rate = df.loc[df[flag_col], "SepsisLabel"].mean()
    unflagged_rate = df.loc[~df[flag_col], "SepsisLabel"].mean()
    lift = flagged_rate / overall_rate if overall_rate > 0 else float("nan")
    n_flagged = int(df[flag_col].sum())
    print(f"  [{label}] flagged={n_flagged:,} ({100*n_flagged/len(df):.2f}% of rows) | "
          f"sepsis rate flagged={flagged_rate:.4f} vs unflagged={unflagged_rate:.4f} "
          f"vs overall={overall_rate:.4f} | lift={lift:.2f}x")
    return {
        "method": label, "n_flagged": n_flagged,
        "pct_flagged": 100 * n_flagged / len(df),
        "sepsis_rate_flagged": flagged_rate,
        "sepsis_rate_unflagged": unflagged_rate,
        "sepsis_rate_overall": overall_rate,
        "lift": lift,
    }


# %%
def run_outlier_analysis():
    df = load_vitals_frame()
    print(f"Loaded {len(df):,} patient-hours, vitals={KEY_VITALS}")

    df, bounds_df = iqr_outlier_flags(df, KEY_VITALS)
    df = isolation_forest_flags(df, KEY_VITALS)

    print("\nIQR fence bounds:")
    print(bounds_df.to_string(index=False))

    print("\nOutlier vs. sepsis rate:")
    rows = []
    rows.append(summarize_against_sepsis(df, "iqr_any_outlier", "iqr_any_vital"))
    rows.append(summarize_against_sepsis(
        df.assign(iqr_2plus=df["iqr_outlier_count"] >= 2), "iqr_2plus", "iqr_2plus_vitals"
    ))
    rows.append(summarize_against_sepsis(df, "iforest_outlier", "isolation_forest"))

    # Per-vital breakdown, so the report can name which single vital's
    # outliers carry the most sepsis signal (expect Lactate/Temp, matching
    # the SHAP ranking in 05_explainability.py).
    print("\nPer-vital IQR outlier lift:")
    for v in KEY_VITALS:
        rows.append(summarize_against_sepsis(df, f"{v}_outlier", f"iqr_{v}"))

    summary_df = pd.DataFrame(rows).sort_values("lift", ascending=False)
    summary_df.to_csv(OUT_DIR / "outlier_sepsis_lift.csv", index=False)
    bounds_df.to_csv(OUT_DIR / "outlier_iqr_bounds.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = summary_df[summary_df["method"] != "iqr_2plus_vitals"]
    ax.barh(plot_df["method"], plot_df["lift"])
    ax.axvline(1.0, color="gray", linestyle="--", label="no lift (baseline rate)")
    ax.set_xlabel("Sepsis-rate lift among flagged hours vs. overall rate")
    ax.set_title("Outlier detection: sepsis lift by method / vital")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "outlier_sepsis_lift.png", dpi=150)
    plt.close()

    print(f"\nSaved: outlier_sepsis_lift.csv, outlier_iqr_bounds.csv, "
          f"figures/outlier_sepsis_lift.png")
    return summary_df, bounds_df


# %%
if __name__ == "__main__":
    run_outlier_analysis()
