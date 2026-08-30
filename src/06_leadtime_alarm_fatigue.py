# %% [markdown]
# # Phase 7 — Lead time & alarm fatigue
#
# 1. For every patient who develops sepsis, find the first hour the model's
#    alert fires (probability >= chosen threshold) and compare it to the
#    actual onset hour (first SepsisLabel==1). Report the distribution of
#    lead time (positive = caught early).
# 2. Compare the false-alarm rate of the model against a naive static-
#    threshold rule (>=2 SIRS-style criteria) at MATCHED sensitivity, so the
#    comparison isn't just "our model fires less often" — it's apples-to-apples.

# %%
import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utility_score import sweep_thresholds_for_utility

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "warehouse" / "sepsis.duckdb"
OUT_DIR = PROJECT_ROOT / "outputs"

# %%
def load_predictions_and_features():
    preds = pd.read_parquet(OUT_DIR / "engineered_oof_predictions.parquet")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    feats = con.execute(
        "SELECT patient_id, hour, partial_sirs_score FROM fact_features"
    ).df()
    con.close()
    df = preds.merge(feats, on=["patient_id", "hour"], how="left")
    return df

# %%
def pick_operating_threshold(df):
    thr_table = sweep_thresholds_for_utility(
        df, patient_col="patient_id", label_col="SepsisLabel", proba_col="engineered_proba"
    )
    return float(thr_table.iloc[0].threshold)

# %%
def lead_time_analysis(df, threshold):
    df = df.sort_values(["patient_id", "hour"]).copy()
    df["alert"] = (df["engineered_proba"] >= threshold).astype(int)

    rows = []
    for pid, g in df.groupby("patient_id"):
        is_septic = g["SepsisLabel"].max() == 1
        if not is_septic:
            continue
        onset_hour = g.loc[g["SepsisLabel"] == 1, "hour"].min()
        alerts = g.loc[g["alert"] == 1, "hour"]
        first_alert_hour = alerts.min() if len(alerts) else np.nan
        lead_time = onset_hour - first_alert_hour if not np.isnan(first_alert_hour) else np.nan
        rows.append({
            "patient_id": pid, "onset_hour": onset_hour,
            "first_alert_hour": first_alert_hour, "lead_time_hours": lead_time,
            "caught": not np.isnan(first_alert_hour),
        })
    lt = pd.DataFrame(rows)
    caught_rate = lt["caught"].mean()
    median_lead = lt.loc[lt["caught"], "lead_time_hours"].median()
    caught_ge6h = (lt.loc[lt["caught"], "lead_time_hours"] >= 6).mean()

    print(f"Septic patients: {len(lt)}")
    print(f"  Caught by at least one alert before/at onset: {caught_rate:.1%}")
    print(f"  Median lead time among caught patients: {median_lead:.1f} h")
    print(f"  % of caught patients warned >= 6h ahead: {caught_ge6h:.1%}")

    lt.to_csv(OUT_DIR / "lead_time_by_patient.csv", index=False)
    return lt

# %%
def alarm_fatigue_comparison(df, threshold):
    """Compare model alerts vs a naive rule (SIRS >= 2) at MATCHED sensitivity
    on non-septic-hour false-alarm rate."""
    df = df.copy()
    non_septic_hours = df["SepsisLabel"] == 0
    y = df["SepsisLabel"]

    # naive rule: fire whenever partial_sirs_score >= 2
    naive_alert = (df["partial_sirs_score"] >= 2).astype(int)
    naive_sensitivity = (naive_alert[y == 1] == 1).mean()
    naive_false_alarm_rate = (naive_alert[non_septic_hours] == 1).mean()

    # find model threshold that matches naive_sensitivity (search a fine grid)
    grid = np.linspace(0.001, 0.999, 400)
    best_thr, best_gap = None, np.inf
    for thr in grid:
        model_alert = (df["engineered_proba"] >= thr).astype(int)
        sens = (model_alert[y == 1] == 1).mean()
        gap = abs(sens - naive_sensitivity)
        if gap < best_gap:
            best_gap, best_thr = gap, thr

    model_alert_matched = (df["engineered_proba"] >= best_thr).astype(int)
    model_sensitivity_matched = (model_alert_matched[y == 1] == 1).mean()
    model_false_alarm_matched = (model_alert_matched[non_septic_hours] == 1).mean()

    result = pd.DataFrame([
        {"rule": "naive_sirs_>=2", "threshold": None,
         "sensitivity": naive_sensitivity, "false_alarm_rate_nonseptic_hours": naive_false_alarm_rate},
        {"rule": "engineered_model_matched_sensitivity", "threshold": best_thr,
         "sensitivity": model_sensitivity_matched, "false_alarm_rate_nonseptic_hours": model_false_alarm_matched},
        {"rule": "engineered_model_operating_point", "threshold": threshold,
         "sensitivity": (df["engineered_proba"] >= threshold)[y == 1].mean(),
         "false_alarm_rate_nonseptic_hours": (df["engineered_proba"] >= threshold)[non_septic_hours].mean()},
    ])
    print("\nAlarm-fatigue comparison (matched sensitivity):")
    print(result.to_string(index=False))
    result.to_csv(OUT_DIR / "alarm_fatigue_comparison.csv", index=False)
    return result

# %%
if __name__ == "__main__":
    df = load_predictions_and_features()
    threshold = pick_operating_threshold(df)
    print(f"Operating threshold (max normalized utility): {threshold:.3f}\n")
    lead_time_analysis(df, threshold)
    alarm_fatigue_comparison(df, threshold)
