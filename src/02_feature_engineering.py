# %% [markdown]
# # Phase 2/4 — Feature engineering as SQL window functions (DuckDB)
#
# Everything here is computed *causally*: every window is
# `ROWS BETWEEN N PRECEDING AND CURRENT ROW` — never FOLLOWING — so no row
# ever sees data from its own future. This is what makes the feature set
# safe to use for prediction instead of just descriptive analytics.
#
# Output: `fact_features` table, grain = (patient_id, hour), which Phase 3/5
# read directly with `SELECT * FROM fact_features`.

# %%
import duckdb
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "warehouse" / "sepsis.duckdb"

VITAL_COLS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
# A clinically-relevant subset of the 26 labs (all 26 are still available raw
# in fact_vitals_hourly if you want to extend this later).
KEY_LAB_COLS = ["WBC", "Lactate", "Creatinine", "Platelets", "BUN",
                 "Bilirubin_total", "Glucose", "Potassium"]
ROLLING_COLS = VITAL_COLS + KEY_LAB_COLS
WINDOWS_HOURS = [3, 6, 12]

# %% [markdown]
# ## Step 1 — missingness + causal forward-fill for every rolling column
#
# For each column X we add:
#   X_missing          : 1 if raw value is NULL at this hour
#   X_hours_since_last  : hours since the last non-null measurement (NULL if
#                          never measured yet for this patient)
#   X_ffill             : last known value carried forward (NULL until first
#                          measurement — we do NOT mean-impute blindly)

# %%
def build_ffill_sql(cols):
    parts = []
    for c in cols:
        parts.append(f"""
            {c} AS {c}_raw,
            CASE WHEN {c} IS NULL THEN 1 ELSE 0 END AS {c}_missing,
            hour - LAST_VALUE(CASE WHEN {c} IS NOT NULL THEN hour END IGNORE NULLS)
                OVER (PARTITION BY patient_id ORDER BY hour
                      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                AS {c}_hours_since_last,
            LAST_VALUE({c} IGNORE NULLS)
                OVER (PARTITION BY patient_id ORDER BY hour
                      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                AS {c}_ffill
        """)
    return ",\n".join(parts)

# %% [markdown]
# ## Step 2 — rolling window stats + slope, computed on the forward-filled series
#
# mean / std / min / max over 3h, 6h, 12h causal windows, plus a linear-regression
# slope (REGR_SLOPE, native DuckDB window aggregate) as a "velocity" feature,
# and a first-difference "acceleration" feature (slope of slope).

# %%
def build_rolling_sql(cols, windows):
    parts = []
    for c in cols:
        for w in windows:
            win = f"(PARTITION BY patient_id ORDER BY hour ROWS BETWEEN {w-1} PRECEDING AND CURRENT ROW)"
            parts.append(f"AVG({c}_ffill)    OVER {win} AS {c}_mean_{w}h")
            parts.append(f"STDDEV({c}_ffill) OVER {win} AS {c}_std_{w}h")
            parts.append(f"MIN({c}_ffill)    OVER {win} AS {c}_min_{w}h")
            parts.append(f"MAX({c}_ffill)    OVER {win} AS {c}_max_{w}h")
            parts.append(f"REGR_SLOPE({c}_ffill, hour) OVER {win} AS {c}_slope_{w}h")
        # velocity (1st difference) and acceleration (2nd difference), causal
        parts.append(
            f"{c}_ffill - LAG({c}_ffill, 1) OVER (PARTITION BY patient_id ORDER BY hour) "
            f"AS {c}_velocity_1h"
        )
    return ",\n".join(parts)

# %% [markdown]
# ## Step 3 — clinical composite scores (computed from forward-filled values)
# - Shock Index = HR / SBP  (elevated = early compensated shock)
# - Pulse Pressure = SBP - DBP
# - Partial SIRS count (2 of 4 canonical criteria are derivable here):
#     Temp > 38 or Temp < 36 ; HR > 90 ; Resp > 20 ; WBC > 12 or WBC < 4
# - Partial qSOFA count (2 of 3 derivable; altered mental status not in data):
#     Resp >= 22 ; SBP <= 100

CLINICAL_SQL = """
    HR_ffill / NULLIF(SBP_ffill, 0)                          AS shock_index,
    SBP_ffill - DBP_ffill                                    AS pulse_pressure,
    (CASE WHEN Temp_ffill > 38 OR Temp_ffill < 36 THEN 1 ELSE 0 END
     + CASE WHEN HR_ffill > 90 THEN 1 ELSE 0 END
     + CASE WHEN Resp_ffill > 20 THEN 1 ELSE 0 END
     + CASE WHEN WBC_ffill > 12 OR WBC_ffill < 4 THEN 1 ELSE 0 END)
                                                               AS partial_sirs_score,
    (CASE WHEN Resp_ffill >= 22 THEN 1 ELSE 0 END
     + CASE WHEN SBP_ffill <= 100 THEN 1 ELSE 0 END)          AS partial_qsofa_score
"""

# %%
def build_features():
    con = duckdb.connect(str(DB_PATH))

    ffill_sql = build_ffill_sql(ROLLING_COLS)
    con.execute(f"""
        CREATE OR REPLACE TABLE fact_ffill AS
        SELECT
            patient_id, hospital_id, hour, ICULOS, SepsisLabel,
            {ffill_sql}
        FROM fact_vitals_hourly
        ORDER BY patient_id, hour
    """)
    print("fact_ffill built:", con.execute("SELECT COUNT(*) FROM fact_ffill").fetchone()[0], "rows")

    rolling_sql = build_rolling_sql(ROLLING_COLS, WINDOWS_HOURS)
    ffill_pass_through = ", ".join(f"{c}_ffill" for c in ROLLING_COLS)
    missing_pass_through = ", ".join(
        f"{c}_missing, {c}_hours_since_last" for c in ROLLING_COLS
    )

    con.execute(f"""
        CREATE OR REPLACE TABLE fact_features AS
        SELECT
            patient_id, hospital_id, hour, ICULOS, SepsisLabel,
            {ffill_pass_through},
            {missing_pass_through},
            {rolling_sql},
            {CLINICAL_SQL}
        FROM fact_ffill
        ORDER BY patient_id, hour
    """)

    n = con.execute("SELECT COUNT(*) FROM fact_features").fetchone()[0]
    n_cols = len(con.execute("SELECT * FROM fact_features LIMIT 0").description)
    print(f"fact_features built: {n:,} rows x {n_cols} columns")
    print("Sample:")
    print(con.execute("SELECT * FROM fact_features LIMIT 3").df().T)

    con.close()

# %%
if __name__ == "__main__":
    build_features()
