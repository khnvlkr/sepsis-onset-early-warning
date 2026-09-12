# %% [markdown]
# # Phase 7 — OLAP operations in DuckDB, then export for Power BI
#
# Demonstrates the four classic OLAP operations (roll-up, drill-down,
# slice, dice) directly in SQL against the existing star schema, so the
# concepts are proven in the warehouse itself -- not just clicked
# together in a BI tool. The same tables are then exported for Power BI,
# where the star-schema relationships and an interactive version of
# these same operations are built per the DWM lab requirement.

# %%
import duckdb
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "warehouse" / "sepsis.duckdb"
OUT_DIR = PROJECT_ROOT / "outputs"
PBI_DIR = OUT_DIR / "powerbi_export"
PBI_DIR.mkdir(parents=True, exist_ok=True)

con = duckdb.connect(str(DB_PATH), read_only=True)

# %% [markdown]
# ## 1. ROLL-UP
# Aggregate from hourly grain -> daily grain -> whole-stay grain.
# Classic OLAP roll-up: moving UP the concept hierarchy (hour -> day -> stay).

# %%
rollup_daily = con.execute("""
    SELECT
        hospital_id,
        patient_id,
        CAST(hour / 24 AS INTEGER) AS day_bucket,
        AVG(HR_ffill) AS avg_HR,
        AVG(Temp_ffill) AS avg_Temp,
        AVG(Lactate_ffill) AS avg_Lactate,
        MAX(SepsisLabel) AS sepsis_that_day
    FROM fact_features
    GROUP BY hospital_id, patient_id, day_bucket
""").df()
print(f"ROLL-UP (hourly -> daily): {rollup_daily.shape[0]:,} rows")
print(rollup_daily.head(3))

rollup_stay = con.execute("""
    SELECT
        hospital_id,
        patient_id,
        AVG(HR_ffill) AS avg_HR_whole_stay,
        AVG(Lactate_ffill) AS avg_Lactate_whole_stay,
        MAX(SepsisLabel) AS ever_septic
    FROM fact_features
    GROUP BY hospital_id, patient_id
""").df()
print(f"\nROLL-UP (daily -> whole stay): {rollup_stay.shape[0]:,} rows")
print(rollup_stay.head(3))

# %% [markdown]
# ## 2. DRILL-DOWN
# The inverse: start from a hospital-level aggregate, then descend back
# into per-patient, then per-hour detail. Same star schema, opposite
# direction of traversal.

# %%
drilldown_hospital = con.execute("""
    SELECT hospital_id, AVG(HR_ffill) AS avg_HR, COUNT(DISTINCT patient_id) AS n_patients
    FROM fact_features
    GROUP BY hospital_id
""").df()
print("DRILL-DOWN level 1 (hospital):")
print(drilldown_hospital.head(3))

# drill into one specific hospital's patients
target_hospital = drilldown_hospital.iloc[0]["hospital_id"]
drilldown_patient = con.execute(f"""
    SELECT patient_id, AVG(HR_ffill) AS avg_HR, MAX(SepsisLabel) AS ever_septic
    FROM fact_features
    WHERE hospital_id = '{target_hospital}'
    GROUP BY patient_id
""").df()
print(f"\nDRILL-DOWN level 2 (patients within hospital {target_hospital}):")
print(drilldown_patient.head(3))

# drill further into one specific patient's hourly detail
target_patient = drilldown_patient.iloc[0]["patient_id"]
drilldown_hourly = con.execute(f"""
    SELECT hour, HR_ffill, Temp_ffill, Lactate_ffill, SepsisLabel
    FROM fact_features
    WHERE patient_id = '{target_patient}'
    ORDER BY hour
""").df()
print(f"\nDRILL-DOWN level 3 (hourly detail for patient {target_patient}):")
print(drilldown_hourly.head(5))

# %% [markdown]
# ## 3. SLICE
# Fix ONE dimension to a single value, keep everything else --
# e.g., "just this hospital" or "just septic hours."

# %%
slice_septic_only = con.execute("""
    SELECT patient_id, hour, HR_ffill, Lactate_ffill, shock_index
    FROM fact_features
    WHERE SepsisLabel = 1
""").df()
print(f"SLICE (SepsisLabel = 1 only): {slice_septic_only.shape[0]:,} rows")

# %% [markdown]
# ## 4. DICE
# Select a sub-cube across MULTIPLE dimensions simultaneously --
# e.g., one hospital AND septic hours AND elevated lactate.

# %%
dice_result = con.execute(f"""
    SELECT patient_id, hour, HR_ffill, Lactate_ffill
    FROM fact_features
    WHERE hospital_id = '{target_hospital}'
      AND SepsisLabel = 1
      AND Lactate_ffill > 2.0
""").df()
print(f"\nDICE (hospital={target_hospital} AND septic AND Lactate>2.0): {dice_result.shape[0]:,} rows")

# %% [markdown]
# ## Save OLAP demo outputs (for report screenshots / appendix)

# %%
rollup_daily.to_csv(OUT_DIR / "olap_rollup_daily.csv", index=False)
rollup_stay.to_csv(OUT_DIR / "olap_rollup_stay.csv", index=False)
drilldown_hourly.to_csv(OUT_DIR / "olap_drilldown_hourly_example.csv", index=False)
slice_septic_only.to_csv(OUT_DIR / "olap_slice_septic.csv", index=False)
dice_result.to_csv(OUT_DIR / "olap_dice_example.csv", index=False)
print("\nSaved OLAP demonstration CSVs to outputs/")

# %% [markdown]
# ## Export star schema tables for Power BI
# Dimension tables as-is; a slimmed fact table (raw vitals + key flags,
# hourly grain) sized reasonably for Power BI's in-memory engine.

# %%
con.execute("SELECT * FROM dim_patient").df().to_csv(PBI_DIR / "dim_patient.csv", index=False)
con.execute("SELECT * FROM dim_hospital").df().to_csv(PBI_DIR / "dim_hospital.csv", index=False)

olap_fact = con.execute("""
    SELECT
        patient_id, hospital_id, hour, ICULOS, SepsisLabel,
        HR_ffill AS HR, O2Sat_ffill AS O2Sat, Temp_ffill AS Temp,
        SBP_ffill AS SBP, MAP_ffill AS MAP, Resp_ffill AS Resp,
        Lactate_ffill AS Lactate, shock_index, partial_sirs_score,
        CAST(hour / 24 AS INTEGER) AS day_bucket
    FROM fact_features
""").df()
olap_fact.to_csv(PBI_DIR / "fact_vitals_olap.csv", index=False)

con.close()
print(f"\nExported Power BI-ready tables to: {PBI_DIR}")
print(f"fact_vitals_olap.csv: {olap_fact.shape[0]:,} rows x {olap_fact.shape[1]} cols")
print("\nNext steps in Power BI Desktop:")
print("  1. Get Data -> Text/CSV -> import all 3 files from powerbi_export/")
print("  2. Model view -> drag patient_id (fact -> dim_patient), hospital_id (fact -> dim_hospital)")
print("  3. Matrix visual: Rows = hospital_id > day_bucket > hour (hierarchy) = rollup/drilldown")
print("  4. Slicer on hospital_id or SepsisLabel = slice")
print("  5. Matrix: Rows=hospital_id, Columns=SepsisLabel, Values=avg(shock_index) = dice")
