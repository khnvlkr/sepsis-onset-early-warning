# %% [markdown]
# # Phase 2 — ETL: raw PSV -> DuckDB star schema
#
# Grain of the fact table: one row per (patient, ICU hour).
# Star schema:
#   fact_vitals_hourly  (patient_id, hour, vitals..., labs..., SepsisLabel)
#   dim_patient         (patient_id, age, gender, hospital_id, source_file)
#   dim_hospital        (hospital_id, hospital_name)
#   dim_time            (hour) -- degenerate dim, hour is already an integer offset

# %%
import os
import glob
import duckdb
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DB_PATH = PROJECT_ROOT / "warehouse" / "sepsis.duckdb"

# Column layout of the official PhysioNet 2019 .psv files
VITAL_COLS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LAB_COLS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]
DEMO_COLS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
LABEL_COL = "SepsisLabel"
ALL_COLS = VITAL_COLS + LAB_COLS + DEMO_COLS + [LABEL_COL]

# %%
def find_hospital_folders():
    """Each subfolder under data/raw is treated as one 'hospital system'."""
    folders = sorted([p for p in RAW_DIR.iterdir() if p.is_dir()])
    if not folders:
        raise FileNotFoundError(
            f"No subfolders found in {RAW_DIR}. Put PhysioNet .psv files in "
            f"{RAW_DIR}/training_setA/ and {RAW_DIR}/training_setB/ first "
            f"(see README.md section 0)."
        )
    return folders

def load_one_patient(psv_path: Path, hospital_id: int) -> pd.DataFrame:
    df = pd.read_csv(psv_path, sep="|")
    missing = [c for c in ALL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{psv_path.name} is missing expected columns: {missing}")
    patient_id = psv_path.stem  # e.g. 'p000001'
    df = df[ALL_COLS].copy()
    df.insert(0, "hour", range(len(df)))       # 0-indexed ICU hour, causal ordering
    df.insert(0, "hospital_id", hospital_id)
    df.insert(0, "patient_id", patient_id)
    return df

# %%
def build_warehouse():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()  # rebuild clean each time this script runs

    con = duckdb.connect(str(DB_PATH))

    hospital_folders = find_hospital_folders()
    print(f"Found {len(hospital_folders)} hospital-system folder(s): "
          f"{[f.name for f in hospital_folders]}")

    frames = []
    for hospital_id, folder in enumerate(hospital_folders, start=1):
        psv_files = sorted(folder.glob("*.psv"))
        print(f"  hospital_id={hospital_id} ({folder.name}): {len(psv_files)} patients")
        for f in psv_files:
            frames.append(load_one_patient(f, hospital_id))

    staging = pd.concat(frames, ignore_index=True)
    print(f"Staging table: {staging.shape[0]:,} patient-hours, "
          f"{staging['patient_id'].nunique():,} patients")

    con.execute("CREATE TABLE staging_vitals AS SELECT * FROM staging")

    # ---- dim_hospital ----
    con.execute("""
        CREATE TABLE dim_hospital AS
        SELECT DISTINCT hospital_id,
               'hospital_system_' || hospital_id AS hospital_name
        FROM staging_vitals
    """)

    # ---- dim_patient (one row per patient; demographics are static per file) ----
    con.execute("""
        CREATE TABLE dim_patient AS
        SELECT
            patient_id,
            FIRST(hospital_id)   AS hospital_id,
            FIRST(Age)           AS age,
            FIRST(Gender)        AS gender,
            FIRST(HospAdmTime)   AS hosp_admit_time,
            MAX(ICULOS)          AS max_iculos,
            MAX(SepsisLabel)     AS is_ever_septic,
            COUNT(*)             AS n_hours_recorded
        FROM staging_vitals
        GROUP BY patient_id
    """)

    # ---- fact_vitals_hourly (grain = patient x hour) ----
    col_list = ", ".join(VITAL_COLS + LAB_COLS)
    con.execute(f"""
        CREATE TABLE fact_vitals_hourly AS
        SELECT
            patient_id,
            hospital_id,
            hour,
            {col_list},
            ICULOS,
            SepsisLabel
        FROM staging_vitals
        ORDER BY patient_id, hour
    """)

    con.execute("DROP TABLE staging_vitals")

    # Helpful indexes for the window-function pass in the next script
    con.execute("CREATE INDEX idx_fact_patient_hour ON fact_vitals_hourly(patient_id, hour)")

    n_patients = con.execute("SELECT COUNT(*) FROM dim_patient").fetchone()[0]
    n_rows = con.execute("SELECT COUNT(*) FROM fact_vitals_hourly").fetchone()[0]
    pos_rate = con.execute(
        "SELECT AVG(SepsisLabel) FROM fact_vitals_hourly"
    ).fetchone()[0]
    print(f"\nWarehouse built: {DB_PATH}")
    print(f"  dim_patient:        {n_patients:,} patients")
    print(f"  fact_vitals_hourly: {n_rows:,} rows")
    print(f"  positive-hour rate: {pos_rate:.3%}")

    con.close()

# %%
if __name__ == "__main__":
    build_warehouse()
