# %% [markdown]
# # Optional — Unsupervised mining: deterioration phenotypes
#
# Adds a genuine data-mining component beyond supervised prediction: k-means
# on a per-patient summary of their vitals trajectory (not raw hourly series
# — that keeps it fast and avoids needing DTW for a capstone-scale project).
# Cut this first if you're short on time; everything else is core.

# %%
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "warehouse" / "sepsis.duckdb"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]

# %%
def build_patient_summary():
    # Not read-only: this script writes a small derived table (patient_summary)
    # back into the warehouse for later reuse (e.g. a Power BI cluster view).
    con = duckdb.connect(str(DB_PATH), read_only=False)
    # Per-patient trajectory summary (mean/std/min/max) — enough signal for
    # phenotype clustering without needing to store full raw sequences.
    con.execute("""
        CREATE OR REPLACE TABLE patient_summary AS
        SELECT
            patient_id,
            MAX(SepsisLabel) AS is_ever_septic,
            AVG(HR_ffill) AS HR_mean, STDDEV(HR_ffill) AS HR_std,
            AVG(O2Sat_ffill) AS O2Sat_mean, STDDEV(O2Sat_ffill) AS O2Sat_std,
            AVG(Temp_ffill) AS Temp_mean, STDDEV(Temp_ffill) AS Temp_std,
            AVG(SBP_ffill) AS SBP_mean, STDDEV(SBP_ffill) AS SBP_std,
            AVG(MAP_ffill) AS MAP_mean, STDDEV(MAP_ffill) AS MAP_std,
            AVG(DBP_ffill) AS DBP_mean, STDDEV(DBP_ffill) AS DBP_std,
            AVG(Resp_ffill) AS Resp_mean, STDDEV(Resp_ffill) AS Resp_std,
            MIN(HR_ffill) AS HR_min, MAX(HR_ffill) AS HR_max,
            MIN(SBP_ffill) AS SBP_min, MAX(SBP_ffill) AS SBP_max
        FROM fact_ffill AS f
        GROUP BY patient_id
    """)
    df = con.execute("SELECT * FROM patient_summary").df()
    con.close()
    return df.dropna()  # patients with too few readings to compute std, etc.

# %%
def run_clustering(k_range=range(2, 8)):
    df = build_patient_summary()
    feature_cols = [c for c in df.columns if c not in ("patient_id", "is_ever_septic")]
    X = StandardScaler().fit_transform(df[feature_cols])

    sil_scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels, sample_size=min(5000, len(X)), random_state=42)
        sil_scores.append({"k": k, "silhouette": sil})
        print(f"k={k}: silhouette={sil:.3f}")

    sil_df = pd.DataFrame(sil_scores)
    best_k = int(sil_df.loc[sil_df["silhouette"].idxmax(), "k"])
    print(f"\nBest k by silhouette: {best_k}")

    km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    df["cluster"] = km.fit_predict(X)

    # profile clusters: mean feature values + sepsis rate per cluster
    profile = df.groupby("cluster")[feature_cols + ["is_ever_septic"]].mean()
    profile["n_patients"] = df.groupby("cluster").size()
    print("\nCluster profile (mean values + sepsis rate):")
    print(profile.to_string())

    profile.to_csv(PROJECT_ROOT / "outputs" / "cluster_profiles.csv")
    df[["patient_id", "cluster", "is_ever_septic"]].to_csv(
        PROJECT_ROOT / "outputs" / "patient_clusters.csv", index=False
    )

    plt.figure()
    plt.bar(sil_df["k"].astype(str), sil_df["silhouette"])
    plt.xlabel("k"); plt.ylabel("silhouette score"); plt.title("Cluster count selection")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cluster_silhouette.png", dpi=150)
    plt.close()

    return df, profile

# %%
if __name__ == "__main__":
    run_clustering()
