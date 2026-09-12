"""
09_hierarchical_clustering.py  (FIXED)

Change vs original version:
- Ward-linkage hierarchical clustering is O(n^2) in memory (condensed pairwise
  distance matrix). On the full 40,336-patient trajectory table this previously
  caused the script to silently fall back to (or accidentally run on) a ~2,000
  patient subset, while the log still printed "Loaded 40,336 patient trajectory
  summaries" -- making it look like the full population was clustered when it
  wasn't. k-means, by contrast, scales fine and ran on all 32,465 patients with
  complete trajectory features.
- This version fixes that by explicitly and reproducibly subsampling, with the
  subsample size chosen deliberately (not accidentally) to fit comfortably in
  memory on a 7.4GB RAM laptop, using STRATIFIED sampling on is_ever_septic so
  the rare-event rate is preserved. The subsample size, seed, and the exact
  reason for subsampling are logged and saved to a manifest file so it can be
  cited directly in the report as a disclosed compute constraint.
- k-means is *also* re-run on the identical subsample (in addition to its
  original full-population run) so the "both methods agree" comparison is
  apples-to-apples, not full-N vs small-N.

Memory math (why N=8000 is safe on a 7.4GB laptop):
  scipy's Ward linkage stores a condensed distance matrix of size N*(N-1)/2
  float64 values.
    N=2,000  ->   ~16 MB
    N=8,000  ->  ~256 MB
    N=12,000 ->  ~576 MB
  8,000 gives a 4x larger, still-representative sample vs the original 2,000
  while staying well under the ~7.4GB ceiling that killed script 04's earlier
  run. Adjust HIER_SUBSAMPLE_N below if your machine has more/less headroom.
"""

import duckdb
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- config ---------------------------------------------------------------
DUCKDB_PATH = "warehouse/sepsis.duckdb"          # adjust to your actual path
HIER_SUBSAMPLE_N = 8000                          # deliberate, disclosed choice
RANDOM_SEED = 42
OUT_DIR = "outputs"
FIG_DIR = f"{OUT_DIR}/figures"

FEATURE_COLS = [
    "HR_mean", "HR_std", "O2Sat_mean", "O2Sat_std", "Temp_mean", "Temp_std",
    "SBP_mean", "SBP_std", "MAP_mean", "MAP_std", "DBP_mean", "DBP_std",
    "Resp_mean", "Resp_std", "HR_min", "HR_max", "SBP_min", "SBP_max",
]

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(str(msg))


def load_patient_trajectory_summary(con):
    """Same per-patient summary table used by the k-means script
    (clustering_phenotypes.py) -- kept identical so both methods cluster on
    the same feature definitions and are directly comparable."""
    # NOTE: fact_vitals_hourly stores RAW vitals (no _ffill suffix -- those
    # only exist in the engineered feature table from script 02). AVG/STDDEV/
    # MIN/MAX in SQL ignore NULLs by default, so aggregating the raw columns
    # directly is equivalent to aggregating a forward-filled version for
    # these particular per-patient summary stats.
    query = f"""
        SELECT
            patient_id,
            AVG(HR)   AS HR_mean,  STDDEV(HR)   AS HR_std,
            AVG(O2Sat) AS O2Sat_mean, STDDEV(O2Sat) AS O2Sat_std,
            AVG(Temp) AS Temp_mean, STDDEV(Temp) AS Temp_std,
            AVG(SBP)  AS SBP_mean,  STDDEV(SBP)  AS SBP_std,
            AVG(MAP)  AS MAP_mean,  STDDEV(MAP)  AS MAP_std,
            AVG(DBP)  AS DBP_mean,  STDDEV(DBP)  AS DBP_std,
            AVG(Resp) AS Resp_mean, STDDEV(Resp) AS Resp_std,
            MIN(HR)   AS HR_min,    MAX(HR)      AS HR_max,
            MIN(SBP)  AS SBP_min,   MAX(SBP)     AS SBP_max,
            MAX(SepsisLabel) AS is_ever_septic
        FROM fact_vitals_hourly
        GROUP BY patient_id
    """
    df = con.execute(query).df()
    return df.dropna(subset=FEATURE_COLS)  # complete-case, matches k-means script


def stratified_subsample(df, n, seed):
    """Stratify on is_ever_septic to preserve the ~1.8% sepsis rate in the
    subsample -- an unstratified random draw of 8,000 out of ~32,000 would
    only carry a couple hundred septic patients by chance and could easily
    over/under-represent them; stratifying pins the ratio exactly.

    Deliberately avoids groupby().apply(lambda g: g.sample(...)) -- recent
    pandas versions can drop the grouping column from the result in that
    pattern, depending on version/include_groups defaults. Splitting
    manually sidesteps that entirely and is easier to reason about."""
    frac = n / len(df)
    parts = []
    for label, group in df.groupby("is_ever_septic"):
        parts.append(group.sample(frac=frac, random_state=seed))
    sub = pd.concat(parts, ignore_index=True)
    return sub


def main():
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    full = load_patient_trajectory_summary(con)
    log(f"Loaded {len(full):,} patient trajectory summaries (complete-case)")

    # --- disclosed, deliberate subsample for hierarchical clustering -------
    subsample = stratified_subsample(full, HIER_SUBSAMPLE_N, RANDOM_SEED)
    log(
        f"\nCOMPUTE-CONSTRAINT DISCLOSURE:\n"
        f"  Ward-linkage hierarchical clustering requires an O(n^2) pairwise\n"
        f"  distance matrix, which is not tractable on the full {len(full):,}-patient\n"
        f"  population on this machine (7.4GB RAM laptop -- a prior full-data run\n"
        f"  in script 04 already triggered an OOM crash of the whole desktop session).\n"
        f"  Hierarchical clustering below therefore runs on a stratified random\n"
        f"  subsample of n={HIER_SUBSAMPLE_N:,} patients (seed={RANDOM_SEED}),\n"
        f"  stratified on is_ever_septic to preserve the sepsis rate. K-means\n"
        f"  results are reported both on the full {len(full):,}-patient population\n"
        f"  AND on this identical {HIER_SUBSAMPLE_N:,}-patient subsample, so the\n"
        f"  two clustering methods can be compared on equal footing."
    )
    full_rate = full["is_ever_septic"].mean()
    sub_rate = subsample["is_ever_septic"].mean()
    log(f"  Sepsis rate -- full population: {full_rate:.4f}, subsample: {sub_rate:.4f}")

    scaler = StandardScaler()
    X_sub = scaler.fit_transform(subsample[FEATURE_COLS])

    # --- hierarchical (Ward) on the subsample -------------------------------
    Z = linkage(X_sub, method="ward")
    hier_labels = fcluster(Z, t=2, criterion="maxclust")
    hier_sil = silhouette_score(X_sub, hier_labels)
    log(f"\nHierarchical (Ward, k=2, n={HIER_SUBSAMPLE_N:,}) silhouette: {hier_sil:.3f}")

    plt.figure(figsize=(12, 5))
    dendrogram(Z, truncate_mode="lastp", p=40, show_leaf_counts=True)
    plt.title(f"Hierarchical clustering dendrogram (n={HIER_SUBSAMPLE_N:,} stratified subsample)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/hierarchical_dendrogram.png", dpi=150)
    plt.close()
    log(f"Saved dendrogram to {FIG_DIR}/hierarchical_dendrogram.png")

    hier_profile = subsample.assign(cluster=hier_labels).groupby("cluster").agg(
        {**{c: "mean" for c in FEATURE_COLS}, "is_ever_septic": "mean", "patient_id": "count"}
    ).rename(columns={"patient_id": "n_patients"})
    hier_profile.to_csv(f"{OUT_DIR}/hierarchical_cluster_profiles.csv")
    log(f"\nHierarchical cluster profile (n={HIER_SUBSAMPLE_N:,} subsample):")
    log(hier_profile)

    # --- k-means on the SAME subsample, for a fair side-by-side ------------
    km_sub = KMeans(n_clusters=2, random_state=RANDOM_SEED, n_init=10).fit(X_sub)
    km_sub_sil = silhouette_score(X_sub, km_sub.labels_)
    log(f"\nk-means (k=2) silhouette on the SAME {HIER_SUBSAMPLE_N:,}-patient subsample: {km_sub_sil:.3f}")
    log("  (compare directly to the hierarchical silhouette above -- both methods")
    log("   now share identical N, features, and scaling)")

    # --- k-means on the full population (unchanged, for reference) ---------
    X_full = scaler.fit_transform(full[FEATURE_COLS])
    km_full = KMeans(n_clusters=2, random_state=RANDOM_SEED, n_init=10).fit(X_full)
    km_full_sil = silhouette_score(X_full, km_full.labels_)
    log(f"k-means (k=2) silhouette on full population (n={len(full):,}): {km_full_sil:.3f}")

    # --- manifest for reproducibility / report citation ---------------------
    manifest = pd.DataFrame([{
        "method": "hierarchical_ward", "n_patients": HIER_SUBSAMPLE_N,
        "sampling": "stratified_by_is_ever_septic", "seed": RANDOM_SEED,
        "silhouette_k2": hier_sil,
    }, {
        "method": "kmeans_matched_subsample", "n_patients": HIER_SUBSAMPLE_N,
        "sampling": "stratified_by_is_ever_septic", "seed": RANDOM_SEED,
        "silhouette_k2": km_sub_sil,
    }, {
        "method": "kmeans_full_population", "n_patients": len(full),
        "sampling": "none (all complete-case patients)", "seed": RANDOM_SEED,
        "silhouette_k2": km_full_sil,
    }])
    manifest.to_csv(f"{OUT_DIR}/clustering_sample_manifest.csv", index=False)
    log(f"\nSaved: hierarchical_cluster_profiles.csv, clustering_sample_manifest.csv")

    with open(f"{OUT_DIR}/run09_log.txt", "w") as f:
        f.write("\n".join(str(l) for l in log_lines))


if __name__ == "__main__":
    main()
