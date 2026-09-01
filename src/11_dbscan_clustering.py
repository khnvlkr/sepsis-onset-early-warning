# %% [markdown]
# # Phase 7c — Density-based clustering (DBSCAN)
#
# DWM Module 6 names density-based clustering as its own technique, distinct
# from the partition-based (k-means, clustering_phenotypes.py) and
# hierarchical (09_hierarchical_clustering.py) methods already in the
# pipeline. This script adds DBSCAN on the same per-patient trajectory
# summary (mean/std/min/max of vitals) used by those two scripts, so all
# three methods are directly comparable — same features, same scaling,
# same sepsis-rate-per-cluster profiling.
#
# DBSCAN doesn't take a k — it takes eps (neighborhood radius) and
# min_samples. eps is chosen via the standard k-distance elbow heuristic
# (Ester et al. 1996) rather than picked arbitrarily, and that choice is
# logged and plotted so it's defensible in the report.
#
# Given the existing null result from k-means and hierarchical clustering
# (both find ~flat sepsis rates across clusters, silhouette 0.10-0.16), the
# expectation here is that DBSCAN corroborates that finding rather than
# discovering a clean phenotype — a third independent method reaching the
# same conclusion is itself useful evidence, and is reported honestly either
# way rather than tuned until something "interesting" appears.

# %%
import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "warehouse" / "sepsis.duckdb"
OUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
MIN_SAMPLES = 10  # rule of thumb: ~2 x n_features (7 vitals x ~2 stats families)


# %%
def load_patient_summary():
    """Identical per-patient trajectory summary to clustering_phenotypes.py
    (k-means) and 09_hierarchical_clustering.py, rebuilt here directly from
    fact_ffill rather than depending on those scripts having been run first,
    so this script works standalone. Same feature definitions -> directly
    comparable cluster profiles across all three methods."""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("""
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
        FROM fact_ffill
        GROUP BY patient_id
    """).df()
    con.close()
    return df.dropna()  # complete-case, matches the other two clustering scripts


# %%
def pick_eps_via_kdistance(X, k, out_path):
    """Standard DBSCAN eps heuristic: sort each point's distance to its k-th
    nearest neighbor, plot it, and take the 'elbow' (point of max curvature)
    as eps. Returns the chosen eps and saves the plot for the report."""
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nn.kneighbors(X)
    k_dist = np.sort(distances[:, -1])

    # Elbow via max distance from the line joining the curve's endpoints —
    # simple, standard, and avoids hand-picking a value by eye.
    n = len(k_dist)
    x = np.arange(n)
    line_vec = np.array([n - 1, k_dist[-1] - k_dist[0]])
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    vecs = np.stack([x - 0, k_dist - k_dist[0]], axis=1)
    proj_len = vecs @ line_vec_norm
    proj = np.outer(proj_len, line_vec_norm)
    dist_from_line = np.linalg.norm(vecs - proj, axis=1)
    elbow_idx = int(np.argmax(dist_from_line))
    eps = float(k_dist[elbow_idx])

    plt.figure(figsize=(8, 5))
    plt.plot(x, k_dist)
    plt.axvline(elbow_idx, color="red", linestyle="--", label=f"elbow -> eps={eps:.2f}")
    plt.xlabel(f"points, sorted by distance to {k}-th nearest neighbor")
    plt.ylabel(f"{k}-NN distance")
    plt.title("DBSCAN eps selection (k-distance elbow)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return eps


# %%
def run_dbscan():
    df = load_patient_summary()
    print(f"Loaded {len(df):,} patient trajectory summaries (complete-case)")

    feature_cols = [c for c in df.columns if c not in ("patient_id", "is_ever_septic")]
    X = StandardScaler().fit_transform(df[feature_cols])

    eps = pick_eps_via_kdistance(X, MIN_SAMPLES, FIG_DIR / "dbscan_kdistance_elbow.png")
    print(f"Chosen eps={eps:.3f} (k-distance elbow, min_samples={MIN_SAMPLES})")

    db = DBSCAN(eps=eps, min_samples=MIN_SAMPLES).fit(X)
    labels = db.labels_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    noise_pct = 100 * n_noise / len(labels)
    print(f"DBSCAN found {n_clusters} cluster(s), {n_noise:,} noise points ({noise_pct:.1f}%)")

    # Silhouette is only meaningful with >=2 clusters and excluding noise
    # (silhouette_score is undefined for label -1 by convention).
    non_noise = labels != -1
    if n_clusters >= 2 and non_noise.sum() > n_clusters:
        sil = silhouette_score(X[non_noise], labels[non_noise])
        print(f"Silhouette (non-noise points only): {sil:.3f}")
    else:
        sil = float("nan")
        print("Silhouette not computed: fewer than 2 real clusters after removing noise "
              "-- this itself is a reportable finding (see docstring: density-based "
              "clustering finding little structure is consistent with k-means/hierarchical).")

    df_out = df.assign(dbscan_cluster=labels)
    profile = df_out.groupby("dbscan_cluster").agg(
        {**{c: "mean" for c in feature_cols}, "is_ever_septic": "mean", "patient_id": "count"}
    ).rename(columns={"patient_id": "n_patients"})
    profile.to_csv(OUT_DIR / "dbscan_cluster_profiles.csv")
    print("\nCluster profile (cluster -1 = noise):")
    print(profile[["n_patients", "is_ever_septic"]].to_string())

    manifest = pd.DataFrame([{
        "method": "dbscan",
        "n_patients": len(df),
        "eps": eps,
        "min_samples": MIN_SAMPLES,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_pct": noise_pct,
        "silhouette_non_noise": sil,
    }])
    manifest.to_csv(OUT_DIR / "dbscan_manifest.csv", index=False)

    print(f"\nSaved: dbscan_cluster_profiles.csv, dbscan_manifest.csv, "
          f"figures/dbscan_kdistance_elbow.png")
    return profile, manifest


# %%
if __name__ == "__main__":
    run_dbscan()
