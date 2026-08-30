# %% [markdown]
# # Phase 9 — Hierarchical clustering (syllabus requires this specific
# method, distinct from the k-means already run in clustering_phenotypes.py)
#
# Reuses the exact same per-patient trajectory summary features as the
# k-means script for a fair, direct comparison. Given the earlier k-means
# result (best silhouette 0.154, clusters not sepsis-differentiated), the
# expectation here is a similarly weak/null result -- reported honestly
# either way, not massaged to look better than it is.

# %%
import duckdb
import pandas as pd
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "warehouse" / "sepsis.duckdb"
OUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Reuse the same per-patient trajectory summary used for k-means
# (mean/std of vitals per patient, is_ever_septic label for post-hoc profiling)

# %%
con = duckdb.connect(str(DB_PATH), read_only=True)
patient_summary = con.execute("""
    SELECT
        patient_id,
        AVG(HR_ffill) AS HR_mean, STDDEV(HR_ffill) AS HR_std,
        AVG(O2Sat_ffill) AS O2Sat_mean, STDDEV(O2Sat_ffill) AS O2Sat_std,
        AVG(Temp_ffill) AS Temp_mean, STDDEV(Temp_ffill) AS Temp_std,
        AVG(SBP_ffill) AS SBP_mean, STDDEV(SBP_ffill) AS SBP_std,
        AVG(MAP_ffill) AS MAP_mean, STDDEV(MAP_ffill) AS MAP_std,
        AVG(DBP_ffill) AS DBP_mean, STDDEV(DBP_ffill) AS DBP_std,
        AVG(Resp_ffill) AS Resp_mean, STDDEV(Resp_ffill) AS Resp_std,
        MIN(HR_ffill) AS HR_min, MAX(HR_ffill) AS HR_max,
        MIN(SBP_ffill) AS SBP_min, MAX(SBP_ffill) AS SBP_max,
        MAX(SepsisLabel) AS is_ever_septic
    FROM fact_features
    GROUP BY patient_id
""").df()
con.close()
print(f"Loaded {len(patient_summary):,} patient trajectory summaries")

# %%
feature_cols = [c for c in patient_summary.columns if c not in ("patient_id", "is_ever_septic")]
X = StandardScaler().fit_transform(patient_summary[feature_cols].fillna(0))

# %% [markdown]
# ## Ward-linkage agglomerative clustering
# Ward minimizes within-cluster variance, generally the most stable linkage
# for this kind of continuous, roughly-Euclidean feature space.

# %%
# Subsample for the linkage/dendrogram step -- full 40k-patient linkage
# matrix is O(n^2) memory and unnecessary to demonstrate the method.
sample_idx = patient_summary.sample(n=min(2000, len(patient_summary)), random_state=42).index
X_sample = X[sample_idx]

Z = linkage(X_sample, method="ward")

plt.figure(figsize=(12, 5))
dendrogram(Z, truncate_mode="lastp", p=20, show_leaf_counts=True)
plt.title("Hierarchical clustering dendrogram (Ward linkage, 2k-patient sample)")
plt.xlabel("Cluster size / sample index")
plt.ylabel("Ward distance")
plt.tight_layout()
plt.savefig(FIG_DIR / "hierarchical_dendrogram.png", dpi=120)
plt.close()
print("Saved dendrogram to outputs/figures/hierarchical_dendrogram.png")

# %% [markdown]
# ## Cut the tree at k=2 (matching k-means for direct comparison) and
# evaluate silhouette on the same sample

# %%
labels = fcluster(Z, t=2, criterion="maxclust")
sil = silhouette_score(X_sample, labels)
print(f"\nHierarchical (Ward, k=2) silhouette: {sil:.3f}")

profile = patient_summary.iloc[sample_idx].copy()
profile["cluster"] = labels
cluster_profile = profile.groupby("cluster")[feature_cols + ["is_ever_septic"]].mean()
cluster_profile["n_patients"] = profile.groupby("cluster").size()
print("\nCluster profile (mean values + sepsis rate):")
print(cluster_profile)

cluster_profile.to_csv(OUT_DIR / "hierarchical_cluster_profiles.csv")
print(f"\nSaved: hierarchical_cluster_profiles.csv")
