# %% [markdown]
# # Phase 6b — TabNet: does attention beat trees on the *same* feature table?
# ### (Kaggle notebook version — full dataset, run 3 of 4 in the 04 -> 16 -> 17 -> 18 sequence)
#
# **Before running this notebook**, upload a private Kaggle Dataset containing:
#   - `sepsis.duckdb` (the warehouse built by `01_etl_warehouse.py` + `02_feature_engineering.py`)
#   - `utility_score.py` and `delong.py` (copied as-is from `src/` — imported
#     straight from the dataset directory, no separate upload step needed)
#   - `engineered_oof_predictions.parquet` (from `04_engineered_model.py`, optional —
#     only needed for the DeLong's-test comparison against XGBoost)
#   - `shap_feature_importance.csv` (from `05_explainability.py`, optional —
#     only needed for the Spearman feature-ranking comparison)
#
# Then attach that dataset to this notebook (**Notebook > Add Input > Datasets**)
# and turn on **Settings > Accelerator > GPU T4 x2** (or similar) before running —
# TabNet trains via backprop and benefits from it far more than XGBoost does.
# Also turn on **Settings > Internet** so the pip installs below (pytorch-tabnet,
# bottleneck) can actually reach PyPI. Paths are auto-detected below by
# searching for sepsis.duckdb under /kaggle/input, so there's no dataset-name
# variable to set by hand.
#
# Everything below is otherwise the same script as the local version: same
# leakage-safe patient-grouped CV, same utility metric, same DeLong's-test
# helper. The only things that differ are (1) `ROW_SUBSAMPLE_FRAC = None` —
# a 35%-patient-subsample run confirmed the real bottlenecks (nanmedian,
# object-dtype patient_id grouping) are fixed, so this now runs the full
# 1.55M-row table for the strongest defensible number, expected to finish
# in ~45min-1hr rather than the original ~2hr — and (2) input/output paths
# auto-detected under `/kaggle/input/` and pointed at `/kaggle/working/`
# instead of the local repo layout.
#
# Script 04 trained XGBoost on `fact_features` (289 columns: forward-filled
# raw values, rolling stats, slopes/velocity, missingness flags, clinical
# ratios). This script is a drop-in swap of the model only — TabNet (Arik &
# Pfister, 2021, "TabNet: Attentive Interpretable Tabular Learning", AAAI)
# reads the exact same 289-column table, the exact same patient-grouped
# folds, and is scored with the exact same utility metric and DeLong's-test
# helper. No new SQL, no new feature engineering.
#
# TabNet uses sequential attention (learned sparse feature masks, one per
# decision step) to pick which features matter at each step, instead of
# greedy axis-aligned splits. The question this script answers is explicitly
# a question, not an assumed win: does that buy anything on this table, or
# does it reproduce the well-published finding that gradient-boosted trees
# tend to beat deep tabular architectures on structured/tabular data at
# moderate sample sizes (Shwartz-Ziv & Armon, 2022, "Tabular Data: Deep
# Learning is Not All You Need")? Either answer is a legitimate result.
#
# Two things TabNet needs that XGBoost's tree splits don't:
#   1. No native missing-value handling -> every NaN must be imputed before
#      it reaches the network. `fact_features` already carries explicit
#      `_missing` / `_hours_since_last` columns for every rolling column
#      (script 02), so imputing the *value* columns with the training
#      fold's median should not destroy the missingness signal -- the model
#      still has it available as its own feature, same as XGBoost does.
#   2. Inputs are actually used at their numeric scale (unlike a tree split,
#      which only cares about relative order), so every fold is
#      standardized (fit on that fold's training rows only, to avoid
#      leaking test-fold statistics).
#
# Also worth comparing directly: script 05's SHAP ranking (from XGBoost) vs.
# TabNet's own built-in `feature_importances_` (aggregated attention masks).
# A high rank correlation would say both architectures converge on the same
# signal from two different mechanisms; a low one says they're leaning on
# the table differently. Spearman's rho is reported at the bottom, in the
# same spirit as script 16's decay-rate-vs-SHAP comparison.

# %%
import gc
import importlib.util
import subprocess
import sys
import time
import warnings
from pathlib import Path

# Kaggle notebooks start from a fresh environment each session -- pytorch-tabnet
# isn't preinstalled. Installing via subprocess (not a `!pip`/`%pip` magic)
# means this cell runs the same way whether it's executed as a notebook cell
# or as a plain `python 17_tabnet_model_kaggle.py` script.
if importlib.util.find_spec("pytorch_tabnet") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytorch-tabnet"], check=True)

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

# np.nanmedian on a large 2D array (millions of rows x hundreds of columns)
# is a known numpy perf trap -- its column-wise NaN-aware median isn't well
# vectorized and can silently take many minutes per call. It was being
# called once per fold with zero logging, which is exactly the kind of gap
# that doesn't show up in the per-epoch fit() timer but eats huge wall time
# between folds. `bottleneck`'s C implementation is ~10-50x faster for this;
# fall back to numpy if it's unavailable rather than hard-failing.
if importlib.util.find_spec("bottleneck") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bottleneck"], check=True)
try:
    import bottleneck as bn
    fast_nanmedian = bn.nanmedian
except ImportError:
    fast_nanmedian = np.nanmedian

warnings.filterwarnings("ignore")

# ---- Kaggle paths ---------------------------------------------------------
# Previously hardcoded as /kaggle/input/<KAGGLE_DATASET_SLUG>/ -- that
# assumed Kaggle's older flat input layout. Current Kaggle notebooks nest
# attached datasets under /kaggle/input/datasets/<slug>/ instead (a real
# platform layout change, not a config mistake), so rather than keep
# guessing the exact nesting convention, search for sepsis.duckdb under
# /kaggle/input directly and use wherever it actually is.
KAGGLE_ROOT = Path("/kaggle/input")
_candidates = sorted(KAGGLE_ROOT.glob("**/sepsis.duckdb"))
if not _candidates:
    sys.exit(
        f"Couldn't find sepsis.duckdb anywhere under {KAGGLE_ROOT}.\n"
        f"Check that your Kaggle Dataset (containing sepsis.duckdb, "
        f"utility_score.py, delong.py, etc.) is attached via "
        f"Notebook > Add Input > Datasets."
    )
if len(_candidates) > 1:
    print(f"Warning: found {len(_candidates)} sepsis.duckdb files under {KAGGLE_ROOT}, "
          f"using the first one: {_candidates[0]}")
DB_PATH = _candidates[0]
KAGGLE_INPUT_DIR = DB_PATH.parent
PRIOR_OUTPUTS_DIR = KAGGLE_INPUT_DIR   # where engineered_oof_predictions.parquet / shap_feature_importance.csv live (read-only), if this notebook is a standalone continuation of an earlier session
OUT_DIR = Path("/kaggle/working/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Using dataset directory: {KAGGLE_INPUT_DIR}")

sys.path.insert(0, str(KAGGLE_INPUT_DIR))
from utility_score import normalized_utility_score, sweep_thresholds_for_utility
from delong import delong_roc_test


def find_prior_output(filename):
    """Look for another script's output first in this session's own working
    outputs (if it already ran earlier in this same Kaggle session -- e.g.
    script 04 immediately before this one), then fall back to the attached
    input dataset (if it was produced in an earlier, separate Kaggle run)."""
    for candidate in (OUT_DIR / filename, PRIOR_OUTPUTS_DIR / filename):
        if candidate.exists():
            return candidate
    return None

# ---- config -------------------------------------------------------------
N_FOLDS = 5                 # must match 03/04 so the DeLong comparison is fold-aligned
RANDOM_STATE = 42           # matches every other script's RANDOM_STATE/seed
INNER_VAL_FRAC = 0.15       # patient-level split carved out of each fold's train for early stopping
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[17_tabnet_model_kaggle] torch device: {DEVICE}"
      + ("" if DEVICE == "cuda" else
         "  -- no GPU detected. On Kaggle: Notebook Settings > Accelerator > "
         "GPU T4 x2 (or similar), then re-run. TabNet trains via backprop and "
         "benefits from a GPU far more than XGBoost does."))

# ROW_SUBSAMPLE_FRAC: fraction of fact_features rows to keep. The local
# version of this script sets this to 0.15 as a memory patch for an 8GB-RAM# laptop that OOM'd at N=1.55M rows x 289 float32 cols. The original
# assumption here was that Kaggle's larger RAM budget removed that
# constraint entirely -- in practice, a full run (ROW_SUBSAMPLE_FRAC=None)
# took ~2 hours to get through 4 folds despite per-epoch fit() timings
# staying fast and unchanged (~26s/epoch), which pointed at a huge,
# previously-unlogged time sink in per-fold preprocessing. Root-caused and
# fixed: (1) np.nanmedian on the full table -> bottleneck.nanmedian, and
# (2) patient_id as an object-dtype string array -> factorized to int32
# before any GroupKFold/np.unique/np.isin call. A 35%-subsample run after
# both fixes finished all 5 folds in ~17 minutes with split=0.0s and
# nanmedian~1s per fold, confirming those were the actual bottlenecks, not
# Kaggle's compute/RAM budget. Back to the full table now for the strongest
# defensible number in the final report -- expect roughly 3x the subsampled
# run's wall time (~45min-1hr), not the original ~2hr+.
ROW_SUBSAMPLE_FRAC = None

# Lighter than pytorch-tabnet's own defaults (n_d=n_a=8 is the library
# default; 16 trades a little capacity for a smaller compute graph -- this
# matters more than it looks since it's refit 5 times). Increase if your
# machine has more headroom -- see script 16's compute-constraint
# disclosure for the same reasoning.
TABNET_PARAMS = dict(
    n_d=16, n_a=16, n_steps=3, gamma=1.5, lambda_sparse=1e-4,
    # Was lr=2e-2 with StepLR(step_size=10, gamma=0.9) + patience=6: every
    # fold's val_auc peaked at epoch 0-2 then degraded (classic overfit),
    # and since patience=6 always fired before epoch 10, the scheduler
    # never actually decayed the LR once -- the whole run trained at the
    # initial (too-high) LR the entire time. Lower initial LR + a shorter
    # step_size so the decay gets at least 1-2 chances to fire before
    # early stopping can end the run.
    optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=8e-3),
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    scheduler_params=dict(step_size=3, gamma=0.8),
    mask_type="sparsemax", seed=RANDOM_STATE, verbose=1,  # verbose=1: prints per-epoch, so it never LOOKS stuck
    device_name=DEVICE,
)
# patience raised 6->8 to match: give the now-more-active scheduler enough
# epochs to actually show whether decayed LR stabilizes val_auc before
# early stopping decides the run is done.
FIT_PARAMS = dict(max_epochs=40, patience=8, batch_size=2048, virtual_batch_size=256)

# compute_importance=False: pytorch-tabnet's fit() otherwise runs an extra,
# undocumented-in-cost full forward pass over the *entire* training set at
# the end of every fold (abstract_model.py: `if self.compute_importance:
# self.feature_importances_ = self._compute_feature_importances(X_train)`,
# which calls `explain()` -> `network.forward_masks()` batch-by-batch with
# CPU-side sparse-matrix bookkeeping per batch). run17's [timing] line
# showed this costing ~520-580s per fold on the full 1.24M-row training
# set -- more than the entire epoch loop that preceded it. We still want
# feature importances (top-15 table, Spearman-vs-SHAP comparison below),
# just not computed on every training row: IMPORTANCE_SUBSAMPLE_FRAC controls
# how much of X_tr gets passed to the public `explain()` API ourselves,
# after fit(), instead of letting the library do it on the full set.
IMPORTANCE_SUBSAMPLE_FRAC = 0.10

NON_FEATURE_COLS = {"patient_id", "hospital_id", "hour", "ICULOS", "SepsisLabel"}

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(str(msg))


# %% [markdown]
# ## Step 1 — load the same `fact_features` table script 04 used

# %%
def load_engineered_frame():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    schema = con.execute("SELECT * FROM fact_features LIMIT 0").df()
    all_cols = list(schema.columns)
    numeric_cols = [c for c in all_cols if c not in NON_FEATURE_COLS]
    select_parts = [
        f'CAST("{c}" AS FLOAT) AS "{c}"' if c in numeric_cols else f'"{c}"'
        for c in all_cols
    ]
    query = f"SELECT {', '.join(select_parts)} FROM fact_features"
    if ROW_SUBSAMPLE_FRAC is not None:
        # sample by patient_id, not by row, so a patient's hours stay
        # together -- GroupKFold still needs every hour for a patient it
        # includes, it just includes fewer patients overall.
        n_patients_full = con.execute("SELECT COUNT(DISTINCT patient_id) FROM fact_features").fetchone()[0]
        query = f"""
            SELECT {', '.join(select_parts)} FROM fact_features
            WHERE patient_id IN (
                SELECT patient_id FROM (
                    SELECT DISTINCT patient_id FROM fact_features
                ) USING SAMPLE {ROW_SUBSAMPLE_FRAC * 100}% (bernoulli, {RANDOM_STATE})
            )
        """
    df = con.execute(query).df()
    con.close()
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    if ROW_SUBSAMPLE_FRAC is not None:
        print(f"ROW_SUBSAMPLE_FRAC={ROW_SUBSAMPLE_FRAC}: kept {df['patient_id'].nunique():,} of "
              f"{n_patients_full:,} patients ({df.shape[0]:,} rows) -- RAM-tractability disclosure "
              f"above the load function.")
    return df, feature_cols


# %% [markdown]
# ## Step 2 — per-fold: inner train/val split (patient-grouped), impute, scale, fit
#
# XGBoost's `GroupKFold` in scripts 03/04 needs no validation split (k-fold
# CV is its own model selection). TabNet needs a held-out `eval_set` for
# early stopping, so each outer fold's training patients are further split
# 85/15 by patient before fitting -- the outer test fold is never touched
# until prediction, so this is a strict superset of the same patient-level
# leakage guard scripts 03/04/16 all assert.

# %%
def fit_fold(X_train, y_train, groups_train, X_test):
    t_split = time.time()
    train_patients = np.unique(groups_train)
    tr_p, val_p = train_test_split(
        train_patients, test_size=INNER_VAL_FRAC, random_state=RANDOM_STATE
    )
    tr_mask = np.isin(groups_train, tr_p)
    val_mask = np.isin(groups_train, val_p)

    scaler = StandardScaler()
    t_median = time.time()
    medians = fast_nanmedian(X_train[tr_mask], axis=0)
    t_median_done = time.time()

    # In-place impute + scale: the previous version did
    # `np.where(...)` (new array) then `scaler.fit_transform(...)` (another
    # new array), so the largest split (train) briefly existed 2-3x over in
    # memory at once. Copying the slice once up front and then mutating that
    # one buffer keeps a single float32 copy alive per split instead of
    # stacking transient copies during imputation/scaling.
    def impute_scale(X, fit=False):
        X = X.copy()  # the one deliberate copy, so we don't mutate X_train/X_test slices in place
        nan_mask = np.isnan(X)
        if nan_mask.any():
            col_idx = np.where(nan_mask)[1]
            X[nan_mask] = medians[col_idx]
        del nan_mask
        if fit:
            scaler.fit(X)
        X -= scaler.mean_
        X /= scaler.scale_
        return X

    t_impute = time.time()
    X_tr = impute_scale(X_train[tr_mask], fit=True)
    X_val = impute_scale(X_train[val_mask])
    X_te = impute_scale(X_test)
    t_impute_done = time.time()

    y_tr, y_val = y_train[tr_mask], y_train[val_mask]

    log(f"  [timing] split={t_median - t_split:.1f}s  "
        f"nanmedian={t_median_done - t_median:.1f}s  "
        f"impute_scale={t_impute_done - t_impute:.1f}s")

    model = TabNetClassifier(**TABNET_PARAMS)
    model.fit(
        X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric=["auc"],
        weights=1,  # TabNet's built-in equivalent of XGBoost's scale_pos_weight
        compute_importance=False,  # see IMPORTANCE_SUBSAMPLE_FRAC comment above
        **FIT_PARAMS,
    )
    t_fit_done = time.time()

    # [timing] predict/importance/cleanup: run17's Kaggle log showed a silent
    # 500-600s gap per fold between the "Early stopping occurred" print and
    # the next `log()` line (the fold's AUROC), with nothing in between to
    # explain it. Root-caused: with compute_importance defaulting to True,
    # fit() itself was running a full attention-mask forward pass over all
    # ~1.24M training rows internally before returning -- not predict_proba,
    # not the property access, not cleanup, all of which were already fast
    # (confirmed by the [timing] line's own predict=4.3s/feature_importances=
    # 0.0s/cleanup=0.2s while fit=870.4s absorbed the whole gap). Now that
    # compute_importance=False skips that internal pass, `fit` here should
    # roughly match the epoch loop's own printed elapsed time, and the
    # replacement subsampled explain() call below is timed on its own line.
    proba_test = model.predict_proba(X_te)[:, 1]
    t_predict_done = time.time()

    # Manual, subsampled replacement for the fit()-internal importance
    # computation: same public explain() API pytorch-tabnet's own
    # _compute_feature_importances uses, just over IMPORTANCE_SUBSAMPLE_FRAC
    # of X_tr instead of all of it. Sampling noise here is a non-issue since
    # run_tabnet() already averages fold_importances across all 5 folds.
    rng = np.random.default_rng(RANDOM_STATE)
    n_sub = max(1, int(len(X_tr) * IMPORTANCE_SUBSAMPLE_FRAC))
    sub_idx = rng.choice(len(X_tr), size=n_sub, replace=False)
    M_explain, _ = model.explain(X_tr[sub_idx], normalize=False)
    sum_explain = M_explain.sum(axis=0)
    importances = sum_explain / np.sum(sum_explain)
    t_importance_done = time.time()

    del X_tr, X_val, X_te, y_tr, y_val, scaler
    gc.collect()
    t_cleanup_done = time.time()

    log(f"  [timing] fit={t_fit_done - t_impute_done:.1f}s  "
        f"predict={t_predict_done - t_fit_done:.1f}s  "
        f"feature_importances(n={n_sub:,})={t_importance_done - t_predict_done:.1f}s  "
        f"cleanup={t_cleanup_done - t_importance_done:.1f}s")
    return proba_test, importances


# %% [markdown]
# ## Step 3 — run all folds, evaluate, compare to XGBoost, compare feature rankings

# %%
def run_tabnet():
    t0 = time.time()
    df, feature_cols = load_engineered_frame()
    log(f"Engineered frame: {df.shape[0]:,} rows, {len(feature_cols)} features "
        f"(same fact_features table as 04_engineered_model.py) "
        f"[loaded in {time.time() - t0:.1f}s]")

    X = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    y = df["SepsisLabel"].astype(int).to_numpy()
    # patient_id is a Python-object string array ("p012345", ...); np.unique/
    # np.isin/train_test_split on ~1.24M object-dtype elements per fold fall
    # back to Python-level string comparisons instead of vectorized ops --
    # this is what fit_fold's new [timing] line just caught as a 627s "split"
    # step. Factorizing to int32 codes once, up front, makes every one of
    # those group-membership operations run on native integers instead.
    groups, _ = pd.factorize(df["patient_id"])
    groups = groups.astype(np.int32)

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_proba = np.zeros(len(df))
    fold_importances = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        train_patients = set(groups[train_idx])
        test_patients = set(groups[test_idx])
        assert train_patients.isdisjoint(test_patients), "PATIENT LEAKAGE DETECTED"

        proba_test, importances = fit_fold(X[train_idx], y[train_idx], groups[train_idx], X[test_idx])
        oof_proba[test_idx] = proba_test
        fold_importances.append(importances)

        auc = roc_auc_score(y[test_idx], proba_test)
        ap = average_precision_score(y[test_idx], proba_test)
        log(f"  fold {fold}: AUROC={auc:.4f}  AUPRC={ap:.4f}  "
            f"(train={len(train_patients):,} pts / test={len(test_patients):,} pts)")

        # `model` no longer escapes fit_fold at all (fit_fold now returns the
        # already-extracted feature_importances_ array instead of the model
        # object itself), so there's one less large object -- optimizer
        # state, network weights, TabNet's internal explain-matrix buffers --
        # to keep alive between folds. `importances` is a small (n_features,)
        # array already appended to fold_importances above, safe to drop too.
        del proba_test, importances, train_patients, test_patients
        gc.collect()

    df["tabnet_proba"] = oof_proba
    overall_auc = roc_auc_score(y, oof_proba)
    overall_ap = average_precision_score(y, oof_proba)
    log(f"\n[TEST] TabNet OOF AUROC={overall_auc:.4f}  AUPRC={overall_ap:.4f}  "
        f"(n_features={len(feature_cols)})")

    thr_table = sweep_thresholds_for_utility(
        df, patient_col="patient_id", label_col="SepsisLabel", proba_col="tabnet_proba"
    )
    best_row = thr_table.iloc[0]
    log(f"Best-threshold normalized utility = {best_row.normalized_utility:.4f} "
        f"at threshold={best_row.threshold:.2f}")

    # --- DeLong's test vs XGBoost engineered model (script 04's OOF predictions) ---
    results_row = {
        "model": "tabnet", "auroc": overall_auc, "auprc": overall_ap,
        "best_threshold": best_row.threshold, "normalized_utility": best_row.normalized_utility,
        "n_features": len(feature_cols),
    }
    engineered_path = find_prior_output("engineered_oof_predictions.parquet")
    if engineered_path is not None:
        xgb_df = pd.read_parquet(engineered_path)
        merged = df[["patient_id", "hour", "SepsisLabel", "tabnet_proba"]].merge(
            xgb_df[["patient_id", "hour", "engineered_proba"]], on=["patient_id", "hour"], how="inner"
        )
        test_result = delong_roc_test(
            merged["SepsisLabel"].to_numpy(),
            merged["engineered_proba"].to_numpy(),
            merged["tabnet_proba"].to_numpy(),
        )
        log(f"\nDeLong's test (XGBoost engineered vs TabNet, {len(merged):,} paired "
            f"predictions, {merged['patient_id'].nunique():,} patients): "
            f"AUC {test_result['auc_a']:.4f} -> {test_result['auc_b']:.4f}, "
            f"z={test_result['z']:.2f}, p={test_result['p_value']:.2e}")
        results_row.update(test_result)
    else:
        log(f"\n(No engineered_oof_predictions.parquet found in /kaggle/working/outputs "
            f"or the attached input dataset -- run 04_engineered_model_kaggle.py first "
            f"to get the DeLong comparison against XGBoost.)")

    # --- feature-ranking comparison: TabNet's attention-mask importances vs script 05's SHAP ---
    mean_importance = np.mean(fold_importances, axis=0)
    tabnet_rank = pd.DataFrame({
        "feature": feature_cols, "tabnet_importance": mean_importance
    }).sort_values("tabnet_importance", ascending=False).reset_index(drop=True)
    tabnet_rank.to_csv(OUT_DIR / "tabnet_feature_importance.csv", index=False)

    log("\nTop 15 features by mean TabNet attention-mask importance "
        "(averaged across 5 folds):")
    log(tabnet_rank.head(15).to_string(index=False))

    shap_path = find_prior_output("shap_feature_importance.csv")
    if shap_path is not None:
        shap_rank = pd.read_csv(shap_path).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        common = sorted(set(tabnet_rank["feature"]) & set(shap_rank["feature"]))
        tn_pos = tabnet_rank.set_index("feature").loc[common]
        sh_pos = shap_rank.set_index("feature").loc[common]
        rho, p = spearmanr(tn_pos["tabnet_importance"], sh_pos["mean_abs_shap"])
        top15_overlap = len(
            set(tabnet_rank["feature"].head(15)) & set(shap_rank["feature"].head(15))
        )
        log(f"\nSpearman rank correlation, TabNet importance vs. XGBoost SHAP "
            f"importance, over {len(common)} shared features: "
            f"rho={rho:.3f}, p={p:.2e}")
        log(f"Top-15 feature overlap between TabNet and SHAP rankings: "
            f"{top15_overlap}/15")
        results_row["shap_rank_spearman_rho"] = rho
        results_row["shap_rank_spearman_p"] = p
        results_row["top15_overlap_with_shap"] = top15_overlap
    else:
        log(f"\n(No shap_feature_importance.csv found in /kaggle/working/outputs "
            f"or the attached input dataset -- run 05_explainability.py first, or add "
            f"its output to the dataset, to get the SHAP-vs-TabNet ranking comparison.)")

    pd.DataFrame([results_row]).to_csv(OUT_DIR / "tabnet_results.csv", index=False)
    df[["patient_id", "hour", "SepsisLabel", "tabnet_proba"]].to_parquet(
        OUT_DIR / "tabnet_oof_predictions.parquet"
    )
    with open(OUT_DIR / "run17_log.txt", "w") as f:
        f.write("\n".join(log_lines))
    log(f"\nSaved: tabnet_results.csv, tabnet_oof_predictions.parquet, "
        f"tabnet_feature_importance.csv, run17_log.txt ({time.time() - t0:.1f}s total)")
    return results_row


# %%
if __name__ == "__main__":
    run_tabnet()
