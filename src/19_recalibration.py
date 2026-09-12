# %% [markdown]
# # Phase 10 — Post-hoc recalibration (Platt scaling) and threshold re-lock
#
# Script 05's calibration diagnostics (`calibration_results.csv`,
# `figures/calibration_curves.png`) showed every model's raw probabilities
# sitting ~18-24x above the true event rate — a `scale_pos_weight` side
# effect from training. AUROC ranking is unaffected (ranking only cares
# about relative order), but the raw probabilities are not usable as
# probabilities: a raw score of 0.4 does NOT mean a 40% chance of sepsis
# within 6h when true prevalence is ~1.8%.
#
# This script fixes that with post-hoc Platt scaling (logistic recalibration
# of the model's own logit-transformed score), the standard fix precisely
# because it doesn't require retraining any of the five underlying models.
# Isotonic regression is deliberately NOT used here: at ~1.8% prevalence the
# positive-class sample in any held-out calibration split is thin enough
# that isotonic's step function can become unstable, whereas Platt's
# two-parameter (a, b) logistic form is far more stable and gives a much
# cleaner methodological story (one intercept, one slope, both reported).
#
# THE ONE RULE THIS SCRIPT IS STRICT ABOUT: never calibrate on the same
# predictions you evaluate.
#   - OOF models (baseline/engineered/tabnet): predictions are already
#     out-of-fold w.r.t. model TRAINING, but calibration is a second fit on
#     top of that, so it needs its OWN disjoint split. Patients (not hours)
#     are split 70/30 -- calib-fit / calib-eval -- deterministically and
#     stratified on ever-septic, matching the stratified-split convention
#     already used in scripts 09/16/18. Splitting by patient, not by hour,
#     matters because a patient's ~38 hourly rows are highly correlated; an
#     hour-level split would leak the same patient's calibration signal
#     across both sides.
#   - Sequence models (GRU-D / Transformer): calibration is fit on the
#     model's VALIDATION predictions (`{model}_val_predictions.parquet`,
#     produced by the patched 16/18 Kaggle scripts) and evaluated once on
#     the untouched TEST predictions. If a given run predates that patch and
#     no val file exists, this script falls back to splitting the existing
#     test cohort itself and prints/logs a loud, explicit caveat -- see
#     FALLBACK_NOTE below. That fallback still respects "don't calibrate on
#     what you evaluate," but it is not as clean as a genuine train/val/test
#     separation and should be replaced by re-running 16/18 when possible.
#
# Threshold consequence: Platt scaling is monotonic, so AUROC/ranking is
# unchanged, but the NUMBER at which you threshold changes completely (a
# raw threshold of 0.37 has no relationship to the right calibrated
# threshold). So this script re-derives the operating threshold from
# scratch on the calib-fit / val side only, LOCKS it there, and then
# applies that single locked number exactly once to the calib-eval / test
# side -- it never re-optimizes on the evaluation side.

# %%
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utility_score import normalized_utility_score, sweep_thresholds_for_utility

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
EPS = 1e-6          # clip before logit() so p=0/1 don't blow up
CALIB_FIT_FRAC = 0.70  # OOF models: 70% patients to fit, 30% to evaluate

# model -> (oof_predictions_filename, proba_column_name)
OOF_MODELS = {
    "baseline":   ("baseline_oof_predictions.parquet", "baseline_proba"),
    "engineered": ("engineered_oof_predictions.parquet", "engineered_proba"),
    "tabnet":     ("tabnet_oof_predictions.parquet", "tabnet_proba"),
}
# model -> (val_predictions_filename, test_predictions_filename, proba_column_name)
SEQUENCE_MODELS = {
    "grud":        ("grud_val_predictions.parquet", "grud_test_predictions.parquet", "grud_proba"),
    "transformer": ("transformer_val_predictions.parquet", "transformer_test_predictions.parquet", "transformer_proba"),
}

FALLBACK_NOTE = (
    "No {vfile} found under outputs/ (this run of {script} predates the patch "
    "that persists validation-set predictions). Falling back to a patient-level "
    "50/50 split of the {n:,} existing test patients: one half fits Platt scaling "
    "and locks the threshold ('fallback-calib'), the other half is evaluated "
    "exactly once ('fallback-eval'). This still obeys 'never calibrate on what "
    "you evaluate', but it is weaker than a genuine train/val/test separation, "
    "because the fallback-calib half was drawn from the same test population the "
    "headline test-set numbers are computed from (same distributional draw, "
    "even though the specific rows never overlap). RECOMMENDATION: re-run the "
    "patched {script} on Kaggle to get a real {vfile}, then re-run this script -- "
    "it will pick it up automatically and this fallback path won't fire."
)

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(str(msg))


# %% [markdown]
# ## Step 1 — deterministic, documented, patient-level splitting

# %%
def patient_split(df, patient_col="patient_id", label_col="SepsisLabel",
                   frac_a=0.70, seed=RANDOM_STATE):
    """Stratified (on ever-positive-per-patient) two-way split of PATIENTS,
    not hours. Deterministic given `seed`, so it's fully reproducible and
    can be documented (this docstring + the seed IS the documentation)."""
    pat = df.groupby(patient_col)[label_col].max().rename("is_septic").reset_index()
    a_ids, b_ids = set(), set()
    for _, g in pat.groupby("is_septic"):
        g = g.sample(frac=1.0, random_state=seed)  # deterministic shuffle
        cut = int(round(len(g) * frac_a))
        a_ids.update(g.iloc[:cut][patient_col].tolist())
        b_ids.update(g.iloc[cut:][patient_col].tolist())
    return a_ids, b_ids


# %% [markdown]
# ## Step 2 — Platt scaling (fit ONCE on the fit split, reused everywhere else)

# %%
def fit_platt(y, p_raw, eps=EPS):
    """logit(p_calibrated) = a * logit(p_raw) + b, fit as a 1-feature
    unpenalized logistic regression on (y, logit(p_raw)) -- this is
    Platt (1999)'s method applied to a probabilistic classifier's own
    output rather than an SVM margin, which is the standard adaptation."""
    p = np.clip(np.asarray(p_raw, dtype=float), eps, 1 - eps)
    x = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    lr.fit(x, y)
    a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])

    def apply(p_new):
        pn = np.clip(np.asarray(p_new, dtype=float), eps, 1 - eps)
        xn = np.log(pn / (1 - pn)).reshape(-1, 1)
        return lr.predict_proba(xn)[:, 1]

    return apply, {"platt_slope_a": a, "platt_intercept_b": b}


def calib_intercept_slope(y, p, eps=EPS):
    """DIAGNOSTIC ONLY: logistic recalibration of y ~ logit(p), fit and
    read off on the SAME set being diagnosed (same convention already used
    for the pre-existing calibration_results.csv). This is a descriptive
    statistic of that set, not a model reused anywhere else -- it must not
    be confused with fit_platt() above, which is fit on one split and
    APPLIED to a different one."""
    pc = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    x = np.log(pc / (1 - pc)).reshape(-1, 1)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    lr.fit(x, y)
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])


def brier_skill(y, p):
    y = np.asarray(y, dtype=float)
    brier = brier_score_loss(y, p)
    noskill = brier_score_loss(y, np.full_like(p, y.mean(), dtype=float))
    return brier, noskill, (1 - brier / noskill if noskill > 0 else np.nan)


def reliability_points(y, p, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append((float(np.mean(p[m])), float(np.mean(np.asarray(y)[m])), int(m.sum())))
    return pd.DataFrame(rows, columns=["mean_predicted", "mean_observed", "n"])


# %% [markdown]
# ## Step 3 — evaluate one model: fit Platt on `fit_df`, score on `eval_df` only

# %%
def evaluate_model(model_name, fit_df, eval_df, proba_col, protocol):
    y_fit, p_fit_raw = fit_df["SepsisLabel"].to_numpy(), fit_df[proba_col].to_numpy()
    y_eval, p_eval_raw = eval_df["SepsisLabel"].to_numpy(), eval_df[proba_col].to_numpy()

    apply_platt, platt_params = fit_platt(y_fit, p_fit_raw)
    p_fit_cal = apply_platt(p_fit_raw)
    p_eval_cal = apply_platt(p_eval_raw)

    auroc_raw = roc_auc_score(y_eval, p_eval_raw)
    auroc_cal = roc_auc_score(y_eval, p_eval_cal)  # must ~= auroc_raw (monotonic transform)

    brier_raw, noskill, skill_raw = brier_skill(y_eval, p_eval_raw)
    brier_cal, _, skill_cal = brier_skill(y_eval, p_eval_cal)

    icpt_raw, slope_raw = calib_intercept_slope(y_eval, p_eval_raw)
    icpt_cal, slope_cal = calib_intercept_slope(y_eval, p_eval_cal)

    # --- threshold: locked on the FIT side, applied ONCE to the EVAL side ---
    fit_df_raw = fit_df.assign(_p=p_fit_raw)
    fit_df_cal = fit_df.assign(_p=p_fit_cal)
    raw_thr_table = sweep_thresholds_for_utility(fit_df_raw, "patient_id", "SepsisLabel", "_p")
    cal_thr_table = sweep_thresholds_for_utility(fit_df_cal, "patient_id", "SepsisLabel", "_p")
    raw_locked_thr = float(raw_thr_table.iloc[0].threshold)
    cal_locked_thr = float(cal_thr_table.iloc[0].threshold)

    eval_df_raw = eval_df.assign(_p=p_eval_raw)
    eval_df_cal = eval_df.assign(_p=p_eval_cal)
    util_raw = normalized_utility_score(eval_df_raw, "patient_id", "SepsisLabel", proba_col="_p", threshold=raw_locked_thr)
    util_cal = normalized_utility_score(eval_df_cal, "patient_id", "SepsisLabel", proba_col="_p", threshold=cal_locked_thr)

    log(f"\n[{model_name}] protocol={protocol}  n_fit_rows={len(fit_df):,} "
        f"({fit_df['patient_id'].nunique():,} pts)  n_eval_rows={len(eval_df):,} "
        f"({eval_df['patient_id'].nunique():,} pts)")
    log(f"  Platt: logit(p_cal) = {platt_params['platt_slope_a']:.4f} * logit(p_raw) + {platt_params['platt_intercept_b']:.4f}")
    log(f"  AUROC        raw={auroc_raw:.4f}  calibrated={auroc_cal:.4f}  (should match: monotonic transform)")
    log(f"  Brier        raw={brier_raw:.4f}  calibrated={brier_cal:.4f}  (no-skill={noskill:.4f})")
    log(f"  Brier skill  raw={skill_raw:.4f}  calibrated={skill_cal:.4f}")
    log(f"  Calib intercept  raw={icpt_raw:.3f} -> calibrated={icpt_cal:.3f}   (0 = perfectly calibrated in the large)")
    log(f"  Calib slope      raw={slope_raw:.3f} -> calibrated={slope_cal:.3f}   (1 = perfectly calibrated)")
    log(f"  Threshold RAW: locked on fit at p>={raw_locked_thr:.3f} -> eval utility={util_raw:.4f}")
    log(f"  Threshold CAL: locked on fit at p>={cal_locked_thr:.3f} -> eval utility={util_cal:.4f}")

    row = {
        "model": model_name, "protocol": protocol,
        "n_fit_rows": len(fit_df), "n_fit_patients": fit_df["patient_id"].nunique(),
        "n_eval_rows": len(eval_df), "n_eval_patients": eval_df["patient_id"].nunique(),
        "prevalence_eval": float(np.mean(y_eval)),
        "brier_raw": brier_raw, "brier_calibrated": brier_cal, "brier_noskill": noskill,
        "brier_skill_raw": skill_raw, "brier_skill_calibrated": skill_cal,
        "calib_intercept_raw": icpt_raw, "calib_intercept_calibrated": icpt_cal,
        "calib_slope_raw": slope_raw, "calib_slope_calibrated": slope_cal,
        "auroc_raw": auroc_raw, "auroc_calibrated": auroc_cal,
        "platt_slope_a": platt_params["platt_slope_a"], "platt_intercept_b": platt_params["platt_intercept_b"],
        "raw_threshold_locked_on_fit": raw_locked_thr, "eval_utility_at_raw_threshold": util_raw,
        "calibrated_threshold_locked_on_fit": cal_locked_thr, "eval_utility_at_calibrated_threshold": util_cal,
    }

    diagnostics = {
        "reliability_raw": reliability_points(y_eval, p_eval_raw),
        "reliability_cal": reliability_points(y_eval, p_eval_cal),
        "eval_calibrated_frame": eval_df.assign(calibrated_proba=p_eval_cal),
        "fit_calibrated_frame": fit_df.assign(calibrated_proba=p_fit_cal),
        "apply_platt": apply_platt,
    }
    return row, diagnostics


# %% [markdown]
# ## Step 4 — run every OOF model

# %%
def run_oof_models():
    rows, diags, manifests = [], {}, []
    for model_name, (fname, proba_col) in OOF_MODELS.items():
        path = OUT_DIR / fname
        if not path.exists():
            log(f"[{model_name}] SKIPPED -- {fname} not found under outputs/")
            continue
        df = pd.read_parquet(path)
        fit_ids, eval_ids = patient_split(df, frac_a=CALIB_FIT_FRAC, seed=RANDOM_STATE)
        fit_df = df[df["patient_id"].isin(fit_ids)].copy()
        eval_df = df[df["patient_id"].isin(eval_ids)].copy()

        row, diag = evaluate_model(model_name, fit_df, eval_df, proba_col, protocol="oof_patient_split_70_30")
        rows.append(row)
        diags[model_name] = diag

        manifest = pd.concat([
            pd.DataFrame({"patient_id": sorted(fit_ids), "split": "calib_fit"}),
            pd.DataFrame({"patient_id": sorted(eval_ids), "split": "calib_eval"}),
        ], ignore_index=True)
        manifest.insert(0, "model", model_name)
        manifests.append(manifest)

        # persist calibrated predictions for the WHOLE cohort (fit + eval),
        # each row tagged with which side of the split it was on, so anyone
        # downstream can see exactly which rows the calibrator was fit vs
        # applied on.
        full = pd.concat([
            diag["fit_calibrated_frame"].assign(split="calib_fit"),
            diag["eval_calibrated_frame"].assign(split="calib_eval"),
        ], ignore_index=True)
        full = full.rename(columns={proba_col: "raw_proba"})
        full.to_parquet(OUT_DIR / f"{model_name}_calibrated_predictions.parquet", index=False)

    return rows, diags, manifests


# %% [markdown]
# ## Step 5 — run GRU-D / Transformer, preferring val predictions, falling
# ## back to a documented test-set split if val predictions aren't present

# %%
def run_sequence_models():
    rows, diags = [], {}
    script_of = {"grud": "16_grud_model_kaggle.py", "transformer": "18_transformer_model_kaggle.py"}
    for model_name, (val_fname, test_fname, proba_col) in SEQUENCE_MODELS.items():
        test_path = OUT_DIR / test_fname
        if not test_path.exists():
            log(f"[{model_name}] SKIPPED -- {test_fname} not found under outputs/")
            continue
        test_df = pd.read_parquet(test_path)
        val_path = OUT_DIR / val_fname

        if val_path.exists():
            fit_df = pd.read_parquet(val_path)
            eval_df = test_df
            protocol = "sequence_val_then_test"
        else:
            log(FALLBACK_NOTE.format(vfile=val_fname, script=script_of[model_name],
                                      n=test_df["patient_id"].nunique()))
            fit_ids, eval_ids = patient_split(test_df, frac_a=0.50, seed=RANDOM_STATE)
            fit_df = test_df[test_df["patient_id"].isin(fit_ids)].copy()
            eval_df = test_df[test_df["patient_id"].isin(eval_ids)].copy()
            protocol = "sequence_fallback_test_split"

        row, diag = evaluate_model(model_name, fit_df, eval_df, proba_col, protocol=protocol)
        rows.append(row)
        diags[model_name] = diag

        full = diag["eval_calibrated_frame"].rename(columns={proba_col: "raw_proba"})
        full["split"] = "eval" if protocol == "sequence_val_then_test" else "fallback_eval"
        full.to_parquet(OUT_DIR / f"{model_name}_calibrated_predictions.parquet", index=False)

    return rows, diags


# %% [markdown]
# ## Step 6 — reliability diagrams, before vs after, eval-only (never fit-side)

# %%
def plot_reliability(diags, out_path):
    models = list(diags.keys())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, key, title in zip(
        axes, ["reliability_raw", "reliability_cal"],
        ["Raw probabilities (held-out eval set)", "Platt-calibrated probabilities (same held-out eval set)"],
    ):
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
        for m in models:
            pts = diags[m][key]
            ax.plot(pts["mean_predicted"], pts["mean_observed"], marker="o", label=m)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability (bin)")
        ax.set_ylabel("Observed sepsis-within-6h rate")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("Reliability before vs. after Platt scaling -- fit and evaluated on disjoint patients")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log(f"\nSaved {out_path.name}")


# %% [markdown]
# ## Step 7 — assemble the "Phase 3" before/after table + threshold-lock table

# %%
def build_summary_tables(rows):
    df = pd.DataFrame(rows)
    phase3 = df[[
        "model", "protocol", "brier_raw", "brier_calibrated",
        "calib_intercept_raw", "calib_intercept_calibrated",
        "calib_slope_raw", "calib_slope_calibrated",
        "auroc_raw", "auroc_calibrated",
    ]].copy()
    phase3.to_csv(OUT_DIR / "calibration_recalibrated_results.csv", index=False)

    threshold_lock = df[[
        "model", "protocol",
        "raw_threshold_locked_on_fit", "eval_utility_at_raw_threshold",
        "calibrated_threshold_locked_on_fit", "eval_utility_at_calibrated_threshold",
    ]].copy()
    threshold_lock.to_csv(OUT_DIR / "threshold_lock_results.csv", index=False)

    log("\n" + "=" * 100)
    log("PHASE 3 TABLE -- discrimination vs. calibration, before/after (all numbers from held-out eval, never fit-side)")
    log("=" * 100)
    log(phase3.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    log("\nTHRESHOLD RE-LOCK TABLE -- old raw threshold vs. new calibrated threshold, both LOCKED on the fit "
        "side, both evaluated (utility) on the untouched eval side")
    log(threshold_lock.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return phase3, threshold_lock


# %%
if __name__ == "__main__":
    oof_rows, oof_diags, manifests = run_oof_models()
    seq_rows, seq_diags = run_sequence_models()
    all_rows = oof_rows + seq_rows
    all_diags = {**oof_diags, **seq_diags}

    if manifests:
        pd.concat(manifests, ignore_index=True).to_csv(OUT_DIR / "calibration_split_manifest.csv", index=False)
        log(f"\nSaved calibration_split_manifest.csv (deterministic patient-level split, seed={RANDOM_STATE}, "
            f"documents exactly which patients were calib_fit vs calib_eval per OOF model)")

    if all_diags:
        plot_reliability(all_diags, FIG_DIR / "calibration_curves_recalibrated.png")
        build_summary_tables(all_rows)

    with open(OUT_DIR / "run19_log.txt", "w") as f:
        f.write("\n".join(log_lines))
    log(f"\nSaved: calibration_recalibrated_results.csv, threshold_lock_results.csv, "
        f"calibration_split_manifest.csv, {{model}}_calibrated_predictions.parquet x{len(all_rows)}, "
        f"figures/calibration_curves_recalibrated.png, run19_log.txt")
