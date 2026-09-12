# %% [markdown]
# # Phase 6 — GRU-D: does a network learn the missingness signal end-to-end?
# ### (Kaggle notebook version — full dataset, run 2 of 4 in the 04 -> 16 -> 17 -> 18 sequence)
#
# **Before running this notebook**, upload a private Kaggle Dataset containing
# `sepsis.duckdb`, `utility_score.py`, and `delong.py`, attach it via
# **Notebook > Add Input > Datasets**, and turn on **Settings > Accelerator >
# GPU T4 x2** (or similar). Paths are auto-detected under `/kaggle/input`.
# Run this script after 04 — it optionally uses 04's `engineered_oof_predictions.parquet`
# for the vs-XGBoost DeLong comparison, found automatically under
# `/kaggle/working/outputs` if both scripts run in the same session, or under
# the attached dataset otherwise. Script 17 and 18 depend on this script's
# outputs in turn (18 needs `grud_test_predictions.parquet`).
#
# Script 02 hand-computed `{vital}_hours_since_last` in SQL as a missingness
# feature. Script 05's SHAP analysis then found that one of those hand-built
# features — `Lactate_hours_since_last` — ranked #3 model-wide by mean |SHAP|,
# right behind `Lactate_max_12h` and `partial_sirs_score`. That's a real
# empirical finding: *when* a lab was last drawn carries predictive signal on
# its own, not just what it read.
#
# GRU-D (Che et al., 2018, "Recurrent Neural Networks for Multivariate Time
# Series with Missing Values", Scientific Reports) is a GRU variant built
# specifically around that idea: it has a learned per-variable decay term
#     gamma_x = exp(-relu(W_x * delta + b_x))
# that decides, at every hour, how much to still trust the last reading of
# a variable given how long ago it was drawn — decaying the imputed value
# toward the population mean the longer a lab has gone unmeasured. This is
# the same concept as `Lactate_hours_since_last`, but learned as part of the
# architecture instead of hand-computed in SQL.
#
# This script is therefore not "swap XGBoost for a fancier model" — it's a
# direct test of whether GRU-D rediscovers the same insight end-to-end. The
# check at the bottom of this script ranks vitals by their *learned* decay
# rate and compares that ranking to script 05's SHAP `hours_since_last`
# ranking (Lactate #3, Bilirubin_total #6).
#
# Baseline for comparison: 04_engineered_model.py's XGBoost model (same
# DeLong's-test helper reused, same normalized-utility metric reused, same
# `patient_id` grouping discipline). Unlike 03/04, GRU-D reads straight from
# `fact_vitals_hourly` (raw, pre-script-02) rather than `fact_features` —
# deliberately, since the whole point is that the network should learn decay
# itself rather than being handed script 02's `_hours_since_last` columns.
#
# EVALUATION PROTOCOL — Kaggle full-dataset version:
# The local version of this script disclosed that a full `GroupKFold(5)`
# over all ~40,336 patients, refit per fold, was not tractable for a
# recurrent BPTT model on an 7.4GB-RAM laptop, and trained on a stratified
# subsample of N_SUBSAMPLE_PATIENTS=20,000 instead. On Kaggle (GPU +
# considerably more RAM), that constraint doesn't apply the same way, so
# this version sets N_SUBSAMPLE_PATIENTS = None to use the FULL population
# of patients, still split 70/15/15 train/val/test by patient (GRU-D still
# needs a held-out validation set for early stopping, which XGBoost's
# k-fold CV in scripts 03/04 doesn't need) and still stratified on
# `is_ever_septic` for the split itself. Script 18 (Transformer) mirrors
# this same full-population choice so the two sequence models remain
# trained/evaluated on identical patients.

# %%
import gc
import sys
import time
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    sys.exit(
        "PyTorch is required for this script. On Kaggle it's preinstalled --\n"
        "if this fires anyway, check Settings > Environment."
    )

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

# ---- Kaggle paths ---------------------------------------------------------
# Same auto-detection convention as 17_tabnet_model_kaggle.py: search for
# sepsis.duckdb under /kaggle/input rather than hardcoding a dataset slug,
# since Kaggle's exact attached-dataset nesting has changed before.
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
OUT_DIR = Path("/kaggle/working/outputs")
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
print(f"Using dataset directory: {KAGGLE_INPUT_DIR}")

sys.path.insert(0, str(KAGGLE_INPUT_DIR))
from utility_score import normalized_utility_score, sweep_thresholds_for_utility
from delong import delong_roc_test, delong_ci_line


def find_prior_output(filename):
    """Look for another script's output first in this session's own working
    outputs (if it already ran earlier in this same Kaggle session -- e.g.
    script 04 immediately before this one), then fall back to the attached
    input dataset (if it was produced in an earlier, separate Kaggle run)."""
    for candidate in (OUT_DIR / filename, KAGGLE_INPUT_DIR / filename):
        if candidate.exists():
            return candidate
    return None


print(f"[16_grud_model_kaggle] torch device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}"
      + ("" if torch.cuda.is_available() else
         "  -- no GPU detected. On Kaggle: Notebook Settings > Accelerator > "
         "GPU T4 x2 (or similar), then re-run. This script trains via backprop "
         "over the full patient population and benefits a lot from a GPU."))

# ---- config -----------------------------------------------------------------
N_SUBSAMPLE_PATIENTS = None      # None = use the FULL patient population (Kaggle has the RAM/
                                  # GPU headroom the local 7.4GB-RAM machine didn't); the local
                                  # version subsampled to 20,000 for laptop tractability (see
                                  # script 18's matching change to stay comparable)
MAX_SEQ_LEN = 336               # PhysioNet 2019 max ICULOS
RANDOM_STATE = 42               # matches every other script's RANDOM_STATE/seed
BATCH_SIZE = 64
HIDDEN_SIZE = 64
EPOCHS = 20
PATIENCE = 4                    # early stopping on val AUPRC
LR = 1e-3
WEIGHT_DECAY = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Exact raw column layout from 01_etl_warehouse.py's VITAL_COLS + LAB_COLS
# (34 columns; fact_vitals_hourly also has patient_id, hospital_id, hour,
# ICULOS, SepsisLabel, which are NOT in this list).
VITAL_COLS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LAB_COLS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]
RAW_VITALS = VITAL_COLS + LAB_COLS

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(str(msg))


# %% [markdown]
# ## Step 1 — stratified patient subsample (same helper as script 09)

# %%
def stratified_subsample(df, n, seed, strat_col="is_ever_septic"):
    """Identical logic/rationale to 09_hierarchical_clustering.py's
    stratified_subsample: manual per-group .sample() rather than
    groupby().apply(), to sidestep pandas' include_groups behavior, and to
    pin the sepsis rate in the subsample rather than leave it to chance."""
    frac = min(1.0, n / len(df))
    parts = []
    for _, group in df.groupby(strat_col):
        parts.append(group.sample(frac=frac, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def load_patient_subsample(con):
    patients = con.execute(
        "SELECT patient_id, is_ever_septic FROM dim_patient"
    ).df()
    if N_SUBSAMPLE_PATIENTS is None:
        # Full population, still shuffled (frac=1.0 sample) rather than
        # taken in table order, and still seeded, for parity with the
        # subsampled path's determinism.
        sub = stratified_subsample(patients, len(patients), RANDOM_STATE)
        log(
            f"\nFULL-DATASET RUN (Kaggle): training on all {len(patients):,} patients -- "
            f"no subsampling. The local version of this script disclosed a stratified\n"
            f"  subsample of n=20,000 patients as a laptop-RAM tractability workaround\n"
            f"  (7.4GB RAM machine that OOM'd on script 04); that constraint doesn't apply\n"
            f"  on Kaggle, so this run uses the full population, split 70/15/15\n"
            f"  train/val/test by patient (stratified on is_ever_septic)."
        )
    else:
        sub = stratified_subsample(patients, N_SUBSAMPLE_PATIENTS, RANDOM_STATE)
        log(
            f"\nEVALUATION PROTOCOL DISCLOSURE:\n"
            f"  Full GroupKFold(5) refit-per-fold over all {len(patients):,} patients is not\n"
            f"  tractable for a BPTT-trained recurrent model on this machine (7.4GB RAM --\n"
            f"  the same machine that OOM'd on script 04). Training below therefore runs on\n"
            f"  a stratified random subsample of n={N_SUBSAMPLE_PATIENTS:,} patients (seed=\n"
            f"  {RANDOM_STATE}), stratified on is_ever_septic, split 70/15/15 train/val/test\n"
            f"  by patient. This mirrors the disclosed subsampling precedent already set by\n"
            f"  09_hierarchical_clustering.py."
        )
    log(f"  Sepsis rate -- full population: {patients['is_ever_septic'].mean():.4f}, "
        f"subsample: {sub['is_ever_septic'].mean():.4f}")
    return sub["patient_id"].tolist()


def load_raw_sequences(con, patient_ids):
    query = f"""
        SELECT patient_id, hour, {", ".join(RAW_VITALS)}, SepsisLabel
        FROM fact_vitals_hourly
        WHERE patient_id = ANY(?)
        ORDER BY patient_id, hour
    """
    df = con.execute(query, [patient_ids]).df()
    log(f"Pulled {len(df):,} raw patient-hours for {df['patient_id'].nunique():,} patients "
        f"({len(RAW_VITALS)} raw vitals/labs, straight from fact_vitals_hourly -- pre-script-02)")
    return df


# %% [markdown]
# ## Step 2 — build GRU-D tensors: value, mask, causal delta, label
#
# `delta[t, d]` = hours since variable d was last observed, computed causally
# (delta[0]=0; a run of missing hours accumulates). `hour` in
# fact_vitals_hourly is a 0-indexed, always-consecutive integer offset per
# patient (per 01_etl_warehouse.py), so a fixed 1-hour step is exact here --
# no need to diff against wall-clock gaps.

# %%
def build_patient_arrays(g, vitals):
    vals = g[vitals].to_numpy(dtype=np.float32)
    mask = (~np.isnan(vals)).astype(np.float32)
    T, D = vals.shape

    delta = np.zeros((T, D), dtype=np.float32)
    for d in range(D):
        last_seen = 0.0
        for t in range(1, T):
            last_seen = 1.0 if mask[t - 1, d] == 1 else last_seen + 1.0
            delta[t, d] = last_seen

    return (
        np.nan_to_num(vals, nan=0.0),
        mask,
        delta,
        g["SepsisLabel"].to_numpy(dtype=np.float32),
        g["hour"].to_numpy(),
    )


def build_dataset(df, vitals):
    sequences = []
    for pid, g in df.groupby("patient_id", sort=False):
        g = g.sort_values("hour")
        if len(g) > MAX_SEQ_LEN:
            g = g.iloc[:MAX_SEQ_LEN]
        vals, mask, delta, labels, hours = build_patient_arrays(g, vitals)
        sequences.append({"patient_id": pid, "values": vals, "mask": mask,
                           "delta": delta, "labels": labels, "hour": hours})
    return sequences


class SepsisSeqDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        s = self.sequences[idx]
        return (
            torch.from_numpy(s["values"]), torch.from_numpy(s["mask"]),
            torch.from_numpy(s["delta"]), torch.from_numpy(s["labels"]),
            s["values"].shape[0],
        )


def collate(batch):
    values, mask, delta, labels, lengths = zip(*batch)
    T_max, B, D = max(lengths), len(batch), values[0].shape[1]
    v = torch.zeros(B, T_max, D)
    m = torch.zeros(B, T_max, D)
    dl = torch.zeros(B, T_max, D)
    y = torch.zeros(B, T_max)
    loss_mask = torch.zeros(B, T_max)
    for i, T in enumerate(lengths):
        v[i, :T], m[i, :T], dl[i, :T], y[i, :T], loss_mask[i, :T] = \
            values[i], mask[i], delta[i], labels[i], 1.0
    return v, m, dl, y, loss_mask, torch.tensor(lengths)


# %% [markdown]
# ## Step 3 — GRU-D cell (Che et al. 2018)

# %%
class GRUD(nn.Module):
    def __init__(self, input_size, hidden_size, x_mean):
        super().__init__()
        self.hidden_size = hidden_size
        self.register_buffer("x_mean", torch.tensor(x_mean, dtype=torch.float32))
        self.gamma_x_lin = nn.Linear(input_size, input_size)   # input decay
        self.gamma_h_lin = nn.Linear(input_size, hidden_size)  # hidden-state decay
        self.gru_cell = nn.GRUCell(input_size * 2, hidden_size)  # fed [x_hat ; mask]
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x, mask, delta):
        B, T, D = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)
        last_obs = self.x_mean.unsqueeze(0).expand(B, D).clone()
        logits = torch.zeros(B, T, device=x.device)
        gamma_x_trace = torch.zeros(B, T, D, device=x.device)

        for t in range(T):
            xt, mt, dt = x[:, t, :], mask[:, t, :], delta[:, t, :]
            gamma_x = torch.exp(-torch.relu(self.gamma_x_lin(dt)))
            gamma_h = torch.exp(-torch.relu(self.gamma_h_lin(dt)))

            x_hat = mt * xt + (1 - mt) * (gamma_x * last_obs + (1 - gamma_x) * self.x_mean.unsqueeze(0))
            last_obs = mt * xt + (1 - mt) * last_obs
            h = gamma_h * h
            h = self.gru_cell(torch.cat([x_hat, mt], dim=1), h)

            logits[:, t] = self.head(h).squeeze(-1)
            gamma_x_trace[:, t, :] = gamma_x

        return logits, gamma_x_trace


def masked_bce_loss(logits, y, loss_mask, pos_weight):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none", pos_weight=pos_weight)
    return (bce * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)


def run_epoch(model, loader, optimizer, pos_weight, train):
    model.train() if train else model.eval()
    total_loss, n_batches = 0.0, 0
    all_probs, all_labels = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for v, m, dl, y, loss_mask, _ in loader:
            v, m, dl, y, loss_mask = (t.to(DEVICE) for t in (v, m, dl, y, loss_mask))
            logits, _ = model(v, m, dl)
            loss = masked_bce_loss(logits, y, loss_mask, pos_weight)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            mnp = loss_mask.cpu().numpy().astype(bool)
            all_probs.append(probs[mnp])
            all_labels.append(y.cpu().numpy()[mnp])
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    return (total_loss / n_batches, roc_auc_score(labels, probs),
            average_precision_score(labels, probs), probs, labels)


# %% [markdown]
# ## Step 4 — train, evaluate, compare to XGBoost via DeLong's test

# %%
def run_grud():
    t0 = time.time()
    con = duckdb.connect(str(DB_PATH), read_only=True)
    patient_ids = load_patient_subsample(con)
    raw_df = load_raw_sequences(con, patient_ids)
    con.close()

    train_ids, temp_ids = train_test_split(patient_ids, test_size=0.30, random_state=RANDOM_STATE)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=RANDOM_STATE)
    log(f"Split: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} patients")

    x_mean = raw_df.loc[raw_df["patient_id"].isin(train_ids), RAW_VITALS].mean().fillna(0).to_numpy(dtype=np.float32)

    train_seqs = build_dataset(raw_df[raw_df["patient_id"].isin(train_ids)], RAW_VITALS)
    val_seqs = build_dataset(raw_df[raw_df["patient_id"].isin(val_ids)], RAW_VITALS)
    test_seqs = build_dataset(raw_df[raw_df["patient_id"].isin(test_ids)], RAW_VITALS)
    del raw_df
    gc.collect()

    train_loader = DataLoader(SepsisSeqDataset(train_seqs), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(SepsisSeqDataset(val_seqs), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(SepsisSeqDataset(test_seqs), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    n_pos = sum(s["labels"].sum() for s in train_seqs)
    n_neg = sum(len(s["labels"]) - s["labels"].sum() for s in train_seqs)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=DEVICE)
    log(f"Train patient-hours: pos={n_pos:.0f} neg={n_neg:.0f} pos_weight={pos_weight.item():.1f}")

    model = GRUD(input_size=len(RAW_VITALS), hidden_size=HIDDEN_SIZE, x_mean=x_mean).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_auprc, best_state, patience_left, epoch = -1.0, None, PATIENCE, 0
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_auroc, tr_auprc, _, _ = run_epoch(model, train_loader, optimizer, pos_weight, train=True)
        val_loss, val_auroc, val_auprc, _, _ = run_epoch(model, val_loader, optimizer, pos_weight, train=False)
        log(f"  epoch {epoch:02d}: train_loss={tr_loss:.4f} train_auroc={tr_auroc:.4f} "
            f"| val_loss={val_loss:.4f} val_auroc={val_auroc:.4f} val_auprc={val_auprc:.4f}")
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                log(f"  early stopping at epoch {epoch} (best val AUPRC={best_val_auprc:.4f})")
                break

    model.load_state_dict(best_state)

    # --- val-set predictions, from the SAME best_state used for test -----
    # Saved to disk (unlike before) for two downstream uses that must NOT
    # touch the test set: (a) post-hoc Platt-scaling calibration (script 19),
    # (b) picking the alarm operating threshold below. Neither the
    # calibrator nor the threshold may be fit/chosen on test data.
    _, val_auroc_final, val_auprc_final, val_probs, val_labels = run_epoch(
        model, val_loader, optimizer, pos_weight, train=False
    )
    val_pids = np.concatenate([[s["patient_id"]] * len(s["labels"]) for s in val_seqs])
    val_hours = np.concatenate([s["hour"] for s in val_seqs])
    val_eval_df = pd.DataFrame({
        "patient_id": val_pids, "hour": val_hours,
        "SepsisLabel": val_labels.astype(int), "grud_proba": val_probs,
    })
    log(f"[VAL, best checkpoint] GRU-D  AUROC={val_auroc_final:.4f}  AUPRC={val_auprc_final:.4f}")

    _, test_auroc, test_auprc, test_probs, test_labels = run_epoch(model, test_loader, optimizer, pos_weight, train=False)
    log(f"\n[TEST] GRU-D  AUROC={test_auroc:.4f}  AUPRC={test_auprc:.4f}")

    # --- assemble eval frame (needs patient_id, hour, SepsisLabel, *_proba
    #     to match utility_score.py's expected schema exactly) ---
    test_pids = np.concatenate([[s["patient_id"]] * len(s["labels"]) for s in test_seqs])
    test_hours = np.concatenate([s["hour"] for s in test_seqs])
    eval_df = pd.DataFrame({
        "patient_id": test_pids, "hour": test_hours,
        "SepsisLabel": test_labels.astype(int), "grud_proba": test_probs,
    })

    # --- threshold LOCKED on val, then applied once to test (not swept on
    #     test) -- see script 19's methodology note on why the previous
    #     version (sweeping on eval_df/test directly) was a leak: it let the
    #     test set influence the very decision rule then evaluated on it. ---
    thr_table = sweep_thresholds_for_utility(
        val_eval_df, patient_col="patient_id", label_col="SepsisLabel", proba_col="grud_proba"
    )
    best_row = thr_table.iloc[0]
    locked_threshold = float(best_row.threshold)
    test_utility_at_locked_threshold = normalized_utility_score(
        eval_df, patient_col="patient_id", label_col="SepsisLabel",
        proba_col="grud_proba", threshold=locked_threshold,
    )
    log(f"[VAL] GRU-D best-threshold normalized utility = {best_row.normalized_utility:.4f} "
        f"at threshold={locked_threshold:.2f} (selected on val, NOT test)")
    log(f"[TEST] GRU-D normalized utility at the VAL-locked threshold={locked_threshold:.2f}: "
        f"{test_utility_at_locked_threshold:.4f}")

    # --- interpretability: did the network learn what script 02/05 found by hand? ---
    log("\nMean learned input-decay rate gamma_x per vital (lower = network decays "
        "that vital's stale readings faster -- treats freshness as more urgent). "
        "Compare against script 05's SHAP ranking of *_hours_since_last "
        "(Lactate #3, Bilirubin_total #6 model-wide).")
    model.eval()
    all_gamma = []
    with torch.no_grad():
        for v, m, dl, y, loss_mask, _ in test_loader:
            v, m, dl = v.to(DEVICE), m.to(DEVICE), dl.to(DEVICE)
            _, gamma_trace = model(v, m, dl)
            mnp = loss_mask.numpy().astype(bool)
            g = gamma_trace.cpu().numpy()
            for i in range(g.shape[0]):
                all_gamma.append(g[i][mnp[i]])
    mean_gamma = np.concatenate(all_gamma, axis=0).mean(axis=0)
    gamma_rank = pd.DataFrame({"vital": RAW_VITALS, "mean_gamma_x": mean_gamma}).sort_values("mean_gamma_x")
    log(gamma_rank.to_string(index=False))
    gamma_rank.to_csv(OUT_DIR / "grud_decay_rates.csv", index=False)
    for vital, shap_rank in [("Lactate", 3), ("Bilirubin_total", 6)]:
        pos = int(gamma_rank.reset_index(drop=True).query("vital == @vital").index[0]) + 1
        log(f"  {vital}: rank #{pos}/{len(RAW_VITALS)} by fastest learned decay "
            f"(SHAP had its hours_since_last feature at #{shap_rank} model-wide)")

    # --- DeLong's test vs XGBoost engineered model (script 04's OOF predictions) ---
    results_row = {
        "model": "grud", "auroc": test_auroc, "auprc": test_auprc,
        "best_threshold": locked_threshold,
        "normalized_utility": test_utility_at_locked_threshold,
        "val_normalized_utility_at_threshold": best_row.normalized_utility,
        "threshold_selection": "locked on val, applied once to test",
        "n_features": len(RAW_VITALS),
        "n_subsample_patients": N_SUBSAMPLE_PATIENTS if N_SUBSAMPLE_PATIENTS is not None else len(patient_ids),
        "n_train_patients": len(train_ids), "n_val_patients": len(val_ids),
        "n_test_patients": len(test_ids), "epochs_run": epoch,
    }
    engineered_path = find_prior_output("engineered_oof_predictions.parquet")
    if engineered_path is not None:
        xgb_df = pd.read_parquet(engineered_path)
        merged = eval_df.merge(
            xgb_df[["patient_id", "hour", "engineered_proba"]], on=["patient_id", "hour"], how="inner"
        )
        test_result = delong_roc_test(
            merged["SepsisLabel"].to_numpy(),
            merged["engineered_proba"].to_numpy(),
            merged["grud_proba"].to_numpy(),
        )
        log(f"\nDeLong's test (XGBoost engineered vs GRU-D, {len(merged):,} paired predictions, "
            f"{merged['patient_id'].nunique():,} patients): "
            f"AUC {test_result['auc_a']:.4f} -> {test_result['auc_b']:.4f}, "
            f"z={test_result['z']:.2f}, p={test_result['p_value']:.2e}")
        ci_lo, ci_hi, ci_line = delong_ci_line(test_result, "vs XGBoost")
        log(ci_line)
        results_row.update(test_result)
        results_row["ci_lower"] = ci_lo
        results_row["ci_upper"] = ci_hi
    else:
        log(f"\n(No engineered_oof_predictions.parquet found in /kaggle/working/outputs "
            f"or the attached input dataset -- run 04_engineered_model_kaggle.py first "
            f"to get the DeLong comparison against XGBoost.)")

    pd.DataFrame([results_row]).to_csv(OUT_DIR / "grud_results.csv", index=False)
    eval_df.rename(columns={"SepsisLabel": "SepsisLabel"}).to_parquet(OUT_DIR / "grud_test_predictions.parquet")
    val_eval_df.to_parquet(OUT_DIR / "grud_val_predictions.parquet")
    with open(OUT_DIR / "run16_log.txt", "w") as f:
        f.write("\n".join(log_lines))
    log(f"\nSaved: grud_results.csv, grud_test_predictions.parquet, grud_val_predictions.parquet, "
        f"grud_decay_rates.csv, run16_log.txt ({time.time() - t0:.1f}s total)")
    return results_row


# %%
if __name__ == "__main__":
    run_grud()
