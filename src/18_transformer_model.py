# %% [markdown]
# # Phase 6c — Transformer: does attention over the *whole stay* beat recurrence?
#
# Script 16 asked whether a recurrent architecture (GRU-D) could learn the
# missingness signal script 02 hand-engineered, one hour at a time. This
# script asks a different question about the same raw hourly data: does
# self-attention over the *entire* ICU stay so far -- rather than a hidden
# state carried hour-to-hour -- find sepsis-relevant patterns a recurrence
# can't? A causally-masked Transformer encoder (Vaswani et al., 2017) can
# attend directly from hour t back to any earlier hour in one layer, instead
# of having to route that information through T sequential GRU updates.
#
# This is explicitly a stretch goal run *after* GRU-D, not instead of it, for
# two reasons the supervisor flagged directly:
#   1. It's more hyperparameter-sensitive than a GRU, and more prone to
#      overfitting at this dataset's ~1.8% stay-level sepsis rate with
#      laptop-scale compute -- so it needs a working recurrent baseline
#      already in hand to sanity-check against.
#   2. Full self-attention is O(T^2) per layer (T=336 max ICULOS), so unlike
#      GRU-D's step-by-step BPTT constraint, the Transformer's constraint is
#      attention memory over long sequences -- the same laptop-tractability
#      problem, different cause. Same subsample precedent as scripts 09/16
#      applies below.
#
# Input construction deliberately differs from both earlier deep models, to
# keep the three architectures' relationship to the missingness signal
# distinct and comparable:
#   - GRU-D (script 16): computes its own learned decay gamma_x from delta,
#     internal to the architecture.
#   - TabNet (script 17): is handed script 02's precomputed
#     `_hours_since_last` columns directly, as ordinary input features.
#   - This script: feeds the *raw* delta tensor (hours-since-last-observed,
#     the same one script 16 computes) plus a binary mask, letting
#     self-attention decide what to do with staleness -- no decay function
#     assumed, no hand-computed missingness feature engineering. Whatever
#     the model does with staleness here, it had to work out from delta and
#     mask alone.
#
# Same evaluation contract as every other model in this repo: same
# `utility_score.py` normalized-utility sweep, same `delong.py` helper,
# same patient-grouped train/val/test split discipline (mirroring script 16
# rather than 03/04's GroupKFold, since this is also a BPTT-adjacent,
# early-stopping-based recurrent-data model, not a k-fold-CV tree model).
#
# EVALUATION PROTOCOL DISCLOSURE (same spirit as scripts 09/16):
# Full GroupKFold(5) refit-per-fold over all patients, at MAX_SEQ_LEN=336,
# is not laptop-tractable for a multi-head self-attention model (O(T^2)
# attention weights per head per layer) on the machine that OOM'd on script
# 04 and needed subsampling for script 16. This script reuses script 16's
# exact stratified-subsample-by-`is_ever_septic` helper and the same
# N_SUBSAMPLE_PATIENTS, so the two deep sequence models are trained and
# evaluated on the *same* patients and the *same* split -- making the
# GRU-D-vs-Transformer comparison itself apples-to-apples, on top of both
# being comparable to XGBoost.

# %%
import gc
import sys
import time
import math
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
        "PyTorch is required for this script and isn't in requirements.txt yet "
        "(the rest of the pipeline is XGBoost/sklearn/duckdb only).\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        "CPU is enough here -- small model, subsampled patients."
    )

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utility_score import normalized_utility_score, sweep_thresholds_for_utility
from delong import delong_roc_test, delong_ci_line

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "warehouse" / "sepsis.duckdb"
OUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# ---- config ---------------------------------------------------------------
N_SUBSAMPLE_PATIENTS = 20000     # bumped from 8,000, still identical to script 16's value so the
                                  # two sequence models share patients -- targeted power increase
                                  # for the vs-XGBoost comparison (p=0.123 at n=8,000); single
                                  # 70/15/15 split kept as-is, no CV rewrite
MAX_SEQ_LEN = 336               # PhysioNet 2019 max ICULOS, same as script 16
RANDOM_STATE = 42               # matches every other script's RANDOM_STATE/seed
BATCH_SIZE = 64
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
FF_DIM = 128
DROPOUT = 0.2                   # higher than GRU-D's implicit regularization -- attention overfits faster here
EPOCHS = 20
PATIENCE = 4                    # early stopping on val AUPRC, same discipline as script 16
LR = 1e-3
WEIGHT_DECAY = 1e-4              # stronger weight decay than script 16, same overfitting concern
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Exact raw column layout from 01_etl_warehouse.py's VITAL_COLS + LAB_COLS,
# identical list/order to script 16 so results are directly comparable.
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
# ## Step 1 — same stratified patient subsample as script 16 (shared patients, shared split)

# %%
def stratified_subsample(df, n, seed, strat_col="is_ever_septic"):
    """Identical to script 16's (and 09's) stratified_subsample: manual
    per-group .sample() rather than groupby().apply(), to sidestep pandas'
    include_groups behavior, and to pin the sepsis rate in the subsample
    rather than leave it to chance."""
    frac = min(1.0, n / len(df))
    parts = []
    for _, group in df.groupby(strat_col):
        parts.append(group.sample(frac=frac, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def load_patient_subsample(con):
    patients = con.execute(
        "SELECT patient_id, is_ever_septic FROM dim_patient"
    ).df()
    sub = stratified_subsample(patients, N_SUBSAMPLE_PATIENTS, RANDOM_STATE)
    log(
        f"\nEVALUATION PROTOCOL DISCLOSURE:\n"
        f"  Full GroupKFold(5) refit-per-fold over all {len(patients):,} patients is not\n"
        f"  tractable for an O(T^2)-attention Transformer on this machine (7.4GB RAM --\n"
        f"  the same machine that OOM'd on script 04 and needed subsampling for script\n"
        f"  16). Training below therefore reuses script 16's exact stratified subsample\n"
        f"  of n={N_SUBSAMPLE_PATIENTS:,} patients (seed={RANDOM_STATE}), same 70/15/15\n"
        f"  train/val/test split by patient, so GRU-D and this Transformer are compared\n"
        f"  on identical patients as well as both being compared to XGBoost."
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
# ## Step 2 — build tensors: mean-imputed value, mask, causal delta, label
#
# Same causal delta construction as script 16 (`hour` is a 0-indexed,
# always-consecutive per-patient integer offset per 01_etl_warehouse.py, so
# a fixed 1-hour step is exact). Unlike GRU-D, there's no learned gamma_x
# here -- missing values are simply mean-imputed (population mean from the
# training split) and the model is given the mask and delta as plain input
# channels, to see what self-attention does with them unassisted.

# %%
def build_patient_arrays(g, vitals, x_mean):
    vals = g[vitals].to_numpy(dtype=np.float32)
    mask = (~np.isnan(vals)).astype(np.float32)
    T, D = vals.shape

    delta = np.zeros((T, D), dtype=np.float32)
    for d in range(D):
        last_seen = 0.0
        for t in range(1, T):
            last_seen = 1.0 if mask[t - 1, d] == 1 else last_seen + 1.0
            delta[t, d] = last_seen
    # normalize delta into roughly [0, 1]-ish scale so it doesn't dominate
    # the linear projection next to standardized values -- MAX_SEQ_LEN is
    # the natural cap since delta can't exceed the sequence length.
    delta_norm = delta / MAX_SEQ_LEN

    vals_imputed = np.where(mask.astype(bool), vals, x_mean[None, :])

    return (
        vals_imputed.astype(np.float32),
        mask,
        delta_norm,
        g["SepsisLabel"].to_numpy(dtype=np.float32),
        g["hour"].to_numpy(),
    )


def build_dataset(df, vitals, x_mean):
    sequences = []
    for pid, g in df.groupby("patient_id", sort=False):
        g = g.sort_values("hour")
        if len(g) > MAX_SEQ_LEN:
            g = g.iloc[:MAX_SEQ_LEN]
        vals, mask, delta, labels, hours = build_patient_arrays(g, vitals, x_mean)
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
    loss_mask = torch.zeros(B, T_max)          # which (b, t) positions are real, not padding
    pad_mask = torch.ones(B, T_max, dtype=torch.bool)  # True = PAD, for the encoder's key_padding_mask
    for i, T in enumerate(lengths):
        v[i, :T], m[i, :T], dl[i, :T], y[i, :T], loss_mask[i, :T] = \
            values[i], mask[i], delta[i], labels[i], 1.0
        pad_mask[i, :T] = False
    return v, m, dl, y, loss_mask, pad_mask, torch.tensor(lengths)


# %% [markdown]
# ## Step 3 — causally-masked Transformer encoder
#
# Standard sinusoidal positional encoding over hour-in-stay, a linear input
# projection of [value ; mask ; delta] (3*D -> D_MODEL), a stack of
# `nn.TransformerEncoderLayer`s with an upper-triangular additive causal
# mask (position t can only attend to positions <= t, so predictions at
# hour t never see the future -- the same causal discipline script 02's SQL
# window functions enforce with `ROWS BETWEEN ... AND CURRENT ROW`), and a
# per-timestep linear head.

# %%
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class CausalSepsisTransformer(nn.Module):
    def __init__(self, input_size, d_model, n_heads, n_layers, ff_dim, dropout, max_len):
        super().__init__()
        self.input_proj = nn.Linear(input_size * 3, d_model)  # [value ; mask ; delta]
        self.pos_enc = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, v, m, dl, pad_mask):
        B, T, D = v.shape
        x = torch.cat([v, m, dl], dim=-1)
        x = self.input_proj(x)
        x = self.pos_enc(x)

        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=v.device), diagonal=1
        )
        h = self.encoder(x, mask=causal_mask, src_key_padding_mask=pad_mask)
        h = self.dropout(h)
        logits = self.head(h).squeeze(-1)
        return logits


# %% [markdown]
# ## Step 4 — train / eval loop (mirrors script 16's, swapped for the pad-mask signature)

# %%
def masked_bce_loss(logits, y, loss_mask, pos_weight):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none", pos_weight=pos_weight)
    return (bce * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)


def run_epoch(model, loader, optimizer, pos_weight, train):
    model.train() if train else model.eval()
    total_loss, n_batches = 0.0, 0
    all_probs, all_labels = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for v, m, dl, y, loss_mask, pad_mask, _ in loader:
            v, m, dl, y, loss_mask, pad_mask = (
                t.to(DEVICE) for t in (v, m, dl, y, loss_mask, pad_mask)
            )
            logits = model(v, m, dl, pad_mask)
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
# ## Step 5 — train, evaluate, compare to XGBoost and to GRU-D via DeLong's test

# %%
def run_transformer():
    t0 = time.time()
    con = duckdb.connect(str(DB_PATH), read_only=True)
    patient_ids = load_patient_subsample(con)
    raw_df = load_raw_sequences(con, patient_ids)
    con.close()

    # identical split call/order to script 16 -> identical train/val/test patient sets
    train_ids, temp_ids = train_test_split(patient_ids, test_size=0.30, random_state=RANDOM_STATE)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=RANDOM_STATE)
    log(f"Split: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} patients")

    x_mean = raw_df.loc[raw_df["patient_id"].isin(train_ids), RAW_VITALS].mean().fillna(0).to_numpy(dtype=np.float32)
    x_std = raw_df.loc[raw_df["patient_id"].isin(train_ids), RAW_VITALS].std().replace(0, 1).fillna(1).to_numpy(dtype=np.float32)

    # standardize in-place before sequence building (mean-impute uses the
    # *raw*-scale mean, so standardize afterwards using the same train stats)
    raw_df = raw_df.copy()
    raw_df[RAW_VITALS] = (raw_df[RAW_VITALS] - x_mean) / x_std
    x_mean_std_scale = np.zeros_like(x_mean)  # post-standardization, impute with 0 (= train mean)

    train_seqs = build_dataset(raw_df[raw_df["patient_id"].isin(train_ids)], RAW_VITALS, x_mean_std_scale)
    val_seqs = build_dataset(raw_df[raw_df["patient_id"].isin(val_ids)], RAW_VITALS, x_mean_std_scale)
    test_seqs = build_dataset(raw_df[raw_df["patient_id"].isin(test_ids)], RAW_VITALS, x_mean_std_scale)
    del raw_df
    gc.collect()

    train_loader = DataLoader(SepsisSeqDataset(train_seqs), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(SepsisSeqDataset(val_seqs), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(SepsisSeqDataset(test_seqs), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    n_pos = sum(s["labels"].sum() for s in train_seqs)
    n_neg = sum(len(s["labels"]) - s["labels"].sum() for s in train_seqs)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=DEVICE)
    log(f"Train patient-hours: pos={n_pos:.0f} neg={n_neg:.0f} pos_weight={pos_weight.item():.1f}")

    model = CausalSepsisTransformer(
        input_size=len(RAW_VITALS), d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, ff_dim=FF_DIM, dropout=DROPOUT, max_len=MAX_SEQ_LEN,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model params: {n_params:,}")
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
    _, test_auroc, test_auprc, test_probs, test_labels = run_epoch(model, test_loader, optimizer, pos_weight, train=False)
    log(f"\n[TEST] Transformer  AUROC={test_auroc:.4f}  AUPRC={test_auprc:.4f}")

    test_pids = np.concatenate([[s["patient_id"]] * len(s["labels"]) for s in test_seqs])
    test_hours = np.concatenate([s["hour"] for s in test_seqs])
    eval_df = pd.DataFrame({
        "patient_id": test_pids, "hour": test_hours,
        "SepsisLabel": test_labels.astype(int), "transformer_proba": test_probs,
    })

    thr_table = sweep_thresholds_for_utility(
        eval_df, patient_col="patient_id", label_col="SepsisLabel", proba_col="transformer_proba"
    )
    best_row = thr_table.iloc[0]
    log(f"[TEST] Transformer best-threshold normalized utility = {best_row.normalized_utility:.4f} "
        f"at threshold={best_row.threshold:.2f}")

    # --- attention-weight summary: which past hours does the model lean on most? ---
    log("\nMean self-attention weight by relative lag (last encoder layer, averaged "
        "over heads/batches/query positions) -- higher = more attended-to distance "
        "back in the stay. Complements GRU-D's per-vital decay-rate table (script 16) "
        "with a per-*lag* view instead of a per-*vital* one.")
    model.eval()
    lag_weight_sum, lag_weight_count = {}, {}
    hook_storage = {}

    def _attn_hook(module, inp, out):
        # MultiheadAttention with need_weights defaults off in fused SDPA path;
        # re-run just the last layer's self-attn explicitly with weights on.
        pass

    with torch.no_grad():
        last_layer = model.encoder.layers[-1].self_attn
        for v, m, dl, y, loss_mask, pad_mask, lengths in test_loader:
            v, m, dl, pad_mask = v.to(DEVICE), m.to(DEVICE), dl.to(DEVICE), pad_mask.to(DEVICE)
            B, T, D = v.shape
            x = torch.cat([v, m, dl], dim=-1)
            x = model.input_proj(x)
            x = model.pos_enc(x)
            h = x
            for layer in model.encoder.layers[:-1]:
                causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=v.device), diagonal=1)
                h = layer(h, src_mask=causal_mask, src_key_padding_mask=pad_mask)
            causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=v.device), diagonal=1)
            _, attn_weights = last_layer(
                h, h, h, attn_mask=causal_mask, key_padding_mask=pad_mask,
                need_weights=True, average_attn_weights=True,
            )
            # attn_weights: (B, T_query, T_key); lag = query_pos - key_pos, only valid (non-pad) queries
            attn_np = attn_weights.cpu().numpy()
            loss_mask_np = loss_mask.numpy().astype(bool)
            for b in range(B):
                Tb = int(lengths[b])
                for tq in range(Tb):
                    if not loss_mask_np[b, tq]:
                        continue
                    for tk in range(tq + 1):
                        lag = tq - tk
                        lag_weight_sum[lag] = lag_weight_sum.get(lag, 0.0) + attn_np[b, tq, tk]
                        lag_weight_count[lag] = lag_weight_count.get(lag, 0) + 1

    lags = sorted(lag_weight_sum.keys())
    lag_df = pd.DataFrame({
        "lag_hours": lags,
        "mean_attention_weight": [lag_weight_sum[l] / lag_weight_count[l] for l in lags],
        "n_query_positions": [lag_weight_count[l] for l in lags],
    })
    lag_df.to_csv(OUT_DIR / "transformer_attention_by_lag.csv", index=False)
    log(lag_df.head(12).to_string(index=False))
    log("...")
    # recency bias check: is lag=0 (attending to "now") the single highest-weighted lag?
    top_lag = lag_df.sort_values("mean_attention_weight", ascending=False).iloc[0]["lag_hours"]
    log(f"Highest-attended lag: {int(top_lag)}h back "
        f"({'recency-biased' if top_lag <= 2 else 'attends further back than immediate recency'})")

    # --- DeLong's test vs XGBoost engineered model (script 04's OOF predictions) ---
    results_row = {
        "model": "transformer", "auroc": test_auroc, "auprc": test_auprc,
        "best_threshold": best_row.threshold, "normalized_utility": best_row.normalized_utility,
        "n_features": len(RAW_VITALS), "n_params": n_params,
        "n_subsample_patients": N_SUBSAMPLE_PATIENTS,
        "n_train_patients": len(train_ids), "n_val_patients": len(val_ids),
        "n_test_patients": len(test_ids), "epochs_run": epoch,
    }
    engineered_path = OUT_DIR / "engineered_oof_predictions.parquet"
    if engineered_path.exists():
        xgb_df = pd.read_parquet(engineered_path)
        merged = eval_df.merge(
            xgb_df[["patient_id", "hour", "engineered_proba"]], on=["patient_id", "hour"], how="inner"
        )
        test_result = delong_roc_test(
            merged["SepsisLabel"].to_numpy(),
            merged["engineered_proba"].to_numpy(),
            merged["transformer_proba"].to_numpy(),
        )
        log(f"\nDeLong's test (XGBoost engineered vs Transformer, {len(merged):,} paired "
            f"predictions, {merged['patient_id'].nunique():,} patients): "
            f"AUC {test_result['auc_a']:.4f} -> {test_result['auc_b']:.4f}, "
            f"z={test_result['z']:.2f}, p={test_result['p_value']:.2e}")
        ci_lo, ci_hi, ci_line = delong_ci_line(test_result, "vs XGBoost")
        log(ci_line)
        results_row.update({f"vs_xgb_{k}": v for k, v in test_result.items()})
        results_row["vs_xgb_ci_lower"] = ci_lo
        results_row["vs_xgb_ci_upper"] = ci_hi
    else:
        log(f"\n(No {engineered_path.name} found -- run 04_engineered_model.py first "
            f"to get the DeLong comparison against XGBoost.)")

    # --- DeLong's test vs GRU-D (script 16's test predictions) -- same patients, same split ---
    grud_path = OUT_DIR / "grud_test_predictions.parquet"
    if grud_path.exists():
        grud_df = pd.read_parquet(grud_path)
        merged_g = eval_df.merge(
            grud_df[["patient_id", "hour", "grud_proba"]], on=["patient_id", "hour"], how="inner"
        )
        if len(merged_g) > 0:
            test_result_g = delong_roc_test(
                merged_g["SepsisLabel"].to_numpy(),
                merged_g["grud_proba"].to_numpy(),
                merged_g["transformer_proba"].to_numpy(),
            )
            log(f"\nDeLong's test (GRU-D vs Transformer, {len(merged_g):,} paired "
                f"predictions, {merged_g['patient_id'].nunique():,} patients -- same "
                f"subsample/split, so this comparison is exact, not approximate): "
                f"AUC {test_result_g['auc_a']:.4f} -> {test_result_g['auc_b']:.4f}, "
                f"z={test_result_g['z']:.2f}, p={test_result_g['p_value']:.2e}")
            ci_lo_g, ci_hi_g, ci_line_g = delong_ci_line(test_result_g, "vs GRU-D")
            log(ci_line_g)
            results_row.update({f"vs_grud_{k}": v for k, v in test_result_g.items()})
            results_row["vs_grud_ci_lower"] = ci_lo_g
            results_row["vs_grud_ci_upper"] = ci_hi_g
    else:
        log(f"\n(No {grud_path.name} found -- run 16_grud_model.py first "
            f"to get the DeLong comparison against GRU-D.)")

    pd.DataFrame([results_row]).to_csv(OUT_DIR / "transformer_results.csv", index=False)
    eval_df.to_parquet(OUT_DIR / "transformer_test_predictions.parquet")
    with open(OUT_DIR / "run18_log.txt", "w") as f:
        f.write("\n".join(log_lines))
    log(f"\nSaved: transformer_results.csv, transformer_test_predictions.parquet, "
        f"transformer_attention_by_lag.csv, run18_log.txt ({time.time() - t0:.1f}s total)")
    return results_row


# %%
if __name__ == "__main__":
    run_transformer()
