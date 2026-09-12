# Early Warning System for Sepsis Onset Using Dynamic Physiological Telemetry

An end-to-end data warehousing + machine learning pipeline that predicts **sepsis onset 6 hours ahead** from hourly ICU vitals and labs, built on the real **PhysioNet/Computing in Cardiology Challenge 2019** dataset. The project takes ~1.55 million patient-hours of raw ICU telemetry, turns them into a queryable data warehouse, engineers a 289-feature predictive representation of each patient-hour, trains and rigorously validates a gradient-boosted model against that target, and then stress-tests the result from every angle a clinical deployment would actually need answered: does it generalize across hospitals, is it fair across patient subgroups, does it know when it's uncertain, and what does it actually buy a nurse at the bedside compared to the rule-of-thumb screening already in use. A later Phase 6 extension (§4) then asks a different question of the same problem: does a deep sequence model — GRU-D, TabNet, or a causal Transformer — actually beat the gradient-boosted tree, or does this dataset reproduce the broader finding that trees are hard to beat on tabular clinical data at this scale?

> **On the title:** this is named *"...Sepsis Onset..."*, not the broader *"...Patient Deterioration..."*, because that is precisely what `SepsisLabel` — the target variable — measures. Calling it "deterioration" would overstate what the model is trained on.

---

## Table of contents

1. [Dataset](#1-dataset)
2. [Pipeline architecture](#2-pipeline-architecture)
3. [Repository structure](#3-repository-structure)
4. [Script-by-script walkthrough, with outputs](#4-script-by-script-walkthrough-with-outputs)
5. [Leakage handling](#5-leakage-handling)
6. [Results summary](#6-results-summary)
7. [Limitations and honest caveats](#7-limitations-and-honest-caveats)
8. [Reproducing this project](#8-reproducing-this-project)

---

## 1. Dataset

The [PhysioNet/Computing in Cardiology Challenge 2019](https://physionet.org/content/challenge-2019/) dataset — real, de-identified ICU telemetry from two hospital systems.

| | |
|---|---|
| Patients | 40,336 |
| Patient-hours (rows in the fact table) | 1,552,210 |
| Positive (septic) hour rate | 1.8% |
| Raw signals per hour | 8 vitals (`HR`, `O2Sat`, `Temp`, `SBP`, `MAP`, `DBP`, `Resp`, `EtCO2`) + 26 labs (`BaseExcess`, `HCO3`, `FiO2`, `pH`, `PaCO2`, `SaO2`, `AST`, `BUN`, `Alkalinephos`, `Calcium`, `Chloride`, `Creatinine`, `Bilirubin_direct`, `Glucose`, `Lactate`, `Magnesium`, `Phosphate`, `Potassium`, `Bilirubin_total`, `TroponinI`, `Hct`, `Hgb`, `PTT`, `WBC`, `Fibrinogen`, `Platelets`) + demographics (`Age`, `Gender`, `HospAdmTime`, `ICULOS`) |
| Target | `SepsisLabel` — derived from Sepsis-3 criteria by the Challenge organizers, defined to flip to 1 six hours before clinically documented onset, so a positive at hour *t* already means "sepsis within the next 6 hours" |
| Hospitals | 2 (`hospital_system_1`: 20,336 patients, `hospital_system_2`: 20,000 patients) |

This is a genuinely hard, severely imbalanced clinical prediction problem. A 1.8% positive rate means a model that always predicts "no sepsis" is already 98.2% "accurate" — which is exactly why this project scores everything on **AUROC**, **AUPRC**, and a **clinical utility score** (§4, `utility_score.py`) instead of accuracy throughout.

Most cells in the raw lab columns are `NULL` in any given hour, because labs are drawn on clinical judgment, not on a fixed hourly schedule — vitals like `HR`/`Temp`/`Resp` are much more densely sampled than labs like `Lactate` or `Bilirubin_total`. This sparsity is not cleaned away; it is treated as a feature in its own right (§4, `02_feature_engineering.py`).

---

## 2. Pipeline architecture

```
raw PhysioNet .psv files (one per patient)
        │
        ▼
01_etl_warehouse.py            →  DuckDB star schema (dim_patient, dim_hospital, fact_vitals_hourly)
        │
        ▼
02_feature_engineering.py      →  fact_features: 294-column engineered table (SQL window functions)
        │
        ├──► 03_baseline_model.py        →  XGBoost, 15 raw features            (control)
        │
        ├──► 04_engineered_model.py      →  XGBoost, 289 engineered features     (main model)
        │           │
        │           ├──► 05_explainability.py           SHAP attribution
        │           ├──► 06_leadtime_alarm_fatigue.py    lead time / false-alarm framing
        │           ├──► 13_fairness_audit.py            subgroup performance audit
        │           ├──► 14_cross_hospital_generalization.py   train-one/test-other hospital
        │           └──► 15_conformal_prediction.py      calibrated uncertainty sets
        │
        ├──► 07_olap_and_export.py       →  OLAP cube demo + Power BI CSV export
        ├──► 08_association_rules.py     →  Apriori rule mining on abnormal-vital flags
        ├──► clustering_phenotypes.py    →  k-means phenotype search
        ├──► 09_hierarchical_clustering.py →  Ward-linkage phenotype search
        ├──► 11_dbscan_clustering.py     →  density-based phenotype search
        ├──► 10_classical_classifiers.py →  Decision Tree + Naive Bayes, raw vs. engineered
        ├──► 12_outlier_analysis.py      →  IQR fencing + Isolation Forest vs. SepsisLabel
        │
        │    ── Phase 6a: does attention beat trees on this same feature table? ──
        └──► 17_tabnet_model_kaggle.py   →  TabNet, same 289 features as script 04

01_etl_warehouse.py's fact_vitals_hourly (raw, pre-script-02)
        │
        │    ── Phase 6b/6c: can a network learn the missingness signal itself? ──
        ├──► 16_grud_model.py            →  GRU-D, learned per-vital missingness decay
        └──► 18_transformer_model.py     →  causal self-attention over the raw stay

delong.py         (shared) statistical test for comparing two correlated AUROCs
utility_score.py  (shared) PhysioNet Challenge's own clinical utility metric
```

Scripts 16–18 are a later addition ("Phase 6" in the scripts' own header comments) that swap in three deep-learning architectures as direct challengers to the XGBoost engineered model, reusing the same `utility_score.py`/`delong.py` evaluation contract throughout. `17_tabnet_model_kaggle.py` is a drop-in *model* swap — it reads the exact same 289-column `fact_features` table script 04 uses. `16_grud_model.py` and `18_transformer_model.py` instead read straight from `fact_vitals_hourly` (raw, pre-script-02) on purpose, since the whole point of both is to test whether a network can learn the missingness/staleness signal script 02 hand-computed in SQL, rather than being handed it as a feature.

Everything downstream of `01_etl_warehouse.py` reads from the DuckDB warehouse, never the raw `.psv` files directly — this keeps every later step fast and lets DuckDB's vectorized SQL engine (window functions, joins, aggregations) do the heavy lifting instead of row-by-row pandas code.

---

## 3. Repository structure

```
sepsis_capstone/
├── src/
│   ├── 01_etl_warehouse.py                    raw .psv -> DuckDB star schema
│   ├── 02_feature_engineering.py              star schema -> 294-column feature table
│   ├── 03_baseline_model.py                   XGBoost control model (15 raw features)
│   ├── 04_engineered_model.py                 XGBoost main model + DeLong test + ablation
│   ├── 05_explainability.py                   SHAP analysis
│   ├── 06_leadtime_alarm_fatigue.py           lead time + false-alarm-rate analysis
│   ├── 07_olap_and_export.py                  OLAP demo (roll-up/drill-down/slice/dice) + Power BI export
│   ├── 08_association_rules.py                Apriori association rule mining
│   ├── 09_hierarchical_clustering.py          Ward-linkage hierarchical clustering
│   ├── 10_classical_classifiers.py            Decision Tree + Naive Bayes (raw vs. engineered)
│   ├── 11_dbscan_clustering.py                DBSCAN density-based clustering
│   ├── 12_outlier_analysis.py                 IQR + Isolation Forest outlier detection
│   ├── 13_fairness_audit.py                   subgroup AUROC/AUPRC/utility audit
│   ├── 14_cross_hospital_generalization.py    train-on-one/test-on-other hospital
│   ├── 15_conformal_prediction.py             MAPIE split-conformal uncertainty sets
│   ├── 16_grud_model.py                       GRU-D recurrent model (learned missingness decay)
│   ├── 17_tabnet_model_kaggle.py              TabNet, local dev run (35% patient subsample)
│   ├── 17_tabnet_model_kaggle (full).py       TabNet, full 1.55M-row Kaggle GPU run
│   ├── 18_transformer_model.py                causal self-attention Transformer over raw vitals
│   ├── clustering_phenotypes.py               k-means clustering
│   ├── delong.py                              DeLong's test (AUROC comparison)
│   └── utility_score.py                       PhysioNet Challenge 2019 utility metric
├── outputs/
│   ├── *.csv               metrics, cluster profiles, rule tables, OLAP demo tables, fairness/
│   │                        cross-hospital/conformal/GRU-D/TabNet/Transformer summaries
│   ├── *.parquet           out-of-fold / held-out test predictions for every model, for
│   │                        independent metric verification without re-training
│   ├── run*_log.txt        captured stdout from each script run (run16/17/18 for the
│   │                        deep-learning scripts; `run17_log(full).txt` for the Kaggle run)
│   ├── figures/            SHAP plots, dendrogram, silhouette plot, DBSCAN elbow,
│   │                        outlier-lift chart, fairness-audit chart
│   │                        (scripts 16–18 report their results as tables/logs only — see
│   │                        §7 for why no new figures were added for these three)
│   └── powerbi_export/     dim_patient.csv, dim_hospital.csv, fact_vitals_olap.csv
├── warehouse/
│   └── sepsis.duckdb       (gitignored — rebuilt locally by 01_etl_warehouse.py)
├── requirements.txt
├── POWERBI_HANDOFF.md      standalone instructions for the Power BI half of the OLAP demo
└── README.md
```

---

## 4. Script-by-script walkthrough, with outputs

### `01_etl_warehouse.py` — raw files → star schema

Parses the raw PhysioNet `.psv` (pipe-separated) files — one file per patient — and loads them into a DuckDB warehouse using a **star schema**:

- **`dim_patient`** — one row per patient: age, gender, hospital, hospital-admission offset, max ICU length-of-stay, an "ever septic" flag, and hours-recorded count.
- **`dim_hospital`** — one row per hospital system.
- **`fact_vitals_hourly`** — the fact table, grain = one row per **patient-hour**: the 8 vitals + 26 labs, `ICULOS` (ICU length of stay in hours), and `SepsisLabel`.

Each subfolder under the raw-data directory is treated as one "hospital system," which is how `hospital_id` gets assigned. A composite index on `(patient_id, hour)` is built on the fact table specifically to make the window-function pass in the next script fast.

This isn't dimensional modeling for its own sake — vitals genuinely are a fact table (one measurement event per patient per hour), with patient and hospital as the natural dimensions to slice and roll up by, which is exactly the structure `07_olap_and_export.py` and the Power BI export later depend on.

### `02_feature_engineering.py` — SQL window-function features

Runs SQL window-function queries directly inside DuckDB (deliberately not pandas, for both speed and correctness) to turn the raw columns into **294 engineered columns**, in four families:

| Family | Count | Logic |
|---|---|---|
| `raw_ffill` | 15 | Forward-filled snapshot of the most recent reading per vital/lab (labs aren't drawn every hour, so most raw cells are `NULL`) |
| `rolling_stats` | 180 | Causal rolling **mean / std / min / max / slope** over **3h / 6h / 12h** windows, per vital, computed on the forward-filled series |
| `slopes_velocity` | 60 | First-difference "velocity" features — is a vital currently moving, and in which direction, not just where it sits |
| `missingness` | 30 | `X_missing` (is this vital `NULL` right now) and `hours_since_last_X` (how long since it was last measured) |
| `clinical_ratios` | 4 | Shock index (`HR/SBP`), pulse pressure (`SBP−DBP`), a partial SIRS score, a partial qSOFA score |

**Why the windows are causal:** every rolling/window feature only looks **backward** — `ROWS BETWEEN N PRECEDING AND CURRENT ROW`, never `FOLLOWING`. A feature computed with `AVG(HR) OVER (... ROWS BETWEEN 11 PRECEDING AND CURRENT ROW)` at hour *t* only ever sees hours ≤ *t*. This is the single most load-bearing anti-leakage decision in the pipeline (§5) — a model that could see future vitals would trivially "predict" sepsis it had already been shown.

**Why missingness is treated as signal, not noise:** clinicians order labs like lactate *more* frequently when they're worried about a patient, so `hours_since_last_Lactate` genuinely encodes clinical suspicion, not just data sparsity. This is confirmed empirically later — `Lactate_hours_since_last` and `Bilirubin_total_hours_since_last` rank 3rd and 6th by SHAP importance model-wide (§4, `05_explainability.py`).

### `03_baseline_model.py` — the control model

Trains XGBoost on **only the 15 raw forward-filled features** — no rolling stats, no engineered ratios — using `GroupKFold(5)` split **by `patient_id`**, not by row (splitting by row would let different hours from the same patient leak across train and test). This exists purely as a control: it establishes what performance the *raw signal alone* buys, so the lift from feature engineering in the next script can be measured against something concrete rather than a vague intuition.

| Metric | Value |
|---|---|
| AUROC | **0.7572** |
| AUPRC | **0.0634** |
| Normalized utility | **0.2504** |
| Best threshold | 0.541 |
| Features | 15 |

### `04_engineered_model.py` — the main model, with statistical proof it's actually better

Trains XGBoost on the full engineered set (289 of the 294 columns from script 02 are used as inputs; the rest are identifiers/labels), same patient-grouped `GroupKFold(5)` split, then runs two further analyses on top of the main fit:

1. **DeLong's test** (`delong.py`) — the statistically correct way to check whether two AUROCs computed on the *same* patients are actually different, versus just numerically different by sampling noise.
2. **Feature-family ablation** — retrain using only one feature family at a time (plus demographics), to see which family is actually carrying the predictive lift.

| Metric | Baseline (15 features) | Engineered (289 features) |
|---|---|---|
| AUROC | 0.7572 | **0.7918** |
| AUPRC | 0.0634 | **0.0821** |
| Normalized utility | 0.2504 | **0.3203** |

**DeLong's test:** z = **−38.30**, p ≈ **0.0** — on 1.55M paired hourly predictions, this is about as statistically decisive a result as this test produces. The AUROC gain is not noise.

**Ablation — which feature family actually carries the lift:**

| Feature family | # features | AUROC | AUPRC |
|---|---|---|---|
| **`rolling_stats`** | 180 | **0.7737** | 0.0695 |
| `raw_ffill` (= baseline) | 15 | 0.7572 | 0.0634 |
| `slopes_velocity` | 60 | 0.7228 | 0.0557 |
| `missingness` | 30 | 0.7163 | 0.0618 |
| `clinical_ratios` | 4 | 0.6561 | 0.0379 |

`rolling_stats` *alone* reaches AUROC 0.7737 — over 80% of the total lift from baseline (0.7572) to full model (0.7918) — while the hand-crafted `clinical_ratios` (shock index, SIRS/qSOFA) actually score *below* the raw baseline in isolation. Broad, mechanical rolling-window statistics beat narrow, textbook clinical-scoring features here: the model doesn't need hand-built domain shortcuts if it has enough raw trend information to reconstruct the equivalent signal itself.

### `05_explainability.py` — SHAP analysis

Computes SHAP (SHapley Additive exPlanations) values for the engineered model — attributing each individual prediction to specific input features via a game-theoretic decomposition, rather than trusting XGBoost's built-in (coarser, less reliable) importance scores.

**Top 15 features by mean |SHAP|:**

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | `Lactate_max_12h` | 0.2523 |
| 2 | `partial_sirs_score` | 0.1255 |
| 3 | `Lactate_hours_since_last` | 0.1029 |
| 4 | `Bilirubin_total_ffill` | 0.0981 |
| 5 | `Temp_max_6h` | 0.0923 |
| 6 | `Bilirubin_total_hours_since_last` | 0.0887 |
| 7 | `Lactate_ffill` | 0.0804 |
| 8 | `Temp_ffill` | 0.0712 |
| 9 | `Resp_mean_12h` | 0.0710 |
| 10 | `Temp_max_12h` | 0.0660 |
| 11 | `BUN_ffill` | 0.0571 |
| 12 | `shock_index` | 0.0500 |
| 13 | `Temp_max_3h` | 0.0493 |
| 14 | `Resp_min_12h` | 0.0486 |
| 15 | `Potassium_mean_12h` | 0.0484 |

![SHAP summary plot](outputs/figures/shap_summary.png)

*Every dot is one patient-hour; horizontal position is that feature's push toward ("sepsis," right) or away from ("no sepsis," left) the prediction, and color is the feature's own value (red = high, blue = low).*

**Clinical sense-check:** none of this is surprising to anyone familiar with Sepsis-3 criteria. **Lactate** — a marker of tissue hypoperfusion — dominates, closely followed by the **partial SIRS score** (the classic bedside sepsis screening heuristic) and **temperature/respiratory** trend features (fever and tachypnea are core SIRS criteria). A gradient-boosted model, given nothing but raw rolling-window statistics and no hand-coded medical knowledge beyond the four clinical-ratio features, independently rediscovers that lactate and SIRS-adjacent signals matter most — that's a meaningful sanity check that it has learned something clinically real rather than a spurious correlation.

The dependence plots below make two of the top relationships concrete:

![SHAP dependence: partial SIRS score](outputs/figures/shap_dependence_partial_sirs_score.png)

*Each additional SIRS criterion met pushes the prediction up in a clear step pattern — 0 criteria pulls the prediction down, 2+ pushes it up, with diminishing separation past 2, consistent with the standard clinical convention of using "≥2 criteria" as the SIRS-positive threshold rather than treating the count as linearly informative.*

![SHAP dependence: shock index](outputs/figures/shap_dependence_shock_index.png)

*Shock index (heart rate ÷ systolic blood pressure) shows a fairly monotonic relationship with SHAP value — as HR climbs relative to SBP (a classic sign of compensated shock), the model's push toward "sepsis" increases smoothly rather than jumping at one cutoff.*

Both plots above are `clinical_ratios` features — composite scores, not raw trend signals. To show what the `slopes_velocity` family (§4, `02_feature_engineering.py`) actually contributes, here's the dependence plot for `HR_slope_6h` (rank #20 by mean |SHAP| — a real but not top-15 feature, included here for what it illustrates rather than for its ranking):

![SHAP dependence: HR slope, 6h](outputs/figures/shap_dependence_HR_slope_6h.png)

*For most of its range, a rising heart rate (`HR_slope_6h > 0`) pushes the prediction toward "sepsis" and a falling or flat heart rate pushes it away — the model has learned that the *direction* a vital is moving matters, not just its current level, which is exactly the motivation for including slope/velocity features at all. The one departure from that pattern is a small cluster of extreme negative slopes (roughly −40 to −45, a heart rate collapsing very fast) sitting back near zero/slightly negative SHAP instead of continuing the downward trend — plausibly a late-stage physiological crash (bradycardic decompensation) that the model has too few examples of to attribute confidently, rather than a sign that rapidly falling HR is reassuring.*

A more surprising finding: two *missingness* features — `Lactate_hours_since_last` (rank 3) and `Bilirubin_total_hours_since_last` (rank 6) — outrank most raw vital-sign features. This validates the missingness-as-signal design decision from script 02: the model is picking up on clinician ordering behavior (more frequent labs when a clinician is worried) as a genuinely useful, if indirect, predictive signal.

Here is what that attribution looks like for one individual, correctly-flagged positive case:

![SHAP waterfall, positive case](outputs/figures/shap_waterfall_case_positive.png)

*Starting from the model's average output (0.029), this patient-hour's elevated temperature (`Temp_ffill`, `Temp_max_6h`, `Temp_max_3h`), elevated lactate (`Lactate_max_12h`), abnormal bilirubin, and a partial SIRS score of 2 stack up to push the final prediction to 2.578 (in log-odds space) — a clear, individually-inspectable justification for the alert, not a black-box number.*

### `06_leadtime_alarm_fatigue.py` — translating AUROC into something a clinician cares about

AUROC and AUPRC are abstract at the bedside. This script reframes the model in operational terms: *how early does it warn, and how many false alarms does a nurse have to tolerate per true warning?* Two things are computed at the model's best-utility operating threshold (0.500):

1. **Lead time distribution** — for every septic patient who received at least one alert before/at their actual sepsis onset, how many hours of advance warning did they get?
2. **Alarm-fatigue comparison** — the engineered model vs. the naive rule-based screen every ICU already has available (**SIRS ≥ 2**, i.e., flag when a patient meets at least 2 of the 4 SIRS criteria), matched to the *same sensitivity* so the comparison isn't rigged by picking favorable operating points.

| | |
|---|---|
| Septic patients in test set | 2,932 |
| Caught by ≥1 alert before/at onset | **87.6%** |
| Median lead time (caught patients) | **22.0 hours** |
| % of caught patients warned ≥6h ahead | **68.6%** |

**Alarm fatigue, matched sensitivity:**

| Rule | Sensitivity | False-alarm rate (non-septic hours) |
|---|---|---|
| Naive SIRS ≥ 2 | 0.5061 | **0.2890** |
| Engineered model (matched) | 0.5072 | **0.1268** |
| Engineered model (default 0.5 threshold) | 0.5998 | 0.1796 |

At matched sensitivity (~50.7% either way), the engineered model cuts the false-alarm rate roughly **in half** (0.127 vs. 0.289) relative to the SIRS≥2 screen every ICU already runs. This — not the raw AUROC number — is the practically meaningful headline of the whole project: for the same number of true cases caught, a nurse would get roughly half as many false pages.

### `07_olap_and_export.py` — OLAP demonstration + Power BI export

Two things happen here: the four classic OLAP cube operations are demonstrated directly against the warehouse in SQL, and the same star schema is exported to CSV so it can be rebuilt interactively in Power BI (full walkthrough in `POWERBI_HANDOFF.md`).

**OLAP operations, in DuckDB SQL:**

| Operation | What it means here | Result |
|---|---|---|
| **Roll-up** (hourly → daily) | Aggregate `fact_vitals_hourly` up to one row per patient per day | 105,665 rows |
| **Roll-up** (daily → whole stay) | Aggregate further to one row per patient's entire ICU stay | 40,336 rows |
| **Drill-down** (hospital → patient → hour) | Start at hospital-level averages, descend into a single patient, then that patient's hour-by-hour detail | Hospital 1: avg HR 84.89 (20,336 patients); Hospital 2: avg HR 83.83 (20,000 patients) |
| **Slice** (`SepsisLabel = 1` only) | Cut the cube down to only septic patient-hours | 27,916 rows |
| **Dice** (hospital 1 AND septic AND `Lactate > 2.0`) | Filter on multiple dimensions simultaneously | 2,438 rows |

**Power BI export** — three CSVs written to `outputs/powerbi_export/`, so the same roll-up/drill-down/slice/dice operations can be rebuilt as interactive Power BI visuals rather than only existing as one-off SQL queries:

| File | Rows | Purpose |
|---|---|---|
| `dim_patient.csv` | 40,336 | Patient dimension |
| `dim_hospital.csv` | 2 | Hospital dimension |
| `fact_vitals_olap.csv` | 1,552,210 × 15 cols | Fact table, ready for `Get Data → Text/CSV` import |

The intended Power BI workflow (detailed in `POWERBI_HANDOFF.md`): import all three CSVs, connect `patient_id`/`hospital_id` as relationships in Model view, build a Matrix visual with a `hospital → day_bucket → hour` row hierarchy (roll-up/drill-down, interactively), and add slicers on `hospital_id`/`SepsisLabel` (slice/dice).

### `08_association_rules.py` — association rule mining on clinical flags

Continuous vitals aren't natural input for Apriori, so this bins six vitals into binary abnormal/normal flags using standard clinical cutoffs (the same spirit as SIRS/qSOFA), then mines which **combinations** of abnormal flags co-occur and how strongly each combination associates with `SepsisLabel = 1` — the same technique retail analytics uses for "customers who bought X also bought Y," applied here to "patient-hours with flag X also tend to have flag Y (and sepsis)."

**Flag prevalence in the data:**

| Flag | Threshold | Prevalence |
|---|---|---|
| `HR_high` | HR > 90 | 33.0% |
| `WBC_abnormal` | WBC > 12 or < 4 | 29.5% |
| `Resp_high` | Resp > 20 | 28.9% |
| `SBP_low` | SBP < 100 | 13.5% |
| `Temp_abnormal` | Temp > 38 or < 36 | 12.1% |
| `Lactate_high` | Lactate > 2.0 | 9.1% |
| `Sepsis` | `SepsisLabel = 1` | 1.8% |

374 total rules were found; 28 have `Sepsis` as a consequent; 9 of those are compound (2+ antecedent flags).

**Top rules by lift, antecedent → Sepsis:**

| Antecedent | Lift | Confidence |
|---|---|---|
| `{Resp_high, Temp_abnormal}` | **2.83** | 5.09% |
| `{HR_high, Temp_abnormal}` | 2.82 | 5.06% |
| `{HR_high, WBC_abnormal, Resp_high}` | 2.28 | 4.11% |
| `{WBC_abnormal, Resp_high}` | 1.92 | 3.44% |
| `{HR_high, Resp_high}` | 1.88 | 3.38% |
| `{HR_high, WBC_abnormal}` | 1.83 | 3.30% |
| `{Temp_abnormal}` (best single-flag) | **1.99** | 3.59% |
| `{Lactate_high}` | 1.69 | 3.04% |

The best 2-flag compound rule (`{Resp_high, Temp_abnormal}`, lift 2.83) beats the best single-flag rule (`Temp_abnormal` alone, lift 1.99) by about **42%**. This is an independent, empirically-derived confirmation of *why* SIRS/qSOFA-style multi-criteria scoring outperforms single-symptom screening: it reaches the same conclusion the SHAP ranking reached (`partial_sirs_score` at rank #2), from a completely different, model-free technique.

### `09_hierarchical_clustering.py` + `clustering_phenotypes.py` — sepsis phenotype discovery

Two clustering algorithms attempt to answer "are there distinct clinical *subtypes* of septic patients?" (e.g. a hyperinflammatory phenotype vs. a hypotensive phenotype), applied to per-patient whole-stay trajectory summaries (mean/std/min/max of each vital across the entire ICU stay, not individual hours).

**k-means** (`clustering_phenotypes.py`), silhouette-scored across k=2 to k=7:

![k-means silhouette by k](outputs/figures/cluster_silhouette.png)

| k | Silhouette |
|---|---|
| **2** | **0.154** (best) |
| 3 | 0.117 |
| 4 | 0.108 |
| 5 | 0.109 |
| 6 | 0.087 |
| 7 | 0.089 |

At k=2, on the full complete-case population (31,857 patients): cluster 0 (19,736 patients, 7.47% sepsis rate) vs. cluster 1 (12,729 patients, 7.01% sepsis rate).

**Hierarchical clustering, Ward linkage** (`09_hierarchical_clustering.py`) — Ward-linkage clustering needs a full O(n²) pairwise distance matrix, which is only tractable on a subsample, not the full 31,857-patient population, so this runs on a **stratified random subsample of 8,000 patients** (seed=42, stratified on sepsis outcome — the 7.11% sepsis rate is preserved exactly between the full population and the subsample). For a fair comparison, k-means was re-run on the identical 8,000-patient subsample:

| Method | N | Silhouette |
|---|---|---|
| Hierarchical (Ward, k=2) | 8,000 (stratified subsample) | **0.095** |
| k-means (k=2), same subsample | 8,000 | 0.157 |
| k-means (k=2), full population | 31,857 | 0.160 |

At k=2, hierarchical cluster 1 (3,245 patients, 7.67% sepsis rate) vs. cluster 2 (4,755 patients, 6.73% sepsis rate).

![Hierarchical dendrogram](outputs/figures/hierarchical_dendrogram.png)

**Reported honestly as a null result:** both methods agree there is **no clinically meaningful sepsis phenotype** in this feature space. Silhouette scores are low across the board (0.095–0.160, well below the ~0.5+ that would indicate genuinely separable clusters), and — more importantly — the sepsis rate is nearly flat across every cluster either method produces (differences of only 0.5–1 percentage points). Whatever these clusters are picking up on (plausibly coarse things like general illness severity or ward assignment), it isn't sepsis subtype. Two independent methods, different sample sizes, different linkage logic, converging on the same negative conclusion is itself a reasonably strong piece of evidence that this null result is real rather than a modeling artifact — it's reported as-is rather than tuned to manufacture a more "interesting" outcome.

### `10_classical_classifiers.py` — Decision Tree & Naive Bayes

Trains `sklearn.tree.DecisionTreeClassifier` and `sklearn.naive_bayes.GaussianNB` on the same warehouse tables, same patient-grouped `GroupKFold(5)` split, and the same AUROC/AUPRC/normalized-utility metrics as the XGBoost models — so these numbers sit directly next to the rest of the results rather than existing in isolation. Each classifier is run twice: once on the 15 raw forward-filled features (matching `03_baseline_model.py`) and once on the full 289-column engineered set (matching `04_engineered_model.py`). Neither model handles missing values natively the way XGBoost does, so a per-column median imputation is applied just for this script — a deliberate, disclosed deviation from the XGBoost scripts, which pass raw (possibly-missing) values straight through.

| Model | AUROC | AUPRC | Normalized utility | Features |
|---|---|---|---|---|
| Decision Tree (engineered) | **0.7152** | **0.0540** | **0.2065** | 289 |
| Naive Bayes (engineered) | 0.7093 | 0.0387 | 0.1431 | 289 |
| Decision Tree (raw) | 0.7056 | 0.0476 | 0.1793 | 15 |
| Naive Bayes (raw) | 0.6957 | 0.0367 | 0.1410 | 15 |

Both classical classifiers land well below XGBoost at every comparable feature count (engineered XGBoost: 0.7918 AUROC vs. engineered Decision Tree: 0.7152, engineered Naive Bayes: 0.7093), which is expected — neither a single decision tree nor a Gaussian-likelihood Bayes model can capture the nonlinear, high-order feature interactions a 300-tree gradient-boosted ensemble can. What's more interesting is that **engineered features still help both classical models** over their raw-feature counterparts (Decision Tree: +0.0096 AUROC, Naive Bayes: +0.0136 AUROC) — smaller gains than XGBoost's (+0.0346), but the same direction. That's a useful cross-check: the value of the engineered feature set isn't an XGBoost-specific artifact, it transfers across model families, with diminishing returns for simpler models that can't exploit as much of the extra feature space.

One caveat worth carrying into any downstream use of Naive Bayes here: its best-utility threshold lands at the sweep floor (0.01) for both feature sets — a symptom of Gaussian Naive Bayes' independence assumption producing poorly-calibrated, extreme predicted probabilities on a heavily correlated feature set (rolling stats and raw values for the same vital are, by construction, highly correlated — exactly what "naive" independence assumes away). This doesn't affect the AUROC/AUPRC ranking (both are threshold-independent), but it does mean Naive Bayes' probability outputs shouldn't be read as literal risk percentages the way XGBoost's or the Decision Tree's more reasonably can.

### `11_dbscan_clustering.py` — density-based clustering

A third, structurally different clustering approach on the same per-patient trajectory summary used by the other two clustering scripts (mean/std/min/max of `HR`, `O2Sat`, `Temp`, `SBP`, `MAP`, `DBP`, `Resp`). Unlike k-means or hierarchical clustering, DBSCAN doesn't take a target cluster count — it takes a neighborhood radius `eps` and a minimum point count `min_samples`. `eps` is chosen via the standard **k-distance elbow heuristic**: plot every point's distance to its `min_samples`-th nearest neighbor, sorted ascending, and pick the point of maximum curvature, rather than hand-tuning a value that produces a preferred outcome. `min_samples=10` follows the common rule-of-thumb of roughly 2× dimensionality for this 18-feature space.

![DBSCAN k-distance elbow](outputs/figures/dbscan_kdistance_elbow.png)

| | |
|---|---|
| Patients (complete-case) | 32,465 |
| Chosen `eps` (k-distance elbow) | 3.551 |
| `min_samples` | 10 |
| Clusters found | **1** |
| Noise points | 662 (2.0%) |

| Cluster | N patients | Sepsis rate |
|---|---|---|
| Noise (`-1`) | 662 | **14.35%** |
| Cluster 0 | 31,803 | 7.14% |

DBSCAN finds essentially **one dense cluster containing the overwhelming majority of patients**, plus a small (2%) noise fraction, rather than multiple genuine clusters. Read alongside the k-means (silhouette 0.154–0.160) and hierarchical (silhouette 0.095) results above, this is a **third, structurally different algorithm reaching the same conclusion**: no clean sepsis phenotype exists in this feature space at the whole-patient-trajectory grain. Density-based clustering is specifically good at finding irregularly-shaped clusters that k-means' spherical assumption or Ward linkage's variance-minimizing objective could miss, so its agreement with the other two isn't just "another vote" — it closes off a specific blind spot the other two methods share.

A genuinely interesting aside: the 662 patients DBSCAN labels as noise (patients who don't fit densely into the main cluster — i.e., outliers in trajectory-feature space) have a sepsis rate **exactly double** the main cluster's (14.35% vs. 7.14%). This aligns with the outlier-analysis finding below, even though the two analyses use different feature grains (whole-stay trajectory summary here vs. individual patient-hours there) and were computed independently.

### `12_outlier_analysis.py` — outlier analysis on key vitals

Extreme lab values (very high lactate, very high or low temperature) are often the clinical signal itself in sepsis, not noise to be cleaned away — so this analysis is expected to *corroborate*, not contradict, the SHAP and association-rule findings above. Two standard, deliberately simple and interpretable outlier-detection methods are run on the six vitals already shown to matter most (`HR`, `Temp`, `Resp`, `SBP`, `Lactate`, `WBC`), at the patient-hour grain:

1. **Univariate IQR (Tukey) fencing**, per vital — flag any hour where a vital falls outside `[Q1 − 1.5×IQR, Q3 + 1.5×IQR]`.
2. **Multivariate Isolation Forest**, across all six vitals jointly, `contamination=0.02` (chosen close to the dataset's own 1.8% positive rate) — catches combinations that look unremarkable one vital at a time but are jointly unusual (e.g. moderately elevated HR *and* moderately low SBP together, neither extreme enough alone to trip an IQR fence).

Every flag is checked against `SepsisLabel` for **lift**: how much more likely a flagged hour is to be septic than the 1.8% overall base rate.

**IQR fence bounds:**

| Vital | Lower fence | Upper fence |
|---|---|---|
| HR | 37.50 | 129.50 |
| Temp | 35.05 | 38.65 |
| Resp | 7.25 | 29.25 |
| SBP | 60.50 | 184.50 |
| Lactate | −0.45 | 3.79 |
| WBC | −1.40 | 22.60 |

**Sepsis lift by method:**

![Outlier sepsis lift](outputs/figures/outlier_sepsis_lift.png)

| Method | % of rows flagged | Sepsis rate (flagged) | Lift |
|---|---|---|---|
| IQR, `Temp` only | 1.54% | 6.94% | **3.86×** |
| IQR, 2+ vitals simultaneously | 1.21% | 5.87% | **3.27×** |
| Isolation Forest (all 6 vitals) | 2.00% | 5.77% | **3.21×** |
| IQR, `Resp` only | 3.78% | 4.51% | 2.51× |
| IQR, `HR` only | 1.12% | 4.43% | 2.46× |
| IQR, `WBC` only | 2.63% | 4.19% | 2.33× |
| IQR, any single vital | 10.96% | 4.13% | 2.30× |
| IQR, `Lactate` only | 2.14% | 3.34% | 1.86× |
| IQR, `SBP` only | 1.14% | 2.17% | 1.20× |

Outlier-flagged hours are consistently, substantially more likely to be septic than the base rate across every method and vital tested — even the weakest (`SBP` alone, 1.20×) shows a positive lift, and the strongest single-vital result (`Temp`, 3.86×) beats the Isolation Forest's full multivariate result. This is a clean validation, from a third completely different angle (unsupervised outlier detection with no model training involved) of the same story SHAP and the association rules already told: **temperature and lactate carry the strongest individual sepsis signal**, and **combinations of simultaneously-abnormal vitals carry more signal than any single vital alone** (2+ vitals: 3.27× lift vs. 2.30× for any single vital) — the same "compound signals beat single flags" pattern found independently by `08_association_rules.py`.

`SBP`'s weak lift (1.20×) is expected rather than a red flag: the Sepsis-3 definition this dataset's label is built on emphasizes lactate and organ-dysfunction scores over blood pressure directly, and hypotension in sepsis is typically a *later* sign than the fever/tachypnea/elevated-lactate signal the stronger-lift vitals more directly capture.

### `13_fairness_audit.py` — subgroup performance audit

Re-scores the engineered model's out-of-fold predictions — the same predictions already saved by `04_engineered_model.py`, so this needs no retraining — grouped by three axes already present in `dim_patient`: `age_bracket`, `gender_label`, `hospital_label`. AUROC, AUPRC, and normalized utility are computed separately per subgroup. This is the standard "does the model work equally well for everyone?" check a clinical model needs before deployment is even a reasonable conversation.

![Fairness audit by subgroup](outputs/figures/fairness_audit_engineered.png)

**By subgroup:**

| Axis | Group | n patients | AUROC | AUPRC | Normalized utility |
|---|---|---|---|---|---|
| Age bracket | <40 | 4,379 | 0.7883 | 0.0738 | 0.3063 |
| Age bracket | 40–59 | 12,489 | 0.7903 | 0.0856 | 0.3158 |
| Age bracket | **60–74** (best AUROC) | 14,073 | **0.8015** | 0.0836 | 0.3462 |
| Age bracket | **75+** (worst AUROC) | 9,395 | **0.7811** | 0.0813 | 0.2944 |
| Gender | Female | 17,770 | 0.7920 | 0.0802 | 0.2914 |
| Gender | Male | 22,566 | 0.7915 | 0.0836 | 0.3401 |
| Hospital | **hospital_system_2** (best AUROC) | 20,000 | **0.8084** | 0.0805 | 0.2966 |
| Hospital | **hospital_system_1** (worst AUROC) | 20,336 | **0.7718** | 0.0837 | 0.3350 |

**Largest gap per axis:**

| Axis | Max group | Max AUROC | Min group | Min AUROC | Gap |
|---|---|---|---|---|---|
| `hospital_label` | hospital_system_2 | 0.8084 | hospital_system_1 | 0.7718 | **0.0366** |
| `age_bracket` | 60–74 | 0.8015 | 75+ | 0.7811 | 0.0204 |
| `gender_label` | Female | 0.7920 | Male | 0.7915 | 0.0005 |

Gender shows essentially no AUROC disparity (0.0005 — noise-level), but **hospital site** is by far the largest fairness axis (0.0366), nearly double the age-bracket gap (0.0204). The oldest patients (75+) are both the worst-served age group by AUROC and the group where a missed or late sepsis call is arguably most consequential — worth flagging explicitly as a deployment caveat.

**AUROC parity does not imply utility parity.** Looking at normalized utility (the metric that reflects clinical value, not just ranking quality) flips two of the three rankings:

- **Hospital:** `hospital_system_2` has the *higher* AUROC (0.8084) but the *lower* utility (0.2966); `hospital_system_1` has the *lower* AUROC (0.7718) but the *higher* utility (0.3350). A single shared decision threshold interacts with each hospital's local prevalence and score distribution differently, so the hospital that ranks patients better in relative terms isn't necessarily the hospital where alerts translate to better time-weighted outcomes.
- **Gender:** near-identical AUROC (0.7920 vs. 0.7915) but a large utility gap (0.2914 for Female vs. 0.3401 for Male — a ~17% relative difference). The ranking-quality metric says "no disparity"; the deployment-relevant metric says otherwise.
- **Age** is the one axis where AUROC and utility agree directionally (60–74 best on both, 75+ worst on both), making it the most straightforwardly interpretable of the three gaps.

A fairness audit that stopped at AUROC would have reported "gender: fine, hospital: some gap, age: some gap" and missed that the threshold-dependent utility metric tells a materially different, and more deployment-relevant, story — which is why both are reported side by side rather than just one.

### `14_cross_hospital_generalization.py` — does the model transfer across hospitals?

A stricter generalization test than ordinary k-fold cross-validation. Instead of training and testing on a random patient-level split drawn from *both* hospitals (as every other script does), this trains on **one hospital entirely** and tests on **the other hospital entirely**, in both directions, and compares against the in-distribution (mixed-hospital) result already on file from `04_engineered_model.py`.

| Direction | Train n | Test n | AUROC | AUPRC | Best-threshold utility |
|---|---|---|---|---|---|
| A: train hospital 1 → test hospital 2 | 20,336 | 20,000 | 0.7381 | 0.0545 | 0.1931 (threshold 0.54) |
| B: train hospital 2 → test hospital 1 | 20,000 | 20,336 | 0.7226 | 0.0609 | 0.2439 (threshold 0.46) |
| In-distribution (mixed, `GroupKFold`) | — | — | **0.7918** | **0.0821** | **0.3203** |

Generalizing across hospital systems costs the model **0.05–0.07 AUROC and roughly 25–40% of its clinical utility**, relative to the in-distribution result. Utility takes the harder hit than AUROC in both directions (utility drops 40% in direction A, 24% in direction B, vs. AUROC drops of 7% and 9% respectively) — consistent with utility being threshold-sensitive and therefore more exposed to a shift in the *score distribution* between hospitals, not just a shift in ranking quality.

This result is the direct causal explanation for the hospital-axis gap already surfaced by `13_fairness_audit.py`: a mixed-hospital model trained with `GroupKFold` sees both hospitals during training, and the 0.0366 AUROC gap in the fairness audit is what's left over from site-specific distribution shift *even after* training on both. This experiment isolates that same shift in its more extreme form — what happens if the model has never seen the target hospital at all — and shows the gap roughly doubles (0.05–0.07 vs. 0.0366) under that harder condition. Read together, the two scripts tell one consistent story: hospital site is a real, non-trivial source of distribution shift in this dataset, more so than age or gender, and a model trained at one site should not be assumed to transfer to another without re-validation.

**Practical implication:** if this model were ever deployed at a hospital not represented in the training data, these numbers — not the headline 0.7918 in-distribution AUROC — are the honest expectation for out-of-the-box performance, and local recalibration or threshold-tuning at minimum (ideally a hospital-specific fine-tune) should be treated as a deployment prerequisite, not a nice-to-have.

### `15_conformal_prediction.py` — calibrated uncertainty via split-conformal classification

Wraps the engineered XGBoost model with **split conformal prediction** (via [MAPIE](https://mapie.readthedocs.io/)'s `SplitConformalClassifier`) to produce, for every patient-hour, a *prediction set* rather than a single point probability. Instead of "12% chance of sepsis," the output is a set like `{no-sepsis}`, `{sepsis}`, or `{no-sepsis, sepsis}` (both — "the model isn't confident enough to commit"), with a **distribution-free statistical guarantee** that the true label falls inside the predicted set at least 90% of the time (`confidence_level=0.9`), regardless of how well-calibrated the underlying model's probabilities actually are.

**Data split** (a three-way split on top of the usual patient-level grouping): 24,201 patients / 932,292 rows for training the base model, 8,067 patients / 309,265 rows held out purely for **conformal calibration** (computing the nonconformity-score threshold), and 8,068 patients / 310,653 rows for final test evaluation — calibration and test sets must be disjoint from each other and from training for the coverage guarantee to hold.

| Metric | Value |
|---|---|
| Target confidence level | 90% |
| **Empirical coverage** (true label in set) | **89.74%** |
| Confident "no sepsis" (singleton set) | 72.94% of hours |
| Confident "sepsis" (singleton set) | 10.51% of hours |
| **Uncertain** (`{both}`, flagged for review) | **16.56% of hours** |
| Empty set (should essentially never happen) | 0.00% of hours |
| Accuracy on confident hours | **87.71%** |
| Accuracy on uncertain hours | **57.52%** |
| Normalized utility, full test cohort | 0.3217 |
| Normalized utility, confident hours only | **0.3871** |

**Empirical coverage matches the target almost exactly:** 89.74% observed vs. 90% target is well within expected sampling noise for a calibration set of this size — exactly what a correctly-implemented split-conformal method should produce. This is the whole point of conformal prediction over an ad-hoc probability threshold: the 90% guarantee isn't a hope, it's a property that's been empirically checked and holds.

**The uncertainty flag is doing real work, not padding:** accuracy on the 16.56% of hours flagged as uncertain (57.52%) is barely better than a coin flip, while accuracy on the 83.44% of hours the model is confident about is 87.71% — a **30-point accuracy gap**. This is exactly the desired behavior: the conformal set is correctly identifying the subset of patient-hours where the point prediction is unreliable, which is a genuinely actionable clinical signal ("route this patient-hour to a human for a second look") that a bare probability score doesn't give you.

**Restricting to confident predictions raises clinical utility by ~20%:** normalized utility on the confident-only subset (0.3871) is notably higher than on the full test cohort (0.3217 — itself consistent with the 0.3203 in-distribution result from `04_engineered_model.py`, a useful cross-check that this held-out split reproduces the earlier headline number). This suggests a two-tier alerting design in practice: act automatically on the ~83% of hours the model is sure about, and route the ~17% flagged as uncertain to clinician review rather than trusting the point estimate blindly. The empty-set rate of exactly 0.00% also confirms the conformal procedure never produces a degenerate "neither label is plausible" output, which would be hard to act on operationally.

### Phase 6 — does a bigger/fancier model actually beat the tree?

Scripts 1–15 answer the core capstone brief with one model family (XGBoost) plus two classical baselines (Decision Tree, Naive Bayes). Scripts 16–18 are a later, self-contained extension that puts three deep-learning architectures up against that same XGBoost engineered model, each one chosen to test a *specific* hypothesis raised earlier in the project rather than "try a neural net and see":

| Script | Architecture | Question it answers | Reads from |
|---|---|---|---|
| `16_grud_model.py` | GRU-D (Che et al., 2018) | Can a network *learn* the missingness-decay signal `02_feature_engineering.py` hand-computed as `hours_since_last_X`, instead of being handed it? | `fact_vitals_hourly` (raw, 34 cols) |
| `17_tabnet_model_kaggle.py` | TabNet (Arik & Pfister, 2021) | Does learned sparse attention over tabular columns beat gradient-boosted trees on the *exact same* 289-feature table? | `fact_features` (same 289 cols as script 04) |
| `18_transformer_model.py` | Causal Transformer encoder (Vaswani et al., 2017) | Does full self-attention over the whole stay-so-far beat a recurrent hidden state at finding sepsis-relevant patterns? | `fact_vitals_hourly` (raw, 34 cols) |

All three reuse `utility_score.py` and `delong.py` exactly as scripts 03–15 do, and all three still split by `patient_id` — but GRU-D and the Transformer use a 70/15/15 patient-level train/val/test split (not `GroupKFold`) because early-stopping a backprop-through-time model needs a held-out validation set that k-fold CV doesn't provide.

**Compute-constraint disclosure, stated once here rather than three times below:** a full `GroupKFold(5)`, refit-per-fold, over all ~40,336 patients is not tractable for either backprop-through-time recurrence (GRU-D) or O(T²) self-attention (Transformer) on commodity hardware — the same class of constraint already disclosed for `09_hierarchical_clustering.py`'s Ward-linkage subsample. GRU-D and the Transformer are therefore both trained on the **same** stratified random subsample of 8,000 patients (seed=42, stratified on `is_ever_septic`, identical helper function to script 09), so the two are compared to each other on identical patients as well as both being compared to XGBoost. TabNet doesn't share this constraint the same way — it needed a *row*-subsample only for local development iteration speed, and was subsequently run on the full 1,552,210-row table on a Kaggle GPU once the actual bottleneck was found (see below), so its headline number is a full-population result like every XGBoost table in this README.

#### `16_grud_model.py` — GRU-D: does a recurrent net learn the missingness signal end-to-end?

GRU-D augments a GRU cell with a learned per-variable decay term, `gamma_x = exp(-relu(W_x · delta + b_x))`, that decides at every hour how much to still trust a variable's last observed value given how long ago (`delta`, hours) it was measured — decaying the imputed value toward the population mean the longer a lab has gone unmeasured. This is the same underlying idea as `Lactate_hours_since_last`, but *learned* as part of the architecture rather than hand-computed in SQL, so this script is a direct end-to-end test of whether the network rediscovers script 05's SHAP finding that `Lactate_hours_since_last` (rank #3) and `Bilirubin_total_hours_since_last` (rank #6) outrank most raw vital-sign features.

| | |
|---|---|
| Train / val / test patients | 5,600 / 1,200 / 1,200 |
| Raw features | 34 (8 vitals + 26 labs, no engineered columns) |
| Epochs run (of 20 max) | 16 (early-stopped on val AUPRC, patience 4) |
| **Test AUROC** | **0.7404** |
| Test AUPRC | 0.0541 |
| Best-threshold normalized utility | 0.2415 (threshold 0.58) |

**DeLong's test vs. XGBoost, on the identical 1,200-patient test subsample (46,561 paired predictions):** XGBoost engineered AUROC 0.7708 → GRU-D 0.7404, z=3.11, p=0.0018 — a statistically real gap, XGBoost still wins. (0.7708 here is the engineered model's score restricted to just this subsample's test hours, not its 0.7918 full-population headline — this is an apples-to-apples comparison on the same rows, not a comparison against the number quoted everywhere else in this README.)

**Did the network rediscover the SHAP missingness ranking? Only partially — reported honestly rather than rounded up to a clean story.** Ranking all 34 vitals/labs by their learned decay rate (lower `gamma_x` = the network decays that variable's stale readings faster, i.e. treats freshness as more urgent):

| Fastest-decaying (most freshness-sensitive) | `gamma_x` | | Slowest-decaying (~never decays) | `gamma_x` |
|---|---|---|---|---|
| `Bilirubin_direct` | 0.042 | | `Temp` | 1.000 |
| `Potassium` | 0.043 | | `Phosphate` | 1.000 |
| `Glucose` | 0.044 | | `FiO2` | 1.000 |
| `DBP` | 0.058 | | `BaseExcess` | 1.000 |
| `SaO2` | 0.072 | | `TroponinI` | 1.000 |

`Lactate` lands at **rank #10/34** by decay speed (`gamma_x` = 0.325) and `Bilirubin_total` at **#11/34** (`gamma_x` = 0.481) — real signal (both decay meaningfully faster than the 22 variables pinned near `gamma_x` ≈ 1.0, which the network has essentially learned to treat as "trust the last value indefinitely"), but not the #3/#6 model-wide dominance SHAP found for their hand-computed `hours_since_last` counterparts. The two methods agree on the qualitative point — recency-of-measurement for a handful of labs (not vitals) carries learnable signal — but disagree on exactly which labs matter most: GRU-D's top-5 are dominated by metabolic/electrolyte labs (`Bilirubin_direct`, `Potassium`, `Glucose`) rather than `Lactate` specifically. Plausible reasons this isn't a clean replication rather than a contradiction: GRU-D's decay term is fit jointly with a much smaller model on 27× fewer patients than the SHAP analysis's full-population XGBoost fit, and it only ever sees the *raw* signal, never the actual `_hours_since_last` feature engineering that gave SHAP such a direct hook to attribute credit to. This is reported as a partial, not full, corroboration — the same reporting-outcomes-honestly standard already applied to the null clustering result in §4.

#### `17_tabnet_model_kaggle.py` — TabNet: does attention beat trees on the same feature table?

A deliberate model-only swap: TabNet reads the identical 289-column `fact_features` table, the identical patient-grouped `GroupKFold(5)`, and is scored with the identical utility metric and DeLong helper script 04 uses — no new SQL, no new feature engineering. The question is treated as genuinely open rather than assumed: does learned sparse attention over tabular columns beat gradient-boosted trees here, or does this dataset reproduce the well-published finding that GBTs tend to beat deep tabular architectures at moderate sample sizes (Shwartz-Ziv & Armon, 2022)?

Two runs exist because of a real debugging story worth documenting rather than hiding: an initial full-table attempt took ~2 hours instead of an expected ~20 minutes. Profiling (the `[timing]` log lines below) traced this to two silent bottlenecks that have nothing to do with GPU compute — an unlogged `np.nanmedian` call over a ~1M-row × 289-col array (swapped for `bottleneck`'s faster implementation) and `patient_id` being carried as an object-dtype string array through every `GroupKFold`/`np.unique`/`np.isin` call (fixed by factorizing to `int32` first). A third, larger bottleneck was found afterward: `pytorch-tabnet`'s `fit()` silently runs an extra full forward pass over the *entire* training fold at the end of every fold to compute feature importances (`compute_importance=True` by default), which alone cost 500–600 seconds per fold on the full table — longer than the training loop itself. The full-table run below sets `compute_importance=False` and instead computes importances itself via the same public `explain()` API, on a 10% subsample of each fold's training rows, averaged across folds.

| Run | Rows | Patients (OOF) | AUROC | AUPRC | Best-threshold utility | vs. XGBoost DeLong (z, p) |
|---|---|---|---|---|---|---|
| Local dev run (`ROW_SUBSAMPLE_FRAC=0.35`, pre-fix) | 545,863 | 14,157 | 0.7089 | 0.0560 | 0.2123 (thr. 0.54) | z=30.62, p≈0 |
| **Full table (Kaggle T4×2 GPU, post-fix)** | **1,552,210** | **40,336** | **0.7597** | **0.0640** | **0.2687 (thr. 0.58)** | **z=29.96, p≈0** |
| XGBoost engineered (script 04, for reference) | 1,552,210 | 40,336 | 0.7918 | 0.0821 | 0.3203 | — |

The full-table run is the number to treat as TabNet's actual result: it uses the same population as every other headline figure in this README, and per-fold timing on it (`fit≈250–272s`, `predict≈3.6s`, `feature_importances≈8s`) confirms the two root-caused bottlenecks — not GPU/RAM budget — were what made the earlier full-table attempt slow. The 0.35-subsample run is kept in the repo and reported here for transparency (it's what informed the debugging), not as a second independent result to average with the first.

**On the full table, TabNet lands above the raw baseline (0.7572) but clearly below the engineered XGBoost model (0.7918)** — DeLong's test rejects "no difference" overwhelmingly (z≈30, p≈0) in XGBoost's favor either way the table is sliced. This is the expected, well-published outcome for a deep tabular architecture against a tuned GBT at this sample size, not a surprise: TabNet's own comparison in Arik & Pfister (2021) shows a similar pattern on several benchmarks, and the finding here is an independent confirmation on real clinical data rather than a contradiction of TabNet's design.

**Feature-attention agreement with SHAP is real but modest.** Both the subsampled and full-table runs put `Lactate_hours_since_last` and `Bilirubin_total_hours_since_last` — the same two missingness features SHAP ranked #3 and #6 model-wide — inside TabNet's own top 4 by attention-mask importance, which is a genuine point of agreement between two structurally different attribution methods. But the rank correlation across all 289 shared features is weak (Spearman ρ=0.189, p=0.0012 on the full run — statistically non-zero, but far from strong agreement), and only 4 of the top-15 features overlap between the two rankings. Read together with `16_grud_model.py`'s partial-replication result above, both deep architectures independently confirm that lab-recency is *a* real signal, without independently reproducing SHAP's *precise* ranking of it — a consistent, honest pattern across two unrelated deep models rather than an isolated fluke in either one.

#### `18_transformer_model.py` — Transformer: does whole-stay attention beat recurrence?

A causally-masked Transformer encoder that can attend from hour *t* directly back to any earlier hour in one layer, instead of routing information through *T* sequential GRU updates the way GRU-D must. To keep the three deep architectures' relationship to the missingness signal distinct and comparable, this script's input construction differs from both other deep models on purpose: it feeds the raw delta tensor (hours-since-last-observed — the same tensor script 16 computes) plus a binary mask directly into the attention mechanism, with no decay function assumed and no hand-computed missingness feature engineering — whatever the model does with staleness here, it worked out from `delta`/`mask` alone. It uses the exact same 8,000-patient stratified subsample and 70/15/15 split as `16_grud_model.py`, specifically so the GRU-D-vs-Transformer comparison is exact (same test patients), not merely similar.

| | |
|---|---|
| Model parameters | 73,601 |
| Train / val / test patients | 5,600 / 1,200 / 1,200 (identical to script 16) |
| Epochs run (of 20 max) | 6 (early-stopped on val AUPRC, patience 4) |
| **Test AUROC** | **0.7839** |
| Test AUPRC | **0.1032** |
| Best-threshold normalized utility | 0.3227 (threshold 0.62) |

**This is the one deep model that is not clearly beaten by XGBoost — reported carefully, given the small test set.** DeLong's test against XGBoost's engineered model on the identical 1,200-patient subsample (46,561 paired predictions) gives AUROC 0.7708 (XGBoost) → 0.7839 (Transformer), **z=−1.54, p=0.123** — numerically higher for the Transformer, but *not* a statistically significant difference at conventional thresholds. The honest reading is "not distinguishable from XGBoost on this subsample," not "beats XGBoost": p=0.123 means the observed +0.013 AUROC gap is well within what sampling noise on a 1,200-patient test set could produce. What *is* statistically decisive is the Transformer's comparison against the other deep model trained on the identical data: **AUROC 0.7404 (GRU-D) → 0.7839 (Transformer), z=−4.99, p=6.1×10⁻⁷** — on this subsample, whole-stay self-attention is a clearly better architecture than GRU-D's recurrent decay mechanism for this task, even though neither can be shown to beat the tree model outright.

**Attention is heavily recency-biased, and that's an independent confirmation of the ablation finding from §4.** Averaging the last encoder layer's self-attention weights by relative lag (how many hours back a query attends to) shows a smooth, monotonic decay: lag 0 (the current hour) receives the most attention (mean weight 0.108), falling to 0.081 at lag 1h, 0.070 at lag 2h, and continuing to decline through the 300-hour range logged. The model was free to attend anywhere in the stay so far and chose to concentrate most heavily on the most recent few hours — independently arriving, via a completely different mechanism (learned attention weights rather than hand-designed window functions), at the same conclusion `04_engineered_model.py`'s ablation already established: `rolling_stats` over short recent windows (3h/6h/12h) carry the great majority of the predictive signal in this problem, and looking much further back adds comparatively little.

**Caveat carried into the results summary and limitations below:** the Transformer's strong showing is measured on the same 1,200-patient held-out subsample as GRU-D, not on the full 40,336-patient population XGBoost's 0.7918 headline reflects — a materially smaller and differently-composed test set. This result is genuinely promising and worth a follow-up full-scale run, but it should not be read as "the Transformer beats XGBoost outright"; it should be read as "on a fair, matched, apples-to-apples comparison against XGBoost restricted to the same patients, the Transformer was competitive and the difference was not statistically significant," which is a meaningfully different and more modest claim.

### `delong.py` and `utility_score.py` — shared helper modules

- **`delong.py`** implements DeLong's test for comparing two correlated AUROCs (used by `04_engineered_model.py`) — the statistically correct way to compare two models' AUROC on the *same* test set, since a naive "0.79 > 0.76, so it's better" comparison ignores that both numbers carry sampling uncertainty.
- **`utility_score.py`** re-implements the PhysioNet/CinC 2019 Challenge's own **clinical utility metric**, verified against the Challenge's reference implementation and vectorized with NumPy per-patient for speed on ~40k patients. It's a time-weighted scoring function that rewards early true-positive predictions, penalizes false positives, and penalizes late or missed true positives, normalized so a perfect predictor scores 1.0 and a "never predict sepsis" predictor scores 0.0. This is a substantially more clinically meaningful headline number than AUROC alone, which is why it's reported alongside AUROC/AUPRC throughout the project, including the fairness audit, cross-hospital test, and conformal-prediction results above.

---

## 5. Leakage handling

Three categories of leakage were explicitly checked and are documented — including the one that couldn't be avoided:

| Type | Status | How it was handled |
|---|---|---|
| **Patient leakage** | Prevented | `GroupKFold(5)` splits by `patient_id` in every model-training script; disjoint train/test patient sets are asserted every fold, every run |
| **Temporal leakage** | Prevented | Every window-function feature in `02_feature_engineering.py` uses causal-only SQL windows (`ROWS BETWEEN N PRECEDING AND CURRENT ROW`, never `FOLLOWING`) |
| **Label-construction leakage** | **Present — a genuine dataset property, disclosed rather than hidden** | `SepsisLabel` is derived from Sepsis-3 criteria that reference some of the same lab values (e.g. lactate) used as model features. This is a known property of the PhysioNet Challenge 2019 dataset itself, not something introduced by this pipeline — but it means the reported AUROC/AUPRC should be read as "how well the model can reconstruct the Sepsis-3 rule from correlated inputs," not as a fully independent clinical prediction task from first principles. |

---

## 6. Results summary

| Model / Analysis | Headline metric |
|---|---|
| Baseline (15 raw features) | AUROC 0.7572, AUPRC 0.0634, utility 0.2504 |
| Engineered (289 features) | AUROC 0.7918, AUPRC 0.0821, utility 0.3203 |
| Statistical significance (DeLong) | z = −38.30, p ≈ 0 |
| Best single feature family | `rolling_stats` alone → AUROC 0.7737 |
| Top SHAP feature | `Lactate_max_12h` |
| Sepsis patients caught (≥1 alert) | 87.6%, median 22h lead time |
| False-alarm rate vs. naive SIRS≥2 (matched sensitivity) | 0.127 vs. 0.289 (~2× fewer) |
| Best association rule | `{Resp_high, Temp_abnormal}` → Sepsis, lift 2.83 |
| Clustering (k-means, k=2) | silhouette 0.154–0.160, no sepsis-rate separation |
| Clustering (hierarchical, k=2) | silhouette 0.095, no sepsis-rate separation |
| Clustering (DBSCAN) | 1 cluster + 2.0% noise; noise has 2× the main cluster's sepsis rate |
| Best classical classifier | Decision Tree (engineered) — AUROC 0.7152, AUPRC 0.0540 |
| Classical vs. XGBoost gap | XGBoost +0.0766 AUROC over best classical classifier (engineered features) |
| Best outlier-vs-sepsis lift | `Temp` IQR outliers → 3.86× sepsis rate vs. base rate |
| Compound outlier lift | 2+ simultaneously-abnormal vitals → 3.27× lift (vs. 2.30× for any single vital) |
| Largest fairness gap (AUROC) | hospital site, 0.0366 (h2=0.8084 vs. h1=0.7718) |
| AUROC-vs-utility disconnect | gender AUROC gap ≈0, but utility gap ≈17% relative |
| Cross-hospital generalization (avg. of both directions) | AUROC ≈0.73, utility ≈0.22 (vs. 0.7918 / 0.3203 in-distribution) |
| Conformal coverage (target 90%) | 89.74% empirical |
| Conformal uncertain-hour flag rate | 16.56% of hours |
| Accuracy: confident vs. uncertain hours (conformal) | 87.71% vs. 57.52% |
| Utility, confident-only subset vs. full cohort (conformal) | 0.3871 vs. 0.3217 (+~20%) |
| TabNet (full 1.55M-row table) vs. XGBoost engineered | AUROC 0.7597 vs. 0.7918 (XGBoost wins, DeLong z≈30, p≈0) |
| GRU-D vs. XGBoost (matched 1,200-patient subsample) | AUROC 0.7404 vs. 0.7708 (XGBoost wins, z=3.11, p=0.0018) |
| Transformer vs. XGBoost (matched 1,200-patient subsample) | AUROC 0.7839 vs. 0.7708 (not significant, z=−1.54, p=0.123) |
| Transformer vs. GRU-D (identical subsample/split) | AUROC 0.7839 vs. 0.7404 (Transformer wins, z=−4.99, p=6.1e-07) |
| Deep-model missingness replication of SHAP ranking | Partial: `Lactate`/`Bilirubin_total` recency matters to both GRU-D and TabNet, but neither reproduces SHAP's exact #3/#6 ranking |

**Reading these together:** the single headline AUROC (0.7918) understates how unevenly this model performs across hospital sites and, to a lesser extent, age groups, and it overstates how well the model would perform at a hospital it wasn't trained on. Conformal prediction offers a partial mitigation — not for the fairness gap itself, but for the more general problem of not knowing which individual predictions to trust — by explicitly separating "confident enough to act on" from "needs a second opinion," which is a meaningfully more honest deployment pattern than a single shared probability threshold.

**On the Phase 6 deep-learning comparison specifically:** the headline conclusion is that gradient-boosted trees remain the right model choice for this table at this scale — TabNet, evaluated on the full population, loses to XGBoost by a wide and statistically decisive margin, consistent with the published literature on tabular deep learning. GRU-D, on a smaller matched subsample, also loses to XGBoost decisively. The one genuinely open result is the Transformer, which was statistically indistinguishable from XGBoost on a matched 1,200-patient subsample — encouraging enough to be worth a full-population re-run, but not yet a result that would justify replacing the tree model in this project's headline numbers.

---

## 7. Limitations and honest caveats

These are stated plainly rather than smoothed over, along with why each one is what it is:

- **Label-construction leakage (§5).** `SepsisLabel` shares some inputs (e.g. lactate) with the model's own features, because that's how the Sepsis-3 definition this dataset's label is built on works. This is a property of the dataset, not a bug in this pipeline, but it means the reported AUROC/AUPRC should be read as "how well the model reconstructs the Sepsis-3 rule from correlated clinical inputs," not as proof of a fully independent early-warning signal.
- **No sepsis phenotype found.** All three clustering methods (k-means, hierarchical, DBSCAN) return a null result. This is reported as-is, not concealed or re-parameterized until a more "interesting" clustering appeared, because a null result honestly obtained is still a real finding.
- **Minor patient-count differences across scripts.** `01_etl_warehouse.py` loads 40,336 patients total, but complete-case filtering (dropping any patient with missing values in a given script's required columns) lands on different final counts depending on exactly which columns that script needs — 32,465 for the k-means run, 31,857 for hierarchical clustering, for example. This is the expected, mechanical consequence of complete-case filtering on slightly different column sets per script, not an inconsistency in the underlying data.
- **The AUROC-optimal decision threshold and the "clinically comfortable" threshold are different.** The best-utility threshold (0.500) used for the alarm-fatigue analysis produces higher sensitivity (59.98%) but a higher false-alarm rate (0.180) than the threshold matched to the naive rule's sensitivity (0.127 false-alarm rate at 50.72% sensitivity). Which threshold is "right" depends on a clinical site's tolerance for false alarms vs. missed cases — a genuine deployment decision the model itself cannot make.
- **The classical classifiers (`10_classical_classifiers.py`) are not tuned to compete with XGBoost.** Decision Tree and Naive Bayes use reasonable, undramatic defaults (`max_depth=6`, `class_weight="balanced"` for the tree; untouched `GaussianNB`), not an exhaustively-optimized configuration. Their purpose is to demonstrate that the engineered feature set's lift transfers across model families, not to claim they're competitive alternatives to the boosted-tree model — the ~0.08 AUROC gap to XGBoost reflects model capacity, not an unfair comparison.
- **DBSCAN finds effectively one cluster.** With `eps` chosen by the standard k-distance elbow method (not hand-tuned to force a particular outcome), DBSCAN places 98% of patients in a single dense cluster. This is the honest result, consistent with the other two clustering methods' own null findings, rather than re-tuning `eps`/`min_samples` until multiple clusters appear — doing that would risk manufacturing structure that isn't really there.
- **The IQR fences and Isolation Forest contamination rate are simple, disclosed heuristics, not optimized thresholds.** The 1.5× IQR multiplier is the standard Tukey convention, and Isolation Forest's `contamination=0.02` was chosen close to the dataset's actual 1.8% sepsis rate rather than fitted to maximize lift. Both are defensible, standard choices, but neither was tuned to produce the best possible lift numbers, so the reported lift values (2.3×–3.9×) should be read as a reasonable first-pass signal, not an optimized outlier-detection system.
- **Age 90+ is capped in the source data** for de-identification, as part of PhysioNet's own release process — this specifically affects the "75+" age bracket used in the fairness audit, which is worth keeping in mind when interpreting that bracket's results, since it isn't a clean, uncensored age distribution.
- **Hospital-site performance gap, both in-distribution and out-of-distribution (§6, §4 scripts 13–14).** The model is measurably worse for `hospital_system_1` by AUROC (though better by utility — see the AUROC/utility disconnect discussed in §4), and generalizes poorly to a hospital it wasn't trained on (AUROC drops to 0.72–0.74, utility drops 25–40%). Any deployment at a new site should treat the in-distribution 0.7918 AUROC as an optimistic ceiling, not an expectation.
- **The 75+ age group is both the worst-served and arguably the highest-stakes group.** A 0.02 AUROC gap sounds small in isolation, but it's the largest age-bracket gap observed, on the subgroup where a missed sepsis alert plausibly carries the most clinical risk — worth flagging explicitly rather than averaging it away in an overall AUROC.
- **The conformal guarantee is marginal, not conditional.** The 90% coverage guarantee from `15_conformal_prediction.py` holds on average across the whole calibration/test distribution, not provably per-subgroup. Given the fairness gaps found in §4/§6, it would be worth checking in a follow-up analysis whether coverage holds equally well within each hospital/age subgroup, or whether the "uncertain" flag is itself unevenly distributed across those same groups — this hasn't been checked yet, and is flagged here rather than assumed away.
- **This is real clinical data, but a research pipeline, not a validated clinical product.** Every result above should be read as evidence for what's learnable from this dataset with this methodology, not as a claim of readiness for bedside deployment — cross-hospital generalization, subgroup fairness, and label-construction leakage are all reasons a real deployment would need substantially more validation than what's shown here.
- **Two of the three Phase 6 deep models (`16_grud_model.py`, `18_transformer_model.py`) are evaluated on an 8,000-patient subsample, not the full 40,336-patient population.** Backprop-through-time recurrence and O(T²) self-attention are both substantially more memory- and compute-intensive per training step than XGBoost's tree building, so a full patient-grouped k-fold refit for either isn't tractable at this project's compute budget. The subsample uses the same stratified-by-`is_ever_septic` helper already established (and disclosed) for `09_hierarchical_clustering.py`'s Ward-linkage run, so the sepsis rate is preserved exactly (7.27% in both the full population and the subsample) — but a 1,200-patient held-out test set is a smaller, higher-variance basis for a DeLong comparison than the 40,336-patient comparisons used everywhere else in this README, and its result should be weighted accordingly. `17_tabnet_model_kaggle.py` does not share this constraint — its headline number is a full-population run — because a tabular model without a recurrent/attention time dimension over a full ICU stay is comparatively cheap to train even on the full row count once its actual bottlenecks (below) were fixed.
- **The Transformer's near-parity with XGBoost (§4, §6) is a promising but unconfirmed result, not a validated one.** It rests on a statistically non-significant comparison (p=0.123) on a 1,200-patient test set — a result this size could plausibly move in either direction with a different random subsample or a full-population re-run. It's reported prominently here because it's genuinely the most interesting Phase 6 finding, not because it's been confirmed at the same evidentiary standard as the XGBoost-vs-baseline result (which rests on 1.55M paired predictions and a DeLong p-value of effectively zero).
- **GRU-D's learned decay ranking only partially matches SHAP's missingness-importance ranking (§4).** `Lactate` and `Bilirubin_total` decay meaningfully faster than most other variables (real signal), but rank #10 and #11 of 34 rather than reproducing SHAP's #3/#6 dominance. This is reported as a partial, not full, replication rather than rounded up to "GRU-D confirms the SHAP finding" — the two methods agree on the general phenomenon (lab recency matters) without agreeing on its precise ranking, plausibly because GRU-D is fit on far fewer patients, with a much smaller model, and never sees the actual `hours_since_last` feature SHAP was attributing credit to.
- **`torch`, `pytorch-tabnet`, and `bottleneck` are not yet in `requirements.txt`.** Scripts 16–18 depend on PyTorch (16, 18) or `pytorch-tabnet`/`bottleneck` (17), none of which are declared alongside the DuckDB/XGBoost/sklearn stack the rest of the pipeline needs — `16_grud_model.py` fails fast with an explicit, actionable error message if PyTorch isn't installed, rather than a confusing stack trace, but this is a packaging gap that should be closed (with CPU-only PyTorch wheels, per the same script's own comment, so scripts 16 and 18 don't force a CUDA install on machines that don't have or need a GPU).
- **`17_tabnet_model_kaggle.py`'s full-table run needs a Kaggle GPU notebook, not this repo's local environment.** The local dev run (`ROW_SUBSAMPLE_FRAC=0.35`) is what's reproducible by running the script as-is; the full-population headline number requires uploading `sepsis.duckdb` plus `utility_score.py`/`delong.py` as a private Kaggle Dataset and running the `(full)` variant there with a GPU accelerator turned on (see the script's own header for the exact steps). This is disclosed rather than smoothed over because it means the full-population TabNet number in this README is not reproducible from a single `python src/17_tabnet_model_kaggle.py` call the way every other headline number in this project is.
- **No plots were generated for scripts 16–18.** Every other script that produces a standalone visual finding (SHAP, clustering, DBSCAN, outliers, fairness) saves a figure to `outputs/figures/`. The three Phase 6 scripts report their results as tables and log lines only — the GRU-D decay-rate ranking and the Transformer's attention-by-lag curve are both good candidates for a bar chart / line plot in a future pass, but weren't prioritized here since the tabular form was sufficient to state the findings precisely.

---

## 8. Reproducing this project

```bash
# 1. Place raw PhysioNet .psv files under data/raw/<hospital_folder>/*.psv
#    (each subfolder is treated as one "hospital system")
python src/01_etl_warehouse.py
python src/02_feature_engineering.py
python src/03_baseline_model.py
python src/04_engineered_model.py
python src/05_explainability.py
python src/06_leadtime_alarm_fatigue.py
python src/07_olap_and_export.py
python src/08_association_rules.py
python src/clustering_phenotypes.py
python src/09_hierarchical_clustering.py
python src/10_classical_classifiers.py
python src/11_dbscan_clustering.py
python src/12_outlier_analysis.py
python src/13_fairness_audit.py
python src/14_cross_hospital_generalization.py
python src/15_conformal_prediction.py

# Phase 6 -- deep-learning comparison (optional; needs torch / pytorch-tabnet,
# not yet in requirements.txt -- see the note below)
python src/16_grud_model.py
python src/18_transformer_model.py
python src/17_tabnet_model_kaggle.py     # local dev run, ROW_SUBSAMPLE_FRAC=0.35
#   (the full-population TabNet number instead needs the "(full)" variant run
#   on a Kaggle GPU notebook -- see 17_tabnet_model_kaggle.py's own header for
#   the exact upload/settings steps; it is not reproducible with a plain
#   `python` call the way every other script in this list is)
```

Install dependencies first with `pip install -r requirements.txt` (this includes `mapie>=1.0`, needed only by script 15 but included so the whole pipeline installs cleanly in one pass). **Scripts 16–18 need `torch` (16, 18) and `pytorch-tabnet` + `bottleneck` (17) as well, and none of these three are in `requirements.txt` yet** (§7) — install them separately, e.g. `pip install torch --index-url https://download.pytorch.org/whl/cpu` (CPU is enough for scripts 16/18's subsampled scale) and `pip install pytorch-tabnet bottleneck` for script 17's local run.

Scripts 10–12 only depend on `02_feature_engineering.py` having already built `fact_features`/`fact_ffill` — they don't need scripts 03–09 to have run first, so they can be run any time after step 2 if you just want to check them in isolation. Scripts 13–15 depend on `04_engineered_model.py`'s trained model and out-of-fold predictions, so run those after script 04. Scripts 16 and 18 only need `01_etl_warehouse.py` (they read raw `fact_vitals_hourly`, not the engineered table), but both look for `outputs/engineered_oof_predictions.parquet` to run their DeLong's-test comparison against XGBoost, so run script 04 first if you want that comparison rather than just the standalone AUROC/AUPRC. Script 18 additionally looks for `outputs/grud_results.csv`, so run script 16 before script 18 if you want the GRU-D-vs-Transformer DeLong comparison. Script 17 needs `02_feature_engineering.py`'s `fact_features` table, same as script 04, but does not depend on script 04 itself running first (only on its *output* being present for the optional DeLong comparison).

For the Power BI half of the OLAP demonstration (importing `outputs/powerbi_export/*.csv` and building the interactive roll-up/drill-down/slice/dice visuals), see `POWERBI_HANDOFF.md`.
