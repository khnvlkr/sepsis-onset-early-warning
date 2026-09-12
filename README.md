# Sepsis Onset Early Warning

**An end-to-end clinical machine-learning pipeline for predicting sepsis onset from hourly ICU physiological data, with temporal feature engineering, statistical model comparison, early-warning analysis, explainability, generalization, fairness, uncertainty, phenotype discovery, and deep-learning challengers.**

> **Status:** Research / retrospective benchmark. This repository is not a clinically validated diagnostic or treatment system.

## Executive summary

This project builds a complete early-warning pipeline around the PhysioNet/Computing in Cardiology Challenge 2019 ICU dataset. The workflow starts from hourly patient telemetry, converts the raw records into a DuckDB analytical warehouse, constructs causal temporal features, trains and evaluates multiple model families, and then examines whether the resulting predictor is useful beyond a single AUROC number.

The central modeling result is a **289-feature XGBoost model** that improves on a 15-feature raw snapshot baseline:

| Model | AUROC | AUPRC | Normalized utility |
|---|---:|---:|---:|
| Raw XGBoost baseline | 0.7572 | 0.0634 | 0.2504 |
| **Engineered XGBoost** | **0.7918** | **0.0821** | **0.3203** |
| Decision Tree, engineered | 0.7152 | 0.0540 | 0.2065 |
| Naive Bayes, engineered | 0.7093 | 0.0387 | 0.1431 |
| TabNet, full OOF run | 0.7597 | 0.0640 | 0.2687 |
| GRU-D, 20k-patient test | 0.7605 | 0.0661 | 0.2854 |
| **Causal Transformer, 20k-patient test** | **0.8230** | **0.1349** | **0.3809** |

The most important conclusion is therefore nuanced: **the engineered XGBoost model is a strong and interpretable tabular baseline, but the later Transformer experiment is the strongest reported model in this repository under its matched 20,000-patient test protocol.** The Transformer result should not be treated as directly interchangeable with the full-population XGBoost OOF number because the evaluation populations and protocols differ; the repository therefore reports paired DeLong comparisons on the shared 3,000-patient deep-learning test set.

The project also shows that the predictive signal is not only the instantaneous value of a vital sign. Temporal behavior, recent laboratory history, measurement frequency, and derived physiological relationships matter. For example, `Lactate_max_12h` is the strongest SHAP feature, while `Lactate_hours_since_last` ranks third, demonstrating that both **what was measured** and **when it was last measured** carry signal.

---

## What the system is trying to predict

The unit of prediction is a **patient-hour**. At each hour of an ICU stay, the system produces a probability associated with the dataset's `SepsisLabel` target.

The Challenge labeling framework is designed around an early-warning horizon: positive labels correspond to the period before documented sepsis onset. This repository therefore treats the task as **early sepsis warning**, not diagnosis at the bedside.

A crucial distinction is worth making:

- the model predicts the **dataset-defined target**;
- the reported lead time is measured relative to the **first `SepsisLabel == 1` hour**;
- therefore, a reported median lead time is a dataset-label lead time, not proof of how many hours before real-world clinical diagnosis a deployed system would warn.

---

## Dataset and cohort

The warehouse contains:

- **40,336 patients**
- **1,552,210 patient-hours**
- **2 hospital systems**
- approximately **1.8% positive patient-hours**
- **34 raw physiological variables:** 8 vital signs + 26 laboratory variables

### Raw variables

**Vitals:**

`HR`, `O2Sat`, `Temp`, `SBP`, `MAP`, `DBP`, `Resp`, `EtCO2`

**Laboratories:**

`BaseExcess`, `HCO3`, `FiO2`, `pH`, `PaCO2`, `SaO2`, `AST`, `BUN`, `Alkalinephos`, `Calcium`, `Chloride`, `Creatinine`, `Bilirubin_direct`, `Glucose`, `Lactate`, `Magnesium`, `Phosphate`, `Potassium`, `Bilirubin_total`, `TroponinI`, `Hct`, `Hgb`, `PTT`, `WBC`, `Fibrinogen`, `Platelets`

The data are sparse, particularly for laboratory variables. Rather than treating missingness as something to erase, the pipeline explicitly models it because the fact that a test was or was not recently measured can itself contain clinical information.

---

# 1. System architecture

```text
PhysioNet 2019 .psv files
          │
          ▼
01_etl_warehouse.py
          │
          ├── dim_hospital
          ├── dim_patient
          └── fact_vitals_hourly
                    │
                    ▼
02_feature_engineering.py
                    │
                    └── fact_features
                          │
       ┌──────────────────┼─────────────────────────────┐
       ▼                  ▼                             ▼
03 baseline         04 engineered                07 OLAP/export
XGBoost             XGBoost                      Power BI tables
       │                  │
       │         ┌────────┼──────────────┐
       │         ▼        ▼              ▼
       │       SHAP    lead time       fairness /
       │                alarm          hospital /
       │                fatigue        conformal
       │
       ├────────────── classical models / clustering / outliers / rules
       │
       └────────────── Phase 6 deep-learning challengers
                         ├── 16 GRU-D
                         ├── 17 TabNet
                         └── 18 causal Transformer

Shared evaluation utilities:
    utility_score.py
    delong.py
```

The architecture deliberately separates **data engineering**, **feature engineering**, **predictive modeling**, and **post-model analysis**. This makes it possible to answer different questions without confusing them:

1. Can raw measurements predict sepsis?
2. How much does temporal feature engineering add?
3. Which feature families carry the improvement?
4. Does the model generalize across hospitals?
5. Is performance similar across age and gender groups?
6. Can the model provide useful uncertainty information?
7. How early does it warn?
8. Does it reduce false alarms relative to a simple SIRS-style rule?
9. Can sequential neural architectures beat the tabular model?

---

# 2. Repository structure

```text
src/
├── 01_etl_warehouse.py
├── 02_feature_engineering.py
├── 03_baseline_model.py
├── 04_engineered_model.py
├── 05_explainability.py
├── 06_leadtime_alarm_fatigue.py
├── 07_olap_and_export.py
├── 08_association_rules.py
├── 09_hierarchical_clustering.py
├── 10_classical_classifiers.py
├── 11_dbscan_clustering.py
├── 12_outlier_analysis.py
├── 13_fairness_audit.py
├── 14_cross_hospital_generalization.py
├── 15_conformal_prediction.py
├── 16_grud_model.py
├── 17_tabnet_model_kaggle.py
├── 17_tabnet_model_kaggle (full).py
├── 18_transformer_model.py
├── clustering_phenotypes.py
├── delong.py
└── utility_score.py

outputs/
├── model result CSVs
├── OOF / held-out prediction Parquet files
├── run logs
├── figures/
└── powerbi_export/

warehouse/
└── sepsis.duckdb   # generated locally; not required in source control
```

---

# 3. Data engineering: `01_etl_warehouse.py`

The first stage converts the raw patient files into a DuckDB star schema.

### `dim_hospital`

One row per hospital system. The project treats each raw-data subdirectory as a hospital system and assigns a stable hospital identifier.

### `dim_patient`

Patient-level attributes and summary information, including age, gender, hospital, admission timing, maximum ICU length of stay, whether the patient was ever septic, and recorded-hour counts.

### `fact_vitals_hourly`

The central fact table has **one row per patient-hour**. It contains the 34 raw physiological variables plus identifiers, ICU hour, and `SepsisLabel`.

This grain is important: almost every downstream predictive result ultimately answers the question:

> **Given everything observable for this patient up to hour `t`, how strongly does the evidence support the dataset's future sepsis label?**

---

# 4. Temporal feature engineering: `02_feature_engineering.py`

The main engineered representation contains **289 predictive features**.

The key design principle is **causality**. Rolling calculations use windows ending at the current hour and never include future observations. Conceptually:

```sql
ROWS BETWEEN N PRECEDING AND CURRENT ROW
```

rather than a window that reaches into the future.

### Feature families

| Family | Features | Purpose |
|---|---:|---|
| Raw / forward-filled | 15 | Current usable snapshot |
| Rolling statistics | 180 | Local temporal state over 3h, 6h and 12h |
| Slopes / velocity | 60 | Direction and rate of change |
| Missingness | 30 | Measurement state and time since last observation |
| Clinical ratios | 4 | Compact physiological summaries |

### Rolling statistics

The rolling layer captures four complementary aspects of recent history:

- mean
- standard deviation
- minimum
- maximum
- slope / temporal trend where applicable

The windows are **3 hours, 6 hours and 12 hours**.

This is important because sepsis deterioration is not necessarily represented by one abnormal measurement. A sustained increase in respiratory rate, a falling pressure trajectory, or repeated abnormal laboratory values can be more informative than a single snapshot.

### Velocity / change features

The pipeline also computes first differences so that the model can distinguish:

```text
stable abnormal value
```

from

```text
rapidly worsening value
```

This is one reason the engineered model can outperform the raw snapshot baseline even when both use the same underlying physiological variables.

### Missingness features

Two forms of missingness are retained:

- whether a measurement is missing now;
- how many hours have passed since that variable was last observed.

This becomes one of the project's most interesting findings: measurement timing itself is predictive.

### Clinical composites

The feature set includes:

- **Shock index:** `HR / SBP`
- **Pulse pressure:** `SBP - DBP`
- **Partial SIRS score**
- **Partial qSOFA score**

These are intentionally partial because the available dataset does not provide every component needed for the full bedside scores.

---

# 5. Baseline XGBoost: `03_baseline_model.py`

The baseline intentionally uses only the **15 raw forward-filled snapshot features**.

It answers a clean control question:

> How well can a gradient-boosted tree perform without temporal feature engineering?

The evaluation uses **5-fold GroupKFold**, grouping on patient ID. This is critical because splitting individual hourly rows would allow hours from the same patient to appear in both training and validation data.

### Baseline result

| Metric | Result |
|---|---:|
| AUROC | **0.7572** |
| AUPRC | **0.0634** |
| Normalized utility | **0.2504** |
| Best threshold | 0.5408 |
| Features | 15 |

The baseline is already substantially better than random ranking, but the next stage tests whether temporal representation improves it.

---

# 6. Main model: `04_engineered_model.py`

The engineered model uses the full **289-feature predictive representation**.

### Main result

| Metric | Raw baseline | Engineered XGBoost | Change |
|---|---:|---:|---:|
| AUROC | 0.7572 | **0.7918** | +0.0347 |
| AUPRC | 0.0634 | **0.0821** | +0.0186 |
| Utility | 0.2504 | **0.3203** | +0.0699 |

The improvement is not merely numerical. The paired DeLong comparison reports:

- ΔAUC = **−0.03466** when calculated as baseline minus engineered;
- z = **−38.30**;
- p ≈ **0** at the reported precision.

The practical interpretation is straightforward: **temporal and missingness-aware representation adds substantial ranking and utility signal beyond the raw snapshot.**

---

# 7. Ablation study: where does the gain come from?

The feature-family ablation is especially useful because it prevents the 289-feature model from becoming a black box whose improvement is simply attributed to “more features.”

| Feature family | N | AUROC | AUPRC |
|---|---:|---:|---:|
| **Rolling statistics** | 180 | **0.7737** | **0.0695** |
| Raw forward-filled | 15 | 0.7572 | 0.0634 |
| Slopes / velocity | 60 | 0.7228 | 0.0557 |
| Missingness | 30 | 0.7163 | 0.0618 |
| Clinical ratios | 4 | 0.6561 | 0.0379 |

### Interpretation

The most important individual feature family is clearly **rolling statistics**. They provide the strongest single-family AUROC and AUPRC.

However, the full model is better than every single family because the information is complementary. A patient can simultaneously have:

- a high recent lactate;
- a worsening temperature trajectory;
- an abnormal SIRS-like state;
- an old laboratory measurement;
- and an adverse short-term trend.

The full model can combine these signals rather than choosing one representation.

A particularly important methodological point is that a low single-family AUROC does **not** mean the family is useless. The missingness family, for example, is much stronger when combined with the physiological values than when used alone.

---

# 8. Explainability: `05_explainability.py`

SHAP is used to inspect the fitted XGBoost model and identify which features contribute most strongly to predictions.

## Global SHAP ranking

Top features by mean absolute SHAP value:

| Rank | Feature |
|---:|---|
| 1 | `Lactate_max_12h` |
| 2 | `partial_sirs_score` |
| 3 | `Lactate_hours_since_last` |
| 4 | `Bilirubin_total_ffill` |
| 5 | `Temp_max_6h` |
| 6 | `Bilirubin_total_hours_since_last` |
| 7 | `Lactate_ffill` |
| 8 | `Temp_ffill` |
| 9 | `Resp_mean_12h` |
| 10 | `Temp_max_12h` |
| 11 | `BUN_ffill` |
| 12 | `shock_index` |
| 13 | `Temp_max_3h` |
| 14 | `Resp_min_12h` |
| 15 | `Potassium_mean_12h` |

### The important finding

`Lactate_hours_since_last` being third overall is one of the strongest pieces of evidence in the project that **measurement timing carries predictive information**.

This does not mean “a missing lactate causes sepsis.” It means the pattern of laboratory observation contains information associated with the target in this dataset. That can reflect clinical monitoring intensity and workflow as well as physiology.

### SHAP summary

![SHAP summary](outputs/figures/shap_summary.png)

### SHAP dependence plots

#### Shock index

![Shock index SHAP dependence](outputs/figures/shap_dependence_shock_index.png)

#### Partial SIRS score

![Partial SIRS SHAP dependence](outputs/figures/shap_dependence_partial_sirs_score.png)

#### HR 6-hour slope

![HR slope SHAP dependence](outputs/figures/shap_dependence_HR_slope_6h.png)

### Patient-level case explanation

![SHAP waterfall case](outputs/figures/shap_waterfall_case_positive.png)

These plots make the repository useful as more than a leaderboard: they show **which physiological and temporal signals the model is using**.

---

# 9. Early warning and alarm fatigue: `06_leadtime_alarm_fatigue.py`

This is arguably the most clinically meaningful analysis in the repository because AUROC alone does not answer the question:

> **How much warning does the system actually provide?**

Using the engineered model's operating threshold of **0.50**, the reported results are:

- **2,932 septic patients** in the lead-time analysis;
- **87.6%** were caught by at least one alert before or at the first positive label hour;
- median lead time among caught patients: **22.0 hours**;
- **68.6%** of caught patients were warned at least **6 hours** ahead.

### Alarm fatigue comparison

At approximately matched sensitivity:

| Rule | Sensitivity | Non-septic-hour false alarm rate |
|---|---:|---:|
| Naive SIRS ≥2 | 0.5061 | **0.2890** |
| Engineered model, matched sensitivity | 0.5072 | **0.1268** |
| Engineered model, operating point | **0.5998** | 0.1796 |

At matched sensitivity, the engineered model's reported non-septic-hour false-alarm rate is therefore less than half the SIRS-style comparator's:

**0.1268 vs 0.2890.**

That is a strong result for an early-warning system because an alarm that is rarely trusted is not operationally useful, even if its AUROC is high.

### Important interpretation of “22 hours”

The 22-hour median is **not** a claim that the model has been clinically proven to predict sepsis 22 hours before diagnosis.

It is the median difference between the first model alert and the first `SepsisLabel == 1` hour among caught patients. The label construction and retrospective nature of the dataset therefore matter.

---

# 10. Classical model comparison: `10_classical_classifiers.py`

The repository also evaluates simpler classifiers using the same raw-versus-engineered framing.

| Model | AUROC | AUPRC | Utility |
|---|---:|---:|---:|
| Decision Tree — engineered | 0.7152 | 0.0540 | 0.2065 |
| Naive Bayes — engineered | 0.7093 | 0.0387 | 0.1431 |
| Decision Tree — raw | 0.7056 | 0.0476 | 0.1793 |
| Naive Bayes — raw | 0.6957 | 0.0367 | 0.1410 |

This reinforces two points:

1. the engineered representation is useful across model families;
2. the XGBoost learner is considerably stronger than these simpler classical baselines on this task.

---

# 11. Deep learning phase

Phase 6 asks a different question:

> **Can sequence models learn the temporal and missingness structure more effectively than the engineered tree model?**

The three deep-learning challengers are deliberately not identical in input representation.

### GRU-D

GRU-D receives the raw 34 variables and explicitly models missingness through learned decay terms.

### TabNet

TabNet receives the engineered 289-feature table, making it a more direct architectural competitor to engineered XGBoost.

### Transformer

The causal Transformer receives the raw 34 variables plus:

- observation masks;
- causal time-since-observation deltas;
- positional information.

A causal attention mask ensures that an hour cannot attend to future hours.

---

# 12. GRU-D: `16_grud_model.py`

The reported GRU-D experiment uses:

- **20,000 patients**;
- 14,000 train / 3,000 validation / 3,000 test;
- 34 raw variables;
- maximum sequence length 336 hours;
- hidden size 64;
- early stopping on validation AUPRC.

### Result

| Metric | GRU-D |
|---|---:|
| AUROC | **0.7605** |
| AUPRC | **0.0661** |
| Utility | **0.2854** |
| Threshold | 0.50 |

On the same 114,486 paired test predictions used for the deep-model comparison, engineered XGBoost scores 0.7923 AUROC versus 0.7605 for GRU-D. DeLong's test reports a statistically significant difference.

### Learned decay interpretation

The model also records learned `gamma_x` values. Lower gamma means faster decay of stale information.

An interesting result is that the GRU-D decay ranking does **not** simply reproduce the SHAP ranking:

- Lactate is fastest-decay rank **34/34**;
- Bilirubin_total is rank **29/34**;
- yet their hand-engineered `hours_since_last` features rank **#3** and **#6** by SHAP.

This is scientifically useful rather than a failure: the engineered XGBoost and GRU-D are extracting temporal missingness information in different ways.

---

# 13. TabNet: `17_tabnet_model_kaggle (full).py`

The repository contains an older TabNet run and a newer full OOF run. The **newer `tabnet_results(full).csv` is treated as the authoritative TabNet result** in this README.

### Full OOF result

| Metric | TabNet |
|---|---:|
| AUROC | **0.7597** |
| AUPRC | **0.0640** |
| Utility | **0.2687** |
| Features | 289 |

Relative to engineered XGBoost:

- XGBoost AUROC: 0.7918
- TabNet AUROC: 0.7597
- absolute gap: approximately **0.0322 AUROC**

The TabNet feature-importance ranking has a modest but statistically detectable Spearman association with the XGBoost SHAP ranking:

- Spearman ρ = **0.1891**
- p = **0.00124**
- top-15 overlap = **4 / 15**

This suggests that the two models share some signal but organize feature importance differently.

---

# 14. Causal Transformer: `18_transformer_model.py`

The Transformer is the strongest model reported in the repository.

### Architecture

- 34 raw physiological variables;
- value + mask + delta input channels;
- input projection to 64 dimensions;
- sinusoidal positional encoding;
- 2 Transformer encoder layers;
- 4 attention heads;
- feed-forward dimension 128;
- dropout 0.2;
- causal attention mask;
- 73,601 trainable parameters;
- maximum sequence length 336 hours.

The model uses a patient-level 70/15/15 split over the same 20,000-patient stratified cohort used for GRU-D.

### Result

| Metric | Transformer |
|---|---:|
| **AUROC** | **0.8230** |
| **AUPRC** | **0.1349** |
| **Utility** | **0.3809** |
| Best threshold | 0.6225 |
| Parameters | 73,601 |

### Transformer vs XGBoost

On the shared 3,000-patient deep-learning test set and **114,486 paired hourly predictions**:

- XGBoost AUROC = **0.7923**
- Transformer AUROC = **0.8230**
- ΔAUC (XGB − Transformer) = **−0.0307**
- 95% CI = **[−0.0400, −0.0213]**
- p = **1.21 × 10⁻¹⁰**

The result is statistically significant under the paired DeLong comparison.

### Transformer vs GRU-D

On exactly the same patients and predictions:

- GRU-D AUROC = **0.7605**
- Transformer AUROC = **0.8230**
- ΔAUC = **−0.0625**
- 95% CI = **[−0.0725, −0.0524]**
- p effectively 0 at the reported precision.

### Attention behavior

The mean attention analysis shows strong recency bias:

| Relative lag | Mean attention weight |
|---:|---:|
| 0 h | **0.1231** |
| 1 h | 0.1002 |
| 2 h | 0.0847 |
| 3 h | 0.0740 |
| 4 h | 0.0638 |
| 5 h | 0.0557 |
| 6 h | 0.0493 |
| 7 h | 0.0450 |
| 8 h | 0.0417 |
| 9 h | 0.0396 |
| 10 h | 0.0377 |
| 11 h | 0.0357 |

The most-attended lag is **0 hours**, meaning the learned representation is strongly recency-biased while still being able to attend to earlier history.

---

# 15. How to read the model leaderboard correctly

A common mistake would be to put all seven model numbers into one table and declare the Transformer “better” solely because 0.8230 > 0.7918.

The repository contains **different evaluation contracts**:

### XGBoost / classical models

- full population;
- patient-grouped 5-fold OOF evaluation;
- approximately 40,336 patients;
- approximately 1.55M hourly predictions.

### GRU-D / Transformer

- fixed stratified **20,000-patient** cohort;
- 70/15/15 patient split;
- 14k train / 3k validation / 3k test;
- early stopping;
- same deep-learning test patients for direct GRU-D/Transformer comparison.

### TabNet full run

- newer full OOF run;
- same engineered feature table as XGBoost;
- treated separately from the older local/subsampled TabNet output.

Therefore the most defensible statements are:

1. **Engineered XGBoost clearly improves over its raw-snapshot baseline.**
2. **Engineered XGBoost is stronger than GRU-D and TabNet under their reported comparisons.**
3. **The causal Transformer is the strongest model in the reported deep-learning test experiment.**
4. **The Transformer-vs-XGBoost comparison is statistically supported on the shared 3,000-patient test cohort, but the full-population XGBoost OOF score and the Transformer test score should not be treated as identical evaluation populations.**

---

# 16. Cross-hospital generalization: `14_cross_hospital_generalization.py`

A major question for clinical ML is whether a model trained in one institution transfers to another.

The repository performs two directions:

| Train → Test | AUROC | AUPRC | Utility | AUROC drop |
|---|---:|---:|---:|---:|
| Hospital 1 → Hospital 2 | 0.7381 | 0.0545 | 0.1931 | 0.0538 |
| Hospital 2 → Hospital 1 | 0.7226 | 0.0609 | 0.2439 | 0.0693 |
| In-distribution reference | 0.7918 | 0.0821 | 0.3203 | — |

The degradation is substantial.

This is one of the most important results in the project because it prevents overclaiming. A model can look strong under pooled cross-validation and still lose performance when moved between hospital systems.

The implication is not that the model is unusable. It is that **external/site-specific validation is essential before deployment**.

---

# 17. Fairness audit: `13_fairness_audit.py`

The fairness analysis evaluates the engineered model without retraining separate models for each subgroup.

### AUROC gaps

| Axis | Highest | Lowest | Gap |
|---|---:|---:|---:|
| Hospital | 0.8084 | 0.7718 | **0.0366** |
| Age | 0.8015 | 0.7811 | **0.0204** |
| Gender | 0.7920 | 0.7915 | **0.0005** |

### Interpretation

Gender performance is extremely close in AUROC.

Age differences are small but non-zero.

The largest measured gap is across hospital systems, reinforcing the cross-hospital generalization result: **site effects appear more important than gender effects in this evaluation.**

These are performance audits, not claims of complete fairness. The available demographic variables and the retrospective dataset cannot establish fairness across every clinically relevant population.

![Fairness audit](outputs/figures/fairness_audit_engineered.png)

---

# 18. Conformal uncertainty: `15_conformal_prediction.py`

The conformal analysis asks a different question:

> When should the system make a confident statement, and when should it admit uncertainty?

At a nominal confidence level of **90%**, the reported results are:

| Measure | Result |
|---|---:|
| Empirical coverage | **89.74%** |
| Confident no-sepsis | 72.94% |
| Confident sepsis | 10.51% |
| Uncertain / both labels | 16.56% |
| Empty sets | 0% |
| Accuracy, confident hours | 87.71% |
| Accuracy, uncertain hours | 57.52% |
| Utility, full cohort | 0.3217 |
| Utility, confident-only | **0.3871** |

The confident-only utility being higher is suggestive of a useful selective-prediction strategy: **the model's strongest predictions are more operationally useful than forcing a hard answer on every hour.**

Coverage remains distribution-dependent and should not be interpreted as a universal guarantee after deployment in a new hospital.

---

# 19. Association rules: `08_association_rules.py`

The association-rule analysis converts selected physiological variables into abnormality flags and mines co-occurrence patterns using Apriori.

The run contains:

- 1,552,210 patient-hour rows;
- 374 rules overall;
- 28 rules with Sepsis as the consequent.

The strongest reported sepsis-consequent rule is:

```text
Resp_high + Temp_abnormal → Sepsis
```

with approximately:

- support = 0.00206
- confidence = 0.0509
- lift = **2.83**

Another strong rule is:

```text
HR_high + Temp_abnormal → Sepsis
```

with lift ≈ **2.82**.

### Important warning

Association rules are **descriptive**, not causal.

A lift above 1 means the combination occurs with the target more often than would be expected under the rule's baseline independence reference. It does not establish that one physiological abnormality causes sepsis or that the rule should be used as a clinical decision rule.

---

# 20. Clustering and phenotype discovery

The repository investigates whether patient trajectories naturally form distinct phenotypes.

## Hierarchical clustering

Ward hierarchical clustering uses an intentional **8,000-patient stratified sample**.

Reported silhouette:

- hierarchical Ward: **0.0949**
- matched K-means: **0.1565**
- full complete-case K-means: **0.1603**

The relatively low hierarchical silhouette indicates that the discovered clusters are not sharply separated in the chosen representation.

![Hierarchical dendrogram](outputs/figures/hierarchical_dendrogram.png)

![Cluster silhouette](outputs/figures/cluster_silhouette.png)

The project should therefore treat these clusters as **exploratory phenotype structure**, not as clinically established patient subtypes.

## DBSCAN

The DBSCAN run uses:

- 32,465 patients;
- epsilon ≈ 3.5512;
- minimum samples = 10;
- 1 dominant cluster;
- 662 noise points;
- 2.04% noise.

![DBSCAN k-distance elbow](outputs/figures/dbscan_kdistance_elbow.png)

Interestingly, the DBSCAN noise group has a higher observed sepsis rate than the dominant cluster. This suggests that some physiologically unusual patients may occupy a sparse region of feature space, but again this is an exploratory association rather than a diagnostic classifier.

---

# 21. Outlier analysis: `12_outlier_analysis.py`

The project compares IQR-based physiological outlier flags with Isolation Forest.

| Method | % flagged | Sepsis-rate lift |
|---|---:|---:|
| IQR Temp | 1.54% | **3.86×** |
| IQR 2+ vitals | 1.21% | **3.27×** |
| Isolation Forest | 2.00% | **3.21×** |
| IQR Resp | 3.78% | 2.51× |
| IQR HR | 1.12% | 2.46× |
| IQR WBC | 2.63% | 2.33× |
| IQR any vital | 10.96% | 2.30× |
| IQR Lactate | 2.14% | 1.86× |
| IQR SBP | 1.14% | 1.20× |

Temperature outliers have the largest reported sepsis-rate lift.

Again, an outlier is not synonymous with sepsis. The analysis asks whether unusual observations are enriched for the target, not whether they diagnose it.

![Outlier sepsis lift](outputs/figures/outlier_sepsis_lift.png)

---

# 22. OLAP and Power BI layer: `07_olap_and_export.py`

The project is not only a machine-learning experiment. It also demonstrates an analytical warehouse workflow.

The OLAP stage demonstrates:

- hourly → daily roll-up;
- daily → stay roll-up;
- hospital → patient → hour drill-down;
- septic slice;
- combined hospital/sepsis/lactate dice.

Reported outputs include:

- **105,665** daily roll-up rows;
- **40,336** stay-level rows;
- **27,916** rows in the septic slice;
- **2,438** rows in the example hospital + septic + lactate > 2 dice;
- Power BI fact export with **1,552,210 rows × 15 columns**.

The exported tables are designed to support downstream dashboards without making the dashboard layer responsible for rebuilding the clinical feature engineering pipeline.

See `POWERBI_HANDOFF.md` for the dashboard-side workflow.

---

# 23. Statistical evaluation methodology

## AUROC

AUROC measures ranking ability across thresholds. It is useful for comparing discrimination but can look deceptively reassuring in severely imbalanced problems.

## AUPRC

AUPRC is emphasized because the positive class is rare. It focuses attention on precision-recall behavior for the clinically important minority class.

## Normalized utility

`utility_score.py` implements a vectorized version of the official PhysioNet/CinC 2019 utility framework. The implementation is documented as verified against the Challenge reference implementation.

The configured timing parameters are:

```text
dt_early   = -12 hours
dt_optimal = -6 hours
dt_late    = +3 hours
```

with:

```text
max_u_tp = +1
min_u_fn = -2
u_fp     = -0.05
u_tn     = 0
```

The normalized score is scaled so that:

- a best-possible prediction sequence is 1;
- doing nothing is 0;
- a model can score below 0 if it is worse than inaction.

This metric rewards timely warnings and penalizes false alarms, which is why it is particularly relevant to early-warning systems.

The utility implementation explicitly evaluates each patient's **chronological sequence**, rather than treating rows as independent classification events.

## DeLong's test

`delong.py` implements the fast DeLong method for comparing **correlated ROC AUCs**. This is appropriate when two models produce predictions for the same held-out rows.

The Transformer comparisons use this paired structure, which is why the reported significance tests are more informative than simply subtracting two independently estimated AUROCs.

---

# 24. Leakage and temporal discipline

Leakage prevention is one of the strongest parts of the pipeline design.

### Patient-level grouping

For the XGBoost and classical models, `GroupKFold` uses `patient_id`. This prevents different hours from the same patient appearing in both training and validation folds.

### Causal windows

The temporal feature engine only uses observations available at or before the prediction hour.

### Training-only normalization

The Transformer standardizes variables using training-set statistics and uses the training mean for imputation after standardization.

### Patient-level deep-learning split

GRU-D and Transformer split patients rather than hourly rows into train/validation/test groups.

### What is still important to validate

No retrospective pipeline can completely remove all sources of dataset construction bias. The `SepsisLabel` target itself is generated according to the Challenge's labeling process, and clinical workflow can influence what measurements are available.

That is why the repository's claims should remain about **benchmark performance on this dataset**, not prospective clinical efficacy.

---

# 25. The central scientific story

The project can be summarized as a sequence of increasingly demanding questions.

### Question 1 — Are raw physiological snapshots enough?

**Answer:** useful, but limited.

Raw XGBoost reaches **0.7572 AUROC**.

### Question 2 — Does temporal context matter?

**Answer:** yes.

Adding rolling statistics, trends, missingness, and clinical composites raises XGBoost to **0.7918 AUROC** and **0.0821 AUPRC**.

### Question 3 — Which temporal representation matters most?

**Answer:** rolling statistics are the strongest single family.

### Question 4 — Does the model use clinically meaningful information?

**Answer:** the SHAP results show strong contributions from lactate, SIRS-like state, temperature, respiratory behavior, bilirubin, BUN, potassium, and shock index, along with measurement-recency features.

### Question 5 — Does this translate into earlier warning?

**Answer:** in the retrospective label-based analysis, 87.6% of septic patients were caught before or at the first positive label, with a median lead time of 22 hours among caught patients.

### Question 6 — Can it reduce alarm burden?

**Answer:** at matched sensitivity, the engineered model has a substantially lower reported non-septic-hour false-alarm rate than the naive SIRS comparator.

### Question 7 — Does it generalize between hospitals?

**Answer:** only partially. AUROC drops by about **0.054–0.069** in the two cross-hospital directions.

### Question 8 — Is performance uniform across groups?

**Answer:** gender performance is almost identical; age and especially hospital show larger differences.

### Question 9 — Does uncertainty information help?

**Answer:** conformal prediction identifies an uncertain subset, and the confident-only subset has higher reported utility.

### Question 10 — Can deep sequence models beat the tree?

**Answer:** GRU-D and TabNet do not beat engineered XGBoost in the reported experiments, but the **causal Transformer does**, reaching **0.8230 AUROC / 0.1349 AUPRC / 0.3809 utility** on its 3,000-patient held-out test set.

---

# 26. Main strengths

1. **End-to-end architecture:** data warehouse → features → models → evaluation → interpretability → deployment-oriented analyses.
2. **Patient-grouped evaluation:** avoids the most obvious repeated-measures leakage failure.
3. **Causal temporal features:** rolling windows do not look forward.
4. **Multiple metrics:** AUROC, AUPRC and a clinical utility metric are all reported.
5. **Statistical model comparison:** paired DeLong testing is used where predictions are directly comparable.
6. **Interpretability:** SHAP connects model performance to specific variables and temporal patterns.
7. **Early-warning analysis:** the project measures lead time rather than stopping at discrimination.
8. **Alarm-fatigue analysis:** compares false-alarm burden with a simple SIRS-style rule.
9. **External-site stress test:** cross-hospital validation exposes meaningful distribution shift.
10. **Fairness audit:** age, gender and hospital subgroup performance are explicitly measured.
11. **Uncertainty analysis:** conformal prediction adds selective-confidence information.
12. **Model diversity:** tree, classical, TabNet, recurrent and Transformer architectures are compared.
13. **Analytical warehouse layer:** OLAP and Power BI exports connect ML to practical analytics workflows.

---

# 27. Limitations and what should happen next

### 1. Retrospective benchmark

The project is evaluated on historical ICU data. Prospective workflow validation is still required.

### 2. Target-definition dependence

All predictive and lead-time results depend on the Challenge's `SepsisLabel` construction. A real deployment would need a carefully specified clinical target and prospective endpoint.

### 3. Cross-hospital degradation

The cross-hospital experiments show meaningful performance loss. This is the clearest evidence that external validation is necessary.

### 4. Class imbalance

Sepsis is rare at the patient-hour level. AUPRC and utility should therefore remain central when interpreting performance.

### 5. Repeated hourly observations

Hourly rows from a patient are temporally correlated. Patient grouping prevents direct train/test patient leakage, but it does not make the observations statistically independent.

### 6. Deep-learning evaluation is not identical to XGBoost evaluation

GRU-D and Transformer use the 20,000-patient fixed split, whereas the main XGBoost model uses full-population GroupKFold OOF evaluation. The README explicitly keeps these protocols separate.

### 7. Clustering is exploratory

The low silhouette values indicate weakly separated phenotypes. These clusters should not be interpreted as validated clinical subtypes.

### 8. Association rules are not causal

High lift indicates association, not intervention effect or mechanism.

### 9. Fairness scope is limited by available variables

The audit covers the demographic/site variables represented in the data. It is not a complete fairness certification.

### 10. Calibration and prospective thresholding

The selected thresholds are useful for the reported retrospective experiments, but a real deployment would require calibration, workflow simulation, alert-frequency constraints, and prospective threshold selection.

### 11. Transformer result needs external validation

The Transformer result is promising, but one held-out test split is not enough to establish general clinical superiority. Replication across temporal splits, institutions, and external datasets would be the natural next step.

---

# 28. Recommended next research steps

If this project were being taken from a strong capstone/research prototype toward a publication-grade system, the next priorities would be:

1. **External validation on another ICU dataset.**
2. **Temporal holdout validation** rather than only random patient splits.
3. **Repeated seeds / repeated patient splits** for the deep models.
4. **Calibration curves and Brier score.**
5. **Patient-level sensitivity and specificity at operational alert rates.**
6. **Time-dependent precision/recall and lead-time distributions**, not only median lead time.
7. **Decision-curve analysis / net benefit.**
8. **Ablation of missingness alone versus value + missingness jointly.**
9. **Transformer ablations:** values only vs values + mask vs values + mask + delta.
10. **Transformer attention stability across seeds and patients.**
11. **Hospital-specific calibration and threshold analysis.**
12. **Prospective-style alarm simulation**, including repeated-alert suppression and cooldown periods.
13. **More rigorous uncertainty evaluation under distribution shift.**

---

# 29. Reproducibility

The intended execution order is approximately:

```bash
python src/01_etl_warehouse.py
python src/02_feature_engineering.py
python src/03_baseline_model.py
python src/04_engineered_model.py
python src/05_explainability.py
python src/06_leadtime_alarm_fatigue.py
python src/07_olap_and_export.py
python src/08_association_rules.py
python src/09_hierarchical_clustering.py
python src/10_classical_classifiers.py
python src/11_dbscan_clustering.py
python src/12_outlier_analysis.py
python src/13_fairness_audit.py
python src/14_cross_hospital_generalization.py
python src/15_conformal_prediction.py
python src/16_grud_model.py
python "src/17_tabnet_model_kaggle (full).py"
python src/18_transformer_model.py
```

The deep-learning scripts require PyTorch in addition to the core scientific Python stack.

The repository's generated `outputs/` directory is retained so that results can be inspected without retraining every model.

---

# 30. Figures and generated evidence

The repository includes the major generated figures used to interpret the analysis:

- SHAP global summary
- SHAP dependence for shock index
- SHAP dependence for partial SIRS
- SHAP dependence for HR slope
- SHAP waterfall case study
- hierarchical dendrogram
- clustering silhouette
- DBSCAN k-distance elbow
- outlier/sepsis lift
- fairness audit

These are intentionally embedded here so the README is also a compact visual research report.

---

# 31. Bottom line

This is a **good and substantial research prototype**, not merely a model-training script.

The strongest parts are the causal temporal feature engineering, patient-level evaluation discipline, explicit ablations, clinical-utility scoring, early-warning analysis, cross-hospital stress testing, SHAP analysis, and the progression from XGBoost to GRU-D, TabNet, and a causal Transformer.

The results tell a coherent story:

> **Temporal context matters. Measurement timing matters. Rolling physiological behavior is more informative than a single snapshot. The engineered XGBoost model is a strong tabular baseline. Cross-hospital transfer is a real weakness. And in the later matched deep-learning experiment, the causal Transformer is the strongest reported architecture.**

The most important caveat is equally clear:

> **These results demonstrate predictive performance on the PhysioNet 2019 benchmark; they do not by themselves establish prospective clinical effectiveness.**

That distinction is what keeps the project scientifically credible.

---

## References

- PhysioNet / Computing in Cardiology Challenge 2019 dataset and evaluation framework.
- Che et al., *Recurrent Neural Networks for Multivariate Time Series with Missing Values*, Scientific Reports (GRU-D).
- Vaswani et al., *Attention Is All You Need* (Transformer architecture).
- Sun & Xu, *Fast Implementation of DeLong's Algorithm for Comparing the Areas Under Correlated Receiver Operating Characteristic Curves* (DeLong comparison).

## License / data note

The repository does not redistribute the underlying clinical dataset. Users should obtain the PhysioNet Challenge 2019 data through its official distribution and comply with the dataset's applicable terms.
