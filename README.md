# Early Sepsis Prediction — DW-Mining Capstone

Predicts sepsis onset 6 hours ahead from hourly ICU vitals/labs (PhysioNet/CinC
Challenge 2019 data), with feature engineering pushed into a DuckDB warehouse
(star schema + SQL window functions), a baseline-vs-engineered model
comparison, the **official Challenge utility score**, SHAP explainability,
lead-time analysis, and alarm-fatigue quantification.

## 1. Environment setup (VSCode)

```bash
cd sepsis_capstone
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

In VSCode: `Ctrl/Cmd+Shift+P` → "Python: Select Interpreter" → pick `.venv`.

## 2. How the code is organized

Every file in `src/` is written with `# %%` cell markers. Open any of them
directly in VSCode with the Python extension installed — it renders as an
interactive Jupyter notebook (no `.ipynb` needed) with "Run Cell" / "Run
Below" affordances above each `# %%` block. Run them **in this order**:

| Order | File | Phase | What it does |
|---|---|---|---|
| 1 | `src/01_etl_warehouse.py` | 2 | Loads raw `.psv` → DuckDB star schema (`dim_patient`, `dim_hospital`, `fact_vitals_hourly`) |
| 2 | `src/02_feature_engineering.py` | 2/4 | SQL window-function features computed **inside DuckDB** (rolling stats, slopes, clinical ratios, missingness) |
| 3 | `src/03_baseline_model.py` | 3 | Raw-snapshot-only baseline model, patient-grouped CV |
| 4 | `src/04_engineered_model.py` | 5 | Full engineered-feature model, same split, statistical comparison |
| 5 | `src/05_explainability.py` | 6 | SHAP + feature-family ablation |
| 6 | `src/06_leadtime_alarm_fatigue.py` | 7 | Lead-time distribution + false-alarm-rate comparison vs a naive threshold rule |
| — | `src/utility_score.py` | — | Reusable module: exact reimplementation of the official Challenge utility function |
| — | `src/clustering_phenotypes.py` | optional | k-means on trajectory summary stats → deterioration phenotypes |

Each script reads/writes from `warehouse/sepsis.duckdb` and drops artifacts
(models, plots, metrics tables) into `outputs/`.

## 3. Grading-relevant design choices (put these in your report)

- **No row-level leakage**: every split uses `GroupKFold` / `GroupShuffleSplit`
  on `patient_id`. A patient's hours never appear in both train and test.
- **No temporal leakage**: every rolling/derivative feature is computed with
  `ROWS BETWEEN N PRECEDING AND CURRENT ROW` — never `FOLLOWING`.
- **Label-construction leakage, flagged not hidden**: `SepsisLabel` is derived
  retrospectively from Sepsis-3 criteria that reference some of the same lab
  values used as features (see `docs/limitations.md`, generate this yourself
  from Phase 9 of the plan). Report it explicitly.
- **Utility score, not just AUC**: `src/utility_score.py` is a line-for-line
  reproduction of PhysioNet's official `compute_prediction_utility`
  (verified against `physionetchallenges/evaluation-2019` on GitHub), so your
  numbers are directly comparable to published Challenge leaderboard results.

## 4. Quick start once data is in place

```bash
python src/01_etl_warehouse.py
python src/02_feature_engineering.py
python src/03_baseline_model.py
python src/04_engineered_model.py
python src/05_explainability.py
python src/06_leadtime_alarm_fatigue.py
```

Or open each in VSCode and run cell-by-cell to inspect intermediate output.
