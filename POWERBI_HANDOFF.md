# Power BI Handoff — Sepsis Early Warning Project (DWM/OLAP portion)

This is context + step-by-step instructions for the Power BI part of a sepsis-prediction capstone project. You (the reader) are building the **OLAP/Power BI demonstration** on top of data someone else already prepared — you don't need to touch any Python or DuckDB code, just import three CSV files and build some visuals in Power BI Desktop.

---

## 1. The context you need

This is a capstone project predicting sepsis 6 hours ahead from ICU vitals, using the real PhysioNet/CinC Challenge 2019 dataset (40,336 patients, ~1.55M hourly readings). The modeling side (XGBoost, feature engineering, SHAP, etc.) is already done. Separately, the course syllabus (DWM — "Data Warehouse and Mining") specifically requires demonstrating **OLAP cube operations** — roll-up, drill-down, slice, and dice — and the professor wants this shown **in Power BI specifically**, not just in SQL. That's your job.

The underlying data lives in a **DuckDB data warehouse** organized as a **star schema** — one central fact table surrounded by dimension tables:

```
        dim_hospital
             |
             |
dim_patient — fact_vitals_olap
```

- **`fact_vitals_olap`** — the fact table. One row per patient, per hour. This is the "measurements" table — every vital sign reading, at every hour, for every patient.
- **`dim_patient`** — the patient dimension. One row per patient (demographics, whether they ever became septic, etc.)
- **`dim_hospital`** — the hospital dimension. One row per hospital (there are only 2 in this dataset).

You don't need to understand DuckDB or SQL at all — someone already ran a script that exported this schema to three plain CSV files, specifically so you can import them into Power BI without touching any code.

---

## 2. Where to get the files

Three CSVs, all inside the repo (or shared Drive folder) at `outputs/powerbi_export/`:

| File | Rows | What it is |
|---|---|---|
| `dim_patient.csv` | 40,336 | Patient dimension |
| `dim_hospital.csv` | 2 | Hospital dimension |
| `fact_vitals_olap.csv` | 1,552,210 | Fact table (this one is large — see the performance note in §6) |

If these aren't in the GitHub repo directly (they're large, so they may be `.gitignore`d and shared via Google Drive instead), ask for the Drive link — that's where the big files live.

**Column reference**, so you know what you're looking at once it's imported:

`dim_patient.csv`: `patient_id`, `hospital_id`, `age`, `gender`, `hosp_admit_time`, `max_iculos`, `is_ever_septic`, `n_hours_recorded`

`dim_hospital.csv`: `hospital_id`, `hospital_name`

`fact_vitals_olap.csv`: `patient_id`, `hospital_id`, `hour`, `ICULOS`, `SepsisLabel`, `HR`, `O2Sat`, `Temp`, `SBP`, `MAP`, `Resp`, `Lactate`, `shock_index`, `partial_sirs_score`, `day_bucket`

The `day_bucket` column is important for you specifically — it's a pre-computed "which day of the ICU stay is this hour in" column, and it's what lets you build a clean `hospital → day → hour` drill-down hierarchy without having to calculate anything yourself in Power BI.

---

## 3. Step-by-step: importing into Power BI Desktop

1. Open **Power BI Desktop**.
2. **Home tab → Get Data → Text/CSV**.
3. Browse to `outputs/powerbi_export/dim_patient.csv`, select it, click **Load** (not "Transform Data" — you don't need to edit anything, just load it as-is).
4. Repeat **Get Data → Text/CSV** for `dim_hospital.csv`. Load it.
5. Repeat once more for `fact_vitals_olap.csv`. **This file is ~100MB and 1.55 million rows — the load will take noticeably longer than the other two.** Let it finish; don't cancel it.
6. Once all three are loaded, you'll see all three tables listed in the **Fields** pane on the right side of the screen (`dim_patient`, `dim_hospital`, `fact_vitals_olap`).

**Data type check (do this before moving on):** click into each table in the Fields pane and confirm:
- `patient_id` and `hospital_name` are **Text**
- `hospital_id`, `hour`, `ICULOS`, `SepsisLabel`, `day_bucket` are **Whole Number**
- `HR`, `O2Sat`, `Temp`, `SBP`, `MAP`, `Resp`, `Lactate`, `shock_index`, `partial_sirs_score`, `age` are **Decimal Number**

Power BI usually infers these correctly automatically, but it's worth a 30-second glance — if `hospital_id` accidentally imports as Text in one table and Whole Number in another, the relationship in the next step will silently fail to match rows.

---

## 4. Step-by-step: building the star-schema relationships

This is the actual "data warehouse" part the professor wants to see — proof that you understand fact/dimension relationships, not just three unconnected tables.

1. Click **Model view** in the left-hand navigation strip (icon looks like three connected boxes).
2. You'll see all three tables as boxes. Drag them apart so they don't overlap.
3. **Create relationship 1:** click and drag from `fact_vitals_olap.patient_id` onto `dim_patient.patient_id`. A relationship line appears. Double-click it to check the settings — it should auto-detect as **Many-to-one** (many fact rows per one patient) with a **single direction** cross-filter.
4. **Create relationship 2:** same thing, drag from `fact_vitals_olap.hospital_id` onto `dim_hospital.hospital_id`. Again should be many-to-one.
5. You should now see a clean star shape: `dim_patient` and `dim_hospital` both connected to `fact_vitals_olap`, with no line directly between `dim_patient` and `dim_hospital` (that's correct — dimensions don't connect to each other in a star schema, they only connect through the fact table).

Take a screenshot of this Model view — it's good visual proof of the star schema for your report/submission.

---

## 5. Step-by-step: demonstrating each OLAP operation

Switch to **Report view** (left nav, the bar-chart icon). You'll build one visual per operation.

### Roll-up (fine-grained → coarse-grained aggregation)

"Roll-up" means aggregating detailed data up to a summary level — e.g., hourly readings summarized into daily, or daily into whole-stay.

1. Insert a **Matrix** visual (Visualizations pane → Matrix icon).
2. Drag `dim_hospital.hospital_name` into **Rows**.
3. Drag `fact_vitals_olap.day_bucket` into **Rows**, below `hospital_name` (this creates a hierarchy: hospital → day).
4. Drag `fact_vitals_olap.HR` into **Values**, set aggregation to **Average**.
5. Do the same for `Lactate` (Average).

You now have a table that rolls hourly data up to daily averages, per hospital. This is the roll-up operation.

### Drill-down (reverse of roll-up — go from summary into detail)

This uses the **same Matrix visual** you just built — drill-down isn't a separate visual, it's an interaction on the hierarchy you already created.

1. In the Matrix visual, click the little **down-arrow / expand icon** in the top-left corner of the visual (or right-click a hospital row → **Drill down**).
2. Clicking into a hospital row expands it to show the `day_bucket` breakdown underneath — that's one level of drill-down.
3. To go a level deeper (down to individual hours), you'd add `fact_vitals_olap.hour` as a third row field below `day_bucket`, extending the hierarchy to hospital → day → hour.

Screenshot both the collapsed (rolled-up) and expanded (drilled-down) states — that's your before/after proof for the report.

### Slice (filter down to one value along one dimension)

"Slice" means cutting the cube down using a filter on a single attribute — e.g., "show me only septic patient-hours."

1. Insert a **Slicer** visual.
2. Drag `fact_vitals_olap.SepsisLabel` into the slicer's field.
3. Click the slicer and select **1** (septic hours only). Every other visual on the page (including your Matrix from before) will now filter down to only septic patient-hours — that's the slice operation in action.

### Dice (filter along multiple dimensions simultaneously)

"Dice" is the same idea as slice but on 2+ dimensions at once — e.g., "only Hospital 1, only septic, only high lactate."

1. Insert a second **Matrix** visual.
2. Drag `dim_hospital.hospital_name` into **Rows**.
3. Drag `fact_vitals_olap.SepsisLabel` into **Columns**.
4. Drag `fact_vitals_olap.shock_index` into **Values**, aggregation **Average**.
5. Add a **Slicer** for `Lactate` (or use a filter card) constrained to `Lactate > 2.0`.

Now you have a table sliced by hospital (rows), diced further by sepsis status (columns), and filtered on a third dimension (lactate threshold) — that's the dice operation, since you're cutting along multiple axes at once rather than just one.

---

## 6. Performance notes (the fact table is genuinely large)

`fact_vitals_olap.csv` is 1.55 million rows. A few things that help if Power BI feels sluggish:

- **File → Options and settings → Data Load**, and turn off **"Auto date/time"** — Power BI creates hidden date hierarchy tables for every date-like column by default, which adds unnecessary overhead here since you don't have a real datetime column, just integer hour offsets.
- Avoid dragging `fact_vitals_olap` fields directly into a Table/Matrix visual without any aggregation — showing 1.5M raw rows in a table will freeze the UI. Always aggregate (Average, Sum, Count) rather than showing raw row-level detail in a visual.
- If your machine struggles with the full fact table, it's fine to mention in your report that a **sample** was used for interactive testing while building visuals, as long as the final submitted file uses the full table.

---

## 7. Cross-checking your work

The person who built the pipeline already ran the equivalent SQL queries directly in DuckDB and saved the expected results, so you can check your Power BI numbers against these files (also in `outputs/`) to confirm you built things correctly:

| Operation | Reference file | Expected result |
|---|---|---|
| Roll-up (hourly → daily) | `outputs/olap_rollup_daily.csv` | 105,665 rows |
| Roll-up (daily → whole stay) | `outputs/olap_rollup_stay.csv` | 40,336 rows |
| Drill-down (hourly detail example) | `outputs/olap_drilldown_hourly_example.csv` | Sample patient's hour-by-hour readings |
| Slice (`SepsisLabel = 1`) | `outputs/olap_slice_septic.csv` | 27,916 rows |
| Dice (hospital=1, septic, Lactate>2.0) | `outputs/olap_dice_example.csv` | 2,438 rows |

If your Power BI slicer set to `SepsisLabel = 1` shows a row count matching 27,916 total fact rows, for example, you know the slice is working correctly.

---

## 8. What to submit / send back

- A `.pbix` file (Power BI's native save format) with the Model view relationships and all four visuals (roll-up, drill-down, slice, dice) built.
- Screenshots of: the Model view star schema, the collapsed and expanded Matrix (roll-up/drill-down), the slicer in action, and the dice table.
- Send the `.pbix` file back so it can be added to the repo/report — `.pbix` files are binary and shouldn't go through GitHub directly if they're large; same Google Drive folder as the other big files works fine.

If anything about the data itself is unclear (what a column means, why a number looks a certain way), the full project README in the repo root covers the entire pipeline in depth — worth a skim for context, though you shouldn't need to touch any of the Python/DuckDB side to do your part.
