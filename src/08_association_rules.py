# %% [markdown]
# # Phase 8 — Association rule mining (Apriori)
#
# Continuous ICU vitals aren't natural Apriori input, so this bins each
# vital into a clinical "abnormal / normal" flag (using standard clinical
# thresholds, the same spirit as SIRS/qSOFA criteria already used in
# feature engineering) and mines which COMBINATIONS of abnormal flags
# co-occur, and how strongly they associate with SepsisLabel=1.
#
# This is intentionally a separate, interpretable analysis -- not a
# replacement for the XGBoost model. It answers a different question:
# "which simple rule-based vital combinations flag sepsis-associated
# hours," which is directly comparable to the naive SIRS>=2 rule already
# used as a baseline in 06_leadtime_alarm_fatigue.py.

# %%
import duckdb
import pandas as pd
from pathlib import Path
from mlxtend.frequent_patterns import apriori, association_rules

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "warehouse" / "sepsis.duckdb"
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# %% [markdown]
# ## Load raw vitals + label (small, not the full feature table)

# %%
con = duckdb.connect(str(DB_PATH), read_only=True)
df = con.execute("""
    SELECT
        patient_id, hour, SepsisLabel,
        HR_ffill AS HR, Temp_ffill AS Temp, Resp_ffill AS Resp,
        SBP_ffill AS SBP, WBC_ffill AS WBC, Lactate_ffill AS Lactate
    FROM fact_features
""").df()
con.close()
print(f"Loaded {len(df):,} rows for binning")

# %% [markdown]
# ## Bin each vital into a clinical abnormal-flag (SIRS/qSOFA-style thresholds)
# These thresholds match standard clinical cutoffs, not arbitrary quantiles,
# so the resulting rules stay clinically interpretable.

# %%
flags = pd.DataFrame({
    "HR_high":       df["HR"] > 90,
    "Temp_abnormal": (df["Temp"] > 38.0) | (df["Temp"] < 36.0),
    "Resp_high":     df["Resp"] > 20,
    "SBP_low":       df["SBP"] < 100,
    "WBC_abnormal":  (df["WBC"] > 12) | (df["WBC"] < 4),
    "Lactate_high":  df["Lactate"] > 2.0,
    "Sepsis":        df["SepsisLabel"] == 1,
})
flags = flags.fillna(False)
print("Flag prevalence:\n", flags.mean().round(4))

# %% [markdown]
# ## Run Apriori on the binary flag table
# min_support kept low since Sepsis=1 is rare (~1.8%) -- otherwise sepsis-
# containing itemsets get pruned before they're ever considered.

# %%
frequent_itemsets = apriori(flags, min_support=0.002, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# %% [markdown]
# ## Focus on rules that predict Sepsis specifically
# Sort by lift: how much more likely Sepsis is given the antecedent flags,
# versus its base rate alone.

# %%
sepsis_rules = rules[rules["consequents"].apply(lambda x: "Sepsis" in x)]
sepsis_rules = sepsis_rules.sort_values("lift", ascending=False)

print(f"\nTotal rules found: {len(rules)}")
print(f"Rules with Sepsis as consequent: {len(sepsis_rules)}\n")
print("Top 20 rules by lift (antecedent -> Sepsis):")
display_cols = ["antecedents", "consequents", "support", "confidence", "lift"]
print(sepsis_rules[display_cols].head(20).to_string(index=False))

# Specifically call out multi-flag (compound) rules, since these are the
# ones directly comparable to a SIRS>=2-style combined-criteria rule.
compound_rules = sepsis_rules[sepsis_rules["antecedents"].apply(lambda x: len(x) >= 2)]
print(f"\nCompound (2+ flag) rules with Sepsis as consequent: {len(compound_rules)}")
if len(compound_rules) > 0:
    print(compound_rules[display_cols].to_string(index=False))

# %%
rules.to_csv(OUT_DIR / "association_rules_all.csv", index=False)
sepsis_rules.to_csv(OUT_DIR / "association_rules_sepsis.csv", index=False)
print(f"\nSaved: association_rules_all.csv, association_rules_sepsis.csv")
