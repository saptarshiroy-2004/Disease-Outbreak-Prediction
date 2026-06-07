"""
Disease Outbreak Prediction
Step 2: Clean & Explore Raw Data
Handles: who_cholera.csv + who_dengue.csv
Run: python src/data/clean_data.py
"""

import pandas as pd
import numpy as np
import os

RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# SECTION 1 — CLEAN CHOLERA DATA
# ─────────────────────────────────────────────

print("\n" + "="*60)
print("  CLEANING CHOLERA DATA (WHO GHO API)")
print("="*60)

cholera_raw = pd.read_csv(f"{RAW_DIR}/who_cholera.csv")

print(f"\nRaw shape      : {cholera_raw.shape}")
print(f"Columns        : {list(cholera_raw.columns)}")
print(f"Missing values : {cholera_raw.isnull().sum().sum()}")

# Keep only the columns we actually need
cholera = cholera_raw[[
    "SpatialDim",        # Country ISO3 code  e.g. IND, BGD
    "ParentLocation",    # WHO Region          e.g. South-East Asia
    "TimeDim",           # Year
    "NumericValue"       # Case count  ← our target
]].copy()

# Rename for clarity
cholera.columns = ["country_code", "region", "year", "cholera_cases"]

# Drop rows with no case count
cholera.dropna(subset=["cholera_cases"], inplace=True)

# Remove zero / negative case counts
cholera = cholera[cholera["cholera_cases"] > 0]

# Convert types
cholera["year"]          = cholera["year"].astype(int)
cholera["cholera_cases"] = cholera["cholera_cases"].astype(float)

# Keep years 2000 onward (aligns with Dengue data)
cholera = cholera[cholera["year"] >= 2000]

# Remove duplicates (same country + year)
cholera = cholera.drop_duplicates(subset=["country_code", "year"])

# Sort
cholera = cholera.sort_values(["country_code", "year"]).reset_index(drop=True)

print(f"\nCleaned shape  : {cholera.shape}")
print(f"Year range     : {cholera['year'].min()} – {cholera['year'].max()}")
print(f"Countries      : {cholera['country_code'].nunique()}")
print(f"Missing values : {cholera.isnull().sum().sum()}")
print(f"\nSample rows:")
print(cholera.head(8).to_string(index=False))

cholera.to_csv(f"{PROCESSED_DIR}/cholera_cleaned.csv", index=False)
print(f"\n✅ Saved → data/processed/cholera_cleaned.csv")


# ─────────────────────────────────────────────
# SECTION 2 — CLEAN DENGUE DATA (OpenDengue)
# ─────────────────────────────────────────────

print("\n" + "="*60)
print("  CLEANING DENGUE DATA (OpenDengue National Extract)")
print("="*60)

dengue_raw = pd.read_csv(f"{RAW_DIR}/who_dengue.csv")

print(f"\nRaw shape      : {dengue_raw.shape}")
print(f"Columns        : {list(dengue_raw.columns)}")
print(f"\nFirst 3 rows:")
print(dengue_raw.head(3).to_string())

# OpenDengue column names — adjust if yours differ slightly
# Common columns: adm_0_name, ISO_A3, calendar_start_date,
#                 calendar_end_date, dengue_total, adm_0_name, WHO_region

# Auto-detect key columns
cols = [c.lower() for c in dengue_raw.columns]
dengue_raw.columns = [c.lower() for c in dengue_raw.columns]

print(f"\nNormalised columns: {list(dengue_raw.columns)}")

# Map to standard names (handles slight column name variations)
col_map = {}
for c in dengue_raw.columns:
    if "iso" in c and ("a3" in c or "a0" in c):           col_map[c] = "country_code"
    elif c == "adm_0_name":                col_map[c] = "country_name"
    elif "dengue_total" in c:              col_map[c] = "dengue_cases"
    elif "calendar_start" in c:            col_map[c] = "start_date"
    elif "calendar_end" in c:              col_map[c] = "end_date"
    elif "who_region" in c or "region" in c: col_map[c] = "region"

dengue_raw.rename(columns=col_map, inplace=True)
print(f"Mapped columns : {col_map}")

# Extract year from start_date
if "start_date" in dengue_raw.columns:
    dengue_raw["year"] = pd.to_datetime(
        dengue_raw["start_date"], errors="coerce"
    ).dt.year
else:
    # Fallback if date column named differently
    date_cols = [c for c in dengue_raw.columns if "date" in c or "year" in c]
    print(f"Date-like cols: {date_cols}")
    dengue_raw["year"] = pd.to_datetime(
        dengue_raw[date_cols[0]], errors="coerce"
    ).dt.year

# Keep needed columns
keep = [c for c in ["country_code","country_name","region","year","dengue_cases"]
        if c in dengue_raw.columns]
dengue = dengue_raw[keep].copy()

# Drop rows missing case count or year
dengue.dropna(subset=["dengue_cases", "year"], inplace=True)
dengue = dengue[dengue["dengue_cases"] > 0]

dengue["year"]         = dengue["year"].astype(int)
dengue["dengue_cases"] = dengue["dengue_cases"].astype(float)

dengue = dengue[dengue["year"] >= 2000]

# Aggregate to yearly national totals (OpenDengue may have monthly rows)
agg_cols = {c: "first" for c in ["country_name","region"] if c in dengue.columns}
agg_cols["dengue_cases"] = "sum"

dengue = (dengue
          .groupby(["country_code", "year"])
          .agg(agg_cols)
          .reset_index())

dengue = dengue.sort_values(["country_code","year"]).reset_index(drop=True)

print(f"\nCleaned shape  : {dengue.shape}")
print(f"Year range     : {dengue['year'].min()} – {dengue['year'].max()}")
print(f"Countries      : {dengue['country_code'].nunique()}")
print(f"Missing values : {dengue.isnull().sum().sum()}")
print(f"\nSample rows:")
print(dengue.head(8).to_string(index=False))

dengue.to_csv(f"{PROCESSED_DIR}/dengue_cleaned.csv", index=False)
print(f"\n✅ Saved → data/processed/dengue_cleaned.csv")


# ─────────────────────────────────────────────
# SECTION 3 — MERGE INTO MASTER DATASET
# ─────────────────────────────────────────────

print("\n" + "="*60)
print("  MERGING INTO MASTER DATASET")
print("="*60)

master = pd.merge(
    dengue,
    cholera[["country_code","year","cholera_cases"]],
    on=["country_code","year"],
    how="outer"
)

# Fill NaN case counts with 0 (no reported cases that year)
master["dengue_cases"]  = master["dengue_cases"].fillna(0)
master["cholera_cases"] = master["cholera_cases"].fillna(0)

master = master.sort_values(["country_code","year"]).reset_index(drop=True)

print(f"\nMaster shape   : {master.shape}")
print(f"Countries      : {master['country_code'].nunique()}")
print(f"Year range     : {master['year'].min()} – {master['year'].max()}")
print(f"\nSample rows:")
print(master.head(10).to_string(index=False))

print(f"\nTop 10 countries by total Dengue cases:")
top_dengue = (master.groupby("country_code")["dengue_cases"]
              .sum().sort_values(ascending=False).head(10))
print(top_dengue.to_string())

print(f"\nTop 10 countries by total Cholera cases:")
top_cholera = (master.groupby("country_code")["cholera_cases"]
               .sum().sort_values(ascending=False).head(10))
print(top_cholera.to_string())

master.to_csv(f"{PROCESSED_DIR}/master_disease_data.csv", index=False)
print(f"\n✅ Saved → data/processed/master_disease_data.csv")

print("\n" + "="*60)
print("  ALL DONE — 3 files saved in data/processed/")
print("  cholera_cleaned.csv")
print("  dengue_cleaned.csv")
print("  master_disease_data.csv  ← use this for modeling")
print("="*60 + "\n")