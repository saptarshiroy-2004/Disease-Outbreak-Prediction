# Disease Outbreak Prediction

This repository contains the data pipeline, exploratory analysis, and modeling infrastructure for a global Disease Outbreak Prediction project. Currently, the project tracks two heavily climate-dependent diseases: **Dengue Fever** and **Cholera**.

## Project Status: Phase 1 (Data Collection & Exploration) - Completed

We have successfully built an automated data pipeline that cleans raw disease data and pairs it with decades of historical climate data to create a master modeling dataset.

### 📁 Directory Structure
- `data/raw/` - Contains the raw CSVs from WHO (Cholera) and OpenDengue, plus the Natural Earth shapefiles.
- `data/processed/` - Contains the cleaned datasets (`cholera_cleaned.csv`, `dengue_cleaned.csv`), the intermediate `master_disease_data.csv`, and the ultimate `final_dataset.csv`.
- `notebooks/` - Contains Jupyter Notebooks. Currently features `01-eda-base-map.ipynb` for plotting disease trends and generating interactive choropleth maps.
- `src/data/` - Contains the Python pipeline scripts.

### ⚙️ Pipeline Scripts

1. **`clean_data.py`**
   - **Purpose:** Parses the raw WHO Cholera CSV and the OpenDengue CSV.
   - **Action:** Normalizes column names (extracting standard ISO_A3 country codes), filters out irrelevant metrics, and concatenates both diseases into a single dataset spanning 2000-2025 across 70+ countries.
   - **Output:** `data/processed/master_disease_data.csv`

2. **`fetch_climate_data.py`**
   - **Purpose:** Automates the retrieval of historical climate data.
   - **Action:** Reads the master dataset to identify all unique countries. Uses `geopandas` to extract the geographical centroid (latitude/longitude) of each country. It then hits the Open-Meteo Archive API to download 25 years of daily maximum temperature, minimum temperature, mean temperature, and precipitation data for each centroid. Features automatic rate-limit handling (429 backoff) and resume capabilities.
   - **Output:** `data/raw/climate_data.csv`

3. **`merge_data.py`**
   - **Purpose:** Creates the final modeling dataset.
   - **Action:** Aggregates the daily Open-Meteo climate data into yearly averages (for temperature) and yearly sums (for precipitation). It then performs a left-join to merge these climate features onto the annual disease case data.
   - **Output:** `data/processed/final_dataset.csv`

## Next Steps: Phase 2 (Modeling)
With the creation of `final_dataset.csv` containing both target variables (disease cases) and features (climate data), the next phase involves building predictive machine learning models to forecast outbreaks based on shifting climate patterns.
