import pandas as pd
import os

def merge_datasets():
    print("Loading data...")
    try:
        disease_df = pd.read_csv("data/processed/master_disease_data.csv")
        climate_df = pd.read_csv("data/raw/climate_data.csv")
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    print("Aggregating climate data to yearly averages/totals...")
    # The climate data is daily, but disease data is yearly. We must aggregate it first!
    climate_df['date'] = pd.to_datetime(climate_df['date'], format='mixed')
    climate_df['year'] = climate_df['date'].dt.year

    # Aggregate: Mean for temperature, Sum for precipitation
    climate_agg = climate_df.groupby(['country_code', 'year']).agg({
        'temp_max_c': 'mean',
        'temp_min_c': 'mean',
        'temp_mean_c': 'mean',
        'precipitation_mm': 'sum',
        'rain_mm': 'sum'
    }).reset_index()

    print("Merging datasets...")
    # Merge on country_code and year (Left join so we keep all disease rows even if climate data is missing)
    final_df = pd.merge(disease_df, climate_agg, on=['country_code', 'year'], how='left')

    # Save to processed folder
    output_path = "data/processed/final_dataset.csv"
    final_df.to_csv(output_path, index=False)
    
    print(f"✅ Final dataset successfully saved to {output_path}")
    print(f"Shape of final dataset: {final_df.shape}")
    print("\nSample rows:")
    print(final_df.head())

if __name__ == "__main__":
    merge_datasets()
