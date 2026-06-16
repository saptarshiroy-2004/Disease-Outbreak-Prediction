import pandas as pd
import numpy as np
import os

def engineer_features():
    print("Loading final_dataset.csv...")
    filepath = "data/processed/final_dataset.csv"
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return
        
    df = pd.read_csv(filepath)
    
    # Ensure data is sorted chronologically per country so lags and rolling windows are correct
    df = df.sort_values(by=['country_code', 'year']).reset_index(drop=True)
    
    print("Adding lag features (1, 2, 4 years)...")
    # Shift rows by 1, 2, and 4 within each country group
    df['dengue_lag1'] = df.groupby('country_code')['dengue_cases'].shift(1)
    df['dengue_lag2'] = df.groupby('country_code')['dengue_cases'].shift(2)
    df['dengue_lag4'] = df.groupby('country_code')['dengue_cases'].shift(4)
    
    print("Adding rolling mean (3-year)...")
    # Calculate rolling 3-year mean (min_periods=1 allows calculating it even for the first few years)
    df['dengue_roll3'] = df.groupby('country_code')['dengue_cases'].transform(lambda x: x.rolling(window=3, min_periods=1).mean().shift(1))
    
    print("Adding high_risk binary column...")
    # 1 if dengue_cases > 1000 else 0
    df['high_risk'] = (df['dengue_cases'] > 1000).astype(int)
    
    output_path = "data/processed/model_ready_dataset.csv"
    df.to_csv(output_path, index=False)
    
    print(f"✅ Feature engineering complete! Saved to {output_path}")
    print(f"New dataset shape: {df.shape}")
    print("\nSample rows showing new features:")
    print(df[['country_code', 'year', 'dengue_cases', 'dengue_lag1', 'dengue_lag2', 'dengue_lag4', 'dengue_roll3', 'high_risk']].head(10))

if __name__ == "__main__":
    engineer_features()
