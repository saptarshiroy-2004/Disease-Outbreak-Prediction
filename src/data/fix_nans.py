import pandas as pd
import pycountry
import numpy as np

def get_country_name(code):
    try:
        country = pycountry.countries.get(alpha_3=code)
        if country:
            return country.name
    except Exception:
        pass
    return np.nan

# Manual dictionary for WHO regions
who_regions = {
    'AFG': 'Eastern Mediterranean', 'AGO': 'Africa', 'AUS': 'Western Pacific',
    'BDI': 'Africa', 'BEN': 'Africa', 'BFA': 'Africa', 'BHR': 'Eastern Mediterranean',
    'BHS': 'Americas', 'CAF': 'Africa', 'CAN': 'Americas', 'CHE': 'Europe',
    'CHL': 'Americas', 'CHN': 'Western Pacific', 'CIV': 'Africa', 'CMR': 'Africa',
    'COD': 'Africa', 'COG': 'Africa', 'CUB': 'Americas', 'DEU': 'Europe',
    'DNK': 'Europe', 'DOM': 'Americas', 'ESP': 'Europe', 'FRA': 'Europe',
    'GBR': 'Europe', 'GHA': 'Africa', 'GIN': 'Africa', 'GNB': 'Africa',
    'HTI': 'Americas', 'IRN': 'Eastern Mediterranean', 'IRQ': 'Eastern Mediterranean',
    'ISR': 'Europe', 'ITA': 'Europe', 'JPN': 'Western Pacific', 'KEN': 'Africa',
    'KOR': 'Western Pacific', 'KWT': 'Eastern Mediterranean', 'LBR': 'Africa',
    'MEX': 'Americas', 'MLI': 'Africa', 'MOZ': 'Africa', 'MWI': 'Africa',
    'MYS': 'Western Pacific', 'NAM': 'Africa', 'NER': 'Africa', 'NGA': 'Africa',
    'NLD': 'Europe', 'NOR': 'Europe', 'OMN': 'Eastern Mediterranean',
    'PAK': 'Eastern Mediterranean', 'PHL': 'Western Pacific', 'RUS': 'Europe',
    'RWA': 'Africa', 'SEN': 'Africa', 'SGP': 'Western Pacific', 'SLE': 'Africa',
    'SOM': 'Eastern Mediterranean', 'SSD': 'Africa', 'SWE': 'Europe',
    'TGO': 'Africa', 'TZA': 'Africa', 'UGA': 'Africa', 'USA': 'Americas',
    'VEN': 'Americas', 'YEM': 'Eastern Mediterranean', 'ZAF': 'Africa',
    'ZMB': 'Africa', 'ZWE': 'Africa'
}

def fix_nans():
    print("Loading master_disease_data.csv...")
    filepath = "data/processed/master_disease_data.csv"
    df = pd.read_csv(filepath)
    
    if 'region' not in df.columns:
        df['region'] = np.nan
        
    missing_names = df['country_name'].isna()
    missing_regions = df['region'].isna()
    
    print(f"Found {missing_names.sum()} missing country names.")
    print(f"Found {missing_regions.sum()} missing regions.")
    
    # Fill country names using pycountry
    print("Filling country names with pycountry...")
    df.loc[missing_names, 'country_name'] = df.loc[missing_names, 'country_code'].apply(get_country_name)
    
    # Fill regions using manual dictionary
    print("Filling regions with manual WHO dictionary...")
    df.loc[missing_regions, 'region'] = df.loc[missing_regions, 'country_code'].map(who_regions)
    
    # Fill any remaining NaNs with 'Unknown'
    df['country_name'] = df['country_name'].fillna('Unknown')
    df['region'] = df['region'].fillna('Unknown')
    
    print(f"Remaining missing names: {df['country_name'].isna().sum()}")
    print(f"Remaining missing regions: {df['region'].isna().sum()}")
    
    df.to_csv(filepath, index=False)
    print(f"Successfully saved fixed data to {filepath}")

if __name__ == "__main__":
    fix_nans()
