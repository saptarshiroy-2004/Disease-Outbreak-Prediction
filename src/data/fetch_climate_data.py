import os
import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_historical_weather(latitude: float, longitude: float, start_date: str, end_date: str):
    """
    Fetch historical daily weather data from Open-Meteo Archive API.
    Variables included: max temp, min temp, mean temp, precipitation.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", "precipitation_sum", "rain_sum"],
        "timezone": "auto"
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    # Extract daily data into a DataFrame
    daily_data = data.get("daily", {})
    if not daily_data:
        return pd.DataFrame()
        
    df = pd.DataFrame({
        "date": pd.to_datetime(daily_data.get("time", [])),
        "temp_max_c": daily_data.get("temperature_2m_max", []),
        "temp_min_c": daily_data.get("temperature_2m_min", []),
        "temp_mean_c": daily_data.get("temperature_2m_mean", []),
        "precipitation_mm": daily_data.get("precipitation_sum", []),
        "rain_mm": daily_data.get("rain_sum", [])
    })
    
    # Add coordinates
    df['latitude'] = latitude
    df['longitude'] = longitude
    
    return df

if __name__ == "__main__":
    import geopandas as gpd
    import time
    import warnings
    warnings.filterwarnings('ignore')
    
    print("Loading master disease dataset...")
    try:
        master_df = pd.read_csv("data/processed/master_disease_data.csv")
    except FileNotFoundError:
        print("Master dataset not found. Please run clean_data.py first.")
        exit(1)
        
    countries = master_df['country_code'].dropna().unique()
    print(f"Found {len(countries)} unique countries to fetch climate data for.")
    
    # Use geopandas to get country centroids from local file
    world = gpd.read_file('data/raw/ne_110m_admin_0_countries.zip')
    world.columns = world.columns.str.lower() # Normalize to lowercase 'iso_a3'
    world['centroid'] = world.geometry.centroid
    
    all_climate_data = []
    output_path = "data/raw/climate_data.csv"
    
    # Resume logic
    fetched_countries = set()
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        if 'country_code' in existing_df.columns:
            fetched_countries = set(existing_df['country_code'].unique())
            all_climate_data.append(existing_df)
            print(f"Resuming... found {len(fetched_countries)} countries already downloaded.")
    
    start = "2000-01-01"
    end = "2024-12-31"
    
    for i, country_code in enumerate(countries):
        if country_code in fetched_countries:
            continue
            
        country_geom = world[world['iso_a3'] == country_code]
        if country_geom.empty:
            print(f"Warning: Could not find geometry for {country_code}. Skipping.")
            continue
            
        lat = country_geom.iloc[0]['centroid'].y
        lon = country_geom.iloc[0]['centroid'].x
        
        print(f"[{i+1}/{len(countries)}] Fetching climate for {country_code} (Lat: {lat:.2f}, Lon: {lon:.2f})...")
        
        retries = 3
        for attempt in range(retries):
            try:
                climate_df = fetch_historical_weather(lat, lon, start, end)
                if not climate_df.empty:
                    climate_df['country_code'] = country_code
                    all_climate_data.append(climate_df)
                    
                    # Incremental save so we don't lose progress!
                    final_df = pd.concat(all_climate_data, ignore_index=True)
                    final_df.to_csv(output_path, index=False)
                break # Success!
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    print(f"  -> Rate limit hit (429). Sleeping for 60 seconds (Attempt {attempt+1}/{retries})...")
                    time.sleep(60)
                else:
                    print(f"Error fetching data for {country_code}: {e}")
                    break
            except Exception as e:
                print(f"Error fetching data for {country_code}: {e}")
                break
                
        # Base sleep to prevent hitting limits too fast
        time.sleep(2.0)
        
    if all_climate_data:
        final_df = pd.concat(all_climate_data, ignore_index=True)
        print(f"\nSuccessfully saved combined climate data to {output_path} (Shape: {final_df.shape})")
    else:
        print("\nNo climate data fetched.")
