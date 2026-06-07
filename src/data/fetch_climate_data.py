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
    print("Fetching sample climate data (e.g., Delhi, India for 2023)...")
    
    # Example coordinates for Delhi, India (a Dengue hotspot)
    lat, lon = 28.6139, 77.2090
    start = "2023-01-01"
    end = "2023-12-31"
    
    climate_df = fetch_historical_weather(lat, lon, start, end)
    print(f"Successfully fetched {len(climate_df)} days of weather data.")
    print(climate_df.head())
    
    # Save sample to data/raw
    os.makedirs("data/raw", exist_ok=True)
    output_path = "data/raw/delhi_climate_2023.csv"
    climate_df.to_csv(output_path, index=False)
    print(f"Saved sample climate data to {output_path}")
