import os
import requests
import pandas as pd

def get_who_indicators():
    """Fetch the list of all available indicators from WHO GHO API."""
    url = "https://ghoapi.azureedge.net/api/Indicator"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json().get('value', [])
    return pd.DataFrame(data)

def fetch_who_data(indicator_code):
    """Fetch disease case data from WHO API for a specific indicator."""
    url = f"https://ghoapi.azureedge.net/api/{indicator_code}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json().get('value', [])
    df = pd.DataFrame(data)
    return df

def fetch_cdc_data():
    """Fetch influenza data from CDC FluView."""
    # TODO: Implement CDC API logic here
    pass

if __name__ == "__main__":
    print("Fetching WHO Cholera case data...")
    
    # Fetch Cholera cases (CHOLERA_0000000001)
    cholera_df = fetch_who_data("CHOLERA_0000000001")
    
    print(f"Successfully fetched {len(cholera_df)} records for Cholera.")
    
    # Ensure the raw data directory exists
    os.makedirs("data/raw", exist_ok=True)
    
    # Save to CSV
    output_path = "data/raw/who_cholera.csv"
    cholera_df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
