import pandas as pd
import numpy as np
import os
import logging
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Suppress cmdstanpy logging to keep terminal output clean
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

def prepare_global_data(df, disease_col):
    # Group by year and sum cases globally
    global_df = df.groupby('year')[disease_col].sum().reset_index()
    
    # Prophet requires 'ds' (date) and 'y' (target)
    global_df['ds'] = pd.to_datetime(global_df['year'].astype(str) + '-01-01')
    global_df = global_df.rename(columns={disease_col: 'y'})
    
    return global_df[['ds', 'y', 'year']].sort_values('ds').reset_index(drop=True)

def train_and_forecast(df, disease_name):
    print(f"\n" + "="*50)
    print(f"  TRAINING PROPHET MODEL: {disease_name.upper()}")
    print("="*50)
    
    # The user asked for a holdout of the last 3 years
    train_df = df.iloc[:-3]
    test_df = df.iloc[-3:]
    
    print(f"Train set: {train_df['year'].min()} to {train_df['year'].max()} ({len(train_df)} years)")
    print(f"Test set : {test_df['year'].min()} to {test_df['year'].max()} ({len(test_df)} years)")
    
    # 1. Train and Evaluate on Holdout
    # We disable weekly/daily seasonality because we have yearly aggregated data
    model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
    model.fit(train_df[['ds', 'y']])
    
    forecast_test = model.predict(test_df[['ds']])
    mae = mean_absolute_error(test_df['y'], forecast_test['yhat'])
    rmse = np.sqrt(mean_squared_error(test_df['y'], forecast_test['yhat']))
    
    print(f"\n{disease_name} Holdout Evaluation (Last 3 Years):")
    print(f"MAE  : {mae:,.2f} cases")
    print(f"RMSE : {rmse:,.2f} cases")
    
    # 2. Retrain on Full Dataset for Real Forecasting
    print(f"\nRetraining on full dataset to forecast future...")
    final_model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
    final_model.fit(df[['ds', 'y']])
    
    # Forecast 5 years ahead (adds 5 rows)
    future = final_model.make_future_dataframe(periods=5, freq='YS')
    forecast = final_model.predict(future)
    
    # 3. Plot the Forecast
    fig = final_model.plot(forecast)
    plt.title(f'Global {disease_name} Forecast (5 Years Ahead)')
    plt.xlabel('Year')
    plt.ylabel('Total Cases')
    
    # Save the plot
    os.makedirs('reports/figures', exist_ok=True)
    plot_path = f'reports/figures/{disease_name.lower()}_forecast.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"Forecast plot saved to {plot_path}")
    
    # Format the forecast output for the CSV
    forecast['disease'] = disease_name
    forecast['year'] = forecast['ds'].dt.year
    
    # Return just the essential columns
    return forecast[['disease', 'year', 'yhat', 'yhat_lower', 'yhat_upper']]

def main():
    filepath = "data/processed/model_ready_dataset.csv"
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
        
    df = pd.read_csv(filepath)
    
    # Prepare data for Prophet
    dengue_data = prepare_global_data(df, 'dengue_cases')
    cholera_data = prepare_global_data(df, 'cholera_cases')
    
    # Train and Forecast for both diseases
    dengue_forecast = train_and_forecast(dengue_data, "Dengue")
    cholera_forecast = train_and_forecast(cholera_data, "Cholera")
    
    # Combine and save
    print("\n" + "="*50)
    all_forecasts = pd.concat([dengue_forecast, cholera_forecast], ignore_index=True)
    os.makedirs('data/processed', exist_ok=True)
    out_path = "data/processed/prophet_forecasts.csv"
    all_forecasts.to_csv(out_path, index=False)
    print(f"✅ All forecasts successfully saved to {out_path}")
    
    print("\nSample Forecast Output (2024+):")
    future_only = all_forecasts[all_forecasts['year'] >= 2024]
    print(future_only.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
