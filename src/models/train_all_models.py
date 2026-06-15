import pandas as pd
import numpy as np
import time
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def get_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

def main():
    print("="*60)
    print("  TRAINING AND EVALUATING ALL MODELS (DENGUE)")
    print("="*60)
    
    filepath = "data/processed/model_ready_dataset.csv"
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return
        
    df = pd.read_csv(filepath)
    
    features = [
        'year', 'temp_mean_c', 'precipitation_mm', 
        'dengue_lag1', 'dengue_lag2', 'dengue_lag4', 
        'dengue_roll3'
    ]
    target = 'dengue_cases'
    
    model_df = df[features + [target, 'country_code']].dropna()
    max_year = model_df['year'].max()
    test_years = [max_year, max_year - 1, max_year - 2]
    
    train_df = model_df[~model_df['year'].isin(test_years)]
    test_df = model_df[model_df['year'].isin(test_years)]
    
    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]
    
    results = []
    
    # 1. Linear Regression
    print("Training Linear Regression...")
    start_time = time.time()
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_time = time.time() - start_time
    lr_preds = lr.predict(X_test)
    mae, rmse, r2 = get_metrics(y_test, lr_preds)
    results.append(['Linear Regression', mae, rmse, r2, lr_time])
    
    # 2. Random Forest
    print("Training Random Forest...")
    start_time = time.time()
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_time = time.time() - start_time
    rf_preds = rf.predict(X_test)
    mae, rmse, r2 = get_metrics(y_test, rf_preds)
    results.append(['Random Forest', mae, rmse, r2, rf_time])
    
    # 3. XGBoost (Saved)
    print("Evaluating XGBoost (saved predictions)...")
    xgb_df = pd.read_csv("data/processed/xgboost_forecasts.csv")
    mae, rmse, r2 = get_metrics(xgb_df['dengue_cases'], xgb_df['predicted_cases'])
    results.append(['XGBoost', mae, rmse, r2, np.nan]) # Training time omitted for saved models
    
    # 4. Prophet (Saved)
    # Prophet predictions are global aggregate, so we evaluate its metrics against the global aggregate actuals
    print("Evaluating Prophet (saved predictions)...")
    try:
        p_df = pd.read_csv("data/processed/prophet_forecasts.csv")
        p_df = p_df[(p_df['disease'] == 'Dengue') & (p_df['year'].isin(test_years))]
        global_actuals = df[df['year'].isin(test_years)].groupby('year')['dengue_cases'].sum().reset_index()
        p_merged = pd.merge(global_actuals, p_df, on='year')
        
        if not p_merged.empty:
            mae, rmse, r2 = get_metrics(p_merged['dengue_cases'], p_merged['yhat'])
            results.append(['Prophet (Global)', mae, rmse, r2, np.nan])
    except Exception as e:
        print(f"Could not evaluate Prophet: {e}")
        
    # Save results
    res_df = pd.DataFrame(results, columns=['model_name', 'MAE', 'RMSE', 'R2_score', 'training_time_seconds'])
    res_df.to_csv("data/processed/model_comparison.csv", index=False)
    print("\n✅ Saved comparison to data/processed/model_comparison.csv")
    print("\n" + res_df.to_string(index=False))

if __name__ == "__main__":
    main()
