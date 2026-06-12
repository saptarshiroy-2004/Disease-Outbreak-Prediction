import pandas as pd
import numpy as np
import os
import pickle
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import shap
import matplotlib.pyplot as plt

def main():
    filepath = "data/processed/model_ready_dataset.csv"
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return
        
    df = pd.read_csv(filepath)
    
    # 1. Define Features and Target
    features = [
        'year', 'temp_mean_c', 'precipitation_mm', 
        'dengue_lag1', 'dengue_lag2', 'dengue_lag4', 
        'dengue_roll3', 'high_risk'
    ]
    target = 'dengue_cases'
    
    # Drop rows where any of these features or target is NaN (Lags create NaNs for first few years)
    model_df = df[features + [target, 'country_code']].dropna()
    
    # 2. Split: last 3 years as test set (2023, 2024, 2025)
    max_year = model_df['year'].max()
    test_years = [max_year, max_year - 1, max_year - 2]
    
    train_df = model_df[~model_df['year'].isin(test_years)]
    test_df = model_df[model_df['year'].isin(test_years)]
    
    X_train = train_df[features]
    y_train = train_df[target]
    
    X_test = test_df[features]
    y_test = test_df[target]
    
    print("\n" + "="*50)
    print("  TRAINING XGBOOST MODEL: DENGUE (Country-Level)")
    print("="*50)
    print(f"Train Set: {train_df['year'].min()} to {train_df['year'].max()} ({len(X_train)} country-years)")
    print(f"Test Set : {test_df['year'].min()} to {test_df['year'].max()} ({len(X_test)} country-years)")
    
    # 3. Train XGBoost with default parameters
    model = XGBRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Evaluate
    preds = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    print("\n[ XGBoost Country-Level Evaluation (Test Set) ]")
    print(f"MAE  : {mae:,.2f} cases per country-year")
    print(f"RMSE : {rmse:,.2f} cases per country-year")
    
    # To compare with Prophet (which was global), aggregate predictions globally by year
    test_df_results = test_df.copy()
    test_df_results['predicted'] = preds
    
    global_test = test_df_results.groupby('year')[[target, 'predicted']].sum()
    global_mae = mean_absolute_error(global_test[target], global_test['predicted'])
    global_rmse = np.sqrt(mean_squared_error(global_test[target], global_test['predicted']))
    
    print("\n[ XGBoost Global-Aggregated Evaluation ]")
    print(f"Prophet MAE Baseline : 427,369.84 cases")
    print(f"XGBoost Global MAE   : {global_mae:,.2f} cases  <-- COMPARISON")
    print("-" * 50)
    print(f"Prophet RMSE Baseline: 452,919.91 cases")
    print(f"XGBoost Global RMSE  : {global_rmse:,.2f} cases  <-- COMPARISON")
    
    # 5. SHAP values & Feature Importance Plot
    print("\nGenerating SHAP values and feature importance plot...")
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)
    
    plt.figure()
    shap.summary_plot(shap_values, X_train, plot_type="bar", max_display=10, show=False)
    
    os.makedirs("reports/figures", exist_ok=True)
    shap_plot_path = "reports/figures/xgboost_dengue_shap.png"
    plt.savefig(shap_plot_path, bbox_inches='tight')
    plt.close()
    print(f"✅ SHAP feature importance saved to {shap_plot_path}")
    
    # 6. Save model
    os.makedirs("src/models", exist_ok=True)
    model_path = "src/models/xgboost_dengue.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Model saved to {model_path}")

if __name__ == "__main__":
    main()
