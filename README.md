# 🌍 Disease Outbreak Prediction Platform

An ML-powered epidemiological dashboard that forecasts global outbreaks of Dengue and Cholera. The platform aggregates multi-source historical disease data, integrates climate and geospatial variables, and utilizes both univariate and multivariate machine learning models (Prophet & XGBoost) to provide actionable insights for global public health.

> 🚀 **Live Demo:** Try the interactive dashboard here: [Streamlit Cloud URL]

---

## 🛠️ Tech Stack

- **Frontend & Visualization:** [Streamlit](https://streamlit.io/), Plotly, Folium
- **Data Engineering:** Pandas, Numpy, GeoPandas
- **Machine Learning:** 
  - **Prophet:** Global aggregate baseline forecasting
  - **XGBoost:** Country-level multivariate forecasting
  - **Scikit-Learn:** Model evaluation and baselines
- **Explainable AI (XAI):** SHAP (SHapley Additive exPlanations)

## 📊 Data Sources

This platform relies on robust, open-source epidemiological and environmental data:
- **[OpenDengue](https://opendengue.org/):** Global Dengue case incidence data (National Extract).
- **[WHO Global Health Observatory (GHO)](https://www.who.int/data/gho):** Historical global Cholera case data.
- **[Open-Meteo](https://open-meteo.com/):** Historical climate data including mean temperature and precipitation.
- **[Natural Earth](https://www.naturalearthdata.com/):** High-resolution geospatial boundaries for choropleth mapping.

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Disease-Outbreak-Prediction.git
cd Disease-Outbreak-Prediction
```

### 2. Install dependencies
It is recommended to use a virtual environment.
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### 3. Run the complete data pipeline (Optional)
If you want to re-download the data and re-train the models from scratch:
```bash
python src/data/fetch_disease_data.py
python data/clean_data.py
python src/data/fix_nans.py
python src/data/merge_data.py
python src/data/feature_engineering.py
python src/models/train_prophet.py
python src/models/train_xgboost.py
python src/models/train_all_models.py
```

### 4. Launch the Streamlit Dashboard
```bash
streamlit run src/app/streamlit_app.py
```
The dashboard will be available in your browser at `http://localhost:8501`.
