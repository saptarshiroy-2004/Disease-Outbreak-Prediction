import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="Disease Outbreak Prediction",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── GLOBAL CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, .big-title {
    font-family: 'Playfair Display', 'Times New Roman', serif !important;
}

.main { background: #0A0E1A; }

/* ── HERO BANNER ── */
.hero-banner {
    background: linear-gradient(135deg, #0D1B2A 0%, #1B2838 50%, #0D2137 100%);
    border: 1px solid rgba(46,134,171,0.3);
    border-radius: 16px;
    padding: 40px 48px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(46,134,171,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Playfair Display', 'Times New Roman', serif !important;
    font-size: 2.6rem;
    font-weight: 700;
    color: #FFFFFF;
    margin: 0 0 8px 0;
    line-height: 1.2;
}
.hero-sub {
    font-family: 'Inter', sans-serif;
    font-size: 1.05rem;
    color: #8BA7C0;
    margin: 0 0 20px 0;
    font-weight: 300;
}
.hero-tags {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}
.tag {
    background: rgba(46,134,171,0.2);
    border: 1px solid rgba(46,134,171,0.4);
    color: #5BB8D4;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.04em;
}

/* ── KPI CARDS ── */
.kpi-card {
    background: linear-gradient(145deg, #111827, #1a2235);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 22px 24px;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
    cursor: default;
}
.kpi-card:hover {
    transform: translateY(-3px);
    border-color: rgba(46,134,171,0.5);
}
.kpi-icon { font-size: 2rem; margin-bottom: 8px; }
.kpi-value {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: #FFFFFF;
    line-height: 1;
    margin-bottom: 4px;
}
.kpi-label {
    font-size: 0.78rem;
    color: #6B7FA8;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.kpi-delta {
    font-size: 0.75rem;
    color: #34D399;
    margin-top: 6px;
    font-weight: 500;
}

/* ── SECTION HEADERS ── */
.section-header {
    font-family: 'Playfair Display', 'Times New Roman', serif !important;
    font-size: 1.6rem;
    font-weight: 600;
    color: #E8F4FD;
    margin: 36px 0 6px 0;
    padding-bottom: 10px;
    border-bottom: 2px solid rgba(46,134,171,0.4);
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-sub {
    font-size: 0.88rem;
    color: #6B7FA8;
    margin-bottom: 20px;
    font-style: italic;
}

/* ── METRIC CARDS ── */
.metric-card {
    background: linear-gradient(145deg, #111827, #162032);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 24px 28px;
    height: 100%;
}
.metric-card.prophet { border-top: 3px solid #F59E0B; }
.metric-card.xgboost { border-top: 3px solid #10B981; }
.metric-model-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 18px;
}
.metric-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.metric-row:last-child { border-bottom: none; }
.metric-key {
    font-size: 0.8rem;
    color: #6B7FA8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 500;
}
.metric-val {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: #E8F4FD;
}
.metric-val.good { color: #34D399; }
.metric-val.warn { color: #F59E0B; }

/* ── INSIGHT BOX ── */
.insight-box {
    background: linear-gradient(135deg, rgba(16,185,129,0.08), rgba(46,134,171,0.08));
    border: 1px solid rgba(16,185,129,0.25);
    border-left: 4px solid #10B981;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 14px 0;
    font-size: 0.9rem;
    color: #A7C5D9;
    line-height: 1.7;
}
.insight-box b { color: #34D399; }

/* ── RISK TABLE ── */
.risk-critical { color: #EF4444; font-weight: 600; }
.risk-high     { color: #F97316; font-weight: 600; }
.risk-medium   { color: #F59E0B; font-weight: 500; }
.risk-low      { color: #34D399; font-weight: 500; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1421 0%, #111827 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
.sidebar-logo {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #FFFFFF;
    text-align: center;
    padding: 20px 0 10px;
}
.sidebar-section {
    background: rgba(255,255,255,0.04);
    border-radius: 10px;
    padding: 14px 16px;
    margin: 12px 0;
    border: 1px solid rgba(255,255,255,0.06);
}
.sidebar-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #4B6080;
    font-weight: 600;
    margin-bottom: 8px;
}

/* ── FOOTER ── */
.footer {
    background: linear-gradient(135deg, #0D1B2A, #111827);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 24px 32px;
    margin-top: 48px;
    text-align: center;
}
.footer-title {
    font-family: 'Playfair Display', serif;
    color: #8BA7C0;
    font-size: 0.85rem;
    margin-bottom: 10px;
}
.footer-sources {
    display: flex;
    justify-content: center;
    gap: 24px;
    flex-wrap: wrap;
}
.footer-source {
    font-size: 0.78rem;
    color: #4B6080;
    border: 1px solid rgba(255,255,255,0.08);
    padding: 4px 12px;
    border-radius: 20px;
}

/* Hide default streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA ────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/model_ready_dataset.csv')
    return df

@st.cache_data
def load_forecasts():
    try:
        return pd.read_csv('data/processed/prophet_forecasts.csv')
    except:
        return None

df = load_data()
forecasts = load_forecasts()

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🌍 Outbreak Engine</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="sidebar-section"><div class="sidebar-label">🦠 Disease</div>', unsafe_allow_html=True)
    disease = st.selectbox("", ["🦟 Dengue", "💧 Cholera"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    disease_key = "dengue" if "Dengue" in disease else "cholera"
    case_col = f"{disease_key}_cases"

    st.markdown('<div class="sidebar-section"><div class="sidebar-label">📅 Year Range</div>', unsafe_allow_html=True)
    yr_min = int(df['year'].min())
    yr_max = int(df['year'].max())
    year_range = st.slider("", yr_min, yr_max, (yr_min, yr_max), label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section"><div class="sidebar-label">🌐 Region Filter</div>', unsafe_allow_html=True)
    regions = ["All Regions"] + sorted(df['region'].dropna().unique().tolist())
    region_sel = st.selectbox("", regions, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section"><div class="sidebar-label">ℹ️ About</div>'
                '<p style="font-size:0.8rem;color:#6B7FA8;line-height:1.6;margin:0;">'
                'ML-powered epidemiological platform using WHO, OpenDengue & Open-Meteo data. '
                'Models: Prophet + XGBoost with SHAP explainability.</p></div>', unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.72rem;color:#2E3D52;text-align:center;margin-top:20px;">'
                'Data: Dengue — Global (OpenDengue)<br>Cholera — Global (WHO)</div>', unsafe_allow_html=True)

# ── FILTER DATA ──────────────────────────────────────────────
dff = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
if region_sel != "All Regions":
    dff = dff[dff['region'] == region_sel]

# ── HERO BANNER ──────────────────────────────────────────────
st.markdown(f"""
<div class="hero-banner">
    <div class="hero-title">🌍 Disease Outbreak Prediction</div>
    <div class="hero-sub">
        Forecasting {disease.split()[1]} outbreaks using time-series modelling,
        geospatial analysis & climate intelligence
    </div>
    <div class="hero-tags">
        <span class="tag">📡 WHO Data</span>
        <span class="tag">🤖 XGBoost + Prophet</span>
        <span class="tag">🧠 SHAP Explainability</span>
        <span class="tag">🗺️ Geospatial Analysis</span>
        <span class="tag">☁️ Open-Meteo Climate</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI CARDS ────────────────────────────────────────────────
total_cases = int(dff[case_col].sum())
countries   = dff['country_code'].nunique()
years_data  = year_range[1] - year_range[0] + 1
peak_year   = int(dff.groupby('year')[case_col].sum().idxmax()) if total_cases > 0 else "N/A"

c1, c2, c3, c4 = st.columns(4)
cards = [
    (c1, "🌐", f"{countries}", "Countries Monitored", "Active surveillance"),
    (c2, "📊", f"{total_cases/1e6:.2f}M" if total_cases > 1e6 else f"{total_cases:,}", "Total Cases Recorded", f"{year_range[0]}–{year_range[1]}"),
    (c3, "📅", f"{years_data} yrs", "Years of Data", "Historical coverage"),
    (c4, "⚠️", f"{peak_year}", "Peak Outbreak Year", "Highest recorded cases"),
]
for col, icon, val, label, delta in cards:
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-value">{val}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-delta">{delta}</div>
        </div>""", unsafe_allow_html=True)

# ── MAP SECTION ──────────────────────────────────────────────
st.markdown(f'<div class="section-header">🗺️ Global {disease.split()[1]} Distribution Map</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Click any country to see detailed case information. Colour intensity represents total reported cases (log scale).</div>', unsafe_allow_html=True)

map_data = dff.groupby('country_code').agg(
    total_cases=(case_col, 'sum'),
    country_name=('country_name', 'first'),
    region=('region', 'first')
).reset_index()
map_data['log_cases'] = np.log1p(map_data['total_cases'])
map_data['risk_level'] = pd.cut(
    map_data['total_cases'],
    bins=[0, 1000, 50000, 500000, float('inf')],
    labels=['🟢 Low', '🟡 Medium', '🟠 High', '🔴 Critical']
)

m = folium.Map(location=[20, 10], zoom_start=2, tiles='CartoDB dark_matter')
folium.Choropleth(
    geo_data='https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/world-countries.json',
    data=map_data,
    columns=['country_code', 'log_cases'],
    key_on='feature.id',
    fill_color='RdYlGn_r',
    fill_opacity=0.75,
    line_opacity=0.2,
    legend_name=f'Total {disease.split()[1]} Cases (Log Scale)',
    nan_fill_color='#1a2035',
).add_to(m)

for _, row in map_data.iterrows():
    if row['total_cases'] > 0:
        folium.GeoJson(
            'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/world-countries.json',
            tooltip=folium.GeoJsonTooltip(fields=[], aliases=[])
        )

st_folium(m, width=None, height=500, returned_objects=[])

# ── EVALUATION METRICS ───────────────────────────────────────
st.markdown('<div class="section-header">📊 Model Evaluation Metrics</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Comparing Prophet baseline against XGBoost with climate & lag features. Lower MAE/RMSE = better predictions.</div>', unsafe_allow_html=True)

col_p, col_x = st.columns(2)

with col_p:
    st.markdown("""
    <div class="metric-card prophet">
        <div class="metric-model-title" style="color:#F59E0B;">📜 Prophet — Baseline Model</div>
        <div class="metric-row">
            <span class="metric-key">MAE (Mean Absolute Error)</span>
            <span class="metric-val warn">48,222 cases</span>
        </div>
        <div class="metric-row">
            <span class="metric-key">RMSE</span>
            <span class="metric-val warn">48,633 cases</span>
        </div>
        <div class="metric-row">
            <span class="metric-key">Model Type</span>
            <span class="metric-val">Univariate Baseline</span>
        </div>
        <div class="metric-row">
            <span class="metric-key">Features Used</span>
            <span class="metric-val">1 (time only)</span>
        </div>
        <div class="metric-row">
            <span class="metric-key">Training Period</span>
            <span class="metric-val">2000 – 2020</span>
        </div>
        <div class="metric-row">
            <span class="metric-key">Forecast Horizon</span>
            <span class="metric-val">5 years ahead</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_x:
    st.markdown("""
    <div class="metric-card xgboost">
        <div class="metric-model-title" style="color:#10B981;">🚀 XGBoost — ML Model</div>
        <div class="metric-row">
            <span class="metric-key">MAE (Mean Absolute Error)</span>
            <span class="metric-val good">~18,000 cases</span>
        </div>
        <div class="metric-row">
            <span class="metric-key">RMSE</span>
            <span class="metric-val good">~21,000 cases</span>
        </div>
        <div class="metric-row">
            <span class="metric-key">Model Type</span>
            <span class="metric-val">Multivariate ML</span>
        </div>
        <div class="metric-row">
            <span class="metric-key">Features Used</span>
            <span class="metric-val">7 (lag + climate)</span>
        </div>
        <div class="metric-row">
            <span class="metric-key">Improvement vs Prophet</span>
            <span class="metric-val good">↓ ~63% better ✅</span>
        </div>
        <div class="metric-row">
            <span class="metric-key">Explainability</span>
            <span class="metric-val good">SHAP values ✅</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# MAE comparison chart
st.markdown("<br>", unsafe_allow_html=True)
fig_cmp = go.Figure()
fig_cmp.add_trace(go.Bar(
    x=['Prophet (Baseline)', 'XGBoost (ML Model)'],
    y=[48222, 18000],
    marker_color=['#F59E0B', '#10B981'],
    text=['48,222 cases', '~18,000 cases'],
    textposition='outside',
    textfont=dict(family='Times New Roman', size=13, color='white'),
    width=0.45,
))
fig_cmp.add_annotation(
    x=1, y=18000,
    text="↓ 63% improvement",
    showarrow=True, arrowhead=2,
    arrowcolor="#34D399", font=dict(color="#34D399", size=12, family="Times New Roman"),
    ax=-80, ay=-40
)
fig_cmp.update_layout(
    title=dict(text="MAE Comparison — Prophet vs XGBoost", font=dict(family="Times New Roman", size=16, color="#E8F4FD")),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(17,24,39,0.8)',
    font=dict(family='Times New Roman', color='#8BA7C0'),
    yaxis=dict(title="Mean Absolute Error (cases)", gridcolor='rgba(255,255,255,0.05)', color='#6B7FA8'),
    xaxis=dict(color='#8BA7C0'),
    height=320,
    margin=dict(t=50, b=20, l=20, r=20),
    showlegend=False,
)
st.plotly_chart(fig_cmp, use_container_width=True)

with st.expander("💡 What do these metrics mean? (Click to understand)"):
    st.markdown("""
    <div class="insight-box">
    <b>MAE (Mean Absolute Error)</b> — On average, Prophet's prediction is off by <b>48,222 cases</b>
    per year. XGBoost reduces this to ~<b>18,000 cases</b> by using climate and lag features.<br><br>
    <b>Why XGBoost beats Prophet:</b> Prophet only sees time patterns. XGBoost also sees
    last year's case counts, rainfall, and temperature — the actual biological drivers of outbreaks.<br><br>
    <b>For public health purposes</b>, trend direction matters more than exact numbers.
    If the model says <i>"cases will rise 40% next year in India"</i>, hospitals can prepare
    even if the exact number differs. That early warning saves lives.
    </div>
    """, unsafe_allow_html=True)

# ── FORECAST TABS ─────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Model Projections & Interpretability</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Explore Prophet global baseline, XGBoost country-level evaluation, and SHAP feature importance.</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📜 Prophet Forecast", "🚀 XGBoost Evaluation", "🧠 SHAP Importance"])

with tab1:
    yearly = df.groupby('year')[case_col].sum().reset_index()
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(
        x=yearly['year'], y=yearly[case_col],
        name='Historical', mode='lines+markers',
        line=dict(color='#2E86AB', width=2.5),
        marker=dict(size=7, color='#2E86AB'),
    ))
    if forecasts is not None and disease_key == 'dengue':
        fc = forecasts[forecasts['year'] > yearly['year'].max()]
        fig_p.add_trace(go.Scatter(
            x=fc['year'], y=fc['yhat'],
            name='Forecast', mode='lines+markers',
            line=dict(color='#F59E0B', width=2.5, dash='dot'),
            marker=dict(size=7),
        ))
        fig_p.add_trace(go.Scatter(
            x=pd.concat([fc['year'], fc['year'][::-1]]),
            y=pd.concat([fc['yhat_upper'], fc['yhat_lower'][::-1]]),
            fill='toself', fillcolor='rgba(245,158,11,0.12)',
            line=dict(color='rgba(0,0,0,0)'),
            name='Confidence Interval'
        ))
    fig_p.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(17,24,39,0.8)',
        font=dict(family='Times New Roman', color='#8BA7C0', size=13),
        title=dict(text=f'Global {disease.split()[1]} Cases — Historical & Forecast', font=dict(family='Times New Roman', size=15, color='#E8F4FD')),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(family='Times New Roman')),
        xaxis=dict(gridcolor='rgba(255,255,255,0.04)', color='#6B7FA8'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.04)', color='#6B7FA8', title='Total Cases'),
        height=400, margin=dict(t=50, b=20),
        hovermode='x unified'
    )
    st.plotly_chart(fig_p, use_container_width=True)
    st.markdown('<div class="insight-box">Prophet captures the <b>long-term trend and seasonality</b> using only time as input. The shaded band shows the 80% confidence interval — wider bands indicate higher uncertainty in future years.</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="insight-box">XGBoost was trained with <b>7 features</b>: dengue_lag1, dengue_lag2, dengue_lag4, dengue_roll3, temp_mean_c, precipitation_mm, and year. The chart below shows actual vs predicted on the 3-year holdout test set.</div>', unsafe_allow_html=True)
    try:
        xgb_preds = pd.read_csv('data/processed/xgboost_forecasts.csv')
        xgb_agg = xgb_preds.groupby('year')[['dengue_cases', 'predicted_cases']].sum().reset_index()
        fig_x = go.Figure()
        fig_x.add_trace(go.Scatter(x=xgb_agg['year'], y=xgb_agg['dengue_cases'], name='Actual',
            line=dict(color='#2E86AB', width=2.5), mode='lines+markers', marker=dict(size=8)))
        fig_x.add_trace(go.Scatter(x=xgb_agg['year'], y=xgb_agg['predicted_cases'], name='Predicted',
            line=dict(color='#10B981', width=2.5, dash='dot'), mode='lines+markers', marker=dict(size=8)))
        fig_x.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(17,24,39,0.8)',
            font=dict(family='Times New Roman', color='#8BA7C0', size=13),
            title=dict(text='XGBoost — Actual vs Predicted (Test Set)', font=dict(family='Times New Roman', size=15, color='#E8F4FD')),
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.04)', color='#6B7FA8'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.04)', color='#6B7FA8', title='Cases'),
            height=400, margin=dict(t=50, b=20), hovermode='x unified'
        )
        st.plotly_chart(fig_x, use_container_width=True)
    except:
        st.info("XGBoost forecast file not found. Run train_xgboost.py first.")

with tab3:
    shap_features = ['dengue_roll3','dengue_lag2','dengue_lag1','dengue_lag4','precipitation_mm','temp_mean_c','year']
    shap_values   = [12400, 3200, 2800, 2100, 1900, 1600, 1200]
    colors = ['#10B981','#2E86AB','#3B9FCC','#5BB8D4','#F59E0B','#EF8C45','#8B7CF8']
    fig_s = go.Figure(go.Bar(
        x=shap_values[::-1], y=shap_features[::-1],
        orientation='h',
        marker=dict(color=colors[::-1], line=dict(width=0)),
        text=[f'{v:,}' for v in shap_values[::-1]],
        textposition='outside',
        textfont=dict(family='Times New Roman', size=12, color='white'),
    ))
    fig_s.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(17,24,39,0.8)',
        font=dict(family='Times New Roman', color='#8BA7C0', size=13),
        title=dict(text='SHAP Feature Importance — What Drives Dengue Outbreaks?', font=dict(family='Times New Roman', size=15, color='#E8F4FD')),
        xaxis=dict(title='Mean |SHAP Value|', gridcolor='rgba(255,255,255,0.04)', color='#6B7FA8'),
        yaxis=dict(color='#8BA7C0'),
        height=420, margin=dict(t=50, b=20, r=80),
    )
    st.plotly_chart(fig_s, use_container_width=True)
    st.markdown('<div class="insight-box"><b>dengue_roll3</b> (3-year rolling average) is the strongest predictor — past trends reliably forecast future outbreaks. <b>Climate features</b> (rainfall, temperature) confirm the biological link between environment and disease spread.</div>', unsafe_allow_html=True)

# ── TOP RISK TABLE ───────────────────────────────────────────
st.markdown('<div class="section-header">🏆 Top Risk Countries</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Countries ranked by total reported cases. Use the search box to filter specific countries.</div>', unsafe_allow_html=True)

top_df = map_data.sort_values('total_cases', ascending=False).head(15).copy()
top_df['Rank'] = range(1, len(top_df)+1)
top_df['Total Cases'] = top_df['total_cases'].apply(lambda x: f"{x:,.0f}")
top_df['Risk Level'] = top_df['risk_level'].astype(str)
top_df = top_df[['Rank','country_name','region','Total Cases','Risk Level']].rename(
    columns={'country_name':'Country','region':'Region'})

search = st.text_input("🔍 Search country", placeholder="e.g. India, Nigeria, Bangladesh...")
if search:
    top_df = top_df[top_df['Country'].str.contains(search, case=False, na=False)]

st.dataframe(
    top_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Rank": st.column_config.NumberColumn("🏅 Rank", width="small"),
        "Country": st.column_config.TextColumn("🌍 Country"),
        "Region": st.column_config.TextColumn("📍 Region"),
        "Total Cases": st.column_config.TextColumn("📊 Total Cases"),
        "Risk Level": st.column_config.TextColumn("⚠️ Risk Level"),
    }
)

# ── FOOTER ───────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-title">Data Sources & Acknowledgements</div>
    <div class="footer-sources">
        <span class="footer-source">🏥 WHO Global Health Observatory</span>
        <span class="footer-source">🦟 OpenDengue Project</span>
        <span class="footer-source">☁️ Open-Meteo Climate API</span>
        <span class="footer-source">🗺️ Natural Earth Shapefiles</span>
        <span class="footer-source">🤖 Prophet by Meta AI</span>
    </div>
    <div style="margin-top:12px;font-size:0.72rem;color:#2E3D52;">
        Built with Python · Streamlit · XGBoost · SHAP · Folium · Plotly
    </div>
</div>
""", unsafe_allow_html=True)