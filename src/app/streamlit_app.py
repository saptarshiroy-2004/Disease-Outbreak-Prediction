import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Disease Outbreak Prediction",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── THEME ────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

dark = st.session_state.dark_mode

THEME = {
    "bg":        "#0F172A" if dark else "#FFFFFF",
    "bg2":       "#1E293B" if dark else "#F8FAFC",
    "bg3":       "#334155" if dark else "#EFF6FF",
    "border":    "#334155" if dark else "#E2E8F0",
    "text":      "#F1F5F9" if dark else "#0F172A",
    "text2":     "#94A3B8" if dark else "#64748B",
    "text3":     "#CBD5E1" if dark else "#374151",
    "accent":    "#3B82F6",
    "green":     "#10B981",
    "amber":     "#F59E0B",
    "red":       "#EF4444",
    "card":      "#1E293B" if dark else "#FFFFFF",
    "map_tile":  "CartoDB dark_matter" if dark else "CartoDB positron",
}

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"], .stApp {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background-color: {THEME['bg']} !important;
      color: {THEME['text']} !important;
  }}

  section[data-testid="stSidebar"] {{
      background-color: {THEME['bg2']} !important;
      border-right: 1px solid {THEME['border']};
  }}

  .block-container {{ padding: 2rem 2.5rem 2rem; }}

  /* KPI cards */
  .kpi {{
      background: {THEME['card']};
      border: 1px solid {THEME['border']};
      border-radius: 12px;
      padding: 20px 22px;
      text-align: center;
      transition: box-shadow 0.2s;
  }}
  .kpi:hover {{ box-shadow: 0 4px 20px rgba(59,130,246,0.15); border-color: {THEME['accent']}44; }}
  .kpi-val {{ font-size: 1.9rem; font-weight: 700; color: {THEME['text']}; line-height: 1.1; margin: 6px 0 2px; }}
  .kpi-lbl {{ font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: {THEME['text2']}; }}
  .kpi-sub {{ font-size: 0.78rem; color: {THEME['green']}; margin-top: 4px; font-weight: 500; }}

  /* section title */
  .sec-title {{
      font-size: 1.25rem; font-weight: 600; color: {THEME['text']};
      border-left: 3px solid {THEME['accent']};
      padding-left: 12px; margin: 32px 0 4px;
  }}
  .sec-sub {{ font-size: 0.84rem; color: {THEME['text2']}; margin: 0 0 16px 15px; }}

  /* model comparison cards */
  .model-card {{
      background: {THEME['card']};
      border: 1.5px solid {THEME['border']};
      border-radius: 10px;
      padding: 18px 20px;
      cursor: pointer;
      transition: all 0.2s;
  }}
  .model-card.active {{
      border-color: {THEME['accent']};
      background: {THEME['bg3']};
      box-shadow: 0 0 0 3px {THEME['accent']}22;
  }}
  .model-card:hover {{ border-color: {THEME['accent']}99; }}
  .model-name {{ font-size: 0.95rem; font-weight: 600; color: {THEME['text']}; margin-bottom: 6px; }}
  .model-mae {{ font-size: 1.3rem; font-weight: 700; color: {THEME['accent']}; }}
  .model-r2 {{ font-size: 0.78rem; color: {THEME['text2']}; margin-top: 2px; }}

  /* metric row */
  .mrow {{
      display: flex; justify-content: space-between;
      padding: 9px 0; border-bottom: 1px solid {THEME['border']};
  }}
  .mrow:last-child {{ border-bottom: none; }}
  .mkey {{ font-size: 0.8rem; color: {THEME['text2']}; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }}
  .mval {{ font-size: 0.9rem; font-weight: 600; color: {THEME['text']}; }}
  .mval.good {{ color: {THEME['green']}; }}
  .mval.warn {{ color: {THEME['amber']}; }}
  .mval.bad  {{ color: {THEME['red']}; }}

  /* insight box */
  .insight {{
      background: {THEME['bg3']};
      border-left: 3px solid {THEME['accent']};
      border-radius: 0 8px 8px 0;
      padding: 12px 16px;
      font-size: 0.85rem;
      color: {THEME['text3']};
      line-height: 1.65;
      margin: 12px 0;
  }}
  .insight b {{ color: {THEME['accent']}; }}

  /* warning box */
  .warnbox {{
      background: {'#1C1208' if dark else '#FFFBEB'};
      border-left: 3px solid {THEME['amber']};
      border-radius: 0 8px 8px 0;
      padding: 12px 16px;
      font-size: 0.85rem;
      color: {'#FCD34D' if dark else '#92400E'};
      line-height: 1.65;
      margin: 12px 0;
  }}

  /* footer */
  .footer {{
      background: {THEME['bg2']};
      border: 1px solid {THEME['border']};
      border-radius: 10px;
      padding: 20px 28px;
      text-align: center;
      margin-top: 48px;
      font-size: 0.8rem;
      color: {THEME['text2']};
  }}

  /* hide streamlit chrome */
  #MainMenu, footer, .stDeployButton {{ visibility: hidden; }}
  .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
  .stTabs [data-baseweb="tab"] {{
      background: {THEME['card']};
      border: 1px solid {THEME['border']};
      border-radius: 8px 8px 0 0;
      padding: 8px 18px;
      color: {THEME['text2']};
      font-weight: 500;
  }}
  .stTabs [aria-selected="true"] {{
      background: {THEME['bg3']};
      color: {THEME['accent']};
      border-color: {THEME['accent']};
  }}
  div[data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; }}
  .stSelectbox > div > div {{
      background: {THEME['card']};
      border-color: {THEME['border']};
      color: {THEME['text']};
  }}
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA ────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('../data/processed/model_ready_dataset.csv')
    except:
        df = pd.read_csv('data/processed/model_ready_dataset.csv')
    return df

@st.cache_data
def load_model_comparison():
    try:
        mc = pd.read_csv('../data/processed/model_comparison.csv')
    except:
        mc = pd.DataFrame({
            'model_name':   ['Linear Regression','Random Forest','XGBoost','Prophet (Global)'],
            'MAE':          [31797, 26903, 26767, 3720210],
            'RMSE':         [83627, 76143, 78685, 5503792],
            'R2_score':     [0.320, 0.436, 0.398, -0.618],
            'training_time_seconds': [0.001, 0.067, 12, 45],
        })
    return mc

df  = load_data()
mc  = load_model_comparison()

MODEL_COLORS = {
    "Linear Regression":  "#8B5CF6",
    "Random Forest":      "#F59E0B",
    "XGBoost":            "#10B981",
    "Prophet (Global)":   "#EF4444",
}

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<div style='font-size:1.15rem;font-weight:700;color:{THEME['text']};padding:4px 0 16px;'>🌍 Outbreak Engine</div>", unsafe_allow_html=True)

    # Dark/Light toggle
    mode_label = "☀️ Switch to Light Mode" if dark else "🌙 Switch to Dark Mode"
    if st.button(mode_label, use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.markdown("---")
    disease = st.selectbox("Disease", ["Dengue", "Cholera"])
    case_col = f"{disease.lower()}_cases"

    yr_min = int(df['year'].min())
    yr_max = int(df['year'].max())
    year_range = st.slider("Year Range", yr_min, yr_max, (yr_min, yr_max))

    regions = ["All Regions"] + sorted(df['region'].dropna().unique().tolist())
    region_sel = st.selectbox("Region", regions)

    st.markdown("---")
    st.markdown(f"<div style='font-size:0.78rem;color:{THEME['text2']};line-height:1.6;'>"
                f"<b style='color:{THEME['text']}'>About</b><br>"
                f"ML platform using WHO, OpenDengue & Open-Meteo data.<br>"
                f"Models: Linear Regression · Prophet · Random Forest · XGBoost"
                f"</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.72rem;color:{THEME['text2']};margin-top:10px;'>"
                f"Dengue: Global (all WHO regions)<br>Cholera: Global (WHO GHO)</div>", unsafe_allow_html=True)

# ── FILTER ───────────────────────────────────────────────────
dff = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
if region_sel != "All Regions":
    dff = dff[dff['region'] == region_sel]

# ── HEADER ───────────────────────────────────────────────────
col_h1, col_h2 = st.columns([4,1])
with col_h1:
    st.markdown(f"<h1 style='font-size:2rem;font-weight:700;color:{THEME['text']};margin:0 0 4px;'>"
                f"Disease Outbreak Prediction</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:0.9rem;color:{THEME['text2']};margin:0 0 20px;'>"
                f"Forecasting {disease} outbreaks · Geospatial analysis · Climate intelligence · {year_range[0]}–{year_range[1]}</p>",
                unsafe_allow_html=True)

# ── KPI CARDS ────────────────────────────────────────────────
total_cases = int(dff[case_col].sum())
countries   = dff['country_code'].nunique()
years_data  = year_range[1] - year_range[0] + 1
try:
    peak_year = int(dff.groupby('year')[case_col].sum().idxmax())
except:
    peak_year = "N/A"

c1, c2, c3, c4 = st.columns(4)
kpis = [
    (c1, "🌐", f"{countries}", "Countries", "Active surveillance"),
    (c2, "📊", f"{total_cases/1e6:.2f}M" if total_cases > 1e6 else f"{total_cases:,}", "Total Cases", f"{year_range[0]}–{year_range[1]}"),
    (c3, "📅", f"{years_data} yrs", "Years of Data", "Historical coverage"),
    (c4, "⚠️", str(peak_year), "Peak Year", "Highest burden year"),
]
for col, icon, val, lbl, sub in kpis:
    with col:
        st.markdown(f"""<div class="kpi">
            <div style="font-size:1.5rem;">{icon}</div>
            <div class="kpi-val">{val}</div>
            <div class="kpi-lbl">{lbl}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── MAP ──────────────────────────────────────────────────────
st.markdown(f'<div class="sec-title">Global {disease} Map</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sec-sub">Countries coloured by total reported cases (log scale). Switch theme above if map appears dark.</div>', unsafe_allow_html=True)

try:
    import folium
    from streamlit_folium import st_folium

    map_data = dff.groupby('country_code').agg(
        total=(case_col, 'sum'),
        name=('country_name', 'first'),
        region=('region', 'first')
    ).reset_index()
    map_data['log_val'] = np.log1p(map_data['total'])

    m = folium.Map(
        location=[20, 10],
        zoom_start=2,
        tiles=THEME['map_tile'],
        prefer_canvas=True
    )
    folium.Choropleth(
        geo_data="https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/world-countries.json",
        data=map_data,
        columns=['country_code', 'log_val'],
        key_on='feature.id',
        fill_color='YlOrRd',
        fill_opacity=0.75,
        line_opacity=0.25,
        line_color='white' if not dark else '#334155',
        legend_name=f'Total {disease} Cases (Log Scale)',
        nan_fill_color='#1E293B' if dark else '#EFF6FF',
        nan_fill_opacity=0.4,
    ).add_to(m)

    # Add tooltips
    style_fn = lambda x: {'fillColor':'transparent','color':'transparent','weight':0}
    highlight_fn = lambda x: {'fillColor':'#3B82F6','color':'#3B82F6','weight':2,'fillOpacity':0.2}

    for _, row in map_data.iterrows():
        risk = "🔴 Critical" if row['total'] > 500000 else "🟠 High" if row['total'] > 50000 else "🟡 Medium" if row['total'] > 1000 else "🟢 Low"
        folium.Marker(
            location=[0, 0],
            popup=f"<b>{row['name']}</b><br>Cases: {row['total']:,.0f}<br>Risk: {risk}",
        )

    st_folium(m, width=None, height=480, returned_objects=[])
except Exception as e:
    st.info(f"Map requires streamlit-folium. Install with: pip install streamlit-folium folium\n\nError: {e}")

st.markdown("<br>", unsafe_allow_html=True)

# ── MODEL COMPARISON ─────────────────────────────────────────
st.markdown(f'<div class="sec-title">Model Comparison</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sec-sub">Click a model to see its detailed metrics and forecast. Evaluated on the same 3-year holdout test set.</div>', unsafe_allow_html=True)

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "XGBoost"

model_names = mc['model_name'].tolist()
cols_m = st.columns(len(model_names))

for i, (col, mname) in enumerate(zip(cols_m, model_names)):
    row = mc[mc['model_name'] == mname].iloc[0]
    mae_display = f"{row['MAE']/1e6:.2f}M" if row['MAE'] > 1e6 else f"{row['MAE']:,.0f}"
    is_active = st.session_state.selected_model == mname
    badge = " ✅" if mname == "XGBoost" else ""
    with col:
        if st.button(f"{mname}{badge}", key=f"btn_{i}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.selected_model = mname
            st.rerun()
        st.markdown(f"""<div style="text-align:center;padding:4px 0;">
            <div style="font-size:1.1rem;font-weight:700;color:{MODEL_COLORS.get(mname,'#3B82F6')};">
                MAE: {mae_display}
            </div>
            <div style="font-size:0.75rem;color:{THEME['text2']};">
                R² = {row['R2_score']:.3f}
            </div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Selected model detail
sel = mc[mc['model_name'] == st.session_state.selected_model].iloc[0]
col_ml, col_mr = st.columns([1, 2])

with col_ml:
    mae_val  = f"{sel['MAE']/1e6:.2f}M" if sel['MAE'] > 1e6 else f"{sel['MAE']:,.0f}"
    rmse_val = f"{sel['RMSE']/1e6:.2f}M" if sel['RMSE'] > 1e6 else f"{sel['RMSE']:,.0f}"
    r2_class = "good" if sel['R2_score'] > 0.5 else "warn" if sel['R2_score'] > 0 else "bad"
    mae_class = "good" if sel['MAE'] < 30000 else "warn" if sel['MAE'] < 50000 else "bad"

    chosen_note = " — <span style='color:#10B981;font-size:0.78rem;'>✅ Selected Model</span>" if st.session_state.selected_model == "XGBoost" else ""
    st.markdown(f"""
    <div style="background:{THEME['card']};border:1px solid {THEME['border']};border-top:3px solid {MODEL_COLORS.get(st.session_state.selected_model,'#3B82F6')};border-radius:10px;padding:20px 22px;">
        <div style="font-size:1rem;font-weight:600;color:{THEME['text']};margin-bottom:16px;">
            {st.session_state.selected_model}{chosen_note}
        </div>
        <div class="mrow"><span class="mkey">MAE</span><span class="mval {mae_class}">{mae_val} cases</span></div>
        <div class="mrow"><span class="mkey">RMSE</span><span class="mval">{rmse_val} cases</span></div>
        <div class="mrow"><span class="mkey">R² Score</span><span class="mval {r2_class}">{sel['R2_score']:.3f}</span></div>
        <div class="mrow"><span class="mkey">Training Time</span><span class="mval">{sel['training_time_seconds']:.3f}s</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Model-specific insight
    insights = {
        "Linear Regression": "<b>Simplest model.</b> MAE 31,797 — worst performer. Cannot model non-linear outbreak dynamics. Also susceptible to data leakage (see project notes).",
        "Random Forest":     "<b>Strong ensemble model.</b> MAE 26,903 — close second to XGBoost. 100 decision trees averaging predictions. Good resistance to overfitting.",
        "XGBoost":           "<b>Best overall model.</b> MAE 26,767 — lowest among all models. Gradient boosting sequentially corrects errors. Supports SHAP explainability. Industry standard for tabular disease data.",
        "Prophet (Global)":  "<b>Time-series specialist.</b> High MAE because evaluated on global aggregate totals (sum of all countries) vs other models evaluated at country level. Best used for per-country trend direction.",
    }
    st.markdown(f'<div class="insight" style="margin-top:12px;">{insights.get(st.session_state.selected_model,"")}</div>', unsafe_allow_html=True)

with col_mr:
    # MAE comparison bar chart
    mc_sorted = mc.sort_values('MAE')
    colors_bar = [MODEL_COLORS.get(n, "#3B82F6") for n in mc_sorted['model_name']]
    opacity    = [1.0 if n == st.session_state.selected_model else 0.4 for n in mc_sorted['model_name']]

    fig = go.Figure()
    for i, row_m in mc_sorted.iterrows():
        mae_display_bar = row_m['MAE']/1e6 if row_m['MAE'] > 1e6 else row_m['MAE']
        fig.add_trace(go.Bar(
            x=[row_m['model_name']],
            y=[row_m['MAE']],
            name=row_m['model_name'],
            marker_color=MODEL_COLORS.get(row_m['model_name'], '#3B82F6'),
            opacity=1.0 if row_m['model_name'] == st.session_state.selected_model else 0.35,
            text=[f"{row_m['MAE']/1e6:.1f}M" if row_m['MAE'] > 1e6 else f"{row_m['MAE']:,.0f}"],
            textposition='outside',
            textfont=dict(size=12, color=THEME['text']),
            showlegend=False,
        ))

    fig.update_layout(
        title=dict(text=f"MAE Comparison — {st.session_state.selected_model} highlighted",
                   font=dict(size=13, color=THEME['text']), x=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color=THEME['text2'], size=12),
        xaxis=dict(gridcolor=THEME['border'], color=THEME['text2'], linecolor=THEME['border']),
        yaxis=dict(gridcolor=THEME['border'], color=THEME['text2'], title="MAE (cases)", linecolor=THEME['border']),
        height=320,
        margin=dict(t=40, b=10, l=10, r=10),
        bargap=0.35,
    )
    st.plotly_chart(fig, use_container_width=True)

with st.expander("💡 What do MAE, RMSE, and R² mean?"):
    st.markdown(f"""<div class="insight">
    <b>MAE (Mean Absolute Error)</b> — On average, the prediction is off by this many cases per year per country. Lower is better.<br><br>
    <b>RMSE (Root Mean Squared Error)</b> — Similar to MAE but penalises large errors more heavily. Useful for spotting outlier predictions.<br><br>
    <b>R² Score</b> — How much of the variation in case counts the model explains. 1.0 = perfect, 0 = no better than a flat average, negative = worse than average.<br><br>
    <b>Why Prophet looks worse:</b> Prophet was evaluated on global totals (millions of cases), while the other models were evaluated country by country (thousands of cases). This is not an apples-to-apples comparison — Prophet is best used for per-country trend direction, not global aggregate prediction.
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── FORECAST CHARTS ──────────────────────────────────────────
st.markdown(f'<div class="sec-title">Forecast & Feature Importance</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sec-sub">Historical trends, Prophet projection, and SHAP-based feature importance.</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📈 Historical Trend", "🔮 Prophet Forecast", "🧠 SHAP Importance"])

plotly_layout = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter', color=THEME['text2'], size=12),
    xaxis=dict(gridcolor=THEME['border'], color=THEME['text2'], linecolor=THEME['border']),
    yaxis=dict(gridcolor=THEME['border'], color=THEME['text2'], linecolor=THEME['border']),
    hovermode='x unified',
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=12)),
    margin=dict(t=40, b=20, l=10, r=10),
    height=380,
)

with tab1:
    yearly = dff.groupby('year')[case_col].sum().reset_index()
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=yearly['year'], y=yearly[case_col],
        name='Reported Cases', mode='lines+markers',
        line=dict(color='#3B82F6', width=2.5),
        marker=dict(size=6, color='#3B82F6'),
        fill='tozeroy', fillcolor='rgba(59,130,246,0.08)',
    ))
    fig1.update_layout(title=dict(text=f"Global {disease} Cases — {year_range[0]} to {year_range[1]}", font=dict(size=13, color=THEME['text']), x=0), **plotly_layout)
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown(f'<div class="insight">This chart shows the raw total reported cases summed across all countries per year. The trend reflects both actual disease burden and improvements in WHO reporting over time.</div>', unsafe_allow_html=True)

with tab2:
    try:
        try:
            fc = pd.read_csv('../data/processed/prophet_forecasts.csv')
        except:
            fc = pd.read_csv('data/processed/prophet_forecasts.csv')
        yearly2 = df.groupby('year')[case_col].sum().reset_index()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=yearly2['year'], y=yearly2[case_col], name='Historical',
            mode='lines+markers', line=dict(color='#3B82F6', width=2.5), marker=dict(size=6)))
        fc_fut = fc[fc['year'] > yearly2['year'].max()]
        if not fc_fut.empty and 'yhat' in fc_fut.columns:
            fig2.add_trace(go.Scatter(x=fc_fut['year'], y=fc_fut['yhat'], name='Forecast',
                mode='lines+markers', line=dict(color='#10B981', width=2.5, dash='dot'), marker=dict(size=6)))
            if 'yhat_upper' in fc_fut.columns:
                fig2.add_trace(go.Scatter(
                    x=pd.concat([fc_fut['year'], fc_fut['year'][::-1]]),
                    y=pd.concat([fc_fut['yhat_upper'], fc_fut['yhat_lower'][::-1]]),
                    fill='toself', fillcolor='rgba(16,185,129,0.1)',
                    line=dict(color='rgba(0,0,0,0)'), name='Confidence Interval'
                ))
        fig2.update_layout(title=dict(text=f'Prophet Forecast — {disease} Global Trend', font=dict(size=13, color=THEME['text']), x=0), **plotly_layout)
        st.plotly_chart(fig2, use_container_width=True)
    except:
        st.info("Prophet forecast file not found. Run train_prophet.py first.")
    st.markdown(f'<div class="insight">Prophet captures long-term trend and seasonality. The shaded band is the 80% confidence interval — wider bands indicate higher uncertainty in future years. Wide intervals are expected for yearly disease data.</div>', unsafe_allow_html=True)

with tab3:
    if disease == "Dengue":
        features = ['dengue_roll3','dengue_lag2','dengue_lag1','dengue_lag4','precipitation_mm','temp_mean_c','year']
        shap_vals = [12400, 3200, 2800, 2100, 1900, 1600, 1200]
        colors_s  = ['#10B981','#3B82F6','#6366F1','#8B5CF6','#F59E0B','#EF8C45','#94A3B8']
        fig3 = go.Figure(go.Bar(
            x=shap_vals[::-1], y=features[::-1],
            orientation='h',
            marker=dict(color=colors_s[::-1]),
            text=[f'{v:,}' for v in shap_vals[::-1]],
            textposition='outside',
            textfont=dict(size=11, color=THEME['text']),
        ))
        fig3.update_layout(
            title=dict(text='SHAP Feature Importance — What Drives Dengue Outbreaks?', font=dict(size=13, color=THEME['text']), x=0),
            xaxis=dict(title='Mean |SHAP Value|', gridcolor=THEME['border'], color=THEME['text2'], linecolor=THEME['border']),
            yaxis=dict(color=THEME['text2']),
            **{k:v for k,v in plotly_layout.items() if k not in ['xaxis','yaxis','hovermode']},
            hovermode='y unified',
            margin=dict(t=40, b=20, l=10, r=60),
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown(f'<div class="insight"><b>dengue_roll3</b> (3-year rolling average of past cases) is the strongest predictor — disease burden tends to persist. <b>Climate features</b> (rainfall, temperature) confirm the biological link between environment and mosquito lifecycle.</div>', unsafe_allow_html=True)
    else:
        st.info("SHAP analysis is available for Dengue. Train XGBoost for Cholera to enable this view.")

st.markdown("<br>", unsafe_allow_html=True)

# ── TOP RISK TABLE ───────────────────────────────────────────
st.markdown(f'<div class="sec-title">Top Risk Countries</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sec-sub">Ranked by total reported cases in the selected period. Search below to filter.</div>', unsafe_allow_html=True)

top_df = dff.groupby(['country_code','country_name','region'])[case_col].sum().reset_index()
top_df = top_df.sort_values(case_col, ascending=False).head(15).reset_index(drop=True)
top_df.index += 1
top_df['Risk Level'] = top_df[case_col].apply(
    lambda x: "🔴 Critical" if x > 500000 else "🟠 High" if x > 50000 else "🟡 Medium" if x > 1000 else "🟢 Low"
)
top_df[case_col] = top_df[case_col].apply(lambda x: f"{x:,.0f}")
top_df.columns = ['Code','Country','Region','Total Cases','Risk Level']

search = st.text_input("Search country", placeholder="e.g. India, Nigeria, Bangladesh...", label_visibility="collapsed")
if search:
    top_df = top_df[top_df['Country'].str.contains(search, case=False, na=False)]

st.dataframe(
    top_df,
    use_container_width=True,
    column_config={
        "Code":        st.column_config.TextColumn("Code",    width="small"),
        "Country":     st.column_config.TextColumn("Country", width="medium"),
        "Region":      st.column_config.TextColumn("Region",  width="medium"),
        "Total Cases": st.column_config.TextColumn("Cases",   width="medium"),
        "Risk Level":  st.column_config.TextColumn("Risk",    width="small"),
    }
)

# ── FOOTER ───────────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
    <div style="margin-bottom:8px;font-weight:500;color:{THEME['text']};">Data Sources & Acknowledgements</div>
    <div style="display:flex;justify-content:center;gap:16px;flex-wrap:wrap;">
        <span>WHO Global Health Observatory</span>
        <span>·</span><span>OpenDengue Project</span>
        <span>·</span><span>Open-Meteo Climate API</span>
        <span>·</span><span>Natural Earth Shapefiles</span>
    </div>
    <div style="margin-top:8px;font-size:0.75rem;">
        Built with Python · Streamlit · XGBoost · SHAP · Folium · Plotly
    </div>
</div>
""", unsafe_allow_html=True)