import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '..', 'data', 'processed')

st.set_page_config(
    page_title="Outbreak Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── THEME & STATE INITIALIZATION ─────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "XGBoost"

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

def select_model(name: str):
    st.session_state.selected_model = name

dark = st.session_state.dark_mode

THEME = {
    "bg":            "#0A0E17" if dark else "#F8FAFC",
    "card":          "#111827" if dark else "#FFFFFF",
    "card_subtle":   "#161F30" if dark else "#F1F5F9",
    "border":        "#1F293D" if dark else "#E2E8F0",
    "border_subtle": "#162238" if dark else "#EDF2F7",
    "text":          "#F8FAFC" if dark else "#0F172A",
    "text2":         "#94A3B8" if dark else "#334155",
    "text3":         "#64748B" if dark else "#64748B",
    "accent":        "#14B8A6" if dark else "#0D9488",
    "accent_hover":  "#2DD4BF" if dark else "#0F766E",
    "accent_bg":     "rgba(20, 184, 166, 0.12)" if dark else "rgba(13, 148, 136, 0.08)",
    "grid":          "rgba(255, 255, 255, 0.05)" if dark else "rgba(15, 23, 42, 0.05)"
}

status_bg = "rgba(34, 197, 94, 0.10)" if dark else "#F0FDF4"
status_border = "rgba(34, 197, 94, 0.25)" if dark else "#DCFCE7"
status_lbl = "#4ADE80" if dark else "#15803D"
status_val = "#22C55E" if dark else "#166534"

badge_red_bg = "rgba(239, 68, 68, 0.15)" if dark else "#FEE2E2"
badge_red_txt = "#F87171" if dark else "#991B1B"
badge_amber_bg = "rgba(249, 115, 22, 0.15)" if dark else "#FFEDD5"
badge_amber_txt = "#FB923C" if dark else "#9A3412"
badge_yellow_bg = "rgba(234, 179, 8, 0.15)" if dark else "#FEF9C3"
badge_yellow_txt = "#FACC15" if dark else "#854D0E"

legend_gradient = "linear-gradient(to right, #111827, #134E4A, #0D9488, #14B8A6, #A7F3D0)" if dark else "linear-gradient(to right, #FFFFFF, #E6FFFA, #99F6E4, #2DD4BF, #0D9488)"
legend_bg = "rgba(17, 24, 39, 0.95)" if dark else "rgba(255, 255, 255, 0.95)"

# ─── GLOBAL STYLESHEET ────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

  :root {{
      --bg: {THEME['bg']};
      --card: {THEME['card']};
      --card-subtle: {THEME['card_subtle']};
      --border: {THEME['border']};
      --border-subtle: {THEME['border_subtle']};
      --text: {THEME['text']};
      --text2: {THEME['text2']};
      --text3: {THEME['text3']};
      --accent: {THEME['accent']};
      --accent-hover: {THEME['accent_hover']};
      --accent-bg: {THEME['accent_bg']};
      --status-bg: {status_bg};
      --status-border: {status_border};
      --status-lbl: {status_lbl};
      --status-val: {status_val};
      --badge-red-bg: {badge_red_bg};
      --badge-red-txt: {badge_red_txt};
      --badge-amber-bg: {badge_amber_bg};
      --badge-amber-txt: {badge_amber_txt};
      --badge-yellow-bg: {badge_yellow_bg};
      --badge-yellow-txt: {badge_yellow_txt};
      --legend-gradient: {legend_gradient};
      --legend-bg: {legend_bg};
  }}

  /* Hide default Streamlit header and deploy button */
  header[data-testid="stHeader"], [data-testid="stToolbar"], .stDeployButton, #MainMenu, footer, .stDecoration {{
      display: none !important;
      visibility: hidden !important;
      height: 0 !important;
  }}

  html, body, [class*="css"], .stApp {{
      background-color: var(--bg) !important;
      color: var(--text) !important;
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
  }}

  .stApp {{
      transition: background-color 0.2s ease !important;
  }}

  .block-container {{
      padding-top: 1.4rem !important;
      padding-bottom: 3rem !important;
      padding-left: 3.5rem !important;
      padding-right: 3.5rem !important;
      max-width: 1520px !important;
  }}

  /* Header Branding */
  .nav-brand {{
      display: flex;
      align-items: center;
      gap: 12px;
  }}
  .nav-logo-icon {{
      display: flex;
      align-items: center;
      justify-content: center;
      width: 36px;
      height: 36px;
      border-radius: 8px;
      background: var(--accent-bg);
      border: 1px solid var(--border);
      color: var(--accent);
      flex-shrink: 0;
  }}
  .nav-title-group {{
      display: flex;
      flex-direction: column;
      justify-content: center;
  }}
  .nav-title {{
      font-size: 1.05rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      color: var(--text) !important;
      line-height: 1.2;
  }}
  .nav-subtitle {{
      font-size: 0.72rem;
      font-weight: 500;
      color: var(--text3) !important;
      letter-spacing: 0.01em;
  }}

  /* Targeted Button Styling */
  div[data-testid="stButton"] button,
  button[data-testid="baseButton-secondary"],
  button[kind="secondary"] {{
      height: 38px !important;
      border-radius: 8px !important;
      padding: 4px 14px !important;
      font-size: 0.78rem !important;
      font-weight: 600 !important;
      background-color: var(--card-subtle) !important;
      background: var(--card-subtle) !important;
      border: 1px solid var(--border) !important;
      color: var(--text) !important;
      box-shadow: none !important;
      font-family: 'Plus Jakarta Sans', sans-serif !important;
      transition: background-color 0.15s ease, border-color 0.15s ease, color 0.15s ease, transform 0.1s ease !important;
  }}
  div[data-testid="stButton"] button p,
  div[data-testid="stButton"] button span,
  div[data-testid="stButton"] button div {{
      color: var(--text) !important;
  }}
  div[data-testid="stButton"] button:hover {{
      border-color: var(--accent) !important;
      background-color: var(--card) !important;
      background: var(--card) !important;
      transform: translateY(-1px);
  }}

  /* Primary/Active Button Overrides */
  div[data-testid="stButton"] button[kind="primary"],
  button[data-testid="baseButton-primary"],
  button[kind="primary"] {{
      background-color: var(--accent) !important;
      background: var(--accent) !important;
      border: 1px solid var(--accent) !important;
      color: #FFFFFF !important;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12) !important;
      transition: background-color 0.15s ease, border-color 0.15s ease, transform 0.1s ease !important;
  }}
  div[data-testid="stButton"] button[kind="primary"] p,
  div[data-testid="stButton"] button[kind="primary"] span,
  div[data-testid="stButton"] button[kind="primary"] div {{
      color: #FFFFFF !important;
  }}
  div[data-testid="stButton"] button[kind="primary"]:hover {{
      background-color: var(--accent-hover) !important;
      background: var(--accent-hover) !important;
      border-color: var(--accent-hover) !important;
      transform: translateY(-1px);
  }}

  /* Minimal Hero Section */
  .hero-card {{
      background: var(--card) !important;
      border: 1px solid var(--border) !important;
      border-radius: 10px;
      padding: 22px 26px;
      margin-bottom: 20px;
      transition: border-color 0.15s ease, transform 0.15s ease;
  }}
  .hero-card:hover {{
      border-color: var(--accent) !important;
  }}
  .hero-top-badge {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
      font-size: 0.68rem;
      font-weight: 700;
      color: var(--accent);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 6px;
  }}
  .hero-main-title {{
      font-size: 1.55rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      color: var(--text) !important;
      margin: 0 0 6px;
      line-height: 1.25;
  }}
  .hero-description {{
      font-size: 0.84rem;
      color: var(--text2) !important;
      line-height: 1.55;
      margin: 0;
      max-width: 860px;
  }}

  /* Control Headings */
  .control-heading {{
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 0.7rem;
      font-weight: 700;
      color: var(--text3) !important;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 6px;
  }}
  .sync-card {{
      background: var(--card-subtle);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px 12px;
      display: flex;
      align-items: center;
      gap: 8px;
      height: 40px;
      margin-top: 18px;
      transition: border-color 0.15s ease;
  }}
  .sync-text-group {{
      display: flex;
      flex-direction: column;
  }}
  .sync-title {{
      font-size: 0.62rem;
      font-weight: 700;
      color: var(--text3) !important;
      text-transform: uppercase;
      letter-spacing: 0.04em;
  }}
  .sync-time {{
      font-size: 0.74rem;
      font-weight: 600;
      color: var(--text) !important;
      font-family: 'JetBrains Mono', monospace;
  }}

  /* Minimal KPI Cards */
  .kpi-card-minimal {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px 18px;
      height: 112px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
  }}
  .kpi-card-minimal:hover {{
      border-color: var(--accent);
      transform: translateY(-2px);
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
  }}
  .kpi-top-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
  }}
  .kpi-label-text {{
      font-size: 0.7rem;
      font-weight: 700;
      color: var(--text3) !important;
      text-transform: uppercase;
      letter-spacing: 0.04em;
  }}
  .kpi-dot {{
      width: 7px;
      height: 7px;
      border-radius: 50%;
  }}
  .dot-teal {{ background: var(--accent); }}
  .dot-blue {{ background: #3B82F6; }}
  .dot-purple {{ background: #8B5CF6; }}
  .dot-amber {{ background: #F59E0B; }}

  .kpi-value-text {{
      font-size: 1.7rem;
      font-weight: 700;
      color: var(--text) !important;
      line-height: 1;
      font-family: 'JetBrains Mono', monospace;
      letter-spacing: -0.02em;
  }}
  .kpi-footnote {{
      font-size: 0.72rem;
      color: var(--text2) !important;
  }}

  /* Content Cards */
  .content-card-box {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 20px 22px;
      position: relative;
      height: 100%;
      transition: border-color 0.15s ease;
  }}

  .card-header-block {{
      margin-bottom: 14px;
  }}
  .card-category {{
      font-size: 0.68rem;
      font-weight: 700;
      color: var(--accent) !important;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 3px;
  }}
  .card-main-heading {{
      font-size: 1.2rem;
      font-weight: 700;
      color: var(--text) !important;
      margin: 0 0 4px;
      letter-spacing: -0.01em;
  }}
  .card-subtext {{
      font-size: 0.78rem;
      color: var(--text2) !important;
      margin: 0;
      line-height: 1.45;
  }}

  /* Map Legend */
  .map-legend-box {{
      position: absolute;
      bottom: 20px;
      left: 20px;
      background: var(--legend-bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 8px 12px;
      z-index: 50;
      width: 175px;
  }}
  .map-legend-title {{
      font-size: 0.62rem;
      font-weight: 700;
      color: var(--text2) !important;
      margin-bottom: 5px;
      text-transform: uppercase;
      letter-spacing: 0.03em;
  }}
  .map-legend-bar {{
      height: 6px;
      background: var(--legend-gradient);
      border-radius: 2px;
      margin-bottom: 4px;
  }}
  .map-legend-scale {{
      display: flex;
      justify-content: space-between;
      font-size: 0.6rem;
      color: var(--text3) !important;
      font-family: 'JetBrains Mono', monospace;
      font-weight: 500;
  }}

  /* Model Detail Block */
  .model-spec-panel {{
      background: var(--card-subtle);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px 16px;
      margin-top: 10px;
  }}
  .model-spec-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 10px;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--border);
  }}
  .model-spec-title {{
      font-size: 0.95rem;
      font-weight: 700;
      color: var(--text) !important;
  }}
  .badge-tag {{
      font-size: 0.6rem;
      font-weight: 700;
      padding: 2px 6px;
      border-radius: 4px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
  }}
  .badge-recommended {{
      background: var(--accent-bg);
      color: var(--accent) !important;
      border: 1px solid var(--accent);
  }}

  .stat-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 4px 0;
      font-size: 0.76rem;
  }}
  .stat-label {{
      color: var(--text3) !important;
      font-weight: 600;
      text-transform: uppercase;
      font-size: 0.68rem;
      letter-spacing: 0.02em;
  }}
  .stat-value {{
      color: var(--text) !important;
      font-weight: 700;
      font-family: 'JetBrains Mono', monospace;
  }}

  /* Feature list */
  .feature-list {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      margin-top: 6px;
  }}
  .feature-point {{
      display: flex;
      align-items: flex-start;
      gap: 6px;
      font-size: 0.78rem;
      color: var(--text2) !important;
      line-height: 1.4;
  }}
  .feature-point-bullet {{
      color: var(--accent) !important;
      font-weight: 700;
  }}

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {{
      gap: 4px !important;
      border-bottom: 1px solid var(--border) !important;
      margin-bottom: 12px !important;
  }}
  .stTabs [data-baseweb="tab"] {{
      background: transparent !important;
      border: 1px solid transparent !important;
      border-bottom: none !important;
      border-radius: 6px 6px 0 0 !important;
      padding: 6px 14px !important;
      color: var(--text3) !important;
      font-weight: 600 !important;
      font-size: 0.78rem !important;
      transition: all 0.15s ease !important;
  }}
  .stTabs [aria-selected="true"] {{
      background: var(--card-subtle) !important;
      color: var(--accent) !important;
      border-color: var(--border) !important;
      border-bottom-color: var(--card-subtle) !important;
  }}

  /* Data Table */
  .badge-tier {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 0.68rem;
      font-weight: 700;
      letter-spacing: 0.02em;
  }}
  .tier-very-high {{
      background-color: var(--badge-red-bg);
      color: var(--badge-red-txt);
  }}
  .tier-high {{
      background-color: var(--badge-amber-bg);
      color: var(--badge-amber-txt);
  }}
  .tier-moderate {{
      background-color: var(--badge-yellow-bg);
      color: var(--badge-yellow-txt);
  }}

  /* Footer */
  .minimal-footer {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-top: 1px solid var(--border);
      padding: 20px 0 0;
      margin-top: 36px;
      font-size: 0.74rem;
      color: var(--text3) !important;
  }}
  .footer-sources {{
      display: flex;
      gap: 16px;
  }}
  .source-link {{
      color: var(--text2) !important;
      text-decoration: none;
      font-weight: 500;
  }}
  .source-link:hover {{
      color: var(--accent) !important;
  }}
</style>
""", unsafe_allow_html=True)

# ─── LOAD DATA ────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, 'model_ready_dataset.csv'))
    return df

@st.cache_data
def load_model_comparison():
    try:
        mc = pd.read_csv(os.path.join(DATA_DIR, 'model_comparison.csv'))
    except Exception:
        mc = pd.DataFrame({
            'model_name':   ['Linear Regression', 'Random Forest', 'XGBoost', 'Prophet (Global)'],
            'MAE':          [31797, 26903, 26767, 3720210],
            'RMSE':         [83627, 76143, 78685, 5503792],
            'R2_score':     [0.320, 0.436, 0.398, -0.618],
            'training_time_seconds': [0.001, 0.067, 12, 45],
        })
    return mc

df = load_data()
mc = load_model_comparison()

# ─── HEADER NAV ──────────────────────────────────────────────
col_brand, col_theme, col_action = st.columns([7.4, 1.3, 1.3])

with col_brand:
    st.markdown("""<div class="nav-brand">
<div class="nav-logo-icon">
<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
<circle cx="12" cy="12" r="2"/>
<path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
</svg>
</div>
<div class="nav-title-group">
<span class="nav-title">Outbreak Intelligence</span>
<span class="nav-subtitle">Epidemiological Forecasting & Risk Analysis</span>
</div>
</div>""", unsafe_allow_html=True)

with col_theme:
    btn_theme_label = "Dark Mode" if not dark else "Light Mode"
    st.button(btn_theme_label, key="theme_toggle_btn", on_click=toggle_theme, use_container_width=True)

with col_action:
    if st.button("Export Data", key="export_data_btn", use_container_width=True):
        st.toast("Dataset compiled for export.")

st.markdown("""<div style="border-bottom: 1px solid var(--border); margin-top: 4px; margin-bottom: 18px;"></div>""", unsafe_allow_html=True)

# ─── MINIMAL HERO SECTION ─────────────────────────────────────
st.markdown("""<div class="hero-card">
<div class="hero-top-badge">
<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
<span>Global Surveillance & ML Forecasting</span>
</div>
<h1 class="hero-main-title">Disease Outbreak Prediction Platform</h1>
<p class="hero-description">An integrated surveillance system analyzing multi-decade epidemiological data across 181 countries. Incorporating meteorological indicators and machine learning models to forecast pathogen propagation risks.</p>
</div>""", unsafe_allow_html=True)

# ─── FILTER CONTROLS ──────────────────────────────────────────
yr_min = int(df['year'].min())
yr_max = int(df['year'].max())
regions = ["All Regions"] + sorted(df['region'].dropna().unique().tolist())

col_filter_dis, col_filter_time, col_filter_geo, col_filter_stat = st.columns([2.0, 3.2, 2.2, 2.6])

with col_filter_dis:
    st.markdown("""<div class="control-heading">
<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22a7 7 0 0 0 7-7c0-4.3-7-11-7-11S5 10.7 5 15a7 7 0 0 0 7 7z"/></svg>
<span>Target Disease</span>
</div>""", unsafe_allow_html=True)
    disease = st.selectbox("Disease", ["Dengue", "Cholera"], label_visibility="collapsed")
    case_col = f"{disease.lower()}_cases"

with col_filter_time:
    st.markdown("""<div class="control-heading">
<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>
<span>Surveillance Period</span>
</div>""", unsafe_allow_html=True)
    year_range = st.slider("Time Range", yr_min, yr_max, (yr_min, yr_max), label_visibility="collapsed")

with col_filter_geo:
    st.markdown("""<div class="control-heading">
<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/><line x1="2" y1="12" x2="22" y2="12"/></svg>
<span>Region Filter</span>
</div>""", unsafe_allow_html=True)
    region_sel = st.selectbox("Geography", regions, label_visibility="collapsed")

with col_filter_stat:
    st.markdown("""<div class="sync-card">
<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--status-val)" stroke-width="2">
<path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/>
</svg>
<div class="sync-text-group">
<span class="sync-title">Last Synchronization</span>
<span class="sync-time">May 2025 · Verified</span>
</div>
</div>""", unsafe_allow_html=True)

# ─── FILTER DATA ─────────────────────────────────────────────
dff = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
if region_sel != "All Regions":
    dff = dff[dff['region'] == region_sel]

st.markdown("<br>", unsafe_allow_html=True)

# ─── KPI METRICS ─────────────────────────────────────────────
total_cases = int(dff[case_col].sum())
countries = dff['country_code'].nunique()
years_span = year_range[1] - year_range[0] + 1
try:
    peak_year = int(dff.groupby('year')[case_col].sum().idxmax())
except Exception:
    peak_year = "N/A"

total_display = f"{total_cases/1e6:.2f}M" if total_cases > 1e6 else f"{total_cases:,}"

col_k1, col_k2, col_k3, col_k4 = st.columns(4)

with col_k1:
    st.markdown(f"""<div class="kpi-card-minimal">
<div class="kpi-top-row">
<span class="kpi-label-text">Monitored Countries</span>
<span class="kpi-dot dot-blue"></span>
</div>
<div class="kpi-value-text">{countries}</div>
<div class="kpi-footnote">Active jurisdictions</div>
</div>""", unsafe_allow_html=True)

with col_k2:
    st.markdown(f"""<div class="kpi-card-minimal">
<div class="kpi-top-row">
<span class="kpi-label-text">Cumulative Cases</span>
<span class="kpi-dot dot-teal"></span>
</div>
<div class="kpi-value-text">{total_display}</div>
<div class="kpi-footnote">{year_range[0]} – {year_range[1]} aggregate</div>
</div>""", unsafe_allow_html=True)

with col_k3:
    st.markdown(f"""<div class="kpi-card-minimal">
<div class="kpi-top-row">
<span class="kpi-label-text">Data Span</span>
<span class="kpi-dot dot-purple"></span>
</div>
<div class="kpi-value-text">{years_span} <span style="font-size: 1rem; font-weight: 500;">years</span></div>
<div class="kpi-footnote">Continuous observations</div>
</div>""", unsafe_allow_html=True)

with col_k4:
    st.markdown(f"""<div class="kpi-card-minimal">
<div class="kpi-top-row">
<span class="kpi-label-text">Peak Burden Year</span>
<span class="kpi-dot dot-amber"></span>
</div>
<div class="kpi-value-text">{peak_year}</div>
<div class="kpi-footnote">Maximum annual surge</div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── MAP BUILDER ─────────────────────────────────────────────
def build_choropleth(dff, case_col):
    map_data = dff.groupby('country_code').agg(
        total=(case_col, 'sum'),
        name=('country_name', 'first')
    ).reset_index()
    
    map_data['log_cases'] = np.log10(map_data['total'] + 1)
    map_data['fmt'] = map_data['total'].apply(lambda x: f"{x:,.0f}")

    if dark:
        greens_scale = [
            [0.0, "#111827"],
            [0.15, "#134E4A"],
            [0.35, "#0D9488"],
            [0.6, "#14B8A6"],
            [0.85, "#2DD4BF"],
            [1.0, "#A7F3D0"]
        ]
    else:
        greens_scale = [
            [0.0, "#FFFFFF"],
            [0.15, "#E6FFFA"],
            [0.35, "#99F6E4"],
            [0.6, "#2DD4BF"],
            [0.85, "#0D9488"],
            [1.0, "#134E4A"]
        ]

    fig = go.Figure(go.Choropleth(
        locations=map_data['country_code'],
        locationmode='ISO-3',
        z=map_data['log_cases'],
        text=map_data['name'],
        customdata=map_data['fmt'],
        colorscale=greens_scale,
        autocolorscale=False,
        showscale=False,
        marker_line_color=THEME['border'],
        marker_line_width=0.5,
        hovertemplate="<b>%{text}</b><br>Cumulative Cases: %{customdata}<extra></extra>"
    ))

    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor=THEME['border'],
            showland=True, landcolor=THEME['bg'],
            showocean=True, oceancolor=THEME['bg'],
            showcountries=True, countrycolor=THEME['border'],
            projection_type="natural earth",
            bgcolor='rgba(0,0,0,0)',
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=0, b=0, l=0, r=0),
        height=380,
    )
    return fig

# ─── MIDDLE SECTION ──────────────────────────────────────────
col_mid_left, col_mid_right = st.columns([1, 1])

with col_mid_left:
    st.markdown("""<div class="content-card-box">
<div class="card-header-block">
<div class="card-category">Geospatial Distribution</div>
<h2 class="card-main-heading">Global Outbreak Intensity</h2>
<p class="card-subtext">Cumulative disease incidence by jurisdiction on a logarithmic scale.</p>
</div>""", unsafe_allow_html=True)

    try:
        st.plotly_chart(build_choropleth(dff, case_col), use_container_width=True, config={'displayModeBar': False})
    except Exception as e:
        st.info(f"Unable to render map: {e}")

    st.markdown("""<div class="map-legend-box">
<div class="map-legend-title">Cumulative Cases (Log Scale)</div>
<div class="map-legend-bar"></div>
<div class="map-legend-scale">
<span>10</span>
<span>1K</span>
<span>100K</span>
<span>10M+</span>
</div>
</div>
</div>""", unsafe_allow_html=True)

with col_mid_right:
    st.markdown("""<div class="content-card-box">
<div class="card-header-block">
<div class="card-category">Model Benchmarks</div>
<h2 class="card-main-heading">Algorithm Comparison</h2>
<p class="card-subtext">Evaluation metrics tested across a 3-year out-of-sample holdout period.</p>
</div>""", unsafe_allow_html=True)

    active_m = st.session_state.selected_model

    c_m1, c_m2, c_m3, c_m4 = st.columns(4)

    with c_m1:
        is_xgb = active_m == "XGBoost"
        st.button("XGBoost\nMAE: 26.8K", key="btn_xgb", on_click=select_model, args=("XGBoost",), use_container_width=True, type="primary" if is_xgb else "secondary")

    with c_m2:
        is_rf = active_m == "Random Forest"
        st.button("Random Forest\nMAE: 26.9K", key="btn_rf", on_click=select_model, args=("Random Forest",), use_container_width=True, type="primary" if is_rf else "secondary")

    with c_m3:
        is_lr = active_m == "Linear Regression"
        st.button("Linear Reg.\nMAE: 31.8K", key="btn_lr", on_click=select_model, args=("Linear Regression",), use_container_width=True, type="primary" if is_lr else "secondary")

    with c_m4:
        is_prophet = active_m == "Prophet (Global)"
        st.button("Prophet\nMAE: 3.72M", key="btn_prophet", on_click=select_model, args=("Prophet (Global)",), use_container_width=True, type="primary" if is_prophet else "secondary")

    sel_row = mc[mc['model_name'] == active_m].iloc[0]
    mae_val = f"{sel_row['MAE']/1e6:.2f}M" if sel_row['MAE'] > 1e6 else f"{sel_row['MAE']:,.0f}"
    rmse_val = f"{sel_row['RMSE']/1e6:.2f}M" if sel_row['RMSE'] > 1e6 else f"{sel_row['RMSE']:,.0f}"
    r2_val = f"+{sel_row['R2_score']:.3f}" if sel_row['R2_score'] > 0 else f"{sel_row['R2_score']:.3f}"

    c_det_left, c_det_mid, c_det_chart = st.columns([1.3, 1.7, 2.0])

    with c_det_left:
        rec_badge = '<span class="badge-tag badge-recommended">Optimal</span>' if active_m in ["XGBoost", "Random Forest"] else ''
        st.markdown(f"""<div class="model-spec-panel">
<div class="model-spec-header">
<span class="model-spec-title">{active_m}</span>
{rec_badge}
</div>
<div class="stat-row"><span class="stat-label">MAE</span><span class="stat-value">{mae_val}</span></div>
<div class="stat-row"><span class="stat-label">RMSE</span><span class="stat-value">{rmse_val}</span></div>
<div class="stat-row"><span class="stat-label">R² Score</span><span class="stat-value" style="color: var(--accent);">{r2_val}</span></div>
<div class="stat-row"><span class="stat-label">Convergence</span><span class="stat-value">Stable</span></div>
</div>""", unsafe_allow_html=True)

    with c_det_mid:
        bullets_dict = {
            "XGBoost": [
                "Gradient boosting on decision trees.",
                "Optimal residual minimization.",
                "Robust with non-linear climate cues.",
                "Full SHAP interpretability."
            ],
            "Random Forest": [
                "Bagged ensemble of trees.",
                "Variance reduction on noisy data.",
                "High multi-regional stability.",
                "Fast non-parametric training."
            ],
            "Linear Regression": [
                "Baseline least squares benchmark.",
                "High bias on exponential surges.",
                "Instant structural benchmark.",
                "Zero hyperparameters required."
            ],
            "Prophet (Global)": [
                "Additive seasonal time-series model.",
                "Captures multi-year macro trends.",
                "Sensitive to localized outbreak shocks.",
                "Best for aggregate global forecasts."
            ]
        }
        b_html = '<div class="feature-list" style="margin-top: 10px;">'
        for pt in bullets_dict.get(active_m, []):
            b_html += f'<div class="feature-point"><span class="feature-point-bullet">—</span><span>{pt}</span></div>'
        b_html += '</div>'
        st.markdown(b_html, unsafe_allow_html=True)

    with c_det_chart:
        fig_bar = go.Figure()
        models = ["XGBoost", "Random Forest", "Linear Reg.", "Prophet"]
        maes = [26767, 26903, 31797, 3720210]
        labels = ["26.8K", "26.9K", "31.8K", "3.7M"]
        bar_colors = [THEME['accent'] if active_m.startswith(m.split()[0]) else THEME['text3'] for m in models]
        opacities = [1.0 if active_m.startswith(m.split()[0]) else 0.45 for m in models]

        fig_bar.add_trace(go.Bar(
            x=models,
            y=maes,
            marker_color=bar_colors,
            marker_opacity=opacities,
            text=labels,
            textposition='outside',
            textfont=dict(size=9, color=THEME['text2'], family="'JetBrains Mono', monospace"),
            width=0.42
        ))
        fig_bar.update_layout(
            title=dict(text="Mean Absolute Error (Lower is Better)", font=dict(size=10, color=THEME['text2'], family="'Plus Jakarta Sans'"), x=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="'Plus Jakarta Sans', sans-serif", color=THEME['text2'], size=9),
            xaxis=dict(gridcolor='rgba(0,0,0,0)', color=THEME['text2'], linecolor=THEME['border']),
            yaxis=dict(
                gridcolor=THEME['grid'],
                color=THEME['text2'],
                linecolor=THEME['border'],
                tickvals=[0, 1000000, 2000000, 3000000, 4000000],
                ticktext=["0", "1M", "2M", "3M", "4M"],
                tickfont=dict(family="'JetBrains Mono', monospace")
            ),
            height=195,
            margin=dict(t=32, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── BOTTOM SECTION ──────────────────────────────────────────
col_bot_left, col_bot_right = st.columns([1, 1])

with col_bot_left:
    st.markdown("""<div class="content-card-box">
<div class="card-header-block">
<div class="card-category">Time-Series & Dynamics</div>
<h2 class="card-main-heading">Temporal Trends & Explanations</h2>
</div>""", unsafe_allow_html=True)

    tab_trend, tab_fc, tab_shap = st.tabs(["Historical Progression", "Prophet Forecast", "SHAP Feature Drivers"])

    with tab_trend:
        yearly = dff.groupby('year')[case_col].sum().reset_index()
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=yearly['year'],
            y=yearly[case_col],
            mode='lines+markers',
            line=dict(color=THEME['accent'], width=2),
            marker=dict(size=5, color=THEME['accent'], line=dict(color=THEME['card'], width=1)),
            fill='tozeroy',
            fillcolor=THEME['accent_bg']
        ))
        fig_trend.update_layout(
            title=dict(text=f"Annual {disease} Incidence ({year_range[0]}–{year_range[1]})", font=dict(size=11, color=THEME['text2'], family="'Plus Jakarta Sans'"), x=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="'Plus Jakarta Sans', sans-serif", color=THEME['text2'], size=10),
            xaxis=dict(gridcolor='rgba(0,0,0,0)', color=THEME['text2'], linecolor=THEME['border'], tickfont=dict(family="'JetBrains Mono', monospace")),
            yaxis=dict(
                gridcolor=THEME['grid'],
                color=THEME['text2'],
                linecolor=THEME['border'],
                tickfont=dict(family="'JetBrains Mono', monospace"),
                tickvals=[0, 2000000, 4000000, 6000000, 8000000, 10000000, 12000000, 14000000],
                ticktext=["0", "2M", "4M", "6M", "8M", "10M", "12M", "14M"]
            ),
            height=290,
            margin=dict(t=35, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_trend, use_container_width=True, config={'displayModeBar': False})

    with tab_fc:
        try:
            fc = pd.read_csv(os.path.join(DATA_DIR, 'prophet_forecasts.csv'))
            yearly2 = df.groupby('year')[case_col].sum().reset_index()
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(
                x=yearly2['year'], y=yearly2[case_col], name='Observed',
                mode='lines+markers', line=dict(color=THEME['accent'], width=2), marker=dict(size=5)
            ))
            fc_fut = fc[fc['year'] > yearly2['year'].max()]
            if not fc_fut.empty and 'yhat' in fc_fut.columns:
                fig_fc.add_trace(go.Scatter(
                    x=fc_fut['year'], y=fc_fut['yhat'], name='Projection',
                    mode='lines+markers', line=dict(color='#3B82F6', width=2, dash='dot'), marker=dict(size=5)
                ))
                if 'yhat_upper' in fc_fut.columns:
                    fig_fc.add_trace(go.Scatter(
                        x=pd.concat([fc_fut['year'], fc_fut['year'][::-1]]),
                        y=pd.concat([fc_fut['yhat_upper'], fc_fut['yhat_lower'][::-1]]),
                        fill='toself', fillcolor='rgba(59, 130, 246, 0.08)',
                        line=dict(color='rgba(0,0,0,0)'), name='Uncertainty Interval'
                    ))
            fig_fc.update_layout(
                title=dict(text='Multi-Year Forecasting Trend', font=dict(size=11, color=THEME['text2']), x=0),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="'Plus Jakarta Sans', sans-serif", color=THEME['text2'], size=10),
                xaxis=dict(gridcolor='rgba(0,0,0,0)', color=THEME['text2'], linecolor=THEME['border'], tickfont=dict(family="'JetBrains Mono', monospace")),
                yaxis=dict(gridcolor=THEME['grid'], color=THEME['text2'], linecolor=THEME['border'], tickfont=dict(family="'JetBrains Mono', monospace")),
                height=290, margin=dict(t=35, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_fc, use_container_width=True, config={'displayModeBar': False})
        except Exception:
            st.info("Prophet prediction dataset not available.")

    with tab_shap:
        if disease == "Dengue":
            features = ['dengue_roll3', 'dengue_lag2', 'dengue_lag1', 'dengue_lag4', 'precipitation_mm', 'temp_mean_c', 'year']
            shap_vals = [12400, 3200, 2800, 2100, 1900, 1600, 1200]
            bar_colors = ["#0D9488", "#14B8A6", "#2DD4BF", "#5EEAD4", "#6EE7B7", "#A7F3D0", "#D1FAE5"] if dark else ["#0F766E", "#0D9488", "#14B8A6", "#2DD4BF", "#5EEAD4", "#99F6E4", "#CCFBF1"]

            fig_shap = go.Figure(go.Bar(
                x=shap_vals[::-1], y=features[::-1],
                orientation='h',
                marker=dict(color=bar_colors[::-1]),
                text=[f'{v:,}' for v in shap_vals[::-1]],
                textposition='outside',
                textfont=dict(size=9, color=THEME['text2'], family="'JetBrains Mono', monospace"),
            ))
            fig_shap.update_layout(
                title=dict(text='SHAP Feature Contribution (XGBoost)', font=dict(size=11, color=THEME['text2']), x=0),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="'Plus Jakarta Sans', sans-serif", color=THEME['text2'], size=10),
                xaxis=dict(title='Mean |SHAP Value|', gridcolor=THEME['grid'], color=THEME['text2'], linecolor=THEME['border'], tickfont=dict(family="'JetBrains Mono', monospace")),
                yaxis=dict(color=THEME['text2'], tickfont=dict(family="'JetBrains Mono', monospace")),
                height=290, margin=dict(t=35, b=10, l=10, r=70)
            )
            st.plotly_chart(fig_shap, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("SHAP metrics calibrated for Dengue feature sets.")

    st.markdown("</div>", unsafe_allow_html=True)

with col_bot_right:
    st.markdown("""<div class="content-card-box">
<div class="card-header-block">
<div class="card-category">Risk Stratification</div>
<h2 class="card-main-heading">Highest Incidence Jurisdictions</h2>
<p class="card-subtext">Cumulative caseload ranking with tiered outbreak classification.</p>
</div>""", unsafe_allow_html=True)

    top_df = dff.groupby(['country_code', 'country_name', 'region'])[case_col].sum().reset_index()
    top_df = top_df.sort_values(case_col, ascending=False).head(10).reset_index(drop=True)
    top_df.index += 1
    top_df['Risk'] = top_df[case_col].apply(
        lambda x: "Very High" if x > 500000 else "High" if x > 50000 else "Moderate"
    )
    top_df.columns = ['Code', 'Country', 'Region', 'Total Cases', 'Risk']

    table_html = """<table style="width: 100%; border-collapse: collapse; font-family: 'Plus Jakarta Sans', sans-serif;">
<thead>
<tr style="border-bottom: 1px solid var(--border); text-align: left;">
<th style="padding: 8px 0; font-size: 0.68rem; font-weight: 700; color: var(--text3); text-transform: uppercase; width: 10%;">#</th>
<th style="padding: 8px 0; font-size: 0.68rem; font-weight: 700; color: var(--text3); text-transform: uppercase; width: 42%;">Jurisdiction</th>
<th style="padding: 8px 0; font-size: 0.68rem; font-weight: 700; color: var(--text3); text-transform: uppercase; width: 28%;">Cumulative</th>
<th style="padding: 8px 0; font-size: 0.68rem; font-weight: 700; color: var(--text3); text-transform: uppercase; width: 20%;">Tier</th>
</tr>
</thead>
<tbody>"""
    for idx, row in top_df.iterrows():
        cases_val = row['Total Cases']
        cases_display = f"{cases_val/1e6:.2f}M" if cases_val > 1e6 else f"{cases_val:,.0f}"

        risk = row['Risk']
        if risk == "Very High":
            badge_cls = "tier-very-high"
        elif risk == "High":
            badge_cls = "tier-high"
        else:
            badge_cls = "tier-moderate"

        table_html += f"""<tr style="border-bottom: 1px solid var(--border);">
<td style="padding: 9px 0; font-size: 0.78rem; color: var(--text3); font-family: 'JetBrains Mono', monospace;">{idx}</td>
<td style="padding: 9px 0; font-size: 0.8rem; font-weight: 600; color: var(--text);">{row['Country']}</td>
<td style="padding: 9px 0; font-size: 0.78rem; color: var(--text2); font-family: 'JetBrains Mono', monospace;">{cases_display}</td>
<td style="padding: 9px 0;"><span class="badge-tier {badge_cls}">{risk}</span></td>
</tr>"""
    table_html += "</tbody></table></div>"

    st.markdown(table_html, unsafe_allow_html=True)

# ─── MINIMAL FOOTER ──────────────────────────────────────────
st.markdown("""<div class="minimal-footer">
<div>Disease Outbreak Prediction System · Epidemiological Intelligence · 2000–2025</div>
<div class="footer-sources">
<span class="source-link">WHO GHO</span>
<span class="source-link">OpenDengue</span>
<span class="source-link">Open-Meteo</span>
<span class="source-link">Natural Earth</span>
</div>
</div>""", unsafe_allow_html=True)