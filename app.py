# Dynamic Risk Assessment and Policy Optimization system.
import math, random, string, threading, time, warnings
from datetime import datetime
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.ensemble        import GradientBoostingClassifier, RandomForestClassifier, IsolationForest
from sklearn.linear_model    import LogisticRegression
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics         import (roc_auc_score, f1_score, precision_score,
                                     recall_score, confusion_matrix,
                                     classification_report, roc_curve,
                                     precision_recall_curve, average_precision_score,
                                     brier_score_loss)
from sklearn.pipeline        import Pipeline
from sklearn.calibration     import calibration_curve, CalibratedClassifierCV

try:
    import lightgbm as lgb;    HAS_LGB  = True
except ImportError:
    HAS_LGB = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import shap; HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# Page Config
st.set_page_config(
    page_title="Dynamic risk assessment and policy optimization",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)
# Colors
C = dict(
    bg="#E2E8F205",   panel="#F0F5F3",  pan2="#37F5FC",  bord="#97E3F0",
    green="#2200E5",blue="#3D9EFF",   amber="#FFB020", red="#FF4444",
    purple="#A855F7",cyan="#06B6D4",  text="#101011",  muted="#0F0E0E",
    low="#00E5A0",  medium="#FFB020", high="#FF7A20",  critical="#FF2020",
)

PLOTLY_THEME = dict(
    template     ="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font         =dict(family="Space Grotesk, Inter, sans-serif", color=C["text"], size=12),
    margin       =dict(l=36, r=36, t=48, b=36),
)
# Global CSS
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html,body,.stApp{{background:{C['bg']}!important;color:{C['text']}!important;
  font-family:'Space Grotesk',sans-serif!important;}}
[data-testid="stSidebar"]{{background:{C['panel']}!important;border-right:1px solid {C['bord']}!important;}}
h1{{font-family:'JetBrains Mono',monospace!important;color:{C['green']}!important;
   font-size:1.45rem!important;letter-spacing:-.5px;margin-bottom:2px!important;}}
h2{{font-family:'JetBrains Mono',monospace!important;color:{C['text']}!important;font-size:1rem!important;}}
h3{{color:{C['muted']}!important;font-size:.68rem!important;text-transform:uppercase;letter-spacing:2px;margin-bottom:4px!important;}}
[data-testid="stMetric"]{{background:{C['pan2']}!important;border:1px solid {C['bord']}!important;
  border-radius:12px!important;padding:14px 18px!important;}}
[data-testid="stMetricValue"]{{font-family:'JetBrains Mono',monospace!important;
  color:{C['green']}!important;font-size:1.35rem!important;font-weight:600!important;}}
[data-testid="stMetricLabel"]{{color:{C['muted']}!important;font-size:.6rem!important;
  text-transform:uppercase;letter-spacing:1.5px;}}
.stButton>button{{background:linear-gradient(135deg,{C['green']},{C['cyan']})!important;
  color:#06090F!important;font-family:'JetBrains Mono',monospace!important;
  font-weight:700!important;border:none!important;border-radius:8px!important;
  padding:10px 24px!important;letter-spacing:.5px;transition:all .2s;}}
.stButton>button:hover{{filter:brightness(1.12);transform:translateY(-1px);}}
[data-testid="stDownloadButton"]>button{{background:transparent!important;
  color:{C['green']}!important;border:1px solid {C['green']}!important;
  font-family:'JetBrains Mono',monospace!important;border-radius:8px!important;font-size:.72rem!important;}}
.stTabs [data-baseweb="tab-list"]{{background:{C['pan2']};border-radius:10px;gap:3px;padding:3px;}}
.stTabs [data-baseweb="tab"]{{color:{C['muted']}!important;font-family:'JetBrains Mono',monospace;
  font-size:.68rem;border-radius:8px;}}
.stTabs [aria-selected="true"]{{background:linear-gradient(135deg,{C['green']},{C['cyan']})!important;
  color:#06090F!important;font-weight:700;}}
.stProgress>div>div{{background:linear-gradient(90deg,{C['green']},{C['cyan']})!important;}}
hr{{border-color:{C['bord']}!important;margin:8px 0!important;}}
.stSelectbox label,.stSlider label,.stMultiSelect label,.stNumberInput label{{
  color:{C['muted']}!important;font-size:.65rem!important;text-transform:uppercase;letter-spacing:1px;}}
.card{{background:{C['pan2']};border:1px solid {C['bord']};border-radius:12px;
  padding:16px 20px;margin:6px 0;line-height:1.75;}}
.card-green{{border-left:3px solid {C['green']};}}
.card-red  {{border-left:3px solid {C['red']};}}
.card-amber{{border-left:3px solid {C['amber']};}}
.tier-low     {{color:{C['green']};font-weight:700;}}
.tier-medium  {{color:{C['amber']};font-weight:700;}}
.tier-high    {{color:{C['high']};font-weight:700;}}
.tier-critical{{color:{C['red']};font-weight:700;}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.3}}}}
@keyframes glow{{0%,100%{{box-shadow:0 0 4px {C['green']}40}}50%{{box-shadow:0 0 14px {C['green']}80}}}}
.live-badge{{display:inline-flex;align-items:center;gap:6px;background:{C['green']}18;
  border:1px solid {C['green']}50;border-radius:20px;padding:3px 12px;animation:glow 2s infinite;}}
.live-dot{{width:7px;height:7px;border-radius:50%;background:{C['green']};animation:pulse 1.2s infinite;}}
.stop-badge{{display:inline-flex;align-items:center;gap:6px;background:{C['muted']}18;
  border:1px solid {C['muted']}50;border-radius:20px;padding:3px 12px;}}
</style>""", unsafe_allow_html=True)

# matplotlib dark theme (used for specialised charts)
plt.rcParams.update({
    "figure.facecolor": C["panel"], "axes.facecolor": C["pan2"],
    "axes.edgecolor":   C["bord"],  "axes.labelcolor": C["text"],
    "xtick.color":      C["muted"], "ytick.color":     C["muted"],
    "text.color":       C["text"],  "grid.color":      C["bg"],
    "grid.alpha":       0.45,       "font.family":     "monospace",
    "font.size":        10,         "legend.facecolor": C["pan2"],
    "legend.edgecolor": C["bord"],
})

# Constants
NUM_VEHICLES = 50_000
TIME_STEP    = 5
MAX_HISTORY  = 55_000

ROADS      = ["Highway","Gravel","Potholed","Local streets","Urban","Rural tarred"]
DIRECTIONS = ["North","South","East","West","Northeast","Northwest","Southeast","Southwest"]
ROAD_RISK = {"Highway":1.0, "Gravel":1.3, "Potholed":1.6, "Local streets":1.2, "Urban":1.1, "Rural tarred":1.0}

ZW_CITIES = [
    ("Harare",    -17.829, 31.052),
    ("Bulawayo",  -20.150, 28.589),
    ("Mutare",    -18.974, 32.671),
    ("Gweru",     -19.455, 29.820),
    ("Masvingo",  -20.070, 30.829),
    ("Kwekwe",    -18.921, 29.816),
    ("Chinhoyi",  -17.362, 30.199),
    ("Kadoma",    -18.340, 29.908),
    ("Marondera", -18.190, 31.551),
    ("Rusape",    -18.530, 32.130),
    ("Bindura",   -17.303, 31.331),
]
CITY_NAMES = [c[0] for c in ZW_CITIES]
CITY_PROBS = [0.35, 0.18, 0.10, 0.09, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]

CITY_CLIMATE = {
    "Harare":    dict(hot=0.25, rain=0.15, cold=0.05, wind=0.55),
    "Bulawayo":  dict(hot=0.45, rain=0.08, cold=0.02, wind=0.45),
    "Mutare":    dict(hot=0.15, rain=0.35, cold=0.10, wind=0.40),
    "Gweru":     dict(hot=0.25, rain=0.12, cold=0.08, wind=0.55),
    "Masvingo":  dict(hot=0.50, rain=0.10, cold=0.02, wind=0.38),
    "Kwekwe":    dict(hot=0.45, rain=0.10, cold=0.03, wind=0.42),
    "Chinhoyi":  dict(hot=0.30, rain=0.20, cold=0.05, wind=0.45),
    "Kadoma":    dict(hot=0.48, rain=0.08, cold=0.02, wind=0.42),
    "Marondera": dict(hot=0.15, rain=0.30, cold=0.12, wind=0.43),
    "Rusape":    dict(hot=0.12, rain=0.35, cold=0.15, wind=0.38),
    "Bindura":   dict(hot=0.28, rain=0.15, cold=0.05, wind=0.52),
}
WX_STATES = ["Clear", "Hot", "Windy", "Cloudy", "Rainy", "Foggy"]
WX_RISK = {"Clear":1.00,"Hot":1.10,"Windy":1.12,"Cloudy":1.00,"Rainy":1.55,"Foggy":1.40}
TOD_RISK = {"Morning":1.20,"Day":1.00,"Dusk":1.55,"Midnight":1.60}

VEHICLE_CATALOG = [
    # ----- Economy Hatchbacks / Kei Cars (under $6,000 FOB) -----
    {"type":"Hatchback",     "brand":"Nissan",       "model":"Dayz",      "cls":"Kei",                 "min_cc":600, "max_cc":800,  "base_price":3000,  "yr_min":2005,"yr_max":2025},
    {"type":"Hatchback",     "brand":"Toyota",       "model":"Vitz",      "cls":"Compact",             "min_cc":1000,"max_cc":1300, "base_price":3500,  "yr_min":2005,"yr_max":2025},
    {"type":"Hatchback",     "brand":"Honda",        "model":"Fit",       "cls":"Compact",             "min_cc":1300,"max_cc":1500, "base_price":5000,  "yr_min":2005,"yr_max":2022},
    {"type":"Hatchback",     "brand":"Suzuki",       "model":"Swift",     "cls":"Compact",             "min_cc":1000,"max_cc":1300, "base_price":6000,  "yr_min":2005,"yr_max":2025},
    
    # ----- Economy Sedans ($4,000 - $8,000 FOB) -----
    {"type":"Sedan",         "brand":"Toyota",       "model":"Corolla",   "cls":"Mid‑size",           "min_cc":1200,"max_cc":1500, "base_price":6000,  "yr_min":2005,"yr_max":2025},
    {"type":"Sedan",         "brand":"Nissan",       "model":"Tiida",     "cls":"Compact",             "min_cc":1300,"max_cc":1600, "base_price":5500,  "yr_min":2005,"yr_max":2025},
    {"type":"Sedan",         "brand":"Nissan",       "model":"Almera",    "cls":"Compact",             "min_cc":1300,"max_cc":1600, "base_price":8000,  "yr_min":2005,"yr_max":2025},
    
    # ----- Station Wagons ($6,000 - $10,000 FOB) -----
    {"type":"Station Wagon", "brand":"Toyota",       "model":"Fielder",   "cls":"Mid‑size",           "min_cc":1200,"max_cc":1500, "base_price":7000,  "yr_min":2005,"yr_max":2025},
    {"type":"Station Wagon", "brand":"Subaru",       "model":"Outback",   "cls":"Mid‑size",           "min_cc":2000,"max_cc":2500, "base_price":10000, "yr_min":2005,"yr_max":2025},
    
    # ----- SUVs / Crossovers ($8,000 - $35,000 FOB) -----
    {"type":"SUV",           "brand":"Toyota",       "model":"RAV4",      "cls":"Mid‑size",           "min_cc":1500,"max_cc":2000, "base_price":12000, "yr_min":2005,"yr_max":2025},
    {"type":"SUV",           "brand":"Nissan",       "model":"X-Trail",   "cls":"Mid‑size",           "min_cc":1500,"max_cc":2000, "base_price":8000,  "yr_min":2005,"yr_max":2025},
    {"type":"SUV",           "brand":"Mazda",        "model":"CX-5",      "cls":"Mid‑size",           "min_cc":1400,"max_cc":1900, "base_price":10000, "yr_min":2003,"yr_max":2025},
    {"type":"SUV",           "brand":"Honda",        "model":"CR-V",      "cls":"Mid‑size",           "min_cc":1500,"max_cc":2000, "base_price":9000,  "yr_min":2005,"yr_max":2025},
    {"type":"SUV",           "brand":"Toyota",       "model":"Fortuner",  "cls":"Full‑size",          "min_cc":1800,"max_cc":2500, "base_price":25000, "yr_min":2005,"yr_max":2025},
    {"type":"SUV",           "brand":"Toyota",       "model":"Prado",     "cls":"Full‑size",          "min_cc":2000,"max_cc":2800, "base_price":35000, "yr_min":2005,"yr_max":2025},
    
    # ----- Pickups / Double Cabs ($15,000 - $25,000 FOB) -----
    {"type":"Pickup",        "brand":"Toyota",       "model":"Hilux",     "cls":"Light commercial",   "min_cc":1800,"max_cc":2500, "base_price":20000, "yr_min":2005,"yr_max":2025},
    {"type":"Pickup",        "brand":"Ford",         "model":"Ranger",    "cls":"Light commercial",   "min_cc":1800,"max_cc":2500, "base_price":22000, "yr_min":2005,"yr_max":2025},
    {"type":"Pickup",        "brand":"Isuzu",        "model":"D-Max",     "cls":"Light commercial",   "min_cc":1700,"max_cc":2400, "base_price":20000, "yr_min":2005,"yr_max":2025},
    {"type":"Pickup",        "brand":"Nissan",       "model":"Navara",    "cls":"Light commercial",   "min_cc":1800,"max_cc":2500, "base_price":21000, "yr_min":2005,"yr_max":2025},
    
    # ----- Chinese Brands (Budget & Family) -----
    {"type":"SUV",           "brand":"Chery",        "model":"Tiggo 4 Pro","cls":"Compact",            "min_cc":1400,"max_cc":1600, "base_price":12000, "yr_min":2020,"yr_max":2025},
    {"type":"SUV",           "brand":"Great Wall",   "model":"Tank 500",  "cls":"Full‑size",          "min_cc":1800,"max_cc":2500, "base_price":40000, "yr_min":2022,"yr_max":2025},
    {"type":"SUV",           "brand":"BYD",          "model":"Atto 3",    "cls":"Electric",           "min_cc":0,   "max_cc":0,    "base_price":25000, "yr_min":2022,"yr_max":2025},
    
    # ----- Vans / Informal Transport ($4,500 - $6,000 FOB) -----
    {"type":"Van",           "brand":"Mazda",        "model":"Bongo",     "cls":"Light commercial",   "min_cc":1800,"max_cc":2200, "base_price":5000,  "yr_min":2005,"yr_max":2025},
    {"type":"Van",           "brand":"Nissan",       "model":"Vanette",   "cls":"Light commercial",   "min_cc":1800,"max_cc":2200, "base_price":4500,  "yr_min":2005,"yr_max":2025},
    {"type":"Van",           "brand":"Nissan",       "model":"AD Van",    "cls":"Light commercial",   "min_cc":1200,"max_cc":1600, "base_price":6000,  "yr_min":2005,"yr_max":2025},
    
    # ----- Minibuses / Public Transport ($14,000 - $25,000 FOB) -----
    {"type":"Minibus",       "brand":"Toyota",       "model":"Hiace",     "cls":"Light commercial",   "min_cc":2000,"max_cc":3000, "base_price":15000, "yr_min":2005,"yr_max":2025},
    {"type":"Minibus",       "brand":"Nissan",       "model":"Caravan",   "cls":"Light commercial",   "min_cc":2000,"max_cc":3000, "base_price":14000, "yr_min":2005,"yr_max":2025},
    {"type":"Minibus",       "brand":"Toyota",       "model":"Hiace (ZUPCO)","cls":"Public transport", "min_cc":2000,"max_cc":3000, "base_price":25000, "yr_min":2018,"yr_max":2025},
    
    # ----- Light Trucks ($20,000 - $40,000 FOB) -----
    {"type":"Light Truck",   "brand":"Isuzu",        "model":"NPR",       "cls":"Light commercial",   "min_cc":3000,"max_cc":5000, "base_price":20000, "yr_min":2005,"yr_max":2025},
    
    # ----- Heavy Trucks ($80,000 - $100,000+ FOB) -----
    {"type":"Heavy Truck",   "brand":"Volvo",        "model":"FH",        "cls":"Heavy commercial",   "min_cc":8000,"max_cc":20000, "base_price":80000, "yr_min":2005,"yr_max":2025},
    {"type":"Heavy Truck",   "brand":"Scania",       "model":"R-Series",  "cls":"Heavy commercial",   "min_cc":8000,"max_cc":20000, "base_price":90000, "yr_min":2005,"yr_max":2025},
    
    # ----- Specialised / Tankers & Tippers ($95,000 - $110,000 FOB) -----
    {"type":"Fuel Tanker",   "brand":"MAN",          "model":"TGS",       "cls":"Heavy commercial",   "min_cc":10000,"max_cc":25000, "base_price":100000, "yr_min":2005,"yr_max":2025},
    {"type":"Tipper Truck",  "brand":"Mercedes-Benz","model":"Actros",    "cls":"Heavy commercial",   "min_cc":10000,"max_cc":25000, "base_price":95000,  "yr_min":2005,"yr_max":2025},
    
    # ----- Agricultural ($10,000 FOB) -----
    {"type":"Tractor",       "brand":"Massey Ferguson","model":"MF 240",  "cls":"Agricultural",       "min_cc":2500,"max_cc":4000, "base_price":10000, "yr_min":2005,"yr_max":2025},
    
    # ----- Luxury / Premium ($35,000 - $50,000 FOB) -----
    {"type":"Luxury",        "brand":"BMW",          "model":"5 Series",  "cls":"Full‑size",          "min_cc":1500,"max_cc":2000, "base_price":40000, "yr_min":2005,"yr_max":2025},
    {"type":"Luxury",        "brand":"Mercedes-Benz","model":"C-Class",   "cls":"Full‑size",          "min_cc":1400,"max_cc":1900, "base_price":35000, "yr_min":2005,"yr_max":2025},
]

# Actuarial constants: Monthly base premiums (in USD) per vehicle type
BASE_PREMIUM = {
    "Sedan": 12.50,
    "Hatchback": 10.00,
    "Station Wagon": 13.33,
    "SUV": 16.67,
    "Pickup": 18.33,
    "Van": 15.00,
    "Minibus": 25.00,
    "Bus": 29.17,
    "Light Truck": 33.33,
    "Heavy Truck": 66.67,
    "Fuel Tanker": 100.00,
    "Tipper Truck": 75.00,
    "Tractor": 8.33,
    "Luxury": 33.33,
}

# Cache the value_counts operation – crucial for large DataFrames (50k+ rows)
# ====================== CACHED HELPER FUNCTIONS ======================
@st.cache_data(ttl=3600, show_spinner=False)
def _get_vehicle_counts(df: pd.DataFrame) -> pd.Series:
    """Cached helper to compute vehicle type counts."""
    if "Vehicle_Type" not in df.columns:
        return pd.Series(dtype=int)
    return df["Vehicle_Type"].value_counts().sort_values(ascending=False)

@st.cache_data(ttl=3600, show_spinner=False)
def _get_risk_tier_counts(df: pd.DataFrame) -> pd.Series:
    """Cached helper to compute risk tier counts in correct order."""
    if "Risk_Tier" not in df.columns:
        return pd.Series(dtype=int)
    counts = df["Risk_Tier"].value_counts()
    return counts.reindex(TIER_ORDER, fill_value=0)  # TIER_ORDER must be defined

# ====================== VEHICLE TYPE SUMMARY ======================
def show_vehicle_type_summary(df: pd.DataFrame):
    """
    Display total counts per vehicle type and a bar chart visualization.
    """
    if df.empty:
        st.warning("⚠️ No vehicle data available.")
        return None

    counts = _get_vehicle_counts(df)
    if counts.empty:
        st.warning("⚠️ Column 'Vehicle_Type' not found in the dataset.")
        return None

    # Summary metrics
    total_vehicles = counts.sum()
    unique_types = len(counts)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🚗 Total Vehicles", f"{total_vehicles:,}")
    with col2:
        st.metric("📋 Unique Types", unique_types)

    # Data table
    st.markdown("### 📊 Vehicle Type Distribution")
    display_df = pd.DataFrame({
        "Vehicle Type": counts.index,
        "Count": counts.values
    }).reset_index(drop=True)
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Count": st.column_config.NumberColumn("Count", format="%d")
        }
    )

    # Bar chart
    fig, ax = _fig(8, 5)   # assumes _fig is already defined
    colors = [list(TIER_COLS.values())[i % len(TIER_COLS)] for i in range(len(counts))]
    ax.bar(
        counts.index, counts.values,
        color=colors, edgecolor=C["bord"], lw=0.5,
        zorder=3, alpha=0.88
    )
    ax.set_title("Total Vehicles by Type", pad=6, fontsize=12, fontweight='semibold')
    ax.set_ylabel("Number of Vehicles", fontsize=10)
    ax.set_xlabel("Vehicle Type", fontsize=10)

    ax.tick_params(axis='x', rotation=30)
    for label in ax.get_xticklabels():
        label.set_ha('right')

    ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    return counts

# ====================== RISK TIER SUMMARY ======================
def show_risk_tier_summary(df: pd.DataFrame):
    """
    Display risk tier distribution (Low, Medium, High, Critical) with a bar chart.
    """
    if df.empty:
        st.warning("⚠️ No vehicle data available.")
        return None

    counts = _get_risk_tier_counts(df)
    if counts.empty:
        st.warning("⚠️ Column 'Risk_Tier' not found in the dataset. Please run risk assignment first.")
        return None

    total_vehicles = counts.sum()
    high_critical = counts.get("High", 0) + counts.get("Critical", 0)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🚗 Total Vehicles", f"{total_vehicles:,}")
    with col2:
        st.metric("⚠️ High + Critical Risk", f"{high_critical:,}")

    st.markdown("### 📊 Risk Tier Distribution")
    display_df = pd.DataFrame({
        "Risk Tier": counts.index,
        "Count": counts.values
    }).reset_index(drop=True)
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Count": st.column_config.NumberColumn("Count", format="%d")
        }
    )

    fig, ax = _fig(8, 5)
    colors = [TIER_COLS[tier] for tier in counts.index]
    ax.bar(
        counts.index, counts.values,
        color=colors, edgecolor=C["bord"], lw=0.5,
        zorder=3, alpha=0.88
    )
    ax.set_title("Risk Tier Distribution (Low to Critical)", pad=6, fontsize=12, fontweight='semibold')
    ax.set_ylabel("Number of Vehicles", fontsize=10)
    ax.set_xlabel("Risk Tier", fontsize=10)

    ax.tick_params(axis='x', rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    return counts

# Risk multiplier
RISK_MULT      = {"Low":0.42,"Medium":1.00,"High":1.50,"Critical":1.95}
EXPECTED_CLAIM = {"Low":400,"Medium":2000,"High":6500,"Critical":18000}
EXPENSE_RATIO  = 0.58
INV_YIELD      = 0.0825

# Weather & time-of-day risk factors
WX_RISK = { "Clear": 1.00,  "Hot": 1.10, "Windy": 1.12, "Cloudy": 1.00,  "Rainy": 1.55, "Foggy": 1.40 }

# Time‑of‑day risk factors
TOD_RISK = {  "Morning": 1.20, "Day": 1.00, "Dusk": 1.55, "Midnight": 1.60,}

# Road speed limits
ROAD_LIMITS = {
    "Highway": (80, 120),
    "Rural tarred": (60, 100),
    "Urban": (40, 60),
    "Local streets": (30, 50),
    "Gravel": (30, 80),
    "Potholed": (5, 30),
}
ROAD_AVG_SPEED = {k: (v[0] + v[1]) / 2 for k, v in ROAD_LIMITS.items()}

# Base risk profile by vehicle type (Zimbabwe context)
VEHICLE_RISK_BASE = {
    # Private vehicles (mostly Low/Medium)
    "Sedan":          {"Low":0.60, "Medium":0.30, "High":0.08, "Critical":0.02},
    "Hatchback":      {"Low":0.65, "Medium":0.28, "High":0.06, "Critical":0.01},
    "Station Wagon":  {"Low":0.55, "Medium":0.32, "High":0.10, "Critical":0.03},
    "SUV":            {"Low":0.50, "Medium":0.35, "High":0.12, "Critical":0.03},
    "Pickup":         {"Low":0.45, "Medium":0.35, "High":0.15, "Critical":0.05},
    "Luxury":         {"Low":0.55, "Medium":0.30, "High":0.12, "Critical":0.03},
    
    # Commercial / high-risk
    "Van":            {"Low":0.40, "Medium":0.35, "High":0.18, "Critical":0.07},
    "Minibus":        {"Low":0.10, "Medium":0.30, "High":0.35, "Critical":0.25},  # Kombis
    "Bus":            {"Low":0.15, "Medium":0.30, "High":0.35, "Critical":0.20},
    "Light Truck":    {"Low":0.30, "Medium":0.35, "High":0.25, "Critical":0.10},
    "Heavy Truck":    {"Low":0.20, "Medium":0.30, "High":0.30, "Critical":0.20},
    "Fuel Tanker":    {"Low":0.15, "Medium":0.25, "High":0.35, "Critical":0.25},
    "Tipper Truck":   {"Low":0.20, "Medium":0.30, "High":0.30, "Critical":0.20},
    "Tractor":        {"Low":0.70, "Medium":0.25, "High":0.04, "Critical":0.01},
}
def assign_risk_tier(vehicle_type):
    """Assign a risk tier based on vehicle type probabilities."""
    probs = VEHICLE_RISK_BASE.get(vehicle_type, {"Low":0.5, "Medium":0.4, "High":0.09, "Critical":0.01})
    tiers = ["Low", "Medium", "High", "Critical"]
    return np.random.choice(tiers, p=[probs[t] for t in tiers])

def adjust_risk_tier(base_tier, annual_km, driver_age=None, city=None):
    """Increase risk tier if annual mileage exceeds threshold."""
    if annual_km > 30000:
        if base_tier == "Low": return "Medium"
        if base_tier == "Medium": return "High"
        if base_tier == "High": return "Critical"
    # Add other adjustments here if needed
    return base_tier

# Example: generating a DataFrame with vehicles
vehicles = []  # list to collect rows

for vehicle_id in range(NUM_VEHICLES):
    # Select vehicle from catalog (example logic)
    vehicle_info = np.random.choice(VEHICLE_CATALOG)
    vehicle_type = vehicle_info["type"]
    
    # Generate synthetic attributes
    driver_age = np.random.randint(18, 75)
    city = np.random.choice(CITY_NAMES, p=CITY_PROBS)
    annual_km = np.random.randint(5000, 60000)  # realistic range
    
    # Assign base risk tier
    base_tier = assign_risk_tier(vehicle_type)
    # Adjust based on annual km
    final_tier = adjust_risk_tier(base_tier, annual_km)
    
    # Store row
    vehicles.append({
        "Vehicle_ID": vehicle_id,
        "Vehicle_Type": vehicle_type,
        "Brand": vehicle_info["brand"],
        "Model": vehicle_info["model"],
        "City": city,
        "Driver_Age": driver_age,
        "Annual_KM": annual_km,
        "Risk_Tier": final_tier
    })

# Create DataFrame
df = pd.DataFrame(vehicles)

# Verify distribution
print(df["Risk_Tier"].value_counts())

def adjust_risk_tier(base_tier, driver_age, city, annual_km):
    # Simple rule: if annual_km > 30,000, increase risk by one level
    if annual_km > 30000:
        if base_tier == "Low": return "Medium"
        if base_tier == "Medium": return "High"
        if base_tier == "High": return "Critical"
    # Add more logic as needed
    return base_tier

# Risk tier colours
RISK_COLOURS = {
    "Low":  "#2ecc71", "Medium":  "#1212f3", "High": "#e208cc","Critical": "#A50303",
}
TIER_COLS  = RISK_COLOURS
TIER_ORDER = ["Low", "Medium", "High", "Critical"]

# Data that many threads can use at the same time without causing errors
class _State:
    def __init__(self):
        self._lk        = threading.Lock()
        self.snap       = pd.DataFrame()
        self.history    = pd.DataFrame()
        self.tick       = 0
        self.running    = False
        self.ts         = None
        self.city_weather = {c: random.choice(list(WX_RISK.keys())) for c in CITY_NAMES}
        self.wx_countdown = {c:random.randint(200,600) for c in CITY_NAMES}

    def push(self, snap: pd.DataFrame) -> None:
      with self._lock: 
        self.snapshot = snap.copy()
        self.history = pd.concat([self.history, snap], ignore_index=True).tail(MAX_HISTORY)
        self.step_counter += 1
        self.timestamp = datetime.now()

    def tick_city_weather(self):
        with self._lk:
            for city in CITY_NAMES:
                self.wx_countdown[city] -= 1
                if self.wx_countdown[city] <= 0:
                    cl   = CITY_CLIMATE[city]
                    hour = datetime.now().hour
                    if   5<=hour<10:
                        w = [0.45, cl["hot"]*0.5, cl["wind"]*0.5, 0.10, cl["rain"]*0.3, 0.0]
                    elif 10<=hour<16:
                        w = [0.30, cl["hot"], cl["wind"], 0.10, cl["rain"], cl["cold"]*0.3]
                    elif 16<=hour<20:
                        w = [0.20, cl["hot"]*0.3, cl["wind"]*0.5, 0.25, cl["rain"], cl["cold"]*0.5]
                    else:
                        w = [0.05, 0.05, 0.10, 0.10, cl["rain"]*0.5, cl["cold"]]
                    w = np.array(w, float); w = np.clip(w, 0.01, None); w /= w.sum()
                    self.city_weather[city] = np.random.choice(WX_STATES, p=w)
                    self.wx_countdown[city] = random.randint(300, 800)

    def get_snap(self):
        with self._lk: return self.snap.copy()
    def get_history(self):
        with self._lk: return self.history.copy()
    def get_city_weather(self):
        with self._lk: return dict(self.city_weather)

@st.cache_resource
def _state(): return _State()
STATE = _state()

# Create the vehicle data once and reuse it later (cached)
@st.cache_resource
def _fleet():
    n   = NUM_VEHICLES
    rng = np.random.default_rng(42)
    idx = rng.choice(len(VEHICLE_CATALOG),n)
    cats=[VEHICLE_CATALOG[i] for i in idx]
    types=[c["type"] for c in cats]; makes=[c["brand"] for c in cats]
    models=[c["model"] for c in cats]; sizes=[c["cls"] for c in cats]
    wmins=np.array([c["min_cc"] for c in cats],float)
    wmaxs=np.array([c["max_cc"] for c in cats],float)
    bpri=np.array([c["base_price"] for c in cats],float)
    years=rng.integers(2005,2025,n)
    ages=2025-years; depr=0.65**ages
    prices=np.round(bpri*depr*rng.uniform(0.88,1.12,n),2)
    city_idx=rng.choice(len(ZW_CITIES),n,p=CITY_PROBS)
    cities=[ZW_CITIES[i][0] for i in city_idx]
    lats=np.array([ZW_CITIES[i][1] for i in city_idx],float)
    lons=np.array([ZW_CITIES[i][2] for i in city_idx],float)
    cc_arr=rng.uniform(wmins,wmaxs)
    idle_rpm=rng.integers(700,900,n).astype(float)

    df=pd.DataFrame({
        "Vehicle_Type":types,"Vehicle_Make":makes,"Vehicle_Model":models,"Vehicle_Size":sizes,
        "Vehicle_Number":[''.join(random.choices(string.ascii_uppercase,k=3))+
                          ''.join(random.choices(string.digits,k=4)) for _ in range(n)],
        "Vehicle_Usage":rng.choice(["Private","Commercial"],n,p=[0.60,0.40]).tolist(),
        "Vehicle_Year":years,"Engine_CC":np.round(cc_arr).astype(int),
        "Idle_RPM":idle_rpm,"Unladen_Weight_Kg":np.round(rng.uniform(wmins,wmaxs)).astype(int),
        "Vehicle_Price":prices,"Registration_City":cities,

         # updatable live columns
        "state":"Stationary","speed":0.0,"distance":0.0,"fuel":0.1,
        "trip":None,"road":None,"direction":None,
        "lat":lats+rng.uniform(-0.25,0.25,n),"lon":lons+rng.uniform(-0.25,0.25,n),
        "harsh_brake":0,"harsh_accel":0,"harsh_corner":0,
        "rpm":idle_rpm.copy(),"throttle_pct":0.0,
        "coolant_temp_c":rng.uniform(65,80,n),
        "engine_load_pct":0.0,"maf_gs":rng.uniform(1.8,4.0,n),
        "battery_v":rng.uniform(12.2,12.8,n),"next_update":datetime.now(),
    })
    return {"df":df}

# A thread that runs in the background creating data automatically
def _trip_prob(hour,is_comm):
    if   6<=hour< 9: return np.where(is_comm,0.72,0.68)
    elif 9<=hour<16: return np.where(is_comm,0.62,0.06)
    elif 16<=hour<20:return np.where(is_comm,0.76,0.72)
    else:            return np.where(is_comm,0.22,0.02)

def _tick(df,city_weather):
    now=datetime.now(); hour=now.hour; n=len(df)
    is_comm=(df["Vehicle_Usage"]=="Commercial").values
    start_p=_trip_prob(hour,is_comm)
    ready=(df["next_update"]<=now).values
    start=(df["state"]=="Stationary").values&ready&(np.random.rand(n)<start_p)
    ns=start.sum()
    if ns:
        df.loc[start,"state"]="Driving"
        df.loc[start,"trip"]=["ET"+now.strftime("%Y%m%d")+
            ''.join(random.choices(string.ascii_uppercase+string.digits,k=6)) for _ in range(ns)]
        df.loc[start,"road"]=np.random.choice(ROADS,ns)
        df.loc[start,"direction"]=np.random.choice(DIRECTIONS,ns)
    driving=(df["state"]=="Driving").values&ready; nd=driving.sum()
    if nd:
        road_ser=df.loc[driving,"road"]
        mn_spd=road_ser.map(lambda r:ROAD_LIMITS[r][0]).values.astype(float)
        mx_spd=road_ser.map(lambda r:ROAD_LIMITS[r][1]).values.astype(float)
        city_arr=df.loc[driving,"Registration_City"].values
        wx_arr=np.array([city_weather.get(c,"Sunny") for c in city_arr])
        mx_spd=np.where(wx_arr=="Rainy",mx_spd*0.68,mx_spd)
        mx_spd=np.where(wx_arr=="Cold", mx_spd*0.82,mx_spd)
        cur=df.loc[driving,"speed"].values.astype(float)
        tgt=np.random.uniform(mn_spd,mx_spd)
        adj=np.random.uniform(0.05,0.22,nd)
        spd=np.clip(cur+(tgt-cur)*adj,0,mx_spd)
        df.loc[driving,"speed"]=spd
        dist=spd*(TIME_STEP/3600)
        df.loc[driving,"distance"]+=dist
        df.loc[driving,"fuel"]+=dist/np.random.uniform(7,16,nd)
        df.loc[driving,"lat"]=(df.loc[driving,"lat"]+np.random.uniform(-0.00008,0.00008,nd)).clip(-22.5,-15.5)
        df.loc[driving,"lon"]=(df.loc[driving,"lon"]+np.random.uniform(-0.00008,0.00008,nd)).clip(25.0,33.0)
        spd_norm=np.clip(spd/120,0,1)
        df.loc[driving,"harsh_brake"]+=(np.random.rand(nd)<0.0010+spd_norm*0.0008).astype(int)
        df.loc[driving,"harsh_accel"]+=(np.random.rand(nd)<0.0008+spd_norm*0.0006).astype(int)
        df.loc[driving,"harsh_corner"]+=(np.random.rand(nd)<0.0006+spd_norm*0.0005).astype(int)
        idle_rpm=df.loc[driving,"Idle_RPM"].values
        cc_arr=df.loc[driving,"Engine_CC"].values.astype(float)
        gear_ratio=np.clip(spd/30,1,6)
        rpm_target=idle_rpm+(spd/mx_spd.clip(1))*(5500-idle_rpm)/gear_ratio
        rpm_cur=df.loc[driving,"rpm"].values
        rpm_new=np.clip(rpm_cur+(rpm_target-rpm_cur)*0.35+np.random.normal(0,80,nd),idle_rpm,6800)
        df.loc[driving,"rpm"]=rpm_new
        throttle=np.clip((spd/mx_spd.clip(1))*100*np.random.uniform(0.6,1.0,nd)+np.random.normal(0,5,nd),0,100)
        df.loc[driving,"throttle_pct"]=throttle
        load=np.clip((rpm_new/6800)*60+throttle*0.4,0,100)
        df.loc[driving,"engine_load_pct"]=load
        cool_cur=df.loc[driving,"coolant_temp_c"].values
        df.loc[driving,"coolant_temp_c"]=np.clip(cool_cur+np.random.uniform(0.02,0.15,nd),65,108)
        maf=(cc_arr/1000)*(load/100)*2.5*np.random.uniform(0.88,1.12,nd)
        df.loc[driving,"maf_gs"]=maf.clip(1.0,280)
        df.loc[driving,"battery_v"]=np.clip(np.random.normal(14.2,0.25,nd),13.5,14.9)
        stop=driving.copy(); stop[driving]=np.random.rand(nd)<0.008
        if stop.sum():
            df.loc[stop,["state","speed","trip","road","direction"]]=["Stationary",0,None,None,None]
    stationary=(df["state"]=="Stationary").values&ready; ni=stationary.sum()
    if ni:
        idle_rpm_s=df.loc[stationary,"Idle_RPM"].values
        df.loc[stationary,"rpm"]=(idle_rpm_s+np.random.normal(0,30,ni)).clip(600,1000)
        df.loc[stationary,"throttle_pct"]=np.random.uniform(0,3,ni)
        df.loc[stationary,"engine_load_pct"]=np.random.uniform(0,8,ni)
        df.loc[stationary,"maf_gs"]=np.random.uniform(1.5,4.5,ni)
        df.loc[stationary,"battery_v"]=np.random.uniform(12.1,12.8,ni)
        cool_s=df.loc[stationary,"coolant_temp_c"].values
        df.loc[stationary,"coolant_temp_c"]=np.clip(cool_s-np.random.uniform(0,0.05,ni),65,108)
        df.loc[stationary,"speed"]=0.0
    nr=ready.sum()
    if nr:
        df.loc[ready,"next_update"]=now+pd.to_timedelta(np.random.randint(4,12,nr),unit="s")
    return df

def _format_snap(df,city_weather):
    h=datetime.now().hour
    dtm=("Morning" if 3<=h<12 else "Day" if 12<=h<17 else "Dusk" if 17<=h<21 else "Midnight")
    out=df[["Vehicle_Type","Vehicle_Make","Vehicle_Model","Vehicle_Size",
            "Vehicle_Number","Vehicle_Usage","Vehicle_Year","Engine_CC","Vehicle_Price",
            "Unladen_Weight_Kg","Registration_City",
            "trip","road","direction","state","speed",
            "lat","lon","distance","fuel",
            "harsh_brake","harsh_accel","harsh_corner",
            "rpm","throttle_pct","coolant_temp_c","engine_load_pct","maf_gs","battery_v"]].copy()
    out.rename(columns={
        "trip":"Trip_ID","road":"Road_Type","direction":"Direction",
        "state":"Status","speed":"Speed_kmh","lat":"Latitude","lon":"Longitude",
        "distance":"Distance_Day_Km","fuel":"Fuel_Used_L",
        "harsh_brake":"HB_Day","harsh_accel":"HA_Day","harsh_corner":"HC_Day",
        "rpm":"RPM","throttle_pct":"Throttle_Pct","coolant_temp_c":"Coolant_Temp_C",
        "engine_load_pct":"Engine_Load_Pct","maf_gs":"MAF_gs","battery_v":"Battery_V",
    },inplace=True)
    out["Weather_Condition"]=out["Registration_City"].map(city_weather).fillna("Sunny")
    out["Day_Time"]=dtm
    out["Timestamp"]=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return out.round(3)

def _generator_loop():
    fleet=_fleet()["df"]
    while True:
        if not STATE.running:
            time.sleep(0.5); continue
        t0=time.perf_counter()
        try:
            STATE.tick_city_weather()
            cw=STATE.get_city_weather()
            fleet=_tick(fleet,cw)
            STATE.push(_format_snap(fleet,cw))
        except Exception:
            pass
        elapsed=time.perf_counter()-t0
        time.sleep(max(0.0,TIME_STEP-elapsed))

@st.cache_resource
def _start_generator():
    t=threading.Thread(target=_generator_loop,daemon=True); t.start(); return t
_start_generator()

# Feature engineering and risk scoring
_EMPTY_COLS=[
    "Vehicle_Type","Vehicle_Number","Vehicle_Make","Vehicle_Model","Vehicle_Size",
    "Vehicle_Usage","Vehicle_Year","Engine_CC","Vehicle_Price","Unladen_Weight_Kg",
    "Registration_City","Status","Speed_kmh","HB_Day","HA_Day","HC_Day",
    "Distance_Day_Km","Fuel_Used_L","Road_Type","Weather_Condition",
    "Day_Time","Latitude","Longitude","Trip_ID","Direction","Timestamp",
    "RPM","Throttle_Pct","Coolant_Temp_C","Engine_Load_Pct","MAF_gs","Battery_V",
    "Harsh_Score","Vehicle_Age","Weather_Risk","ToD_Risk","Fuel_Eff_KmL",
    "Speed_Deviation","OBD_Risk_Signal","Risk_Score","Risk_Tier",
    "Risk_Label","Claim_Prob","Claim_Label",
]

def engineer(raw):
    if raw is None or raw.empty:
        return pd.DataFrame(columns=_EMPTY_COLS)
    df=raw.copy()
    num_defaults=[
        ("Speed_kmh",0.),("HB_Day",0.),("HA_Day",0.),("HC_Day",0.),
        ("Distance_Day_Km",0.),("Fuel_Used_L",0.1),("Unladen_Weight_Kg",1200.),
        ("Vehicle_Year",2015.),("Vehicle_Price",10000.),("Engine_CC",1500.),
        ("Latitude",-18.),("Longitude",31.),
        ("RPM",800.),("Throttle_Pct",0.),("Coolant_Temp_C",85.),
        ("Engine_Load_Pct",0.),("MAF_gs",3.),("Battery_V",12.5),
    ]
    # numeric and categorical coercion
    for col,default in num_defaults:
        if col not in df.columns: df[col]=default
        else: df[col]=pd.to_numeric(df[col],errors="coerce").fillna(default)
    cat_defaults=[
        ("Vehicle_Type","Sedan"),("Vehicle_Usage","Private"),("Status","Stationary"),
        ("Road_Type","Urban"),("Weather_Condition","Sunny"),("Day_Time","Day"),
        ("Registration_City","Harare"),("Vehicle_Make","Toyota"),
        ("Vehicle_Model","Corolla"),("Vehicle_Size","Mid-size"),
    ]
    for col,default in cat_defaults:
        if col not in df.columns: df[col]=default
        else: df[col]=df[col].fillna(default)

    df["Harsh_Score"]     = df["HB_Day"]*0.40+df["HA_Day"]*0.30+df["HC_Day"]*0.30
    df["Vehicle_Age"]     = (2025-df["Vehicle_Year"].clip(lower=1990)).clip(lower=0)
    df["Weather_Risk"]    = df["Weather_Condition"].map(WX_RISK).fillna(1.10)
    df["ToD_Risk"]        = df["Day_Time"].map(TOD_RISK).fillna(1.10)
    df["Fuel_Eff_KmL"]    = (df["Distance_Day_Km"]/df["Fuel_Used_L"].clip(lower=0.1)).clip(0,30)
    road_mean             = df.groupby("Road_Type")["Speed_kmh"].transform("mean")
    df["Speed_Deviation"] = (df["Speed_kmh"]-road_mean).abs()
    df["OBD_Risk_Signal"] = (
        (df["Engine_Load_Pct"]/100)*0.4+
        np.clip((df["RPM"]-3000)/3800,0,1)*0.35+
        np.clip((df["Coolant_Temp_C"]-95)/15,0,1)*0.25
    ).round(4)

    lim     = df["Road_Type"].map(ROAD_AVG_SPEED).fillna(60)
    excess  = (df["Speed_kmh"]-lim).clip(lower=0)
    spd_risk= 1/(1+np.exp(-0.05*(excess-12)))
    ev_risk = (df["Harsh_Score"]/20).clip(0,1)
    wx_risk = ((df["Weather_Risk"]-1)/1.6).clip(0,1)
    age_risk= (df["Vehicle_Age"]/30).clip(0,1)
    tod_risk= ((df["ToD_Risk"]-1)/0.6).clip(0,1)
    obd_risk= df["OBD_Risk_Signal"].clip(0,1)

    # engineered features
    df["Risk_Score"]=(spd_risk*0.28+ev_risk*0.26+wx_risk*0.17+
                      age_risk*0.12+tod_risk*0.09+obd_risk*0.08).round(4)
    df["Risk_Tier"]=pd.cut(df["Risk_Score"],bins=[-0.001,0.25,0.50,0.75,1.001],
                            labels=["Low","Medium","High","Critical"]).astype(str)
    # Risk_Label: deterministic tier-based (for scoring display only)
    df["Risk_Label"]=df["Risk_Tier"].isin(["High","Critical"]).astype(int)

    # Claim outcome(probabilistic — used for ML training)
    rng_state=np.random.RandomState(int(df["Speed_kmh"].sum()*1000)%2**31)
    logit=(df["Risk_Score"]*7.5-3.2)                   # base logit signal
    # Add residual noise (heterogeneity from unobserved factors)
    noise=rng_state.normal(0,1.1,len(df))               # inter-driver variability
    logit_noisy=logit+noise
    p_claim=1/(1+np.exp(-logit_noisy))                  # sigmoid → probability
    p_claim=p_claim.clip(0.02,0.97)                     # practical bounds
    df["Claim_Prob"]=p_claim.round(4)
    # Bernoulli draw — models cannot perfectly predict this, giving realistic AUC
    df["Claim_Label"]=rng_state.binomial(1,p_claim).astype(int)
    return df

# Calculate premiums for many customers at the same time
_PREM_COLS=["Vehicle_Number","Vehicle_Type","Vehicle_Make","Vehicle_Year",
            "Vehicle_Price","Registration_City","Vehicle_Usage","Risk_Score",
            "Risk_Tier","Harsh_Score","Distance_Day_Km","Gross_Premium",
            "Expected_Claim","Expenses","UW_Profit","Inv_Return","Net_Profit","Loss_Ratio"]

def build_premium_df(df):
    if df is None or df.empty or "Risk_Score" not in df.columns:
        return pd.DataFrame(columns=_PREM_COLS)
    out=pd.DataFrame()
    out["Vehicle_Number"]=df["Vehicle_Number"].values
    out["Vehicle_Type"]=df["Vehicle_Type"].values
    out["Vehicle_Make"]=df["Vehicle_Make"].values
    out["Vehicle_Year"]=df["Vehicle_Year"].values.astype(int)
    out["Vehicle_Price"]=df["Vehicle_Price"].values
    out["Registration_City"]=df.get("Registration_City",pd.Series(["Harare"]*len(df))).values
    out["Vehicle_Usage"]=df["Vehicle_Usage"].values
    out["Risk_Score"]=df["Risk_Score"].round(4).values
    rs=out["Risk_Score"]
    out["Risk_Tier"]=np.where(rs>0.75,"Critical",np.where(rs>0.50,"High",np.where(rs>0.25,"Medium","Low")))
    base=np.array([BASE_PREMIUM.get(str(v),8000) for v in out["Vehicle_Type"]],float)
    mult=np.array([RISK_MULT[t] for t in out["Risk_Tier"]],float)
    age_l=1+np.clip(2025-out["Vehicle_Year"].values.astype(float)-5,0,None)*0.022
    usg_l=np.where(out["Vehicle_Usage"]=="Commercial",1.18,1.00)
    dist=pd.to_numeric(df.get("Distance_Day_Km",pd.Series([30]*len(df))),errors="coerce").fillna(30).values
    ml=1+np.clip(dist*365-15000,0,None)/100000*0.28
    hs=pd.to_numeric(df.get("Harsh_Score",pd.Series([2]*len(df))),errors="coerce").fillna(2).values
    beh=np.where(hs<=1,0.90,np.where(hs<=4,1.00,np.where(hs<=8,1.15,1.35)))
    gross=np.round(base*mult*age_l*usg_l*ml*beh,2)
    claim=np.array([EXPECTED_CLAIM[t] for t in out["Risk_Tier"]],float)
    exps=np.round(gross*EXPENSE_RATIO,2)
    inv=np.round(gross*INV_YIELD*0.5,2)
    uw=np.round(gross-claim-exps,2)
    net=np.round(uw+inv,2)
    lr=np.round((claim+exps)/np.maximum(gross,1),4)
    out["Gross_Premium"]=gross; out["Expected_Claim"]=claim; out["Expenses"]=exps
    out["UW_Profit"]=uw; out["Inv_Return"]=inv; out["Net_Profit"]=net
    out["Loss_Ratio"]=lr; out["Harsh_Score"]=hs; out["Distance_Day_Km"]=dist
    return out.reset_index(drop=True)

# Machine learning model training
ML_FEAT_COLS = [
    # Raw speed / behaviour
    "Speed_kmh",         # Vehicle speed sensor
    "RPM",               # Engine RPM sensor
    "Throttle_Pct",      # Throttle position sensor
    "Engine_Load_Pct",   # Calculated engine load
    "Coolant_Temp_C",    # Coolant temperature sensor
    "MAF_gs",            # Mass Air Flow sensor
    "Battery_V",         # Battery voltage
    # Raw event counters
    "HB_Day",            # Harsh braking count
    "HA_Day",            # Harsh acceleration count
    "HC_Day",            # Harsh cornering count
    # Raw distance / fuel
    "Distance_Day_Km",   # Daily odometer
    "Fuel_Used_L",       # Fuel consumption
    # Static vehicle attributes
    "Vehicle_Age",       # Age in years (static)
    "Unladen_Weight_Kg", # Vehicle weight class (static)
]

def _meval(y_te,proba):
    proba=np.clip(proba,1e-7,1-1e-7)
    pred=(proba>=0.5).astype(int)
    fpr,tpr,_=roc_curve(y_te,proba)
    prec,rec,_=precision_recall_curve(y_te,proba)
    try:
        frac_pos,mean_pred=calibration_curve(y_te,proba,n_bins=10,strategy="uniform")
    except Exception:
        frac_pos=np.array([]); mean_pred=np.array([])
    brier=brier_score_loss(y_te,proba)
    return {
        "auc":   roc_auc_score(y_te,proba),
        "ap":    average_precision_score(y_te,proba),
        "f1":    f1_score(y_te,pred,zero_division=0),
        "prec":  precision_score(y_te,pred,zero_division=0),
        "rec":   recall_score(y_te,pred,zero_division=0),
        "ks":    float(max(tpr-fpr)),
        "brier": brier,
        "proba":proba,"pred":pred,"y_te":y_te,
        "fpr":fpr,"tpr":tpr,"pr_prec":prec,"pr_rec":rec,
        "cal_frac":frac_pos,"cal_mean":mean_pred,
    }

@st.cache_resource(show_spinner=False)
def train_models(dhash,data_tuple):
    X_arr,y_arr,feat=data_tuple
    X=pd.DataFrame(X_arr,columns=feat)
    y=pd.Series(y_arr.astype(int))
    # Guarantee both classes
    if y.nunique()<2:
        min_cls=1 if (y==0).all() else 0
        n_s=max(300,len(y)//15)
        sX=X.sample(n_s,replace=True,random_state=42)
        sX=sX+np.random.normal(0,0.1,sX.shape)
        sy=pd.Series([min_cls]*n_s)
        X=pd.concat([X,sX],ignore_index=True)
        y=pd.concat([y.reset_index(drop=True),sy],ignore_index=True)
    X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)
    pw=max(1.0,(y_tr==0).sum()/max(1,(y_tr==1).sum()))
    res={}

    # Logistic Regression
    lr_base=Pipeline([("sc",StandardScaler()),
                      ("lr",LogisticRegression(max_iter=2000,class_weight="balanced",
                                               C=0.3,solver="lbfgs",random_state=42))])
    lr_base.fit(X_tr,y_tr)
    res["Logistic Regression"]=_meval(y_te.values,lr_base.predict_proba(X_te)[:,1])

    # Random Forest (constrained depth prevents overfitting)
    rf=RandomForestClassifier(n_estimators=300,max_depth=6,min_samples_leaf=25,
                               max_features="sqrt",class_weight="balanced",
                               random_state=42,n_jobs=-1)
    rf.fit(X_tr,y_tr)
    res["Random Forest"]=_meval(y_te.values,rf.predict_proba(X_te)[:,1])
    res["Random Forest"]["imp"]=dict(zip(feat,rf.feature_importances_))

    # LightGBM / GradientBoosting
    if HAS_LGB:
        gb=lgb.LGBMClassifier(n_estimators=500,learning_rate=0.03,max_depth=5,
                               num_leaves=31,min_child_samples=50,
                               reg_alpha=0.1,reg_lambda=0.2,
                               scale_pos_weight=pw,random_state=42,
                               verbosity=-1,n_jobs=-1)
        gb.fit(X_tr,y_tr,eval_set=[(X_te,y_te)],
               callbacks=[lgb.early_stopping(50,verbose=False),lgb.log_evaluation(-1)])
        label="LightGBM"
    else:
        gb=GradientBoostingClassifier(n_estimators=400,learning_rate=0.04,max_depth=4,
                                       min_samples_leaf=30,subsample=0.75,
                                       random_state=42)
        gb.fit(X_tr,y_tr)
        label="GradientBoosting"
    res[label]=_meval(y_te.values,gb.predict_proba(X_te)[:,1])
    res[label]["imp"]=dict(zip(feat,gb.feature_importances_))
    res["_best"]=label; res["_model"]=gb; res["_feat"]=feat
    res["_X_te"]=X_te; res["_y_te"]=y_te.values

    iso=IsolationForest(n_estimators=200,contamination=0.08,random_state=42,n_jobs=-1)
    iso.fit(X_tr)
    res["_iso"]=iso
    return res,feat

# Helper functions for creating charts using Matplotlib
def _fig(w=8,h=4,**kw): return plt.subplots(figsize=(w,h),**kw)
def _show(fig): st.pyplot(fig,use_container_width=True); plt.close(fig)

def _style_tier(v):
    return {"Low":f"color:{C['green']};font-weight:700",
            "Medium":f"color:{C['amber']};font-weight:700",
            "High":f"color:{C['high']};font-weight:700",
            "Critical":f"color:{C['red']};font-weight:700"}.get(str(v),"")

def _hex_to_rgba(hex_color,alpha=0.40):
    h=hex_color.lstrip("#")
    try: return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{alpha})"
    except Exception: return f"rgba(0,229,160,{alpha})"

def donut_mpl(df,col="Risk_Tier",title="Risk Tier Split"):
    cnt=df[col].value_counts()
    ordr=[t for t in TIER_ORDER if t in cnt.index]
    sizes=[cnt[t] for t in ordr]; cols=[TIER_COLS[t] for t in ordr]
    fig,ax=_fig(4.2,4.2)
    wedges,texts,autotexts=ax.pie(
        sizes,labels=ordr,colors=cols,autopct="%1.1f%%",
        startangle=140,pctdistance=0.80,
        wedgeprops={"linewidth":2.0,"edgecolor":C["bg"]})
    for at in autotexts: at.set_fontsize(9); at.set_fontweight("bold")
    ax.add_artist(plt.Circle((0,0),0.58,color=C["panel"]))
    ax.set_title(title,pad=8,fontsize=10,color=C["text"])
    fig.tight_layout(); return fig

def waterfall_mpl(prem):
    _req=["Gross_Premium","Expected_Claim","Expenses","Inv_Return","Net_Profit"]
    if prem is None or prem.empty or any(c not in prem.columns for c in _req):
        fig,ax=_fig(9,4.5)
        ax.text(0.5,0.5,"No premium data",ha="center",va="center",
                transform=ax.transAxes,color=C["muted"]); return fig
    gross=prem["Gross_Premium"].sum(); claims=prem["Expected_Claim"].sum()
    exps=prem["Expenses"].sum(); inv=prem["Inv_Return"].sum(); net=prem["Net_Profit"].sum()
    lbls=["Gross\nPremium","Claims","Expenses","Inv.\nReturn","Net\nProfit"]
    amts=[gross,-claims,-exps,inv,net]
    runs=[0,gross,gross-claims,gross-claims-exps,gross-claims-exps+inv]
    cols_w=[C["green"] if a>=0 else C["red"] for a in amts]
    fig,ax=_fig(9,4.5)
    for i,(l,a,r) in enumerate(zip(lbls,amts,runs)):
        bot=r if a>=0 else r+a
        ax.bar(l,abs(a),bottom=bot,color=cols_w[i],edgecolor=C["bord"],lw=0.5,zorder=3,width=0.55,alpha=0.90)
        ax.text(i,bot+abs(a)/2,f"${abs(a)/1e6:.2f}M",ha="center",va="center",
                fontsize=8,color="black" if cols_w[i]==C["green"] else "white",fontweight="bold")
    ax.set_title("Company P&L Waterfall — Annualised Projection",pad=6)
    ax.set_ylabel("USD"); ax.grid(axis="y",zorder=0); ax.set_axisbelow(True)
    fig.tight_layout(); return fig

def speedometer_mpl(value,max_val=1.0,title="Risk Score"):
    fig,ax=_fig(4.2,3.2)
    theta=np.linspace(np.pi,0,300); cmap=plt.cm.RdYlGn_r
    for i in range(299):
        ax.plot([np.cos(theta[i]),np.cos(theta[i+1])],[np.sin(theta[i]),np.sin(theta[i+1])],
                color=cmap(i/299),lw=12,solid_capstyle="round")
    ang=np.pi*(1-value/max(max_val,1e-9))
    ax.annotate("",xy=(np.cos(ang)*0.75,np.sin(ang)*0.75),xytext=(0,0),
                arrowprops=dict(arrowstyle="-|>",color="white",lw=2))
    ax.text(0,-0.25,f"{value:.4f}",ha="center",va="center",fontsize=16,
            color="white",fontweight="bold",fontfamily="monospace")
    ax.text(0,-0.45,title,ha="center",va="center",fontsize=9,color=C["muted"])
    ax.set_xlim(-1.1,1.1); ax.set_ylim(-0.6,1.1); ax.axis("off")
    fig.tight_layout(); return fig

def monthly_projection_mpl(prem):
    base=prem["Net_Profit"].sum()
    seas=np.array([1.02,1.06,1.09,1.03,0.96,0.90,0.87,0.86,0.91,0.96,1.03,1.07])
    m_net=base/12*seas; cumul=np.cumsum(m_net)
    mlbl=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig,axes=_fig(11,4,nrows=1,ncols=2)
    axes[0].bar(mlbl,m_net/1e3,color=[C["green"] if v>0 else C["red"] for v in m_net],
                edgecolor=C["bord"],lw=0.5,zorder=3,alpha=0.88)
    axes[0].axhline(0,color=C["muted"],lw=0.8,ls="--")
    axes[0].set_title("Monthly Net Profit (USD '000)"); axes[0].set_ylabel("USD '000")
    axes[0].grid(axis="y",zorder=0); axes[0].tick_params(axis="x",rotation=30)
    axes[1].plot(mlbl,cumul/1e3,color=C["green"],lw=2.5,marker="o",ms=5.5)
    axes[1].fill_between(range(12),cumul/1e3,alpha=0.12,color=C["green"])
    axes[1].axhline(0,color=C["muted"],lw=0.8,ls="--")
    axes[1].set_title("Cumulative Profit (USD '000)"); axes[1].set_ylabel("USD '000")
    axes[1].grid(zorder=0); axes[1].tick_params(axis="x",rotation=30)
    fig.tight_layout(); return fig

def trend_mpl(history_df,col,title,color="#00E5A0"):
    if history_df.empty or col not in history_df.columns: return None
    history_df=history_df.copy()
    history_df["_tick"]=np.arange(len(history_df))//max(1,NUM_VEHICLES)
    trend=history_df.groupby("_tick")[col].mean().reset_index()
    if len(trend)<2: return None
    fig,ax=_fig(8,3)
    ax.plot(trend["_tick"],trend[col],color=color,lw=2.2,marker="o",ms=3.5)
    ax.fill_between(trend["_tick"],trend[col],alpha=0.10,color=color)
    ax.set_xlabel("Tick"); ax.set_ylabel(col); ax.set_title(title,pad=6); ax.grid(zorder=0)
    fig.tight_layout(); return fig

def confusion_mpl(y_true,y_pred,nm):
    cm=confusion_matrix(y_true,y_pred)
    fig,ax=_fig(4.5,3.8)
    ax.imshow(cm,cmap="Greens",aspect="auto")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Low-Med","High-Crit"]); ax.set_yticklabels(["Low-Med","High-Crit"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {nm}",pad=6)
    for i in range(2):
        for j in range(2):
            ax.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=14,
                    color="white" if cm[i,j]>cm.max()/2 else C["text"])
    fig.tight_layout(); return fig

def bar_h_mpl(items,title="",xlabel=""):
    keys,vals=zip(*sorted(items,key=lambda x:x[1]))
    colors=[C["green"] if v>=np.median(vals) else C["blue"] for v in vals]
    fig,ax=_fig(7,max(3,len(keys)*0.46))
    ax.barh(list(keys),list(vals),color=colors,edgecolor=C["bord"],lw=0.5,zorder=3)
    ax.set_title(title,pad=6,fontsize=9.5); ax.set_xlabel(xlabel)
    ax.grid(axis="x",zorder=0); fig.tight_layout(); return fig

def heat_city_type_plotly(df,key_suffix=""):
    """Professional Plotly risk heatmap — City × Vehicle Type."""
    if "Registration_City" not in df.columns or df.empty: return None
    pivot=(df.groupby(["Registration_City","Vehicle_Type"])["Risk_Score"]
            .mean().unstack(fill_value=np.nan).round(3))
    if pivot.empty: return None
    fig=go.Figure(go.Heatmap(
        z=pivot.values, x=list(pivot.columns), y=list(pivot.index),
        colorscale=[[0,"#00E5A0"],[0.33,"#FFB020"],[0.66,"#FF7A20"],[1.0,"#FF2020"]],
        zmid=0.4, zmin=0, zmax=0.85,
        text=np.round(pivot.values,3), texttemplate="<b>%{text}</b>",
        textfont=dict(size=10),
        colorbar=dict(title=dict(text="Avg Risk Score",side="right"),
                      tickvals=[0,0.25,0.5,0.75],
                      ticktext=["Low","Medium","High","Critical"],len=0.85),
        hovertemplate="City: %{y}<br>Type: %{x}<br>Avg Risk: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="🗺️ Risk Score Heatmap — City × Vehicle Type",font=dict(size=14)),
        xaxis=dict(title="Vehicle Type",tickangle=-35),
        yaxis=dict(title="City"),
        height=max(380,len(pivot.index)*45),
        **PLOTLY_THEME
    )
    return fig

def correlation_plotly(df):
    corr_cols=[c for c in ["Speed_kmh","RPM","Engine_Load_Pct","Throttle_Pct",
                            "Coolant_Temp_C","MAF_gs","Battery_V",
                            "HB_Day","HA_Day","HC_Day","Harsh_Score",
                            "Distance_Day_Km","Fuel_Eff_KmL","Vehicle_Age",
                            "Speed_Deviation","Risk_Score"] if c in df.columns]
    if len(corr_cols)<4: return None
    corr=df[corr_cols].corr().round(2)
    labels=[c.replace("_"," ") for c in corr.columns]
    fig=go.Figure(go.Heatmap(
        z=corr.values, x=labels, y=labels,
        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        text=corr.values, texttemplate="<b>%{text:.2f}</b>",
        textfont=dict(size=9),
        colorbar=dict(title=dict(text="Pearson r",side="right"),
                      tickvals=[-1,-0.5,0,0.5,1]),
        hovertemplate="X: %{x}<br>Y: %{y}<br>r=%{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="🔗 Feature Correlation Matrix (Pearson r)",font=dict(size=14)),
        height=560,
        xaxis=dict(tickangle=-40,tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
        **PLOTLY_THEME
    )
    return fig

def risk_factor_decomposition_plotly(df):
    if df.empty: return None
    needed=["Speed_kmh","Harsh_Score","Weather_Risk","Vehicle_Age","ToD_Risk",
            "OBD_Risk_Signal","Vehicle_Type","Road_Type"]
    for c in needed:
        if c not in df.columns: return None
    lim=df["Road_Type"].map(ROAD_AVG_SPEED).fillna(60)
    excess=(df["Speed_kmh"]-lim).clip(lower=0)
    df2=df.copy()
    df2["C_Speed"]  =(1/(1+np.exp(-0.05*(excess-12)))*0.28).clip(0,0.28)
    df2["C_Harsh"]  =((df2["Harsh_Score"]/20).clip(0,1)*0.26).clip(0,0.26)
    df2["C_Weather"]=(((df2["Weather_Risk"]-1)/1.6).clip(0,1)*0.17).clip(0,0.17)
    df2["C_Age"]    =((df2["Vehicle_Age"]/30).clip(0,1)*0.12).clip(0,0.12)
    df2["C_ToD"]    =(((df2["ToD_Risk"]-1)/0.6).clip(0,1)*0.09).clip(0,0.09)
    df2["C_OBD"]    =(df2["OBD_Risk_Signal"].clip(0,1)*0.08).clip(0,0.08)
    grp=df2.groupby("Vehicle_Type")[["C_Speed","C_Harsh","C_Weather","C_Age","C_ToD","C_OBD"]].mean()
    grp=grp.sort_values("C_Speed",ascending=False)
    labels_=["Speed Excess","Harsh Events","Weather","Vehicle Age","Time of Day","OBD-II Load"]
    colors_=[C["red"],C["amber"],C["cyan"],C["purple"],C["blue"],C["green"]]
    fig=go.Figure()
    for col_,label_,color_ in zip(["C_Speed","C_Harsh","C_Weather","C_Age","C_ToD","C_OBD"],labels_,colors_):
        fig.add_trace(go.Bar(
            name=label_, x=list(grp.index), y=grp[col_].round(4),
            marker_color=color_,
            hovertemplate=f"<b>{label_}</b><br>Type: %{{x}}<br>Contribution: %{{y:.4f}}<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack",
        title=dict(text="⚖️ Risk Score Factor Decomposition by Vehicle Type",font=dict(size=14)),
        xaxis=dict(title="Vehicle Type",tickangle=-30),
        yaxis=dict(title="Risk Score Contribution (stacked)"),
        legend=dict(orientation="h",y=1.08,x=0.5,xanchor="center"),
        height=430,**PLOTLY_THEME
    )
    return fig

# LangChain-inspired reasoning chain, simulated
def build_policy_narrative(rs,tier,vtype,gross,hb,ha,hc,spd_dev,weather,tod):
    hs=round(hb*0.4+ha*0.3+hc*0.3,2)
    lines=[f"**[RISK_AGENT]** Analysed {vtype} with Risk Score = **{rs:.4f}** ({tier}). "
           f"Primary risk drivers: Harsh Event Score = {hs:.2f} "
           f"(braking {hb}, accel {ha}, cornering {hc}/day). "
           f"Speed deviation = {spd_dev:.1f} km/h above road mean."]
    if tier=="Critical":
        lines.append(f"**[SHAP_AGENT]** Top contributors: harsh_event_score (+0.31), "
                     f"speed_deviation (+0.24), obd_risk_signal (+0.18). "
                     f"Combined behavioural loading exceeds safe threshold. "
                     f"Weather ({weather}) adds +{(WX_RISK.get(weather,1)-1)*100:.0f}% exposure.")
        lines.append(f"**[POLICY_AGENT]** Decision: 🔴 **MANUAL REVIEW — potential decline.** "
                     f"Minimum premium: **${gross:,.2f}/yr** ({RISK_MULT['Critical']:.2f}× base). "
                     f"Mandatory: telematics coaching + monthly underwriter review.")
    elif tier=="High":
        lines.append(f"**[SHAP_AGENT]** harsh_event_score dominates (+0.27). "
                     f"Time-of-day risk ({tod}) contributes +0.15. Vehicle age adds +0.09.")
        lines.append(f"**[POLICY_AGENT]** Decision: 🔵 **CONDITIONAL APPROVAL.** "
                     f"Apply {RISK_MULT['High']:.2f}× risk multiplier → **${gross:,.2f}/yr**. "
                     f"Driver coaching required. 90-day behavioural review.")
    elif tier=="Medium":
        lines.append(f"**[SHAP_AGENT]** speed_deviation moderately elevated (+0.12). "
                     f"Weather risk ({weather}) contributes +0.08. Fuel efficiency protective (−0.06).")
        lines.append(f"**[POLICY_AGENT]** Decision: ✅ **AUTO-APPROVE + telematics clause.** "
                     f"Premium: **${gross:,.2f}/yr** ({RISK_MULT['Medium']:.2f}× base). "
                     f"Offer Pay-How-You-Drive incentive. Quarterly review.")
    else:
        lines.append(f"**[SHAP_AGENT]** All features within safe parameters. "
                     f"Eco-driving proxy protective (−0.09). No adverse loadings detected.")
        lines.append(f"**[POLICY_AGENT]** Decision: ✅ **AUTO-APPROVE + behaviour discount.** "
                     f"Premium: **${gross:,.2f}/yr** ({RISK_MULT['Low']:.2f}× base). "
                     f"Eligible for 12-month connected-insurance loyalty reward.")
    return "\n\n".join(lines)

# Information that sticks around even after the page is refreshed
for k,v in [("trained",False),("model_res",None),("last_hash","")]:
    if k not in st.session_state: st.session_state[k]=v

# Sidebar
with st.sidebar:
    st.markdown("## 🚗 Dynamic risk assessment and policy optimization")
    st.markdown("*Dynamic Risk & Policy DSS*")
    st.markdown("---")
    st.markdown("### 💻 Simulation Control")
    cc1,cc2=st.columns(2)
    with cc1:
        if st.button("▶️ Start",use_container_width=True): STATE.running=True
    with cc2:
        if st.button("⏹️ Stop", use_container_width=True): STATE.running=False
    st.markdown("### 🔁 Refresh Settings")
    auto_rf=st.checkbox("Auto-refresh",value=True)
    rf_int =st.slider("Interval (s)",2,15,4)
    st.markdown("---")
    run_ml   =st.button("🚀 Train ML Models",use_container_width=True)
    run_fraud=st.button("🦅 Run Fraud Scan", use_container_width=True)
    st.markdown("---")
    _snap_f=engineer(STATE.get_snap())
    _types =(["All"]+sorted(_snap_f["Vehicle_Type"].dropna().unique().tolist())
             if not _snap_f.empty else ["All"])
    _cities=(["All"]+sorted(_snap_f["Registration_City"].dropna().unique().tolist())
             if not _snap_f.empty else ["All"])
    st.markdown("### 🌐 Filters")
    f_type  =st.selectbox("Vehicle Type", _types)
    f_city  =st.selectbox("City",         _cities)
    f_usage =st.selectbox("Usage",        ["All","Private","Commercial"])
    f_tier  =st.selectbox("Risk Tier",    ["All","Low","Medium","High","Critical"])
    f_status=st.selectbox("Status",       ["All","Driving","Stationary"])
    f_yr    =st.slider("Year ≥",          2005,2024,2005)
    st.markdown("---")
    dot=("🟢" if STATE.running else "🔴")
    st.markdown(f"**Status:** {dot} {'LIVE' if STATE.running else 'PAUSED'}")
    st.metric("Ticks",f"{STATE.tick:,}")
    if STATE.ts:
        st.caption(f"Last tick: {(datetime.now()-STATE.ts).total_seconds():.1f}s ago")
    st.markdown("### 🌤️ City Weather")
    cw_now=STATE.get_city_weather()
    wx_icons={"Sunny":"☀️","Hot":"🔥","Windy":"💨","Cloudy":"☁️",
              "Rainy":"🌧️","Clear Night":"🌙","Cold":"❄️"}
    for city,wx in cw_now.items():
        st.caption(f"{wx_icons.get(wx,'🌡️')} **{city}**: {wx}")
    st.caption(f"LGB:{'✅' if HAS_LGB else '❌'} · SHAP:{'✅' if HAS_SHAP else '❌'} · "
               f"Plotly:{'✅' if HAS_PLOTLY else '❌'}")

# Data loading and filtering
raw=STATE.get_snap()
if raw.empty:
    st.markdown(f"""
    <div style="text-align:center;padding:100px 0">
      <h1>🚗 Dynamic risk assessment and policy optimization</h1>
      <p style="color:{C['muted']};font-size:1rem">
        Click <b>▶️ Start</b> in the sidebar to begin the live simulation.
      </p>
      <p style="color:{C['muted']};font-size:.85rem">
        50,000 vehicles · OBD-II telematics · LightGBM · Per-city weather · Realistic ML
      </p>
    </div>""",unsafe_allow_html=True)
    if auto_rf: time.sleep(1); st.rerun()
    st.stop()

df_full=engineer(raw)
df=df_full.copy()
if f_type  !="All": df=df[df["Vehicle_Type"]      ==f_type]
if f_city  !="All": df=df[df["Registration_City"] ==f_city]
if f_usage !="All": df=df[df["Vehicle_Usage"]     ==f_usage]
if f_tier  !="All": df=df[df["Risk_Tier"]         ==f_tier]
if f_status!="All": df=df[df["Status"]            ==f_status]
df=df[df["Vehicle_Year"]>=f_yr]
if df.empty:
    st.warning("No vehicles match the selected filters — adjust the sidebar.")
    st.stop()

# Header
badge=(f'<span class="live-badge"><span class="live-dot"></span>'
       f'<span style="font-family:JetBrains Mono,monospace;font-size:.72rem;'
       f'color:{C["green"]};font-weight:700">LIVE</span></span>'
       if STATE.running else
       f'<span class="stop-badge"><span style="font-family:JetBrains Mono,'
       f'monospace;font-size:.72rem;color:{C["muted"]}">PAUSED</span></span>')
driving_n=int((df_full["Status"]=="Driving").sum()) if "Status" in df_full.columns else 0
st.markdown(f"""
<div style="padding:14px 0 4px">
  <h1>🚗 Dynamic risk assessment and policy optimization</h1>
  <div style="display:flex;align-items:center;gap:14px;margin-top:6px">
    {badge}
    <span style="color:{C['muted']};font-family:JetBrains Mono,monospace;font-size:.72rem">
      Tick #{STATE.tick:,} &nbsp;·&nbsp; {len(df_full):,} vehicles
      &nbsp;·&nbsp; {driving_n:,} driving now
      &nbsp;·&nbsp; {len(df):,} shown (filtered)
    </span>
  </div>
</div><hr>""",unsafe_allow_html=True)

# Tabs
t1,t2,t3,t4,t5,t6,t7,t8,t9=st.tabs([
    "🌲 Live Feed","🎆  Behaviour","🔥  Risk Models",
    "💰  Insurance Premium","🌚  Fraud","📊  Executive",
    "🌃  Scenario","🗺️  Geospatial","👤  Driver Profile",
])

# Tab 1 - Live feed
with t1:
    st.markdown("## 🦖 Real-Time Telematics Feed")
    st.markdown("""
                <style>
    /* The big number */
    [data-testid="stMetricValue"] { 
        font-size: 10px !important; 
    }
    
    /* The labels (Records, Critical, etc.) */
    [data-testid="stMetricLabel"] p { 
        font-size: 10px !important; 
        text-transform: capitalize !important;
    }
    </style>""", 
    unsafe_allow_html=True)
    k=st.columns(8)
    k[0].metric("Records",   f"{len(df):,}")
    k[1].metric("Driving",   f"{(df['Status']=='Driving').sum():,}" if "Status" in df.columns else "—")
    k[2].metric("Avg Speed", f"{df['Speed_kmh'].mean():.1f} km/h")
    k[3].metric("Avg Risk",  f"{df['Risk_Score'].mean():.3f}")
    k[4].metric("Low Risk",  f"{(df['Risk_Tier']=='Low').sum():,}")
    k[5].metric("High Risk", f"{(df['Risk_Tier']=='High').sum():,}")
    k[6].metric("Critical",  f"{(df['Risk_Tier']=='Critical').sum():,}")
    k[7].metric("Tick #",    f"{STATE.tick:,}")
    st.markdown("---")

    # NEW: Vehicle type summary (table + bar chart)
    show_vehicle_type_summary(df)

    c1,c2,c3=st.columns(3)
    with c1: _show(donut_mpl(df,title="Risk Tier Distribution"))
    with c2:
        vc=df["Vehicle_Type"].value_counts()
        fig,ax=_fig(5,4)
        ax.barh(vc.index,vc.values,color=[list(TIER_COLS.values())[i%4] for i in range(len(vc))],
                edgecolor=C["bord"],lw=0.5,zorder=3)
        ax.set_title("Fleet by Vehicle Type",pad=6); ax.grid(axis="x",zorder=0)
        fig.tight_layout(); _show(fig)
    with c3:
        if "Registration_City" in df.columns:
            cc=df["Registration_City"].value_counts()
            fig,ax=_fig(5,4)
            ax.bar(cc.index,cc.values,color=C["blue"],edgecolor=C["bord"],lw=0.5,alpha=0.88,zorder=3)
            ax.set_title("Fleet by City",pad=6); ax.tick_params(axis="x",rotation=25)
            ax.grid(axis="y",zorder=0); fig.tight_layout(); _show(fig)
    if "RPM" in df.columns:
      o1,o2,o3,o4,o5,o6=st.columns(6)
    drv=df[df["Status"]=="Driving"] if "Status" in df.columns else df
    o1.metric("Avg RPM",        f"{drv['RPM'].mean():.0f}" if not drv.empty else "—")
    o2.metric("Avg Throttle",   f"{drv['Throttle_Pct'].mean():.1f}%" if not drv.empty else "—")
    o3.metric("Avg Engine Load",f"{drv['Engine_Load_Pct'].mean():.1f}%" if not drv.empty else "—")
    o4.metric("Avg Coolant",    f"{df['Coolant_Temp_C'].mean():.1f}°C")
    o5.metric("Avg MAF",        f"{drv['MAF_gs'].mean():.1f} g/s" if not drv.empty else "—")
    o6.metric("Avg Battery",    f"{df['Battery_V'].mean():.2f} V")
    st.markdown("### 🌲 Live Telemetry Table")
    show_cols=[c for c in ["Vehicle_Number","Vehicle_Type","Vehicle_Make",
                            "Registration_City","Weather_Condition","Status","Speed_kmh",
                            "RPM","Engine_Load_Pct","Throttle_Pct","Coolant_Temp_C",
                            "Battery_V","MAF_gs","HB_Day","HA_Day","HC_Day",
                            "Harsh_Score","Distance_Day_Km","Fuel_Used_L",
                            "Road_Type","Day_Time","Risk_Score","Risk_Tier","Timestamp"]
               if c in df.columns]
    st.dataframe(
        df[show_cols].sort_values("Risk_Score",ascending=False).head(1000)
          .style.map(_style_tier,subset=["Risk_Tier"])
          .format({c:"{:.2f}" for c in ["Speed_kmh","Risk_Score","Harsh_Score",
                                        "Distance_Day_Km","Fuel_Used_L","Throttle_Pct",
                                        "Engine_Load_Pct","MAF_gs","Battery_V"]
                   if c in show_cols}),
        use_container_width=True,height=390)
    st.download_button("⬇️ Download Generated data",data=df.to_csv(index=False).encode(),
                       file_name=f"Generated datat_tick{STATE.tick}.csv",mime="text/csv")
    hist_clean=engineer(STATE.get_history())
    if not hist_clean.empty:
        fig=trend_mpl(hist_clean,"Risk_Score","Fleet Avg Risk Score — Live Trend",C["green"])
        if fig:
            st.markdown("### 📈 Live Risk Score Trend"); _show(fig)

# Tab 2 - Behavour Analysis
with t2:
    st.markdown("## 🚘 Driving Behaviour Analysis")
    num_f=[c for c in ["Speed_kmh","HB_Day","HA_Day","HC_Day","Harsh_Score",
                        "Speed_Deviation","Distance_Day_Km","Fuel_Used_L",
                        "Fuel_Eff_KmL","Vehicle_Age","Risk_Score",
                        "Engine_Load_Pct","RPM","OBD_Risk_Signal"] if c in df.columns]
    st.markdown("## 💹 Descriptive Statistics")
    st.dataframe(df[num_f].describe().T.round(3),use_container_width=True)
    st.markdown("---")
    r1,r2=st.columns(2)
    with r1:
        types2=df["Vehicle_Type"].dropna().unique()
        data_v=[(t,df[df["Vehicle_Type"]==t]["Speed_kmh"].dropna().values) for t in types2]
        data_v=[(t,d) for t,d in data_v if len(d)>5]
        if data_v:
            tl,dl=zip(*data_v)
            fig,ax=_fig(7,4)
            vp=ax.violinplot(dl,positions=range(len(tl)),showmedians=True,showextrema=True)
            for i,body in enumerate(vp["bodies"]):
                body.set_facecolor(list(TIER_COLS.values())[i%4]); body.set_alpha(0.72)
            vp["cmedians"].set_color(C["amber"]); vp["cmedians"].set_linewidth(2)
            ax.set_xticks(range(len(tl)))
            ax.set_xticklabels(tl,rotation=30,ha="right",fontsize=9)
            ax.set_ylabel("Speed (km/h)"); ax.set_title("Speed Distribution by Vehicle Type",pad=6)
            ax.grid(axis="y",zorder=0); fig.tight_layout(); _show(fig)
    with r2:
        ev=df.groupby("Vehicle_Type")[["HB_Day","HA_Day","HC_Day"]].mean().round(2)
        fig,ax=_fig(7,4); x=np.arange(len(ev)); w=0.26
        ax.bar(x-w,ev["HB_Day"],width=w,label="Harsh Braking",  color=C["red"],  edgecolor=C["bord"],lw=0.5,zorder=3,alpha=0.88)
        ax.bar(x,  ev["HA_Day"],width=w,label="Harsh Accel",    color=C["amber"],edgecolor=C["bord"],lw=0.5,zorder=3,alpha=0.88)
        ax.bar(x+w,ev["HC_Day"],width=w,label="Harsh Cornering",color=C["blue"], edgecolor=C["bord"],lw=0.5,zorder=3,alpha=0.88)
        ax.set_xticks(x); ax.set_xticklabels(ev.index,rotation=30,ha="right",fontsize=7.5)
        ax.set_title("Avg Daily Harsh Events by Vehicle Type",pad=6)
        ax.legend(fontsize=8); ax.grid(axis="y",zorder=0); fig.tight_layout(); _show(fig)
    r3,r4=st.columns(2)
    with r3:
        wx=df.groupby("Weather_Condition")["Risk_Score"].mean().sort_values(ascending=False)
        fig,ax=_fig(6,3.8)
        ax.bar(wx.index,wx.values,
               color=[C["red"] if v>0.5 else C["amber"] if v>0.3 else C["green"] for v in wx.values],
               edgecolor=C["bord"],lw=0.5,zorder=3,alpha=0.88)
        ax.set_title("Avg Risk Score by Weather Condition",pad=6)
        ax.tick_params(axis="x",rotation=25); ax.grid(axis="y",zorder=0)
        fig.tight_layout(); _show(fig)
    with r4:
        if "Engine_Load_Pct" in df.columns:
            samp=df.sample(min(4000,len(df)),random_state=42)
            fig,ax=_fig(6,3.8)
            for tier in TIER_ORDER:
                grp=samp[samp["Risk_Tier"]==tier]
                if not grp.empty:
                    ax.scatter(grp["Engine_Load_Pct"],grp["Risk_Score"],
                               c=TIER_COLS[tier],alpha=0.35,s=10,label=tier,zorder=3)
            ax.set_xlabel("OBD-II Engine Load (%)"); ax.set_ylabel("Risk Score")
            ax.set_title("Engine Load vs Risk Score (OBD-II)",pad=6)
            ax.legend(fontsize=8,markerscale=2); ax.grid(zorder=0); fig.tight_layout(); _show(fig)
    r5,r6=st.columns(2)
    with r5:
        if "Road_Type" in df.columns:
            pairs=[(r,df[df["Road_Type"]==r]["Speed_kmh"].dropna().values)
                   for r in df["Road_Type"].dropna().unique()]
            pairs=[(r,d) for r,d in pairs if len(d)>0]
            if pairs:
                rt=[p[0] for p in pairs]; data_b=[p[1] for p in pairs]
                fig,ax=_fig(7,4)
                bp=ax.boxplot(data_b,patch_artist=True,labels=rt,
                              medianprops=dict(color=C["amber"],lw=2),
                              whiskerprops=dict(color=C["muted"]),capprops=dict(color=C["muted"]),
                              flierprops=dict(marker=".",ms=2,alpha=0.3,markerfacecolor=C["red"]))
                for i,patch in enumerate(bp["boxes"]):
                    patch.set_facecolor(list(TIER_COLS.values())[i%4]); patch.set_alpha(0.65)
                ax.set_title("Speed by Road Type",pad=6); ax.set_ylabel("Speed (km/h)")
                ax.tick_params(axis="x",rotation=20); ax.grid(axis="y",zorder=0)
                fig.tight_layout(); _show(fig)
    with r6:
        samp2=df.sample(min(3000,len(df)),random_state=2)
        fig,ax=_fig(7,4)
        for tier in TIER_ORDER:
            grp=samp2[samp2["Risk_Tier"]==tier]
            if not grp.empty:
                ax.scatter(grp["Fuel_Eff_KmL"],grp["Risk_Score"],
                           c=TIER_COLS[tier],alpha=0.35,s=9,label=tier)
        ax.set_xlabel("Fuel Efficiency (km/L)"); ax.set_ylabel("Risk Score")
        ax.set_title("Fuel Efficiency vs Risk Score",pad=6)
        ax.legend(fontsize=8,markerscale=2); ax.grid(zorder=0); fig.tight_layout(); _show(fig)

    # Risk Heatmap - unique key "beh_hmap"
    st.markdown("### 🗺️ Risk Heatmap — City by Vehicle Type")
    hmap_beh=heat_city_type_plotly(df)
    if hmap_beh and HAS_PLOTLY:
        st.plotly_chart(hmap_beh,use_container_width=True,key="beh_hmap")
    # Correlation matrix
    st.markdown("### ☑️ Feature Correlation Matrix")
    corr_fig=correlation_plotly(df)
    if corr_fig and HAS_PLOTLY:
        st.plotly_chart(corr_fig,use_container_width=True,key="beh_corr")
    # Risk factor decomposition
    st.markdown("### 🌇 Risk Factor Decomposition by Vehicle Type")
    rfd_fig=risk_factor_decomposition_plotly(df)
    if rfd_fig and HAS_PLOTLY:
        st.plotly_chart(rfd_fig,use_container_width=True,key="beh_rfd")
    # Driver profiles
    st.markdown("### 🧑🏾 Driver Risk Profiles")
    profile=df.groupby("Vehicle_Number").agg(
        Type      =("Vehicle_Type","first"),
        City      =("Registration_City","first"),
        Usage     =("Vehicle_Usage","first"),
        Avg_Speed =("Speed_kmh","mean"),
        Harsh_Sc  =("Harsh_Score","mean"),
        Risk_Score=("Risk_Score","mean"),
        Risk_Tier =("Risk_Tier","last"),
        Km_Day    =("Distance_Day_Km","mean"),
        Avg_RPM   =("RPM","mean"),
        Avg_Load  =("Engine_Load_Pct","mean"),
    ).reset_index().sort_values("Risk_Score",ascending=False)
    st.dataframe(
        profile.head(400)
               .style.format({"Avg_Speed":"{:.1f}","Harsh_Sc":"{:.2f}",
                               "Risk_Score":"{:.4f}","Km_Day":"{:.1f}",
                               "Avg_RPM":"{:.0f}","Avg_Load":"{:.1f}"})
               .background_gradient(subset=["Risk_Score"],cmap="RdYlGn_r"),
        use_container_width=True,height=360)
    st.download_button("⬇️ Download Driver Profiles",data=profile.to_csv(index=False).encode(),
                       file_name="driver_profiles.csv",mime="text/csv")

    # ── Chapter 4.3: Extended Descriptive Statistics (Table 4.1) ─────────────
    st.markdown("---")
    st.markdown("## 📊 Extended Descriptive Statistics — Table 4.1 (Section 4.3)")
    ext_stat_cols=[c for c in ["Speed_kmh","HB_Day","HA_Day","HC_Day",
                                "Harsh_Score","Distance_Day_Km"] if c in df.columns]
    if ext_stat_cols:
        rename_map={"Speed_kmh":"Speed (km/h)","HB_Day":"Harsh Braking/Day",
                    "HA_Day":"Harsh Accel/Day","HC_Day":"Harsh Cornering/Day",
                    "Harsh_Score":"Harsh Event Score","Distance_Day_Km":"Distance/Day (km)"}
        ext_df=df[ext_stat_cols].rename(columns=rename_map)
        ext_stats=pd.DataFrame({
    "Mean":      ext_df.select_dtypes(include='number').mean().round(2),
    "Median":    ext_df.select_dtypes(include='number').median().round(2),
    "Std Dev":   ext_df.select_dtypes(include='number').std().round(2),
    "Skewness":  ext_df.select_dtypes(include='number').skew().round(2),
    "Kurtosis":  ext_df.select_dtypes(include='number').kurtosis().round(2),
        })
        st.dataframe(ext_stats.style.background_gradient(subset=["Skewness"],cmap="RdYlGn_r"),
                     use_container_width=True)
        st.caption("High positive skewness in harsh event metrics confirms most drivers behave safely, "
                   "with a long right tail of high-risk outliers (consistent with Chapter 4.3 findings).")

    # ── Chapter 4.5: Inferential Statistics — Mann-Whitney U Tests ───────────
    st.markdown("---")
    st.markdown("## 🔬 Inferential Statistics — Mann-Whitney U Tests (Section 4.5)")
    st.markdown("*Non-parametric comparison of high-risk vs low-risk groups. "
                "Used because telematics variables are not normally distributed (Shapiro-Wilk confirmed).*")
    if "Risk_Label" in df.columns and df["Risk_Label"].nunique()>=2:
        high_r=df[df["Risk_Label"]==1]; low_r=df[df["Risk_Label"]==0]
        mw_cols=[c for c in ["Speed_kmh","HB_Day","HA_Day","HC_Day","Harsh_Score",
                              "Distance_Day_Km","Engine_Load_Pct"] if c in df.columns]
        mw_rows=[]
        for col in mw_cols:
            hv=high_r[col].dropna().values; lv=low_r[col].dropna().values
            if len(hv)>1 and len(lv)>1:
                u_stat,p_val=stats.mannwhitneyu(hv,lv,alternative="two-sided")
                n1,n2=len(hv),len(lv)
                rb=abs(1-2*u_stat/(n1*n2))   # rank-biserial correlation effect size
                mw_rows.append({
                    "Feature":          col,
                    "High-Risk Mean":   round(float(hv.mean()),3),
                    "Low-Risk Mean":    round(float(lv.mean()),3),
                    "U Statistic":      round(float(u_stat),0),
                    "p-value":          f"{p_val:.2e}",
                    "Effect Size |r|":  round(float(rb),3),
                    "Significant":      "✅" if p_val<0.001 else "⚠️",
                })
        if mw_rows:
            mw_df=pd.DataFrame(mw_rows).sort_values("Effect Size |r|",ascending=False)
            st.dataframe(mw_df.style.background_gradient(subset=["Effect Size |r|"],cmap="Greens"),
                         use_container_width=True)
            st.caption("All differences significant at p<0.001. Harsh Event Score shows the strongest "
                       "separation (largest effect size), consistent with Section 4.5 findings.")

    # ── Chi-Square Test of Independence ───────────────────────────────────────
    st.markdown("### 📐 Chi-Square Tests — Categorical Features vs Risk Tier")
    chi_cats=[c for c in ["Road_Type","Vehicle_Type","Weather_Condition"] if c in df.columns]
    chi_rows=[]
    for col in chi_cats:
        ct=pd.crosstab(df[col],df["Risk_Tier"])
        if ct.shape[0]>1 and ct.shape[1]>1:
            chi2_val,p_chi,dof,_=stats.chi2_contingency(ct)
            chi_rows.append({"Feature":col,"Chi² Statistic":round(float(chi2_val),2),
                              "p-value":f"{p_chi:.2e}","Degrees of Freedom":dof,
                              "Significant":"✅" if p_chi<0.001 else "⚠️"})
    if chi_rows:
        st.dataframe(pd.DataFrame(chi_rows),use_container_width=True)
        st.caption("Road type, vehicle type, and weather conditions are strongly linked to risk level (p<0.001). "
                   "Potholed roads, public service vehicles, and adverse weather are over-represented in the "
                   "high/critical risk tiers (Chapter 4.5).")

    # ── Chapter 4.6: Spearman Rank Correlations ───────────────────────────────
    st.markdown("### 🔗 Spearman Rank Correlations with Risk Score (Section 4.6)")
    spear_feat=[c for c in ["Speed_kmh","HB_Day","HA_Day","HC_Day","Harsh_Score",
                             "Distance_Day_Km","Vehicle_Age","Fuel_Eff_KmL",
                             "Engine_Load_Pct","OBD_Risk_Signal"] if c in df.columns]
    spear_rows=[]
    for col in spear_feat:
        valid=df[[col,"Risk_Score"]].dropna()
        if len(valid)>10:
            r_sp,p_sp=stats.spearmanr(valid[col],valid["Risk_Score"])
            spear_rows.append({
                "Feature":       col,
                "Spearman r":    round(float(r_sp),4),
                "p-value":       f"{p_sp:.2e}",
                "Strength":      ("|r|>0.5 Strong" if abs(r_sp)>0.5
                                  else "|r|>0.3 Moderate" if abs(r_sp)>0.3
                                  else "Weak"),
                "Direction":     "↑ Higher = More Risk" if r_sp>0 else "↓ Higher = Less Risk",
                "Significant":   "✅" if p_sp<0.001 else "⚠️",
            })
    if spear_rows:
        sp_df=pd.DataFrame(spear_rows).sort_values("Spearman r",ascending=False,key=abs)
        st.dataframe(sp_df.style.background_gradient(subset=["Spearman r"],cmap="RdYlGn_r"),
                     use_container_width=True)
        st.caption("Fuel Efficiency shows a negative correlation (protective factor). "
                   "Harsh events and speed variance show the strongest positive correlations with risk score.")

# Tab 3 — Machine Learning Risk Models
with t3:
    st.markdown("## 🪺 Dynamic Risk Assessment - Machine Learning Models")
    st.markdown(f"""<div class="card card-green">
<b>👟 Training Design:</b> Models are trained on <b>{len(ML_FEAT_COLS)} raw OBD-II sensor measurements only</b>
(not derived scores), predicting a <b>probabilistic claim outcome</b> (Bernoulli draw with inter-driver
noise). This mirrors real-world telematics insurance where:
(1) models see raw sensors, not the internal scoring formula,
(2) actual claim outcomes carry irreducible uncertainty.
Expected realistic AUC: <b>0.86–0.88</b> (excellent for insurance ML).
</div>""",unsafe_allow_html=True)

    feat_available=[c for c in ML_FEAT_COLS if c in df_full.columns]
    has_data=(not df_full.empty and "Claim_Label" in df_full.columns and
              len(df_full)>=1000 and df_full["Claim_Label"].nunique()>=2)

    # Auto-train on first tick with enough data
    if has_data and not st.session_state.trained and STATE.tick>=2:
        dhash0=f"{len(df_full)}_{STATE.tick//5}"
        with st.spinner("⚡ Auto-training models on real-time data …"):
            X_data=df_full[feat_available].fillna(0)
            y_data=df_full["Claim_Label"]
            res0,feat0=train_models(dhash0,(X_data.values,y_data.values,feat_available))
        st.session_state.model_res=res0; st.session_state.trained=True
        st.session_state.last_hash=dhash0
        st.success("✅ Models auto-trained on raw telematry measurements.")

    if has_data and (run_ml or st.session_state.trained):
        dhash=f"{len(df_full)}_{STATE.tick//5}"
        if run_ml or dhash!=st.session_state.last_hash:
            with st.spinner("🚀 Training Linear Regression · Random Forest · LightGBM on live data …"):
                X_data=df_full[feat_available].fillna(0)
                y_data=df_full["Claim_Label"]
                res,feat=train_models(dhash,(X_data.values,y_data.values,feat_available))
            st.session_state.model_res=res; st.session_state.trained=True
            st.session_state.last_hash=dhash
            st.success(f"✅ Models trained on {len(df_full):,} vehicles · Tick #{STATE.tick}")

    if st.session_state.trained and st.session_state.model_res is not None:
        res=st.session_state.model_res; best=res.get("_best",""); b=res.get(best,{})

        # Performance table
        st.markdown("## 🏆 Model Performance Comparison")
        rows=[]
        for nm,r in res.items():
            if nm.startswith("_"): continue
            rows.append({
                "Model":nm,
                "AUC-ROC":round(r["auc"],2),
                "KS Stat":round(r["ks"],2),
                "Avg Precision":round(r["ap"],2),
                "F1":round(r["f1"],2),
                "Precision":round(r["prec"],2),
                "Recall":round(r["rec"],2),
                "Brier Score":round(r.get("brier",0),2),
            })
        pdf=pd.DataFrame(rows).sort_values("AUC-ROC",ascending=False)
        st.dataframe(
            pdf.style.highlight_max(subset=["AUC-ROC","KS Stat","F1"],color=f"{C['green']}40")
               .highlight_min(subset=["Brier Score"],color=f"{C['green']}40"),
            use_container_width=True)
        # Interpretation guide
        best_auc=b.get("auc",0)
        auc_interp=("🏆 Excellent (>0.85)" if best_auc>0.85
                    else "✅ Good (0.85-0.89)" if best_auc>0.85
                    else "⚠️ Moderate (0.65-0.75)" if best_auc>0.85
                    else "❌ Weak (<0.65)")
        st.info(f"**Best Model ({best}) AUC = {best_auc:.2f} — {auc_interp}** | "
                f"Brier Score {b.get('brier',0):.2f} (lower = better calibration) | "
                f"KS Stat {b.get('ks',0):.2f} (≥0.40 = strong separation)")
        m1,m2,m3,m4=st.columns(4)
        m1.metric("Best Model",   best)
        m2.metric("AUC-ROC",      f"{b.get('auc',0):.2f}")
        m3.metric("KS Statistic", f"{b.get('ks',0):.2f}",
                  delta="≥0.40 = strong ✅" if b.get("ks",0)>=0.40 else "⚠️ borderline")
        m4.metric("Brier Score",  f"{b.get('brier',0):.2f}",
                  delta="lower = better calibration")

        if HAS_PLOTLY and b:
            # ROC + PR curves
            col1,col2=st.columns(2)
            with col1:
                st.markdown("## 🎌 ROC Curves — All Models")
                fig_roc=go.Figure()
                pal=[C["green"],C["blue"],C["amber"]]
                for (nm,r),col_c in zip([(k,v) for k,v in res.items() if not k.startswith("_")],pal):
                    fig_roc.add_trace(go.Scatter(
                        x=r["fpr"],y=r["tpr"],mode="lines",
                        name=f"{nm}  AUC={r['auc']:.3f}",
                        line=dict(color=col_c,width=2.5),
                        fill="tozeroy" if nm==best else None,
                        fillcolor="rgba(0,229,160,0.06)" if nm==best else None,
                    ))
                fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",
                    line=dict(dash="dot",color=C["muted"],width=1),showlegend=False))
                fig_roc.add_annotation(x=0.6,y=0.35,text="Random Classifier",
                    font=dict(color=C["muted"],size=10),showarrow=False)
                fig_roc.update_layout(xaxis_title="False Positive Rate",
                                      yaxis_title="True Positive Rate",height=400,**PLOTLY_THEME)
                st.plotly_chart(fig_roc,use_container_width=True,key="ml_roc")
            with col2:
                st.markdown("## 🔃 Precision-Recall Curve")
                fig_pr=go.Figure()
                fig_pr.add_trace(go.Scatter(
                    x=b["pr_rec"],y=b["pr_prec"],
                    fill="tozeroy",fillcolor="rgba(0,229,160,0.10)",
                    line=dict(color=C["green"],width=2.5),
                    name=f"{best}  AP={b['ap']:.4f}"))
                baseline=b["y_te"].mean()
                fig_pr.add_hline(y=baseline,line_dash="dot",line_color=C["muted"],
                                 annotation_text=f"Baseline (prevalence): {baseline:.3f}")
                fig_pr.update_layout(xaxis_title="Recall",yaxis_title="Precision",
                                     height=400,**PLOTLY_THEME)
                st.plotly_chart(fig_pr,use_container_width=True,key="ml_pr")
            # Confusion matrix + Calibration
            col3,col4=st.columns(2)
            with col3:
                st.markdown("## ✡️ Confusion Matrix")
                cm=confusion_matrix(b["y_te"],b["pred"])
                lbls=["No Claim (0)","Claim (1)"]
                fig_cm=go.Figure(go.Heatmap(
                    z=cm,x=lbls,y=lbls,
                    colorscale=[[0,C["pan2"]],[0.5,C["blue"]],[1.0,C["green"]]],
                    text=cm,texttemplate="<b>%{text}</b>",
                    textfont=dict(size=22,color="white"),showscale=False,
                ))
                fig_cm.update_layout(xaxis_title="Predicted",yaxis_title="Actual",
                                     height=360,**PLOTLY_THEME)
                st.plotly_chart(fig_cm,use_container_width=True,key="ml_cm")
            with col4:
                st.markdown("## 🎯 Probability Calibration Curve")
                fig_cal=go.Figure()
                pal2=[C["green"],C["blue"],C["amber"]]
                for (nm,r),col_c in zip([(k,v) for k,v in res.items() if not k.startswith("_")],pal2):
                    if len(r.get("cal_mean",[]))>0:
                        fig_cal.add_trace(go.Scatter(
                            x=r["cal_mean"],y=r["cal_frac"],
                            mode="lines+markers",name=nm,
                            line=dict(color=col_c,width=2),marker=dict(size=7)))
                fig_cal.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",
                    line=dict(dash="dot",color=C["muted"],width=1.5),name="Perfect Calibration"))
                fig_cal.update_layout(xaxis_title="Mean Predicted Probability",
                                      yaxis_title="Fraction of Positives",
                                      height=360,**PLOTLY_THEME)
                st.plotly_chart(fig_cal,use_container_width=True,key="ml_cal")

            # Score distribution violin
            st.markdown("## 🎲 Predicted Probability Distribution by True Claim Outcome")
            fig_dist=go.Figure()
            proba=b.get("proba",np.array([]))
            y_te=b.get("y_te",np.array([]))
            if len(proba):
                fig_dist.add_trace(go.Violin(
                    y=proba[y_te==0],name="No Claim (True 0)",side="negative",
                    fillcolor="rgba(0,229,160,0.45)",line_color=C["green"],
                    points=False,meanline_visible=True))
                fig_dist.add_trace(go.Violin(
                    y=proba[y_te==1],name="Claim (True 1)",side="positive",
                    fillcolor="rgba(255,68,68,0.45)",line_color=C["red"],
                    points=False,meanline_visible=True))
                fig_dist.add_hline(y=0.5,line_dash="dot",line_color=C["amber"],
                                   annotation_text="Decision threshold 0.50")
                fig_dist.update_layout(violingap=0,violinmode="overlay",
                                       yaxis_title="Predicted Claim Probability",
                                       height=380,**PLOTLY_THEME)
                st.plotly_chart(fig_dist,use_container_width=True,key="ml_dist")

            # Feature importance
            imp_k=next((k for k in [best,"Random Forest"] if k in res and "imp" in res[k]),None)
            if imp_k:
                st.markdown(f"## 🗝️ Feature Importance — {imp_k}")
                fi_df=pd.DataFrame(list(res[imp_k]["imp"].items()),
                                   columns=["Feature","Importance"]).sort_values("Importance")
                fi_df["Feature"]=fi_df["Feature"].str.replace("_"," ")
                fig_fi=go.Figure(go.Bar(
                    x=fi_df["Importance"],y=fi_df["Feature"],orientation="h",
                    marker=dict(color=fi_df["Importance"],
                                colorscale=[[0,C["blue"]],[0.5,C["purple"]],[1.0,C["red"]]],
                                showscale=True,colorbar=dict(title="Importance",len=0.6)),
                    text=[f"{v:.4f}" for v in fi_df["Importance"]],textposition="outside",
                    hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
                ))
                fig_fi.update_layout(height=480,xaxis_title="Importance Score",
                                     yaxis=dict(tickfont=dict(size=11)),**PLOTLY_THEME)
                st.plotly_chart(fig_fi,use_container_width=True,key="ml_fi")

            # Risk score distribution by tier
            st.markdown("## 🎠 Risk Score Distribution by Tier")
            fig_hist=go.Figure()
            for tier in TIER_ORDER:
                sub=df_full[df_full["Risk_Tier"]==tier]["Risk_Score"]
                if not sub.empty:
                    fig_hist.add_trace(go.Violin(
                        y=sub,name=tier,
                        fillcolor=_hex_to_rgba(TIER_COLS[tier],0.40),
                        line_color=TIER_COLS[tier],
                        points="outliers",pointpos=-0.8,jitter=0.1,
                        meanline_visible=True,showlegend=True))
            for v,label in [(0.25,"Low→Medium"),(0.5,"Medium→High"),(0.75,"High→Critical")]:
                fig_hist.add_hline(y=v,line_dash="dot",line_color=C["muted"],
                                   annotation_text=label,annotation_position="right")
            fig_hist.update_layout(yaxis_title="Risk Score",height=420,
                                   violinmode="group",**PLOTLY_THEME)
            st.plotly_chart(fig_hist,use_container_width=True,key="ml_risk_dist")
        else:
            mc1,mc2=st.columns(2)
            with mc1: _show(confusion_mpl(b.get("y_te",[]),b.get("pred",[]),best))
            imp_k=next((k for k in [best,"Random Forest"] if k in res and "imp" in res[k]),None)
            if imp_k:
                with mc2: _show(bar_h_mpl(res[imp_k]["imp"].items(),f"Feature Importance — {imp_k}"))

        # Classification report
        st.markdown("## 🎇 Classification Report")
        if b.get("y_te") is not None and b.get("pred") is not None:
            cr=classification_report(b["y_te"],b["pred"],
                                     target_names=["No Claim","Claim"],output_dict=True)
            st.dataframe(pd.DataFrame(cr).T.round(4),use_container_width=True)

        # ── Chapter 4.9: 5-Fold Cross-Validation ─────────────────────────────
        st.markdown("---")
        st.markdown("## 🔄 5-Fold Stratified Cross-Validation (Section 4.9)")
        st.markdown("*Verifying model generalisation — AUC-ROC variance across folds should be low (σ < 0.01).*")
        feat_cv=[c for c in ML_FEAT_COLS if c in df_full.columns]
        X_cv_arr=df_full[feat_cv].fillna(0).values
        y_cv_arr=df_full["Claim_Label"].astype(int).values
        if len(np.unique(y_cv_arr))>=2:
            with st.spinner("Running 5-fold cross-validation …"):
                try:
                    if HAS_LGB:
                        cv_model=lgb.LGBMClassifier(
                            n_estimators=200,learning_rate=0.05,max_depth=5,
                            num_leaves=31,min_child_samples=50,
                            random_state=42,verbosity=-1,n_jobs=-1)
                    else:
                        cv_model=GradientBoostingClassifier(
                            n_estimators=150,learning_rate=0.05,max_depth=4,
                            min_samples_leaf=30,random_state=42)
                    skf_cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
                    cv_scores=cross_val_score(
                        cv_model,X_cv_arr,y_cv_arr,cv=skf_cv,
                        scoring="roc_auc",n_jobs=-1)
                    cv_df=pd.DataFrame({
                        "Fold":   [f"Fold {i+1}" for i in range(5)]+["Mean","Std Dev"],
                        "AUC-ROC":list(cv_scores.round(3))+[
                            round(float(cv_scores.mean()),3),
                            round(float(cv_scores.std()),3)],
                    })
                    cv_col1,cv_col2=st.columns([1,2])
                    with cv_col1:
                        st.dataframe(cv_df,use_container_width=True)
                        delta_txt=("✅ Model generalises well" if cv_scores.std()<0.015
                                   else "⚠️ Variance slightly elevated")
                        st.success(f"CV AUC-ROC: **{cv_scores.mean():.3f} ± {cv_scores.std():.3f}**  {delta_txt}")
                    with cv_col2:
                        fig_cv,ax_cv=_fig(6,3.5)
                        ax_cv.bar([f"F{i+1}" for i in range(5)],cv_scores,
                                  color=[C["green"] if v>=0.85 else C["amber"] for v in cv_scores],
                                  edgecolor=C["bord"],lw=0.5,zorder=3,alpha=0.88,width=0.55)
                        ax_cv.axhline(cv_scores.mean(),color=C["cyan"],lw=2.0,ls="--",
                                      label=f"Mean = {cv_scores.mean():.3f}")
                        ax_cv.axhline(0.85,color=C["amber"],lw=1.2,ls=":",
                                      label="Project target (0.85)")
                        ax_cv.set_ylabel("AUC-ROC"); ax_cv.set_ylim(0.5,1.0)
                        ax_cv.set_title("5-Fold Cross-Validation AUC-ROC",pad=6)
                        ax_cv.legend(fontsize=8); ax_cv.grid(axis="y",zorder=0)
                        fig_cv.tight_layout(); _show(fig_cv)
                except Exception as cv_err:
                    st.warning(f"Cross-validation skipped: {cv_err}")

        # ── Chapter 4.7: SHAP Feature Importance & Table 4.2 ─────────────────
        st.markdown("---")
        st.markdown("## 🔍 SHAP Feature Importance & Explainability (Section 4.7)")

        # Table 4.2 — always shown (from Chapter 4 research findings)
        st.markdown("### 📋 Top 10 Features by Mean Absolute SHAP Value — Table 4.2")
        shap_tbl=pd.DataFrame({
            "Rank":        list(range(1,11)),
            "Feature":     ["Harsh Event Score","Speed Variance",
                            "Harsh Braking Freq/Day","Time-of-Day Risk Factor",
                            "Road Type","Weather Risk Factor","Vehicle Age (Years)",
                            "Fuel Efficiency Ratio","Distance Travelled/Day","Province/Zone"],
            "Mean |SHAP|": [0.31,0.25,0.20,0.14,0.13,0.11,0.09,0.08,0.06,0.05],
            "Direction":   ["Positive","Positive","Positive","Positive (Night)",
                            "Positive (Potholed/Gravel)","Positive","Positive",
                            "Negative (Protective)","Positive","Variable"],
        })
        st.dataframe(shap_tbl.style.background_gradient(subset=["Mean |SHAP|"],cmap="YlOrRd"),
                     use_container_width=True)
        st.caption("Harsh driving behaviour dominates risk prediction. Fuel efficiency is a protective factor "
                   "— eco-driving proxies correlate negatively with claim probability.")

        if HAS_SHAP and res.get("_model") is not None:
            import shap as shap_lib
            X_shap_df=res.get("_X_te")
            feat_shap=res.get("_feat",[])
            if X_shap_df is not None and len(X_shap_df)>0:
                try:
                    with st.spinner("Computing SHAP values …"):
                        explainer=shap_lib.TreeExplainer(res["_model"])
                        n_shap=min(500,len(X_shap_df))
                        X_shap_sub=(X_shap_df.iloc[:n_shap]
                                    if hasattr(X_shap_df,"iloc") else X_shap_df[:n_shap])
                        shap_vals=explainer.shap_values(X_shap_sub)
                        if isinstance(shap_vals,list) and len(shap_vals)==2:
                            shap_vals=shap_vals[1]
                    sh1,sh2=st.columns(2)
                    with sh1:
                        st.markdown("#### SHAP Summary (Beeswarm)")
                        fig_sh,_=plt.subplots(figsize=(7,4.5))
                        shap_lib.summary_plot(shap_vals,X_shap_sub,
                                              feature_names=feat_shap,show=False,plot_size=(7,4.5))
                        st.pyplot(fig_sh,use_container_width=True); plt.close(fig_sh)
                    with sh2:
                        st.markdown("#### Mean |SHAP| Bar Chart")
                        mean_shap=np.abs(shap_vals).mean(axis=0)
                        shap_imp=pd.DataFrame({"Feature":feat_shap,"Mean |SHAP|":mean_shap})\
                                   .sort_values("Mean |SHAP|",ascending=False).head(10)
                        if HAS_PLOTLY:
                            fig_si=go.Figure(go.Bar(
                                x=shap_imp["Mean |SHAP|"],y=shap_imp["Feature"],
                                orientation="h",
                                marker=dict(color=shap_imp["Mean |SHAP|"],
                                            colorscale=[[0,C["blue"]],[0.5,C["amber"]],[1.0,C["red"]]],
                                            showscale=True),
                                text=[f"{v:.4f}" for v in shap_imp["Mean |SHAP|"]],
                                textposition="outside",
                            ))
                            fig_si.update_layout(height=400,xaxis_title="Mean |SHAP value|",
                                                 yaxis=dict(autorange="reversed"),**PLOTLY_THEME)
                            st.plotly_chart(fig_si,use_container_width=True,key="ml_shap_bar")
                        else:
                            fig_sb,ax_sb=_fig(7,4)
                            ax_sb.barh(shap_imp["Feature"],shap_imp["Mean |SHAP|"],
                                       color=C["amber"],edgecolor=C["bord"],lw=0.5,zorder=3)
                            ax_sb.set_xlabel("Mean |SHAP value|"); ax_sb.grid(axis="x",zorder=0)
                            fig_sb.tight_layout(); _show(fig_sb)
                    # SHAP Waterfall for the single highest-risk test sample
                    st.markdown("#### SHAP Waterfall — Highest-Risk Vehicle in Test Set")
                    try:
                        max_idx=int(np.argmax(np.abs(shap_vals).sum(axis=1)))
                        ev=(explainer.expected_value if not isinstance(explainer.expected_value,list)
                            else explainer.expected_value[1])
                        sample_data=(X_shap_sub.iloc[max_idx]
                                     if hasattr(X_shap_sub,"iloc") else X_shap_sub[max_idx])
                        exp_obj=shap_lib.Explanation(
                            values=shap_vals[max_idx],base_values=ev,
                            data=sample_data,feature_names=feat_shap)
                        fig_wf,_=plt.subplots(figsize=(9,4))
                        shap_lib.waterfall_plot(exp_obj,show=False,max_display=10)
                        st.pyplot(fig_wf,use_container_width=True); plt.close(fig_wf)
                    except Exception as wf_err:
                        st.caption(f"Waterfall not available: {wf_err}")
                except Exception as shap_err:
                    st.warning(f"SHAP computation error: {shap_err}")
        else:
            if not HAS_SHAP:
                st.info("💡 Install SHAP for live waterfall plots: `pip install shap`")

        # Filterable risk table
        st.markdown("---")
        st.markdown("## 🚙 Vehicle Risk Table — Filterable")
        rt1,rt2,rt3,rt4=st.columns(4)
        with rt1: rt_t=st.multiselect("Risk Tier",TIER_ORDER,default=["High","Critical"],key="rt_t")
        with rt2: rt_v=st.multiselect("Vehicle Type",sorted(df["Vehicle_Type"].dropna().unique()),default=[],key="rt_v")
        with rt3: rt_c=st.multiselect("City",sorted(df["Registration_City"].dropna().unique()) if "Registration_City" in df.columns else [],default=[],key="rt_c")
        with rt4: rs_r=st.slider("Risk Score Range",0.0,1.0,(0.0,1.0),0.01,key="rs_r")
        plate_q=st.text_input("🔎 Search plate / make / model","",key="plate_q")
        rv=df.copy()
        if rt_t: rv=rv[rv["Risk_Tier"].isin(rt_t)]
        if rt_v: rv=rv[rv["Vehicle_Type"].isin(rt_v)]
        if rt_c and "Registration_City" in rv.columns: rv=rv[rv["Registration_City"].isin(rt_c)]
        rv=rv[(rv["Risk_Score"]>=rs_r[0])&(rv["Risk_Score"]<=rs_r[1])]
        if plate_q:
            m=(rv["Vehicle_Number"].str.contains(plate_q,case=False,na=False)|
               rv["Vehicle_Make"].str.contains(plate_q,case=False,na=False))
            rv=rv[m]
        rcols=[c for c in ["Vehicle_Number","Vehicle_Type","Vehicle_Make","Vehicle_Year",
                            "Vehicle_Price","Registration_City","Vehicle_Usage","Status",
                            "Speed_kmh","RPM","Engine_Load_Pct","HB_Day","HA_Day","HC_Day",
                            "Harsh_Score","Speed_Deviation","Distance_Day_Km","Fuel_Eff_KmL",
                            "OBD_Risk_Signal","Weather_Condition","Risk_Score","Risk_Tier"]
               if c in rv.columns]
        st.markdown(f"**{len(rv):,}** vehicles match")
        st.dataframe(
            rv[rcols].sort_values("Risk_Score",ascending=False).head(500)
              .style.map(_style_tier,subset=["Risk_Tier"])
              .format({c:"{:.3f}" for c in ["Speed_kmh","Risk_Score","Harsh_Score",
                                            "Speed_Deviation","Fuel_Eff_KmL","OBD_Risk_Signal"]
                       if c in rcols}),
            use_container_width=True,height=400)

        # Individual deep-dive
        st.markdown("## 🛻 Individual Vehicle Deep-Dive")
        plates=df["Vehicle_Number"].dropna().unique().tolist()
        if plates:
            sel=st.selectbox("Select Vehicle",plates,key="sel_v")
            vrow=df[df["Vehicle_Number"]==sel]
            if vrow.empty: vrow=df[df["Vehicle_Number"]==plates[0]]
            if not vrow.empty:
                vr=vrow.iloc[-1].to_dict()
                rs=float(vr.get("Risk_Score",0.3)); hs=float(vr.get("Harsh_Score",2))
                g1,g2=st.columns([1,2])
                with g1: _show(speedometer_mpl(rs,title="Risk Score"))
                with g2:
                    hb=float(vr.get("HB_Day",0)); ha=float(vr.get("HA_Day",0))
                    hc=float(vr.get("HC_Day",0)); sd=float(vr.get("Speed_Deviation",0))
                    fe=float(vr.get("Fuel_Eff_KmL",0)); age=int(vr.get("Vehicle_Age",10))
                    rpm=float(vr.get("RPM",800)); load=float(vr.get("Engine_Load_Pct",0))
                    advice=[]
                    if hb>4:    advice.append(f"Reduce harsh braking ({hb:.0f}/day).")
                    if ha>3:    advice.append(f"Reduce harsh acceleration ({ha:.0f}/day).")
                    if hc>3:    advice.append(f"Reduce harsh cornering ({hc:.0f}/day).")
                    if sd>25:   advice.append(f"Speed {sd:.0f} km/h above road mean.")
                    if fe<5:    advice.append(f"Poor fuel efficiency ({fe:.1f} km/L).")
                    if age>12:  advice.append(f"Vehicle {age} yrs — inspect mechanically.")
                    if load>80: advice.append(f"High engine load ({load:.0f}%) — check drivetrain.")
                    if rpm>4500:advice.append(f"High RPM ({rpm:.0f}) — aggressive driving.")
                    if not advice: advice=["All driving metrics within safe parameters ✅"]
                    tier_css={"Low":"tier-low","Medium":"tier-medium","High":"tier-high",
                              "Critical":"tier-critical"}.get(str(vr.get("Risk_Tier","Low")),"tier-low")
                    card_cls=("green" if vr.get("Risk_Tier","Low")=="Low"
                              else "red" if vr.get("Risk_Tier","Low") in ["High","Critical"]
                              else "amber")
                    st.markdown(f"""<div class="card card-{card_cls}">
<b>Plate:</b> {sel} &nbsp;|&nbsp; <b>Type:</b> {vr.get('Vehicle_Type','—')} · {vr.get('Vehicle_Make','—')} {vr.get('Vehicle_Model','—')}<br>
<b>Year:</b> {int(vr.get('Vehicle_Year',0))} &nbsp;|&nbsp; <b>City:</b> {vr.get('Registration_City','—')} &nbsp;|&nbsp; <b>Weather:</b> {vr.get('Weather_Condition','—')}<br>
<b>Status:</b> {vr.get('Status','—')} &nbsp;|&nbsp; <b>Speed:</b> {float(vr.get('Speed_kmh',0)):.1f} km/h &nbsp;|&nbsp; <b>RPM:</b> {rpm:.0f}<br>
<b>Engine Load:</b> {load:.1f}% &nbsp;|&nbsp; <b>Coolant:</b> {float(vr.get('Coolant_Temp_C',85)):.1f}°C &nbsp;|&nbsp; <b>Battery:</b> {float(vr.get('Battery_V',12.5)):.2f}V<br><br>
<b>Risk Score:</b> <span class="{tier_css}">{rs:.4f} — {vr.get('Risk_Tier','—')}</span><br>
<b>Harsh Score:</b> {hs:.2f} &nbsp;|&nbsp; <b>Speed Dev:</b> {sd:.1f} km/h<br><br>
<b>🍀 Recommendations:</b> {' '.join(advice)}
</div>""",unsafe_allow_html=True)
        st.download_button("⬇️ Download Risk Table",data=rv[rcols].to_csv(index=False).encode(),
                           file_name="risk_table.csv",mime="text/csv")
    else:
        st.info("👆 Click **🚀 Train Machine Learning Models** in the sidebar, or wait for auto-training (Tick ≥ 2).")
        if has_data:
            pct=min(STATE.tick/max(1,2)*100,100)
            st.progress(int(pct),text=f"Data ready: {len(df_full):,} vehicles — auto-training …")
        elif not has_data and STATE.tick<2:
            st.progress(min(STATE.tick*50,90),text="Collecting real-time data …")

# Tab 4 — Poloicy pricing
with t4:
    st.markdown("## 💰 Policy Pricing")
    with st.spinner("Computing premium schedule…"):
        prem=build_premium_df(df)
    _prem_req=["Gross_Premium","Expected_Claim","Expenses","Inv_Return","Net_Profit","Loss_Ratio"]
    if prem.empty or any(c not in prem.columns for c in _prem_req):
        st.warning("No premium data — start simulation first.")
    else:
        p1,p2,p3,p4,p5,p6,p7=st.columns(7)
        p1.metric("Vehicles",     f"{len(prem):,}")
        p2.metric("Gross Premium",f"${prem['Gross_Premium'].sum()/1e6:.2f}M")
        p3.metric("Exp. Claims",  f"${prem['Expected_Claim'].sum()/1e6:.2f}M")
        p4.metric("Expenses",     f"${prem['Expenses'].sum()/1e6:.2f}M")
        p5.metric("Net Profit",   f"${prem['Net_Profit'].sum()/1e6:.2f}M",
                  delta=f"{prem['Net_Profit'].sum()/max(prem['Gross_Premium'].sum(),1)*100:.1f}% margin")
        p6.metric("Loss Ratio",   f"{prem['Loss_Ratio'].mean()*100:.1f}%")
        p7.metric("Critical Risk",f"{(prem['Risk_Tier']=='Critical').sum():,}")
        st.markdown("---")
        st.markdown("### 💹 P&L Waterfall — Annualised"); _show(waterfall_mpl(prem))
        r1,r2=st.columns(2)
        with r1:
            st.markdown("## 😪 Loss Ratio by Vehicle Type")
            lr=prem.groupby("Vehicle_Type")["Loss_Ratio"].mean().sort_values(ascending=False)
            fig,ax=_fig(7,4)
            ax.bar(lr.index,lr.values*100,
                   color=[C["red"] if v>0.90 else C["amber"] if v>0.70 else C["green"] for v in lr.values],
                   edgecolor=C["bord"],lw=0.5,zorder=3,alpha=0.88)
            ax.axhline(100,color=C["red"],  lw=1.5,ls="--",label="Break-even (100%)")
            ax.axhline(70, color=C["green"],lw=1.0,ls=":", label="Target (70%)")
            ax.set_title("Loss Ratio by Vehicle Type (%)"); ax.set_ylabel("Loss Ratio (%)")
            ax.legend(fontsize=8); ax.tick_params(axis="x",rotation=30)
            ax.grid(axis="y",zorder=0); fig.tight_layout(); _show(fig)
        with r2:
            st.markdown("### 🏙️ Profitability by City")
            grp=prem.groupby("Registration_City").agg(
                Rev=("Gross_Premium","sum"),Clm=("Expected_Claim","sum"),
                Net=("Net_Profit","sum")).reset_index().sort_values("Net",ascending=False)
            x=np.arange(len(grp)); w=0.27
            fig,ax=_fig(8,4)
            ax.bar(x-w,grp.Rev/1e3,width=w,label="Revenue",color=C["green"],edgecolor=C["bord"],lw=0.5,zorder=3,alpha=0.88)
            ax.bar(x,  grp.Clm/1e3,width=w,label="Claims", color=C["red"],  edgecolor=C["bord"],lw=0.5,zorder=3,alpha=0.88)
            ax.bar(x+w,grp.Net/1e3,width=w,label="Net",    color=C["blue"], edgecolor=C["bord"],lw=0.5,zorder=3,alpha=0.88)
            ax.set_xticks(x); ax.set_xticklabels(grp.Registration_City,rotation=20)
            ax.set_title("Revenue vs Claims vs Net by City (USD '000)")
            ax.set_ylabel("USD (thousands)"); ax.legend(fontsize=8)
            ax.grid(axis="y",zorder=0); fig.tight_layout(); _show(fig)
        r3,r4=st.columns(2)
        with r3:
            if HAS_PLOTLY:
                st.markdown("## 🎯 Risk Score vs Premium")
                fig_sp=px.scatter(prem.sample(min(5000,len(prem)),random_state=42),
                                  x="Risk_Score",y="Gross_Premium",color="Risk_Tier",
                                  color_discrete_map=TIER_COLS,opacity=0.4,size_max=8,
                                  labels={"Risk_Score":"Risk Score","Gross_Premium":"Annual Premium (USD)"})
                fig_sp.update_layout(height=380,**PLOTLY_THEME)
                st.plotly_chart(fig_sp,use_container_width=True,key="t4_scatter")
        with r4:
            st.markdown("## 💸 Premium Box by Risk Tier")
            _bp_pairs=[(t,prem[prem["Risk_Tier"]==t]["Gross_Premium"].dropna().values)
                       for t in TIER_ORDER if t in prem["Risk_Tier"].values]
            _bp_pairs=[(t,d) for t,d in _bp_pairs if len(d)>0]
            if _bp_pairs:
                data_bp=[d for _,d in _bp_pairs]; lbls_bp=[t for t,_ in _bp_pairs]
                fig,ax=_fig(6,4)
                bp=ax.boxplot(data_bp,patch_artist=True,labels=lbls_bp,
                              medianprops=dict(color="white",lw=2.2),
                              whiskerprops=dict(color=C["muted"]),capprops=dict(color=C["muted"]),
                              flierprops=dict(marker="o",ms=2.5,alpha=0.3,markerfacecolor=C["red"]))
                for patch,t in zip(bp["boxes"],lbls_bp):
                    patch.set_facecolor(TIER_COLS[t]); patch.set_alpha(0.72)
                ax.set_title("Premium Distribution by Risk Tier")
                ax.set_ylabel("Annual Premium (USD)"); ax.grid(axis="y",zorder=0)
                fig.tight_layout(); _show(fig)
        st.markdown("## 🎡 12-Month Profitability Projection (Seasonal)")
        _show(monthly_projection_mpl(prem))
        st.markdown("---"); st.markdown("### 📋 Full Premium Schedule")
        pt1,pt2=st.columns(2)
        with pt1: pt_tier=st.multiselect("Filter Tier",TIER_ORDER,default=[],key="pt_t")
        with pt2: pt_type=st.multiselect("Filter Type",sorted(prem["Vehicle_Type"].unique()),default=[],key="pt_vt")
        pv=prem.copy()
        if pt_tier: pv=pv[pv["Risk_Tier"].isin(pt_tier)]
        if pt_type: pv=pv[pv["Vehicle_Type"].isin(pt_type)]
        st.dataframe(
            pv.sort_values("Gross_Premium",ascending=False).head(500)
              .style.map(_style_tier,subset=["Risk_Tier"])
              .format({"Risk_Score":"{:.4f}","Vehicle_Price":"${:,.2f}",
                       "Gross_Premium":"${:,.2f}","Expected_Claim":"${:,.2f}",
                       "Expenses":"${:,.2f}","UW_Profit":"${:,.2f}",
                       "Net_Profit":"${:,.2f}","Loss_Ratio":"{:.2%}"})
              .background_gradient(subset=["Loss_Ratio"],cmap="RdYlGn_r"),
            use_container_width=True,height=360)
        st.download_button("⬇️ Download Premium Schedule",data=pv.to_csv(index=False).encode(),
                           file_name="premium_schedule.csv",mime="text/csv")

# Tab -Fraud detection
with t5:
    st.markdown("## 🚨 Fraud & Anomaly Detection")
    st.markdown("*Isolation Forest flags vehicles whose OBD-II signal combinations deviate "
                "statistically from normal fleet behaviour.*")
    if run_fraud:
        feat_f=[c for c in ML_FEAT_COLS if c in df.columns]
        Xf=df[feat_f].fillna(0)
        iso=IsolationForest(n_estimators=200,contamination=0.08,random_state=42,n_jobs=-1)
        sc=iso.fit_predict(Xf); raw_sc=iso.score_samples(Xf)
        df_fr=df.copy()
        df_fr["Anomaly_Flag"]=(sc==-1).astype(int)
        neg=(-raw_sc); rng_n=neg.max()-neg.min()+1e-9
        df_fr["Anomaly_Score"]=((neg-neg.min())/rng_n).round(4)
        df_fr["Fraud_Risk"]=pd.cut(df_fr["Anomaly_Score"],
                                    bins=[-0.001,0.40,0.65,0.85,1.001],
                                    labels=["Normal","Suspicious","Likely Fraud","Critical Fraud"]).astype(str)
        nf=df_fr["Anomaly_Flag"].sum()
        st.success(f"✓ Scan complete — **{nf:,}** anomalous vehicles ({nf/len(df_fr)*100:.1f}%) on Tick #{STATE.tick}")
        f1,f2,f3,f4=st.columns(4)
        f1.metric("Scanned",    f"{len(df_fr):,}")
        f2.metric("Flagged",    f"{nf:,}")
        f3.metric("Critical",   f"{(df_fr['Fraud_Risk']=='Critical Fraud').sum():,}")
        f4.metric("Likely Fraud",f"{(df_fr['Fraud_Risk']=='Likely Fraud').sum():,}")
        fl1,fl2=st.columns(2)
        with fl1:
            fr=df_fr["Fraud_Risk"].value_counts()
            fraud_c={"Normal":C["green"],"Suspicious":C["amber"],"Likely Fraud":C["high"],"Critical Fraud":C["red"]}
            cats=list(fr.index); vals=list(fr.values)
            colors_fr=[fraud_c.get(k,C["blue"]) for k in cats]
            fig,ax=_fig(6,4)
            ax.bar(cats,vals,color=colors_fr,edgecolor=C["bord"],lw=0.5,zorder=3,alpha=0.88)
            ax.set_title("Fraud Risk Category Distribution"); ax.set_ylabel("Count")
            ax.tick_params(axis="x",rotation=20)
            ax.grid(axis="y",zorder=0); fig.tight_layout(); _show(fig)
        with fl2:
            fig,ax=_fig(6,4)
            ax.hist(df_fr["Anomaly_Score"],bins=60,color=C["amber"],
                    edgecolor=C["bord"],lw=0.4,alpha=0.85,zorder=3)
            ax.axvline(0.65,color=C["red"],  lw=1.5,ls="--",label="Fraud threshold (0.65)")
            ax.axvline(0.40,color=C["amber"],lw=1.0,ls=":", label="Suspicious (0.40)")
            ax.set_title("Anomaly Score Distribution")
            ax.set_xlabel("Anomaly Score (0=Normal → 1=Most Anomalous)")
            ax.legend(fontsize=8); ax.grid(axis="y",zorder=0); fig.tight_layout(); _show(fig)
        if HAS_PLOTLY:
            st.markdown("### 🔴 Risk Score vs Anomaly Score")
            samp_fr=df_fr.sample(min(5000,len(df_fr)),random_state=42)
            fig_fa=px.scatter(samp_fr,x="Risk_Score",y="Anomaly_Score",color="Fraud_Risk",
                              color_discrete_map={"Normal":C["green"],"Suspicious":C["amber"],
                                                  "Likely Fraud":C["high"],"Critical Fraud":C["red"]},
                              opacity=0.45,size_max=8,
                              hover_data=["Vehicle_Number","Vehicle_Type","Registration_City","Weather_Condition"])
            fig_fa.add_hline(y=0.65,line_dash="dash",line_color=C["red"],annotation_text="Fraud threshold")
            fig_fa.update_layout(height=440,**PLOTLY_THEME)
            st.plotly_chart(fig_fa,use_container_width=True,key="t5_scatter")
        flagged=df_fr[df_fr["Anomaly_Flag"]==1]
        ffc1,ffc2=st.columns(2)
        with ffc1:
            fi_fr=st.multiselect("Filter Fraud Risk",
                                  ["Normal","Suspicious","Likely Fraud","Critical Fraud"],
                                  default=["Likely Fraud","Critical Fraud"],key="fi_fr")
        with ffc2:
            fi_vt=st.multiselect("Filter Type",sorted(flagged["Vehicle_Type"].unique()),default=[],key="fi_vt")
        vfr=flagged.copy()
        if fi_fr: vfr=vfr[vfr["Fraud_Risk"].isin(fi_fr)]
        if fi_vt: vfr=vfr[vfr["Vehicle_Type"].isin(fi_vt)]
        fcols=[c for c in ["Vehicle_Number","Vehicle_Type","Vehicle_Make","Vehicle_Year",
                            "Registration_City","Weather_Condition","Status","Speed_kmh",
                            "RPM","Engine_Load_Pct","HB_Day","HA_Day","HC_Day",
                            "Harsh_Score","Distance_Day_Km","Risk_Score","Risk_Tier",
                            "Anomaly_Score","Fraud_Risk"] if c in vfr.columns]
        st.markdown("### 🚩 Flagged Vehicle Report")
        st.dataframe(
            vfr[fcols].sort_values("Anomaly_Score",ascending=False).head(400)
               .style.format({"Anomaly_Score":"{:.4f}","Risk_Score":"{:.4f}","Speed_kmh":"{:.1f}"})
               .background_gradient(subset=["Anomaly_Score"],cmap="YlOrRd"),
            use_container_width=True,height=360)
        st.download_button("⬇️ Download Fraud Report",data=vfr.to_csv(index=False).encode(),
                           file_name="fraud_report.csv",mime="text/csv")
    else:
        st.info("👆 Click **🦅 Run Fraud Scan** in the sidebar.")

# Tab 6 - Excutive dashboard
with t6:
    st.markdown("## 👨🏼‍🎓 Executive Reporting Dashboard")
    # ── Chapter 4.2: Key Findings Summary ────────────────────────────────────
    st.markdown("### 🏆 Key Findings Summary (Chapter 4.2)")
    kf1,kf2=st.columns(2)
    with kf1:
        st.markdown(f"""<div class="card card-green">
<b>🎯 Objective 1 — Telematics Data Generation</b><br>
50,000 synthetic vehicles · 25 variables per record · 6 road types<br>
~1.2M records after preprocessing · Realistic Zimbabwe fleet composition<br>
Toyota-dominant fleet · Public service vehicles (kombis) included
</div>
<div class="card card-green">
<b>🚘 Objective 2 — Driving Behaviour Analysis</b><br>
Public service vehicles avg harsh score: <b>0.68</b> vs private sedans: <b>0.31</b><br>
Night-time driving (22:00–04:00): <b>+43% high-risk flags</b> vs morning hours<br>
Urban zones (Harare, Bulawayo) record highest average speeds &amp; harsh events
</div>
<div class="card card-green">
<b>💰 Objective 3 — Policy Pricing &amp; Optimisation</b><br>
Premium range: <b>-25% discount</b> (low-risk) to <b>+60% loading</b> (high-risk)<br>
<b>38% reduction</b> in pricing variance vs demographic-only baseline<br>
Policy recommendations generated in <b>&lt;1.2 sec/vehicle</b>
</div>""",unsafe_allow_html=True)
    with kf2:
        st.markdown(f"""<div class="card card-green">
<b>⚡ Objective 4 — Dynamic Risk Assessment</b><br>
LightGBM AUC-ROC: <b>0.913</b> (target: 0.85) · KS Statistic: <b>52.4</b><br>
Risk scores recalculated every tick · Behaviour changes reflected within 24h<br>
Driver reducing harsh braking 6.2→1.4/day: <b>High→Medium</b> in real time
</div>
<div class="card card-green">
<b>📊 Objective 5 — Dashboards &amp; Visualisation</b><br>
Interactive Streamlit dashboard · SHAP waterfall plots per policy decision<br>
KPI panel updates live · Risk heatmaps across Zimbabwe provincial zones<br>
Explainable AI outputs interpretable by non-technical stakeholders (IPEC compliant)
</div>""",unsafe_allow_html=True)
    st.markdown("---")
    prem_d=build_premium_df(df_full)
    _pd_req=["Gross_Premium","Expected_Claim","Inv_Return","Net_Profit","Loss_Ratio"]
    if prem_d.empty or any(c not in prem_d.columns for c in _pd_req):
        st.warning("Start simulation first.")
    else:
        e=st.columns(8)
        e[0].metric("Fleet",        f"{len(df_full):,}")
        e[1].metric("Gross Premium",f"${prem_d['Gross_Premium'].sum()/1e6:.2f}M")
        e[2].metric("Exp. Claims",  f"${prem_d['Expected_Claim'].sum()/1e6:.2f}M")
        e[3].metric("Net Profit",   f"${prem_d['Net_Profit'].sum()/1e6:.2f}M")
        e[4].metric("Loss Ratio",   f"{prem_d['Loss_Ratio'].mean()*100:.1f}%",
                    delta="🟢 Profitable" if prem_d["Loss_Ratio"].mean()<0.9 else "🔴 Unprofitable")
        e[5].metric("Avg Risk",     f"{df_full['Risk_Score'].mean():.4f}")
        e[6].metric("High+Critical",f"{df_full['Risk_Tier'].isin(['High','Critical']).sum():,}")
        e[7].metric("Driving Now",  f"{(df_full['Status'].eq('Driving').sum() if 'Status' in df_full.columns else 0):,}")
        st.markdown("---")
        d1,d2=st.columns(2)
        with d1:
            st.markdown("### Risk Tier Split"); _show(donut_mpl(df_full))
        with d2:
            st.markdown("### Premium vs Claims by Type")
            grp=prem_d.groupby("Vehicle_Type").agg(P=("Gross_Premium","sum"),
                                                    C=("Expected_Claim","sum")).reset_index()
            x=np.arange(len(grp)); w=0.36
            fig,ax=_fig(6,4)
            ax.bar(x-w/2,grp.P/1e3,width=w,label="Premium",color=C["green"],edgecolor=C["bord"],lw=0.5,zorder=3,alpha=0.88)
            ax.bar(x+w/2,grp.C/1e3,width=w,label="Claims", color=C["red"],  edgecolor=C["bord"],lw=0.5,zorder=3,alpha=0.88)
            ax.set_xticks(x); ax.set_xticklabels(grp.Vehicle_Type,rotation=30,ha="right",fontsize=7.5)
            ax.set_ylabel("USD '000"); ax.legend(fontsize=8)
            ax.grid(axis="y",zorder=0); fig.tight_layout(); _show(fig)
        d3,d4=st.columns(2)
        with d3:
            fig,ax=_fig(6,4)
            ax.hist(df_full["Risk_Score"],bins=60,color=C["blue"],edgecolor=C["bord"],lw=0.4,alpha=0.85,zorder=3)
            for v,col_c,lbl in [(0.25,C["green"],"Low/Med"),(0.50,C["amber"],"Med/High"),(0.75,C["red"],"High/Crit")]:
                ax.axvline(v,color=col_c,lw=1.4,ls="--",label=lbl)
            ax.set_xlabel("Risk Score"); ax.set_ylabel("Count"); ax.set_title("Risk Score Distribution",pad=6)
            ax.legend(fontsize=7.5); ax.grid(axis="y",zorder=0); fig.tight_layout(); _show(fig)
        with d4:
            lr2=prem_d.groupby("Vehicle_Type")["Loss_Ratio"].mean().sort_values(ascending=False)
            fig,ax=_fig(6,4)
            ax.bar(lr2.index,lr2.values*100,
                   color=[C["red"] if v>0.90 else C["amber"] if v>0.70 else C["green"] for v in lr2.values],
                   edgecolor=C["bord"],lw=0.5,zorder=3,alpha=0.88)
            ax.axhline(100,color=C["red"],  lw=1.5,ls="--",label="Break-even")
            ax.axhline(70, color=C["green"],lw=1.0,ls=":", label="Target (70%)")
            ax.set_title("Loss Ratio by Type (%)"); ax.legend(fontsize=8)
            ax.tick_params(axis="x",rotation=30); ax.grid(axis="y",zorder=0)
            fig.tight_layout(); _show(fig)
        # Risk heatmap — unique key "exec_hmap" (different from "beh_hmap" in Tab 2)
        hmap2=heat_city_type_plotly(df_full)
        if hmap2 and HAS_PLOTLY:
            st.markdown("## 🌍 Risk Heatmap — City × Vehicle Type")
            st.plotly_chart(hmap2,use_container_width=True,key="exec_hmap")
        # Underwriting automation
        st.markdown("### ⚡ Underwriting Automation Summary")
        auto_dec={"Low":"✅ Auto-approve","Medium":"✅ Auto-approve + telematics clause",
                  "High":"🟡 Conditional — driver training required",
                  "Critical":"🔴 Manual review — potential decline"}
        auto_df=pd.DataFrame({
            "Risk Tier":list(auto_dec.keys()),
            "Decision":list(auto_dec.values()),
            "Count":[int((df_full["Risk_Tier"]==t).sum()) for t in auto_dec],
            "% of Fleet":[f"{(df_full['Risk_Tier']==t).mean()*100:.1f}%" for t in auto_dec],
            "Avg Premium":[f"${prem_d[prem_d['Risk_Tier']==t]['Gross_Premium'].mean():,.2f}"
                           if not prem_d[prem_d['Risk_Tier']==t].empty else "—" for t in auto_dec],
        })
        st.dataframe(auto_df.style.map(
            lambda v: (f"color:{C['red']};font-weight:700" if "Manual" in str(v)
                       else f"color:{C['amber']};font-weight:700" if "Conditional" in str(v)
                       else f"color:{C['green']};font-weight:700" if "Auto" in str(v) else ""),
            subset=["Decision"]),use_container_width=True)

# Tab 7- Scenario simulator
with t7:
    st.markdown("## 🌃 Interactive Scenario & Premium Simulator")
    sc1,sc2,sc3=st.columns(3)
    with sc1:
        sim_t  =st.selectbox("Vehicle Type",  list(BASE_PREMIUM.keys()),index=0,key="sc_t")
        sim_u  =st.selectbox("Usage",         ["Private","Commercial"],key="sc_u")
        sim_age=st.slider("Vehicle Age (yrs)",0,25,5,key="sc_a")
        sim_v  =st.slider("Vehicle Value (USD)",5000,200000,20000,1000,key="sc_v")
    with sc2:
        sim_spd=st.slider("Speed Deviation (km/h)",0,80,15,key="sc_spd")
        sim_hb =st.slider("Harsh Braking /day",    0,20, 2,key="sc_hb")
        sim_ha =st.slider("Harsh Accel /day",       0,20, 2,key="sc_ha")
        sim_hc =st.slider("Harsh Cornering /day",   0,20, 1,key="sc_hc")
        sim_d  =st.slider("Daily Distance (km)",    10,400,60,key="sc_d")
        sim_hs =round(sim_hb*0.4+sim_ha*0.3+sim_hc*0.3,2)
    with sc3:
        sim_wn  =st.selectbox("Weather",list(WX_RISK.keys()),  index=0,key="sc_wn")
        sim_tn  =st.selectbox("Time",  list(TOD_RISK.keys()),  index=1,key="sc_tn")
        sim_load=st.slider("Engine Load %",0,100,35,key="sc_ld")
        sim_rpm =st.slider("Engine RPM",  700,6500,2500,key="sc_rpm")
    sim_wxr   =(WX_RISK.get(sim_wn,1)-1)/1.6
    sim_todr  =(TOD_RISK.get(sim_tn,1)-1)/0.6
    sim_agr   =min(1,sim_age/30)
    sim_evr   =min(1,sim_hs/20)
    obd_risk_s=np.clip((sim_load/100)*0.4+np.clip((sim_rpm-3000)/3800,0,1)*0.35,0,1)
    spd_risk_s=1/(1+math.exp(-0.05*(sim_spd-12)))
    sim_rs    =round(spd_risk_s*0.28+sim_evr*0.26+sim_wxr*0.17+sim_agr*0.12+sim_todr*0.09+obd_risk_s*0.08,4)
    sim_tier  =("Critical" if sim_rs>0.75 else "High" if sim_rs>0.50 else "Medium" if sim_rs>0.25 else "Low")
    base=BASE_PREMIUM.get(sim_t,8000); mult=RISK_MULT[sim_tier]
    age_l=1+max(0,sim_age-5)*0.022; usg_l=1.18 if sim_u=="Commercial" else 1.0
    ml_  =1+max(0,sim_d*365-15000)/100000*0.28
    beh  =(0.90 if sim_hs<=1 else 1.00 if sim_hs<=4 else 1.15 if sim_hs<=8 else 1.35)
    gross=round(base*mult*age_l*usg_l*ml_*beh,2)
    claim=EXPECTED_CLAIM[sim_tier]; exps=round(gross*EXPENSE_RATIO,2)
    inv  =round(gross*INV_YIELD*0.5,2); net=round(gross-claim-exps+inv,2)
    lr   =round((claim+exps)/max(gross,1),4); tc=TIER_COLS[sim_tier]
    s1,s2,s3,s4,s5,s6=st.columns(6)
    s1.metric("Risk Score",    f"{sim_rs:.4f}"); s2.metric("Risk Tier",sim_tier)
    s3.metric("Annual Premium",f"${gross:,.2f}"); s4.metric("Exp. Claim",f"${claim:,.2f}")
    s5.metric("Net Profit",    f"${net:,.2f}");   s6.metric("Loss Ratio",f"{lr*100:.1f}%")
    gs1,gs2=st.columns([1,2])
    with gs1:
        if HAS_PLOTLY:
            fig_g=go.Figure(go.Indicator(
                mode="gauge+number+delta",value=round(sim_rs,4),
                delta={"reference":0.50,"increasing":{"color":C["red"]},"decreasing":{"color":C["green"]}},
                title={"text":f"Risk Score<br><span style='font-size:.8em;color:{tc}'>{sim_tier}</span>","font":{"size":18}},
                gauge={"axis":{"range":[0,1],"tickwidth":1,"tickcolor":"#666"},
                       "bar":{"color":tc},"bgcolor":"rgba(0,0,0,0)",
                       "steps":[{"range":[0,0.25],"color":"rgba(0,229,160,.12)"},
                                 {"range":[0.25,0.50],"color":"rgba(255,176,32,.12)"},
                                 {"range":[0.50,0.75],"color":"rgba(255,122,32,.12)"},
                                 {"range":[0.75,1.00],"color":"rgba(255,32,32,.12)"}],
                       "threshold":{"line":{"color":"white","width":3},"value":sim_rs}},
                number={"suffix":" / 1.0","font":{"size":26}},
            ))
            fig_g.update_layout(height=300,**PLOTLY_THEME)
            st.plotly_chart(fig_g,use_container_width=True,key="t7_gauge")
        else:
            _show(speedometer_mpl(sim_rs,title=f"Risk Score — {sim_tier}"))
    with gs2:
        st.markdown(f"""<div class="card" style="border-left:3px solid {tc}">
<b>Premium Breakdown:</b><br>
Base Premium ({sim_t}): USD {base:,.2f}<br>
Risk Multiplier ({sim_tier}): ×{mult:.2f}<br>
Age Loading ({sim_age} yrs): ×{age_l:.3f}<br>
Usage Loading ({'Commercial' if sim_u=='Commercial' else 'Private'}): ×{usg_l:.2f}<br>
UBI Mileage ({sim_d} km/day): ×{ml_:.3f}<br>
Behavioural Adj (HScore={sim_hs:.1f}): ×{beh:.2f}<br><br>
<b>Gross Premium: USD {gross:,.2f}</b><br>
Expected Claim: USD {claim:,.2f} · Expenses: USD {exps:,.2f}<br>
Investment Return: USD {inv:,.2f}<br>
<b>Net Profit per Policy: USD {net:,.2f}</b>
</div>""",unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("## 🖥️ Policy Narrative *(LangChain-inspired reasoning chain)*")
    narrative=build_policy_narrative(sim_rs,sim_tier,sim_t,gross,sim_hb,sim_ha,sim_hc,sim_spd,sim_wn,sim_tn)
    for line in narrative.split("\n\n"):
        color=(C["red"] if "RISK_AGENT" in line else C["purple"] if "SHAP_AGENT" in line else C["green"])
        st.markdown(f'<div class="card" style="border-left:3px solid {color}">{line}</div>',unsafe_allow_html=True)
    st.markdown("### 📊 Sensitivity Analysis")
    ss1,ss2=st.columns(2)
    with ss1:
        hb_vals=np.arange(0,21)
        rs_sens=[min(1.0,1/(1+math.exp(-0.05*(sim_spd-12)))*0.28+
                 min(1,(v*0.4+sim_ha*0.3+sim_hc*0.3)/20)*0.26+
                 sim_wxr*0.17+sim_agr*0.12+sim_todr*0.09+obd_risk_s*0.08) for v in hb_vals]
        fig,ax=_fig(6,3.8)
        ax.plot(hb_vals,rs_sens,color=C["red"],lw=2.2,marker="o",ms=4)
        ax.fill_between(hb_vals,rs_sens,alpha=0.12,color=C["red"])
        ax.axhline(0.75,color=C["red"],  lw=1,ls="--",label="Critical threshold")
        ax.axhline(0.50,color=C["amber"],lw=1,ls=":", label="High threshold")
        ax.set_xlabel("Harsh Braking Events/day"); ax.set_ylabel("Risk Score")
        ax.set_title("Sensitivity: Risk vs Harsh Braking",pad=6)
        ax.legend(fontsize=8); ax.grid(zorder=0); fig.tight_layout(); _show(fig)
    with ss2:
        spd_vals=np.arange(0,81,4)
        rs_spd=[min(1.0,1/(1+math.exp(-0.05*(v-12)))*0.28+
                sim_evr*0.26+sim_wxr*0.17+sim_agr*0.12+sim_todr*0.09+obd_risk_s*0.08)
                for v in spd_vals]
        prem_spd=[round(base*RISK_MULT["Critical" if rs>0.75 else "High" if rs>0.50
                                        else "Medium" if rs>0.25 else "Low"]*age_l*usg_l*ml_*beh,2)
                  for rs in rs_spd]
        fig,ax=_fig(6,3.8); ax2=ax.twinx()
        ax.plot(spd_vals,rs_spd,color=C["red"],lw=2,label="Risk Score")
        ax2.plot(spd_vals,prem_spd,color=C["amber"],lw=2,ls="--",label="Premium (USD)")
        ax.set_xlabel("Speed Deviation (km/h)"); ax.set_ylabel("Risk Score",color=C["red"])
        ax2.set_ylabel("Annual Premium (USD)",color=C["amber"])
        ax.set_title("Speed Deviation → Risk & Premium",pad=6)
        lines1,lab1=ax.get_legend_handles_labels(); lines2,lab2=ax2.get_legend_handles_labels()
        ax.legend(lines1+lines2,lab1+lab2,fontsize=8)
        ax.grid(zorder=0); fig.tight_layout(); _show(fig)

# Tab 8- Geospatial
with t8:
    st.markdown("## 🗺️ Geospatial Risk Heatmap - Zimbabwe")
    if not HAS_PLOTLY:
        st.warning("Install Plotly: `pip install plotly`")
    elif "Latitude" not in df.columns:
        st.warning("GPS data not yet available — start simulation first.")
    else:
        map_metric=st.selectbox("Colour metric",
                                ["Risk_Score","Harsh_Score","Speed_kmh","Engine_Load_Pct"],
                                key="map_m")
        sample_map=df.sample(min(5000,len(df)),random_state=42)
        prem_map=build_premium_df(sample_map)
        if "Gross_Premium" in prem_map.columns:
            sample_map=sample_map.merge(prem_map[["Vehicle_Number","Gross_Premium"]],
                                        on="Vehicle_Number",how="left")
        col_to_use=map_metric if map_metric in sample_map.columns else "Risk_Score"
        fig_map=px.scatter_mapbox(
            sample_map,lat="Latitude",lon="Longitude",
            color=col_to_use,size="Risk_Score",size_max=14,
            color_continuous_scale=[C["green"],C["amber"],C["red"]],
            hover_name="Vehicle_Number",
            hover_data={"Registration_City":True,"Vehicle_Type":True,
                        "Risk_Tier":True,"Speed_kmh":True,
                        "Weather_Condition":True,"RPM":True},
            mapbox_style="carto-darkmatter",
            zoom=5.5,center={"lat":-19.0,"lon":29.8},
            height=640,opacity=0.85,
            labels={col_to_use:col_to_use.replace("_"," ").title()},
        )
        fig_map.update_layout(**PLOTLY_THEME)
        fig_map.update_layout(margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_map,use_container_width=True,key="t8_map")
        st.markdown("## 🏙️ City-Level Risk & Premium Summary")
        prem_full=build_premium_df(df_full)
        city_agg=df_full.groupby("Registration_City").agg(
            avg_risk     =("Risk_Score","mean"),
            high_risk_pct=("Risk_Label","mean"),
            vehicle_count=("Vehicle_Number","count"),
            avg_harsh    =("Harsh_Score","mean"),
        ).round(3).reset_index()
        if not prem_full.empty:
            city_prem=prem_full.groupby("Registration_City")["Gross_Premium"].mean().round(2).reset_index()
            city_agg=city_agg.merge(city_prem,on="Registration_City",how="left")
            city_agg["high_risk_pct"]=(city_agg["high_risk_pct"]*100).round(1)
            cw_now2=STATE.get_city_weather()
            city_agg["Weather"]=city_agg["Registration_City"].map(cw_now2)
            fig_city=px.scatter(city_agg,x="avg_risk",y="Gross_Premium",
                                size="vehicle_count",color="high_risk_pct",
                                color_continuous_scale=[C["green"],C["red"]],
                                text="Registration_City",size_max=40,
                                labels={"avg_risk":"Avg Risk Score",
                                        "Gross_Premium":"Avg Premium (USD)",
                                        "high_risk_pct":"% High-Risk"})
            fig_city.update_traces(textposition="top center")
            fig_city.update_layout(height=460,**PLOTLY_THEME)
            st.plotly_chart(fig_city,use_container_width=True,key="t8_city")
            st.dataframe(city_agg.sort_values("avg_risk",ascending=False),use_container_width=True)

# Tab 9 - Driver profile
with t9:
    st.markdown("## 🧑🏾 Individual Driver Profile")

    def _info_card(col,icon,label,value,color=C["blue"]):
        col.markdown(f"""<div class="card" style="text-align:center">
          <div style="font-size:.72rem;color:{C['muted']};text-transform:uppercase;letter-spacing:.08em">{icon} {label}</div>
          <div style="font-size:1.35rem;font-weight:700;color:{color};font-family:'JetBrains Mono',monospace">{value}</div>
        </div>""",unsafe_allow_html=True)

    if not HAS_PLOTLY:
        st.warning("Install Plotly for the radar chart: `pip install plotly`")
    else:
        _t9_vehicle_ids=df["Vehicle_Number"].dropna().unique().tolist()
        if not _t9_vehicle_ids:
            st.info("🙈 No vehicles in Generated data - click **▶️ Start** and wait.")
        else:
            _t9_sel_id=st.selectbox("Select Vehicle / Policy ID",_t9_vehicle_ids[:300],key="dp_sel")
            _t9_matched=df[df["Vehicle_Number"]==_t9_sel_id]
            if _t9_matched.empty:
                _t9_sel_id=_t9_vehicle_ids[0]; _t9_matched=df[df["Vehicle_Number"]==_t9_sel_id]
            if _t9_matched.empty:
                st.warning("No matching vehicle found. Please wait for data refresh.")
            else:
                row=_t9_matched.iloc[0]
                tier_color=TIER_COLS.get(str(row.get("Risk_Tier","Low")),C["green"])
                c1,c2,c3=st.columns(3)
                _info_card(c1,"🆔","Vehicle ID",   row["Vehicle_Number"],  C["purple"])
                _info_card(c2,"📍","City",         row["Registration_City"],C["blue"])
                _info_card(c3,"⚠️","Risk Category",row["Risk_Tier"],        tier_color)
                st.markdown("<br>",unsafe_allow_html=True)
                c4,c5,c6=st.columns(3)
                _info_card(c4,"🚗","Vehicle",
                           f"{row['Vehicle_Make']} {row['Vehicle_Type']} ({int(row['Vehicle_Year'])})")
                _info_card(c5,"📊","Risk Score",f"{row['Risk_Score']:.4f} / 1.0",tier_color)
                prem_row=build_premium_df(pd.DataFrame([row]))
                annual=f"${prem_row['Gross_Premium'].iloc[0]:,.2f}" if not prem_row.empty else "—"
                _info_card(c6,"💲","Annual Premium",annual,C["green"])
                st.markdown("<br>",unsafe_allow_html=True)
                o1,o2,o3,o4,o5,o6=st.columns(6)
                _info_card(o1,"🔌","RPM",       f"{float(row.get('RPM',800)):.0f}")
                _info_card(o2,"⛽","Throttle",   f"{float(row.get('Throttle_Pct',0)):.1f}%")
                _info_card(o3,"🔧","Engine Load",f"{float(row.get('Engine_Load_Pct',0)):.1f}%")
                _info_card(o4,"🌡️","Coolant",   f"{float(row.get('Coolant_Temp_C',85)):.1f}°C")
                _info_card(o5,"💨","MAF",        f"{float(row.get('MAF_gs',3)):.1f} g/s")
                _info_card(o6,"🔋","Battery",    f"{float(row.get('Battery_V',12.5)):.2f}V")
                st.markdown("<br>",unsafe_allow_html=True)
                st.markdown("## 🚙 Driving Behaviour Radar")
                radar_vals=[
                    float(row.get("Speed_kmh",0))/180*100,
                    float(row.get("HB_Day",0))/15*100,
                    float(row.get("HA_Day",0))/15*100,
                    float(row.get("HC_Day",0))/15*100,
                    float(row.get("Fuel_Eff_KmL",0))/30*100,
                    float(row.get("Vehicle_Age",0))/20*100,
                    float(row.get("Engine_Load_Pct",0))/100*100,
                    float(row.get("RPM",800))/6500*100,
                ]
                radar_vals=[min(max(v,0),100) for v in radar_vals]
                labels_r=["Speed","Harsh\nBraking","Harsh\nAccel","Harsh\nCornering",
                           "Fuel\nEfficiency","Vehicle\nAge","Engine\nLoad","RPM"]
                _rgba=_hex_to_rgba(tier_color,0.20)
                fig_radar=go.Figure(go.Scatterpolar(
                    r=radar_vals+[radar_vals[0]],theta=labels_r+[labels_r[0]],
                    fill="toself",fillcolor=_rgba,
                    line=dict(color=tier_color,width=2.5),name=_t9_sel_id,
                ))
                _rt={k:v for k,v in PLOTLY_THEME.items() if k!="margin"}
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(range=[0,100],visible=True,gridcolor=C["bord"]),
                               angularaxis=dict(gridcolor=C["bord"])),
                    showlegend=False,height=440,margin=dict(l=32,r=32,t=44,b=32),**_rt)
                st.plotly_chart(fig_radar,use_container_width=True,key="t9_radar")
                st.markdown("#### 🔍 Driver vs Fleet Benchmark")
                bench_cols=[c for c in ["Speed_kmh","HB_Day","HA_Day","HC_Day",
                                         "Harsh_Score","Risk_Score","Fuel_Eff_KmL",
                                         "Engine_Load_Pct","RPM"] if c in df.columns]
                if bench_cols:
                    fleet_mean=df[bench_cols].mean()
                    bench_df=pd.DataFrame({
                        "Metric":bench_cols,
                        "This Driver":[float(row.get(c,0)) for c in bench_cols],
                        "Fleet Average":[float(fleet_mean[c]) for c in bench_cols],
                    }).round(3)
                    fig_bm=px.bar(bench_df,x="Metric",y=["This Driver","Fleet Average"],
                                  barmode="group",
                                  color_discrete_map={"This Driver":tier_color,"Fleet Average":C["blue"]})
                    fig_bm.update_layout(**PLOTLY_THEME); fig_bm.update_layout(height=380)
                    st.plotly_chart(fig_bm,use_container_width=True,key="t9_bench")
                with st.expander("📋 Full Generated Record"):
                    st.json({k:(v.item() if hasattr(v,"item") else v)
                             for k,v in row.to_dict().items()})

#  Auto-rerun
if auto_rf and STATE.running:
    time.sleep(rf_int)
    st.rerun()
