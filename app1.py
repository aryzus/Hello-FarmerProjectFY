import os
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mandi Price Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Force light theme + custom styles ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

/* Force light background everywhere */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="block-container"],
.main, .block-container {
    background-color: #f4f6fb !important;
    color: #1c1f2e !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Fix dropdown portal (renders outside main DOM) */
div[data-baseweb="popover"],
div[data-baseweb="popover"] *,
div[data-baseweb="menu"],
div[data-baseweb="menu"] *,
div[data-baseweb="select"] *,
ul[data-testid="stSelectboxVirtualDropdown"],
ul[data-testid="stSelectboxVirtualDropdown"] * {
    background-color: #ffffff !important;
    color: #1c1f2e !important;
    font-family: 'DM Sans', sans-serif !important;
}
div[data-baseweb="popover"] li:hover,
div[data-baseweb="menu"] li:hover {
    background-color: #eff6ff !important;
    color: #1d4ed8 !important;
}
/* Selected option highlight */
div[data-baseweb="menu"] [aria-selected="true"] {
    background-color: #dbeafe !important;
    color: #1d4ed8 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e3e7f0 !important;
}
[data-testid="stSidebar"] * {
    color: #1c1f2e !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.9rem !important;
    font-weight: 500 !important;
}

/* All text */
p, span, label, div, li {
    color: #1c1f2e !important;
    font-family: 'DM Sans', sans-serif !important;
}

h1, h2, h3, h4 {
    font-family: 'DM Serif Display', serif !important;
    color: #1c1f2e !important;
}

/* Radio buttons */
[data-testid="stRadio"] > div { gap: 6px; }
[data-testid="stRadio"] label {
    background: #f4f6fb;
    border: 1px solid #e3e7f0;
    border-radius: 8px;
    padding: 8px 14px !important;
    transition: all 0.15s;
}
[data-testid="stRadio"] label:hover { background: #e8ecf7; }

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: #ffffff !important;
    border: 1px solid #e3e7f0 !important;
    border-radius: 8px !important;
    color: #1c1f2e !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    gap: 6px;
    border-bottom: 2px solid #e3e7f0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border: none;
    color: #6b7280 !important;
    font-weight: 600;
    font-size: 0.88rem;
    padding: 8px 16px;
    border-radius: 6px 6px 0 0;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color: #2563eb !important;
    border-bottom: 2px solid #2563eb !important;
}

/* Info boxes */
[data-testid="stAlert"] {
    background: #eff6ff !important;
    border: 1px solid #bfdbfe !important;
    border-radius: 10px !important;
    color: #1d4ed8 !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    border: 1px solid #e3e7f0;
    overflow: hidden;
}

/* Download button */
[data-testid="stDownloadButton"] button {
    background: #2563eb !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.2rem !important;
}

/* Slider */
[data-testid="stSlider"] * { color: #1c1f2e !important; }

/* Number input */
[data-testid="stNumberInput"] input {
    background: #ffffff !important;
    border: 1px solid #e3e7f0 !important;
    border-radius: 8px !important;
    color: #1c1f2e !important;
}

/* ── Custom components ── */
.hero {
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 50%, #1d4ed8 100%);
    border-radius: 18px;
    padding: 2.8rem 3rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '🌾';
    position: absolute;
    right: 3rem; top: 50%;
    transform: translateY(-50%);
    font-size: 6rem;
    opacity: 0.15;
}
.hero h1 {
    color: #ffffff !important;
    font-size: 2.4rem;
    margin: 0 0 0.5rem;
    line-height: 1.2;
}
.hero p { color: #bfdbfe !important; font-size: 1rem; margin: 0; }
.hero .accent { color: #7dd3fc !important; }

.kpi-card {
    background: #ffffff;
    border: 1px solid #e3e7f0;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    box-shadow: 0 1px 8px rgba(37,99,235,0.06);
    transition: transform 0.15s, box-shadow 0.15s;
    margin-bottom: 1rem;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(37,99,235,0.1);
}
.kpi-card .lbl {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6b7280 !important;
    margin-bottom: 0.4rem;
}
.kpi-card .val {
    font-size: 1.85rem;
    font-weight: 700;
    color: #1c1f2e !important;
    font-family: 'DM Sans', sans-serif !important;
    line-height: 1.1;
}
.kpi-card .dlt {
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 0.3rem;
}

.section-hdr {
    font-family: 'DM Serif Display', serif;
    font-size: 1.2rem;
    color: #1c1f2e !important;
    margin: 2rem 0 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #e3e7f0;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.pill {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.04em;
}
.pill-green  { background: #dcfce7; color: #15803d; }
.pill-blue   { background: #dbeafe; color: #1d4ed8; }
.pill-purple { background: #ede9fe; color: #6d28d9; }
.pill-orange { background: #ffedd5; color: #c2410c; }

.callout {
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    font-size: 0.9rem;
    font-weight: 500;
    margin: 0.6rem 0;
}
.callout-green  { background: #f0fdf4; border-left: 4px solid #22c55e; color: #15803d !important; }
.callout-red    { background: #fef2f2; border-left: 4px solid #ef4444; color: #b91c1c !important; }
.callout-blue   { background: #eff6ff; border-left: 4px solid #3b82f6; color: #1d4ed8 !important; }
.callout * { color: inherit !important; }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib light style ─────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  '#ffffff',
    'axes.facecolor':    '#ffffff',
    'axes.edgecolor':    '#e3e7f0',
    'axes.labelcolor':   '#6b7280',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'xtick.color':       '#9ca3af',
    'ytick.color':       '#9ca3af',
    'grid.color':        '#f0f2f8',
    'grid.linestyle':    '-',
    'grid.linewidth':    0.8,
    'text.color':        '#1c1f2e',
    'legend.facecolor':  '#ffffff',
    'legend.edgecolor':  '#e3e7f0',
    'legend.framealpha': 1,
    'font.family':       'sans-serif',
    'font.size':         10,
})

PAL = {
    'navy':   '#1e3a8a',
    'blue':   '#2563eb',
    'sky':    '#38bdf8',
    'teal':   '#0d9488',
    'green':  '#16a34a',
    'amber':  '#d97706',
    'red':    '#dc2626',
    'purple': '#7c3aed',
    'gray':   '#9ca3af',
    'seq':    ['#1e3a8a','#2563eb','#0d9488','#d97706','#dc2626','#7c3aed'],
}

# ── Data ───────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, 'Agriculture_price_dataset.csv'))
    df['Price Date']  = pd.to_datetime(df['Price Date'])
    df['Commodity']   = df['Commodity'].str.strip()
    df['Market Name'] = df['Market Name'].str.strip()
    if 'State Name'    in df.columns: df['State Name']    = df['State Name'].str.strip()
    if 'District Name' in df.columns: df['District Name'] = df['District Name'].str.strip()
    return df

raw_df = load_data()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌾 Mandi Intelligence")
    st.markdown("---")
    page = st.radio("", [
        "🏠  Home",
        "📈  Price Prediction",
        "🔍  Market Explorer",
        "🔮  Future Forecast",
        "⚔️  Model Comparison",
        "📅  Seasonal Analysis",
        "🌦️  Climate Analysis",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**📍 Location**")
    has_state    = 'State Name'    in raw_df.columns
    has_district = 'District Name' in raw_df.columns
    states       = sorted(raw_df['State Name'].dropna().unique()) if has_state else []
    selected_state    = st.selectbox("State", ["All"] + states) if states else "All"

    dist_pool = []
    if has_district:
        dist_pool = (raw_df[raw_df['State Name'] == selected_state]['District Name'].dropna().unique()
                     if selected_state != "All" else raw_df['District Name'].dropna().unique())
    selected_district = st.selectbox("District", ["All"] + sorted(dist_pool)) if len(dist_pool) else "All"

    filtered_df = raw_df.copy()
    if selected_state    != "All" and has_state:    filtered_df = filtered_df[filtered_df['State Name']    == selected_state]
    if selected_district != "All" and has_district: filtered_df = filtered_df[filtered_df['District Name'] == selected_district]

    markets     = sorted(filtered_df['Market Name'].dropna().unique())
    commodities = sorted(filtered_df['Commodity'].dropna().unique())

    st.markdown("**🛒 Selection**")
    selected_market    = st.selectbox("Market",    markets     or ["—"])
    selected_commodity = st.selectbox("Commodity", commodities or ["—"])

    st.markdown("---")
    st.markdown("**🔔 Price Alert**")
    alert_enabled   = st.toggle("Enable alert", value=False)
    alert_threshold = st.number_input("Alert threshold (₹)", min_value=0, value=2000, step=100) if alert_enabled else None

# ── Helpers ────────────────────────────────────────────────────────────────────
def prepare_series(df, commodity, market):
    ts = df[
        (df['Commodity'].str.lower() == commodity.lower()) &
        (df['Market Name'] == market)
    ][['Price Date','Modal_Price']].copy()
    ts = ts.sort_values('Price Date').set_index('Price Date')
    return ts[~ts.index.duplicated(keep='last')]

def add_features(ts):
    t = ts.copy()
    t['lag_1']          = t['Modal_Price'].shift(1)
    t['lag_7']          = t['Modal_Price'].shift(7)
    t['rolling_mean_7'] = t['Modal_Price'].rolling(7).mean()
    t['rolling_std_7']  = t['Modal_Price'].rolling(7).std()
    return t.dropna()

def do_split(ts_feat, ratio=0.8):
    n = int(len(ts_feat) * ratio)
    X = ts_feat[['lag_1','lag_7','rolling_mean_7','rolling_std_7']]
    y = ts_feat['Modal_Price']
    return X[:n], X[n:], y[:n], y[n:]

def calc_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true))) * 100
    return mae, rmse, mape

def kpi_card(label, value, delta="", delta_color="#6b7280"):
    return f"""<div class="kpi-card">
        <div class="lbl">{label}</div>
        <div class="val">{value}</div>
        <div class="dlt" style="color:{delta_color} !important">{delta}</div>
    </div>"""

def sec(title):
    st.markdown(f'<div class="section-hdr">{title}</div>', unsafe_allow_html=True)

def callout(msg, kind="blue"):
    st.markdown(f'<div class="callout callout-{kind}">{msg}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":
    st.markdown(f"""
    <div class="hero">
        <h1>Mandi Price <span class="accent">Intelligence</span></h1>
        <p>AI-powered agricultural price prediction across Indian mandis.<br>
        Select your location and commodity from the sidebar to explore.</p>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, lbl, val in zip([c1,c2,c3,c4],
        ["Total Markets","Commodities","Data Points","Date Range"],
        [f"{raw_df['Market Name'].nunique():,}",
         f"{raw_df['Commodity'].nunique():,}",
         f"{len(raw_df):,}",
         f"{raw_df['Price Date'].min().strftime('%b %Y')} – {raw_df['Price Date'].max().strftime('%b %Y')}"]):
        col.markdown(kpi_card(lbl, val), unsafe_allow_html=True)

    sec("📍 Your Selection")
    i1, i2 = st.columns(2)
    i1.info(f"**State:** {selected_state}  \n**District:** {selected_district}  \n**Market:** {selected_market}")
    i2.info(f"**Commodity:** {selected_commodity}  \n**Records in filter:** {len(filtered_df):,}")

    sec("📊 Price Snapshot")
    snap = prepare_series(filtered_df, selected_commodity, selected_market)
    if len(snap) > 10:
        latest = snap['Modal_Price'].iloc[-1]
        prev   = snap['Modal_Price'].iloc[-2]
        chg    = (latest - prev) / prev * 100 if prev else 0
        s1,s2,s3,s4 = st.columns(4)
        for col, lbl, val, dlt, dc in zip(
            [s1,s2,s3,s4],
            ["Latest Price","All-Time High","All-Time Low","Average"],
            [f"₹{latest:,.0f}", f"₹{snap['Modal_Price'].max():,.0f}",
             f"₹{snap['Modal_Price'].min():,.0f}", f"₹{snap['Modal_Price'].mean():,.0f}"],
            [f"{'▲' if chg>=0 else '▼'} {abs(chg):.1f}% vs prev","","",""],
            [PAL['green'] if chg>=0 else PAL['red'],'#6b7280','#6b7280','#6b7280']):
            col.markdown(kpi_card(lbl, val, dlt, dc), unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(12, 3.5))
        ax.plot(snap.index, snap['Modal_Price'], color=PAL['blue'], linewidth=2)
        ax.fill_between(snap.index, snap['Modal_Price'], snap['Modal_Price'].min(),
                        alpha=0.08, color=PAL['blue'])
        ax.set_title(f"{selected_commodity} · {selected_market} — Full Price History",
                     fontsize=12, fontweight='600', pad=12, color='#1c1f2e')
        ax.set_ylabel("Price (₹)", fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.xticks(rotation=30, fontsize=9)
        ax.grid(True, alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig); plt.close()
    else:
        st.warning("Not enough data for this selection. Try adjusting the filters.")

# ══════════════════════════════════════════════════════════════════════════════
# PRICE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Price Prediction":
    st.markdown('<h2>📈 Price Prediction</h2>', unsafe_allow_html=True)
    st.caption(f"{selected_commodity} · {selected_market}")

    ts = prepare_series(filtered_df, selected_commodity, selected_market)
    if len(ts) < 30:
        st.error("Need at least 30 records. Adjust filters."); st.stop()

    ts_feat = add_features(ts)
    X_tr, X_te, y_tr, y_te = do_split(ts_feat)

    with st.spinner("Training XGBoost model…"):
        model = XGBRegressor(n_estimators=200, learning_rate=0.08, max_depth=4, random_state=42)
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        y_pred = model.predict(X_te)

    mae, rmse, mape = calc_metrics(y_te, y_pred)

    m1,m2,m3 = st.columns(3)
    for col, lbl, val, note in zip([m1,m2,m3],
        ["MAE","RMSE","MAPE"],
        [f"₹{mae:.1f}", f"₹{rmse:.1f}", f"{mape:.2f}%"],
        ["Mean Absolute Error","Root Mean Sq. Error","Mean Abs. % Error"]):
        col.markdown(kpi_card(lbl, val, note), unsafe_allow_html=True)

    st.markdown("")
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(12,8), gridspec_kw={'height_ratios':[3,1]})
    ax1.plot(y_te.values, color=PAL['navy'],  linewidth=2,   label='Actual')
    ax1.plot(y_pred,      color=PAL['amber'], linewidth=1.8, linestyle='--', label='Predicted')
    ax1.fill_between(range(len(y_te)), y_te.values, y_pred, alpha=0.06, color=PAL['amber'])
    ax1.set_title("Actual vs Predicted Prices", fontsize=13, fontweight='600', pad=10, color='#1c1f2e')
    ax1.set_ylabel("Price (₹)"); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.6)

    residuals = y_te.values - y_pred
    ax2.bar(range(len(residuals)), residuals,
            color=[PAL['green'] if r>=0 else PAL['red'] for r in residuals], alpha=0.7, width=0.8)
    ax2.axhline(0, color=PAL['gray'], linewidth=1.2)
    ax2.set_title("Residuals", fontsize=10, pad=8, color='#1c1f2e')
    ax2.set_ylabel("Error (₹)"); ax2.grid(True, alpha=0.6)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    sec("🔑 Feature Importance")
    feat_names  = ['Lag 1 day','Lag 7 days','Rolling Mean 7d','Rolling Std 7d']
    feat_colors = [PAL['navy'], PAL['blue'], PAL['teal'], PAL['amber']]
    fig2, ax3 = plt.subplots(figsize=(8,3))
    bars = ax3.barh(feat_names, model.feature_importances_, color=feat_colors, height=0.5)
    for bar, val in zip(bars, model.feature_importances_):
        ax3.text(val+0.003, bar.get_y()+bar.get_height()/2,
                 f'{val:.3f}', va='center', fontsize=9, color='#4b5563')
    ax3.set_xlabel("Importance Score", fontsize=9)
    ax3.set_title("Feature Importances", fontsize=11, fontweight='600', pad=10, color='#1c1f2e')
    ax3.grid(True, alpha=0.5, axis='x')
    plt.tight_layout(); st.pyplot(fig2); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# MARKET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍  Market Explorer":
    st.markdown('<h2>🔍 Market Explorer</h2>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📉 Price Trend", "🏪 Market Comparison", "📦 Commodity Comparison"])

    with tab1:
        ts = prepare_series(filtered_df, selected_commodity, selected_market)
        if len(ts) < 10: st.warning("Not enough data."); st.stop()
        ts['MA_7']  = ts['Modal_Price'].rolling(7).mean()
        ts['MA_30'] = ts['Modal_Price'].rolling(30).mean()
        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(ts.index, ts['Modal_Price'], color='#cbd5e1', linewidth=1, label='Daily')
        ax.plot(ts.index, ts['MA_7'],  color=PAL['amber'], linewidth=1.8, label='7-day MA')
        ax.plot(ts.index, ts['MA_30'], color=PAL['navy'],  linewidth=2.2, label='30-day MA')
        ax.fill_between(ts.index, ts['Modal_Price'], ts['Modal_Price'].min(), alpha=0.04, color=PAL['blue'])
        ax.set_title(f"{selected_commodity} · {selected_market}", fontsize=13, fontweight='600', color='#1c1f2e')
        ax.set_ylabel("Price (₹)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.xticks(rotation=30, fontsize=9); ax.legend(fontsize=9); ax.grid(True, alpha=0.6)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        sec("📋 Statistics")
        stats = ts['Modal_Price'].describe().rename({
            'count':'Count','mean':'Mean (₹)','std':'Std Dev (₹)',
            'min':'Min (₹)','25%':'Q1 (₹)','50%':'Median (₹)','75%':'Q3 (₹)','max':'Max (₹)'})
        st.dataframe(stats.to_frame("Value").style.format("₹{:.2f}"), use_container_width=True)

    with tab2:
        top_markets = (
            filtered_df[filtered_df['Commodity'].str.lower() == selected_commodity.lower()]
            .groupby('Market Name')['Modal_Price'].count().nlargest(6).index.tolist())
        fig, ax = plt.subplots(figsize=(12,5))
        for mkt, color in zip(top_markets, PAL['seq']):
            mts = prepare_series(filtered_df, selected_commodity, mkt)
            if len(mts) > 10:
                ax.plot(mts.index, mts['Modal_Price'].rolling(30).mean(),
                        label=mkt, color=color, linewidth=1.8)
        ax.set_title(f"{selected_commodity} — 30-day MA across top markets", fontsize=13, fontweight='600', color='#1c1f2e')
        ax.set_ylabel("Price (₹)"); ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.xticks(rotation=30, fontsize=9); ax.grid(True, alpha=0.6)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab3:
        top_comms = (
            filtered_df[filtered_df['Market Name'] == selected_market]
            .groupby('Commodity')['Modal_Price'].count().nlargest(6).index.tolist())
        fig, ax = plt.subplots(figsize=(12,5))
        for comm, color in zip(top_comms, PAL['seq']):
            cts = prepare_series(filtered_df, comm, selected_market)
            if len(cts) > 10:
                norm = (cts['Modal_Price'] / cts['Modal_Price'].iloc[0]) * 100
                ax.plot(cts.index, norm, label=comm, color=color, linewidth=1.8)
        ax.set_title(f"Price Index at {selected_market} (Base=100)", fontsize=13, fontweight='600', color='#1c1f2e')
        ax.set_ylabel("Index"); ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.xticks(rotation=30, fontsize=9); ax.grid(True, alpha=0.6)
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# FUTURE FORECAST
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮  Future Forecast":
    st.markdown('<h2>🔮 Future Forecast</h2>', unsafe_allow_html=True)
    st.caption(f"{selected_commodity} · {selected_market}")

    horizon = st.slider("Forecast horizon (days)", 7, 60, 30, 7)
    ts = prepare_series(filtered_df, selected_commodity, selected_market)
    if len(ts) < 30: st.error("Not enough data."); st.stop()

    ts_feat = add_features(ts)
    X_all = ts_feat[['lag_1','lag_7','rolling_mean_7','rolling_std_7']]
    y_all = ts_feat['Modal_Price']

    with st.spinner("Generating forecast…"):
        model = XGBRegressor(n_estimators=200, learning_rate=0.08, max_depth=4, random_state=42)
        model.fit(X_all, y_all)
        history = ts['Modal_Price'].tolist()
        future_prices = []
        for _ in range(horizon):
            lag_1 = history[-1]; lag_7 = history[-7] if len(history)>=7 else history[0]
            pred  = float(model.predict(pd.DataFrame(
                [[lag_1, lag_7, np.mean(history[-7:]), np.std(history[-7:])]],
                columns=['lag_1','lag_7','rolling_mean_7','rolling_std_7']))[0])
            future_prices.append(pred); history.append(pred)

    last_date    = ts.index[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq='D')
    fdf          = pd.DataFrame({'Forecast': future_prices}, index=future_dates)
    res_std      = np.std(y_all.values - model.predict(X_all))
    fdf['Upper'] = fdf['Forecast'] + 1.5*res_std
    fdf['Lower'] = fdf['Forecast'] - 1.5*res_std

    if alert_enabled and alert_threshold:
        breaches = fdf[fdf['Forecast'] > alert_threshold]
        if not breaches.empty:
            callout(f"⚠️ Price alert! Forecast exceeds ₹{alert_threshold:,} from <b>{breaches.index[0].strftime('%d %b %Y')}</b>.", "red")
        else:
            callout(f"✅ Forecast stays below ₹{alert_threshold:,} for the entire {horizon}-day period.", "green")

    hist_plot = ts['Modal_Price'].iloc[-60:]
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(hist_plot.index, hist_plot.values, color=PAL['navy'], linewidth=2, label='Historical')
    ax.plot(fdf.index, fdf['Forecast'], color=PAL['teal'], linewidth=2.2, linestyle='--', label='Forecast')
    ax.fill_between(fdf.index, fdf['Lower'], fdf['Upper'], alpha=0.12, color=PAL['teal'], label='Confidence band')
    ax.axvline(x=last_date, color=PAL['gray'], linestyle=':', linewidth=1.5, label='Forecast start')
    if alert_enabled and alert_threshold:
        ax.axhline(y=alert_threshold, color=PAL['red'], linestyle='--', linewidth=1.2, label=f'Alert ₹{alert_threshold:,}')
    ax.set_title(f"{horizon}-Day Forecast · {selected_commodity} · {selected_market}",
                 fontsize=13, fontweight='600', color='#1c1f2e')
    ax.set_ylabel("Price (₹)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.xticks(rotation=30, fontsize=9); ax.legend(fontsize=9); ax.grid(True, alpha=0.6)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    sec("📋 Forecast Table")
    disp = fdf.copy(); disp.index = disp.index.strftime('%d %b %Y')
    disp.columns = ['Forecast (₹)','Upper Band (₹)','Lower Band (₹)']
    st.dataframe(disp.style.format("₹{:.2f}"), use_container_width=True)
    csv = disp.to_csv().encode('utf-8')
    st.download_button("⬇️ Download Forecast CSV", csv,
                       f"forecast_{selected_commodity}_{selected_market}.csv", "text/csv")

    sec("📌 Summary")
    current = ts['Modal_Price'].iloc[-1]; end_p = fdf['Forecast'].iloc[-1]
    chg = (end_p - current) / current * 100
    f1,f2,f3 = st.columns(3)
    for col, lbl, val, dc in zip([f1,f2,f3],
        ["Current Price", f"Price in {horizon} Days", "Expected Change"],
        [f"₹{current:,.0f}", f"₹{end_p:,.0f}", f"{'▲' if chg>=0 else '▼'} {abs(chg):.1f}%"],
        ['#6b7280','#6b7280', PAL['green'] if chg>=0 else PAL['red']]):
        col.markdown(kpi_card(lbl, val, delta_color=dc), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚔️  Model Comparison":
    st.markdown('<h2>⚔️ Model Comparison</h2>', unsafe_allow_html=True)
    st.caption(f"XGBoost vs Linear Regression vs ARIMA · {selected_commodity} · {selected_market}")

    ts = prepare_series(filtered_df, selected_commodity, selected_market)
    if len(ts) < 40: st.error("Need at least 40 records."); st.stop()

    ts_feat = add_features(ts)
    X_tr, X_te, y_tr, y_te = do_split(ts_feat)

    results = {}
    with st.spinner("Training all models…"):
        xgb = XGBRegressor(n_estimators=200, learning_rate=0.08, max_depth=4, random_state=42)
        xgb.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        results['XGBoost'] = {'pred': xgb.predict(X_te), 'color': PAL['amber'], 'pill': 'pill-orange'}

        lr = LinearRegression()
        lr.fit(X_tr, y_tr)
        results['Linear Regression'] = {'pred': lr.predict(X_te), 'color': PAL['blue'], 'pill': 'pill-blue'}

        try:
            arima_model = ARIMA(y_tr.values, order=(5,1,0)).fit()
            results['ARIMA'] = {'pred': arima_model.forecast(steps=len(y_te)), 'color': PAL['purple'], 'pill': 'pill-purple'}
        except Exception:
            st.warning("ARIMA failed to converge for this dataset.")

    sec("📊 Accuracy Metrics")
    rows = []
    for name, res in results.items():
        mae, rmse, mape = calc_metrics(y_te.values, res['pred'])
        rows.append({'Model': name, 'MAE (₹)': round(mae,2), 'RMSE (₹)': round(rmse,2), 'MAPE (%)': round(mape,2)})
    mdf = pd.DataFrame(rows).set_index('Model')
    best_rmse = mdf['RMSE (₹)'].idxmin()
    best_mape = mdf['MAPE (%)'].idxmin()

    st.dataframe(
        mdf.style
           .highlight_min(axis=0, color='#dcfce7')
           .format({'MAE (₹)':'₹{:.2f}','RMSE (₹)':'₹{:.2f}','MAPE (%)':'{:.2f}%'}),
        use_container_width=True)
    callout(f"🏆 Best on RMSE: <b>{best_rmse}</b> &nbsp;·&nbsp; Best on MAPE: <b>{best_mape}</b>", "blue")

    sec("📈 All Models vs Actual")
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(y_te.values, color=PAL['navy'], linewidth=2.5, label='Actual', zorder=5)
    for name, res in results.items():
        ax.plot(res['pred'], color=res['color'], linewidth=1.8, linestyle='--', label=name, alpha=0.9)
    ax.set_title("Model Predictions vs Actual", fontsize=13, fontweight='600', color='#1c1f2e', pad=10)
    ax.set_ylabel("Price (₹)"); ax.legend(fontsize=9); ax.grid(True, alpha=0.6)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    sec("📉 Residual Distributions")
    fig2, axes = plt.subplots(1, len(results), figsize=(5*len(results), 4))
    if len(results) == 1: axes = [axes]
    for ax, (name, res) in zip(axes, results.items()):
        errs = np.array(y_te.values) - np.array(res['pred'])
        ax.hist(errs, bins=20, color=res['color'], alpha=0.85, edgecolor='white')
        ax.axvline(0, color=PAL['navy'], linewidth=1.8)
        ax.set_title(name, fontsize=10, fontweight='600', color='#1c1f2e')
        ax.set_xlabel("Error (₹)", fontsize=9); ax.grid(True, alpha=0.5)
    plt.tight_layout(); st.pyplot(fig2); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# SEASONAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📅  Seasonal Analysis":
    st.markdown('<h2>📅 Seasonal Analysis</h2>', unsafe_allow_html=True)
    st.caption(f"{selected_commodity} · {selected_market}")

    ts = prepare_series(filtered_df, selected_commodity, selected_market)
    if len(ts) < 30: st.error("Not enough data."); st.stop()

    ts = ts.copy()
    ts['Month']     = ts.index.month
    ts['MonthName'] = ts.index.strftime('%b')
    ts['Year']      = ts.index.year
    ts['Quarter']   = ts.index.quarter

    month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    monthly_avg = ts.groupby('MonthName')['Modal_Price'].mean().reindex(month_order).dropna()
    monthly_std = ts.groupby('MonthName')['Modal_Price'].std().reindex(month_order).dropna()
    best_month  = monthly_avg.idxmin()
    worst_month = monthly_avg.idxmax()

    sec("🗓️ Monthly Average Prices")
    b1,b2 = st.columns(2)
    b1.markdown(kpi_card("📉 Cheapest Month", best_month,
                         f"Avg ₹{monthly_avg.min():,.0f}", PAL['green']), unsafe_allow_html=True)
    b2.markdown(kpi_card("📈 Costliest Month", worst_month,
                         f"Avg ₹{monthly_avg.max():,.0f}", PAL['red']),   unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(12,4.5))
    bar_colors = [PAL['red'] if m==worst_month else PAL['green'] if m==best_month else '#bfdbfe'
                  for m in monthly_avg.index]
    bars = ax.bar(monthly_avg.index, monthly_avg.values, color=bar_colors, width=0.6,
                  yerr=monthly_std.reindex(monthly_avg.index).values,
                  error_kw={'ecolor':'#9ca3af','capsize':4})
    for bar, val in zip(bars, monthly_avg.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+18,
                f'₹{val:,.0f}', ha='center', fontsize=7.5, color='#4b5563')
    ax.set_title(f"Average Price by Month · {selected_commodity}",
                 fontsize=13, fontweight='600', color='#1c1f2e', pad=10)
    ax.set_ylabel("Avg Price (₹)"); ax.grid(True, alpha=0.5, axis='y')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    sec("📆 Year-over-Year Trends")
    yearly = ts.groupby(['Year','MonthName'])['Modal_Price'].mean().reset_index()
    years  = sorted(yearly['Year'].unique())
    blues  = plt.cm.Blues(np.linspace(0.35, 0.9, len(years)))
    fig2, ax2 = plt.subplots(figsize=(12,5))
    for year, color in zip(years, blues):
        yd = yearly[yearly['Year']==year].set_index('MonthName').reindex(month_order)
        ax2.plot(month_order, yd['Modal_Price'].values, label=str(year),
                 color=color, linewidth=1.8, marker='o', markersize=4)
    ax2.set_title(f"Year-over-Year · {selected_commodity}",
                  fontsize=13, fontweight='600', color='#1c1f2e')
    ax2.set_ylabel("Avg Price (₹)")
    ax2.legend(title='Year', fontsize=8, title_fontsize=8)
    ax2.grid(True, alpha=0.5)
    plt.tight_layout(); st.pyplot(fig2); plt.close()

    sec("📊 Quarterly Distribution")
    q_data = [ts[ts['Quarter']==q]['Modal_Price'].dropna().values for q in [1,2,3,4]]
    fig3, ax3 = plt.subplots(figsize=(9,4.5))
    bp = ax3.boxplot(q_data, labels=['Q1\nJan–Mar','Q2\nApr–Jun','Q3\nJul–Sep','Q4\nOct–Dec'],
                     patch_artist=True, widths=0.45,
                     medianprops={'color': PAL['navy'], 'linewidth':2.5})
    for patch, color in zip(bp['boxes'], ['#dbeafe','#dcfce7','#fef3c7','#ede9fe']):
        patch.set_facecolor(color); patch.set_alpha(0.85)
    ax3.set_title(f"Quarterly Price Distribution · {selected_commodity}",
                  fontsize=13, fontweight='600', color='#1c1f2e')
    ax3.set_ylabel("Price (₹)"); ax3.grid(True, alpha=0.5, axis='y')
    plt.tight_layout(); st.pyplot(fig3); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# CLIMATE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌦️  Climate Analysis":
    st.markdown('<h2>🌦️ Climate & Price Analysis</h2>', unsafe_allow_html=True)
    st.caption("Nashik District · Lasalgaon Market · Onion")

    # ── Generate structured Nashik climate data based on real IMD seasonal patterns ──
    @st.cache_data
    def generate_climate_data(start, end):
        dates = pd.date_range(start, end, freq='D')
        np.random.seed(42)
        n = len(dates)
        months = dates.month

        # Real Nashik avg temperature ranges by month (IMD based)
        temp_base  = {1:18,2:21,3:26,4:31,5:34,6:30,7:26,8:26,9:27,10:26,11:22,12:18}
        rain_base  = {1:1,2:1,3:2,4:3,5:8,6:90,7:160,8:140,9:80,10:30,11:5,12:2}
        humid_base = {1:55,2:50,3:45,4:42,5:48,6:72,7:88,8:87,9:80,10:68,11:60,12:57}

        temp     = np.array([temp_base[m]  for m in months], dtype=float)
        rainfall = np.array([rain_base[m]  for m in months], dtype=float)
        humidity = np.array([humid_base[m] for m in months], dtype=float)

        # Add daily noise
        temp     += np.random.normal(0, 1.5, n)
        rainfall  = np.maximum(0, rainfall/30 + np.random.exponential(0.3, n))
        humidity += np.random.normal(0, 4, n)
        humidity  = np.clip(humidity, 30, 98)

        return pd.DataFrame({
            'Temperature': temp.round(1),
            'Rainfall':    rainfall.round(2),
            'Humidity':    humidity.round(1),
        }, index=dates)

    # Get price series for Lasalgaon onion
    ts_climate = prepare_series(raw_df, 'Onion', 'Lasalgaon')
    if len(ts_climate) < 10:
        st.error("Not enough Lasalgaon Onion data for climate analysis."); st.stop()

    climate_df = generate_climate_data(ts_climate.index.min(), ts_climate.index.max())

    # Merge price + climate on common dates
    merged = ts_climate.join(climate_df, how='inner').dropna()

    if len(merged) < 10:
        st.error("Not enough overlapping dates between price and climate data."); st.stop()

    # ── KPIs ──
    sec("🌡️ Climate Overview — Nashik District")
    k1, k2, k3, k4 = st.columns(4)
    for col, lbl, val in zip([k1,k2,k3,k4],
        ["Avg Temperature","Avg Rainfall/day","Avg Humidity","Data Points"],
        [f"{merged['Temperature'].mean():.1f} °C",
         f"{merged['Rainfall'].mean():.2f} mm",
         f"{merged['Humidity'].mean():.1f} %",
         f"{len(merged):,} days"]):
        col.markdown(kpi_card(lbl, val), unsafe_allow_html=True)

    # ── Chart 1: Price + Climate overlay ──
    sec("📊 Price vs Climate Variables Over Time")
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    axes[0].plot(merged.index, merged['Modal_Price'], color=PAL['navy'], linewidth=1.8)
    axes[0].fill_between(merged.index, merged['Modal_Price'], merged['Modal_Price'].min(), alpha=0.07, color=PAL['blue'])
    axes[0].set_ylabel("Price (₹/quintal)"); axes[0].set_title("Modal Price", fontsize=11, fontweight='600', color='#1c1f2e')
    axes[0].grid(True, alpha=0.5)

    axes[1].plot(merged.index, merged['Temperature'], color='#dc2626', linewidth=1.5)
    axes[1].fill_between(merged.index, merged['Temperature'], merged['Temperature'].min(), alpha=0.08, color='#dc2626')
    axes[1].set_ylabel("°C"); axes[1].set_title("Temperature (°C)", fontsize=11, fontweight='600', color='#1c1f2e')
    axes[1].grid(True, alpha=0.5)

    axes[2].bar(merged.index, merged['Rainfall'], color='#2563eb', alpha=0.7, width=1)
    axes[2].set_ylabel("mm"); axes[2].set_title("Daily Rainfall (mm)", fontsize=11, fontweight='600', color='#1c1f2e')
    axes[2].grid(True, alpha=0.5)

    axes[3].plot(merged.index, merged['Humidity'], color='#0d9488', linewidth=1.5)
    axes[3].fill_between(merged.index, merged['Humidity'], merged['Humidity'].min(), alpha=0.08, color='#0d9488')
    axes[3].set_ylabel("%"); axes[3].set_title("Humidity (%)", fontsize=11, fontweight='600', color='#1c1f2e')
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    axes[3].grid(True, alpha=0.5)

    plt.xticks(rotation=30, fontsize=9)
    plt.suptitle("Lasalgaon Onion Price vs Nashik Climate Variables", fontsize=13, fontweight='600', y=1.01, color='#1c1f2e')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # ── Chart 2: Scatter plots ──
    sec("🔗 Price–Climate Correlation")
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))
    climate_vars = [('Temperature','#dc2626'), ('Rainfall','#2563eb'), ('Humidity','#0d9488')]

    for ax, (var, color) in zip(axes2, climate_vars):
        ax.scatter(merged[var], merged['Modal_Price'], alpha=0.35, color=color, s=18, edgecolors='none')
        # Trend line
        z = np.polyfit(merged[var], merged['Modal_Price'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(merged[var].min(), merged[var].max(), 100)
        ax.plot(x_line, p(x_line), color='#1c1f2e', linewidth=2, linestyle='--', label='Trend')
        corr = merged[var].corr(merged['Modal_Price'])
        ax.set_xlabel(var, fontsize=10)
        ax.set_ylabel("Price (₹/quintal)", fontsize=10)
        ax.set_title(f"{var} vs Price\n(r = {corr:.3f})", fontsize=11, fontweight='600', color='#1c1f2e')
        ax.grid(True, alpha=0.5)
        strength = "Strong" if abs(corr)>0.5 else "Moderate" if abs(corr)>0.3 else "Weak"
        direction = "positive" if corr > 0 else "negative"
        ax.text(0.05, 0.95, f"{strength} {direction}", transform=ax.transAxes,
                fontsize=9, color=color, fontweight='600', va='top')

    plt.suptitle("Scatter Plots: Climate Variables vs Onion Price", fontsize=13, fontweight='600', y=1.02, color='#1c1f2e')
    plt.tight_layout()
    st.pyplot(fig2); plt.close()

    # ── Correlation table ──
    sec("📋 Correlation Summary")
    corr_rows = []
    for var in ['Temperature', 'Rainfall', 'Humidity']:
        r = merged[var].corr(merged['Modal_Price'])
        strength = "Strong" if abs(r)>0.5 else "Moderate" if abs(r)>0.3 else "Weak"
        direction = "Positive ▲" if r>0 else "Negative ▼"
        corr_rows.append({'Climate Variable': var, 'Pearson r': round(r,4),
                          'Strength': strength, 'Direction': direction})
    corr_df = pd.DataFrame(corr_rows).set_index('Climate Variable')
    st.dataframe(corr_df.style.background_gradient(subset=['Pearson r'], cmap='Blues'), use_container_width=True)

    # ── Chart 3: Monthly avg climate vs monthly avg price ──
    sec("📅 Monthly Climate vs Price Patterns")
    merged['Month'] = merged.index.month
    monthly = merged.groupby('Month').agg({
        'Modal_Price': 'mean',
        'Temperature': 'mean',
        'Rainfall':    'mean',
        'Humidity':    'mean'
    }).round(1)

    month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    monthly.index = [month_labels[i-1] for i in monthly.index]

    fig3, ax_main = plt.subplots(figsize=(12, 5))
    ax_twin = ax_main.twinx()

    bars = ax_main.bar(monthly.index, monthly['Modal_Price'],
                       color=[PAL['blue'] if p < monthly['Modal_Price'].mean() else PAL['navy']
                              for p in monthly['Modal_Price']],
                       alpha=0.75, width=0.4, label='Avg Price (₹)')
    ax_twin.plot(monthly.index, monthly['Rainfall'], color='#2563eb', linewidth=2,
                 marker='o', markersize=5, label='Avg Rainfall (mm)', linestyle='--')
    ax_twin.plot(monthly.index, monthly['Temperature'], color='#dc2626', linewidth=2,
                 marker='s', markersize=5, label='Avg Temp (°C)')

    ax_main.set_ylabel("Avg Price (₹/quintal)", color=PAL['navy'], fontsize=10)
    ax_twin.set_ylabel("Rainfall (mm) / Temp (°C)", fontsize=10)
    ax_main.set_title("Monthly Avg Price vs Climate Variables", fontsize=13, fontweight='600', color='#1c1f2e', pad=10)

    lines1, labels1 = ax_main.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax_main.legend(lines1+lines2, labels1+labels2, fontsize=9, loc='upper left')
    ax_main.grid(True, alpha=0.4, axis='y')
    plt.tight_layout()
    st.pyplot(fig3); plt.close()

    # ── XGBoost with climate features ──
    sec("🤖 XGBoost Model: With vs Without Climate Features")

    ts_feat_base = prepare_series(raw_df, 'Onion', 'Lasalgaon').copy()
    ts_feat_base['lag_1']          = ts_feat_base['Modal_Price'].shift(1)
    ts_feat_base['lag_7']          = ts_feat_base['Modal_Price'].shift(7)
    ts_feat_base['rolling_mean_7'] = ts_feat_base['Modal_Price'].rolling(7).mean()
    ts_feat_base['rolling_std_7']  = ts_feat_base['Modal_Price'].rolling(7).std()
    ts_feat_base = ts_feat_base.dropna()

    # Join climate
    ts_feat_clim = ts_feat_base.join(climate_df, how='inner').dropna()

    def run_xgb(df, features, label):
        X = df[features]; y = df['Modal_Price']
        split = int(len(df) * 0.8)
        model = XGBRegressor(n_estimators=200, learning_rate=0.08, max_depth=4, random_state=42)
        model.fit(X[:split], y[:split], eval_set=[(X[split:], y[split:])], verbose=False)
        pred = model.predict(X[split:])
        mae  = mean_absolute_error(y[split:], pred)
        rmse = np.sqrt(mean_squared_error(y[split:], pred))
        mape = np.mean(np.abs((y[split:].values - pred) / y[split:].values)) * 100
        return {'Model': label, 'MAE (₹)': round(mae,2), 'RMSE (₹)': round(rmse,2), 'MAPE (%)': round(mape,2)}, pred, y[split:]

    BASE_FEATS   = ['lag_1','lag_7','rolling_mean_7','rolling_std_7']
    CLIMATE_FEATS = BASE_FEATS + ['Temperature','Rainfall','Humidity']

    with st.spinner("Training both models…"):
        res_base,   pred_base,   y_base   = run_xgb(ts_feat_base, BASE_FEATS,    'XGBoost (No Climate)')
        res_climate, pred_climate, y_clim = run_xgb(ts_feat_clim, CLIMATE_FEATS, 'XGBoost + Climate')

    cmp_df = pd.DataFrame([res_base, res_climate]).set_index('Model')
    st.dataframe(cmp_df.style
                 .highlight_min(axis=0, color='#dcfce7')
                 .format({'MAE (₹)':'₹{:.2f}','RMSE (₹)':'₹{:.2f}','MAPE (%)':'{:.2f}%'}),
                 use_container_width=True)

    # Improvement
    mae_imp  = res_base['MAE (₹)']  - res_climate['MAE (₹)']
    rmse_imp = res_base['RMSE (₹)'] - res_climate['RMSE (₹)']
    if rmse_imp > 0:
        callout(f"✅ Adding climate features improved RMSE by ₹{rmse_imp:.1f} and MAE by ₹{mae_imp:.1f}", "green")
    else:
        callout(f"ℹ️ Climate features did not improve accuracy on this dataset — consistent with simulated weather data. Real IMD data would likely show stronger correlation.", "blue")

    # Comparison chart
    fig4, ax4 = plt.subplots(figsize=(12, 4.5))
    ax4.plot(y_base.values,    color=PAL['navy'],  linewidth=2,   label='Actual')
    ax4.plot(pred_base,        color=PAL['amber'],  linewidth=1.8, linestyle='--', label='XGBoost (No Climate)', alpha=0.85)
    ax4.plot(pred_climate,     color=PAL['teal'],   linewidth=1.8, linestyle='--', label='XGBoost + Climate',    alpha=0.85)
    ax4.set_title("XGBoost: With vs Without Climate Features", fontsize=13, fontweight='600', color='#1c1f2e')
    ax4.set_ylabel("Price (₹/quintal)"); ax4.legend(fontsize=9); ax4.grid(True, alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig4); plt.close()

    # ── Disclaimer ──
    st.markdown("")
    callout("📌 <b>Note:</b> Climate data shown here is synthetically generated based on real IMD seasonal patterns for Nashik district (temperature ranges, monsoon rainfall distribution, humidity profiles). Integration with live IMD/OpenWeatherMap API data is identified as future scope.", "blue")
