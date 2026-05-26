import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')
 
# ══════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="⚡ ElectroSense Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# ══════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;500;600&display=swap');
 
html, body, [class*="css"] { font-family: 'Exo 2', sans-serif; }
 
/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080c14 0%, #0d1525 50%, #080c14 100%);
    border-right: 1px solid #1a2a4a;
}
[data-testid="stSidebar"] * { color: #a0b4d0 !important; }
 
/* ── Main background ── */
.stApp { background: #070b12; }
.main .block-container { padding: 2rem 2.5rem; }
 
/* ── Header glow ── */
h1 {
    font-family: 'Orbitron', monospace !important;
    color: #00d4ff !important;
    text-shadow: 0 0 30px rgba(0,212,255,0.5), 0 0 60px rgba(0,212,255,0.2);
    letter-spacing: 2px;
}
 
/* ── Metric card ── */
.mcard {
    background: linear-gradient(135deg, #0d1525 0%, #111e35 100%);
    border: 1px solid #1a3a5c;
    border-top: 2px solid #00d4ff;
    border-radius: 12px;
    padding: 20px 16px;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-bottom: 8px;
}
.mcard::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, #00d4ff, transparent);
}
.mcard-label {
    font-size: 10px;
    color: #4a7a9b;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-family: 'Orbitron', monospace;
    margin-bottom: 8px;
}
.mcard-value {
    font-size: 26px;
    font-weight: 700;
    color: #00d4ff;
    font-family: 'Orbitron', monospace;
    text-shadow: 0 0 20px rgba(0,212,255,0.4);
}
.mcard-value.green { color: #00ff9f; text-shadow: 0 0 20px rgba(0,255,159,0.4); }
.mcard-value.orange { color: #ff9f00; text-shadow: 0 0 20px rgba(255,159,0,0.4); }
.mcard-value.red { color: #ff4060; text-shadow: 0 0 20px rgba(255,64,96,0.4); }
.mcard-sub {
    font-size: 11px;
    color: #3a5a7a;
    margin-top: 4px;
}
 
/* ── Section title ── */
.stitle {
    font-family: 'Orbitron', monospace;
    font-size: 14px;
    color: #00d4ff;
    text-transform: uppercase;
    letter-spacing: 3px;
    border-bottom: 1px solid #1a3a5c;
    padding-bottom: 10px;
    margin: 28px 0 16px 0;
}
 
/* ── Insight box ── */
.insight {
    background: linear-gradient(135deg, #0a1520, #0d1e30);
    border: 1px solid #1a3a5c;
    border-left: 3px solid #00d4ff;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 10px 0;
    font-size: 13px;
    color: #7090b0;
    line-height: 1.6;
}
.insight strong { color: #00d4ff; }
 
/* ── Best model box ── */
.champion {
    background: linear-gradient(135deg, #061a0a, #0a2510);
    border: 1px solid #00ff9f44;
    border-radius: 16px;
    padding: 30px;
    text-align: center;
    margin-top: 24px;
    position: relative;
    overflow: hidden;
}
.champion-label {
    font-family: 'Orbitron', monospace;
    font-size: 11px;
    color: #00a060;
    letter-spacing: 3px;
    text-transform: uppercase;
}
.champion-name {
    font-family: 'Orbitron', monospace;
    font-size: 36px;
    font-weight: 900;
    color: #00ff9f;
    text-shadow: 0 0 40px rgba(0,255,159,0.5);
    margin: 10px 0;
}
.champion-stats { color: #00a060; font-size: 14px; }
 
/* ── Anomaly box ── */
.abox {
    background: linear-gradient(135deg, #1a0608, #25080c);
    border: 1px solid #ff406044;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
}
.abox-count {
    font-family: 'Orbitron', monospace;
    font-size: 52px;
    font-weight: 900;
    color: #ff4060;
    text-shadow: 0 0 30px rgba(255,64,96,0.6);
}
.abox-label { font-size: 11px; color: #804050; text-transform: uppercase; letter-spacing: 2px; }
 
/* ── Guide card ── */
.guide-card {
    background: #0d1525;
    border: 1px solid #1a3a5c;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 6px 0;
    display: flex;
    align-items: center;
    gap: 12px;
}
.guide-icon { font-size: 20px; }
.guide-text { font-size: 13px; color: #5a7a9a; }
.guide-text strong { color: #a0c0e0; }
 
/* ── Tab styling ── */
.stTabs [data-baseweb="tab"] {
    font-family: 'Orbitron', monospace;
    font-size: 11px;
    letter-spacing: 1.5px;
    color: #4a7a9b;
}
.stTabs [aria-selected="true"] { color: #00d4ff !important; }
 
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)
 
# ══════════════════════════════════════════════════════
# MATPLOTLIB THEME
# ══════════════════════════════════════════════════════
plt.rcParams.update({
    'figure.facecolor':  '#0d1525',
    'axes.facecolor':    '#0a1020',
    'axes.edgecolor':    '#1a3a5c',
    'axes.labelcolor':   '#5a8ab0',
    'xtick.color':       '#3a6080',
    'ytick.color':       '#3a6080',
    'text.color':        '#c0d8f0',
    'grid.color':        '#1a3050',
    'grid.linewidth':    0.5,
    'legend.facecolor':  '#0d1525',
    'legend.edgecolor':  '#1a3a5c',
    'legend.labelcolor': '#a0c0e0',
    'font.size':         10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})
 
BLUE   = '#00d4ff'
GREEN  = '#00ff9f'
ORANGE = '#ff9f00'
RED    = '#ff4060'
PURPLE = '#c060ff'
YELLOW = '#ffe040'
 
# ══════════════════════════════════════════════════════
# DATA LOADING & CACHING
# ══════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv('data/final_electricity_dataset.csv', index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df
 
@st.cache_data
def prepare_features(_df):
    data = _df.copy()
    data['hour']            = data.index.hour
    data['day']             = data.index.day
    data['month']           = data.index.month
    data['weekday']         = data.index.weekday
    data['lag_1']           = data['Global_active_power'].shift(1)
    data['lag_24']          = data['Global_active_power'].shift(24)
    data['rolling_mean_24'] = data['Global_active_power'].rolling(24).mean()
    data['rolling_std_24']  = data['Global_active_power'].rolling(24).std()
    data.dropna(inplace=True)
    return data
 
@st.cache_resource
def train_models(_df):
    feature_cols = ['hour','day','month','weekday','lag_1','lag_24','rolling_mean_24','rolling_std_24']
    X = _df[feature_cols]
    y = _df['Global_active_power']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
 
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
 
    try:
        from xgboost import XGBRegressor
        xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, verbosity=0)
        xgb.fit(X_train, y_train)
        xgb_pred     = xgb.predict(X_test)
        xgb_ok       = True
        xgb_importances = xgb.feature_importances_
    except ImportError:
        xgb_pred        = rf_pred.copy()
        xgb_ok          = False
        xgb_importances = rf.feature_importances_
 
    rf_importances = rf.feature_importances_
 
    def calc(actual, predicted):
        mae  = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2   = r2_score(actual, predicted)
        mask = actual != 0
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        return round(mae,4), round(rmse,4), round(r2,4), round(mape,2)
 
    return dict(
        feature_cols    = feature_cols,
        X_test          = X_test,
        y_test          = y_test,
        rf_pred         = rf_pred,
        xgb_pred        = xgb_pred,
        rf_metrics      = calc(y_test, rf_pred),
        xgb_metrics     = calc(y_test, xgb_pred),
        xgb_ok          = xgb_ok,
        rf_importances  = rf_importances,
        xgb_importances = xgb_importances,
    )
 
# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:20px 0 10px;'>
        <div style='font-family:Orbitron,monospace;font-size:22px;color:#00d4ff;
                    text-shadow:0 0 20px rgba(0,212,255,0.5);'>⚡ ELECTRO</div>
        <div style='font-family:Orbitron,monospace;font-size:22px;color:#00ff9f;
                    text-shadow:0 0 20px rgba(0,255,159,0.5);letter-spacing:4px;'>SENSE</div>
        <div style='font-size:10px;color:#2a4a6a;letter-spacing:2px;margin-top:6px;'>
            HOUSEHOLD POWER ANALYTICS
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
 
    page = st.radio("", [
        "🏠  Overview",
        "📊  EDA",
        "🔮  Forecasting",
        "🚨  Anomaly Detection",
        "📋  Model Comparison",
        "🧠  Feature Insights",
    ], label_visibility="collapsed")
 
    st.markdown("---")
    st.markdown("""
    <div style='font-size:10px;color:#1a3a5a;text-align:center;line-height:1.8;'>
        UCI Household Power Dataset<br>
        4 Years · Hourly Resolution<br>
        ML + Anomaly Detection
    </div>
    """, unsafe_allow_html=True)
 
# ══════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════
try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ File not found: `data/final_electricity_dataset.csv`  \nMake sure it's in the `data/` folder.")
    st.stop()
 
df_feat = prepare_features(df)
GAP     = 'Global_active_power'
 
# ══════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════
if page == "🏠  Overview":
    st.markdown("<h1>⚡ Smart Electricity Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#2a5a7a;font-size:14px;margin-top:-10px;margin-bottom:30px;'>UCI Individual Household Electric Power Consumption · Machine Learning & Anomaly Detection</p>", unsafe_allow_html=True)
 
    # KPI row
    c1,c2,c3,c4,c5 = st.columns(5)
    stats = [
        ("Total Records",    f"{len(df):,}",                                         "blue",   "hourly observations"),
        ("Date From",        df.index.min().strftime("%b %Y"),                        "green",  df.index.min().strftime("%Y-%m-%d")),
        ("Date To",          df.index.max().strftime("%b %Y"),                        "green",  df.index.max().strftime("%Y-%m-%d")),
        ("Avg Power",        f"{df[GAP].mean():.3f} kW",                             "orange", "global active power"),
        ("Peak Power",       f"{df[GAP].max():.2f} kW",                              "red",    "maximum recorded"),
    ]
    for col, (label, val, clr, sub) in zip([c1,c2,c3,c4,c5], stats):
        with col:
            st.markdown(f"""
            <div class="mcard">
                <div class="mcard-label">{label}</div>
                <div class="mcard-value {clr}" style="font-size:20px;">{val}</div>
                <div class="mcard-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # Mini trend + data preview
    col_a, col_b = st.columns([2,1])
    with col_a:
        st.markdown('<div class="stitle">Live Consumption Trend</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10,3))
        daily = df[GAP].resample('D').mean()
        ax.fill_between(daily.index, daily.values, alpha=0.15, color=BLUE)
        ax.plot(daily.index, daily.values, color=BLUE, linewidth=1.2)
        ax.plot(daily.index, daily.rolling(30).mean().values, color=ORANGE, linewidth=1.8, linestyle='--', label='30-day avg')
        ax.legend()
        ax.set_ylabel("Power (kW)")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
 
    with col_b:
        st.markdown('<div class="stitle">Quick Stats</div>', unsafe_allow_html=True)
        stat_df = df[[GAP]].describe().round(4)
        st.dataframe(stat_df, use_container_width=True)
 
    # Dataset preview
    st.markdown('<div class="stitle">Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(8), use_container_width=True)
 
    # Guide cards
    st.markdown('<div class="stitle">Navigation Guide</div>', unsafe_allow_html=True)
    guides = [
        ("📊","EDA","5 interactive charts — trends, hourly/monthly patterns, heatmap, rolling statistics"),
        ("🔮","Forecasting","Random Forest & XGBoost predictions with metrics, residuals, and error distribution"),
        ("🚨","Anomaly Detection","Z-Score & Isolation Forest — find unusual consumption with interactive thresholds"),
        ("📋","Model Comparison","Side-by-side metrics table and comparison charts for all models"),
        ("🧠","Feature Insights","Feature importance rankings and correlation deep-dive"),
    ]
    for icon, title, desc in guides:
        st.markdown(f"""
        <div class="guide-card">
            <span class="guide-icon">{icon}</span>
            <span class="guide-text"><strong>{title}</strong> — {desc}</span>
        </div>""", unsafe_allow_html=True)
 
# ══════════════════════════════════════════════════════
# PAGE 2 — EDA
# ══════════════════════════════════════════════════════
elif page == "📊  EDA":
    st.markdown("<h1>📊 Exploratory Data Analysis</h1>", unsafe_allow_html=True)
 
    # ── Year filter ──
    years = sorted(df.index.year.unique().tolist())
    sel_years = st.multiselect("Filter by Year", years, default=years)
    df_view = df[df.index.year.isin(sel_years)] if sel_years else df
 
    # ── Full Trend ──
    st.markdown('<div class="stitle">Electricity Consumption Over Time</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(14, 3.5))
    daily = df_view[GAP].resample('D').mean()
    ax.fill_between(daily.index, daily.values, alpha=0.12, color=BLUE)
    ax.plot(daily.index, daily.values, color=BLUE, linewidth=0.8, alpha=0.7, label='Daily Avg')
    ax.plot(daily.index, daily.rolling(30, min_periods=1).mean(), color=ORANGE, linewidth=2, label='30-day Rolling Mean')
    ax.axhline(daily.mean(), color=GREEN, linewidth=1, linestyle=':', label=f'Overall Mean ({daily.mean():.2f} kW)')
    ax.legend(fontsize=9)
    ax.set_ylabel("Power (kW)")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()
 
    # ── Monthly & Hourly ──
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="stitle">Monthly Average Consumption</div>', unsafe_allow_html=True)
        # FIX: use 'ME' instead of 'M'
        monthly = df_view[GAP].resample('ME').mean()
        month_labels = [d.strftime("%b %Y") for d in monthly.index]
        fig, ax = plt.subplots(figsize=(7, 4))
        bar_colors = [GREEN if v >= monthly.mean() else BLUE for v in monthly.values]
        bars = ax.bar(range(len(monthly)), monthly.values, color=bar_colors, width=0.7, alpha=0.85)
        ax.axhline(monthly.mean(), color=ORANGE, linewidth=1.5, linestyle='--', label=f'Mean: {monthly.mean():.2f}')
        ax.set_xticks(range(len(monthly)))
        ax.set_xticklabels(month_labels, rotation=75, fontsize=7)
        ax.set_ylabel("Avg Power (kW)")
        ax.legend(fontsize=9)
        ax.grid(True, axis='y', alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown(f'<div class="insight">📌 <strong>Peak month:</strong> {monthly.idxmax().strftime("%B %Y")} ({monthly.max():.2f} kW) &nbsp;|&nbsp; <strong>Lowest:</strong> {monthly.idxmin().strftime("%B %Y")} ({monthly.min():.2f} kW)</div>', unsafe_allow_html=True)
 
    with c2:
        st.markdown('<div class="stitle">Hourly Consumption Pattern</div>', unsafe_allow_html=True)
        hourly_avg = df_view.groupby(df_view.index.hour)[GAP].mean()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.fill_between(hourly_avg.index, hourly_avg.values, alpha=0.15, color=PURPLE)
        ax.plot(hourly_avg.index, hourly_avg.values, color=PURPLE, linewidth=2.5, marker='o', markersize=5)
        peak_h = hourly_avg.idxmax()
        ax.annotate(f'Peak\n{peak_h}:00', xy=(peak_h, hourly_avg[peak_h]),
                    xytext=(peak_h+1.5, hourly_avg[peak_h]+0.03),
                    arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.5),
                    color=ORANGE, fontsize=9)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Avg Power (kW)")
        ax.set_xticks(range(0,24))
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown(f'<div class="insight">📌 <strong>Peak hour:</strong> {peak_h}:00 ({hourly_avg[peak_h]:.2f} kW) &nbsp;|&nbsp; <strong>Quietest:</strong> {hourly_avg.idxmin()}:00 ({hourly_avg.min():.2f} kW)</div>', unsafe_allow_html=True)
 
    # ── Weekday & Heatmap ──
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="stitle">Weekday vs Weekend Pattern</div>', unsafe_allow_html=True)
        day_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        day_avg = df_view.groupby(df_view.index.weekday)[GAP].mean()
        clrs = [RED if i >= 5 else BLUE for i in range(7)]
        fig, ax = plt.subplots(figsize=(7,4))
        bars = ax.bar(day_names, day_avg.values, color=clrs, width=0.6, alpha=0.85)
        for bar, val in zip(bars, day_avg.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f'{val:.2f}', ha='center', fontsize=9, color='#c0d8f0')
        ax.set_ylabel("Avg Power (kW)")
        ax.grid(True, axis='y', alpha=0.2)
        legend_els = [mpatches.Patch(color=BLUE,label='Weekday'), mpatches.Patch(color=RED,label='Weekend')]
        ax.legend(handles=legend_els)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
 
    with c4:
        st.markdown('<div class="stitle">Correlation Heatmap</div>', unsafe_allow_html=True)
        corr_cols = [c for c in ['Global_active_power','Global_reactive_power','Voltage','Global_intensity'] if c in df.columns]
        fig, ax = plt.subplots(figsize=(7,4))
        mask = np.triu(np.ones_like(df[corr_cols].corr(), dtype=bool))
        sns.heatmap(df[corr_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm',
                    ax=ax, linewidths=0.5, linecolor='#1a3050',
                    annot_kws={'size':12,'weight':'bold'}, mask=mask)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
 
    # ── Seasonal Heatmap (Hour x Month) ──
    st.markdown('<div class="stitle">Seasonal Heatmap — Hour of Day × Month</div>', unsafe_allow_html=True)
    pivot = df_view.groupby([df_view.index.month, df_view.index.hour])[GAP].mean().unstack()
    pivot.index = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][:len(pivot)]
    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(pivot, cmap='YlOrRd', ax=ax, linewidths=0, cbar_kws={'label':'Avg kW'},
                annot=False)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Month")
    ax.set_title("Average Power Consumption by Month and Hour", color='#a0c0e0', fontsize=11)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.markdown('<div class="insight">📌 This heatmap reveals <strong>seasonal patterns</strong> — darker cells = higher consumption. Look for winter mornings and summer evenings as typical peak zones.</div>', unsafe_allow_html=True)
 
# ══════════════════════════════════════════════════════
# PAGE 3 — FORECASTING
# ══════════════════════════════════════════════════════
elif page == "🔮  Forecasting":
    st.markdown("<h1>🔮 Electricity Forecasting</h1>", unsafe_allow_html=True)
 
    with st.spinner("Training models... ⚙️"):
        res = train_models(df_feat)
 
    col_sel, col_pts = st.columns([2,2])
    with col_sel:
        model_choice = st.selectbox("Select Forecasting Model", ["Random Forest", "XGBoost"])
    with col_pts:
        n = st.slider("Data points to display", 100, 600, 250, 50)
 
    if model_choice == "Random Forest":
        pred = res['rf_pred']
        mae, rmse, r2, mape = res['rf_metrics']
        color = BLUE
    else:
        pred = res['xgb_pred']
        mae, rmse, r2, mape = res['xgb_metrics']
        color = ORANGE
        if not res['xgb_ok']:
            st.warning("XGBoost not installed — showing Random Forest. Run: `pip install xgboost`")
 
    y_test = res['y_test']
 
    # Metrics
    c1,c2,c3,c4 = st.columns(4)
    for col, (label, val, clr) in zip([c1,c2,c3,c4],[
        ("MAE",      mae,        "blue"),
        ("RMSE",     rmse,       "orange"),
        ("MAPE",     f"{mape}%", "green"),
        ("R² Score", r2,         "green" if r2 > 0.9 else "orange"),
    ]):
        with col:
            st.markdown(f"""<div class="mcard">
                <div class="mcard-label">{label}</div>
                <div class="mcard-value {clr}">{val}</div>
            </div>""", unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    y_actual = y_test.values[:n]
    y_pred   = pred[:n]
 
    # Actual vs Predicted
    st.markdown(f'<div class="stitle">{model_choice} — Actual vs Predicted</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(range(n), y_actual, color='#c0d8f0', linewidth=1.4, label='Actual', alpha=0.9)
    ax.plot(range(n), y_pred, color=color, linewidth=1.4, label='Predicted', linestyle='--', alpha=0.9)
    ax.fill_between(range(n), np.minimum(y_actual, y_pred), np.maximum(y_actual, y_pred),
                    alpha=0.1, color=color, label='Error band')
    ax.legend(fontsize=10)
    ax.set_xlabel("Time Steps (hours)")
    ax.set_ylabel("Global Active Power (kW)")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()
 
    # Residuals + Error Distribution side by side
    c1, c2 = st.columns([2,1])
    with c1:
        st.markdown('<div class="stitle">Prediction Residuals</div>', unsafe_allow_html=True)
        residuals = y_actual - y_pred
        fig, ax = plt.subplots(figsize=(10, 3))
        bar_colors = [GREEN if r >= 0 else RED for r in residuals]
        ax.bar(range(n), residuals, color=bar_colors, width=1.0, alpha=0.8)
        ax.axhline(0, color=YELLOW, linewidth=1.2)
        ax.axhline(residuals.mean(), color=ORANGE, linewidth=1, linestyle='--', label=f'Mean residual: {residuals.mean():.4f}')
        ax.legend(fontsize=9)
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Residual (kW)")
        ax.grid(True, axis='y', alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
 
    with c2:
        st.markdown('<div class="stitle">Error Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(residuals, bins=40, color=color, alpha=0.7, edgecolor='none')
        ax.axvline(0, color=YELLOW, linewidth=1.5)
        ax.axvline(residuals.mean(), color=ORANGE, linewidth=1.5, linestyle='--', label=f'Mean: {residuals.mean():.4f}')
        ax.set_xlabel("Residual")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
 
    # Scatter actual vs predicted
    st.markdown('<div class="stitle">Actual vs Predicted Scatter</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_actual, y_pred, color=color, alpha=0.3, s=8)
    lim = [min(y_actual.min(), y_pred.min()), max(y_actual.max(), y_pred.max())]
    ax.plot(lim, lim, color=YELLOW, linewidth=1.5, linestyle='--', label='Perfect fit')
    ax.set_xlabel("Actual (kW)")
    ax.set_ylabel("Predicted (kW)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()
 
# ══════════════════════════════════════════════════════
# PAGE 4 — ANOMALY DETECTION
# ══════════════════════════════════════════════════════
elif page == "🚨  Anomaly Detection":
    st.markdown("<h1>🚨 Anomaly Detection</h1>", unsafe_allow_html=True)
 
    series = df[GAP].copy().dropna()
 
    tab1, tab2, tab3 = st.tabs(["⚡  Z-Score Method", "🌲  Isolation Forest", "📈  Rolling Threshold"])
 
    # ── Z-Score ──
    with tab1:
        threshold = st.slider("Z-Score Threshold", 2.0, 4.0, 3.0, 0.1, key='zs')
        z_scores  = zscore(series)
        z_ser     = pd.Series(z_scores, index=series.index)
        anomalies = series[abs(z_ser) > threshold]
        pct       = round(len(anomalies)/len(series)*100, 2)
 
        c1,c2,c3 = st.columns([1,1,2])
        with c1:
            st.markdown(f'<div class="abox"><div class="abox-label">Anomalies</div><div class="abox-count">{len(anomalies)}</div><div class="abox-label">detected</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="mcard"><div class="mcard-label">Anomaly Rate</div><div class="mcard-value red">{pct}%</div><div class="mcard-sub">of all records</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="insight" style="height:100%;display:flex;align-items:center;"><span>Z-Score flags data points that are <strong>more than {threshold}σ</strong> away from the mean. A threshold of 3 covers 99.7% of normal data under a Gaussian distribution.</span></div>', unsafe_allow_html=True)
 
        st.markdown("<br>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(14, 4.5))
        ax.plot(series.index, series.values, color=BLUE, linewidth=0.5, alpha=0.6, label='Normal')
        ax.scatter(anomalies.index, anomalies.values, color=RED, s=15, zorder=5, label=f'Anomaly ({len(anomalies)})', alpha=0.9)
        ax.set_ylabel("Power (kW)")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
 
        st.markdown('<div class="stitle">Top 10 Anomalies</div>', unsafe_allow_html=True)
        top10 = anomalies.abs().nlargest(10).reset_index()
        top10.columns = ['Datetime','Global Active Power (kW)']
        top10['Z-Score'] = top10['Global Active Power (kW)'].apply(lambda x: round(abs((x - series.mean())/series.std()), 2))
        st.dataframe(top10, use_container_width=True)
 
    # ── Isolation Forest ──
    with tab2:
        contamination = st.slider("Contamination rate", 0.01, 0.10, 0.02, 0.01, key='iso')
        iso     = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        iso_lbl = iso.fit_predict(series.values.reshape(-1,1))
        iso_an  = series[iso_lbl == -1]
        pct_iso = round(len(iso_an)/len(series)*100, 2)
 
        c1,c2,c3 = st.columns([1,1,2])
        with c1:
            st.markdown(f'<div class="abox"><div class="abox-label">Anomalies</div><div class="abox-count">{len(iso_an)}</div><div class="abox-label">detected</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="mcard"><div class="mcard-label">Anomaly Rate</div><div class="mcard-value red">{pct_iso}%</div><div class="mcard-sub">of all records</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="insight" style="height:100%;display:flex;align-items:center;"><span>Isolation Forest uses <strong>random partitioning</strong> — anomalies are isolated with fewer splits. No Gaussian assumption needed. Great for catching non-obvious outliers.</span></div>', unsafe_allow_html=True)
 
        st.markdown("<br>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(14, 4.5))
        ax.plot(series.index, series.values, color=BLUE, linewidth=0.5, alpha=0.6, label='Normal')
        ax.scatter(iso_an.index, iso_an.values, color=RED, s=15, zorder=5, label=f'Anomaly ({len(iso_an)})', alpha=0.9)
        ax.set_ylabel("Power (kW)")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
 
        st.markdown('<div class="stitle">Top 10 Isolation Forest Anomalies</div>', unsafe_allow_html=True)
        top10_iso = iso_an.abs().nlargest(10).reset_index()
        top10_iso.columns = ['Datetime','Global Active Power (kW)']
        st.dataframe(top10_iso, use_container_width=True)
 
    # ── Rolling Threshold ──
    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            window    = st.slider("Rolling window (hours)", 6, 72, 24, 6, key='rw')
        with c2:
            multiplier = st.slider("Std multiplier for threshold", 1.0, 4.0, 2.0, 0.5, key='rm')
 
        roll_mean = series.rolling(window, min_periods=1).mean()
        roll_std  = series.rolling(window, min_periods=1).std().fillna(0)
        upper     = roll_mean + multiplier * roll_std
        lower     = roll_mean - multiplier * roll_std
        roll_an   = series[(series > upper) | (series < lower)]
        pct_roll  = round(len(roll_an)/len(series)*100, 2)
 
        c1,c2,c3 = st.columns([1,1,2])
        with c1:
            st.markdown(f'<div class="abox"><div class="abox-label">Anomalies</div><div class="abox-count">{len(roll_an)}</div><div class="abox-label">detected</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="mcard"><div class="mcard-label">Anomaly Rate</div><div class="mcard-value red">{pct_roll}%</div><div class="mcard-sub">of all records</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="insight" style="height:100%;display:flex;align-items:center;"><span>Rolling threshold detects <strong>sudden spikes/drops</strong> relative to recent behavior using a {window}-hour window and {multiplier}σ bands. Great for real-time monitoring.</span></div>', unsafe_allow_html=True)
 
        st.markdown("<br>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(14, 4.5))
        ax.plot(series.index, series.values, color=BLUE, linewidth=0.5, alpha=0.5, label='Consumption')
        ax.plot(roll_mean.index, roll_mean.values, color=GREEN, linewidth=1, alpha=0.7, label='Rolling Mean')
        ax.fill_between(upper.index, lower.values, upper.values, alpha=0.1, color=GREEN, label='Normal band')
        ax.scatter(roll_an.index, roll_an.values, color=RED, s=15, zorder=5, label=f'Anomaly ({len(roll_an)})')
        ax.set_ylabel("Power (kW)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
 
# ══════════════════════════════════════════════════════
# PAGE 5 — MODEL COMPARISON
# ══════════════════════════════════════════════════════
elif page == "📋  Model Comparison":
    st.markdown("<h1>📋 Model Comparison</h1>", unsafe_allow_html=True)
 
    with st.spinner("Computing metrics... ⚙️"):
        res = train_models(df_feat)
 
    rf_mae,  rf_rmse,  rf_r2,  rf_mape  = res['rf_metrics']
    xgb_mae, xgb_rmse, xgb_r2, xgb_mape = res['xgb_metrics']
 
    comp = pd.DataFrame({
        'Model':    ['Random Forest', 'XGBoost'],
        'MAE':      [rf_mae,  xgb_mae],
        'RMSE':     [rf_rmse, xgb_rmse],
        'MAPE (%)': [rf_mape, xgb_mape],
        'R² Score': [rf_r2,   xgb_r2],
    })
 
    # Summary cards
    best_r2_idx  = comp['R² Score'].idxmax()
    best_mae_idx = comp['MAE'].idxmin()
    c1,c2,c3,c4 = st.columns(4)
    for col, (label, val, clr) in zip([c1,c2,c3,c4],[
        ("Best R²",    f"{comp['R² Score'].max():.4f}",   "green"),
        ("Best MAE",   f"{comp['MAE'].min():.4f}",        "blue"),
        ("Best RMSE",  f"{comp['RMSE'].min():.4f}",       "blue"),
        ("Best MAPE",  f"{comp['MAPE (%)'].min():.2f}%",  "orange"),
    ]):
        with col:
            st.markdown(f'<div class="mcard"><div class="mcard-label">{label}</div><div class="mcard-value {clr}">{val}</div></div>', unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="stitle">Full Metrics Table</div>', unsafe_allow_html=True)
    st.dataframe(comp.set_index('Model').style.highlight_max(subset=['R² Score'], color='#0d2a1a').highlight_min(subset=['MAE','RMSE','MAPE (%)'], color='#0d1a2a'), use_container_width=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="stitle">R² Score (Higher = Better)</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6,4))
        clrs = [GREEN if i == comp['R² Score'].idxmax() else BLUE for i in range(len(comp))]
        bars = ax.bar(comp['Model'], comp['R² Score'], color=clrs, width=0.5, alpha=0.85)
        for bar, val in zip(bars, comp['R² Score']):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                    f'{val:.4f}', ha='center', fontsize=12, color='#c0d8f0', fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("R² Score")
        ax.grid(True, axis='y', alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
 
    with c2:
        st.markdown('<div class="stitle">MAE — Lower is Better</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6,4))
        clrs2 = [GREEN if i == comp['MAE'].idxmin() else ORANGE for i in range(len(comp))]
        bars = ax.bar(comp['Model'], comp['MAE'], color=clrs2, width=0.5, alpha=0.85)
        for bar, val in zip(bars, comp['MAE']):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0003,
                    f'{val:.4f}', ha='center', fontsize=12, color='#c0d8f0', fontweight='bold')
        ax.set_ylabel("MAE")
        ax.grid(True, axis='y', alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
 
    # Radar-style RMSE & MAPE
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="stitle">RMSE Comparison</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6,3))
        bars = ax.barh(comp['Model'], comp['RMSE'], color=[BLUE, ORANGE], height=0.45, alpha=0.85)
        for bar, val in zip(bars, comp['RMSE']):
            ax.text(bar.get_width()+0.0002, bar.get_y()+bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=11, color='#c0d8f0')
        ax.set_xlabel("RMSE")
        ax.grid(True, axis='x', alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
 
    with c4:
        st.markdown('<div class="stitle">MAPE % Comparison</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6,3))
        bars = ax.barh(comp['Model'], comp['MAPE (%)'], color=[BLUE, ORANGE], height=0.45, alpha=0.85)
        for bar, val in zip(bars, comp['MAPE (%)']):
            ax.text(bar.get_width()+0.05, bar.get_y()+bar.get_height()/2,
                    f'{val:.2f}%', va='center', fontsize=11, color='#c0d8f0')
        ax.set_xlabel("MAPE (%)")
        ax.grid(True, axis='x', alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
 
    # Champion box
    best_model = comp.loc[best_r2_idx, 'Model']
    best_r2_v  = comp.loc[best_r2_idx, 'R² Score']
    best_mae_v = comp.loc[best_mae_idx, 'MAE']
    st.markdown(f"""
    <div class="champion">
        <div class="champion-label">🏆 Best Performing Model</div>
        <div class="champion-name">{best_model}</div>
        <div class="champion-stats">
            R² Score: <strong>{best_r2_v}</strong> &nbsp;·&nbsp;
            MAE: <strong>{best_mae_v}</strong> &nbsp;·&nbsp;
            MAPE: <strong>{comp.loc[best_r2_idx,'MAPE (%)']:.2f}%</strong>
        </div>
    </div>""", unsafe_allow_html=True)
 
# ══════════════════════════════════════════════════════
# PAGE 6 — FEATURE INSIGHTS
# ══════════════════════════════════════════════════════
elif page == "🧠  Feature Insights":
    st.markdown("<h1>🧠 Feature Insights</h1>", unsafe_allow_html=True)
 
    with st.spinner("Loading feature analysis... ⚙️"):
        res = train_models(df_feat)
 
    feature_names = res['feature_cols']
    rf_imp  = res['rf_importances']
    xgb_imp = res['xgb_importances']
 
    # Feature importance comparison
    st.markdown('<div class="stitle">Feature Importance — Random Forest vs XGBoost</div>', unsafe_allow_html=True)
    imp_df = pd.DataFrame({
        'Feature':        feature_names,
        'Random Forest':  rf_imp,
        'XGBoost':        xgb_imp,
    }).sort_values('Random Forest', ascending=True)
 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (col, clr) in zip(axes, [('Random Forest', BLUE), ('XGBoost', ORANGE)]):
        vals = imp_df[col]
        bars = ax.barh(imp_df['Feature'], vals, color=clr, alpha=0.8, height=0.6)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=9, color='#a0c0e0')
        ax.set_xlabel("Importance Score")
        ax.set_title(col, color=clr, fontsize=12, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()
 
    st.markdown('<div class="insight">📌 <strong>lag_1</strong> (previous hour) and <strong>rolling_mean_24</strong> (24-hour average) typically dominate because electricity usage is highly auto-correlated — what happened recently strongly predicts what happens next.</div>', unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # Lag analysis
    st.markdown('<div class="stitle">Autocorrelation — How much does past affect future?</div>', unsafe_allow_html=True)
    lags  = range(1, 49)
    acorr = [df[GAP].autocorr(lag=l) for l in lags]
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.bar(lags, acorr, color=[BLUE if v > 0 else RED for v in acorr], alpha=0.7, width=0.8)
    ax.axhline(0, color=YELLOW, linewidth=1)
    ax.axhline(0.05, color=GREEN, linewidth=1, linestyle='--', alpha=0.5, label='5% threshold')
    ax.axhline(-0.05, color=GREEN, linewidth=1, linestyle='--', alpha=0.5)
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Autocorrelation")
    ax.set_xticks(list(lags))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.markdown('<div class="insight">📌 Strong autocorrelation at lag 1, 24 (same hour yesterday), and 48 (same hour two days ago) confirms that time-based lag features are the most powerful predictors for this dataset.</div>', unsafe_allow_html=True)
 
    # Distribution of target
    st.markdown('<div class="stitle">Target Variable Distribution</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.hist(df[GAP].dropna(), bins=80, color=BLUE, alpha=0.75, edgecolor='none')
        ax.axvline(df[GAP].mean(), color=ORANGE, linewidth=2, linestyle='--', label=f'Mean: {df[GAP].mean():.3f}')
        ax.axvline(df[GAP].median(), color=GREEN, linewidth=2, linestyle='--', label=f'Median: {df[GAP].median():.3f}')
        ax.set_xlabel("Global Active Power (kW)")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
 
    with c2:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        bp = ax.boxplot(df[GAP].dropna(), vert=False, patch_artist=True,
                        boxprops=dict(facecolor='#0d2040', color=BLUE),
                        medianprops=dict(color=ORANGE, linewidth=2),
                        whiskerprops=dict(color=BLUE),
                        capprops=dict(color=BLUE),
                        flierprops=dict(marker='o', color=RED, markersize=2, alpha=0.3))
        ax.set_xlabel("Global Active Power (kW)")
        ax.set_yticks([])
        ax.grid(True, axis='x', alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()