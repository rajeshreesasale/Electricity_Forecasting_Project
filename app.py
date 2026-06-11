import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="⚡ ElectroSense Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Exo 2', sans-serif; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080c14 0%, #0d1525 50%, #080c14 100%);
    border-right: 1px solid #1a2a4a;
}
[data-testid="stSidebar"] * { color: #a0b4d0 !important; }
.stApp { background: #070b12; }
.main .block-container { padding: 2rem 2.5rem; }
h1 {
    font-family: 'Orbitron', monospace !important;
    color: #00d4ff !important;
    text-shadow: 0 0 30px rgba(0,212,255,0.5), 0 0 60px rgba(0,212,255,0.2);
    letter-spacing: 2px;
}
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
.mcard-sub { font-size: 11px; color: #3a5a7a; margin-top: 4px; }
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
.champion {
    background: linear-gradient(135deg, #061a0a, #0a2510);
    border: 1px solid #00ff9f44;
    border-radius: 16px;
    padding: 30px;
    text-align: center;
    margin-top: 24px;
}
.champion-label { font-family: 'Orbitron', monospace; font-size: 11px; color: #00a060; letter-spacing: 3px; text-transform: uppercase; }
.champion-name { font-family: 'Orbitron', monospace; font-size: 36px; font-weight: 900; color: #00ff9f; text-shadow: 0 0 40px rgba(0,255,159,0.5); margin: 10px 0; }
.champion-stats { color: #00a060; font-size: 14px; }
.abox { background: linear-gradient(135deg, #1a0608, #25080c); border: 1px solid #ff406044; border-radius: 12px; padding: 24px; text-align: center; }
.abox-count { font-family: 'Orbitron', monospace; font-size: 52px; font-weight: 900; color: #ff4060; text-shadow: 0 0 30px rgba(255,64,96,0.6); }
.abox-label { font-size: 11px; color: #804050; text-transform: uppercase; letter-spacing: 2px; }
.guide-card { background: #0d1525; border: 1px solid #1a3a5c; border-radius: 10px; padding: 14px 18px; margin: 6px 0; }
.guide-text { font-size: 13px; color: #5a7a9a; }
.guide-text strong { color: #a0c0e0; }
.split-box {
    background: linear-gradient(135deg, #0a1520, #0d1e30);
    border: 1px solid #1a3a5c;
    border-left: 4px solid #00ff9f;
    border-radius: 8px;
    padding: 18px 22px;
    margin: 8px 0;
}
.split-box-title { font-family: 'Orbitron', monospace; font-size: 11px; color: #00ff9f; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 6px; }
.split-box-val { font-size: 16px; color: #c0d8f0; font-weight: 600; }
.split-box-sub { font-size: 12px; color: #3a6080; margin-top: 4px; }
.stTabs [data-baseweb="tab"] { font-family: 'Orbitron', monospace; font-size: 11px; letter-spacing: 1.5px; color: #4a7a9b; }
.stTabs [aria-selected="true"] { color: #00d4ff !important; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    'figure.facecolor': '#0d1525', 'axes.facecolor': '#0a1020',
    'axes.edgecolor': '#1a3a5c', 'axes.labelcolor': '#5a8ab0',
    'xtick.color': '#3a6080', 'ytick.color': '#3a6080',
    'text.color': '#c0d8f0', 'grid.color': '#1a3050',
    'grid.linewidth': 0.5, 'legend.facecolor': '#0d1525',
    'legend.edgecolor': '#1a3a5c', 'legend.labelcolor': '#a0c0e0',
    'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
})

BLUE = '#00d4ff'; GREEN = '#00ff9f'; ORANGE = '#ff9f00'
RED = '#ff4060'; PURPLE = '#c060ff'; YELLOW = '#ffe040'

@st.cache_data
def load_data():
    df = pd.read_csv('final_electricity_dataset.csv', index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()

def load_colab_results():
    import json
    with open('metrics.json', 'r') as f:
        m = json.load(f)
    rf_pred  = np.load('rf_pred.npy')
    xgb_pred = np.load('xgb_pred.npy')
    lr_pred  = np.load('lr_pred.npy')
    y_test   = np.load('y_test.npy')
    rf_imp   = np.load('rf_importance.npy')
    xgb_imp  = np.load('xgb_importance.npy')
    return dict(
        feature_cols = [
            'hour', 'day', 'month', 'weekday',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
            'hour_of_week',
            'lag_1', 'lag_2', 'lag_3',
            'lag_24', 'lag_48', 'lag_168',
            'rolling_mean_6h', 'rolling_mean_24h',
            'rolling_mean_72h', 'rolling_std_24h'
        ],
        y_test      = pd.Series(y_test),
        rf_pred     = rf_pred,
        xgb_pred    = xgb_pred,
        lr_pred     = lr_pred,
        rf_metrics  = (m['rf_mae'],  m['rf_rmse'],  m['rf_r2'],  m['rf_mape']),
        xgb_metrics = (m['xgb_mae'], m['xgb_rmse'], m['xgb_r2'], m['xgb_mape']),
        lr_metrics  = (m['lr_mae'],  m['lr_rmse'],  m['lr_r2'],  m['lr_mape']),
        xgb_ok      = True,
        split_idx   = m['split_index'],
        train_start = pd.Timestamp(m['train_start']),
        train_end   = pd.Timestamp(m['train_end']),
        test_start  = pd.Timestamp(m['test_start']),
        test_end    = pd.Timestamp(m['test_end']),
        rf_imp      = rf_imp,
        xgb_imp     = xgb_imp,
    )

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:20px 0 10px;'>
        <div style='font-family:Orbitron,monospace;font-size:22px;color:#00d4ff;
                    text-shadow:0 0 20px rgba(0,212,255,0.5);'>⚡ ELECTRO</div>
        <div style='font-family:Orbitron,monospace;font-size:22px;color:#00ff9f;
                    text-shadow:0 0 20px rgba(0,255,159,0.5);letter-spacing:4px;'>SENSE</div>
        <div style='font-size:10px;color:#2a4a6a;letter-spacing:2px;margin-top:6px;'>
            HOUSEHOLD POWER ANALYTICS</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("", [
        "🏠  Overview",
        "📊  EDA",
        "🔮  Forecasting",
        "🚨  Anomaly Detection",
        "📋  Model Comparison",
        "🧪  Model Testing & Evaluation",
        "🧠  Feature Insights",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("<div style='font-size:10px;color:#1a3a5a;text-align:center;line-height:1.8;'>UCI Household Power Dataset<br>4 Years · Hourly Resolution<br>ML + Anomaly Detection</div>", unsafe_allow_html=True)

try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ File not found: `final_electricity_dataset.csv`")
    st.stop()

try:
    res = load_colab_results()
except Exception as e:
    st.error(f"❌ Could not load model results: {e}")
    st.stop()

GAP = 'Global_active_power'

# ══════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════
if page == "🏠  Overview":
    st.markdown("<h1>⚡ Smart Electricity Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#2a5a7a;font-size:14px;margin-top:-10px;margin-bottom:30px;'>UCI Individual Household Electric Power Consumption · Machine Learning & Anomaly Detection</p>", unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    stats = [
        ("Total Records", f"{len(df):,}",                 "blue",   "hourly observations"),
        ("Date From",     df.index.min().strftime("%b %Y"),"green",  df.index.min().strftime("%Y-%m-%d")),
        ("Date To",       df.index.max().strftime("%b %Y"),"green",  df.index.max().strftime("%Y-%m-%d")),
        ("Avg Power",     f"{df[GAP].mean():.3f} kW",     "orange", "global active power"),
        ("Peak Power",    f"{df[GAP].max():.2f} kW",      "red",    "maximum recorded"),
    ]
    for col, (label, val, clr, sub) in zip([c1,c2,c3,c4,c5], stats):
        with col:
            st.markdown(f'<div class="mcard"><div class="mcard-label">{label}</div><div class="mcard-value {clr}" style="font-size:20px;">{val}</div><div class="mcard-sub">{sub}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns([2,1])
    with col_a:
        st.markdown('<div class="stitle">Consumption Trend</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10,3))
        daily = df[GAP].resample('D').mean()
        ax.fill_between(daily.index, daily.values, alpha=0.15, color=BLUE)
        ax.plot(daily.index, daily.values, color=BLUE, linewidth=1.2)
        ax.plot(daily.index, daily.rolling(30).mean().values, color=ORANGE, linewidth=1.8, linestyle='--', label='30-day avg')
        ax.legend(); ax.set_ylabel("Power (kW)"); ax.grid(True, alpha=0.25)
        fig.tight_layout(); st.pyplot(fig); plt.close()
    with col_b:
        st.markdown('<div class="stitle">Quick Stats</div>', unsafe_allow_html=True)
        st.dataframe(df[[GAP]].describe().round(4), use_container_width=True)

    st.markdown('<div class="stitle">Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(8), use_container_width=True)

    st.markdown('<div class="stitle">Navigation Guide</div>', unsafe_allow_html=True)
    guides = [
        ("📊","EDA","Consumption trends, patterns, seasonal heatmap, correlations"),
        ("🔮","Forecasting","Model predictions with actual vs predicted graphs and metrics"),
        ("🚨","Anomaly Detection","Z-Score, Isolation Forest and Rolling Threshold methods"),
        ("📋","Model Comparison","Side-by-side performance of all models"),
        ("🧪","Model Testing & Evaluation","Train-test split info, testing period, full evaluation"),
        ("🧠","Feature Insights","Feature importance, autocorrelation, target distribution"),
    ]
    for icon, title, desc in guides:
        st.markdown(f'<div class="guide-card"><span class="guide-text"><strong>{icon} {title}</strong> — {desc}</span></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# PAGE 2 — EDA
# ══════════════════════════════════════════════════════
elif page == "📊  EDA":
    st.markdown("<h1>📊 Exploratory Data Analysis</h1>", unsafe_allow_html=True)

    years = sorted(df.index.year.unique().tolist())
    sel_years = st.multiselect("Filter by Year", years, default=years)
    df_view = df[df.index.year.isin(sel_years)] if sel_years else df

    st.markdown('<div class="stitle">Electricity Consumption Over Time</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(14, 3.5))
    daily = df_view[GAP].resample('D').mean()
    ax.fill_between(daily.index, daily.values, alpha=0.12, color=BLUE)
    ax.plot(daily.index, daily.values, color=BLUE, linewidth=0.8, alpha=0.7, label='Daily Avg')
    ax.plot(daily.index, daily.rolling(30, min_periods=1).mean(), color=ORANGE, linewidth=2, label='30-day Rolling Mean')
    ax.axhline(daily.mean(), color=GREEN, linewidth=1, linestyle=':', label=f'Mean ({daily.mean():.2f} kW)')
    ax.legend(fontsize=9); ax.set_ylabel("Power (kW)"); ax.grid(True, alpha=0.2)
    fig.tight_layout(); st.pyplot(fig); plt.close()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="stitle">Monthly Average Consumption</div>', unsafe_allow_html=True)
        monthly = df_view[GAP].resample('ME').mean()
        month_labels = [d.strftime("%b %Y") for d in monthly.index]
        fig, ax = plt.subplots(figsize=(7, 4))
        bar_colors = [GREEN if v >= monthly.mean() else BLUE for v in monthly.values]
        ax.bar(range(len(monthly)), monthly.values, color=bar_colors, width=0.7, alpha=0.85)
        ax.axhline(monthly.mean(), color=ORANGE, linewidth=1.5, linestyle='--', label=f'Mean: {monthly.mean():.2f}')
        ax.set_xticks(range(len(monthly))); ax.set_xticklabels(month_labels, rotation=75, fontsize=7)
        ax.set_ylabel("Avg Power (kW)"); ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.2)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown(f'<div class="insight">📌 <strong>Peak:</strong> {monthly.idxmax().strftime("%B %Y")} ({monthly.max():.2f} kW) &nbsp;|&nbsp; <strong>Lowest:</strong> {monthly.idxmin().strftime("%B %Y")} ({monthly.min():.2f} kW)</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="stitle">Hourly Consumption Pattern</div>', unsafe_allow_html=True)
        hourly_avg = df_view.groupby(df_view.index.hour)[GAP].mean()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.fill_between(hourly_avg.index, hourly_avg.values, alpha=0.15, color=PURPLE)
        ax.plot(hourly_avg.index, hourly_avg.values, color=PURPLE, linewidth=2.5, marker='o', markersize=5)
        peak_h = hourly_avg.idxmax()
        ax.annotate(f'Peak\n{peak_h}:00', xy=(peak_h, hourly_avg[peak_h]),
                    xytext=(peak_h+1.5, hourly_avg[peak_h]+0.03),
                    arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.5), color=ORANGE, fontsize=9)
        ax.set_xlabel("Hour of Day"); ax.set_ylabel("Avg Power (kW)")
        ax.set_xticks(range(0,24)); ax.grid(True, alpha=0.2)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown(f'<div class="insight">📌 <strong>Peak hour:</strong> {peak_h}:00 ({hourly_avg[peak_h]:.2f} kW) &nbsp;|&nbsp; <strong>Quietest:</strong> {hourly_avg.idxmin()}:00 ({hourly_avg.min():.2f} kW)</div>', unsafe_allow_html=True)

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
        ax.set_ylabel("Avg Power (kW)"); ax.grid(True, axis='y', alpha=0.2)
        legend_els = [mpatches.Patch(color=BLUE,label='Weekday'), mpatches.Patch(color=RED,label='Weekend')]
        ax.legend(handles=legend_els); fig.tight_layout(); st.pyplot(fig); plt.close()

    with c4:
        st.markdown('<div class="stitle">Correlation Heatmap</div>', unsafe_allow_html=True)
        corr_cols = [c for c in ['Global_active_power','Global_reactive_power','Voltage','Global_intensity'] if c in df.columns]
        fig, ax = plt.subplots(figsize=(7,4))
        mask = np.triu(np.ones_like(df[corr_cols].corr(), dtype=bool))
        sns.heatmap(df[corr_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
                    linewidths=0.5, linecolor='#1a3050', annot_kws={'size':12,'weight':'bold'}, mask=mask)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="stitle">Seasonal Heatmap — Hour × Month</div>', unsafe_allow_html=True)
    pivot = df_view.groupby([df_view.index.month, df_view.index.hour])[GAP].mean().unstack()
    pivot.index = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][:len(pivot)]
    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(pivot, cmap='YlOrRd', ax=ax, linewidths=0, cbar_kws={'label':'Avg kW'})
    ax.set_xlabel("Hour of Day"); ax.set_ylabel("Month")
    fig.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown('<div class="insight">📌 Darker cells = higher consumption. Winter mornings and summer evenings are typical peak zones.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# PAGE 3 — FORECASTING
# ══════════════════════════════════════════════════════
elif page == "🔮  Forecasting":
    st.markdown("<h1>🔮 Electricity Forecasting</h1>", unsafe_allow_html=True)

    col_sel, col_pts = st.columns([2,2])
    with col_sel:
        model_choice = st.selectbox("Select Model", ["Random Forest", "XGBoost", "Linear Regression"])
    with col_pts:
        n = st.slider("Points to display", 100, 600, 250, 50)

    if model_choice == "Random Forest":
        pred = res['rf_pred']; mae,rmse,r2,mape = res['rf_metrics']; color = BLUE
    elif model_choice == "XGBoost":
        pred = res['xgb_pred']; mae,rmse,r2,mape = res['xgb_metrics']; color = ORANGE
    else:
        pred = res['lr_pred']; mae,rmse,r2,mape = res['lr_metrics']; color = PURPLE

    y_test = res['y_test']
    c1,c2,c3,c4 = st.columns(4)
    for col,(label,val,clr) in zip([c1,c2,c3,c4],[
        ("MAE",      mae,        "blue"),
        ("RMSE",     rmse,       "orange"),
        ("MAPE",     f"{mape}%", "green"),
        ("R² Score", r2,         "green" if r2 > 0.85 else "orange"),
    ]):
        with col:
            st.markdown(f'<div class="mcard"><div class="mcard-label">{label}</div><div class="mcard-value {clr}">{val}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    y_actual = y_test.values[:n]; y_pred = pred[:n]

    st.markdown(f'<div class="stitle">{model_choice} — Actual vs Predicted</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(range(n), y_actual, color='#c0d8f0', linewidth=1.4, label='Actual', alpha=0.9)
    ax.plot(range(n), y_pred, color=color, linewidth=1.4, label='Predicted', linestyle='--', alpha=0.9)
    ax.fill_between(range(n), np.minimum(y_actual,y_pred), np.maximum(y_actual,y_pred), alpha=0.1, color=color)
    ax.legend(fontsize=10); ax.set_xlabel("Time Steps (hours)"); ax.set_ylabel("Power (kW)"); ax.grid(True, alpha=0.2)
    fig.tight_layout(); st.pyplot(fig); plt.close()

    c1, c2 = st.columns([2,1])
    with c1:
        st.markdown('<div class="stitle">Prediction Residuals</div>', unsafe_allow_html=True)
        residuals = y_actual - y_pred
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.bar(range(n), residuals, color=[GREEN if r >= 0 else RED for r in residuals], width=1.0, alpha=0.8)
        ax.axhline(0, color=YELLOW, linewidth=1.2)
        ax.axhline(residuals.mean(), color=ORANGE, linewidth=1, linestyle='--', label=f'Mean: {residuals.mean():.4f}')
        ax.legend(fontsize=9); ax.set_xlabel("Time Steps"); ax.set_ylabel("Residual (kW)"); ax.grid(True, axis='y', alpha=0.2)
        fig.tight_layout(); st.pyplot(fig); plt.close()
    with c2:
        st.markdown('<div class="stitle">Error Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(residuals, bins=40, color=color, alpha=0.7, edgecolor='none')
        ax.axvline(0, color=YELLOW, linewidth=1.5)
        ax.axvline(residuals.mean(), color=ORANGE, linewidth=1.5, linestyle='--', label=f'Mean: {residuals.mean():.4f}')
        ax.set_xlabel("Residual"); ax.set_ylabel("Frequency"); ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
        fig.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════
# PAGE 4 — ANOMALY DETECTION
# ══════════════════════════════════════════════════════
elif page == "🚨  Anomaly Detection":
    st.markdown("<h1>🚨 Anomaly Detection</h1>", unsafe_allow_html=True)
    series = df[GAP].copy().dropna()
    tab1, tab2, tab3 = st.tabs(["⚡  Z-Score Method", "🌲  Isolation Forest", "📈  Rolling Threshold"])

    with tab1:
        threshold = st.slider("Z-Score Threshold", 2.0, 4.0, 3.0, 0.1, key='zs')
        z_scores = zscore(series); z_ser = pd.Series(z_scores, index=series.index)
        anomalies = series[abs(z_ser) > threshold]; pct = round(len(anomalies)/len(series)*100, 2)
        c1,c2,c3 = st.columns([1,1,2])
        with c1: st.markdown(f'<div class="abox"><div class="abox-label">Anomalies</div><div class="abox-count">{len(anomalies)}</div><div class="abox-label">detected</div></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="mcard"><div class="mcard-label">Anomaly Rate</div><div class="mcard-value red">{pct}%</div></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="insight">Z-Score flags points <strong>more than {threshold}σ</strong> from the mean. Threshold of 3 covers 99.7% of normal data.</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(14, 4.5))
        ax.plot(series.index, series.values, color=BLUE, linewidth=0.5, alpha=0.6, label='Normal')
        ax.scatter(anomalies.index, anomalies.values, color=RED, s=15, zorder=5, label=f'Anomaly ({len(anomalies)})')
        ax.set_ylabel("Power (kW)"); ax.legend(fontsize=10); ax.grid(True, alpha=0.2)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('<div class="stitle">Top 10 Anomalies</div>', unsafe_allow_html=True)
        top10 = anomalies.abs().nlargest(10).reset_index()
        top10.columns = ['Datetime','Global Active Power (kW)']
        top10['Z-Score'] = top10['Global Active Power (kW)'].apply(lambda x: round(abs((x - series.mean())/series.std()), 2))
        st.dataframe(top10, use_container_width=True)

    with tab2:
        contamination = st.slider("Contamination rate", 0.01, 0.10, 0.02, 0.01, key='iso')
        iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        iso_lbl = iso.fit_predict(series.values.reshape(-1,1))
        iso_an = series[iso_lbl == -1]; pct_iso = round(len(iso_an)/len(series)*100, 2)
        c1,c2,c3 = st.columns([1,1,2])
        with c1: st.markdown(f'<div class="abox"><div class="abox-label">Anomalies</div><div class="abox-count">{len(iso_an)}</div><div class="abox-label">detected</div></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="mcard"><div class="mcard-label">Anomaly Rate</div><div class="mcard-value red">{pct_iso}%</div></div>', unsafe_allow_html=True)
        with c3: st.markdown('<div class="insight">Isolation Forest uses <strong>random partitioning</strong> — anomalies are isolated with fewer splits. No Gaussian assumption needed.</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(14, 4.5))
        ax.plot(series.index, series.values, color=BLUE, linewidth=0.5, alpha=0.6, label='Normal')
        ax.scatter(iso_an.index, iso_an.values, color=RED, s=15, zorder=5, label=f'Anomaly ({len(iso_an)})')
        ax.set_ylabel("Power (kW)"); ax.legend(fontsize=10); ax.grid(True, alpha=0.2)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('<div class="stitle">Top 10 Isolation Forest Anomalies</div>', unsafe_allow_html=True)
        top10_iso = iso_an.abs().nlargest(10).reset_index(); top10_iso.columns = ['Datetime','Global Active Power (kW)']
        st.dataframe(top10_iso, use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1: window = st.slider("Rolling window (hours)", 6, 72, 24, 6, key='rw')
        with c2: multiplier = st.slider("Std multiplier", 1.0, 4.0, 2.0, 0.5, key='rm')
        roll_mean = series.rolling(window, min_periods=1).mean()
        roll_std  = series.rolling(window, min_periods=1).std().fillna(0)
        upper = roll_mean + multiplier * roll_std; lower = roll_mean - multiplier * roll_std
        roll_an = series[(series > upper) | (series < lower)]; pct_roll = round(len(roll_an)/len(series)*100, 2)
        c1,c2,c3 = st.columns([1,1,2])
        with c1: st.markdown(f'<div class="abox"><div class="abox-label">Anomalies</div><div class="abox-count">{len(roll_an)}</div><div class="abox-label">detected</div></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="mcard"><div class="mcard-label">Anomaly Rate</div><div class="mcard-value red">{pct_roll}%</div></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="insight">Rolling threshold detects <strong>sudden spikes/drops</strong> using a {window}-hour window with ±{multiplier}σ bands.</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(14, 4.5))
        ax.plot(series.index, series.values, color=BLUE, linewidth=0.5, alpha=0.5, label='Consumption')
        ax.plot(roll_mean.index, roll_mean.values, color=GREEN, linewidth=1, alpha=0.7, label='Rolling Mean')
        ax.fill_between(upper.index, lower.values, upper.values, alpha=0.1, color=GREEN, label='Normal band')
        ax.scatter(roll_an.index, roll_an.values, color=RED, s=15, zorder=5, label=f'Anomaly ({len(roll_an)})')
        ax.set_ylabel("Power (kW)"); ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
        fig.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════
# PAGE 5 — MODEL COMPARISON
# ══════════════════════════════════════════════════════
elif page == "📋  Model Comparison":
    st.markdown("<h1>📋 Model Comparison</h1>", unsafe_allow_html=True)

    lr_mae,  lr_rmse,  lr_r2,  lr_mape  = res['lr_metrics']
    rf_mae,  rf_rmse,  rf_r2,  rf_mape  = res['rf_metrics']
    xgb_mae, xgb_rmse, xgb_r2, xgb_mape = res['xgb_metrics']

    comp = pd.DataFrame({
        'Model':    ['Linear Regression', 'Random Forest', 'XGBoost'],
        'MAE':      [lr_mae,  rf_mae,  xgb_mae],
        'RMSE':     [lr_rmse, rf_rmse, xgb_rmse],
        'MAPE (%)': [lr_mape, rf_mape, xgb_mape],
        'R² Score': [lr_r2,   rf_r2,   xgb_r2],
    })

    best_r2_idx = comp['R² Score'].idxmax()
    c1,c2,c3,c4 = st.columns(4)
    for col,(label,val,clr) in zip([c1,c2,c3,c4],[
        ("Best R²",   f"{comp['R² Score'].max():.4f}", "green"),
        ("Best MAE",  f"{comp['MAE'].min():.4f}",      "blue"),
        ("Best RMSE", f"{comp['RMSE'].min():.4f}",     "blue"),
        ("Best MAPE", f"{comp['MAPE (%)'].min():.2f}%","orange"),
    ]):
        with col:
            st.markdown(f'<div class="mcard"><div class="mcard-label">{label}</div><div class="mcard-value {clr}">{val}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="stitle">Full Metrics Table</div>', unsafe_allow_html=True)
    st.dataframe(comp.set_index('Model').style
                 .highlight_max(subset=['R² Score'], color='#0d2a1a')
                 .highlight_min(subset=['MAE','RMSE','MAPE (%)'], color='#0d1a2a'),
                 use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="stitle">R² Score (Higher = Better)</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6,4))
        clrs = [GREEN if i == best_r2_idx else BLUE for i in range(len(comp))]
        bars = ax.bar(comp['Model'], comp['R² Score'], color=clrs, width=0.5, alpha=0.85)
        for bar, val in zip(bars, comp['R² Score']):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f'{val:.4f}', ha='center', fontsize=11, color='#c0d8f0', fontweight='bold')
        ax.set_ylim(0, 1.15); ax.set_ylabel("R² Score"); ax.grid(True, axis='y', alpha=0.2)
        ax.tick_params(axis='x', rotation=10); fig.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        st.markdown('<div class="stitle">MAE — Lower is Better</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6,4))
        clrs2 = [GREEN if i == comp['MAE'].idxmin() else ORANGE for i in range(len(comp))]
        bars = ax.bar(comp['Model'], comp['MAE'], color=clrs2, width=0.5, alpha=0.85)
        for bar, val in zip(bars, comp['MAE']):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                    f'{val:.4f}', ha='center', fontsize=11, color='#c0d8f0', fontweight='bold')
        ax.set_ylabel("MAE"); ax.grid(True, axis='y', alpha=0.2)
        ax.tick_params(axis='x', rotation=10); fig.tight_layout(); st.pyplot(fig); plt.close()

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="stitle">RMSE Comparison</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6,3))
        bars = ax.barh(comp['Model'], comp['RMSE'], color=[BLUE,ORANGE,PURPLE], height=0.45, alpha=0.85)
        for bar, val in zip(bars, comp['RMSE']):
            ax.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=11, color='#c0d8f0')
        ax.set_xlabel("RMSE"); ax.grid(True, axis='x', alpha=0.2)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with c4:
        st.markdown('<div class="stitle">MAPE % Comparison</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6,3))
        bars = ax.barh(comp['Model'], comp['MAPE (%)'], color=[BLUE,ORANGE,PURPLE], height=0.45, alpha=0.85)
        for bar, val in zip(bars, comp['MAPE (%)']):
            ax.text(bar.get_width()+0.1, bar.get_y()+bar.get_height()/2,
                    f'{val:.2f}%', va='center', fontsize=11, color='#c0d8f0')
        ax.set_xlabel("MAPE (%)"); ax.grid(True, axis='x', alpha=0.2)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    best_model = comp.loc[best_r2_idx,'Model']
    st.markdown(f"""
    <div class="champion">
        <div class="champion-label">🏆 Best Performing Model</div>
        <div class="champion-name">{best_model}</div>
        <div class="champion-stats">
            R² Score: <strong>{comp.loc[best_r2_idx,'R² Score']}</strong> &nbsp;·&nbsp;
            MAE: <strong>{comp.loc[best_r2_idx,'MAE']}</strong> &nbsp;·&nbsp;
            MAPE: <strong>{comp.loc[best_r2_idx,'MAPE (%)']:.2f}%</strong>
        </div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# PAGE 6 — MODEL TESTING & EVALUATION
# ══════════════════════════════════════════════════════
elif page == "🧪  Model Testing & Evaluation":
    st.markdown("<h1>🧪 Model Testing & Evaluation</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#2a5a7a;font-size:14px;margin-top:-10px;margin-bottom:20px;'>Proper chronological train-test split · Real metrics from Colab · No data leakage</p>", unsafe_allow_html=True)

    st.markdown('<div class="stitle">1. Train-Test Split Information</div>', unsafe_allow_html=True)
    total     = res['split_idx'] + len(res['y_test'])
    n_train   = res['split_idx']
    n_test    = len(res['y_test'])
    pct_train = round(n_train / total * 100, 1)
    pct_test  = round(n_test  / total * 100, 1)

    c1,c2,c3,c4 = st.columns(4)
    split_items = [
        ("Split Method",  "Chronological",                       "No shuffle — time order preserved"),
        ("Training Size", f"{n_train:,} records ({pct_train}%)", f"{res['train_start'].strftime('%Y-%m-%d')} to {res['train_end'].strftime('%Y-%m-%d')}"),
        ("Testing Size",  f"{n_test:,} records ({pct_test}%)",   f"{res['test_start'].strftime('%Y-%m-%d')} to {res['test_end'].strftime('%Y-%m-%d')}"),
        ("Total Records", f"{total:,}",                          "Hourly resampled data"),
    ]
    for col,(title,val,sub) in zip([c1,c2,c3,c4], split_items):
        with col:
            st.markdown(f'<div class="split-box"><div class="split-box-title">{title}</div><div class="split-box-val">{val}</div><div class="split-box-sub">{sub}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(14, 1.2))
    ax.barh(0, n_train, color=BLUE, alpha=0.85, height=0.6)
    ax.barh(0, n_test, left=n_train, color=ORANGE, alpha=0.85, height=0.6)
    ax.text(n_train/2, 0, f'TRAINING  {pct_train}%', ha='center', va='center', color='white', fontsize=11, fontweight='bold')
    ax.text(n_train + n_test/2, 0, f'TESTING  {pct_test}%', ha='center', va='center', color='white', fontsize=11, fontweight='bold')
    ax.set_xlim(0, total); ax.axis('off')
    fig.patch.set_facecolor('#0d1525'); fig.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown(f'<div class="insight">📌 <strong>Training period:</strong> {res["train_start"].strftime("%d %b %Y")} → {res["train_end"].strftime("%d %b %Y")} &nbsp;|&nbsp; <strong>Testing period:</strong> {res["test_start"].strftime("%d %b %Y")} → {res["test_end"].strftime("%d %b %Y")}<br>Split is strictly chronological — the model never sees future data during training. This prevents data leakage.</div>', unsafe_allow_html=True)

    st.markdown('<div class="stitle">2. Testing Period — Actual Consumption</div>', unsafe_allow_html=True)
    y_test = res['y_test']
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.fill_between(range(len(y_test)), y_test.values, alpha=0.15, color=GREEN)
    ax.plot(range(len(y_test)), y_test.values, color=GREEN, linewidth=0.8, label='Actual (Test Set)')
    ax.axhline(y_test.mean(), color=ORANGE, linewidth=1.5, linestyle='--', label=f'Test Mean: {y_test.mean():.3f} kW')
    ax.set_xlabel("Time Steps (hours)"); ax.set_ylabel("Power (kW)"); ax.legend(fontsize=10); ax.grid(True, alpha=0.2)
    ax.set_title(f"Test Period: {res['test_start'].strftime('%d %b %Y')} to {res['test_end'].strftime('%d %b %Y')}", color='#a0c0e0', fontsize=11)
    fig.tight_layout(); st.pyplot(fig); plt.close()

    c1,c2,c3 = st.columns(3)
    with c1: st.markdown(f'<div class="mcard"><div class="mcard-label">Test Min</div><div class="mcard-value blue">{y_test.min():.3f} kW</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="mcard"><div class="mcard-label">Test Mean</div><div class="mcard-value orange">{y_test.mean():.3f} kW</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="mcard"><div class="mcard-label">Test Max</div><div class="mcard-value red">{y_test.max():.3f} kW</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="stitle">3. Model Metrics on Test Data</div>', unsafe_allow_html=True)
    selected_model = st.selectbox("Select Model to Evaluate", ["XGBoost", "Random Forest", "Linear Regression"], key='eval_model')

    if selected_model == "Random Forest":
        pred = res['rf_pred']; mae,rmse,r2,mape = res['rf_metrics']; color = BLUE
    elif selected_model == "XGBoost":
        pred = res['xgb_pred']; mae,rmse,r2,mape = res['xgb_metrics']; color = ORANGE
    else:
        pred = res['lr_pred']; mae,rmse,r2,mape = res['lr_metrics']; color = PURPLE

    c1,c2,c3,c4 = st.columns(4)
    for col,(label,val,clr,desc) in zip([c1,c2,c3,c4],[
        ("MAE",      mae,        "blue",   "Mean Absolute Error"),
        ("RMSE",     rmse,       "orange", "Root Mean Squared Error"),
        ("MAPE",     f"{mape}%", "green",  "Mean Absolute % Error"),
        ("R² Score", r2,         "green" if r2 > 0.85 else "orange", "Coefficient of Determination"),
    ]):
        with col:
            st.markdown(f'<div class="mcard"><div class="mcard-label">{label}</div><div class="mcard-value {clr}">{val}</div><div class="mcard-sub">{desc}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="insight">📌 These are the <strong>exact metrics from your Colab notebook</strong> — computed on the test set only (data the model never saw during training).</div>', unsafe_allow_html=True)

    st.markdown('<div class="stitle">4. Actual vs Predicted — Test Set</div>', unsafe_allow_html=True)
    n_pts = st.slider("Test points to display", 100, min(600, len(y_test)), 300, 50, key='eval_pts')
    y_actual = y_test.values[:n_pts]; y_pred = pred[:n_pts]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(range(n_pts), y_actual, color='#c0d8f0', linewidth=1.5, label='Actual', alpha=0.9, zorder=2)
    ax.plot(range(n_pts), y_pred, color=color, linewidth=1.5, label=f'{selected_model} Predicted', linestyle='--', alpha=0.9, zorder=3)
    ax.fill_between(range(n_pts), np.minimum(y_actual,y_pred), np.maximum(y_actual,y_pred), alpha=0.12, color=color, label='Error band', zorder=1)
    ax.legend(fontsize=11); ax.set_xlabel("Time Steps (hours on test set)"); ax.set_ylabel("Global Active Power (kW)")
    ax.set_title(f"{selected_model} — Test Set  |  R²={r2}  MAE={mae}", color='#a0c0e0', fontsize=11)
    ax.grid(True, alpha=0.2); fig.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="stitle">5. Residual Analysis</div>', unsafe_allow_html=True)
    residuals = y_actual - y_pred
    c1, c2 = st.columns([2,1])
    with c1:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.bar(range(n_pts), residuals, color=[GREEN if r >= 0 else RED for r in residuals], width=1.0, alpha=0.8)
        ax.axhline(0, color=YELLOW, linewidth=1.5)
        ax.axhline(residuals.mean(), color=ORANGE, linewidth=1.2, linestyle='--', label=f'Mean residual: {residuals.mean():.4f}')
        ax.axhline(residuals.std(), color=PURPLE, linewidth=1, linestyle=':', label=f'Std: {residuals.std():.4f}')
        ax.legend(fontsize=9); ax.set_xlabel("Time Steps"); ax.set_ylabel("Residual (kW)"); ax.grid(True, axis='y', alpha=0.2)
        fig.tight_layout(); st.pyplot(fig); plt.close()
    with c2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(residuals, bins=40, color=color, alpha=0.75, edgecolor='none')
        ax.axvline(0, color=YELLOW, linewidth=2, label='Zero error')
        ax.axvline(residuals.mean(), color=ORANGE, linewidth=1.5, linestyle='--', label=f'Mean: {residuals.mean():.4f}')
        ax.set_xlabel("Residual (kW)"); ax.set_ylabel("Frequency"); ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
        ax.set_title("Error Distribution", color='#a0c0e0', fontsize=10)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    c1,c2,c3,c4 = st.columns(4)
    for col,(label,val,clr) in zip([c1,c2,c3,c4],[
        ("Mean Residual", f"{residuals.mean():.4f}", "blue"),
        ("Std Residual",  f"{residuals.std():.4f}",  "orange"),
        ("Max Error",     f"{residuals.max():.4f}",  "red"),
        ("Min Error",     f"{residuals.min():.4f}",  "green"),
    ]):
        with col:
            st.markdown(f'<div class="mcard"><div class="mcard-label">{label}</div><div class="mcard-value {clr}" style="font-size:20px;">{val}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="stitle">6. Full Model Comparison Table</div>', unsafe_allow_html=True)
    lr_mae,  lr_rmse,  lr_r2,  lr_mape  = res['lr_metrics']
    rf_mae,  rf_rmse,  rf_r2,  rf_mape  = res['rf_metrics']
    xgb_mae, xgb_rmse, xgb_r2, xgb_mape = res['xgb_metrics']

    comp_df = pd.DataFrame({
        'Model':    ['Linear Regression', 'Random Forest', 'XGBoost'],
        'MAE':      [lr_mae,  rf_mae,  xgb_mae],
        'RMSE':     [lr_rmse, rf_rmse, xgb_rmse],
        'MAPE (%)': [lr_mape, rf_mape, xgb_mape],
        'R² Score': [lr_r2,   rf_r2,   xgb_r2],
        'Train Size':[f"{n_train:,}", f"{n_train:,}", f"{n_train:,}"],
        'Test Size': [f"{n_test:,}",  f"{n_test:,}",  f"{n_test:,}"],
    })

    best_idx = comp_df['R² Score'].idxmax()
    st.dataframe(
        comp_df.set_index('Model').style
               .highlight_max(subset=['R² Score'], color='#0d2a1a')
               .highlight_min(subset=['MAE','RMSE','MAPE (%)'], color='#0d1a2a'),
        use_container_width=True
    )

    best = comp_df.iloc[best_idx]
    st.markdown(f"""
    <div class="champion">
        <div class="champion-label">🏆 Best Model on Test Set</div>
        <div class="champion-name">{best['Model']}</div>
        <div class="champion-stats">
            R² Score: <strong>{best['R² Score']}</strong> &nbsp;·&nbsp;
            MAE: <strong>{best['MAE']}</strong> &nbsp;·&nbsp;
            RMSE: <strong>{best['RMSE']}</strong> &nbsp;·&nbsp;
            MAPE: <strong>{best['MAPE (%)']:.2f}%</strong>
        </div>
        <p style="color:#3a7a4a;font-size:13px;margin-top:12px;">
            Trained on {n_train:,} hours · Tested on {n_test:,} hours · Chronological split · No data leakage
        </p>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# PAGE 7 — FEATURE INSIGHTS
# ══════════════════════════════════════════════════════
elif page == "🧠  Feature Insights":
    st.markdown("<h1>🧠 Feature Insights</h1>", unsafe_allow_html=True)

    feature_names = res['feature_cols']
    rf_imp = res['rf_imp']; xgb_imp = res['xgb_imp']

    st.markdown('<div class="stitle">Feature Importance — Random Forest vs XGBoost</div>', unsafe_allow_html=True)
    imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Random Forest': rf_imp,
        'XGBoost': xgb_imp
    }).sort_values('Random Forest', ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (col, clr) in zip(axes, [('Random Forest', BLUE), ('XGBoost', ORANGE)]):
        vals = imp_df[col]
        bars = ax.barh(imp_df['Feature'], vals, color=clr, alpha=0.8, height=0.6)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=8, color='#a0c0e0')
        ax.set_xlabel("Importance Score")
        ax.set_title(col, color=clr, fontsize=12, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.2)
    fig.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown('<div class="insight">📌 <strong>lag_1</strong> and <strong>rolling_mean_24h</strong> dominate — electricity usage is highly auto-correlated. Past values strongly predict future ones.</div>', unsafe_allow_html=True)

    st.markdown('<div class="stitle">Autocorrelation — How Much Does Past Affect Future?</div>', unsafe_allow_html=True)
    lags = range(1, 49)
    acorr = [df[GAP].autocorr(lag=l) for l in lags]
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.bar(lags, acorr, color=[BLUE if v > 0 else RED for v in acorr], alpha=0.7, width=0.8)
    ax.axhline(0, color=YELLOW, linewidth=1)
    ax.axhline(0.05, color=GREEN, linewidth=1, linestyle='--', alpha=0.5, label='5% threshold')
    ax.axhline(-0.05, color=GREEN, linewidth=1, linestyle='--', alpha=0.5)
    ax.set_xlabel("Lag (hours)"); ax.set_ylabel("Autocorrelation")
    ax.set_xticks(list(lags)); ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
    fig.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown('<div class="insight">📌 Strong autocorrelation at lag 1, 24, and 48 confirms time-based lag features are the most powerful predictors.</div>', unsafe_allow_html=True)

    st.markdown('<div class="stitle">Target Variable Distribution</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.hist(df[GAP].dropna(), bins=80, color=BLUE, alpha=0.75, edgecolor='none')
        ax.axvline(df[GAP].mean(), color=ORANGE, linewidth=2, linestyle='--', label=f'Mean: {df[GAP].mean():.3f}')
        ax.axvline(df[GAP].median(), color=GREEN, linewidth=2, linestyle='--', label=f'Median: {df[GAP].median():.3f}')
        ax.set_xlabel("Global Active Power (kW)"); ax.set_ylabel("Frequency")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
        fig.tight_layout(); st.pyplot(fig); plt.close()
    with c2:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.boxplot(df[GAP].dropna(), vert=False, patch_artist=True,
                   boxprops=dict(facecolor='#0d2040', color=BLUE),
                   medianprops=dict(color=ORANGE, linewidth=2),
                   whiskerprops=dict(color=BLUE), capprops=dict(color=BLUE),
                   flierprops=dict(marker='o', color=RED, markersize=2, alpha=0.3))
        ax.set_xlabel("Global Active Power (kW)"); ax.set_yticks([])
        ax.grid(True, axis='x', alpha=0.2)
        fig.tight_layout(); st.pyplot(fig); plt.close()

