"""
Edge AI Vehicle Health Monitoring Dashboard
dashboard.py — Premium Real-Time Streamlit Dashboard

HOW TO RUN (in a SEPARATE terminal):
    streamlit run dashboard.py

WHAT THIS DOES:
    - Reads live_prediction.csv every second (written by predict_live.py)
    - Displays:  metric cards (Total / Healthy / Faulty)
                 bar chart    (GOOD vs BAD)
                 donut chart  (percentage split)
                 live table   (all 5 vehicles, always visible)
    - Dark glassmorphism theme
    - Green = GOOD,  Red = BAD
    - Auto-refreshes every 1 second — no manual refresh needed
"""

import os
import time
import pandas as pd
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime

matplotlib.use("Agg")

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Edge AI Vehicle Health Monitoring",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
LIVE_CSV     = os.path.join(BASE_DIR, "live_prediction.csv")
VEHICLE_IDS  = ["veh0", "veh1", "veh2", "veh3", "veh4"]
COLOR_GOOD   = "#10b981"
COLOR_BAD    = "#ef4444"
REFRESH_SEC  = 1

# ──────────────────────────────────────────────────────────────
# PREMIUM CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1526 50%, #0a0f1e 100%);
}

#MainMenu, header, footer { visibility: hidden; }

/* ── Hero ── */
.hero-title {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 60%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0; padding: 0; line-height: 1.2;
}
.hero-sub {
    color: #64748b; font-size: 0.9rem; margin-top: 4px;
}

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    padding: 30px 20px;
    border-radius: 18px;
    border: 1px solid #1e3a5f;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04);
    text-align: center;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.metric-card:hover { transform: translateY(-4px); border-color: #334155; }
.metric-icon  { font-size: 2.2rem; margin-bottom: 8px; display: block; }
.metric-title { color: #94a3b8; font-size: 0.72rem; font-weight: 700;
                text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 10px; }
.metric-value { font-size: 3.8rem; font-weight: 800; line-height: 1; }
.val-total    { color: #38bdf8; text-shadow: 0 0 20px rgba(56,189,248,0.35); }
.val-healthy  { color: #10b981; text-shadow: 0 0 20px rgba(16,185,129,0.35); }
.val-faulty   { color: #ef4444; text-shadow: 0 0 20px rgba(239,68,68,0.35); }

/* ── Section headings ── */
.sec-head {
    color: #e2e8f0; font-size: 1rem; font-weight: 700; letter-spacing: 0.04em;
    padding: 10px 0 8px 0; border-bottom: 1px solid #1e293b; margin-bottom: 14px;
}

/* ── Divider ── */
.divider { border: none; border-top: 1px solid #1e293b; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# CHART THEME HELPER
# ──────────────────────────────────────────────────────────────
BG_DARK = "#0f172a"
BG_CARD = "#1e293b"
TXT     = "#e2e8f0"


def dark_fig(w=6.5, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_CARD)
    ax.tick_params(colors=TXT, labelsize=10)
    for sp in ax.spines.values():
        sp.set_color("#1e293b")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.22, color="#334155")
    ax.set_axisbelow(True)
    return fig, ax


# ──────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────
def load_data():
    """
    Read live_prediction.csv.
    Always returns a 5-row DataFrame — missing vehicles get placeholder '—'.
    """
    base = pd.DataFrame({
        "vehicle_id"      : VEHICLE_IDS,
        "speed"           : [0.0] * 5,
        "engine_temp"     : [0]   * 5,
        "vibration"       : [0.0] * 5,
        "predicted_status": ["—"] * 5,
    })

    if not os.path.exists(LIVE_CSV):
        return base, "waiting"

    try:
        df = pd.read_csv(LIVE_CSV)
    except Exception:
        return base, "error"

    if df.empty:
        return base, "empty"

    required = {"vehicle_id", "speed", "engine_temp", "vibration", "predicted_status"}
    if not required.issubset(df.columns):
        return base, "schema"

    # Keep only latest row per vehicle_id
    df = df.drop_duplicates(subset="vehicle_id", keep="last")

    # Left-join so all 5 vehicles always appear
    merged = base[["vehicle_id"]].merge(df, on="vehicle_id", how="left")
    merged["predicted_status"] = merged["predicted_status"].fillna("—")
    merged["speed"]            = merged["speed"].fillna(0.0)
    merged["engine_temp"]      = merged["engine_temp"].fillna(0)
    merged["vibration"]        = merged["vibration"].fillna(0.0)

    return merged, "ok"


# ──────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────
ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
h_left, h_right = st.columns([3, 1])
with h_left:
    st.markdown(
        "<p class='hero-title'>🚗 Edge AI Vehicle Health Monitoring</p>"
        "<p class='hero-sub'>Real-time predictive maintenance · Random Forest classifier · SUMO simulation</p>",
        unsafe_allow_html=True
    )
with h_right:
    st.markdown(f"""
        <div style="text-align:right;padding-top:16px;">
            <span style="background:rgba(56,189,248,0.1);border:1px solid rgba(56,189,248,0.25);
                         border-radius:8px;padding:6px 14px;color:#38bdf8;
                         font-size:0.8rem;font-weight:600;">🕐 {ts}</span>
        </div>""",
        unsafe_allow_html=True
    )

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# LOAD + HANDLE STATES
# ──────────────────────────────────────────────────────────────
df, state = load_data()

if state == "waiting":
    st.info("⏳ Waiting for live data — start `python predict_live.py` in another terminal.")
    time.sleep(REFRESH_SEC); st.rerun()

if state in ("error", "schema"):
    st.warning("⚠️ Could not read live_prediction.csv — retrying …")
    time.sleep(REFRESH_SEC); st.rerun()

if state == "empty":
    st.info("📡 CSV detected — waiting for first predictions …")
    time.sleep(REFRESH_SEC); st.rerun()

# ──────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────
total_v   = len(VEHICLE_IDS)
healthy_v = int((df["predicted_status"] == "GOOD").sum())
faulty_v  = int((df["predicted_status"] == "BAD").sum())

c1, c2, c3 = st.columns(3)
for col, icon, title, cls, val in [
    (c1, "🚗", "Total Monitored Fleet",  "val-total",   total_v),
    (c2, "✅", "Healthy Vehicles (GOOD)", "val-healthy", healthy_v),
    (c3, "⚠️", "Faulty Vehicles (BAD)",  "val-faulty",  faulty_v),
]:
    with col:
        st.markdown(f"""
            <div class="metric-card">
                <span class="metric-icon">{icon}</span>
                <div class="metric-title">{title}</div>
                <div class="metric-value {cls}">{val}</div>
            </div>""",
            unsafe_allow_html=True
        )

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# CHARTS
# ──────────────────────────────────────────────────────────────
labels = ["Healthy (GOOD)", "Faulty (BAD)"]
counts = [healthy_v, faulty_v]
colors = [COLOR_GOOD, COLOR_BAD]

bc, pc = st.columns(2)

# ── Bar Chart ──────────────────────────────────────────────────
with bc:
    st.markdown("<div class='sec-head'>📊 Fleet Health Distribution</div>",
                unsafe_allow_html=True)
    fig, ax = dark_fig()
    bars = ax.bar(labels, counts, color=colors, edgecolor="#0f172a", width=0.4, zorder=3)
    ax.set_ylim(0, total_v + 1)
    ax.set_yticks(range(0, total_v + 2))
    ax.tick_params(axis="x", colors=TXT, labelsize=11)
    ax.tick_params(axis="y", colors="#475569")
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.06,
                    str(int(h)), ha="center", va="bottom",
                    color=TXT, fontweight="bold", fontsize=14)
    fig.tight_layout(pad=1.5)
    st.pyplot(fig)
    plt.close(fig)

# ── Donut Chart ────────────────────────────────────────────────
with pc:
    st.markdown("<div class='sec-head'>🥧 Fleet Health Split</div>",
                unsafe_allow_html=True)
    fig2, ax2 = dark_fig()
    ax2.set_facecolor(BG_DARK)
    pie_counts = counts if sum(counts) > 0 else [1, 0]
    wedges, texts, autos = ax2.pie(
        pie_counts, labels=labels, autopct="%1.0f%%", startangle=90,
        colors=colors, pctdistance=0.78,
        wedgeprops=dict(width=0.52, edgecolor=BG_DARK, linewidth=2),
        textprops=dict(color=TXT, fontsize=10, fontweight="600"),
    )
    for at in autos:
        at.set_color("white"); at.set_fontweight("bold"); at.set_fontsize(12)
    ax2.text(0, 0, f"{total_v}\nVehicles", ha="center", va="center",
             color=TXT, fontsize=13, fontweight="bold")
    fig2.tight_layout(pad=1.5)
    st.pyplot(fig2)
    plt.close(fig2)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# LIVE VEHICLE TABLE
# ──────────────────────────────────────────────────────────────
st.markdown("<div class='sec-head'>📡 Live Vehicle Telemetry &amp; Status</div>",
            unsafe_allow_html=True)

disp = df[["vehicle_id", "speed", "engine_temp", "vibration", "predicted_status"]].copy()
disp.columns = ["Vehicle ID", "Speed (m/s)", "Engine Temp (°C)", "Vibration", "Status"]
disp["Speed (m/s)"]       = disp["Speed (m/s)"].round(2)
disp["Vibration"]         = disp["Vibration"].round(2)
disp["Engine Temp (°C)"]  = disp["Engine Temp (°C)"].astype(int)


def style_status(val):
    if val == "GOOD":
        return "color:#10b981;font-weight:700;background:rgba(16,185,129,0.1);"
    if val == "BAD":
        return "color:#ef4444;font-weight:700;background:rgba(239,68,68,0.1);"
    return "color:#64748b;"


def style_vid(val):
    return "color:#38bdf8;font-weight:700;"


styled = (
    disp.style
    .map(style_status, subset=["Status"])
    .map(style_vid,    subset=["Vehicle ID"])
    .set_properties(**{
        "background-color": "#0f172a",
        "color"           : "#e2e8f0",
        "border-color"    : "#1e293b",
        "font-size"       : "14px",
        "padding"         : "10px 14px",
    })
    .set_table_styles([
        {"selector": "thead th", "props": [
            ("background-color", "#1e293b"), ("color", "#94a3b8"),
            ("font-weight", "700"), ("text-transform", "uppercase"),
            ("letter-spacing", "0.06em"), ("font-size", "11px"),
            ("padding", "12px 14px"),
        ]},
        {"selector": "tbody tr:hover td", "props": [("background-color", "#1a2744")]},
    ])
)

st.dataframe(styled, use_container_width=True, height=260)

# ──────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#334155;font-size:0.75rem;padding-bottom:12px;">
    Edge AI Vehicle Health Monitoring System &nbsp;|&nbsp;
    Random Forest · SUMO · TraCI · Streamlit
</div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# AUTO-REFRESH every 1 second
# ──────────────────────────────────────────────────────────────
time.sleep(REFRESH_SEC)
st.rerun()
