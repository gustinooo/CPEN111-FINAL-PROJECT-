import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime
import matplotlib.pyplot as plt
from image_base64 import LETTUCE_BASE64

# ---------------- Page Config ----------------
st.set_page_config(page_title="Lettuce Growth Monitoring Dashboard", layout="wide")

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'
if 'selected_metric' not in st.session_state:
    st.session_state.selected_metric = None

# Ensure `st.query_params` is available and use it (experimental_get_query_params removed)
try:
    _ = st.query_params
except Exception:
    # If Streamlit doesn't expose `query_params` for some reason, fall back to empty dict
    st.query_params = {}


def safe_rerun():
    """Rerun the Streamlit script."""
    st.rerun()

# ---------------- Constants ----------------
CARD_HEIGHT = 180
refresh_interval = 30
num_results = 100

# ---------------- CSS ----------------
css = f"""
<style>
/* Apple-like clean system font */
html, body, [data-testid='stAppViewContainer'] {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  color: #ffffff;
}}

[data-testid='stAppViewContainer'] {{
    /* keep lettuce background but darkened with overlay */
    background-image: linear-gradient(rgba(0,0,0,0.4), rgba(0,0,0,0.4)), url('data:image/jpeg;base64,{LETTUCE_BASE64}');
    background-size: cover;
    background-attachment: fixed;
    background-repeat: no-repeat;
    -webkit-font-smoothing: antialiased;
}}

[data-testid='stMainBlockContainer'] {{ background-color: transparent !important; }}

/* Top-right date/time box */
.datetime-box {{
    background-color: rgba(20,20,20,0.75);
    border-radius: 16px;
    padding: 10px 14px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    font-size: 13px;
    font-weight: 600;
    color: #111111;
    line-height: 1.2;
    backdrop-filter: blur(6px);
    border: 1px solid rgba(255,255,255,0.1);
}}

/* Metric cards - dark translucent glass panels */
.metric-card-visual {{
    background: rgba(20,20,20,0.75);
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    height: {CARD_HEIGHT}px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    text-align: center;
    backdrop-filter: blur(6px);
    border: 1px solid rgba(255,255,255,0.1);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}}
.metric-card-visual:hover {{ transform: none; box-shadow: 0 10px 30px rgba(0,0,0,0.3); }}

.metric-label {{ font-size: 12px; font-weight: 600; color: #ffffff; text-transform: uppercase; letter-spacing: 0.6px; text-align: left; width:100%; padding-left:6px; text-shadow: 0 2px 4px rgba(0,0,0,0.5); }}
.metric-value {{ font-size: 40px; font-weight: 700; color: #ffffff; text-align: center; width: 100%; margin: 6px 0; text-shadow: 0 2px 4px rgba(0,0,0,0.5); }}
.metric-status {{ font-size: 12px; color: #cccccc; text-align: left; padding-left:6px; text-shadow: 0 2px 4px rgba(0,0,0,0.5); }}

/* Smaller global tweaks */
.stTitle {{ font-weight:700; }}
.stMetric {{ color: #111111; }}

</style>
"""
st.markdown(css, unsafe_allow_html=True)

# Ensure metric wrapper/button overlay styles (in case Streamlit wraps forms differently)
st.markdown(
    """
    <style>
    .metric-wrapper { position: relative; display: flex; flex-direction: column; }
    .metric-wrapper form { position: relative; margin: 0; padding: 0; }
    /* Place a visible View button under each card, full width with spacing */
    .metric-wrapper .stButton { position: relative; margin-top: 12px; width: 100%; display:flex; justify-content:center; }
    .metric-wrapper .stButton button { width: 100% !important; font-size: 14px !important; color: #ffffff !important; background: linear-gradient(180deg,#2b7cff,#0b60ff) !important; border-radius: 10px !important; padding: 8px 16px !important; border: none !important; box-shadow: 0 6px 18px rgba(11,96,255,0.16) !important; cursor: pointer !important; }

    /* Remove hover animation and overlay so cards are static */
    .metric-card-visual:hover { transform: none !important; box-shadow: 0 10px 30px rgba(15,15,15,0.06) !important; }
    .metric-overlay { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- Title and Date/Time ----------------
col_title, col_datetime = st.columns([3, 1])
with col_title:
    st.title("🥬 Lettuce Growth Monitoring Dashboard")
with col_datetime:
    current_date = datetime.now().strftime("%B %d, %Y")
    current_time = datetime.now().strftime("%H:%M")
    st.markdown(f'<div class="datetime-box">Date: {current_date}<br>Time: {current_time}</div>', unsafe_allow_html=True)

# ---------------- ThingSpeak Config ----------------
CHANNEL_ID = "3186362"
READ_API_KEY = "767PVA4E7GDFP1HC"
THINKSPEAK_URL_BASE = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json"

def fetch_thingspeak(results: int = 100):
    """Fetch feeds from ThingSpeak and return a DataFrame."""
    params = {"api_key": READ_API_KEY, "results": results}
    try:
        resp = requests.get(THINKSPEAK_URL_BASE, params=params, timeout=10)
        if resp.status_code != 200:
            st.error(f"API Error: {resp.status_code}")
            return pd.DataFrame()
        data = resp.json()
        feeds = data.get("feeds", [])
        if not feeds:
            cols = ["created_at", "field1", "field2", "field3", "field4", "field5"]
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame(feeds)
        return df
    except Exception as exc:
        st.error(f"Error: {exc}")
        return pd.DataFrame()

def last_valid_value(df: pd.DataFrame, col: str):
    """Return the last non-null numeric value."""
    if df is None or col not in df.columns:
        return None
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    return None if series.empty else series.iloc[-1]

# ---------------- Page Logic ----------------
if st.session_state.current_page == 'detail' and st.session_state.selected_metric:
    # --- DETAIL PAGE ---
    if st.button("← Back to Home"):
        st.session_state.current_page = 'home'
        st.session_state.selected_metric = None
        st.rerun()
    
    st.markdown("---")

    metric_names = {
        "temp": ("Temperature", "field1"),
        "humidity": ("Humidity", "field2"),
        "rain": ("Rain", "field3"),
        "soil": ("Soil", "field4"),
        "light": ("Light", "field5")
    }

    metric_name, field_key = metric_names[st.session_state.selected_metric]
    st.subheader(f"📈 {metric_name} - Historical Data")

    df_history_raw = fetch_thingspeak(results=500)
    if df_history_raw.empty or field_key not in df_history_raw.columns:
        st.warning("No historical data available.")
    else:
        df_history = df_history_raw[["created_at", field_key]].rename(columns={field_key: metric_name})
        df_history[metric_name] = pd.to_numeric(df_history[metric_name], errors="coerce")
        df_history = df_history.dropna()
        if df_history.empty:
            st.warning("No valid numeric historical data.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Current", f"{df_history[metric_name].iloc[-1]:.2f}")
            with col2: st.metric("Average", f"{df_history[metric_name].mean():.2f}")
            with col3: st.metric("Max", f"{df_history[metric_name].max():.2f}")
            with col4: st.metric("Min", f"{df_history[metric_name].min():.2f}")

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(range(len(df_history)), df_history[metric_name].values, linewidth=2, color="#1f77b4")
            ax.fill_between(range(len(df_history)), df_history[metric_name].values, alpha=0.3, color="#1f77b4")
            ax.set_title(f"{metric_name} Over Time")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            with st.expander("View Raw Data"):
                st.dataframe(df_history, use_container_width=True)

else:
    # --- HOME PAGE ---
    df_raw = fetch_thingspeak(results=num_results)
    if df_raw.empty:
        df_clean = pd.DataFrame(columns=["created_at", "Temperature", "Humidity", "Rain", "Soil", "Light"])
    else:
        df_clean = df_raw[["created_at", "field1", "field2", "field3", "field4", "field5"]].rename(
            columns={
                "field1": "Temperature", "field2": "Humidity",
                "field3": "Rain", "field4": "Soil", "field5": "Light"
            }
        )

    col1, col2, col3, col4, col5 = st.columns(5)
    
    metrics_data = [
        (col1, "Temperature", last_valid_value(df_clean, "Temperature"), "temp", "°C", "Normal"),
        (col2, "Humidity", last_valid_value(df_clean, "Humidity"), "humidity", "%", "Normal"),
        (col3, "Rain", last_valid_value(df_clean, "Rain"), "rain", "mm", "Normal"),
        (col4, "Soil", last_valid_value(df_clean, "Soil"), "soil", "mV", "Normal"),
        (col5, "Light", last_valid_value(df_clean, "Light"), "light", "lux", "Normal"),
    ]

    # Render cards with View buttons
    for col, label, value, metric_key, unit, status in metrics_data:
        display_value = "N/A" if value is None else f"{value:.2f}"
        with col:
            # Render card and button in a wrapper container
            st.markdown(f"""
            <div class="metric-wrapper">
              <div class="metric-card-visual">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{display_value}<span style="font-size: 20px">{unit if display_value != 'N/A' else ''}</span></div>
                <div class="metric-status">{status}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add spacing and button
            st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)

            # Add a visible 'View' button under the card that opens the detail inline
            if st.button("View", key=f"view_{metric_key}", use_container_width=True):
                st.session_state.selected_metric = metric_key
                st.session_state.current_page = 'detail'
                try:
                    st.query_params = {}
                except Exception:
                    try:
                        st.experimental_set_query_params()
                    except Exception:
                        pass
                safe_rerun()

    # ---------------- Growth Predictions (placeholder) ----------------
    st.subheader("Growth Predictions")
    # Single lettuce placeholder card
    st.markdown(
        """
        <div style="display:flex; gap:16px; margin-top:8px;">
            <div class="metric-card-visual" style="height:120px; width:320px;">
                <div class="metric-label">Lettuce</div>
                <div class="metric-value">7<span style="font-size:18px"> days</span></div>
                <div class="metric-status">Estimated days until harvest</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Auto-refresh
    if refresh_interval and st.session_state.current_page == 'home':
        time.sleep(refresh_interval)
        st.rerun()
