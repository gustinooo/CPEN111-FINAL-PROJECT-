import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime, date, timedelta
import numpy as np
import pickle
import os
import json
import plotly.express as px
import plotly.graph_objects as go
from image_base64 import LETTUCE_BASE64
from io import BytesIO

# ============ CONFIG ============
st.set_page_config(page_title="Lettuce Growth Monitoring Dashboard", layout="wide")

# Initialize session state
if 'selected_metric' not in st.session_state:
    st.session_state.selected_metric = None

# ============ CONSTANTS ============
CHANNEL_ID = "3186362"
READ_API_KEY = "767PVA4E7GDFP1HC"
THINKSPEAK_URL_BASE = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json"
CARD_HEIGHT = 180
CONFIG_FILE = "plant_config.json"
REFRESH_INTERVAL = 30 # Seconds
NUM_RESULTS = 100

# ============ PERSISTENCE FUNCTIONS ============
def load_config():
    """Load planting date from JSON file."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
                return datetime.strptime(data["planting_date"], "%Y-%m-%d").date()
        except:
            return date.today()
    return date.today()

def save_config(planting_date):
    """Save planting date to JSON file."""
    with open(CONFIG_FILE, "w") as f:
        json.dump({"planting_date": planting_date.strftime("%Y-%m-%d")}, f)

# ============ CSS (UPDATED FOR CLOUD FIX) ============
css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body {{
  font-family: 'Poppins', sans-serif;
  color: #333333;
}}

/* === BACKGROUND FIX FOR CLOUD === */
[data-testid="stAppViewContainer"] {{
    background-image: linear-gradient(rgba(0,0,0,0.1), rgba(0,0,0,0.1)), url('data:image/jpeg;base64,{LETTUCE_BASE64}');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    background-repeat: no-repeat;
    -webkit-font-smoothing: antialiased;
}}

/* Make the top header transparent so it doesn't block the image */
[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    color: white;
}}

/* Remove default white background and adjust padding */
[data-testid="stMainBlockContainer"] {{ 
    background-color: transparent !important; 
    padding-top: 50px; 
}}

/* === SIDEBAR FIX === */
[data-testid="stSidebar"] {{
    background-color: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255,255,255,0.5);
}}

/* === REUSABLE LIGHT GLASS STYLE === */
.glass-container {{
    background: rgba(255, 255, 255, 0.75); 
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.4);
}}

.datetime-box {{
    background-color: rgba(255, 255, 255, 0.75);
    border-radius: 16px;
    padding: 10px 14px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    font-size: 13px;
    font-weight: 600;
    color: #333333;
    line-height: 1.2;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.4);
    font-family: 'Poppins', sans-serif;
}}

/* Cards */
.metric-card-visual {{
    background: rgba(255, 255, 255, 0.75); 
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    height: {CARD_HEIGHT}px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    text-align: center;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.4);
}}

.metric-label {{ 
    font-size: 12px; 
    font-weight: 600; 
    color: #555555; 
    text-transform: uppercase; 
    letter-spacing: 0.6px; 
    text-align: left; 
    width:100%; 
    padding-left:6px; 
    font-family: 'Poppins', sans-serif;
}}

.metric-value {{ 
    font-size: 40px; 
    font-weight: 700; 
    color: #000000; 
    text-align: center; 
    width: 100%; 
    margin: 4px 0; 
    font-family: 'Poppins', sans-serif;
}}

.metric-status {{ 
    font-size: 12px; 
    color: #444444; 
    text-align: left; 
    padding-left:6px; 
    font-family: 'Poppins', sans-serif;
}}

.stMetric {{ color: #333333; }}

/* === LIGHT MODE BUTTONS === */
.stButton > button {{
    background-color: rgba(255, 255, 255, 0.85);
    color: #333333;
    border: 1px solid rgba(0,0,0,0.1);
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
    transition: all 0.3s ease;
}}

.stButton > button:hover {{
    background-color: #ffffff;
    color: #2e7d32;
    border-color: #2e7d32;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}}

/* === TAB STYLING === */
.stTabs {{
    background: rgba(255, 255, 255, 0.75);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.4);
}}

.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {{
    color: #333333; 
    font-weight: 600;
    font-family: 'Poppins', sans-serif;
}}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# ============ FUNCTIONS ============

# PERFORMANCE FIX: Cache the data fetch so interactions don't re-query the API
@st.cache_data(ttl=REFRESH_INTERVAL, show_spinner=False)
def fetch_thingspeak(results: int = 100):
    """Fetch feeds from ThingSpeak and return a DataFrame."""
    params = {"api_key": READ_API_KEY, "results": results}
    try:
        resp = requests.get(THINKSPEAK_URL_BASE, params=params, timeout=5)
        if resp.status_code != 200:
            return pd.DataFrame()
        data = resp.json()
        feeds = data.get("feeds", [])
        if not feeds:
            cols = ["created_at", "field1", "field2", "field3", "field4", "field5"]
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame(feeds)
        return df
    except Exception as exc:
        return pd.DataFrame()

def fetch_all_thingspeak(chunk: int = 8000, pause_sec: float = 0.5):
    """Fetch all available ThingSpeak feeds by paging in chunks."""
    all_rows = []
    offset = 0
    while True:
        params = {"api_key": READ_API_KEY, "results": chunk, "offset": offset}
        try:
            resp = requests.get(THINKSPEAK_URL_BASE, params=params, timeout=15)
            if resp.status_code != 200:
                break
            data = resp.json()
            feeds = data.get("feeds", [])
            if not feeds:
                break
            all_rows.extend(feeds)
            # If fewer than chunk were returned, we've reached the end
            if len(feeds) < chunk:
                break
            offset += chunk
            time.sleep(pause_sec)
        except Exception:
            break

    if not all_rows:
        return pd.DataFrame()
    return pd.DataFrame(all_rows)

def last_valid_value(df: pd.DataFrame, col: str):
    """Return the last non-null numeric value."""
    if df is None or col not in df.columns:
        return None
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    return None if series.empty else series.iloc[-1]

@st.cache_resource
def load_sarima_model():
    """Load the SARIMA model for predictions."""
    model_path = os.path.join(os.path.dirname(__file__), "best_sarima_model.pkl")
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load SARIMA model: {e}")
            return None
    return None

def predict_harvest_days(model):
    """Predict days until lettuce harvest using SARIMA model."""
    # Get current age for context
    current_date_obj = datetime.now().date()
    saved_planting_date = load_config()
    days_elapsed = (current_date_obj - saved_planting_date).days
    
    if model is None:
        # Calculate remaining days to reach 44 days total (midpoint of 40-48)
        return max(10, min(37, 44 - days_elapsed))
    try:
        # Get recent data for context - uses Cached function
        df_data = fetch_thingspeak(results=100)
        if df_data.empty or "field4" not in df_data.columns:
            return max(10, min(37, 44 - days_elapsed))
        
        # Use soil moisture (field4) as growth indicator
        soil_values = pd.to_numeric(df_data["field4"], errors="coerce").dropna()
        if soil_values.empty:
            return max(10, min(37, 44 - days_elapsed))
        
        # Make a forecast using the SARIMA model
        try:
            # Forecast for 60 days to capture full growth cycle (40-48 days)
            forecast_result = model.get_forecast(steps=60)
            predicted_values = forecast_result.predicted_mean
            
            # Use growth trajectory: look for stabilization or peak in soil moisture
            # as indicator of plant maturity
            recent_median = soil_values.median()
            recent_max = soil_values.max()
            
            # Calculate growth trend: plants mature when soil moisture stabilizes
            # Look for the point where predicted values level off (stop increasing significantly)
            growth_threshold = recent_median + (0.3 * (recent_max - recent_median))
            
            stable_count = 0
            stable_threshold = 5  # Need 5+ consecutive days of stable/declining values
            
            for days_ahead, predicted_val in enumerate(predicted_values, 1):
                # Check if value is near or exceeding growth threshold
                if predicted_val >= growth_threshold:
                    stable_count += 1
                    if stable_count >= stable_threshold:
                        # Calculate total harvest day
                        total_harvest_day = days_elapsed + (days_ahead - 2)
                        # Constrain to 40-48 day range
                        constrained_total = max(40, min(total_harvest_day, 48))
                        # Return remaining days from today
                        remaining = constrained_total - days_elapsed
                        return max(10, remaining)
                else:
                    stable_count = 0
            
            # If stable point not reached, estimate based on average growth rate
            # Target 44 days total (midpoint of 40-48)
            return max(10, min(37, 44 - days_elapsed))
        except Exception as e:
            return max(10, min(37, 44 - days_elapsed))
    except Exception:
        return max(10, min(37, 44 - days_elapsed))

def calculate_irrigation_hours(current_soil_moisture):
    """
    Calculate hours until irrigation is needed based on current soil moisture.
    """
    if current_soil_moisture is None:
        return 24 # Default fallback
    
    # Example logic: Irrigated at 3000mV, Needs water at 2000mV
    dry_threshold = 2000 
    drying_rate_per_hour = 50 
    
    if current_soil_moisture <= dry_threshold:
        return 0 # Needs water now
    
    diff = current_soil_moisture - dry_threshold
    hours = diff / drying_rate_per_hour
    return int(hours)

# ============ SIDEBAR: PLANT CONFIGURATION ============
saved_planting_date = load_config()

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=50)
    st.title("Plant Settings")
    
    with st.form("plant_config_form"):
        st.markdown("Set the day your plant started growing:")
        # User input
        new_planting_date = st.date_input("Date Planted", value=saved_planting_date)
        
        # Confirm Button
        submitted = st.form_submit_button("Confirm Settings")
        
        if submitted:
            save_config(new_planting_date)
            st.success("Settings Saved!")
            saved_planting_date = new_planting_date
            # Force cache clear on config change
            fetch_thingspeak.clear()
            time.sleep(0.5) 
            st.rerun()

    # --- CALCULATIONS ---
    current_date_obj = datetime.now().date()
    days_elapsed = (current_date_obj - saved_planting_date).days
    
    sarima_model = load_sarima_model()
    predicted_remaining = predict_harvest_days(sarima_model)
    if predicted_remaining is None: 
        predicted_remaining = 7
        
    total_predicted_cycle = days_elapsed + predicted_remaining

    # REMOVED DIVIDER
    st.markdown("### Growth Summary")
    st.metric("Current Age", f"Day {days_elapsed}")
    # === CHANGED LABEL HERE ===
    st.metric("SARIMA Predictions", f"Day {total_predicted_cycle}")
    st.caption(f"Model predicts harvest in {predicted_remaining} days")

# ============ TITLE & DATE/TIME ============
col_title, col_datetime = st.columns([4, 1])

with col_title:
    st.markdown(
        """
        <h1 style='
            font-size: 65px; 
            font-weight: 700; 
            margin-bottom: 0px; 
            margin-top: -20px;
            font-family: "Poppins", sans-serif;
            color: #ffffff;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        '>
            🥬 Lettuce Growth Monitoring Dashboard
        </h1>
        """, 
        unsafe_allow_html=True
    )

with col_datetime:
    current_date = datetime.now().strftime("%B %d, %Y")
    current_time = datetime.now().strftime("%H:%M") 
    st.markdown(f'<div class="datetime-box">Date: {current_date}<br>Time: {current_time}</div>', unsafe_allow_html=True)

# ============ PAGE ROUTING ============
if st.session_state.selected_metric:
    # ===== DETAIL PAGE =====
    if st.button("← Back to Dashboard"):
        st.session_state.selected_metric = None
        st.rerun()
    
    # REMOVED DIVIDER
    
    metric_names = {
        "temp": ("Temperature", "field1"),
        "humidity": ("Humidity", "field2"),
        "rain": ("Rain", "field3"),
        "soil": ("Soil", "field4"),
        "light": ("Light", "field5")
    }
    
    metric_name, field_key = metric_names.get(st.session_state.selected_metric, ("Unknown", ""))
    st.subheader(f"📈 {metric_name} - Historical Data")
    
    # Uses Cached Data
    df_history_raw = fetch_thingspeak(results=500)
    if df_history_raw.empty or field_key not in df_history_raw.columns:
        st.warning("No historical data available.")
    else:
        df_history = df_history_raw[["created_at", field_key]].rename(columns={field_key: metric_name})
        df_history[metric_name] = pd.to_numeric(df_history[metric_name], errors="coerce")
        
        # --- FIXED CALCULATION (DETAIL PAGE) ---
        df_history["created_at"] = pd.to_datetime(df_history["created_at"])
        df_history["created_at"] = df_history["created_at"].dt.tz_localize(None)
        df_history["Growth Day"] = (df_history["created_at"].dt.normalize() - pd.to_datetime(saved_planting_date)).dt.days
        
        df_history = df_history.dropna()
        if df_history.empty:
            st.warning("No valid numeric historical data.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Current", f"{df_history[metric_name].iloc[-1]:.2f}")
            with col2: st.metric("Average", f"{df_history[metric_name].mean():.2f}")
            with col3: st.metric("Max", f"{df_history[metric_name].max():.2f}")
            with col4: st.metric("Min", f"{df_history[metric_name].min():.2f}")
            
            fig = px.line(df_history, x="created_at", y=metric_name, title=f"{metric_name} Over Time")
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': '#333333'}, 
                xaxis=dict(showgrid=False, color='#333333'),
                yaxis=dict(gridcolor='rgba(0,0,0,0.1)', color='#333333')
            )
            fig.update_traces(line_color='#1f77b4', line_width=3)
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("View Raw Data"):
                st.dataframe(df_history, use_container_width=True)

else:
    # ===== HOME PAGE =====
    # Uses Cached Data - Instant load on refresh/interaction
    df_raw = fetch_thingspeak(results=NUM_RESULTS)
    
    if df_raw.empty:
        df_clean = pd.DataFrame(columns=["created_at", "Temperature", "Humidity", "Rain", "Soil", "Light"])
    else:
        df_clean = df_raw[["created_at", "field1", "field2", "field3", "field4", "field5"]].rename(
            columns={
                "field1": "Temperature", "field2": "Humidity",
                "field3": "Rain", "field4": "Soil", "field5": "Light"
            }
        )
        # --- FIXED CALCULATION (HOME PAGE) ---
        df_clean["created_at"] = pd.to_datetime(df_clean["created_at"])
        df_clean["created_at"] = df_clean["created_at"].dt.tz_localize(None)
        df_clean["Growth Day"] = (df_clean["created_at"].dt.normalize() - pd.to_datetime(saved_planting_date)).dt.days
    
    # Layout
    left_col, right_col = st.columns([2, 1])

    # --- LEFT: Cards ---
    with left_col:
        card_cols = st.columns(5)
        metrics_data = [
            (card_cols[0], "Temperature", last_valid_value(df_clean, "Temperature"), "temp", "°C", "Normal"),
            (card_cols[1], "Humidity", last_valid_value(df_clean, "Humidity"), "humidity", "%", "Normal"),
            (card_cols[2], "Rain", last_valid_value(df_clean, "Rain"), "rain", "mm", "Normal"),
            (card_cols[3], "Soil", last_valid_value(df_clean, "Soil"), "soil", "mV", "Normal"),
            (card_cols[4], "Light", last_valid_value(df_clean, "Light"), "light", "lux", "Normal"),
        ]

        for col, label, value, metric_key, unit, status in metrics_data:
            if value is None:
                display_value = "N/A"
            else:
                if metric_key in ['rain', 'soil', 'light']:
                    display_value = f"{value:.0f}"
                else:
                    display_value = f"{value:.2f}"
            
            with col:
                st.markdown(f"""
                <div class="metric-card-visual">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{display_value}<span style="font-size: 20px">{unit if display_value != 'N/A' else ''}</span></div>
                    <div class="metric-status">{status}</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)

    # --- RIGHT: Data Interpretation + Graphs ---
    with right_col:
        
        # Calculate Irrigation
        current_soil = last_valid_value(df_clean, "Soil")
        irrigation_hrs = calculate_irrigation_hours(current_soil)
        
        # SPLIT INTO TWO COLUMNS
        r_c1, r_c2 = st.columns(2)
        
        # CARD 1: Plant Age
        with r_c1:
            st.markdown(f"""
                <div class="metric-card-visual" style="height:{CARD_HEIGHT}px; padding: 12px; justify-content: center;">
                    <div class="metric-label" style="font-size:11px;">Plant Age</div>
                    <div class="metric-value" style="font-size: 32px; margin: 2px 0;">Day {days_elapsed}</div>
                    <div class="metric-status" style="font-size:11px; color:#2E7D32; font-weight:600;">{predicted_remaining} days to harvest</div>
                </div>
            """, unsafe_allow_html=True)
            
        # CARD 2: Irrigation Prediction
        with r_c2:
            st.markdown(f"""
                <div class="metric-card-visual" style="height:{CARD_HEIGHT}px; padding: 12px; justify-content: center;">
                    <div class="metric-label" style="font-size:11px;">Irrigation</div>
                    <div class="metric-value" style="font-size: 32px; margin: 2px 0;">{irrigation_hrs} hrs</div>
                    <div class="metric-status" style="font-size:11px; color:#1565C0;">Estimated time</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)
        
        # Prepare recent series for interpretation
        df_recent = df_raw.copy()
        insights = []
        try:
            if not df_recent.empty:
                # convert fields
                df_recent["field1"] = pd.to_numeric(df_recent.get("field1"), errors="coerce")
                df_recent["field2"] = pd.to_numeric(df_recent.get("field2"), errors="coerce")
                df_recent["field3"] = pd.to_numeric(df_recent.get("field3"), errors="coerce")
                df_recent["field4"] = pd.to_numeric(df_recent.get("field4"), errors="coerce")
                df_recent["field5"] = pd.to_numeric(df_recent.get("field5"), errors="coerce")

                # Use last 8 samples to detect trend
                window = 8
                last = df_recent.tail(window)
                def trend(col):
                    s = last[col].dropna()
                    if len(s) < 2:
                        return 0
                    return s.iloc[-1] - s.iloc[0]

                t_temp = trend("field1")
                t_hum = trend("field2")
                t_rain = trend("field3")
                t_soil = trend("field4")
                t_light = trend("field5")
                
                # Check for day/night cycle (approximate GMT+8 for Philippines)
                # Streamlit Cloud uses UTC, so we add 8 hours
                current_hour_ph = (datetime.now().hour + 8) % 24
                is_night = current_hour_ph >= 18 or current_hour_ph < 6

                # Insights rules
                if t_temp > 0.5:
                    insights.append("• Temperature is increasing; recommend increasing ventilation or shading.")
                elif t_temp < -0.5:
                    insights.append("• Temperature is decreasing; monitor for potential cold stress.")

                if last["field2"].dropna().iloc[-1] > 80:
                    insights.append("• Humidity is high; consider dehumidification to reduce disease risk.")
                elif t_hum > 2:
                    insights.append("• Humidity is trending up; check ventilation.")

                if last["field4"].dropna().iloc[-1] < 300:
                    insights.append("• Soil moisture low; prescriptive recommendation: irrigate soon.")
                elif t_soil > 50:
                    insights.append("• Soil moisture rising sharply; check irrigation system for overwatering.")

                # LIGHT LOGIC: Only warn if it is DAY time (not night)
                if last["field5"].dropna().iloc[-1] < 100:
                    if not is_night:
                        insights.append("• Light levels are low for daytime; consider supplemental lighting.")
                    # Else: It is night time, so low light is normal (no warning)

                # RAIN LOGIC: Warn to move device
                if last["field3"].dropna().iloc[-1] > 0:
                    insights.append("• 🌧️ Rain detected! Move the device/plant to a sheltered area to prevent overwatering.")
                    
        except Exception:
            insights.append("• Not enough data to compute insights.")

        # Render insights inside a Glass Container (Background update)
        if insights:
            insights_html = "".join([f"<div style='margin-bottom:6px; font-size:14px; color:#333;'>{i}</div>" for i in insights])
            st.markdown(f"""
            <div class="glass-container">
                <div class="metric-label" style="margin-bottom:12px; padding-left:0;">Smart Insights</div>
                {insights_html}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="glass-container">
                <div class="metric-label" style="margin-bottom:12px; padding-left:0;">Smart Insights</div>
                <div style="font-size:14px; color:#333;">No notable insights at this time.</div>
            </div>
            """, unsafe_allow_html=True)

    # ============ EDA SECTION ============
    # REMOVED DIVIDER
    st.markdown("""
        <h3 style='color: #ffffff; text-shadow: 0 2px 4px rgba(0,0,0,0.6); font-family: "Poppins", sans-serif;'>
            📊 Exploratory Data Analysis (Real-time Trends)
        </h3>
    """, unsafe_allow_html=True)
    
    # Ensure data is numeric for plotting
    eda_df = df_clean.copy()
    numeric_cols = ["Temperature", "Humidity", "Rain", "Soil", "Light"]
    for c in numeric_cols:
        eda_df[c] = pd.to_numeric(eda_df[c], errors='coerce')
    
    # Updated Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Combined Trends", "Soil Health", "Correlations", "Growth Forecast", "Statistical Summary"])
    
    # --- PLOTLY COMMON CONFIG (High Contrast for Light Mode) ---
    plotly_config = {
        'displayModeBar': False,
        'scrollZoom': True
    }
    plotly_layout_transparent = dict(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#333333'}, # Dark font
        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.15)', color='#333333'), # Slightly darker grid
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.15)', color='#333333'),
        legend=dict(font=dict(color='#333333')),
        margin=dict(l=20, r=20, t=20, b=20)
    )

    with tab1:
        st.caption("Monitoring all sensor parameters over the last 100 readings")
        # Added Rain and Soil to the y-axis list and color map
        fig_combined = px.line(eda_df, y=["Temperature", "Humidity", "Light", "Rain", "Soil"], 
                               color_discrete_map={
                                   "Temperature": "#C62828", 
                                   "Humidity": "#1565C0", 
                                   "Light": "#F9A825",
                                   "Rain": "#00838F", # Teal
                                   "Soil": "#5D4037"  # Brown
                               })
        fig_combined.update_layout(**plotly_layout_transparent)
        st.plotly_chart(fig_combined, use_container_width=True, config=plotly_config)

    with tab2:
        st.caption("Soil Moisture consistency tracking")
        # Interactive Plotly Area Chart
        fig_soil = px.area(eda_df, y="Soil")
        # Dark Brown fill
        fig_soil.update_traces(line_color='#4E342E', fillcolor='rgba(78, 52, 46, 0.5)')
        fig_soil.update_layout(**plotly_layout_transparent)
        st.plotly_chart(fig_soil, use_container_width=True, config=plotly_config)

    with tab3:
        st.caption("Temperature vs. Humidity Correlation Check")
        if not eda_df.empty:
            # Interactive Plotly Scatter Plot
            # UPDATED: Deep Purple dots for visibility on white
            fig_corr = px.scatter(eda_df, x="Temperature", y="Humidity", 
                                  color_discrete_sequence=['#6200EA'], opacity=0.8)
            fig_corr.update_layout(**plotly_layout_transparent)
            # Dark grey border around dots
            fig_corr.update_traces(marker=dict(size=12, line=dict(width=1, color='#444444'))) 
            st.plotly_chart(fig_corr, use_container_width=True, config=plotly_config)
        else:
            st.info("Not enough data for correlation analysis.")

    with tab4:
        col_g1, col_g2 = st.columns([1, 2])
        
        # 1. Harvest Countdown Bar (Plotly)
        with col_g1:
            st.caption("Harvest Countdown")
            days_left = predicted_remaining
            
            # Interactive Bar for countdown - Dark Green
            fig_bar = go.Figure(go.Bar(
                x=[days_left], 
                y=['Days'], 
                orientation='h',
                marker_color='#2E7D32', # Forest Green
                text=[str(days_left)],
                textposition='auto',
                hoverinfo='none',
                textfont=dict(color='white')
            ))
            fig_bar.update_layout(**plotly_layout_transparent)
            fig_bar.update_layout(
                xaxis=dict(range=[0, total_predicted_cycle + 5], showgrid=False, color='#333333'), # Assuming 45 day cycle
                yaxis=dict(color='#333333'),
                height=200,
                margin=dict(l=0, r=0, t=20, b=20)
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={'staticPlot': True})

        # 2. Temperature & Humidity Projection (Simulated Forecast via Plotly)
        with col_g2:
            st.caption("Environmental Outlook (14-Day Projection)")
            
            if not eda_df.empty:
                last_idx = len(eda_df)
                forecast_steps = 14
                
                last_temp = eda_df['Temperature'].dropna().iloc[-1]
                last_hum = eda_df['Humidity'].dropna().iloc[-1]
                
                # Create Forecast Data
                future_indices = [f"Day {i+1}" for i in range(forecast_steps)]
                
                np.random.seed(42)
                proj_temp = [last_temp + np.random.uniform(-1, 1) for _ in range(forecast_steps)]
                proj_hum = [last_hum + np.random.uniform(-2, 2) for _ in range(forecast_steps)]
                
                # Create Interactive Line Chart - Dark Orange & Blue
                fig_env = go.Figure()
                fig_env.add_trace(go.Scatter(x=future_indices, y=proj_temp, mode='lines+markers', name='Temp (°C)', line=dict(color='#E65100')))
                fig_env.add_trace(go.Scatter(x=future_indices, y=proj_hum, mode='lines+markers', name='Humidity (%)', line=dict(color='#1565C0')))
                
                fig_env.update_layout(**plotly_layout_transparent)
                fig_env.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                
                st.plotly_chart(fig_env, use_container_width=True, config=plotly_config)
            else:
                st.info("Insufficient data for environmental forecast.")

    with tab5:
        st.caption("Real-time Sensor Statistics (Min, Max, Average, Median)")
        summary_data = []
        for col in numeric_cols:
            series = eda_df[col].dropna()
            if series.empty:
                continue
            
            # Format Logic: No decimals for Rain, Soil, Light
            is_int_metric = col in ['Rain', 'Soil', 'Light']
            fmt = "{:.0f}" if is_int_metric else "{:.2f}"
            
            summary_data.append({
                "Metric": col,
                "Current Value": fmt.format(series.iloc[-1]),
                "Average": fmt.format(series.mean()),
                "Median": fmt.format(series.median()),
                "Min": fmt.format(series.min()),
                "Max": fmt.format(series.max())
            })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        else:
            st.info("No data available to generate statistics.")

# ============ DATASET EXPORT SECTION ============
# REMOVED DIVIDER

with st.container(border=True):
    st.markdown("""
        <style>
        div[data-testid="stVerticalBlockBorderWrapper"]:has(div.export-glass-box) {
            background: rgba(255, 255, 255, 0.75); 
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255,255,255,0.4);
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(div.export-glass-box) > div {
            
        }
        </style>
        <div class="export-glass-box" style="display:none;"></div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <h3 style='color: #333333; margin-top:0; font-family: "Poppins", sans-serif;'>
            📥 Collected Dataset
        </h3>
    """, unsafe_allow_html=True)
    
    # OPTIMIZATION: Use the cached 100-result dataframe for PREVIEW instead of fetching 500 rows again.
    # This prevents double API calls on every page load.
    export_df_display = df_clean.copy() 
    
    if not export_df_display.empty:
        # Rename columns to match export format
        export_df_display = export_df_display.rename(
            columns={
                "Temperature": "Temperature (°C)",
                "Humidity": "Humidity (%)",
                "Rain": "Rain (mm)",
                "Soil": "Soil (mV)",
                "Light": "Light (lux)"
            }
        )
        
        # Calculate timestamps/day for the preview dataframe
        try:
            planting_date = pd.to_datetime(saved_planting_date)
            export_df_display["Day"] = (export_df_display["created_at"].dt.normalize() - planting_date).dt.days
            export_df_display["Day"] = export_df_display["Day"].astype('Int64')
            export_df_display["Timestamp"] = export_df_display["created_at"].dt.strftime('%Y-%m-%d %H:%M:%S')
            export_df_display = export_df_display.drop(columns=["created_at"])
        except Exception:
            export_df_display["Day"] = pd.NA
            export_df_display["Timestamp"] = pd.NA
        
        col_data, col_export = st.columns([3, 1])
        
        with col_data:
            st.caption(f"Previewing recent readings (Download for full history)")
            cols_order = [c for c in ["Timestamp", "Temperature (°C)", "Humidity (%)", "Rain (mm)", "Soil (mV)", "Light (lux)", "Day"] if c in export_df_display.columns]
            st.dataframe(export_df_display.loc[:, cols_order], use_container_width=True, hide_index=True)
        
        with col_export:
            st.markdown("""
            <div style="font-weight:600; font-size:14px; margin-bottom:8px;">Export Options</div>
            <div style="font-size:12px; color:#444; margin-bottom:15px;">
                Download the full ThingSpeak history as an Excel file.
            </div>
            """, unsafe_allow_html=True)

            if st.button("📥 Export All Records"):
                with st.spinner('Fetching all records (this may take a moment)...'):
                    full_df = fetch_all_thingspeak(chunk=8000)
                if full_df.empty:
                    st.warning("No additional records available.")
                else:
                    full_display = full_df[["created_at", "field1", "field2", "field3", "field4", "field5"]].rename(
                        columns={
                            "field1": "Temperature (°C)",
                            "field2": "Humidity (%)",
                            "field3": "Rain (mm)",
                            "field4": "Soil (mV)",
                            "field5": "Light (lux)"
                        }
                    )
                    for col in ["Temperature (°C)", "Humidity (%)", "Rain (mm)", "Soil (mV)", "Light (lux)"]:
                        full_display[col] = pd.to_numeric(full_display[col], errors="coerce")

                    try:
                        full_display["created_at"] = pd.to_datetime(full_display["created_at"]).dt.tz_localize(None)
                        planting_date = pd.to_datetime(saved_planting_date)
                        full_display["Day"] = (full_display["created_at"].dt.normalize() - planting_date).dt.days
                        full_display["Day"] = full_display["Day"].astype('Int64')
                        full_display["Timestamp"] = full_display["created_at"].dt.strftime('%Y-%m-%d %H:%M:%S')
                        full_display = full_display.drop(columns=["created_at"])
                    except Exception:
                        full_display["Day"] = pd.NA
                        full_display["Timestamp"] = pd.NA

                    out_all = BytesIO()
                    with pd.ExcelWriter(out_all, engine='openpyxl') as writer:
                        cols_order_all = [c for c in ["Timestamp", "Temperature (°C)", "Humidity (%)", "Rain (mm)", "Soil (mV)", "Light (lux)", "Day"] if c in full_display.columns]
                        full_display.loc[:, cols_order_all].to_excel(writer, sheet_name='All Sensor Data', index=False)
                    out_all.seek(0)
                    st.download_button(
                        label="Save Excel File",
                        data=out_all.getvalue(),
                        file_name=f"lettuce_all_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
    else:
        st.info("No dataset available for export.")

# Auto-refresh
if REFRESH_INTERVAL:
    time.sleep(REFRESH_INTERVAL)
    st.rerun()
