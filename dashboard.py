import streamlit as st
import pandas as pd
import os
import requests
from streamlit_autorefresh import st_autorefresh

st.set_page_config(layout="wide", page_title="CrowdShield Dashboard")
st.title("🚨 CrowdShield AI - Monitoring Dashboard")

# Refresh every 1.5 seconds for real-time feel
st_autorefresh(interval=1500, key="data_refresh")

video_path = r"D:\CrowdShield_AI\crowd.mp4"

# ---------- DATA LOADER ----------
def load_data():
    csv_file = "crowd_log.csv"
    if not os.path.exists(csv_file):
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_file, on_bad_lines="skip")
        df.columns = df.columns.str.strip() 
        df = df.dropna(how="all")
        return df
    except Exception as e:
        return pd.DataFrame()

df = load_data()
latest = df.iloc[-1] if not df.empty else None

# ---------- LOCAL LLM FUNCTION (LLAMA 3) ----------
def get_ai_advisory(zone, density, pressure):
    prompt = f"""
    You are an expert crowd control AI. A critical situation has been detected.
    Location: {zone}
    People Count: {density}
    Pressure Level: {pressure} (Scale > 5 is dangerous)
    
    Provide a fast, 3-bullet point actionable evacuation and mitigation plan for the ground security team. Keep it under 50 words. Be direct and authoritative.
    """
    
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3",  # Tera chosen Llama 3 model
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
        return response.json()['response']
    except requests.exceptions.ConnectionError:
        return "⚠️ LLM Backend Error: Ollama is not running. Please run 'ollama serve' or 'ollama run llama3' in terminal."
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# ==========================================
# 📺 TOP ROW: VIDEO (LEFT) & ALERTS (RIGHT)
# ==========================================
vid_col, alert_col = st.columns([2, 1])

with vid_col:
    st.subheader("📹 Live Camera Feed")
    if os.path.exists(video_path):
        st.video(video_path)
    else:
        st.error("❌ Video file not found! Check path.")

with alert_col:
    st.subheader("⚠️ System Status & Alerts")
    
    if latest is not None:
        risk_label = str(latest.get("risk_label", "N/A")).strip().upper()
        current_zone = str(latest.get("gate_id", "Unknown Zone"))
        current_pressure = round(float(latest.get("pressure", 0)), 2)
        current_count = int(latest.get("people_count", 0))
        
        # Big visual alert box & AI Advisory Trigger
        if "DANGER" in risk_label or "CRITICAL" in risk_label:
            st.error(f"🚨 **ACTIVE CROWD ALERT IN {current_zone}**\n\nImmediate mitigation required at bottleneck points.")
            
            st.markdown("---")
            st.subheader("🧠 GenAI Advisory Board")
            
            # Button dabane par LLM call hoga
            if st.button("Generate Mitigation Plan (LLaMA 3)"):
                with st.spinner("AI is analyzing spatial data..."):
                    advice = get_ai_advisory(current_zone, current_count, current_pressure)
                    st.info(f"**Action Plan:**\n{advice}")

        elif "SURGE" in risk_label or "CROWDING" in risk_label:
            st.warning(f"⚠️ **DENSITY SURGE IN {current_zone}**\n\nMonitor crowd flow closely.")
        else:
            st.success("✅ **SYSTEM STABLE**\n\nCrowd movement is currently normal.")
            
        if "DANGER" not in risk_label and "CRITICAL" not in risk_label:
            st.info("💡 **AI Recommendation Engine:** System is monitoring optical flow and local density grids. No immediate action required.")
    else:
        st.info("⏳ Waiting for AI Vision Engine to send data...")

st.markdown("---") 

# ==========================================
# 📊 MIDDLE ROW: METRICS
# ==========================================
st.subheader("📊 Live Gate Metrics")

if latest is not None:
    def safe(val):
        try:
            return round(float(val), 2)
        except:
            return 0.0

    m1, m2, m3, m4, m5, m6 = st.columns(6)

    with m1:
        st.metric("Critical Zone", current_zone)
    with m2:
        st.metric("People Count", current_count)
    with m3:
        st.metric("Velocity", safe(latest.get("avg_velocity", 0)))
    with m4:
        st.metric("Pressure", current_pressure)
    with m5:
        st.metric("Predicted Pressure", safe(latest.get("predicted_pressure", 0)))
    with m6:
        st.metric("Risk Status", risk_label)

# ==========================================
# 📈 BOTTOM ROW: CHART
# ==========================================
if not df.empty and "pressure" in df.columns:
    st.subheader("📈 Pressure Trend Analysis")
    clean_pressure = pd.to_numeric(df["pressure"], errors="coerce").dropna()
    
    if not clean_pressure.empty:
        st.line_chart(clean_pressure.tail(100))