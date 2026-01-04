import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import plotly.graph_objects as go
from datetime import datetime

# --- 1. PAGE SETUP (Must be the first Streamlit command) ---
st.set_page_config(page_title="SeismicGuard Pro", page_icon="🌋", layout="wide")

# --- 2. USER DATA MANAGEMENT ---
USER_DB = "users.json"

def load_users():
    if not os.path.exists(USER_DB):
        return {"admin": "seismic2024"} # Default admin
    with open(USER_DB, "r") as f:
        return json.load(f)

def save_user(username, password):
    users = load_users()
    if username in users: return False
    users[username] = password
    with open(USER_DB, "w") as f:
        json.dump(users, f)
    return True

# --- 3. CUSTOM CSS FOR A REAL WEBSITE LOOK ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .prediction-card {
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        border: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 20px;
    }
    .main-title { 
        font-size: 3rem; 
        font-weight: 800; 
        text-align: center;
        background: linear-gradient(45deg, #ff4b4b, #00d4ff); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    div[data-testid="stExpander"] { background-color: #161b22; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- 4. SESSION STATE INITIALIZATION ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'user' not in st.session_state:
    st.session_state['user'] = None
if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- 5. LOAD AI MODELS ---
@st.cache_resource
def load_assets():
    try:
        # These paths must match your folder structure
        with open('output/final_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('output/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('output/feature_names.pkl', 'rb') as f:
            feat_names = pickle.load(f)
        return model, scaler, feat_names
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None

model, scaler, feature_names = load_assets()

# --- 6. LOGIN & REGISTRATION PAGE ---
def auth_page():
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.5, 1])
    
    with col2:
        st.markdown('<h1 style="text-align:center;">🌋 SeismicGuard Pro</h1>', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Create Account"])
        
        with tab1:
            user = st.text_input("Username", key="login_user")
            pw = st.text_input("Password", type="password", key="login_pw")
            if st.button("Sign In", use_container_width=True, type="primary"):
                users = load_users()
                if user in users and users[user] == pw:
                    st.session_state['logged_in'] = True
                    st.session_state['user'] = user
                    st.rerun()
                else:
                    st.error("Invalid Username or Password")
        
        with tab2:
            new_user = st.text_input("New Username", key="reg_user")
            new_pw = st.text_input("New Password", type="password", key="reg_pw")
            if st.button("Register Now", use_container_width=True):
                if new_user and new_pw:
                    if save_user(new_user, new_pw):
                        st.success("Account created! Please switch to Login tab.")
                    else:
                        st.error("Username already exists.")
                else:
                    st.warning("Please enter both username and password.")

# --- 7. MAIN DASHBOARD PAGE ---
def main_dashboard():
    # Sidebar Navigation
    st.sidebar.markdown(f"### 🛡️ System Access: {st.session_state['user'].upper()}")
    nav = st.sidebar.radio("Navigation", ["Prediction Center", "Analysis History", "System Status"])
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state['logged_in'] = False
        st.rerun()

    if nav == "Prediction Center":
        st.markdown('<p class="main-title">Seismic Impact Intelligence</p>', unsafe_allow_html=True)
        
        # Input Section
        with st.container(border=True):
            st.subheader("⌨️ Seismic Parameter Input")
            c1, c2, c3 = st.columns(3)
            mag = c1.number_input("Magnitude (Richter)", 0.0, 10.0, 7.2)
            dep = c1.number_input("Depth (km)", 0.0, 700.0, 15.0)
            mmi = c2.number_input("MMI Intensity", 1.0, 12.0, 8.0)
            cdi = c2.number_input("CDI Intensity", 1.0, 12.0, 7.5)
            sig = c3.number_input("Significance Score", 0, 1000, 600)
            st.write("")
            analyze = st.button("🚀 RUN AI ANALYSIS", use_container_width=True, type="primary")

        if analyze:
            if model is None:
                st.error("Model not loaded. Check your 'output' folder.")
                return

            # ML Logic
            input_df = pd.DataFrame([[mag, dep, cdi, mmi, sig]], columns=feature_names)
            scaled = scaler.transform(input_df)
            pred_idx = model.predict(scaled)[0]
            probs = model.predict_proba(scaled)[0]

            # UI Mapping
            levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
            colors = ["#28a745", "#ffc107", "#fd7e14", "#dc3545"]
            descriptions = [
                "Minor shaking. No damage reported. Routine monitoring.",
                "Moderate shaking. Potential minor damage to old buildings.",
                "Severe shaking. Structural damage likely. High alert.",
                "Disastrous impact. Extreme damage expected. Emergency response active."
            ]
            
            # Save to history
            st.session_state['history'].insert(0, {
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Magnitude": mag,
                "Level": levels[pred_idx]
            })

            # RESULTS UI
            st.markdown("---")
            res_col1, res_col2 = st.columns([1.5, 1])
            
            with res_col1:
                st.markdown(f"""
                    <div class="prediction-card" style="background-color:{colors[pred_idx]};">
                        <p style="margin:0; text-transform:uppercase; font-weight:bold; opacity:0.8;">Predicted Impact Level</p>
                        <h1 style="font-size:5.5rem; margin:10px 0; font-weight:900; line-height:1;">{levels[pred_idx]}</h1>
                        <h3 style="margin-top:0; opacity:0.9;">{descriptions[pred_idx]}</h3>
                        <hr style="border:0.5px solid rgba(255,255,255,0.3)">
                        <p style="font-size:1.3rem;">AI Confidence Score: <b>{probs[pred_idx]*100:.1f}%</b></p>
                    </div>
                """, unsafe_allow_html=True)

            with res_col2:
                # Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = probs[pred_idx] * 100,
                    title = {'text': "Confidence Gauge", 'font': {'size': 20, 'color': "white"}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickcolor': "white"},
                        'bar': {'color': "white"},
                        'bgcolor': "rgba(0,0,0,0.1)",
                        'steps': [{'range': [0, 100], 'color': colors[pred_idx]}]
                    }
                ))
                fig.update_layout(height=300, margin=dict(t=50, b=0, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
                st.plotly_chart(fig, use_container_width=True)

    elif nav == "Analysis History":
        st.title("📜 Past Event Analysis")
        if st.session_state['history']:
            st.dataframe(pd.DataFrame(st.session_state['history']), use_container_width=True)
        else:
            st.info("No records found in current session.")

    elif nav == "System Status":
        st.title("📡 System Health")
        st.success("Model Status: Online")
        st.success("Database: Connected")
        st.info("Version: 2.1.0 (Pro Edition)")

# --- 8. RUN APP ---
if st.session_state['logged_in']:
    main_dashboard()
else:
    auth_page()