import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

# Load model
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE_DIR, 'data', 'models', 'win_prob_model.pkl'))

# Page config
st.set_page_config(
    page_title="IPL Win Probability Engine",
    page_icon="🏏",
    layout="wide"
)

st.title("🏏 IPL Win Probability Engine")
st.caption("Ball-by-ball win prediction · LightGBM · Trained on 1241 IPL matches (2008–2024)")

st.divider()

# Input section
st.subheader("Enter Match State")

col1, col2, col3 = st.columns(3)

with col1:
    inning = st.radio("Innings", [1, 2], horizontal=True)
    over = st.slider("Current Over", 0, 19, 10)
    ball = st.slider("Ball in Over", 1, 6, 3)

with col2:
    runs = st.number_input("Runs Scored", 0, 300, 80)
    wickets = st.number_input("Wickets Fallen", 0, 10, 2)

with col3:
    target = st.number_input("Target (2nd innings only)", 0, 300, 160)
    toss_advantage = st.radio("Did batting team win toss?", [1, 0],
                               format_func=lambda x: "Yes" if x == 1 else "No",
                               horizontal=True)

# Calculate features
legal_ball = over * 6 + ball
balls_remaining = 120 - legal_ball
wickets_remaining = 10 - wickets
crr = (runs / (legal_ball / 6)) if legal_ball > 0 else 0

if inning == 2:
    runs_required = max(target - runs, 0)
    rrr = (runs_required / (balls_remaining / 6)) if balls_remaining > 0 else 99
    rr_diff = crr - rrr
else:
    runs_required = 0
    rrr = 0
    rr_diff = 0

# Determine phase
if over <= 5:
    phase = 0
elif over <= 14:
    phase = 1
else:
    phase = 2

# Build feature row
features = pd.DataFrame([{
    'inning': inning,
    'over': over,
    'legal_ball': legal_ball,
    'balls_remaining': balls_remaining,
    'wickets_remaining': wickets_remaining,
    'runs_so_far': runs,
    'current_run_rate': crr,
    'runs_required': runs_required,
    'required_run_rate': rrr,
    'run_rate_diff': rr_diff,
    'phase': phase,
    'toss_advantage': toss_advantage
}])

# Predict
prob = model.predict_proba(features)[0][1]

st.divider()

# Display result
st.subheader("Win Probability")

col_gauge, col_stats = st.columns([1, 1])

with col_gauge:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={'suffix': '%', 'font': {'size': 48}},
        title={'text': "Batting Team Wins", 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#1D9E75" if prob >= 0.5 else "#D85A30"},
            'steps': [
                {'range': [0, 30], 'color': '#FAECE7'},
                {'range': [30, 70], 'color': '#FFF8E8'},
                {'range': [70, 100], 'color': '#E1F5EE'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

with col_stats:
    st.metric("Current Run Rate", f"{crr:.2f}")
    st.metric("Required Run Rate", f"{rrr:.2f}" if inning == 2 else "N/A")
    st.metric("Balls Remaining", balls_remaining)
    st.metric("Wickets Remaining", wickets_remaining)
    if inning == 2:
        st.metric("Runs Required", runs_required)

st.divider()
st.caption("Model: LightGBM · Accuracy: 68.2% · Brier Score: 0.197 · Data: Cricsheet.org")