"""
Logistics Warehouse Robot — Data-Driven MDP Simulation (Streamlit)
==================================================================
Run with:
    pip install streamlit numpy matplotlib pandas
    streamlit run wall_following_streamlit.py
"""

import time
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Logistics Warehouse MDP - Robot Navigation", layout="wide")

# ---------------------------------------------------------------------
# 1. Load Dataset & Optimization Results
# ---------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('sensor_readings_4.csv', header=None, names=['SD_front', 'SD_left', 'SD_right', 'SD_back', 'Class'])
    except Exception:
        # Fallback dummy data if file is missing
        df = pd.DataFrame({
            'SD_front': [1.5]*100, 'SD_left': [0.6]*100, 'SD_right': [2.0]*100, 'SD_back': [1.0]*100,
            'Class': ['Move-Forward']*100
        })
    
    try:
        opt_df = pd.read_csv('optimal_value_function.csv')
        policy = dict(zip(opt_df['State'], opt_df['Optimal_Action']))
        values = dict(zip(opt_df['State'], opt_df['Optimal_Value']))
    except Exception:
        policy = {'Too-Close': 'Sharp-Right-Turn', 'Ideal': 'Move-Forward', 'Too-Far': 'Slight-Left-Turn'}
        values = {'Too-Close': 100.0, 'Ideal': 100.0, 'Too-Far': 100.0}
        
    return df, policy, values

DATA, POLICY, VALUES = load_data()

R = {
    'Too-Close': {'Move-Forward': -10, 'Slight-Right-Turn': 5, 'Sharp-Right-Turn': 10, 'Slight-Left-Turn': -10},
    'Ideal':     {'Move-Forward': 10, 'Slight-Right-Turn': 2, 'Sharp-Right-Turn': -5, 'Slight-Left-Turn': 2},
    'Too-Far':   {'Move-Forward': -2, 'Slight-Right-Turn': -10, 'Sharp-Right-Turn': -10, 'Slight-Left-Turn': 10},
}

# Action kinematic mapping for dead-reckoning reconstruction (~5 corners, ~6 laps)
ACTION_STEPS = {
    'Move-Forward':      {'turn': 0.0,   'step': 0.12},
    'Slight-Right-Turn': {'turn': -0.09, 'step': 0.10},
    'Sharp-Right-Turn':  {'turn': -0.25, 'step': 0.08},
    'Slight-Left-Turn':  {'turn': 0.09,  'step': 0.10},
}

# ---------------------------------------------------------------------
# 2. Sidebar Controls
# ---------------------------------------------------------------------
st.sidebar.header("Shared Model Parameters")
too_close_thresh = st.sidebar.slider("Too-Close Threshold (m)", 0.3, 0.7, 0.53, 0.01,
                                     help="Aligned with data 33rd percentile (~0.53m)")
too_far_thresh = st.sidebar.slider("Too-Far Threshold (m)", 0.7, 1.2, 0.71, 0.01,
                                   help="Aligned with data 66th percentile (~0.71m)")
max_rows = st.sidebar.slider("Simulation Dataset Rows", 500, len(DATA), 2000, 100)

def classify_state(sd_left):
    if sd_left < too_close_thresh:
        return 'Too-Close'
    elif sd_left > too_far_thresh:
        return 'Too-Far'
    else:
        return 'Ideal'

# ---------------------------------------------------------------------
# 3. Session State Initialization
# ---------------------------------------------------------------------
def init_simulation():
    st.session_state.idx = 0
    st.session_state.x = 3.0
    st.session_state.y = 3.0
    st.session_state.theta = 0.0
    st.session_state.cum_reward = 0
    st.session_state.trail = [{'x': 3.0, 'y': 3.0}]
    st.session_state.wall_points = []
    st.session_state.log = []
    st.session_state.running = False
    st.session_state.reached_end = False

if 'idx' not in st.session_state:
    init_simulation()

def do_step():
    ss = st.session_state
    if ss.idx >= max_rows:
        ss.reached_end = True
        ss.running = False
        return

    row = DATA.iloc[ss.idx]
    sd_front, sd_left, sd_right, sd_back = row['SD_front'], row['SD_left'], row['SD_right'], row['SD_back']
    dataset_action = row['Class']

    state = classify_state(sd_left)
    action = POLICY.get(state, dataset_action)
    if action not in R[state]:
        action = 'Move-Forward'

    reward = R[state][action]
    ss.cum_reward += reward

    # Update pose using dead-reckoning kinematics based on action label
    mv = ACTION_STEPS.get(action, {'turn': 0.0, 'step': 0.1})
    ss.theta += mv['turn']
    ss.x += math.cos(ss.theta) * mv['step']
    ss.y += math.sin(ss.theta) * mv['step']

    ss.trail.append({'x': ss.x, 'y': ss.y})

    # Estimate wall point from left sensor reading for environment reconstruction plot
    wall_x = ss.x + math.cos(ss.theta + math.pi/2) * sd_left
    wall_y = ss.y + math.sin(ss.theta + math.pi/2) * sd_left
    ss.wall_points.append({'x': wall_x, 'y': wall_y, 'opening': sd_left > 1.2})

    ss.log.append(f"Row #{ss.idx:04d} | State: {state:<10s} | Action: {action:<18s} | Left: {sd_left:.2f}m")
    if len(ss.log) > 100:
        ss.log.pop(0)

    ss.idx += 1

# ---------------------------------------------------------------------
# 4. Main UI Layout
# ---------------------------------------------------------------------
st.title("🤖 Data-Driven Warehouse Robot Navigation")
st.caption("Reconstructing robot trajectory and mapped loop directly from `sensor_readings_4.csv` data and MDP policy evaluation.")

col_plot, col_side = st.columns([2.2, 1.0])

with col_plot:
    b1, b2, b3, b4 = st.columns([1, 1, 1, 2])
    if b1.button("▶ Play / Pause"):
        st.session_state.running = not st.session_state.running
    if b2.button("Step"):
        do_step()
    if b3.button("Reset Simulation"):
        init_simulation()
    speed = b4.slider("Simulation Speed (s/step)", 0.01, 0.2, 0.03, 0.01)
    plot_ph = st.empty()

with col_side:
    st.subheader("Navigation Metrics")
    m1, m2 = st.columns(2)
    m1.metric("Dataset Row", f"{st.session_state.idx} / {max_rows}")
    m2.metric("Total Reward", st.session_state.cum_reward)
    
    if st.session_state.reached_end:
        st.success("Simulation Reached Dataset Limit!")

    current_state = classify_state(DATA.iloc[min(st.session_state.idx, len(DATA)-1)]['SD_left'])
    st.write(f"**Current State:** `{current_state}`")
    st.write(f"Left Sensor: `{DATA.iloc[min(st.session_state.idx, len(DATA)-1)]['SD_left']:.2f}m`")
    st.write(f"Front Sensor: `{DATA.iloc[min(st.session_state.idx, len(DATA)-1)]['SD_front']:.2f}m`")

    st.subheader("Event Log")
    st.code("\n".join(st.session_state.log[-10:]) if st.session_state.log else "—", language=None)

# ---------------------------------------------------------------------
# 5. Render Visualization
# ---------------------------------------------------------------------
def render_plot():
    ss = st.session_state
    fig, ax = plt.subplots(figsize=(7, 5.0))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#ffffff')
    ax.set_aspect('equal')

    ax.grid(True, linestyle=':', alpha=0.5, color='#cccccc')

    # Plot inferred wall points from sensor data
    if ss.wall_points:
        wx = [p['x'] for p in ss.wall_points if not p['opening']]
        wy = [p['y'] for p in ss.wall_points if not p['opening']]
        ox = [p['x'] for p in ss.wall_points if p['opening']]
        oy = [p['y'] for p in ss.wall_points if p['opening']]
        
        ax.scatter(wx, wy, color="#adb5bd", s=6, alpha=0.5, label="Mapped Walls")
        if ox:
            ax.scatter(ox, oy, color="#e76f51", s=25, label="Wall Openings (Doorways)")

    # Plot Robot Trail
    if len(ss.trail) > 1:
        tx = [p['x'] for p in ss.trail]
        ty = [p['y'] for p in ss.trail]
        ax.plot(tx, ty, color="#3a86ff", linewidth=2.5, label="Robot Trajectory Loop", zorder=3)

    # Plot Robot Position
    ax.scatter([ss.x], [ss.y], color="#f72585", s=130, zorder=4, edgecolors='black', linewidths=1.2, label="Robot")
    ax.arrow(ss.x, ss.y, math.cos(ss.theta)*0.3, math.sin(ss.theta)*0.3, 
             head_width=0.15, head_length=0.18, fc='#f72585', ec='black', zorder=5)

    ax.legend(loc='upper right', framealpha=0.9, fontsize=8)
    ax.set_title("Data-Driven Robot Trajectory & Wall Reconstruction (~6 Laps)", fontsize=11, fontweight='bold', pad=10)
    
    plot_ph.pyplot(fig, use_container_width=True)
    plt.close(fig)

render_plot()

if st.session_state.running and not st.session_state.reached_end:
    for _ in range(15):
        if not st.session_state.running or st.session_state.reached_end:
            break
        do_step()
        render_plot()
        time.sleep(speed)
    st.rerun()
