"""
Logistics Warehouse Robot — Spatial MDP Simulation (Streamlit)
==============================================================
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
# 1. Load Optimization Results & MDP Definition
# ---------------------------------------------------------------------
@st.cache_data
def load_mdp_data():
    try:
        opt_df = pd.read_csv('optimal_value_function.csv')
        policy = dict(zip(opt_df['State'], opt_df['Optimal_Action']))
        values = dict(zip(opt_df['State'], opt_df['Optimal_Value']))
    except Exception:
        policy = {'Too-Close': 'Sharp-Right-Turn', 'Ideal': 'Move-Forward', 'Too-Far': 'Slight-Left-Turn'}
        values = {'Too-Close': 100.0, 'Ideal': 100.0, 'Too-Far': 100.0}
    return policy, values

POLICY, VALUES = load_mdp_data()

STATES = ['Too-Close', 'Ideal', 'Too-Far']

# Empirical transition probabilities
T = {
    'Too-Close': {'Move-Forward': [0.622, 0.333, 0.044], 'Slight-Right-Turn': [0.955, 0.042, 0.002],
                  'Sharp-Right-Turn': [0.968, 0.027, 0.005], 'Slight-Left-Turn': [0.333, 0.333, 0.333]},
    'Ideal':     {'Move-Forward': [0.015, 0.976, 0.009], 'Slight-Right-Turn': [0.333, 0.333, 0.333],
                  'Sharp-Right-Turn': [0.024, 0.961, 0.014], 'Slight-Left-Turn': [0.333, 0.333, 0.333]},
    'Too-Far':   {'Move-Forward': [0.0, 0.5, 0.5], 'Slight-Right-Turn': [0.333, 0.333, 0.333],
                  'Sharp-Right-Turn': [0.034, 0.188, 0.778], 'Slight-Left-Turn': [0.003, 0.055, 0.942]},
}

R = {
    'Too-Close': {'Move-Forward': -10, 'Slight-Right-Turn': 5, 'Sharp-Right-Turn': 10, 'Slight-Left-Turn': -10},
    'Ideal':     {'Move-Forward': 10, 'Slight-Right-Turn': 2, 'Sharp-Right-Turn': -5, 'Slight-Left-Turn': 2},
    'Too-Far':   {'Move-Forward': -2, 'Slight-Right-Turn': -10, 'Sharp-Right-Turn': -10, 'Slight-Left-Turn': 10},
}

MOVES = {
    'Move-Forward':      {'turn': 0.0,   'step': 0.35},
    'Slight-Right-Turn': {'turn': -0.07, 'step': 0.30},
    'Sharp-Right-Turn':  {'turn': -0.20, 'step': 0.24},
    'Slight-Left-Turn':  {'turn': 0.07,  'step': 0.30},
}

# ---------------------------------------------------------------------
# 2. 5-Corner Polygon Room (~3m x 5.5m) Based on Analysis
# ---------------------------------------------------------------------
ENTRANCE = (1.5, 1.2)
EXIT = (3.5, 5.0)  # Located at the left-wall opening / gap

def get_polygon_walls():
    """Defines a closed 5-corner polygon room (approx 3m x 5.5m) with a wall opening."""
    # 5 vertices forming the perimeter
    pts = [
        (1.0, 1.0),   # Corner 1 (Bottom-Left)
        (4.0, 1.0),   # Corner 2 (Bottom-Right)
        (5.2, 3.5),   # Corner 3 (Mid-Right)
        (4.2, 5.8),   # Corner 4 (Top-Right)
        (1.2, 5.0)    # Corner 5 (Top-Left)
    ]
    
    walls = []
    n = len(pts)
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        
        # Introduce a wall gap / opening on the left wall segment (e.g., segment between Corner 5 and Corner 1)
        if i == n - 1:
            # Split wall to create a doorway/opening
            mid_p = (p1[0] * 0.5 + p2[0] * 0.5, p1[1] * 0.5 + p2[1] * 0.5)
            # Add wall before opening and wall after opening
            walls.append((p1, (mid_p[0] - 0.3, mid_p[1] - 0.3)))
            walls.append(((mid_p[0] + 0.3, mid_p[1] + 0.3), p2))
        else:
            walls.append((p1, p2))
            
    return walls

# ---------------------------------------------------------------------
# 3. Sidebar Controls (Shared Threshold Parameters)
# ---------------------------------------------------------------------
st.sidebar.header("Shared Model Parameters")
too_close_thresh = st.sidebar.slider("Too-Close Threshold (m)", 0.3, 0.7, 0.53, 0.01,
                                     help="Aligned with data 33rd percentile (~0.53m)")
too_far_thresh = st.sidebar.slider("Too-Far Threshold (m)", 0.7, 1.2, 0.71, 0.01,
                                   help="Aligned with data 66th percentile (~0.71m)")
front_safety = st.sidebar.slider("Front Collision Safety (m)", 0.2, 0.6, 0.38, 0.05)
sensor_range = 6.0

def classify_state(sd_left):
    if sd_left < too_close_thresh:
        return 'Too-Close'
    elif sd_left > too_far_thresh:
        return 'Too-Far'
    else:
        return 'Ideal'

# ---------------------------------------------------------------------
# 4. Ray Casting & Collision Detection
# ---------------------------------------------------------------------
def ray_hit(walls, ox, oy, dx, dy):
    best = sensor_range
    for (ax, ay), (bx, by) in walls:
        sx, sy = bx - ax, by - ay
        denom = dx * sy - dy * sx
        if abs(denom) < 1e-9:
            continue
        qpx, qpy = ax - ox, ay - oy
        t = (qpx * sy - qpy * sx) / denom
        u = (qpx * dy - qpy * dx) / denom
        if t >= 0 and 0 <= u <= 1 and t < best:
            best = t
    return best

def get_sensors(walls, x, y, theta):
    dx, dy = math.cos(theta), math.sin(theta)
    ldx, ldy = -math.sin(theta), math.cos(theta) # Left sensor (+90 deg CCW)
    front = ray_hit(walls, x, y, dx, dy)
    left = ray_hit(walls, x, y, ldx, ldy)
    return front, left, (dx, dy), (ldx, ldy)

def check_collision(walls, x, y, radius=0.18):
    for (ax, ay), (bx, by) in walls:
        px, py = bx - ax, by - ay
        length_sq = px**2 + py**2
        if length_sq == 0:
            dist = math.hypot(x - ax, y - ay)
        else:
            t = max(0, min(1, ((x - ax) * px + (y - ay) * py) / length_sq))
            proj_x, proj_y = ax + t * px, ay + t * py
            dist = math.hypot(x - proj_x, y - proj_y)
        if dist < radius:
            return True
    return False

# ---------------------------------------------------------------------
# 5. Session State Initialization
# ---------------------------------------------------------------------
def init_simulation():
    st.session_state.walls = get_polygon_walls()
    st.session_state.robot = {'x': ENTRANCE[0], 'y': ENTRANCE[1], 'theta': 0.0}
    st.session_state.state = 'Ideal'
    st.session_state.step = 0
    st.session_state.cum_reward = 0
    st.session_state.safety_count = 0
    st.session_state.trail = [{'x': ENTRANCE[0], 'y': ENTRANCE[1]}]
    st.session_state.log = []
    st.session_state.running = False
    st.session_state.reached_exit = False
    st.session_state.last_action = '—'
    st.session_state.last_reward = None
    st.session_state.last_sensors = (None, None)

if 'robot' not in st.session_state:
    init_simulation()

def do_step():
    ss = st.session_state
    if ss.reached_exit:
        return

    front, left, _, _ = get_sensors(ss.walls, ss.robot['x'], ss.robot['y'], ss.robot['theta'])
    ss.state = classify_state(left)

    action = POLICY.get(ss.state, 'Move-Forward')
    overridden = False
    if front < front_safety:
        action = 'Sharp-Right-Turn'
        overridden = True
        ss.safety_count += 1

    reward = R[ss.state][action]
    ss.cum_reward += reward
    ss.step += 1

    mv = MOVES[action]
    new_theta = ss.robot['theta'] + mv['turn']
    new_x = ss.robot['x'] + math.cos(new_theta) * mv['step']
    new_y = ss.robot['y'] + math.sin(new_theta) * mv['step']

    if not check_collision(ss.walls, new_x, new_y, radius=0.18):
        ss.robot['x'] = new_x
        ss.robot['y'] = new_y
        ss.robot['theta'] = new_theta
    else:
        ss.robot['theta'] -= 0.5  # smooth turn away from wall

    ss.trail.append({'x': ss.robot['x'], 'y': ss.robot['y']})

    if math.hypot(ss.robot['x'] - EXIT[0], ss.robot['y'] - EXIT[1]) < 0.6:
        ss.reached_exit = True
        ss.log.append(f"🎉 Robot successfully reached the Exit opening!")

    ss.last_action = action + (' (safety)' if overridden else '')
    ss.last_reward = reward
    ss.last_sensors = (left, front)

    tag = "SAFETY" if overridden else ss.state
    ss.log.append(f"#{ss.step:03d} | State: {tag:<10s} | Action: {action:<18s} | Reward: {reward:+d}")
    if len(ss.log) > 100:
        ss.log.pop(0)

# ---------------------------------------------------------------------
# 6. Main UI Layout
# ---------------------------------------------------------------------
st.title("🤖 5-Corner Polygon Room — Robot Navigation MDP")
st.caption("Refactored based on empirical data analysis: single closed loop (~3m × 5.5m) with 5 corners, clockwise wall-following, and a left-wall opening.")

col_plot, col_side = st.columns([2.2, 1.0])

with col_plot:
    b1, b2, b3, b4 = st.columns([1, 1, 1, 2])
    if b1.button("▶ Play / Pause"):
        st.session_state.running = not st.session_state.running
    if b2.button("Step"):
        do_step()
    if b3.button("Reset Robot"):
        init_simulation()
    speed = b4.slider("Simulation Speed (s/step)", 0.02, 0.3, 0.05, 0.02)
    plot_ph = st.empty()

with col_side:
    st.subheader("Navigation Status")
    m1, m2 = st.columns(2)
    m1.metric("Step", st.session_state.step)
    m2.metric("Total Reward", st.session_state.cum_reward)
    
    if st.session_state.reached_exit:
        st.success("Target Exit Reached!")

    st.write(f"**Current State:** `{st.session_state.state}`")
    if st.session_state.last_sensors[0] is not None:
        st.write(f"Left / Front Sensor: `{st.session_state.last_sensors[0]:.2f}m` / `{st.session_state.last_sensors[1]:.2f}m`")
    st.write(f"Selected Action: `{st.session_state.last_action}`")
    st.write(f"Safety Overrides: `{st.session_state.safety_count}`")

    st.subheader("Event Log")
    st.code("\n".join(st.session_state.log[-12:]) if st.session_state.log else "—", language=None)

# ---------------------------------------------------------------------
# 7. Render Visualization
# ---------------------------------------------------------------------
def render_room():
    ss = st.session_state
    fig, ax = plt.subplots(figsize=(7, 5.0))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#ffffff')
    ax.set_xlim(0.0, 6.0)
    ax.set_ylim(0.0, 6.5)
    ax.set_aspect('equal')

    ax.grid(True, linestyle=':', alpha=0.5, color='#cccccc')

    # Draw polygon walls
    for (ax0, ay0), (bx0, by0) in ss.walls:
        ax.plot([ax0, bx0], [ay0, by0], color="#2b2d42", linewidth=5, solid_capstyle='round', zorder=2)

    # Draw Entrance & Exit markers
    ax.scatter([ENTRANCE[0]], [ENTRANCE[1]], color="#2a9d8f", s=160, marker='s', zorder=3, label="Entrance")
    ax.text(ENTRANCE[0], ENTRANCE[1] - 0.35, "Entrance", color="#2a9d8f", fontweight='bold', ha='center', fontsize=9)

    ax.scatter([EXIT[0]], [EXIT[1]], color="#e76f51", s=160, marker='*', zorder=3, label="Exit (Opening)")
    ax.text(EXIT[0], EXIT[1] + 0.35, "Exit", color="#e76f51", fontweight='bold', ha='center', fontsize=9)

    # Draw sensor rays
    rx, ry = ss.robot['x'], ss.robot['y']
    front, left, fdir, ldir = get_sensors(ss.walls, rx, ry, ss.robot['theta'])
    ax.plot([rx, rx + ldir[0] * left], [ry, ry + ldir[1] * left], color="#adb5bd", linestyle='--', linewidth=1.2, zorder=3)
    ax.plot([rx, rx + fdir[0] * front], [ry, ry + fdir[1] * front], color="#adb5bd", linestyle='--', linewidth=1.2, zorder=3)

    # Draw Robot Path Trace
    if len(ss.trail) > 1:
        xs = [p['x'] for p in ss.trail]
        ys = [p['y'] for p in ss.trail]
        ax.plot(xs, ys, color="#3a86ff", linewidth=2.5, label="Robot Path Loop", zorder=4)

    # Draw Robot Position & Heading
    ax.scatter([rx], [ry], color="#f72585", s=120, zorder=5, edgecolors='black', linewidths=1.2, label="Robot")
    ax.arrow(rx, ry, math.cos(ss.robot['theta'])*0.35, math.sin(ss.robot['theta'])*0.35, 
             head_width=0.18, head_length=0.2, fc='#f72585', ec='black', zorder=6)

    ax.legend(loc='upper right', framealpha=0.9, fontsize=8)
    ax.set_title("5-Corner Polygon Room & Clockwise Wall-Following Trace", fontsize=11, fontweight='bold', pad=10)
    
    plot_ph.pyplot(fig, use_container_width=True)
    plt.close(fig)

render_room()

if st.session_state.running and not st.session_state.reached_exit:
    for _ in range(15):
        if not st.session_state.running or st.session_state.reached_exit:
            break
        do_step()
        render_room()
        time.sleep(speed)
    st.rerun()
