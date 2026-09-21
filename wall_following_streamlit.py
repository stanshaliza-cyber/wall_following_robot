"""
Wall-Following Robot — Spatial MDP Simulation (Streamlit)
==========================================================
Run with:
    pip install streamlit numpy matplotlib
    streamlit run wall_following_streamlit.py

The robot ray-casts a left sensor and a front sensor against real walls.
The MDP policy (trained in wall_following_mdp.py) steers from the left
sensor reading; a front-proximity check only overrides at corners, so you
can watch the policy hug the wall for a full lap without a collision.
"""

import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Wall-Following Robot — MDP Simulation", layout="wide")

# ---------------------------------------------------------------------
# MDP definition (from wall_following_mdp.py)
# ---------------------------------------------------------------------
STATES = ['Too-Close', 'Ideal', 'Too-Far']
COLORS = {'Too-Close': '#f2745a', 'Ideal': '#4fd18b', 'Too-Far': '#5b9df2'}

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
POLICY = {'Too-Close': 'Sharp-Right-Turn', 'Ideal': 'Move-Forward', 'Too-Far': 'Slight-Left-Turn'}

# turn deltas are in a standard (CCW-positive, y-up) math frame:
# negative = turn right (clockwise), positive = turn left (counter-clockwise)
MOVES = {
    'Move-Forward':      {'turn': 0.0,   'step': 5.5},
    'Slight-Right-Turn': {'turn': -0.09, 'step': 4.5},
    'Sharp-Right-Turn':  {'turn': -0.30, 'step': 3.0},
    'Slight-Left-Turn':  {'turn': 0.09,  'step': 4.5},
}

# ---------------------------------------------------------------------
# Maze geometry (generated maze instead of a single open room)
# ---------------------------------------------------------------------
COLS, ROWS, CELL, M = 9, 6, 1.3, 0.6
W = COLS * CELL + 2 * M
H = ROWS * CELL + 2 * M
SENSOR_RANGE = 8.0
TOO_CLOSE, TOO_FAR = 0.32, 0.85
FRONT_SAFETY = 0.45


def generate_maze(seed):
    """Randomized DFS (recursive backtracker) maze on a COLS x ROWS grid."""
    rng = random.Random(seed)
    cells = [[{'N': True, 'S': True, 'E': True, 'W': True} for _ in range(COLS)] for _ in range(ROWS)]
    visited = [[False] * COLS for _ in range(ROWS)]
    dirs = [('N', 1, 0, 'S'), ('S', -1, 0, 'N'), ('E', 0, 1, 'W'), ('W', 0, -1, 'E')]
    stack = [(0, 0)]
    visited[0][0] = True
    while stack:
        r, c = stack[-1]
        options = []
        for d, dr, dc, opp in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS and not visited[nr][nc]:
                options.append((d, nr, nc, opp))
        if options:
            d, nr, nc, opp = rng.choice(options)
            cells[r][c][d] = False
            cells[nr][nc][opp] = False
            visited[nr][nc] = True
            stack.append((nr, nc))
        else:
            stack.pop()

    segs = []
    for r in range(ROWS):
        for c in range(COLS):
            x0, y0 = M + c * CELL, M + r * CELL
            x1, y1 = x0 + CELL, y0 + CELL
            cw = cells[r][c]
            if cw['S']:
                segs.append(((x0, y0), (x1, y0)))
            if cw['W']:
                segs.append(((x0, y0), (x0, y1)))
            if c == COLS - 1 and cw['E']:
                segs.append(((x1, y0), (x1, y1)))
            if r == ROWS - 1 and cw['N']:
                segs.append(((x0, y1), (x1, y1)))
    return segs


def ray_hit(walls, ox, oy, dx, dy):
    best = SENSOR_RANGE
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


def sensors(walls, x, y, theta):
    dx, dy = math.cos(theta), math.sin(theta)
    ldx, ldy = -math.sin(theta), math.cos(theta)   # left = +90 deg (CCW) from heading
    front = ray_hit(walls, x, y, dx, dy)
    left = ray_hit(walls, x, y, ldx, ldy)
    return front, left, (dx, dy), (ldx, ldy)


def classify(d):
    if d < TOO_CLOSE:
        return 'Too-Close'
    if d > TOO_FAR:
        return 'Too-Far'
    return 'Ideal'


def sample_next(probs):
    r, acc = np.random.random(), 0.0
    for i, p in enumerate(probs):
        acc += p
        if r <= acc:
            return STATES[i]
    return STATES[-1]


# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------
def init_state(new_maze=True, seed=None):
    if new_maze or 'maze_seed' not in st.session_state:
        st.session_state.maze_seed = seed if seed is not None else random.randint(0, 1_000_000)
        st.session_state.walls = generate_maze(st.session_state.maze_seed)
    st.session_state.robot = {'x': M + CELL / 2, 'y': M + CELL / 2, 'theta': 0.0}
    st.session_state.state = 'Ideal'
    st.session_state.step = 0
    st.session_state.cum_reward = 0
    st.session_state.safety_count = 0
    st.session_state.trail = []
    st.session_state.log = []
    st.session_state.running = False
    st.session_state.last_action = '—'
    st.session_state.last_reward = None
    st.session_state.last_sensors = (None, None)


if 'robot' not in st.session_state:
    init_state(new_maze=True, seed=42)


def do_step():
    ss = st.session_state
    front, left, _, _ = sensors(ss.walls, ss.robot['x'], ss.robot['y'], ss.robot['theta'])
    ss.state = classify(left)

    action = POLICY[ss.state]
    overridden = False
    if front < FRONT_SAFETY:
        action = 'Sharp-Right-Turn'
        overridden = True
        ss.safety_count += 1

    reward = R[ss.state][POLICY[ss.state]]
    ss.cum_reward += reward
    ss.step += 1

    mv = MOVES[action]
    ss.robot['theta'] += mv['turn']
    ss.robot['x'] += math.cos(ss.robot['theta']) * mv['step'] * 0.02
    ss.robot['y'] += math.sin(ss.robot['theta']) * mv['step'] * 0.02
    ss.robot['x'] = min(max(ss.robot['x'], M + 0.05), W - M - 0.05)
    ss.robot['y'] = min(max(ss.robot['y'], M + 0.05), H - M - 0.05)

    ss.trail.append({'x': ss.robot['x'], 'y': ss.robot['y']})
    if len(ss.trail) > 3000:
        ss.trail.pop(0)

    ss.last_action = action + (' (safety)' if overridden else '')
    ss.last_reward = reward
    ss.last_sensors = (left, front)

    tag = "SAFETY" if overridden else ss.state
    ss.log.append(f"#{ss.step:04d}  {tag:<10s}  {action:<18s}  reward {reward:+d}")
    if len(ss.log) > 200:
        ss.log.pop(0)

    ss.state = sample_next(T[ss.state][POLICY[ss.state]])


def reset(new_maze=False):
    init_state(new_maze=new_maze)


# ---------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------
st.title("Wall-Following Robot — MDP Maze Trace")
st.caption("A maze defines the boundaries; the trained policy steers from the left-sensor "
           "reading, with a front-proximity check that only intervenes to avoid a collision. "
           "The red trace is the path actually taken.")

col_plot, col_side = st.columns([2.2, 1])

with col_plot:
    b1, b2, b3, b4, b5 = st.columns([1, 1, 1, 1, 2])
    if b1.button("▶ Play" if not st.session_state.running else "⏸ Pause"):
        st.session_state.running = not st.session_state.running
    if b2.button("Step"):
        do_step()
    if b3.button("Reset"):
        reset(new_maze=False)
    if b4.button("New Maze"):
        reset(new_maze=True)
    speed = b5.slider("Speed (sec/step)", 0.02, 0.5, 0.06, 0.02)
    plot_ph = st.empty()

with col_side:
    st.subheader("Current state")
    m1, m2 = st.columns(2)
    m1.metric("Step", st.session_state.step)
    m2.metric("Cumulative reward", st.session_state.cum_reward)
    state_ph = st.empty()
    sens_ph = st.empty()
    action_ph = st.empty()
    reward_ph = st.empty()
    safety_ph = st.empty()

    st.subheader("P(next state | current, action)")
    bars_ph = st.empty()

    st.subheader("Event log")
    log_ph = st.empty()


def draw_dotted_border(ax, x0, y0, x1, y1, spacing=0.22, size=28):
    """Small alternating black/orange squares tracing the outer frame,
    matching the classic maze-print border look."""
    pts = []
    n = int((x1 - x0) / spacing)
    for i in range(n + 1):
        pts.append((x0 + i * spacing, y0))
        pts.append((x0 + i * spacing, y1))
    n = int((y1 - y0) / spacing)
    for i in range(n + 1):
        pts.append((x0, y0 + i * spacing))
        pts.append((x1, y0 + i * spacing))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cols = ['#111111' if i % 2 == 0 else '#f2a33a' for i in range(len(pts))]
    ax.scatter(xs, ys, c=cols, s=size, marker='s', zorder=1, edgecolors='none')


def render():
    ss = st.session_state
    fig, ax = plt.subplots(figsize=(7, 4.7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_xlim(-0.15, W + 0.15)
    ax.set_ylim(-0.15, H + 0.15)
    ax.set_aspect('equal')
    ax.axis('off')

    draw_dotted_border(ax, 0, 0, W, H)

    # maze walls
    for (ax0, ay0), (bx0, by0) in ss.walls:
        ax.plot([ax0, bx0], [ay0, by0], color="#111111", linewidth=6, solid_capstyle='round', zorder=2)

    # sensors (subtle, drawn under the trail)
    front, left, fdir, ldir = sensors(ss.walls, ss.robot['x'], ss.robot['y'], ss.robot['theta'])
    rx, ry = ss.robot['x'], ss.robot['y']
    ax.plot([rx, rx + ldir[0] * left], [ry, ry + ldir[1] * left], color="#c9c9c9", linestyle='--', linewidth=1, zorder=3)
    ax.plot([rx, rx + fdir[0] * front], [ry, ry + fdir[1] * front], color="#c9c9c9", linestyle='--', linewidth=1, zorder=3)

    # red path trace, with beads every few points for the "solved maze" look
    if len(ss.trail) > 1:
        xs = [p['x'] for p in ss.trail]
        ys = [p['y'] for p in ss.trail]
        ax.plot(xs, ys, color="#e8352b", linewidth=2, zorder=4)
        ax.scatter(xs[::4], ys[::4], color="#e8352b", s=6, zorder=4)

    # robot as a red dot
    ax.scatter([rx], [ry], color="#e8352b", s=90, zorder=5, edgecolors='black', linewidths=1)

    plot_ph.pyplot(fig, use_container_width=True)
    plt.close(fig)

    state_ph.markdown(f"**Left sensor state:** :{'red' if ss.state=='Too-Close' else 'green' if ss.state=='Ideal' else 'blue'}[{ss.state}]")
    if ss.last_sensors[0] is not None:
        sens_ph.write(f"Left / front dist: `{ss.last_sensors[0]:.2f}` / `{ss.last_sensors[1]:.2f}`")
    action_ph.write(f"Action: `{ss.last_action}`")
    if ss.last_reward is not None:
        reward_ph.write(f"Step reward: `{ss.last_reward:+d}`")
    safety_ph.write(f"Safety overrides: `{ss.safety_count}`")

    a = POLICY[ss.state]
    probs = T[ss.state][a]
    bars_ph.bar_chart({"P(next state)": dict(zip(STATES, probs))})

    log_ph.code("\n".join(ss.log[-15:]) if ss.log else "—", language=None)


render()

if st.session_state.running:
    do_step()
    time.sleep(speed)
    st.rerun()
