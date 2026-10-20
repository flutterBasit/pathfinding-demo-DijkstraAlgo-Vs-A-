import streamlit as st
import numpy as np
import plotly.graph_objects as go
import heapq
import time

# Page configuration
st.set_page_config(
    page_title="AI Pathfinding Lab | IMSciences Demo", 
    page_icon="🤖", 
    layout="centered"
)

# Custom Styling & Header
st.title("🤖 AI-Powered Pathfinding Lab")
st.caption("Advanced Algorithm & Heuristic Benchmark Dashboard | IMSciences Demo")

# Grid dimensions
GRID_SIZE = 12

# Initialize session state for grid layout
if "grid" not in st.session_state:
    st.session_state.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    # Default maze walls
    st.session_state.grid[3, 1:9] = 1
    st.session_state.grid[7, 3:11] = 1
    st.session_state.grid[5, 5] = 1

start_pos = (0, 0)
goal_pos = (GRID_SIZE - 1, GRID_SIZE - 1)

# Helper: Generate random obstacle layout
def generate_random_maze():
    np.random.seed(int(time.time()))
    new_grid = np.random.choice([0, 1], size=(GRID_SIZE, GRID_SIZE), p=[0.75, 0.25])
    new_grid[start_pos] = 0
    new_grid[goal_pos] = 0
    st.session_state.grid = new_grid

# Heuristic Function: Manhattan Distance
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Robust Pathfinding Solver returning visited nodes and path
def run_pathfinding(use_heuristic=False):
    grid = st.session_state.grid
    pq = []
    heapq.heappush(pq, (0, start_pos))
    came_from = {}
    g_score = {(r, c): float('inf') for r in range(GRID_SIZE) for c in range(GRID_SIZE)}
    g_score[start_pos] = 0
    
    visited_order = []

    while pq:
        current_f, current = heapq.heappop(pq)
        if current not in visited_order:
            visited_order.append(current)

        if current == goal_pos:
            break

        r, c = current
        neighbors = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]

        for nr, nc in neighbors:
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and grid[nr, nc] == 0:
                tentative_g = g_score[current] + 1
                if tentative_g < g_score[(nr, nc)]:
                    came_from[(nr, nc)] = current
                    g_score[(nr, nc)] = tentative_g
                    h_val = heuristic((nr, nc), goal_pos) if use_heuristic else 0
                    f_score = tentative_g + h_val
                    heapq.heappush(pq, (f_score, (nr, nc)))

    # Reconstruct Path
    path = []
    curr = goal_pos
    if curr in came_from or curr == start_pos:
        while curr in came_from:
            path.append(curr)
            curr = came_from[curr]
        path.append(start_pos)
        path.reverse()

    return visited_order, path

# Plotly Visual Grid Generator
def plot_grid(visited_nodes, path, title_text):
    grid_matrix = st.session_state.grid.copy()
    
    # Color mapping: 0=Open(White), 1=Wall(DarkGray), Visited=LightBlue, Path=Gold, Start/Goal=Green/Red
    # We will build a numeric matrix for Plotly Heatmap
    # 0: Open, 1: Wall, 2: Visited, 3: Path, 4: Start, 5: Goal
    display_matrix = np.zeros((GRID_SIZE, GRID_SIZE))
    
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if grid_matrix[r, c] == 1:
                display_matrix[r, c] = 1 # Wall
                
    for node in visited_nodes:
        if node != start_pos and node != goal_pos:
            display_matrix[node] = 2 # Visited search wave
            
    for node in path:
        if node != start_pos and node != goal_pos:
            display_matrix[node] = 3 # Final Shortest Path
            
    display_matrix[start_pos] = 4 # Start
    display_matrix[goal_pos] = 5 # Goal

    # Custom Color Scale
    colorscale = [
        [0.0, "#ffffff"], # Open Path (White)
        [0.2, "#333333"], # Wall (Dark Gray)
        [0.4, "#93c5fd"], # Visited Search Wave (Soft Blue)
        [0.6, "#f59e0b"], # Shortest Path (Gold/Orange)
        [0.8, "#10b981"], # Start Node (Emerald Green)
        [1.0, "#ef4444"]  # Goal Node (Crimson Red)
    ]

    fig = go.Figure(data=go.Heatmap(
        z=display_matrix,
        colorscale=colorscale,
        showscale=False,
        xgap=3,
        ygap=3
    ))

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=14)),
        xaxis=dict(showticklabels=False, ticks=""),
        yaxis=dict(showticklabels=False, ticks="", autorange="reversed"),
        margin=dict(l=10, r=10, t=30, b=10),
        height=300
    )
    return fig

# --- UI CONTROLS ---
st.subheader("🗺️ 1. Obstacle Configuration")
col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    if st.button("🎲 Generate Random Maze", use_container_width=True):
        generate_random_maze()
        st.rerun()
with col_btn2:
    if st.button("🔄 Reset Default Maze", use_container_width=True):
        st.session_state.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        st.session_state.grid[3, 1:9] = 1
        st.session_state.grid[7, 3:11] = 1
        st.rerun()

# Preview Initial Grid
fig_preview = plot_grid([], [], "Current Maze Layout (🚀 Start -> 🎯 Goal)")
st.plotly_chart(fig_preview, use_container_width=True)

st.divider()

st.subheader("🚀 2. Execute Algorithm Benchmark")
if st.button("⚡ Run Dijkstra vs AI A* Comparison", type="primary", use_container_width=True):
    with st.spinner("Executing search algorithms..."):
        time.sleep(0.3)
        d_visited, d_path = run_pathfinding(use_heuristic=False)
        a_visited, a_path = run_pathfinding(use_heuristic=True)

    st.success("Benchmark Complete!")

    # Metrics Row
    m1, m2 = st.columns(2)
    with m1:
        st.metric(label="🔴 Dijkstra Nodes Explored", value=len(d_visited))
    with m2:
        st.metric(label="🟢 A* Search Nodes Explored", value=len(a_visited), delta=f"{len(d_visited) - len(a_visited)} fewer nodes!")

    # Side-by-Side Plotly Visual Comparison
    col_map1, col_map2 = st.columns(2)
    
    with col_map1:
        fig_d = plot_grid(d_visited, d_path, "🔴 Dijkstra (Blind Search Wave)")
        st.plotly_chart(fig_d, use_container_width=True)
        
    with col_map2:
        fig_a = plot_grid(a_visited, a_path, "🟢 A* (AI Heuristic Beam)")
        st.plotly_chart(fig_a, use_container_width=True)

    if len(a_visited) < len(d_visited):
        efficiency = round((len(d_visited) - len(a_visited)) / len(d_visited) * 100, 1)
        st.balloons()
        st.info(f"💡 **AI Advantage:** A* Search was **{efficiency}% more efficient** because its heuristic compass steered it directly toward the target destination while Dijkstra spread out blindly!")
