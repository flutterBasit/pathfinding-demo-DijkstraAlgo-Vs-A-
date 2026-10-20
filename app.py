import streamlit as st
import numpy as np
import heapq
import time

# Page configuration
st.set_page_config(
    page_title="Pathfinding Benchmark: Dijkstra vs A*", 
    page_icon="🗺️", 
    layout="centered"
)

st.title("🗺️ Smart Pathfinding: Dijkstra vs A* Search")
st.caption("IMSciences Demonstration Web Application | Powered by Streamlit & Python")

# Grid dimensions
GRID_SIZE = 10

# Initialize default grid session state
if "grid" not in st.session_state:
    st.session_state.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    # Default initial obstacle walls for demonstration
    st.session_state.grid[3, 2:8] = 1
    st.session_state.grid[6, 1:7] = 1

start_pos = (0, 0)
goal_pos = (GRID_SIZE - 1, GRID_SIZE - 1)

# Reset Grid Function
def reset_grid():
    st.session_state.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    st.session_state.grid[3, 2:8] = 1
    st.session_state.grid[6, 1:7] = 1

st.write("### 1. Interactive Grid Canvas")
st.write("Tap any white square to toggle custom obstacle walls (⬛ = Wall, ⬜ = Open Path):")

# Render interactive grid layout
for r in range(GRID_SIZE):
    cols = st.columns(GRID_SIZE)
    for c in range(GRID_SIZE):
        if (r, c) == start_pos:
            cols[c].button("🚀", key=f"btn_{r}_{c}", disabled=True)
        elif (r, c) == goal_pos:
            cols[c].button("🎯", key=f"btn_{r}_{c}", disabled=True)
        else:
            val = st.session_state.grid[r, c]
            label = "⬛" if val == 1 else "⬜"
            if cols[c].button(label, key=f"btn_{r}_{c}"):
                st.session_state.grid[r, c] = 1 if val == 0 else 0
                st.rerun()

st.button("🔄 Reset Obstacles to Default", on_click=reset_grid)

st.divider()

# Heuristic Function: Manhattan Distance h(n) = |x1 - x2| + |y1 - y2|
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Pathfinding Algorithm Solver
def solve_pathfinding(use_heuristic=False):
    grid = st.session_state.grid
    pq = []
    # Priority queue stores tuples of (f_score, position)
    heapq.heappush(pq, (0, start_pos))
    came_from = {}
    
    g_score = {(r, c): float('inf') for r in range(GRID_SIZE) for c in range(GRID_SIZE)}
    g_score[start_pos] = 0
    
    visited_nodes = 0

    while pq:
        current_f, current = heapq.heappop(pq)
        visited_nodes += 1

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
                    
                    # f(n) = g(n) + h(n)
                    h_val = heuristic((nr, nc), goal_pos) if use_heuristic else 0
                    f_score = tentative_g + h_val
                    heapq.heappush(pq, (f_score, (nr, nc)))

    # Path Reconstruction
    path = []
    curr = goal_pos
    if curr in came_from or curr == start_pos:
        while curr in came_from:
            path.append(curr)
            curr = came_from[curr]
        path.append(start_pos)
        path.reverse()

    return visited_nodes, len(path)

st.write("### 2. Benchmark Execution")

if st.button("🚀 Run Dijkstra vs A* Race", type="primary", use_container_width=True):
    with st.spinner("Executing algorithms..."):
        time.sleep(0.2)
        
        d_visited, d_path_len = solve_pathfinding(use_heuristic=False)
        a_visited, a_path_len = solve_pathfinding(use_heuristic=True)

    st.success("Algorithm Race Complete!")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔴 Dijkstra (Classic DSA)")
        st.caption("Formula: f(n) = g(n)")
        st.metric(label="Nodes Explored", value=d_visited)
        st.metric(label="Path Length", value=d_path_len if d_path_len > 1 else "No Path Found")

    with col2:
        st.subheader("🟢 A* Search (AI Heuristic)")
        st.caption("Formula: f(n) = g(n) + h(n)")
        st.metric(label="Nodes Explored", value=a_visited, delta=f"{d_visited - a_visited} fewer nodes!")
        st.metric(label="Path Length", value=a_path_len if a_path_len > 1 else "No Path Found")

    if a_visited < d_visited and a_path_len > 1:
        efficiency = round((d_visited - a_visited) / d_visited * 100, 1)
        st.info(f"💡 **AI Advantage:** A* Search explored **{efficiency}% fewer nodes** than Dijkstra while finding the exact same optimal path!")
