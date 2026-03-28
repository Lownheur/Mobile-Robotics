import heapq
import time
import numpy as np

def load_gazebo_map(scale=3, inflation_cells=0):
    with open('/home/ubuntu/project/circuit_map.pgm', 'rb') as f:
        magic = f.readline()
        line = f.readline()
        while line.startswith(b'#'):
            line = f.readline()
        width, height = map(int, line.split())
        maxval = int(f.readline())
        data = f.read()
        
    img = np.frombuffer(data, dtype=np.uint8).reshape((height, width))
    base_grid = (img < 250).astype(int)
    
    new_h, new_w = height // scale, width // scale
    small_grid = np.zeros((new_h, new_w))
    
    for r in range(new_h):
        for c in range(new_w):
            block = base_grid[r*scale:(r+1)*scale, c*scale:(c+1)*scale]
            small_grid[r, c] = 1 if np.sum(block) > 0 else 0
            
    # Par defaut on n'inflate pas: on utilise exactement la map PGM.
    if inflation_cells <= 0:
        return small_grid, height, width

    inflated = np.copy(small_grid)
    for r in range(new_h):
        for c in range(new_w):
            if small_grid[r, c] == 1:
                rmin, rmax = max(0, r - inflation_cells), min(new_h, r + inflation_cells + 1)
                cmin, cmax = max(0, c - inflation_cells), min(new_w, c + inflation_cells + 1)
                inflated[rmin:rmax, cmin:cmax] = 1

    return inflated, height, width

ORIGIN_X = -4.0
ORIGIN_Y_TOP = 6.0
BASE_RESOLUTION = 0.05
SCALE = 3 

# Dynamic globals cache
_grid_h = 240
_grid_w = 160

def set_grid_dims(h, w):
    global _grid_h, _grid_w
    _grid_h = h
    _grid_w = w

def create_gazebo_env_grid():
    # Grille de planning: petite inflation pour eloigner le chemin des murs.
    grid, h, w = load_gazebo_map(scale=SCALE, inflation_cells=1)
    set_grid_dims(h, w)
    return grid

def create_collision_env_grid():
    # Grille de collision: map brute sans inflation (murs reels Gazebo).
    grid, h, w = load_gazebo_map(scale=SCALE, inflation_cells=0)
    set_grid_dims(h, w)
    return grid

def world_to_grid(x, y):
    col_base = int((x - ORIGIN_X) / BASE_RESOLUTION)
    row_base = int((ORIGIN_Y_TOP - y) / BASE_RESOLUTION)
    
    r = row_base // SCALE
    c = col_base // SCALE
    return max(0, min((_grid_h//SCALE)-1, r)), max(0, min((_grid_w//SCALE)-1, c))

def grid_to_world(r, c):
    row_base = r * SCALE + (SCALE / 2.0)
    col_base = c * SCALE + (SCALE / 2.0)
    
    world_x = (col_base * BASE_RESOLUTION) + ORIGIN_X
    world_y = ORIGIN_Y_TOP - (row_base * BASE_RESOLUTION) 
    return world_x, world_y

def get_neighbors(node, grid):
    r, c = node
    candidates = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
    valid = []
    for nr, nc in candidates:
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] == 0:
            valid.append((nr, nc))
    return valid

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def run_dijkstra(grid, start, goal):
    t_start = time.time()
    pq = [(0, start)]
    dist = {start: 0}
    parent = {start: None}
    explored = []
    
    while pq:
        d, current = heapq.heappop(pq)
        explored.append(current)
        if current == goal: break
        for nxt in get_neighbors(current, grid):
            new_dist = dist[current] + 1
            if nxt not in dist or new_dist < dist[nxt]:
                dist[nxt] = new_dist
                parent[nxt] = current
                heapq.heappush(pq, (new_dist, nxt))
                
    path = []
    curr = goal
    if curr in parent:
        while curr is not None:
            path.append(curr)
            curr = parent[curr]
    return path[::-1], explored, time.time() - t_start

def run_greedy(grid, start, goal):
    t_start = time.time()
    pq = [(heuristic(start, goal), start)]
    parent = {start: None}
    explored = []
    visited = {start}
    
    while pq:
        h, current = heapq.heappop(pq)
        explored.append(current)
        if current == goal: break
        for nxt in get_neighbors(current, grid):
            if nxt not in visited:
                visited.add(nxt)
                parent[nxt] = current
                heapq.heappush(pq, (heuristic(nxt, goal), nxt))
                
    path = []
    curr = goal
    if curr in parent:
        while curr is not None:
            path.append(curr)
            curr = parent[curr]
    return path[::-1], explored, time.time() - t_start

def run_astar(grid, start, goal):
    t_start = time.time()
    pq = [(heuristic(start, goal), 0, start)]
    g_score = {start: 0}
    parent = {start: None}
    explored = []
    
    while pq:
        f, g, current = heapq.heappop(pq)
        explored.append(current)
        if current == goal: break
        for nxt in get_neighbors(current, grid):
            new_g = g_score[current] + 1
            if nxt not in g_score or new_g < g_score[nxt]:
                g_score[nxt] = new_g
                f_score = new_g + heuristic(nxt, goal)
                parent[nxt] = current
                heapq.heappush(pq, (f_score, new_g, nxt))
                
    path = []
    curr = goal
    if curr in parent:
        while curr is not None:
            path.append(curr)
            curr = parent[curr]
    return path[::-1], explored, time.time() - t_start
