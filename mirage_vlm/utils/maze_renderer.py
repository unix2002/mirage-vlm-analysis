import numpy as np

def generate_maze_traces(map_desc, full_path, center_x, center_y, scale=0.8, y_stretch=1.0):
    """
    Generates batch-ready trace coordinates for a single micro-maze.
    Draws a literal grid (e.g. 5x5 lines) and a red path on top.
    """
    if not map_desc or not isinstance(map_desc, list) or len(map_desc) == 0:
        return {
            'grid_x': [], 'grid_y': [], 
            'path_x': [], 'path_y': [], 
            'obstacle_x': [], 'obstacle_y': [],
            'start_x': [], 'start_y': [],
            'end_x': [], 'end_y': []
        }
        
    rows = len(map_desc)
    cols = len(map_desc[0])
    
    cell_w = scale / max(1, cols)
    cell_h = (scale / y_stretch) / max(1, rows)
    
    # Top-left corner of the grid
    start_x = center_x - scale / 2
    start_y = center_y + (scale / y_stretch) / 2
    
    grid_x, grid_y = [], []
    
    # Draw horizontal lines for the grid
    for r in range(rows + 1):
        y = start_y - r * cell_h
        grid_x.extend([start_x, start_x + scale, None])
        grid_y.extend([y, y, None])
        
    # Draw vertical lines for the grid
    for c in range(cols + 1):
        x = start_x + c * cell_w
        grid_x.extend([x, x, None])
        grid_y.extend([start_y, start_y - (scale / y_stretch), None])
        
    # Helper to find the center of a specific cell (for the path)
    def get_center(r, c):
        x = start_x + (c + 0.5) * cell_w
        y = start_y - (r + 0.5) * cell_h
        return x, y
        
    path_x, path_y = [], []
    start_px, start_py = [], []
    end_px, end_py = [], []
    start_pos = None
    
    # Find start position (where cell value is 1)
    for r in range(rows):
        for c in range(cols):
            if map_desc[r][c] == 1:
                start_pos = (r, c)
                break
        if start_pos: break
        
    if start_pos and full_path:
        curr_r, curr_c = start_pos
        px, py = get_center(curr_r, curr_c)
        path_x.append(px)
        path_y.append(py)
        start_px.append(px)
        start_py.append(py)
        
        # Clean the string if it is in \boxed{...} format
        clean_path = full_path.replace('\\boxed{', '').replace('}', '').strip()
        moves = [m.strip().upper() for m in clean_path.split(',')]
        
        move_map = {
            'UP': (-1, 0), 'DOWN': (1, 0),
            'LEFT': (0, -1), 'RIGHT': (0, 1),
            'U': (-1, 0), 'D': (1, 0),
            'L': (0, -1), 'R': (0, 1)
        }
        
        for move in moves:
            if move in move_map:
                dr, dc = move_map[move]
                curr_r += dr
                curr_c += dc
                
                # Stop drawing if we go out of bounds
                if not (0 <= curr_r < rows and 0 <= curr_c < cols):
                    break
                    
                px, py = get_center(curr_r, curr_c)
                path_x.append(px)
                path_y.append(py)
        
        if len(path_x) > 0:
            end_px.append(path_x[-1])
            end_py.append(path_y[-1])
                
        path_x.append(None)
        path_y.append(None)

    return {
        'grid_x': grid_x, 'grid_y': grid_y,
        'path_x': path_x, 'path_y': path_y,
        'obstacle_x': [], 'obstacle_y': [],
        'start_x': start_px, 'start_y': start_py,
        'end_x': end_px, 'end_y': end_py
    }
