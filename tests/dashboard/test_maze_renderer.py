import pytest
import numpy as np
from mirage_vlm.utils.maze_renderer import generate_maze_traces

def test_maze_renderer_bounds():
    """Verify that different size mazes fit exactly in the [-scale/2, scale/2] box relative to center."""
    
    # 4x4 Maze centered at (10, 10) with scale 1.0
    map_4x4 = [[0, 0, 0, 0] for _ in range(4)]
    res_4x4 = generate_maze_traces(map_4x4, "", center_x=10, center_y=10, scale=1.0)
    
    # Grid should be exactly from 9.5 to 10.5
    # (Filtering out None separators before taking min/max)
    grid_x_4 = [x for x in res_4x4['grid_x'] if x is not None]
    grid_y_4 = [y for y in res_4x4['grid_y'] if y is not None]
    
    assert min(grid_x_4) == 9.5
    assert max(grid_x_4) == 10.5
    assert min(grid_y_4) == 9.5
    assert max(grid_y_4) == 10.5
    
    # 8x8 Maze centered at (-5, -5) with scale 2.0
    map_8x8 = [[0] * 8 for _ in range(8)]
    res_8x8 = generate_maze_traces(map_8x8, "", center_x=-5, center_y=-5, scale=2.0)
    
    grid_x_8 = [x for x in res_8x8['grid_x'] if x is not None]
    grid_y_8 = [y for y in res_8x8['grid_y'] if y is not None]
    
    # Box should be [-6, -4]
    assert min(grid_x_8) == -6.0
    assert max(grid_x_8) == -4.0
    assert min(grid_y_8) == -6.0
    assert max(grid_y_8) == -4.0

def test_maze_renderer_path():
    """Verify that the solution path is traced correctly."""
    
    # 3x3 Map, start at (0,0), move RIGHT, DOWN
    map_desc = [
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    path_str = r"\boxed{RIGHT, DOWN}"
    
    res = generate_maze_traces(map_desc, path_str, center_x=0, center_y=0, scale=2.0)
    
    path_x = res['path_x']
    path_y = res['path_y']
    
    # Ignore the final None
    px = [x for x in path_x if x is not None]
    py = [y for y in path_y if y is not None]
    
    assert len(px) == 3
    # Centers of cells (0,0), (0,1), (1,1)
    np.testing.assert_allclose(px, [-2/3, 0.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(py, [2/3, 2/3, 0.0], atol=1e-8)
