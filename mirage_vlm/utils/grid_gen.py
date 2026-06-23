import re
from pathlib import Path
from PIL import Image, ImageDraw
from io import BytesIO
import base64
# from dashboard.mock_data import MOCK_DATA
# python -m grid_gen.grid_gen_full

# DATA = MOCK_DATA

def pil_to_data_url(img, fmt="PNG"):
    buf = BytesIO()
    img.save(buf, format=fmt)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{data}"

def analyze_tile(tile):
    """Samples a tile and returns its classification: 'hole', 'player', 'goal', or 'ice'."""
    if tile == 1:
        return 'player'
    elif tile == -1:
        return 'hole'
    elif tile == 2:
        return 'goal'
    # Default is the snowy/icy paths
    else:
        return "ice"

def add_path(draw, path, start_pos, scale, width=10, color="blue"):
    """Draws a path on the maze image using a chosen color."""
    move_map = {
        'UP': (-1, 0), 'DOWN': (1, 0),
        'LEFT': (0, -1), 'RIGHT': (0, 1),
        'U': (-1, 0), 'D': (1, 0),
        'L': (0, -1), 'R': (0, 1)
    }

    if not path:
        return start_pos

    coords = [start_pos]
    current_r, current_c = start_pos
    for move in path:
        if move not in move_map:
            continue
        dr, dc = move_map[move]
        current_r += dr
        current_c += dc
        coords.append((current_r, current_c))

    if len(coords) >= 2:
        draw.line(
            [
                (c * scale + scale / 2, r * scale + scale / 2)
                for r, c in coords
            ],
            fill=color,
            width=width,
            joint="curve"
        )

    return coords[-1]

def draw_paths(draw, orig_path=None, ablated_path=None, player_pos=None, scale=50):
    """Draws both original and ablated paths on the maze image."""

    if not player_pos:
        return None

    # Draw start position of the player
    r, c = player_pos
    new_left, new_top = c * scale + 0.375 * scale, r * scale + 0.375 * scale
    draw.rectangle(
        [new_left, new_top, new_left + scale * 0.25, new_top + scale * 0.25],
        fill="black"
    )

    orig_end = None
    ablated_end = None

    if ablated_path:
        ablated_end = add_path(
            draw, ablated_path, player_pos, scale,
            width=10, color="red"
        )

    if orig_path:
        orig_end = add_path(
            draw, orig_path, player_pos, scale,
            width=5 if ablated_path else 10, color="blue"
        )

    if ablated_end:
        r, c = ablated_end
        red_box_size = scale * 0.6
        new_left = c * scale + (scale - red_box_size) / 2
        new_top = r * scale + (scale - red_box_size) / 2
        draw.rectangle(
            [new_left, new_top, new_left + red_box_size, new_top + red_box_size],
            fill="red"
        )

    if orig_end:
        r, c = orig_end
        blue_box_size = scale * 0.5
        new_left = c * scale + (scale - blue_box_size) / 2
        new_top = r * scale + (scale - blue_box_size) / 2
        draw.rectangle(
            [new_left, new_top, new_left + blue_box_size, new_top + blue_box_size],
            fill="blue"
        )

    return orig_end

def maze_renderer(map_desc, orig_path, ablated_path=None, scale=50):
    """Processes all steps within a specific map instance folder."""

    if not map_desc or not isinstance(map_desc, list) or len(map_desc) == 0:
        return None

    grid_size = len(map_desc)

    canvas_dim = grid_size * scale
    new_img = Image.new("RGB", (canvas_dim, canvas_dim), "white")
    draw = ImageDraw.Draw(new_img)
    player_pos = None

    # Draw maze
    for r in range(grid_size):
        for c in range(grid_size):
            tile = map_desc[r][c]
            tile_type = analyze_tile(tile)

            # Check if this tile is the predetermined goal
            if tile_type == "goal":
                fill_color = "lightgreen"
            elif tile_type == "player":
                fill_color = "white"
                player_pos = (r, c)
            elif tile_type == "hole":
                fill_color = "black"
            else:
                fill_color = "white"

            # Draw grid tile
            new_left, new_top = c * scale, r * scale
            draw.rectangle(
                [new_left, new_top, new_left + scale, new_top + scale],
                fill=fill_color, outline="gray"
            )

    # Draw paths in the requested order: ablated first (bottom), original second.
    orig_end = None
    ablated_end = None

    if ablated_path:
        ablated_end = add_path(
            draw, ablated_path, player_pos, scale,
            width=10, color="red"
        )

    if orig_path:
        orig_end = add_path(
            draw, orig_path, player_pos, scale,
            width=5 if ablated_path else 10, color="blue"
        )

    # Draw start position of the player
    r, c = player_pos
    new_left, new_top = c * scale + 0.375 * scale, r * scale + 0.375 * scale
    draw.rectangle(
        [new_left, new_top, new_left + scale * 0.25, new_top + scale * 0.25],
        fill="black"
    )

    # Draw final player position boxes.
    if ablated_end:
        r, c = ablated_end
        red_box_size = scale * 0.6
        new_left = c * scale + (scale - red_box_size) / 2
        new_top = r * scale + (scale - red_box_size) / 2
        draw.rectangle(
            [new_left, new_top, new_left + red_box_size, new_top + red_box_size],
            fill="red"
        )

    if orig_end:
        r, c = orig_end
        blue_box_size = scale * 0.5
        new_left = c * scale + (scale - blue_box_size) / 2
        new_top = r * scale + (scale - blue_box_size) / 2
        draw.rectangle(
            [new_left, new_top, new_left + blue_box_size, new_top + blue_box_size],
            fill="blue"
        )

    # Draw the outer canvas frame boundary
    draw.rectangle([0, 0, canvas_dim - 1, canvas_dim - 1], outline="gray")

    return pil_to_data_url(new_img)
