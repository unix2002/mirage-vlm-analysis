
from dash import dcc
import plotly.graph_objects as go
import pandas as pd
import numpy as np

try:
    from scipy.spatial import ConvexHull
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def create_level1_landscape(data_source, color_metric='avg_kl'):
    if not data_source:
        return dcc.Graph(id='level1-scatter', figure=go.Figure().update_layout(title="No Data Available"))

    # 1. Prepare data
    rows = []
    for s in data_source:
        avg_kl = np.mean([t['kl_divergence'] for t in s['tokens']])

        symbol_map = {
            'UP': 'triangle-up',
            'DOWN': 'triangle-down',
            'LEFT': 'triangle-left',
            'RIGHT': 'triangle-right',
            'UNKNOWN': 'circle'
        }

        rows.append({
            'sample_id': s['sample_id'],
            'umap_x': s['umap_x'],
            'umap_y': s['umap_y'],
            'umap_uncertainty': s.get('umap_uncertainty', 0),
            'correctness': s.get('correctness', False),
            'move_direction': s.get('move_direction', 'UNKNOWN'),
            'symbol': symbol_map.get(s.get('move_direction', 'UNKNOWN'), 'circle'),
            'avg_kl': avg_kl,
            'num_tokens': len(s['tokens']),
            'level_id': s.get('level_id', 0),
            'seq_len': s.get('seq_len', 0),
            'num_latent': s.get('num_latent', 6)
        })

    df = pd.DataFrame(rows)

    # If UMAP has not been computed yet (lazy startup), add tiny jitter so points
    # are visible until the first slider interaction triggers real projection.
    if np.allclose(df['umap_x'].values, 0.0) and np.allclose(df['umap_y'].values, 0.0):
        rng = np.random.default_rng(42)
        df['umap_x'] = df['umap_x'] + rng.normal(0, 0.05, len(df))
        df['umap_y'] = df['umap_y'] + rng.normal(0, 0.05, len(df))

    # Resolve Color Metric Label
    metric_labels = {
        'avg_kl': 'Reasoning Intensity',
        'correctness': 'Correctness',
        'level_id': 'Level ID',
        'seq_len': 'Seq Length',
        'num_latent': 'Latent Tokens',
        'umap_uncertainty': 'Projection Error'
    }
    color_title = metric_labels.get(color_metric, color_metric)


    # 2. Build Scientific Figure
    fig = go.Figure()

    # Add Cluster Boundaries (Convex Hulls)
    if HAS_SCIPY and len(df) > 5:
        colors = {
            'UP': 'rgba(31, 119, 180, 0.15)',    # Blue
            'DOWN': 'rgba(255, 127, 14, 0.15)',  # Orange
            'LEFT': 'rgba(44, 160, 44, 0.15)',   # Green
            'RIGHT': 'rgba(214, 39, 40, 0.15)',  # Red
            'UNKNOWN': 'rgba(127, 127, 127, 0.1)'# Gray
        }

        for direction, color in colors.items():
            subset = df[df['move_direction'] == direction]
            if len(subset) > 2:
                points = subset[['umap_x', 'umap_y']].values
                if np.unique(points, axis=0).shape[0] > 2:
                    try:
                        hull = ConvexHull(points)
                        hull_points = points[hull.vertices]
                        hull_points = np.vstack([hull_points, hull_points[0]])

                        fig.add_trace(go.Scatter(
                            x=hull_points[:, 0],
                            y=hull_points[:, 1],
                            fill="toself",
                            fillcolor=color,
                            line=dict(color='rgba(0,0,0,0)'),
                            hoverinfo='skip',
                            showlegend=False,
                            name=f"{direction} Cluster"
                        ))
                    except: pass

    # Ensure correctness is numeric for coloring if selected
    if color_metric == 'correctness':
        color_data = df['correctness'].astype(int)
        colorscale = [[0, '#e74c3c'], [1, '#2ecc71']] # Red to Green
    else:
        color_data = df[color_metric]
        colorscale = 'Viridis'

    # Add the main scatter trace
    fig.add_trace(go.Scatter(
        x=df['umap_x'],
        y=df['umap_y'],
        mode='markers',
        marker=dict(
            size=12,
            symbol=df['symbol'],
            color=color_data,
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(
                title=color_title,
                thickness=15,
                len=0.5,
                y=0.5,
                x=1.15,
                tickfont=dict(size=10)
            ),
            line=dict(
                width=1.5,
                color='rgba(0,0,0,0.5)'
            )
        ),
    )
    )
from dash import dcc
import plotly.graph_objects as go
import pandas as pd
import numpy as np

try:
    from scipy.spatial import ConvexHull
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from mirage_vlm.utils.maze_renderer import generate_maze_traces


def create_level1_landscape(data_source, color_metric='avg_kl', zoom_level=1.0, viewport=None):
    if not data_source:
        return dcc.Graph(id='level1-scatter', figure=go.Figure().update_layout(title="No Data Available"))

    # 1. Prepare data
    rows = []
    
    # Pre-calculate global min/max for the color scale BEFORE culling
    # so that the colorbar remains static during zoom.
    global_color_vals = []
    for s in data_source:
        if color_metric == 'correctness':
            global_color_vals.append(int(s.get('correctness', False)))
        elif color_metric == 'avg_kl':
            global_color_vals.append(np.mean([t['kl_divergence'] for t in s['tokens']]))
        elif color_metric in s:
            global_color_vals.append(s[color_metric])
        else:
            global_color_vals.append(0)
            
    cmin = min(global_color_vals) if global_color_vals else 0
    cmax = max(global_color_vals) if global_color_vals else 1

    for s in data_source:
        # Check viewport culling
        if viewport:
            x, y = s['umap_x'], s['umap_y']
            if not (viewport['x_min'] <= x <= viewport['x_max'] and viewport['y_min'] <= y <= viewport['y_max']):
                continue

        avg_kl = np.mean([t['kl_divergence'] for t in s['tokens']])

        symbol_map = {
            'UP': 'triangle-up',
            'DOWN': 'triangle-down',
            'LEFT': 'triangle-left',
            'RIGHT': 'triangle-right',
            'UNKNOWN': 'circle'
        }

        rows.append({
            'sample_id': s['sample_id'],
            'umap_x': s['umap_x'],
            'umap_y': s['umap_y'],
            'umap_uncertainty': s.get('umap_uncertainty', 0),
            'correctness': s.get('correctness', False),
            'move_direction': s.get('move_direction', 'UNKNOWN'),
            'symbol': symbol_map.get(s.get('move_direction', 'UNKNOWN'), 'circle'),
            'avg_kl': avg_kl,
            'num_tokens': len(s['tokens']),
            'level_id': s.get('level_id', 0),
            'seq_len': s.get('seq_len', 0),
            'num_latent': s.get('num_latent', 6),
            'map_desc': s.get('map_desc'),
            'full_path': s.get('full_path')
        })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        return go.Figure()

    # Calculate dynamic sizes
    # Uncertainty Aura Size
    df['aura_size'] = 10 + df['umap_uncertainty'] * 60

    # Velocity / Confidence Size
    kl_min, kl_max = df['avg_kl'].min(), df['avg_kl'].max()
    kl_norm = (df['avg_kl'] - kl_min) / (kl_max - kl_min + 1e-8)
    df['velocity_size'] = 8 + kl_norm * 14

    # Resolve Color Metric Label
    metric_labels = {
        'avg_kl': 'Reasoning Intensity',
        'correctness': 'Correctness',
        'level_id': 'Level ID',
        'seq_len': 'Seq Length',
        'num_latent': 'Latent Tokens',
        'umap_uncertainty': 'Projection Error'
    }
    color_title = metric_labels.get(color_metric, color_metric)


    # 2. Build Scientific Figure
    fig = go.Figure()

    # Add Cluster Boundaries (Convex Hulls)
    if HAS_SCIPY and len(df) > 5 and zoom_level < 3.0:
        colors = {
            'UP': 'rgba(31, 119, 180, 0.15)',    # Blue
            'DOWN': 'rgba(255, 127, 14, 0.15)',  # Orange
            'LEFT': 'rgba(44, 160, 44, 0.15)',   # Green
            'RIGHT': 'rgba(214, 39, 40, 0.15)',  # Red
            'UNKNOWN': 'rgba(127, 127, 127, 0.1)'# Gray
        }

        for direction, color in colors.items():
            subset = df[df['move_direction'] == direction]
            if len(subset) > 2:
                points = subset[['umap_x', 'umap_y']].values
                if np.unique(points, axis=0).shape[0] > 2:
                    try:
                        hull = ConvexHull(points)
                        hull_points = points[hull.vertices]
                        hull_points = np.vstack([hull_points, hull_points[0]])

                        fig.add_trace(go.Scatter(
                            x=hull_points[:, 0],
                            y=hull_points[:, 1],
                            fill="toself",
                            fillcolor=color,
                            line=dict(color='rgba(0,0,0,0)'),
                            hoverinfo='skip',
                            showlegend=False,
                            name=f"{direction} Cluster"
                        ))
                    except: pass

    # Transition logic based on zoom level
    show_macro = zoom_level < 20.0
    show_micro = zoom_level >= 15.0
    
    # Base macro opacity (the triangles fade out at 15.0)
    macro_opacity = max(0.0, min(1.0, 1.0 - (zoom_level - 15.0) / 5.0)) if zoom_level > 15.0 else 1.0
    
    # Aura opacity (starts fading in at zoom 3.0, fully opaque at 8.0, fades out with macro at 15.0)
    if zoom_level < 3.0:
        aura_opacity = 0.0
    elif zoom_level < 8.0:
        aura_opacity = 0.15 * ((zoom_level - 3.0) / 5.0)
    else:
        aura_opacity = 0.15 * macro_opacity

    # Ensure correctness is numeric for coloring if selected
    if color_metric == 'correctness':
        color_data = df['correctness'].astype(int)
        colorscale = [[0, '#e74c3c'], [1, '#2ecc71']] # Red to Green
    else:
        color_data = df[color_metric]
        colorscale = 'Viridis'

    if show_macro:
        # Add the Uncertainty Aura trace (behind main points)
        if aura_opacity > 0:
            fig.add_trace(go.Scatter(
                x=df['umap_x'],
                y=df['umap_y'],
                mode='markers',
                marker=dict(
                    size=df['aura_size'],
                    symbol='circle',
                    color='#bdc3c7', # Uniform clean gray color for uncertainty
                    showscale=False,
                    opacity=aura_opacity,
                    line=dict(width=0)
                ),
                hoverinfo='skip',
                showlegend=False,
                name="Uncertainty Aura"
            ))

        # Add the main scatter trace (Velocity / Confidence glyphs)
        fig.add_trace(go.Scatter(
            x=df['umap_x'],
            y=df['umap_y'],
            mode='markers',
            marker=dict(
                size=df['velocity_size'],
                symbol=df['symbol'],
                color=color_data,
                colorscale=colorscale,
                cmin=cmin,
                cmax=cmax,
                showscale=True,
                opacity=macro_opacity,
                colorbar=dict(
                    title=color_title,
                    thickness=15,
                    len=0.5,
                    y=0.5,
                    x=1.15,
                    tickfont=dict(size=10)
                ) if macro_opacity > 0.5 else None,
                line=dict(
                    width=1.5,
                    color='rgba(0,0,0,0.5)'
                )
            ),
            text=df['sample_id'],
            hovertext=df['sample_id'],
            customdata=df[['move_direction', 'correctness', 'avg_kl', 'seq_len', 'level_id', 'umap_uncertainty']],
            hovertemplate=(
                "<b>%{text}</b><br>" +
                "Predicted: %{customdata[0]}<br>" +
                "Correct: %{customdata[1]}<br>" +
                "Avg KL Div: %{customdata[2]:.2f}<br>" +
                "Seq Len: %{customdata[3]}<br>" +
                "Level: %{customdata[4]}<br>" +
                "Projection Error: %{customdata[5]:.4f}<br>" +
                "<extra></extra>"
            )
        ))

    if show_micro:
        # Generate batch traces for all visible mazes
        all_grid_x, all_grid_y = [], []
        all_path_x, all_path_y = [], []
        all_start_x, all_start_y = [], []
        all_end_x, all_end_y = [], []

        # Dynamic scaling: at zoom 15.0, scale is small, grows as you zoom in
        # We want the mazes to stay relatively consistent in screen space and prevent clashing
        base_scale = 3.0 / max(1.0, zoom_level)

        for _, row in df.iterrows():
            traces = generate_maze_traces(
                row.get('map_desc'), 
                row.get('full_path'), 
                row['umap_x'], 
                row['umap_y'], 
                scale=base_scale
            )
            all_grid_x.extend(traces['grid_x'])
            all_grid_y.extend(traces['grid_y'])
            all_path_x.extend(traces['path_x'])
            all_path_y.extend(traces['path_y'])
            all_start_x.extend(traces['start_x'])
            all_start_y.extend(traces['start_y'])
            all_end_x.extend(traces['end_x'])
            all_end_y.extend(traces['end_y'])

        # 1. Background Grid (Light Gray Lines)
        if all_grid_x:
            fig.add_trace(go.Scatter(
                x=all_grid_x, y=all_grid_y,
                mode='lines',
                line=dict(color='rgba(200, 200, 200, 0.5)', width=1),
                hoverinfo='skip',
                showlegend=False,
                name='Maze Grid'
            ))

        # 2. The Path (Bold Red Line)
        if all_path_x:
            fig.add_trace(go.Scatter(
                x=all_path_x, y=all_path_y,
                mode='lines',
                line=dict(color='rgba(231, 76, 60, 0.9)', width=2.5),
                hoverinfo='skip',
                showlegend=False,
                name='Solution Path'
            ))

        # 3. Start Points (Blue)
        if all_start_x:
            fig.add_trace(go.Scatter(
                x=all_start_x, y=all_start_y,
                mode='markers',
                marker=dict(color='#3498db', size=5, symbol='circle', line=dict(color='white', width=0.5)),
                hoverinfo='skip',
                showlegend=False,
                name='Start Points',
                unselected=dict(marker=dict(opacity=1))
            ))

        # 4. End Points (Green)
        if all_end_x:
            fig.add_trace(go.Scatter(
                x=all_end_x, y=all_end_y,
                mode='markers',
                marker=dict(color='#2ecc71', size=5, symbol='circle', line=dict(color='white', width=0.5)),
                hoverinfo='skip',
                showlegend=False,
                name='End Points',
                unselected=dict(marker=dict(opacity=1))
            ))

        # 5. Invisible Hover Targets (Since grid/path skip hover)
        fig.add_trace(go.Scatter(
            x=df['umap_x'],
            y=df['umap_y'],
            mode='markers',
            marker=dict(size=20, color='rgba(0,0,0,0)'), # Invisible
            text=df['sample_id'],
            hovertext=df['sample_id'],
            customdata=df[['move_direction', 'correctness', 'avg_kl', 'seq_len', 'level_id', 'umap_uncertainty']],
            hovertemplate=(
                "<b>%{text}</b><br>" +
                "Predicted: %{customdata[0]}<br>" +
                "Correct: %{customdata[1]}<br>" +
                "Avg KL Div: %{customdata[2]:.2f}<br>" +
                "Seq Len: %{customdata[3]}<br>" +
                "Level: %{customdata[4]}<br>" +
                "Projection Error: %{customdata[5]:.4f}<br>" +
                "<b>[Micro-Maze View]</b><br>" +
                "<extra></extra>"
            ),
            showlegend=False,
            name='Maze Hover'
        ))

    fig.update_layout(
        template='plotly_white',
        clickmode='event',
        margin=dict(l=40, r=80, t=30, b=40),
        showlegend=False,
        autosize=True,
        xaxis=dict(
            title="UMAP Latent Dimension 1",
            showgrid=True,
            gridwidth=1,
            gridcolor='#f0f0f0',
            zeroline=True,
            zerolinecolor='#e0e0e0',
            showticklabels=True,
            tickfont=dict(size=10, color='#666')
        ),
        yaxis=dict(
            title="UMAP Latent Dimension 2",
            showgrid=True,
            gridwidth=1,
            gridcolor='#f0f0f0',
            zeroline=True,
            zerolinecolor='#e0e0e0',
            showticklabels=True,
            tickfont=dict(size=10, color='#666')
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=12, color="#333"),
        title=dict(
            text="Fig 1. Latent Reasoning Landscape",
            x=0.05,
            y=0.98,
            font=dict(size=14, color="#111")
        )
    )

    return fig
