
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

_DIRECTION_COLORS = {
    'UP': 'rgba(31, 119, 180, 0.15)',
    'DOWN': 'rgba(255, 127, 14, 0.15)',
    'LEFT': 'rgba(44, 160, 44, 0.15)',
    'RIGHT': 'rgba(214, 39, 40, 0.15)',
    'UNKNOWN': 'rgba(127, 127, 127, 0.1)',
}

_HOVER_TEMPLATE = (
    "<b>%{text}</b><br>"
    "Predicted: %{customdata[0]}<br>"
    "Correct: %{customdata[1]}<br>"
    "Avg KL Div: %{customdata[2]:.2f}<br>"
    "Seq Len: %{customdata[3]}<br>"
    "Level: %{customdata[4]}<br>"
    "Projection Error: %{customdata[5]:.4f}<br>"
)


def _compute_y_stretch(viewport, df):
    """Compute y-stretch factor to preserve maze aspect ratio."""
    if viewport:
        vp_w = viewport['x_max'] - viewport['x_min']
        vp_h = viewport['y_max'] - viewport['y_min']
    else:
        vp_w = df['umap_x'].max() - df['umap_x'].min()
        vp_h = df['umap_y'].max() - df['umap_y'].min()
    return (vp_w / vp_h) / 0.74 if vp_h > 0 else 1.0


def create_level1_landscape(data_source, color_metric='avg_kl', zoom_level=1.0, viewport=None, highlight_flippers=False):
    if not data_source:
        return dcc.Graph(id='level1-scatter', figure=go.Figure().update_layout(title="No Data Available"))

    # 1. Prepare data
    rows = []
    
    # Pre-calculate global min/max for the color scale BEFORE culling
    # so that the colorbar remains static during zoom.
    global_color_vals = []
    for s in data_source:
        if color_metric == 'avg_kl':
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
            'has_plan_flip': s.get('has_plan_flip', False),
            'map_desc': s.get('map_desc'),
            'full_path': s.get('full_path')
        })

    df = pd.DataFrame(rows)

    # Mark plan-flippers from the free-run reroute data (regen_gen). Only loaded
    # when the highlight is on. Matches by namespaced id ('sample_NNN'/'test_NNN').
    if highlight_flippers and 'sample_id' in df.columns and len(df):
        from ..gen_data import flipper_ids
        df['has_plan_flip'] = df['sample_id'].isin(flipper_ids())

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
        'level_id': 'Level ID',
        'seq_len': 'Seq Length',
        'num_latent': 'Latent Tokens',
        'umap_uncertainty': 'Projection Error'
    }
    color_title = metric_labels.get(color_metric, color_metric)


    # Add Cluster Boundaries (Convex Hulls)
    fig = go.Figure()

    if HAS_SCIPY and len(df) > 5 and zoom_level < 3.0:
        for direction, color in _DIRECTION_COLORS.items():
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
                    title=dict(text=color_title, side='bottom', font=dict(size=10)),
                    orientation='h',
                    thickness=12,
                    len=0.6,
                    y=-0.28,
                    x=0.5,
                    xanchor='center',
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
            hovertemplate=_HOVER_TEMPLATE + "<extra></extra>"
        ))

    if highlight_flippers and macro_opacity > 0:
        flip_df = df[df['has_plan_flip'] == True]
        if len(flip_df) > 0:
            fig.add_trace(go.Scatter(
                x=flip_df['umap_x'],
                y=flip_df['umap_y'],
                mode='markers',
                marker=dict(
                    size=flip_df['velocity_size'] + 10,
                    symbol='circle-open',
                    color='#dc3545',
                    opacity=0.6 * macro_opacity,
                    line=dict(width=2),
                ),
                hoverinfo='skip',
                showlegend=False,
                name='Plan Flippers',
            ))

    if show_micro:
        # Generate batch traces for all visible mazes
        all_grid_x, all_grid_y = [], []
        all_path_x, all_path_y = [], []
        all_start_x, all_start_y = [], []
        all_end_x, all_end_y = [], []

        # Dynamic scaling: keep mazes consistent on screen, prevent clashing
        base_scale = 3.0 / max(1.0, zoom_level)
        y_stretch = _compute_y_stretch(viewport, df)

        for _, row in df.iterrows():
            traces = generate_maze_traces(
                row.get('map_desc'), 
                row.get('full_path'), 
                row['umap_x'], 
                row['umap_y'], 
                scale=base_scale,
                y_stretch=y_stretch
            )
            all_grid_x.extend(traces['grid_x'])
            all_grid_y.extend(traces['grid_y'])
            all_path_x.extend(traces['path_x'])
            all_path_y.extend(traces['path_y'])
            all_start_x.extend(traces['start_x'])
            all_start_y.extend(traces['start_y'])
            all_end_x.extend(traces['end_x'])
            all_end_y.extend(traces['end_y'])

        # Background Grid
        if all_grid_x:
            fig.add_trace(go.Scatter(
                x=all_grid_x, y=all_grid_y,
                mode='lines',
                line=dict(color='rgba(200, 200, 200, 0.5)', width=1),
                hoverinfo='skip',
                showlegend=False,
                name='Maze Grid'
            ))

        # Solution Path
        if all_path_x:
            fig.add_trace(go.Scatter(
                x=all_path_x, y=all_path_y,
                mode='lines',
                line=dict(color='rgba(231, 76, 60, 0.9)', width=2.5),
                hoverinfo='skip',
                showlegend=False,
                name='Solution Path'
            ))

        # Start Points
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

        # End Points
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

        # Invisible hover targets (grid/path skip hoverinfo)
        fig.add_trace(go.Scatter(
            x=df['umap_x'],
            y=df['umap_y'],
            mode='markers',
            marker=dict(size=20, color='rgba(0,0,0,0)'), # Invisible
            text=df['sample_id'],
            hovertext=df['sample_id'],
            customdata=df[['move_direction', 'correctness', 'avg_kl', 'seq_len', 'level_id', 'umap_uncertainty']],
            hovertemplate=_HOVER_TEMPLATE + "<b>[Micro-Maze View]</b><br><extra></extra>",
            showlegend=False,
            name='Maze Hover'
        ))

    fig.update_layout(
        template='plotly_white',
        clickmode='event',
        margin=dict(l=40, r=80, t=30, b=65),
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
            tickfont=dict(size=10, color='#666'),
            scaleanchor='x',
            scaleratio=1,
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=12, color="#333"),
        title=dict(
            text="Fig 1. Latent Reasoning Landscape",
            x=0.05,
            y=0.98,
            font=dict(size=15, color="#111")
        )
    )

    if len(df) > 0:
        x_vals, y_vals = df['umap_x'], df['umap_y']
        x_pad = (x_vals.max() - x_vals.min()) * 0.05 or 1.0
        y_pad = (y_vals.max() - y_vals.min()) * 0.05 or 1.0
        fig.update_xaxes(range=[x_vals.min() - x_pad, x_vals.max() + x_pad])
        fig.update_yaxes(range=[y_vals.min() - y_pad, y_vals.max() + y_pad])

    return fig
