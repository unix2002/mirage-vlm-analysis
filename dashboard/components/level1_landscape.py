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

    # Resolve Color Metric Label
    metric_labels = {
        'avg_kl': 'Reasoning Intensity',
        'correctness': 'Correctness',
        'level_id': 'Level ID',
        'seq_len': 'Seq Length',
        'num_latent': 'Latent Tokens'
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
        text=df['sample_id'],
        hovertext=df['sample_id'],
        customdata=df[['move_direction', 'correctness', 'avg_kl', 'seq_len', 'level_id']],
        hovertemplate=(
            "<b>%{text}</b><br>" +
            "Predicted: %{customdata[0]}<br>" +
            "Correct: %{customdata[1]}<br>" +
            "Avg KL Div: %{customdata[2]:.2f}<br>" +
            "Seq Len: %{customdata[3]}<br>" +
            "Level: %{customdata[4]}<br>" +
            "<extra></extra>"
        )
    ))

    fig.update_layout(
        template='plotly_white',
        clickmode='event+select',
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
