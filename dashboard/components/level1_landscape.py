from dash import dcc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from ..mock_data import MOCK_DATA


def create_level1_landscape():
    # 1. Prepare data with metrics for glyph encoding
    rows = []
    for s in MOCK_DATA:
        # Calculate a continuous metric: Average KL divergence across tokens
        avg_kl = np.mean([t['kl_divergence'] for t in s['tokens']])
        # Map directions to symbols
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
            'correctness': s['correctness'],
            'move_direction': s['move_direction'],
            'symbol': symbol_map.get(s['move_direction'], 'circle'),
            'avg_kl': avg_kl,
            'complexity': len(s['tokens']) * 10 # Scaling for size
        })
    
    df = pd.DataFrame(rows)

    # 2. Build Advanced Glyph Scatter with graph_objects
    fig = go.Figure()

    # Split into Correct and Incorrect for distinct border styling if needed
    # but plotly supports vectorized color/line mapping
    fig.add_trace(go.Scatter(
        x=df['umap_x'],
        y=df['umap_y'],
        mode='markers',
        marker=dict(
            size=12,
            symbol=df['symbol'],
            color=df['avg_kl'],
            colorscale='Viridis',
            showscale=False,
            line=dict(
                width=2,
                color=['#2ecc71' if c else '#e74c3c' for c in df['correctness']]
            )
        ),
        text=df['sample_id'],
        hovertext=df['sample_id'],
        customdata=df[['move_direction', 'correctness', 'avg_kl']],
        hovertemplate=(
            "<b>Sample: %{text}</b><br>" +
            "Direction: %{customdata[0]}<br>" +
            "Correct: %{customdata[1]}<br>" +
            "Avg KL: %{customdata[2]:.2f}<br>" +
            "<extra></extra>"
        )
    ))

    fig.update_layout(
        clickmode='event+select',
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        height=300,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    return dcc.Graph(id='level1-scatter', figure=fig, style={'height': '100%'})
