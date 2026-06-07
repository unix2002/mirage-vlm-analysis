from dash import dcc
import plotly.express as px
import pandas as pd
from ..mock_data import MOCK_DATA

def create_level1_landscape():
    df = pd.DataFrame([{
        'sample_id': s['sample_id'],
        'umap_x': s['umap_x'],
        'umap_y': s['umap_y'],
        'correctness': str(s['correctness']),
        'move_direction': s['move_direction']
    } for s in MOCK_DATA])
    
    fig = px.scatter(
        df, x='umap_x', y='umap_y', 
        color='correctness', symbol='move_direction',
        hover_name='sample_id'
    )
    # Compact layout for narrow sidebar
    fig.update_layout(
        clickmode='event+select', 
        margin=dict(l=5, r=5, t=5, b=5),
        showlegend=False,
        height=300 # Will be overridden by card body height if using responsive graph
    )
    
    return dcc.Graph(id='level1-scatter', figure=fig, style={'height': '100%'})
