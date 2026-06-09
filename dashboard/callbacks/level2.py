from dash.dependencies import Input, Output
import dash
import plotly.graph_objects as go
from dash import html
import numpy as np
from ..mock_data import MOCK_DATA


def update_level2_logic(clickData):
    if not clickData:
        return html.Div("Select a Sample (Level 1)", className="p-2 text-muted small"), go.Figure()

    sample_id = clickData['points'][0]['hovertext']
    sample = next(s for s in MOCK_DATA if s['sample_id'] == sample_id)

    # Build Left Panel: Image-Anchored View
    glyphs = []
    for i, token in enumerate(sample['tokens']):
        size = max(10, min(35, token['kl_divergence'] * 20))
        r = int(255 * (1 - token['probe_accuracy']))
        g = int(255 * token['probe_accuracy'])
        color = f"rgba({r}, {g}, 0, 0.8)"

        top = f"{20 + i * 12}%"
        left = f"{10 + (i%2)*20 + i*10}%"

        glyph = html.Div(
            id={'type': 'token-glyph', 'index': token['token_id']},
            style={
                'position': 'absolute', 'top': top, 'left': left,
                'width': f"{size}px", 'height': f"{size}px",
                'backgroundColor': color, 'borderRadius': '50%',
                'border': '1px solid black', 'cursor': 'pointer',
            },
            title=f"{token['token_id']}"
        )
        glyphs.append(glyph)

    maze_bg = html.Div(
        glyphs, style={'width': '100%', 'height': '100%', 'position': 'relative'})

    # Build Right Panel: Sequential Flow
    attn = np.array(sample['attention_weights'])
    fig_flow = go.Figure(data=go.Heatmap(
        z=attn, x=[f"T{i}" for i in range(6)], y=[f"T{i}" for i in range(6)],
        colorscale='Blues', showscale=False
    ))
    fig_flow.update_layout(
        margin=dict(l=5, r=5, t=5, b=5),
        xaxis=dict(tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
        font=dict(size=10)
    )

    return maze_bg, fig_flow


def register_level2_callbacks(app):
    @app.callback(
        [Output('level2-image-pane', 'children'),
         Output('level2-flow-pane', 'figure')],
        [Input('level1-scatter', 'clickData')]
    )
    def update_level2(clickData):
        return update_level2_logic(clickData)
