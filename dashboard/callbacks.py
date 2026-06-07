from dash.dependencies import Input, Output, State, ALL
import dash
import plotly.graph_objects as go
import plotly.express as px
from dash import html
import numpy as np
import json
from .mock_data import MOCK_DATA

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
        
    maze_bg = html.Div(glyphs, style={'width': '100%', 'height': '100%', 'position': 'relative'})
    
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

def update_level3_logic(token_clicks, clickData, triggered_id_full):
    if not any(token_clicks) or not clickData:
        return go.Figure(), go.Figure(), go.Figure(), "Level 3: Token Details", ""

    token_id = json.loads(triggered_id_full)['index']
    sample_id = clickData['points'][0]['hovertext']
    sample = next(s for s in MOCK_DATA if s['sample_id'] == sample_id)
    token = next(t for t in sample['tokens'] if t['token_id'] == token_id)
    
    # RQ1: Spatial Focus Heatmap
    fig_heatmap = px.imshow(token['spatial_focus'], color_continuous_scale='Viridis')
    fig_heatmap.update_layout(
        margin=dict(l=5, r=5, t=20, b=5), 
        title=dict(text=f"RQ1: Focus {token_id}", font=dict(size=10)),
        coloraxis_showscale=False
    )
    
    # RQ2: Information Content Probe Bar
    dirs = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    accs = [token['probe_accuracy'] if d == sample['move_direction'] else np.random.uniform(0.1, 0.4) for d in dirs]
    fig_bar = px.bar(x=dirs, y=accs)
    fig_bar.update_layout(
        margin=dict(l=5, r=5, t=20, b=5), 
        title=dict(text=f"RQ2: Info {token_id}", font=dict(size=10)), 
        yaxis=dict(range=[0,1], tickfont=dict(size=8)),
        xaxis=dict(tickfont=dict(size=8))
    )
    
    # RQ3: Causal Dependence Curve
    steps = list(range(10))
    kls = [token['kl_divergence'] * np.exp(-0.2 * s) for s in steps]
    fig_curve = px.line(x=steps, y=kls)
    fig_curve.update_layout(
        margin=dict(l=5, r=5, t=20, b=5), 
        title=dict(text=f"RQ3: Causal {token_id}", font=dict(size=10)),
        xaxis=dict(tickfont=dict(size=8)),
        yaxis=dict(tickfont=dict(size=8))
    )
    
    return fig_heatmap, fig_bar, fig_curve, f"Details: {token_id} ({sample_id})", ""

def ablate_token_logic(n_clicks):
    if n_clicks:
        return f"Ablation Result: DOWN (95%) -> LEFT (40%)"
    return ""

def register_callbacks(app):
    
    @app.callback(
        [Output('level2-image-pane', 'children'),
         Output('level2-flow-pane', 'figure')],
        [Input('level1-scatter', 'clickData')]
    )
    def update_level2(clickData):
        return update_level2_logic(clickData)

    @app.callback(
        [Output('token-heatmap', 'figure'),
         Output('token-probe-bar', 'figure'),
         Output('token-dependency-curve', 'figure'),
         Output('level3-instructions', 'children'),
         Output('ablate-output', 'children')],
        [Input({'type': 'token-glyph', 'index': ALL}, 'n_clicks'),
         Input('ablate-btn', 'n_clicks')],
        [State('level1-scatter', 'clickData')]
    )
    def update_level3(token_clicks, ablate_clicks, clickData):
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        triggered_id_full = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if triggered_id_full == 'ablate-btn':
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, ablate_token_logic(ablate_clicks)
        
        return update_level3_logic(token_clicks, clickData, triggered_id_full)
        
    @app.callback(
        Output('sensitivity-subplots', 'figure'),
        Input('level1-scatter', 'id')
    )
    def update_sensitivity(_):
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=1, cols=3, subplot_titles=("Temp", "Res", "Seed"))
        fig.add_trace(go.Scatter(y=np.random.rand(10), mode='lines'), row=1, col=1)
        fig.add_trace(go.Bar(y=np.random.rand(5)), row=1, col=2)
        fig.add_trace(go.Box(y=np.random.rand(20)), row=1, col=3)
        fig.update_layout(height=150, margin=dict(l=5, r=5, t=20, b=5), showlegend=False, font=dict(size=8))
        return fig
