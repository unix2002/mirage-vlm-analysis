from dash.dependencies import Input, Output, State, ALL
import dash
import plotly.express as px
import plotly.graph_objects as go
import json
from ..mock_data import MOCK_DATA


def _extract_active_click(token_clicks):
    if not token_clicks:
        return None
    for item in token_clicks:
        if item:
            return item
    return None


def update_level3_logic(token_clicks, clickData, triggered_id_full):
    active_click = _extract_active_click(token_clicks)
    if not active_click or not clickData:
        return go.Figure(), go.Figure(), go.Figure(), "Level 3: Token Details", ""

    # Strip property if present (e.g. .n_clicks)
    if '.' in triggered_id_full:
        triggered_id_full = triggered_id_full.split('.')[0]

    token_id = json.loads(triggered_id_full)['index']
    sample_id = clickData['points'][0]['hovertext']
    sample = next(s for s in MOCK_DATA if s['sample_id'] == sample_id)
    token = next(t for t in sample['tokens'] if t['token_id'] == token_id)

    # RQ1: Spatial Focus Heatmap
    fig_heatmap = px.imshow(
        token['spatial_focus'], color_continuous_scale='Viridis')
    fig_heatmap.update_layout(
        margin=dict(l=5, r=5, t=20, b=5),
        title=dict(text=f"RQ1: Spatial Focus Heatmap (Token {token_id})", font=dict(size=10)),
        coloraxis_showscale=True,
        xaxis=dict(title="Column"),
        yaxis=dict(title="Row")
    )

    # RQ2: Information Content Probe Bar
    dirs = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    base = max(0.0, min(1.0, float(token['probe_accuracy'])))
    off_value = max(0.0, min(1.0, base * 0.35))
    accs = [base if d == sample['move_direction'] else off_value for d in dirs]
    fig_bar = px.bar(x=dirs, y=accs, labels={'x': 'Direction', 'y': 'Probe Accuracy'})
    fig_bar.update_layout(
        margin=dict(l=5, r=5, t=20, b=5),
        title=dict(text=f"RQ2: Directional Probe Accuracy (Token {token_id})", font=dict(size=10)),
        yaxis=dict(range=[0, 1], tickfont=dict(size=8)),
        xaxis=dict(tickfont=dict(size=8))
    )

    # RQ3: Causal Dependence Curve
    steps = list(range(10))
    decay = 0.82
    kls = [token['kl_divergence'] * (decay ** s) for s in steps]
    fig_curve = px.line(x=steps, y=kls, labels={'x': 'Step', 'y': 'KL Divergence'})
    fig_curve.update_layout(
        margin=dict(l=5, r=5, t=20, b=5),
        title=dict(text=f"RQ3: Causal KL Dependence (Token {token_id})", font=dict(size=10)),
        xaxis=dict(tickfont=dict(size=8)),
        yaxis=dict(tickfont=dict(size=8))
    )

    return fig_heatmap, fig_bar, fig_curve, f"Details: {token_id} ({sample_id})", ""


def ablate_token_logic(n_clicks):
    if n_clicks:
        return f"Ablation Result: DOWN (95%) -> LEFT (40%)"
    return ""


def register_level3_callbacks(app):
    @app.callback(
        [Output('token-detail-heatmap', 'figure'),
         Output('token-detail-probe-bar', 'figure'),
         Output('token-detail-dependency-curve', 'figure'),
         Output('level3-instructions', 'children'),
         Output('ablate-output', 'children')],
        [Input({'type': 'token-heatmap', 'index': ALL}, 'clickData'),
         Input('ablate-btn', 'n_clicks')],
        [State('level1-scatter', 'clickData')]
    )
    def update_level3(token_clicks, ablate_clicks, clickData):
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        triggered_id_full = ctx.triggered[0]['prop_id']

        # Check if ablate-btn triggered
        if 'ablate-btn' in triggered_id_full:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, ablate_token_logic(ablate_clicks)

        active_click = _extract_active_click(token_clicks)
        if not active_click:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        token_index = None
        if 'token-heatmap' in triggered_id_full:
            try:
                token_index = json.loads(triggered_id_full.split('.')[0])['index']
            except Exception:
                token_index = None

        if token_index is None:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        return update_level3_logic(active_click, clickData, json.dumps({'index': token_index}))
