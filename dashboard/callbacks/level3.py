from dash.dependencies import Input, Output, State, ALL
import dash
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np
from ..mock_data import MOCK_DATA
from .ablation_v2 import load_per_token_summary
from ..data_loader import get_layer_heatmap


def _extract_active_click(token_clicks):
    if not token_clicks:
        return None
    for item in token_clicks:
        if item:
            return item
    return None


def _style_detail_fig(fig, title):
    """Apply the shared compact margin + title used by all Level 3 figures."""
    fig.update_layout(
        margin=dict(l=5, r=5, t=20, b=5),
        title=dict(text=title, font=dict(size=10)),
    )
    return fig


def update_level3_logic(token_clicks, clickData, triggered_id_full):
    active_click = _extract_active_click(token_clicks)
    if not active_click or not clickData:
        return go.Figure(), go.Figure(), go.Figure(), "Level 3: Token Details", {}

    if '.' in triggered_id_full:
        triggered_id_full = triggered_id_full.split('.')[0]

    token_id = json.loads(triggered_id_full)['index']
    sample_id = clickData['points'][0]['hovertext']
    sample = next(s for s in MOCK_DATA if s['sample_id'] == sample_id)
    token = next(t for t in sample['tokens'] if t['token_id'] == token_id)
    token_rank = int(token_id[1:])

    grid = np.array(token['spatial_focus'])
    grid = np.flipud(grid)
    fig_heatmap = px.imshow(grid, color_continuous_scale='Viridis')
    _style_detail_fig(fig_heatmap, f"RQ1: Spatial Focus Heatmap (Token {token_id})")
    n = grid.shape[0]
    fig_heatmap.update_layout(
        coloraxis_showscale=True,
        xaxis=dict(title="Column"),
        yaxis=dict(title="Row",
                   tickvals=[0, n // 2, n - 1],
                   ticktext=[str(n - 1), str(n // 2), "0"]),
    )

    dirs = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    base = max(0.0, min(1.0, float(token['probe_accuracy'])))
    off_value = max(0.0, min(1.0, base * 0.35))
    accs = [base if d == sample['move_direction'] else off_value for d in dirs]
    fig_bar = px.bar(x=dirs, y=accs, labels={'x': 'Direction', 'y': 'Probe Accuracy'})
    _style_detail_fig(fig_bar, f"RQ2: Directional Probe Accuracy (Token {token_id})")
    fig_bar.update_layout(
        yaxis=dict(range=[0, 1], tickfont=dict(size=8)),
        xaxis=dict(tickfont=dict(size=8))
    )

    abl_data, _ = load_per_token_summary(sample_id, token_rank)
    if abl_data and abl_data.get('kl_positions'):
        kls = abl_data['kl_positions']
        x = list(range(len(kls)))
        kls_label = 'KL Divergence (per position)'
    else:
        kls = [token['kl_divergence'] * (0.82 ** s) for s in range(10)]
        x = list(range(10))
        kls_label = 'KL Divergence (synthetic decay)'
    fig_curve = px.line(x=x, y=kls, labels={'x': 'Position', 'y': kls_label})
    _style_detail_fig(fig_curve, f"RQ3: Per-Position KL after Zeroing Token {token_id}")
    fig_curve.update_layout(
        xaxis=dict(tickfont=dict(size=8)),
        yaxis=dict(tickfont=dict(size=8))
    )

    store_data = {"sample_id": sample_id, "token_id": token_id, "token_rank": token_rank}
    return fig_heatmap, fig_bar, fig_curve, f"Details: {token_id} ({sample_id})", store_data




def register_level3_callbacks(app):
    @app.callback(
        [Output('token-detail-heatmap', 'figure'),
         Output('token-detail-probe-bar', 'figure'),
         Output('token-detail-dependency-curve', 'figure'),
         Output('level3-instructions', 'children'),
         Output('current-token-state', 'data')],
        [Input({'type': 'token-heatmap', 'index': ALL}, 'clickData')],
        [State('level1-scatter', 'clickData')]
    )
    def update_level3_detail(token_clicks, clickData):
        ctx = dash.callback_context
        if not ctx.triggered:
            return (dash.no_update,) * 5

        triggered_id_full = ctx.triggered[0]['prop_id']
        active_click = _extract_active_click(token_clicks)
        if not active_click:
            return (dash.no_update,) * 5

        token_index = None
        if 'token-heatmap' in triggered_id_full:
            try:
                token_index = json.loads(triggered_id_full.split('.')[0])['index']
            except Exception:
                token_index = None

        if token_index is None:
            return (dash.no_update,) * 5

        return update_level3_logic(active_click, clickData, json.dumps({'index': token_index}))

    @app.callback(
        Output('token-detail-heatmap', 'figure', allow_duplicate=True),
        [Input('layer-slider', 'value')],
        [State('current-token-state', 'data')],
        prevent_initial_call=True,
    )
    def update_layer_heatmap(layer, token_state):
        if not token_state:
            return dash.no_update
        sid = token_state.get('sample_id')
        token_id = token_state.get('token_id', 'T0')
        token_idx = int(token_id[1:]) if token_id.startswith('T') else 0
        grid = get_layer_heatmap(sid, token_idx, layer)
        if grid is None:
            return dash.no_update
        grid = np.array(grid)
        grid = np.flipud(grid)
        fig = px.imshow(grid, color_continuous_scale='Viridis')
        n = grid.shape[0]
        fig.update_layout(
            margin=dict(l=5, r=5, t=20, b=5),
            coloraxis_showscale=True,
            xaxis=dict(title="Column"),
            yaxis=dict(title="Row",
                       tickvals=[0, n // 2, n - 1],
                       ticktext=[str(n - 1), str(n // 2), "0"]),
            title=dict(text=f"RQ1: Spatial Focus ({token_id}, layer {layer})", font=dict(size=10)),
        )
        return fig

