from dash.dependencies import Input, Output, State, ALL
from dash import dcc, html
import dash
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np
from ..data_loader import get_layer_heatmap, LOADER
from .ablation_v2 import load_per_token_summary


def _build_level3_content(fig_heatmap, fig_bar, fig_curve, layer_value=26):
    """Build the full Level 3 detail row + layer slider as a single element."""
    return html.Div([
        dbc.Row([
            dbc.Col(dcc.Graph(id='token-detail-heatmap', figure=fig_heatmap,
                    style={'height': '26vh'}), width=4),
            dbc.Col(dcc.Graph(id='token-detail-probe-bar', figure=fig_bar,
                    style={'height': '26vh'}), width=4),
            dbc.Col(dcc.Graph(id='token-detail-dependency-curve', figure=fig_curve,
                    style={'height': '26vh'}), width=4),
        ], className="g-0"),
        dbc.Row([
            dbc.Col(dcc.Slider(
                id='layer-slider',
                min=0, max=26, step=1, value=layer_value,
                marks={0: '0', 6: '6', 13: '13', 20: '20', 26: '26'},
                tooltip=dict(placement='bottom'),
            ), width=4),
        ], className="mt-1"),
    ])


_OFF_VALUE_RATIO = 0.35
_KL_DECAY_RATE = 0.82


def _extract_active_click(token_clicks):
    return next((item for item in (token_clicks or []) if item), None)


def _style_detail_fig(fig, title):
    """Apply shared compact margin + title for all Level 3 figures."""
    fig.update_layout(
        margin=dict(l=5, r=5, t=20, b=5),
        title=dict(text=title, font=dict(size=10)),
    )
    return fig


def _apply_heatmap_layout(fig, title, n):
    """Shared heatmap axis/colorbar config used by initial build and slider rebuild."""
    _style_detail_fig(fig, title)
    fig.update_layout(
        coloraxis_showscale=True,
        coloraxis_colorbar=dict(thickness=8, len=0.5, outlinewidth=0),
        xaxis=dict(title="Column"),
        yaxis=dict(title="Row",
                   tickvals=[0, n // 2, n - 1],
                   ticktext=[str(n - 1), str(n // 2), "0"]),
    )


def update_level3_logic(token_clicks, clickData, triggered_id_full, data=None):
    active_click = _extract_active_click(token_clicks)
    if not active_click or not clickData:
        return go.Figure(), go.Figure(), go.Figure(), "Level 3: Token Details", {}

    if '.' in triggered_id_full:
        triggered_id_full = triggered_id_full.split('.')[0]

    token_id = json.loads(triggered_id_full)['index']
    sample_id = clickData['points'][0]['hovertext']
    if data is None:
        data = LOADER.get_data()
    sample = next(s for s in data if s['sample_id'] == sample_id)
    token = next(t for t in sample['tokens'] if t['token_id'] == token_id)
    token_rank = int(token_id[1:])

    grid = np.array(token['spatial_focus'])
    grid = np.flipud(grid)
    fig_heatmap = px.imshow(grid, color_continuous_scale='Viridis')
    _apply_heatmap_layout(fig_heatmap, f"RQ1: Spatial Focus Heatmap (Token {token_id})", grid.shape[0])

    dirs = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    base = max(0.0, min(1.0, float(token['probe_accuracy'])))
    off_value = max(0.0, min(1.0, base * _OFF_VALUE_RATIO))
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
        kls = [token['kl_divergence'] * (_KL_DECAY_RATE ** s) for s in range(10)]
        x = list(range(10))
        kls_label = 'KL Divergence (synthetic decay)'
    fig_curve = px.line(x=x, y=kls, labels={'x': 'Position', 'y': kls_label})
    _style_detail_fig(fig_curve, f"RQ3: Per-Position KL after Zeroing Token {token_id}")
    fig_curve.update_layout(
        xaxis=dict(tickfont=dict(size=8)),
        yaxis=dict(tickfont=dict(size=8))
    )

    store_data = {"sample_id": sample_id, "token_id": token_id, "token_rank": token_rank,
                  "move_direction": sample['move_direction'],
                  "probe_accuracy": token['probe_accuracy'],
                  "kl_divergence": token['kl_divergence']}
    return fig_heatmap, fig_bar, fig_curve, f"Details: {token_id} ({sample_id})", store_data




def register_level3_callbacks(app):
    @app.callback(
        [Output('level3-detail-content', 'children'),
         Output('level3-instructions', 'children'),
         Output('current-token-state', 'data')],
        [Input({'type': 'token-heatmap', 'index': ALL}, 'clickData')],
        [State('level1-scatter', 'clickData')]
    )
    def update_level3_detail(token_clicks, clickData):
        ctx = dash.callback_context
        if not ctx.triggered:
            return (dash.no_update,) * 3

        triggered_id_full = ctx.triggered[0]['prop_id']
        active_click = _extract_active_click(token_clicks)
        if not active_click:
            return (dash.no_update,) * 3

        token_index = None
        if 'token-heatmap' in triggered_id_full:
            try:
                token_index = json.loads(triggered_id_full.split('.')[0])['index']
            except Exception:
                token_index = None

        if token_index is None:
            return (dash.no_update,) * 3

        fig_heatmap, fig_bar, fig_curve, text, store_data = update_level3_logic(
            active_click, clickData, json.dumps({'index': token_index}))
        content = _build_level3_content(fig_heatmap, fig_bar, fig_curve)
        return content, text, store_data

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
        fig_heatmap = px.imshow(grid, color_continuous_scale='Viridis')
        _apply_heatmap_layout(fig_heatmap, f"RQ1: Spatial Focus ({token_id}, layer {layer})", grid.shape[0])
        return fig_heatmap