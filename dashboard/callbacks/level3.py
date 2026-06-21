from dash.dependencies import Input, Output, State, ALL
import dash
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np
from pathlib import Path
from ..data_loader import get_layer_heatmap, LOADER
from .ablation_v2 import token_marginal_contributions

_PROBE_PATH = Path("data/processed/rq2/probe_results_per_sample.json")

try:
    with open(_PROBE_PATH) as f:
        _PROBE_CACHE = json.load(f)
except Exception:
    _PROBE_CACHE = None

_HIDDEN = {'display': 'none'}
_VISIBLE = {'display': 'flex', 'flexDirection': 'column', 'overflow': 'hidden', 'height': '100%'}


def _extract_active_click(token_clicks):
    return next((item for item in (token_clicks or []) if item), None)


def _style_detail_fig(fig, title):
    """Apply shared compact margin + title for all Level 3 figures."""
    fig.update_layout(
        margin=dict(l=5, r=5, t=20, b=5),
        title=dict(text=title, font=dict(size=10)),
    )
    return fig


def _apply_heatmap_layout(fig, title):
    """Shared heatmap axis/colorbar config used by initial build and slider rebuild."""
    _style_detail_fig(fig, title)
    fig.update_layout(
        coloraxis_showscale=True,
        coloraxis_colorbar=dict(thickness=8, len=0.5, outlinewidth=0),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
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
    _apply_heatmap_layout(fig_heatmap, f"RQ1: Spatial Focus Heatmap (Token {token_id})")

    # RQ2: direction probe probabilities from real probe data
    fig_bar = go.Figure()
    if _PROBE_CACHE:
        per_sample = _PROBE_CACHE.get("per_sample", {})
        sample_key = sample_id.replace("sample_", "").lstrip("0") or "0"
        probe_sample = per_sample.get(sample_key)
        if probe_sample:
            layers = sorted(int(k) for k in probe_sample.get("per_layer", {}).keys())
            best_layer = str(layers[-1]) if layers else "26"
            tdata = probe_sample["per_layer"].get(best_layer, {}).get(
                "token_step_direction_probs", {}).get(str(token_rank), {})
            if tdata:
                step = sorted(tdata.keys(), key=int)[0]
                probs = tdata[step]
                dirs = list(probs.keys())
                values = [probs[d] for d in dirs]
                true_dir = probe_sample.get("true_move_sequence", [None])[0]
                colors = ['#06b6d4' if d == true_dir else '#94a3b8' for d in dirs]
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(x=dirs, y=values, marker_color=colors, showlegend=False))
    _style_detail_fig(fig_bar, f"RQ2: Direction Probe (Token {token_id})")
    fig_bar.update_layout(
        yaxis=dict(type='log', tickfont=dict(size=8)),
        xaxis=dict(tickfont=dict(size=8))
    )

    contribs = token_marginal_contributions(sample_id)
    if contribs:
        ranks = [f'T{c["rank"]}' for c in contribs]
        individs = [c['individual'] for c in contribs]
        margins = [c['marginal'] for c in contribs]
        bar_colors = ['#06b6d4' if c['rank'] == token_rank else '#94a3b8' for c in contribs]
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Bar(x=ranks, y=individs, name='individual KL',
                                     marker_color='#e2e8f0'))
        fig_curve.add_trace(go.Bar(x=ranks, y=margins, name='marginal contribution',
                                     marker_color=bar_colors))
        fig_curve.update_layout(barmode='group')
        _style_detail_fig(fig_curve, f"RQ3: Token Ablation Contributions (Token {token_id})")
        fig_curve.update_layout(
            xaxis=dict(tickfont=dict(size=8)),
            yaxis=dict(tickfont=dict(size=8)),
            legend=dict(font=dict(size=7)),
        )
    else:
        fig_curve = go.Figure()
        _style_detail_fig(fig_curve, f"RQ3: No ablation data (Token {token_id})")

    store_data = {"sample_id": sample_id, "token_id": token_id, "token_rank": token_rank,
                  "move_direction": sample['move_direction'],
                  "probe_accuracy": token['probe_accuracy'],
                  "kl_divergence": token['kl_divergence']}
    return fig_heatmap, fig_bar, fig_curve, f"Details: {token_id} ({sample_id})", store_data




def register_level3_callbacks(app):
    @app.callback(
        [Output('level3-placeholder', 'style'),
         Output('level3-content', 'style'),
         Output('token-detail-heatmap', 'figure'),
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
            return (dash.no_update,) * 7

        triggered_id_full = ctx.triggered[0]['prop_id']
        active_click = _extract_active_click(token_clicks)
        if not active_click:
            return (dash.no_update,) * 7

        token_index = None
        if 'token-heatmap' in triggered_id_full:
            try:
                token_index = json.loads(triggered_id_full.split('.')[0])['index']
            except Exception:
                token_index = None

        if token_index is None:
            return (dash.no_update,) * 7

        fig_heatmap, fig_bar, fig_curve, text, store_data = update_level3_logic(
            active_click, clickData, json.dumps({'index': token_index}))
        return _HIDDEN, _VISIBLE, fig_heatmap, fig_bar, fig_curve, text, store_data

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
        _apply_heatmap_layout(fig_heatmap, f"RQ1: Spatial Focus ({token_id}, layer {layer})")
        return fig_heatmap