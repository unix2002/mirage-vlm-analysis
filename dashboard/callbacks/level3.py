from dash.dependencies import Input, Output, State, ALL
import dash
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np
from pathlib import Path
from ..data_loader import get_layer_heatmap, LOADER
from .ablation_v2 import token_marginal_contributions
from .level2 import _load_maze_image

_PROBE_PATH = Path("data/processed/rq2/probe_results_per_sample.json")

try:
    with open(_PROBE_PATH) as f:
        _PROBE_CACHE = json.load(f)
except Exception:
    _PROBE_CACHE = None

_HIDDEN = {'display': 'none'}
_VISIBLE = {'position': 'absolute', 'inset': 0, 'display': 'flex', 'flexDirection': 'column'}


def _extract_active_click(token_clicks):
    return next((item for item in (token_clicks or []) if item), None)


def _style_detail_fig(fig, title):
    """Apply shared compact margin + title for all Level 3 figures."""
    fig.update_layout(
        margin=dict(l=5, r=5, t=30, b=5),
        title=dict(text=title, font=dict(size=16, color='#1f2937')),
        font=dict(size=14, color='#1f2937'),
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


_HEATMAP_TITLE_PAD = 28  # px reserved for the figure title; maze bg starts below it


def _overlay_heatmap_fig(z, title):
    """Latent-token attention heatmap styled as a translucent overlay (like the Step 2
    token tiles): semi-opaque, no colorbar, transparent background so the maze image
    behind it shows through."""
    fig = go.Figure(data=go.Heatmap(
        z=z, colorscale='Viridis', showscale=False, opacity=0.55, hoverinfo='skip'))
    fig.update_layout(
        margin=dict(l=0, r=0, t=_HEATMAP_TITLE_PAD, b=0),
        title=dict(text=title, font=dict(size=16, color='#1f2937')),
        font=dict(size=14, color='#1f2937'),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


def _maze_bg_style(image_src):
    """Background style for the Step 3 heatmap container: the original maze image,
    sitting behind the translucent heatmap overlay.

    The box is the heatmap's plot area exactly (full width, from just below the title
    strip to the bottom), and the image is stretched to fill it (100% 100%) so it lines
    up cell-for-cell with the heatmap, which also fills that same box."""
    return {
        'position': 'absolute', 'top': f'{_HEATMAP_TITLE_PAD}px', 'left': 0, 'right': 0, 'bottom': 0,
        'zIndex': 0,
        'backgroundImage': f'url("{image_src}")' if image_src else 'none',
        'backgroundSize': '100% 100%', 'backgroundPosition': 'center', 'backgroundRepeat': 'no-repeat',
    }


def update_level3_logic(token_clicks, clickData, triggered_id_full, data=None):
    active_click = _extract_active_click(token_clicks)
    if not active_click or not clickData:
        return go.Figure(), go.Figure(), go.Figure(), "Step 3: Token Details", {}

    if '.' in triggered_id_full:
        triggered_id_full = triggered_id_full.split('.')[0]

    token_id = json.loads(triggered_id_full)['index']
    sample_id = clickData['points'][0]['hovertext']
    if data is None:
        data = LOADER.get_data()
    sample = next(s for s in data if s['sample_id'] == sample_id)
    token = next(t for t in sample['tokens'] if t['token_id'] == token_id)
    token_rank = int(token_id[1:])

    fig_heatmap = _overlay_heatmap_fig(
        token['spatial_focus'], f"RQ1: Latent Token Heatmap (Token {token_id})")

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
        margin=dict(l=48, r=5, t=30, b=5),
        yaxis=dict(type='log', title=dict(text='Probability', font=dict(size=13, color='#1f2937')),
                   tickfont=dict(size=14, color='#1f2937')),
        xaxis=dict(tickfont=dict(size=14, color='#1f2937'))
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
            margin=dict(l=48, r=5, t=30, b=5),
            showlegend=False,
            xaxis=dict(tickfont=dict(size=14, color='#1f2937')),
            yaxis=dict(title=dict(text='KL divergence', font=dict(size=13, color='#1f2937')),
                       tickfont=dict(size=14, color='#1f2937')),
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
         Output('current-token-state', 'data'),
         Output('token-detail-heatmap-bg', 'style')],
        [Input({'type': 'token-heatmap', 'index': ALL}, 'clickData')],
        [State('level1-scatter', 'clickData')]
    )
    def update_level3_detail(token_clicks, clickData):
        ctx = dash.callback_context
        if not ctx.triggered:
            return (dash.no_update,) * 8

        triggered_id_full = ctx.triggered[0]['prop_id']
        active_click = _extract_active_click(token_clicks)
        if not active_click:
            return (dash.no_update,) * 8

        token_index = None
        if 'token-heatmap' in triggered_id_full:
            try:
                token_index = json.loads(triggered_id_full.split('.')[0])['index']
            except Exception:
                token_index = None

        if token_index is None:
            return (dash.no_update,) * 8

        fig_heatmap, fig_bar, fig_curve, text, store_data = update_level3_logic(
            active_click, clickData, json.dumps({'index': token_index}))

        # Original maze image to sit behind the translucent heatmap overlay.
        image_src = None
        try:
            sid = clickData['points'][0]['hovertext']
            sample = next((s for s in LOADER.get_data() if s['sample_id'] == sid), None)
            if sample:
                image_src = _load_maze_image(sample.get('metadata', {}).get('image_input'))
        except Exception:
            image_src = None

        return (_HIDDEN, _VISIBLE, fig_heatmap, fig_bar, fig_curve, text, store_data,
                _maze_bg_style(image_src))

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
        fig_heatmap = _overlay_heatmap_fig(
            grid, f"RQ1: Latent Token Heatmap ({token_id}, layer {layer})")
        return fig_heatmap

    _PLACEHOLDER_SHOWN = {'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center',
                          'height': '100%', 'color': '#1f2937', 'fontSize': '14px'}

    @app.callback(
        [Output('level3-placeholder', 'style', allow_duplicate=True),
         Output('level3-content', 'style', allow_duplicate=True),
         Output('current-token-state', 'data', allow_duplicate=True)],
        [Input('level1-scatter', 'clickData')],
        prevent_initial_call=True,
    )
    def reset_level3_on_sample(clickData):
        """Gate the Step 3 placeholder on sample selection.

        Selecting a new sample clears any open token detail and shows the
        'select a latent token' prompt. With no sample selected the prompt stays
        hidden (the global help page covers that empty state instead).
        """
        if not clickData:
            return _HIDDEN, _HIDDEN, {}
        return _PLACEHOLDER_SHOWN, _HIDDEN, {}