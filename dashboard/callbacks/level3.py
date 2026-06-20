from dash.dependencies import Input, Output, State, ALL
import dash
import plotly.express as px
import plotly.graph_objects as go
import json
from ..mock_data import MOCK_DATA
from .ablation_v2 import load_per_token_summary, build_bitmask, load_combo_file, _get_token_positions
from dash import html


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
        return go.Figure(), go.Figure(), go.Figure(), "Level 3: Token Details", {}

    if '.' in triggered_id_full:
        triggered_id_full = triggered_id_full.split('.')[0]

    token_id = json.loads(triggered_id_full)['index']
    sample_id = clickData['points'][0]['hovertext']
    sample = next(s for s in MOCK_DATA if s['sample_id'] == sample_id)
    token = next(t for t in sample['tokens'] if t['token_id'] == token_id)
    token_rank = int(token_id[1:])

    fig_heatmap = px.imshow(
        token['spatial_focus'], color_continuous_scale='Viridis')
    fig_heatmap.update_layout(
        margin=dict(l=5, r=5, t=20, b=5),
        title=dict(text=f"RQ1: Spatial Focus Heatmap (Token {token_id})", font=dict(size=10)),
        coloraxis_showscale=True,
        xaxis=dict(title="Column"),
        yaxis=dict(title="Row")
    )

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
    fig_curve.update_layout(
        margin=dict(l=5, r=5, t=20, b=5),
        title=dict(text=f"RQ3: Per-Position KL after Zeroing Token {token_id}", font=dict(size=10)),
        xaxis=dict(tickfont=dict(size=8)),
        yaxis=dict(tickfont=dict(size=8))
    )

    store_data = {"sample_id": sample_id, "token_id": token_id, "token_rank": token_rank}
    return fig_heatmap, fig_bar, fig_curve, f"Details: {token_id} ({sample_id})", store_data


def _format_combo_output(mask, data, ablated_str):
    kl = data['kl_mean']
    top1 = data['top1']
    acc = data.get('acc_abl', data.get('gt_acc_ablated', '?'))
    nll_delta = data.get('nll_delta', data.get('gt_nll_delta', '?'))
    em = data.get('em_abl', data.get('exact_match_ablated', '?'))

    is_top1_flip = top1 < 1.0
    is_em_flip = not em
    has_flip = is_top1_flip or is_em_flip

    if is_top1_flip:
        label = f"⚠️ FLIP (top1={top1:.3f})"
        color = "#dc3545"
    elif is_em_flip:
        label = f"⚠️ partial flip (acc={acc:.3f})"
        color = "#fd7e14"
    else:
        label = "✓ stable"
        color = "#28a745"

    parts = [f"Ablated {ablated_str}",
             f"KL={kl:.4f}", f"top1={top1:.3f}",
             f"NLL Δ={nll_delta:.4f}"]

    return html.Span([
        html.Span(label, style={
            'fontWeight': 'bold', 'color': color,
            'backgroundColor': '#fff3cd' if has_flip else 'transparent',
            'padding': '1px 6px', 'borderRadius': '3px',
            'marginRight': '8px'
        }),
        " | ".join(parts)
    ])


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
        [Output('ablate-output', 'children'),
         Output('ablation-state', 'data', allow_duplicate=True)],
        [Input('ablate-btn', 'n_clicks')],
        [State('current-token-state', 'data'),
         State('ablation-state', 'data')],
        prevent_initial_call=True,
    )
    def toggle_ablate(n_clicks, token_state, ablation_state):
        if not n_clicks:
            return "", ablation_state or {'ablated_ranks': []}

        state = ablation_state or {'ablated_ranks': []}
        token_state = token_state or {}
        sample_id = token_state.get('sample_id')
        rank = token_state.get('token_rank')

        if sample_id is None or rank is None:
            return "Select a token first", state

        ranks = list(state.get('ablated_ranks', []))
        # If rank is already in set, remove it (toggle off)
        if rank in ranks:
            ranks.remove(rank)
            if not ranks:
                return "No tokens ablated", {'ablated_ranks': []}
            mask, result = build_bitmask(sample_id, ranks)
            if result is None:
                return f"Error: {mask}", {'ablated_ranks': []}
            ablated_str = f"T{' + T'.join(str(r) for r in ranks)}"
            return _format_combo_output(mask, result, ablated_str), {'ablated_ranks': ranks}

        # Trying to add a token — check if it's valid first
        combo = load_combo_file(sample_id)
        if combo is None:
            return "Combinatorial data not available for this sample", state

        token_pos_map = _get_token_positions(sample_id)
        pos = token_pos_map.get(rank) if token_pos_map else None
        if pos is None or pos not in combo['positions']:
            existing = ', '.join(str(t) for t in sorted(state.get('ablated_ranks', [])))
            valid = ', '.join(sorted(f"T{r}" for r in range(6)
                                     if token_pos_map and r in combo['positions']))
            return (f"Token T{rank} is not in the combinatorial study. "
                    f"Only {valid} can be ablated together." +
                    (f" Currently ablated: T{existing}." if existing else "")), state

        ranks.append(rank)
        ranks.sort()
        mask, result = build_bitmask(sample_id, ranks)
        if result is None:
            return f"Error: {mask}", {'ablated_ranks': []}
        ablated_str = f"T{' + T'.join(str(r) for r in ranks)}"
        output = _format_combo_output(mask, result, ablated_str)
        return output, {'ablated_ranks': ranks}
