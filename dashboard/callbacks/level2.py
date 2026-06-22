import base64
import json
import tarfile
from pathlib import Path
from functools import lru_cache

from dash.dependencies import Input, Output, State, ALL
from dash import dcc, html, callback_context, no_update
import plotly.graph_objects as go
from mirage_vlm.utils.grid_gen import maze_renderer
from ..data_loader import LOADER
from ..gen_data import reroute_plan
from .ablation_v2 import (
    load_moves, load_combo_dist, load_clean_plan,
    sample_dose_response, current_combo_metrics, mask_for_ranks,
    subset_lattice,
)
from ..rq2_viz import build_rq2_static_bar, build_rq2_dynamic_grid


def _load_maze_image(path):
    if not path:
        return None
    img_path = Path(path)
    if not img_path.exists():
        archive_path = Path('data/vsp_spatial_planning.tar.gz')
        if not archive_path.exists():
            return None

        member_suffix = path.split('/img/', 1)[-1] if '/img/' in path else path.lstrip('./')
        member_name = f"./img/{member_suffix}" if not member_suffix.startswith('img/') else f"./{member_suffix}"
        data = _read_tar_member(archive_path, member_name)
        if data is None:
            return None
        return f"data:image/png;base64,{base64.b64encode(data).decode('ascii')}"

    data = base64.b64encode(img_path.read_bytes()).decode('ascii')
    suffix = img_path.suffix.lower().lstrip('.') or 'png'
    return f"data:image/{suffix};base64,{data}"


@lru_cache(maxsize=256)
def _read_tar_member(archive_path, member_name):
    archive_path = Path(archive_path)
    with tarfile.open(archive_path, 'r:gz') as tar:
        try:
            extracted = tar.extractfile(member_name)
        except KeyError:
            extracted = None
        if extracted is None:
            return None
        return extracted.read()


def _get_sample_id(clickData):
    """Extract sample_id from a Level 1 scatter click, or None."""
    return clickData['points'][0]['hovertext'] if clickData else None


def _ablated_ranks(state):
    """Get ablated token ranks from ablation-state store dict."""
    return (state or {}).get('ablated_ranks', [])


def _kl_fingerprint(sample_id, ablated_ranks):
    """Combinatorial KL fingerprint: one cell per subset, grouped by k, shaded by KL,
    amber outline on current selection, red border on plan-flipping combos."""
    lat = subset_lattice(sample_id)
    if not lat or not lat['cells']:
        return html.Div('No combinatorial data for this sample',
                        className='small text-muted', style={'padding': '8px'})

    n = lat['n']
    max_kl = lat['max_kl'] or 1e-9
    cur_mask = mask_for_ranks(ablated_ranks, n) if ablated_ranks else None

    def _dot(filled):
        return html.Span(style={
            'width': '4px', 'height': '4px', 'borderRadius': '50%', 'display': 'inline-block',
            'backgroundColor': '#0f172a' if filled else 'rgba(15,23,42,0.18)',
        })

    def _cell(c):
        alpha = 0.12 + 0.88 * (c['kl'] / max_kl)
        selected = cur_mask is not None and c['mask'] == cur_mask
        changed = c.get('changed', False)
        label = '+'.join('T' + str(r) for r in c['ranks'])
        border_color = '#eab308' if selected else ('#dc3545' if changed else 'transparent')
        hover = f"{label}  ·  KL {c['kl']:.5f}"
        if changed:
            hover += '  ·  plan changed'
        hover += '  ·  click to ablate this set'
        return html.Div(
            [_dot(i in c['ranks']) for i in range(n)],
            id={'type': 'fingerprint-cell', 'mask': c['mask']}, n_clicks=0,
            title=hover,
            style={
                'display': 'flex', 'gap': '1px', 'justifyContent': 'center', 'alignItems': 'center',
                'padding': '3px 2px', 'borderRadius': '3px', 'cursor': 'pointer',
                'backgroundColor': f'rgba(6,182,212,{alpha:.3f})',
                'border': f'1.5px solid {border_color}',
            })

    columns = []
    for k in range(1, n + 1):
        kcells = sorted((c for c in lat['cells'] if c['k'] == k), key=lambda c: -c['kl'])
        columns.append(html.Div([
            html.Div(f'k{k}', style={
                'fontSize': '0.45rem', 'color': '#94a3b8', 'fontFamily': 'monospace',
                'textAlign': 'center', 'marginBottom': '2px',
            }),
            html.Div([_cell(c) for c in kcells],
                     style={'display': 'flex', 'flexDirection': 'column', 'gap': '2px'}),
        ], style={'flex': 1, 'minWidth': 0}))

    return html.Div([
        html.Div('RQ3: KL Fingerprint', style={
            'fontSize': '0.75rem', 'color': '#333',
            'fontFamily': '"Open Sans", verdana, arial, sans-serif',
            'padding': '2px 4px 1px',
        }),
        html.Div('shade = KL · dots = tokens zeroed (T0–T5) · red = plan changed · hover for value', style={
            'fontSize': '0.5rem', 'color': '#666', 'padding': '0 4px 4px',
        }),
        html.Div(columns, style={
            'display': 'flex', 'gap': '3px', 'padding': '0 4px 4px', 'alignItems': 'flex-start',
        }),
    ], style={'height': '100%'})


def _maze_view(sample, ablated_ranks=None):
    sample_id = sample.get('sample_id')
    map_desc = sample.get('map_desc')
    base_plan = load_clean_plan(sample_id) or []
    ablated_path = None

    if ablated_ranks is not None:
        rr = reroute_plan(sample_id, ablated_ranks)
        if rr:
            _, ablated_path = rr

    # Prefer the generated grid image when map data is available.
    if map_desc:
        image_src = maze_renderer(map_desc, base_plan, ablated_path=ablated_path)
    else:
        print(f"Warning: No map_desc for sample {sample_id}, falling back to image_input.")
        image_src = _load_maze_image(sample.get('metadata', {}).get('image_input'))

    if not image_src:
        return html.Div('Maze image unavailable.', className='small text-muted p-2')

    return html.Div([
        html.Img(src=image_src, style={
            'maxWidth': '100%',
            'maxHeight': '100%',
            'objectFit': 'contain',
            'border': '1px solid #dee2e6',
            'backgroundColor': '#f8f9fa'
        })
    ], style={'width': '100%', 'height': '100%', 'display': 'flex',
              'alignItems': 'center', 'justifyContent': 'center',
              'overflow': 'hidden'})


def _token_grid(sample):
    tiles = []
    metadata = sample.get('metadata', {})
    image_src = _load_maze_image(metadata.get('image_input'))
    for i, token in enumerate(sample['tokens'][:6]):
        heatmap = go.Figure(data=go.Heatmap(
            z=token['spatial_focus'],
            colorscale='Viridis',
            showscale=False,
            opacity=0.45
        ))
        heatmap.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title=None,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        tiles.append(
            html.Div([
                html.Div(style={
                    'position': 'absolute',
                    'inset': 0,
                    'backgroundImage': f'url("{image_src}")' if image_src else 'none',
                    'backgroundSize': 'cover',
                    'backgroundPosition': 'center',
                    'backgroundRepeat': 'no-repeat',
                    'opacity': 0.92,
                }),
                dcc.Graph(
                    id={'type': 'token-heatmap', 'index': token['token_id']},
                    figure=heatmap,
                    config={'displayModeBar': False, 'staticPlot': False},
                    style={'height': '100%', 'position': 'relative', 'zIndex': 1, 'backgroundColor': 'transparent'}
                ),
                html.Div('ABLATED', id={'type': 'token-ablated-badge', 'index': i}, style={
                    'position': 'absolute', 'top': '2px', 'right': '2px',
                    'backgroundColor': '#dc3545', 'color': 'white',
                    'fontSize': '8px', 'padding': '1px 4px', 'borderRadius': '3px',
                    'zIndex': 3, 'display': 'none'
                })
            ], id={'type': 'token-tile-wrapper', 'index': i}, style={
                **_TOKEN_TILE_BASE,
                'border': '1px solid #dee2e6',
            })
        )

    return html.Div([
        html.Div("Spatial Focus (latent token heatmaps — clickable)", style={
            'fontSize': '0.55rem', 'color': '#9ca3af',
            'textTransform': 'uppercase', 'letterSpacing': '0.05em',
            'padding': '1px 4px', 'flexShrink': 0,
        }),
        html.Div(tiles, style={
            'display': 'flex', 'gap': '4px', 'alignItems': 'center',
            'height': '10vh', 'overflowX': 'auto', 'flex': 1,
        }),
        html.Div(id='token-flip-readout'),
    ], style={'display': 'flex', 'flexDirection': 'column', 'height': '100%'})


_DIR_GLYPH = {'UP': '↑', 'DOWN': '↓', 'LEFT': '←', 'RIGHT': '→'}

_TOKEN_TILE_BASE = {
    'position': 'relative',
    'width': '10vh', 'height': '10vh', 'flexShrink': 0,
    'overflow': 'hidden',
    'borderRadius': '4px',
    'backgroundColor': '#f8f9fa',
}


def _plan_row(clean_plan, mean_clean_conf, mean_abl_conf, new_plan=None):
    """Predicted-plan label with direction glyphs. If rerouted, shows new plan in red and original struck-through."""
    rerouted = new_plan is not None and list(new_plan) != list(clean_plan)
    plan = list(new_plan) if rerouted else list(clean_plan)
    accent = '#dc3545' if rerouted else '#06b6d4'

    glyphs = [html.Span(_DIR_GLYPH.get(d, '?'), style={
        'fontSize': '0.95rem', 'color': accent, 'fontWeight': 'bold',
    }) for d in plan]

    children = [
        html.Span('Predicted plan — rerouted' if rerouted else 'Predicted plan', style={
            'fontSize': '0.55rem', 'color': accent if rerouted else '#888',
            'textTransform': 'uppercase', 'letterSpacing': '0.1em', 'marginBottom': '2px',
            'fontWeight': 'bold' if rerouted else 'normal',
        }),
        html.Div(glyphs or html.Span('—', style={'fontSize': '0.8rem', 'color': '#bbb'}), style={
            'display': 'flex', 'alignItems': 'center', 'gap': '5px', 'padding': '2px 10px',
            'border': f'1px solid {accent}{"80" if rerouted else "40"}', 'borderRadius': '5px',
            'backgroundColor': f'{accent}{"14" if rerouted else "0d"}',
        }),
    ]
    if rerouted:
        children.append(html.Div(
            [html.Span('was', style={'color': '#aaa', 'marginRight': '4px'})] +
            [html.Span(_DIR_GLYPH.get(d, '?'), style={'color': '#aaa', 'marginLeft': '2px'})
             for d in clean_plan],
            style={'fontSize': '0.5rem', 'marginTop': '2px', 'textDecoration': 'line-through'}))

    return html.Div(children, style={
        'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'padding': '4px 0'})


def _dose_response_graph(sample_id, ablated_ranks):
    """KL-vs-(tokens zeroed) curve with min-max band, both flip-rate series on a
    secondary axis, and the current selection marked as a 'you are here' dot."""
    dr = sample_dose_response(sample_id)
    if not dr or not dr['rows']:
        return html.Div('No ablation landscape data', className='small text-muted', style={'padding': '4px 8px'})

    rows = dr['rows']
    ks = [r['k'] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=[r['kl_hi'] for r in rows], mode='lines',
                             line=dict(width=0), hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=ks, y=[r['kl_lo'] for r in rows], mode='lines',
                             line=dict(width=0), fill='tonexty', fillcolor='rgba(6,182,212,0.12)',
                             hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=ks, y=[r['kl_median'] for r in rows], mode='lines+markers',
                             line=dict(color='#06b6d4', width=2), marker=dict(size=5),
                             hovertemplate='zero %{x} tokens<br>median KL %{y:.4f}<extra></extra>',
                             name='median KL', showlegend=True))
    fig.add_trace(go.Scatter(x=ks, y=[r['plan_flip_pct'] for r in rows], mode='lines', yaxis='y2',
                             line=dict(color='#94a3b8', width=1.5, dash='dot'),
                             hovertemplate='plan flip %{y:.0f}%<extra></extra>',
                             name='plan flip', showlegend=True))

    cur = current_combo_metrics(sample_id, ablated_ranks)
    if cur:
        fig.add_trace(go.Scatter(x=[cur['k']], y=[cur['kl']], mode='markers',
                                 marker=dict(size=11, color='#eab308',
                                             line=dict(color='#92400e', width=1.5)),
                                 hovertemplate='current: zero %{x} tokens<br>KL %{y:.4f}<extra></extra>',
                                 name='current selection', showlegend=True))

    fig.update_layout(
        title=dict(text='RQ3: Dose-Response Curve', font=dict(size=12)),
        template='plotly_white',
        margin=dict(l=30, r=28, t=40, b=18), font=dict(size=8),
        hovermode='x unified', paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='v', x=1, y=1, xanchor='left', yanchor='top',
                    font=dict(size=7), bgcolor='rgba(255,255,255,0.7)'),
        xaxis=dict(title=dict(text='latent tokens zeroed', font=dict(size=8)),
                   dtick=1, tickfont=dict(size=7)),
        yaxis=dict(title=dict(text='KL', font=dict(size=8)), tickfont=dict(size=7), rangemode='tozero'),
        yaxis2=dict(overlaying='y', side='right', range=[0, 100], tickfont=dict(size=7),
                    tickvals=[0, 50, 100], ticktext=['0', '50', '100%']),
    )
    return html.Div([
        html.Div(dcc.Graph(figure=fig, config={'displayModeBar': False},
                           style={'height': '100%', 'width': '100%'}),
                 style={'flex': 1, 'minHeight': 0, 'overflow': 'hidden'}),
    ], style={'height': '32vh', 'display': 'flex',
              'flexDirection': 'column', 'padding': '0 4px'})


def _mean_top_conf(dists):
    """Mean of the top-probability per move distribution, or None if empty."""
    confs = [d[max(d, key=d.get)] for d in dists if d]
    return sum(confs) / len(confs) if confs else None


def _ablated_plan_row(sample_id, ablated_ranks):
    """Return (clean_plan, mean_clean_conf, mean_abl_conf) from per-move distributions.
    Confidences default to 1.0 when distributions are unavailable or argmax-saturated."""
    if len(ablated_ranks) == 1:
        moves = load_moves(sample_id, ablated_ranks[0])
        if moves:
            clean_plan = [max(m['clean'], key=m['clean'].get) for m in moves]
            mean_cc = _mean_top_conf([m['clean'] for m in moves])
            mean_ac = _mean_top_conf([m['ablated'] for m in moves])
            return clean_plan, mean_cc or 1.0, mean_ac or 1.0

    clean_plan = load_clean_plan(sample_id) or []
    combo_mc, combo_ma, _, _ = load_combo_dist(sample_id, mask_for_ranks(ablated_ranks))
    if combo_mc and combo_ma:
        mean_cc = _mean_top_conf([mc.get('probs', mc) for mc in combo_mc])
        mean_ac = _mean_top_conf(combo_ma)
        return clean_plan, mean_cc or 1.0, mean_ac or 1.0
    return clean_plan, 1.0, 1.0


def _render_output_panel(sample_id, ablated_ranks, show_strip=False):
    """Dose-response graph for the ablation tab."""
    if not show_strip:
        return html.Div()
    return _dose_response_graph(sample_id, ablated_ranks)


def update_level2_logic(clickData, data=None):
    """Return maze view for a sample click; used by callback and test suite."""
    if not clickData:
        return html.Div(className='p-2')
    if data is None:
        data = LOADER.get_data()
    sample_id = _get_sample_id(clickData)
    sample = next(s for s in data if s['sample_id'] == sample_id)
    return _maze_view(sample)


def register_level2_callbacks(app):
    @app.callback(
        [Output('level2-maze-pane', 'children'),
         Output('ablation-state', 'data'),
         Output('level2-header-title', 'children')],
        [Input('level1-scatter', 'clickData'),
         Input('ablation-state', 'data')]
    )
    def update_level2(clickData, ablation_state):
        if not clickData:
            return html.Div(className='p-2'), {'ablated_ranks': []}, "Level 2: Reasoning Path Analysis"
        sample_id = _get_sample_id(clickData)
        sample = next(s for s in LOADER.get_data() if s['sample_id'] == sample_id)
        return _maze_view(sample, _ablated_ranks(ablation_state)), {'ablated_ranks': _ablated_ranks(ablation_state)}, f"Level 2: Reasoning Path Analysis — {sample_id}"

    @app.callback(
        Output('level2-kl-pane', 'children'),
        [Input('level1-scatter', 'clickData'),
         Input('level2-tab-selector', 'value'),
         Input('ablation-state', 'data')]
    )
    def update_kl_pane(clickData, active_tab, ablation_state):
        if active_tab != 'ablation':
            return html.Div(style={'height': '100%'})
        if not clickData:
            return html.Div(className='p-2')
        sample_id = _get_sample_id(clickData)
        return html.Div(_kl_fingerprint(sample_id, _ablated_ranks(ablation_state)),
                        style={'flex': 1, 'minWidth': 0, 'overflow': 'auto', 'height': '100%'})

    @app.callback(
        Output('level2-probing-pane', 'children'),
        [Input('level1-scatter', 'clickData'),
         Input('level2-tab-selector', 'value')]
    )
    def update_probing_pane(clickData, active_tab):
        if not clickData:
            return html.Div("Select a Sample (Level 1)", style={
                'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center',
                'height': '100%', 'color': '#888', 'fontSize': '0.65rem',
                'textTransform': 'uppercase', 'letterSpacing': '0.05em',
                'padding': '8px',
            })
        if active_tab != 'probing':
            return html.Div()
        sample_id = _get_sample_id(clickData)
        return html.Div([
            dcc.Graph(id='rq2-dynamic-grid',
                      figure=build_rq2_dynamic_grid(sample_id),
                      config={'responsive': True},
                      style={'flex': 1, 'minWidth': 0}),
            dcc.Graph(id='rq2-static-bar',
                      figure=build_rq2_static_bar(),
                      config={'responsive': True},
                      style={'flex': 1, 'minWidth': 0}),
        ], style={'display': 'flex', 'flexDirection': 'row', 'height': '100%'})

    @app.callback(
        Output('level2-plan-status-row', 'children'),
        [Input('level1-scatter', 'clickData'),
         Input('ablation-state', 'data')]
    )
    def update_plan_status(clickData, ablation_state):
        if not clickData:
            return html.Div(className='p-1')
        sample_id = _get_sample_id(clickData)
        ablated = _ablated_ranks(ablation_state)
        plan, cc, ac = _ablated_plan_row(sample_id, ablated)
        rr = reroute_plan(sample_id, ablated)
        if rr:
            base, new = rr
            return html.Div(_plan_row(base, cc, ac, new_plan=new),
                            style={'display': 'flex', 'justifyContent': 'center'})
        return html.Div(_plan_row(plan, cc, ac),
                        style={'display': 'flex', 'justifyContent': 'center'})

    @app.callback(
        Output('level2-token-grid', 'children'),
        [Input('level1-scatter', 'clickData')]
    )
    def update_token_grid(clickData):
        if not clickData:
            return html.Div(className='p-2')
        sample_id = _get_sample_id(clickData)
        sample = next(s for s in LOADER.get_data() if s['sample_id'] == sample_id)
        return _token_grid(sample)

    @app.callback(
        Output({'type': 'token-tile-wrapper', 'index': ALL}, 'style'),
        Output({'type': 'token-ablated-badge', 'index': ALL}, 'style'),
        [Input('ablation-state', 'data')],
        [State({'type': 'token-tile-wrapper', 'index': ALL}, 'id')]
    )
    def update_token_tile_styles(ablation_state, ids):
        ablated = _ablated_ranks(ablation_state)
        wrapper_styles = []
        badge_styles = []
        for wrapper_id in ids:
            i = wrapper_id['index']
            is_ablated = i in ablated
            border_color = '#dc3545' if is_ablated else '#dee2e6'
            border_width = '3px' if is_ablated else '1px'
            wrapper_styles.append({
                **_TOKEN_TILE_BASE,
                'border': f'{border_width} solid {border_color}',
            })
            badge_styles.append({
                'position': 'absolute', 'top': '2px', 'right': '2px',
                'backgroundColor': '#dc3545', 'color': 'white',
                'fontSize': '8px', 'padding': '1px 4px', 'borderRadius': '3px',
                'zIndex': 3, 'display': 'block' if is_ablated else 'none'
            })
        return wrapper_styles, badge_styles

    @app.callback(
        Output('level2-output-pane', 'children'),
        [Input('level1-scatter', 'clickData'),
         Input('ablation-state', 'data'),
         Input('level2-tab-selector', 'value')]
    )
    def update_output_pane(clickData, ablation_state, active_tab):
        if not clickData:
            return html.Div()
        sample_id = _get_sample_id(clickData)
        return _render_output_panel(sample_id, _ablated_ranks(ablation_state), show_strip=(active_tab == 'ablation'))

    @app.callback(
        Output('ablation-state', 'data', allow_duplicate=True),
        Input({'type': 'fingerprint-cell', 'mask': ALL}, 'n_clicks'),
        State('ablation-state', 'data'),
        prevent_initial_call=True,
    )
    def set_ablation_from_fingerprint(_n_clicks, ablation_state):
        ctx = callback_context
        if not ctx.triggered:
            return no_update
        trig = ctx.triggered[0]
        if not trig['value']:  # recreation / no real click
            return no_update
        try:
            mask = json.loads(trig['prop_id'].split('.n_clicks')[0])['mask']
        except Exception:
            return no_update
        ranks = [i for i, b in enumerate(mask) if b == '1']
        current = sorted(_ablated_ranks(ablation_state))
        new_ranks = [] if ranks == current else ranks
        return {'ablated_ranks': new_ranks}
