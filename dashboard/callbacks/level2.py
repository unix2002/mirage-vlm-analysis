import base64
import json
import math
import tarfile
from pathlib import Path
from functools import lru_cache

from dash.dependencies import Input, Output, State, ALL
from dash import dcc, html, callback_context, no_update
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from ..mock_data import MOCK_DATA
from .ablation_v2 import (
    load_moves, load_combo_dist, load_clean_plan, load_per_token_summary,
    sample_dose_response, token_marginal_contributions, current_combo_metrics, mask_for_ranks,
    toggle_rank,
)


ABLATED_DATA = json.loads(Path('data/ablation_results.json').read_text())




def _format_metric(value, digits=3):
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def _ablation_key(sample):
    meta = sample.get('metadata', {})
    raw_id = meta.get('sample_id', sample.get('sample_id'))
    if isinstance(raw_id, int):
        return str(raw_id)
    if isinstance(raw_id, str) and raw_id.startswith('sample_'):
        suffix = raw_id.split('sample_', 1)[1]
        try:
            return str(int(suffix))
        except ValueError:
            return suffix
    try:
        return str(int(str(raw_id).split('_')[-1]))
    except Exception:
        return str(raw_id)


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


def _ablation_summary(sample_id):
    sample_key = str(sample_id)
    per_sample = ABLATED_DATA.get('per_sample', {})

    modes = ['zero_out', 'shuffle', 'noise', 'random', 'visual_zero']
    kl_means, top1s, accs = [], [], []
    present_modes = []
    for mode in modes:
        stats = per_sample.get(mode, {}).get(sample_key)
        if stats is None:
            continue
        present_modes.append(mode)
        kl_means.append(stats.get('kl_mean', 0))
        top1s.append(stats.get('top1_agreement', 0))
        accs.append(stats.get('gt_acc_ablated', 0))

    if not present_modes:
        return html.Div('No ablation results for this sample.', className='small text-muted p-2')

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=('KL Mean', 'Top1 Agreement', 'Acc Ablated'),
        vertical_spacing=0.06
    )
    fig.add_trace(go.Bar(x=present_modes, y=kl_means, showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=present_modes, y=top1s, showlegend=False), row=2, col=1)
    fig.add_trace(go.Bar(x=present_modes, y=accs, showlegend=False), row=3, col=1)
    fig.update_layout(
        margin=dict(l=5, r=5, t=20, b=5),
        font=dict(size=7),
        hovermode=False
    )
    fig.update_annotations(font_size=6)
    fig.update_xaxes(tickfont=dict(size=6), row=3, col=1)
    for r in (1, 2):
        fig.update_xaxes(visible=False, row=r, col=1)
    for r in (1, 2, 3):
        fig.update_yaxes(tickfont=dict(size=6), row=r, col=1)

    return dcc.Graph(figure=fig, config={'displayModeBar': False}, style={'height': '100%', 'width': '100%'})


def _maze_view(sample):
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


def _token_grid(sample, ablated_ranks=None):
    ablated_ranks = ablated_ranks or []
    tiles = []
    image_src = _load_maze_image(sample.get('metadata', {}).get('image_input'))
    for i, token in enumerate(sample['tokens'][:6]):
        is_ablated = i in ablated_ranks
        border_color = '#dc3545' if is_ablated else '#dee2e6'
        border_width = '3px' if is_ablated else '1px'
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
                html.Div('ABLATED', style={
                    'position': 'absolute', 'top': '2px', 'right': '2px',
                    'backgroundColor': '#dc3545', 'color': 'white',
                    'fontSize': '8px', 'padding': '1px 4px', 'borderRadius': '3px',
                    'zIndex': 3, 'display': 'block' if is_ablated else 'none'
                }) if is_ablated else None
            ], style={
                'position': 'relative',
                'width': '10vh', 'height': '10vh', 'flexShrink': 0,
                'border': f'{border_width} solid {border_color}',
                'overflow': 'hidden',
                'borderRadius': '4px',
                'backgroundColor': '#f8f9fa'
            })
        )

    return html.Div([
        html.Div("Spatial Focus (latent token heatmaps)", style={
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


def _compute_move_kl(clean_dist, ablated_dist):
    kls = []
    for d in ('UP', 'DOWN', 'LEFT', 'RIGHT'):
        c = clean_dist.get(d, 1e-9) or 1e-9
        a = ablated_dist.get(d, 1e-9) or 1e-9
        kls.append(c * math.log(c / a))
    return sum(kls)


def _plan_row(clean_plan, mean_clean_conf, mean_abl_conf):
    """Predicted-plan label with the direction glyphs boxed beside it, then an
    optional 'unchanged' badge and mean-confidence readout pushed to the right."""
    glyphs = [html.Span(_DIR_GLYPH.get(d, '?'), style={
        'fontSize': '0.95rem', 'color': '#06b6d4', 'fontWeight': 'bold',
    }) for d in clean_plan]

    row = [
        html.Span('Predicted plan', style={
            'fontSize': '0.55rem', 'color': '#888', 'textTransform': 'uppercase',
            'letterSpacing': '0.1em', 'marginRight': '8px',
        }),
        html.Div(glyphs or html.Span('—', style={'fontSize': '0.8rem', 'color': '#bbb'}), style={
            'display': 'flex', 'alignItems': 'center', 'gap': '5px',
            'padding': '2px 10px', 'marginRight': 'auto',
            'border': '1px solid #06b6d440', 'borderRadius': '5px',
            'backgroundColor': '#06b6d40d',
        }),
    ]

    if mean_abl_conf is not None:
        row.append(html.Span('unchanged', style={
            'fontSize': '0.5rem', 'color': '#06b6d4', 'fontFamily': 'monospace',
            'textTransform': 'uppercase', 'letterSpacing': '0.05em',
            'padding': '1px 6px', 'border': '1px solid #06b6d433',
            'borderRadius': '3px', 'backgroundColor': '#06b6d410', 'marginRight': '8px',
        }))

    conf = [
        html.Span('mean confidence ', style={'fontSize': '0.55rem', 'color': '#888', 'fontFamily': 'monospace'}),
        html.Span(f'{mean_clean_conf:.4f}', style={'fontSize': '0.55rem', 'color': '#06b6d4', 'fontFamily': 'monospace'}),
    ]
    if mean_abl_conf is not None:
        conf += [
            html.Span(' → ', style={'fontSize': '0.55rem', 'color': '#666'}),
            html.Span(f'{mean_abl_conf:.4f}', style={'fontSize': '0.55rem', 'color': '#eab308', 'fontFamily': 'monospace'}),
        ]
    row.append(html.Div(conf, style={'display': 'flex', 'alignItems': 'baseline', 'gap': '2px'}))

    return html.Div(row, style={'display': 'flex', 'alignItems': 'center', 'padding': '4px 8px',
                                'borderBottom': '1px solid #e9ecef'})


def _per_step_cards(moves, rank):
    """One horizontal strip of narrow cards, each showing confidence + KL shift.

    Two compact bars per step:
    1. Clean confidence: width = winner probability (always near 100 %)
    2. KL shift:  width = log10-scaled move-KL, visible even for tiny shifts
    """
    if not moves:
        return html.Div('No per-move data for this token', className='small text-muted',
                        style={'padding': '8px'})

    max_kl = max((_compute_move_kl(m['clean'], m['ablated']) for m in moves), default=1e-9) or 1e-9
    eps = 1e-9

    def _kl_pct(kl_val):
        """log10 scaling: 0 → 0 %, max_kl → 100 %, tiny values stay visible."""
        if kl_val <= 0:
            return 0.0
        lo = math.log10(eps)
        hi = math.log10(max_kl + eps)
        span = hi - lo or 1.0
        return (math.log10(kl_val + eps) - lo) / span * 100

    cards = []
    for i, m in enumerate(moves):
        winner = max(m['clean'], key=m['clean'].get)
        conf = m['clean'][winner]
        kl = _compute_move_kl(m['clean'], m['ablated'])
        kl_pct = _kl_pct(kl)

        cards.append(html.Div([
            # Header: step number + winner direction glyph
            html.Div([
                html.Span(f'step {i}', style={
                    'fontSize': '0.5rem', 'color': '#888',
                    'fontFamily': 'monospace',
                }),
                html.Span(_DIR_GLYPH.get(winner, '?'), style={
                    'fontSize': '0.65rem', 'color': '#06b6d4',
                    'fontWeight': 'bold', 'marginLeft': '4px',
                }),
            ], style={'display': 'flex', 'alignItems': 'baseline', 'marginBottom': '3px'}),

            # Confidence row
            html.Div([
                html.Span('conf', style={
                    'fontSize': '0.45rem', 'color': '#aaa', 'fontFamily': 'monospace',
                    'minWidth': '2.2em',
                }),
                html.Div(style={
                    'flex': '1', 'position': 'relative', 'height': '5px',
                    'margin': '0 4px',
                    'backgroundColor': '#f0f0f0', 'borderRadius': '2px',
                    'overflow': 'hidden',
                }, children=[
                    html.Div(style={
                        'position': 'absolute', 'inset': 0,
                        'width': f'{conf * 100:.1f}%', 'height': '100%',
                        'backgroundColor': '#06b6d470', 'borderRadius': '2px',
                    }),
                ]),
                html.Span(f'{conf:.4f}', style={
                    'fontSize': '0.45rem', 'color': '#06b6d4',
                    'fontFamily': 'monospace', 'minWidth': '3.2em',
                    'textAlign': 'right',
                }),
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '2px'}),

            # KL shift row (log-scale bar)
            html.Div([
                html.Span('shift', style={
                    'fontSize': '0.45rem', 'color': '#aaa', 'fontFamily': 'monospace',
                    'minWidth': '2.2em',
                }),
                html.Div(style={
                    'flex': '1', 'position': 'relative', 'height': '5px',
                    'margin': '0 4px',
                    'backgroundColor': '#f0f0f0', 'borderRadius': '2px',
                    'overflow': 'hidden',
                }, children=[
                    html.Div(style={
                        'position': 'absolute', 'inset': 0,
                        'width': f'{min(kl_pct, 100):.1f}%', 'height': '100%',
                        'backgroundColor': '#eab308', 'borderRadius': '2px',
                    }),
                ]),
                html.Span(f'{kl:.4f} KL', style={
                    'fontSize': '0.45rem', 'color': '#eab308',
                    'fontFamily': 'monospace', 'minWidth': '4.2em',
                    'textAlign': 'right',
                }),
            ], style={'display': 'flex', 'alignItems': 'center'}),
        ], style={
            'width': '160px', 'flexShrink': 0,
            'backgroundColor': '#fafafa', 'border': '1px solid #e9ecef',
            'borderRadius': '4px', 'padding': '3px 4px',
        }))

    return html.Div(children=cards, style={
        'display': 'flex', 'gap': '4px',
        'overflowX': 'auto', 'padding': '4px 8px',
    })


def _render_unablated_path(sample_id):
    """Clean path view — identical layout to `_plan_row`, minus ablated side."""
    moves = load_moves(sample_id, 0)
    if not moves:
        return html.Div('N/A', className='small text-muted p-2')

    clean_plan = [max(m['clean'], key=m['clean'].get) for m in moves]
    clean_conf = [m['clean'][d] for d, m in zip(clean_plan, moves)]
    mean_cc = sum(clean_conf) / len(clean_conf)
    return _plan_row(clean_plan, mean_cc, None)


def _token_contribution_strip(sample_id, ablated_ranks):
    """Six per-latent-token cells showing marginal KL contribution.

    Individually every token is ~0; the marginal contribution (effect of adding
    the token to a subset) is where the structure lives. Selected tokens are
    highlighted in the ablation colour.

    Falls back to single-token KL values from results.json when the full
    combinatorial ablated_plans data is unavailable.
    """
    contribs = token_marginal_contributions(sample_id)
    if not contribs:
        # Fallback: build a simple strip from per-token zero_token_* results.
        contribs = []
        for rank in range(6):
            entry, err = load_per_token_summary(sample_id, rank)
            if entry is None:
                continue
            kl = entry.get('kl_mean', 0.0)
            contribs.append({
                'rank': rank,
                'label': f'T{rank}',
                'individual': kl,
                'marginal': kl,
            })
    if not contribs:
        return html.Div('No per-token data', className='small text-muted', style={'padding': '4px 8px'})

    max_marg = max((c['marginal'] for c in contribs), default=0.0) or 1e-9
    cells = []
    for c in contribs:
        selected = c['rank'] in ablated_ranks
        frac = max(0.0, c['marginal'] / max_marg)
        cells.append(html.Div([
            html.Div([
                html.Span(f"T{c['rank']}", style={
                    'fontSize': '0.6rem', 'fontWeight': 'bold', 'fontFamily': 'monospace',
                    'color': '#b45309' if selected else '#475569',
                    'marginRight': '4px',
                }),
                html.Span(c['label'], style={
                    'fontSize': '0.4rem', 'color': '#94a3b8',
                    'whiteSpace': 'nowrap', 'overflow': 'hidden', 'textOverflow': 'ellipsis',
                }),
            ], style={'display': 'flex', 'alignItems': 'baseline'}),
            html.Div(style={
                'position': 'relative', 'height': '5px', 'marginTop': '2px',
                'backgroundColor': '#f0f0f0', 'borderRadius': '2px', 'overflow': 'hidden',
            }, children=[html.Div(style={
                'position': 'absolute', 'inset': 0, 'width': f'{frac * 100:.0f}%',
                'backgroundColor': '#eab308' if selected else '#06b6d4', 'borderRadius': '2px',
            })]),
            html.Div(f"{c['marginal']:.3f}", style={
                'fontSize': '0.45rem', 'color': '#64748b', 'fontFamily': 'monospace', 'marginTop': '1px',
            }),
        ], id={'type': 'token-contrib', 'rank': c['rank']}, n_clicks=0, style={
            'flexShrink': 0, 'padding': '2px 3px', 'cursor': 'pointer',
            'border': f"1px solid {'#eab308' if selected else '#e9ecef'}",
            'borderRadius': '4px',
            'backgroundColor': '#fffbeb' if selected else '#fafafa',
        }))

    return html.Div([
        html.Div(cells, style={'display': 'flex', 'flexDirection': 'column', 'gap': '2px', 'flex': 1}),
        html.Div('Marginal KL', style={
            'fontSize': '0.45rem', 'color': '#888', 'writingMode': 'vertical-rl',
            'textTransform': 'uppercase', 'letterSpacing': '0.05em',
            'textAlign': 'center', 'flexShrink': 0, 'marginLeft': '2px',
        }),
    ], style={'display': 'flex', 'flexDirection': 'row', 'padding': '2px 4px'})


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
                             hovertemplate='zero %{x} tokens<br>median KL %{y:.4f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=ks, y=[r['em_flip_pct'] for r in rows], mode='lines', yaxis='y2',
                             line=dict(color='#dc3545', width=1.5, dash='dash'),
                             hovertemplate='text flip %{y:.0f}%<extra></extra>'))
    fig.add_trace(go.Scatter(x=ks, y=[r['plan_flip_pct'] for r in rows], mode='lines', yaxis='y2',
                             line=dict(color='#94a3b8', width=1.5, dash='dot'),
                             hovertemplate='plan flip %{y:.0f}%<extra></extra>'))

    cur = current_combo_metrics(sample_id, ablated_ranks)
    if cur:
        fig.add_trace(go.Scatter(x=[cur['k']], y=[cur['kl']], mode='markers',
                                 marker=dict(size=11, color='#eab308',
                                             line=dict(color='#92400e', width=1.5)),
                                 hovertemplate='current: zero %{x} tokens<br>KL %{y:.4f}<extra></extra>'))

    fig.update_layout(
        margin=dict(l=30, r=28, t=6, b=18), font=dict(size=8),
        hovermode='x unified', showlegend=False, paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title=dict(text='latent tokens zeroed', font=dict(size=8)),
                   dtick=1, tickfont=dict(size=7)),
        yaxis=dict(title=dict(text='KL', font=dict(size=8)), tickfont=dict(size=7), rangemode='tozero'),
        yaxis2=dict(overlaying='y', side='right', range=[0, 100], tickfont=dict(size=7),
                    tickvals=[0, 50, 100], ticktext=['0', '50', '100%']),
    )
    return html.Div([
        html.Div(_dose_legend(), style={'flexShrink': 0}),
        html.Div(dcc.Graph(figure=fig, config={'displayModeBar': False},
                           style={'height': '100%', 'width': '100%'}),
                 style={'flex': 1, 'minHeight': 0, 'overflow': 'hidden'}),
    ], style={'height': '24vh', 'display': 'flex',
              'flexDirection': 'column', 'padding': '0 4px'})


def _dose_legend():
    """Compact HTML legend for the dose-response graph."""
    def item(swatch, label):
        return html.Span([swatch, html.Span(label, style={'marginLeft': '4px'})], style={
            'display': 'flex', 'alignItems': 'center', 'fontSize': '0.5rem',
            'color': '#64748b', 'fontFamily': 'monospace',
        })

    def line(color, dash=False):
        return html.Span(style={
            'display': 'inline-block', 'width': '14px',
            'borderTop': f"2px {'dashed' if dash else 'solid'} {color}",
        })

    band = html.Span(style={'display': 'inline-block', 'width': '14px', 'height': '8px',
                            'backgroundColor': 'rgba(6,182,212,0.25)'})
    dot = html.Span(style={'display': 'inline-block', 'width': '8px', 'height': '8px',
                           'borderRadius': '50%', 'backgroundColor': '#eab308',
                           'border': '1px solid #92400e'})
    return html.Div([
        item(line('#06b6d4'), 'median KL'),
        item(band, 'min–max'),
        item(line('#dc3545', dash=True), 'text flip %'),
        item(line('#94a3b8', dash=True), 'plan flip %'),
        item(dot, 'current'),
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '10px', 'padding': '2px 2px 3px'})


def _flip_readout(cur):
    """One-line current-combo summary: KL + both flip definitions."""
    if not cur:
        return None

    def _badge(text, flipped):
        on = bool(flipped)
        return html.Span(text, style={
            'fontSize': '0.62rem', 'fontFamily': 'monospace', 'padding': '1px 7px',
            'borderRadius': '3px', 'whiteSpace': 'nowrap',
            'color': '#b91c1c' if on else '#15803d',
            'backgroundColor': '#fee2e2' if on else '#dcfce7',
        })

    return html.Div([
        html.Span(f"zero {cur['k']} tokens", style={
            'fontSize': '0.66rem', 'color': '#475569', 'fontFamily': 'monospace',
            'fontWeight': 'bold', 'whiteSpace': 'nowrap',
        }),
        html.Span(f"KL {cur['kl']:.6f}", style={
            'fontSize': '0.66rem', 'color': '#a16207', 'fontFamily': 'monospace',
            'fontWeight': 'bold', 'whiteSpace': 'nowrap',
        }),
        _badge('text ' + ('flipped' if cur['em_flipped'] else 'stable'), cur['em_flipped']),
        _badge('plan ' + ('flipped' if cur['plan_flipped'] else 'stable'), cur['plan_flipped']),
    ], style={
        'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap', 'gap': '12px',
        'padding': '2px 8px',
    })


def _mean_top_conf(dists):
    """Mean of the top-probability per move distribution, or None if empty."""
    confs = [d[max(d, key=d.get)] for d in dists if d]
    return sum(confs) / len(confs) if confs else None


def _ablated_plan_row(sample_id, ablated_ranks):
    """Return (clean_plan, mean_clean_conf, mean_abl_conf) for the plan-row header.

    Confidences come from the per-move distributions (single-token via load_moves,
    combos via load_combo_dist); they default to 1.0 when unavailable, since the
    stored distributions are argmax-saturated.
    """
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
    """Pinned plan-row + aggregate footer for the selected ablation state."""
    header = []
    footer = []

    if not ablated_ranks:
        header.append(_render_unablated_path(sample_id))
    else:
        header.append(_plan_row(*_ablated_plan_row(sample_id, ablated_ranks)))

    if show_strip:
        footer.append(_dose_response_graph(sample_id, ablated_ranks))

    return html.Div([
        html.Div(header, style={'flexShrink': 0}),
        html.Div(footer, style={'overflowY': 'auto', 'maxHeight': '100%'}) if footer else None,
    ], style={'display': 'flex', 'flexDirection': 'column', 'height': '100%'})


def _render_ablation_tab(sample_id, ablated_ranks):
    """Ablation tab: KL bar chart for the selected sample."""
    return _ablation_summary(sample_id)


def update_level2_logic(clickData):
    if not clickData:
        return (
            html.Div(className='p-2'),
        )

    sample_id = clickData['points'][0]['hovertext']
    sample = next(s for s in MOCK_DATA if s['sample_id'] == sample_id)

    return _maze_view(sample)


def register_level2_callbacks(app):
    @app.callback(
        [Output('level2-maze-pane', 'children'),
         Output('ablation-state', 'data')],
        [Input('level1-scatter', 'clickData')]
    )
    def update_level2(clickData):
        a = update_level2_logic(clickData)
        return a, {'ablated_ranks': []}

    @app.callback(
        Output('level2-kl-pane', 'children'),
        [Input('level1-scatter', 'clickData'),
         Input('level2-tab-selector', 'value'),
         Input('ablation-state', 'data')]
    )
    def update_kl_pane(clickData, active_tab, ablation_state):
        if not clickData:
            return html.Div(className='p-2')
        if active_tab != 'ablation':
            return html.Div([
                html.Div("RQ2: Probe accuracy visualizations — data pending",
                         style={'fontSize': '0.6rem', 'color': '#888', 'padding': '8px',
                                'textTransform': 'uppercase', 'letterSpacing': '0.05em'}),
                html.Div(id='rq2-static-bar-placeholder'),
                html.Div(id='rq2-dynamic-grid-placeholder'),
            ], style={'height': '100%'})
        sample_id = clickData['points'][0]['hovertext']
        sample = next(s for s in MOCK_DATA if s['sample_id'] == sample_id)
        sid = _ablation_key(sample)
        ablated = (ablation_state or {}).get('ablated_ranks', [])
        return html.Div([
            html.Div(_token_contribution_strip(sid, ablated), style={'flexShrink': 0, 'marginRight': '6px'}),
            html.Div(_ablation_summary(sid), style={'flex': 1, 'minWidth': 0, 'overflow': 'hidden'}),
        ], style={'display': 'flex', 'flexDirection': 'row', 'height': '100%'})

    @app.callback(
        Output('token-flip-readout', 'children'),
        [Input('level1-scatter', 'clickData'),
         Input('ablation-state', 'data')]
    )
    def update_flip_readout(clickData, ablation_state):
        if not clickData:
            return None
        sample_id = clickData['points'][0]['hovertext']
        ablated = (ablation_state or {}).get('ablated_ranks', [])
        return _flip_readout(current_combo_metrics(sample_id, ablated))

    @app.callback(
        Output('level2-token-grid', 'children'),
        [Input('level1-scatter', 'clickData'),
         Input('ablation-state', 'data')]
    )
    def update_token_grid(clickData, ablation_state):
        if not clickData:
            return html.Div(className='p-2')
        sample_id = clickData['points'][0]['hovertext']
        sample = next(s for s in MOCK_DATA if s['sample_id'] == sample_id)
        ablated = (ablation_state or {}).get('ablated_ranks', [])
        return _token_grid(sample, ablated)

    @app.callback(
        Output('level2-output-pane', 'children'),
        [Input('level1-scatter', 'clickData'),
         Input('ablation-state', 'data'),
         Input('level2-tab-selector', 'value')]
    )
    def update_output_pane(clickData, ablation_state, active_tab):
        if not clickData:
            return html.Div("Select a Sample (Level 1)", className="p-2 text-muted small")
        sample_id = clickData['points'][0]['hovertext']
        ablated = (ablation_state or {}).get('ablated_ranks', [])
        return _render_output_panel(sample_id, ablated, show_strip=(active_tab == 'ablation'))

    @app.callback(
        Output('ablation-state', 'data', allow_duplicate=True),
        Input({'type': 'token-contrib', 'rank': ALL}, 'n_clicks'),
        [State('level1-scatter', 'clickData'),
         State('ablation-state', 'data')],
        prevent_initial_call=True,
    )
    def toggle_from_strip(_n_clicks, clickData, ablation_state):
        ctx = callback_context
        if not ctx.triggered or not clickData:
            return no_update
        trig = ctx.triggered[0]
        if not trig['value']:  # recreation / no real click
            return no_update
        try:
            rank = json.loads(trig['prop_id'].split('.n_clicks')[0])['rank']
        except Exception:
            return no_update
        sample_id = clickData['points'][0]['hovertext']
        ranks = (ablation_state or {}).get('ablated_ranks', [])
        new_ranks, _ = toggle_rank(sample_id, ranks, rank)
        return {'ablated_ranks': new_ranks}
