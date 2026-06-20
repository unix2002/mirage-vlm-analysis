import base64
import json
import tarfile
from pathlib import Path
from functools import lru_cache

from dash.dependencies import Input, Output
import dash
from dash import dcc, html
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from ..mock_data import MOCK_DATA
from .ablation_v2 import load_moves


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
            'width': '100%',
            'maxHeight': '100%',
            'objectFit': 'contain',
            'border': '1px solid #dee2e6',
            'backgroundColor': '#f8f9fa'
        })
    ], style={'width': '100%', 'height': '100%', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})


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
            dbc.Col(
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
                    'height': '14vh',
                    'border': f'{border_width} solid {border_color}',
                    'overflow': 'hidden',
                    'borderRadius': '4px',
                    'backgroundColor': '#f8f9fa'
                }),
                width=4,
                className='p-1'
            )
        )

    return dbc.Row(tiles[:3], className='g-0') if len(tiles) <= 3 else html.Div([
        dbc.Row(tiles[:3], className='g-0'),
        dbc.Row(tiles[3:6], className='g-0')
    ])


_DIR_GLYPH = {'UP': '↑', 'DOWN': '↓', 'LEFT': '←', 'RIGHT': '→'}


def _arrow_span(direction, prob, color):
    glyph = _DIR_GLYPH.get(direction, '?')
    if not direction:
        return html.Div()
    size = 0.6 + prob * 0.8
    return html.Div([
        html.Div(glyph, style={
            'fontSize': f'{size}rem', 'color': color,
            'fontWeight': 'bold', 'lineHeight': 1, 'textAlign': 'center'
        }),
        html.Div(f'{prob:.3f}', style={
            'fontSize': '0.55rem', 'color': '#888', 'textAlign': 'center'
        })
    ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'minWidth': '1.5rem'})


def _arrow_path_row(directions_probs, color):
    return html.Div([
        _arrow_span(d, p, color) for d, p in directions_probs
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '2px', 'alignItems': 'center'})


def _outputs_panel(sample):
    meta = sample.get('metadata', {})
    sample_id = meta.get('sample_id', sample.get('sample_id'))

    # Load clean moves from T0 (same for all tokens)
    moves = load_moves(sample_id, 0)
    if moves:
        clean_dirs = [(max(m['clean'], key=m['clean'].get), m['clean'][max(m['clean'], key=m['clean'].get)]) for m in moves]
        gt_dirs = [(m['gt'], 1.0) for m in moves]
        model_body = _arrow_path_row(clean_dirs, '#3b82f6')
        true_body = _arrow_path_row(gt_dirs, '#22c55e')
    else:
        model_body = html.Pre('N/A', className='small mb-0')
        true_body = html.Pre('N/A', className='small mb-0')

    return dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader('Model Output', className='py-1 small'),
            dbc.CardBody(model_body, className='py-1 px-2'),
        ], className='h-100'), width=6),
        dbc.Col(dbc.Card([
            dbc.CardHeader('True Output', className='py-1 small'),
            dbc.CardBody(true_body, className='py-1 px-2'),
        ], className='h-100'), width=6),
    ], className='g-0', style={'height': '100%'})


def _render_arrow_rows(sample_id, ablated_ranks):
    if len(ablated_ranks) != 1:
        return None

    moves = load_moves(sample_id, ablated_ranks[0])
    if not moves:
        return None

    clean_dirs = [(max(m['clean'], key=m['clean'].get), m['clean'][max(m['clean'], key=m['clean'].get)]) for m in moves]
    abl_dirs = [(max(m['ablated'], key=m['ablated'].get), m['ablated'][max(m['ablated'], key=m['ablated'].get)]) for m in moves]
    gt_dirs = [(m['gt'], 1.0) for m in moves]

    flips = [c[0] != a[0] for c, a in zip(clean_dirs, abl_dirs)]
    has_flip = any(flips)

    heading = f'Model Output (ablated T{ablated_ranks[0]})'
    if has_flip:
        heading += ' ★ FLIP'

    label_clean = html.Div('clean', style={'fontSize': '0.55rem', 'color': '#3b82f6', 'marginRight': '4px'})
    label_abl = html.Div('abl', style={'fontSize': '0.55rem', 'color': '#ef4444', 'marginRight': '4px'})

    model_body = html.Div([
        html.Div([label_clean, _arrow_path_row(clean_dirs, '#3b82f6')],
                 style={'display': 'flex', 'alignItems': 'center'}),
        html.Div([label_abl, _arrow_path_row(abl_dirs, '#ef4444')],
                 style={'display': 'flex', 'alignItems': 'center', 'marginTop': '2px'}),
    ])
    true_body = _arrow_path_row(gt_dirs, '#22c55e')

    return dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader(heading, className='py-1 small'),
            dbc.CardBody(model_body, className='py-1 px-2'),
        ], className='h-100'), width=6),
        dbc.Col(dbc.Card([
            dbc.CardHeader('True Output (ground truth)', className='py-1 small'),
            dbc.CardBody(true_body, className='py-1 px-2'),
        ], className='h-100'), width=6),
    ], className='g-0', style={'height': '100%'})


def update_level2_logic(clickData):
    if not clickData:
        return (
            html.Div("Select a Sample (Level 1)", className="p-2 text-muted small"),
            html.Div(className='p-2'),
        )

    sample_id = clickData['points'][0]['hovertext']
    sample = next(s for s in MOCK_DATA if s['sample_id'] == sample_id)

    return _ablation_summary(_ablation_key(sample)), _maze_view(sample)


def register_level2_callbacks(app):
    @app.callback(
        [Output('level2-ablation-pane', 'children'),
         Output('level2-maze-pane', 'children'),
         Output('ablation-state', 'data')],
        [Input('level1-scatter', 'clickData')]
    )
    def update_level2(clickData):
        a, b = update_level2_logic(clickData)
        return a, b, {'ablated_ranks': []}

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
         Input('ablation-state', 'data')]
    )
    def update_output_pane(clickData, ablation_state):
        if not clickData:
            return html.Div("Select a Sample (Level 1)", className="p-2 text-muted small")
        sample_id = clickData['points'][0]['hovertext']
        sample = next(s for s in MOCK_DATA if s['sample_id'] == sample_id)
        ablated = (ablation_state or {}).get('ablated_ranks', [])

        if ablated:
            result = _render_arrow_rows(sample_id, ablated)
            if result:
                return result

        return _outputs_panel(sample)
