import base64
import json
import re
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


ABLATED_DATA = json.loads(Path('real_data/ablation_results.json').read_text())


def _load_truth_index():
    truth_path = Path('real_data/train_direct.jsonl')
    if not truth_path.exists():
        return {}

    index = {}
    for line in truth_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        image_input = row.get('image_input')
        if image_input:
            index[image_input] = row
    return index


GROUND_TRUTH_INDEX = _load_truth_index()


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
        archive_path = Path('data/vsp_spatial_planning/vsp_spatial_planning.tar.gz')
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


def _token_grid(sample):
    tiles = []
    image_src = _load_maze_image(sample.get('metadata', {}).get('image_input'))
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
                    )
                ], style={
                    'position': 'relative',
                    'height': '14vh',
                    'border': '1px solid #dee2e6',
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


def _outputs_panel(sample):
    meta = sample.get('metadata', {})
    model_output = meta.get('text_output_short') or meta.get('text_output') or 'N/A'
    truth_row = GROUND_TRUTH_INDEX.get(meta.get('image_input'))
    true_output = (truth_row or {}).get('text_output') or meta.get('true_output') or meta.get('ground_truth') or 'N/A'

    def _clean_output(text):
        if not text:
            return 'N/A'
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = text.replace('<output_image>', '')
        return text.replace('\\boxed{', '').replace('}', '').strip()

    def _card(title, body):
        return dbc.Card([
            dbc.CardHeader(title, className='py-1 small'),
            dbc.CardBody(html.Pre(_clean_output(body), className='small mb-0', style={'whiteSpace': 'pre-wrap', 'maxHeight': '4vh', 'overflowY': 'auto'}), className='py-1')
        ], className='h-100')

    return dbc.Row([
        dbc.Col(_card('Model Output', model_output), width=6),
        dbc.Col(_card('True Output', true_output), width=6),
    ], className='g-0', style={'height': '100%'})


def update_level2_logic(clickData):
    if not clickData:
        return (
            html.Div("Select a Sample (Level 1)", className="p-2 text-muted small"),
            html.Div(className='p-2'),
            html.Div(className='p-2'),
            html.Div(className='p-2')
        )

    sample_id = clickData['points'][0]['hovertext']
    sample = next(s for s in MOCK_DATA if s['sample_id'] == sample_id)

    return _ablation_summary(_ablation_key(sample)), _maze_view(sample), _token_grid(sample), _outputs_panel(sample)


def register_level2_callbacks(app):
    @app.callback(
        [Output('level2-ablation-pane', 'children'),
         Output('level2-maze-pane', 'children'),
         Output('level2-token-grid', 'children'),
         Output('level2-output-pane', 'children')],
        [Input('level1-scatter', 'clickData')]
    )
    def update_level2(clickData):
        return update_level2_logic(clickData)
