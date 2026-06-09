import dash_bootstrap_components as dbc
from dash import html, dcc
from .components.level1_landscape import create_level1_landscape
from .components.level2_path import create_level2_path
from .components.level3_detail import create_level3_detail
from .mock_data import MOCK_DATA


def create_header():
    return dbc.Row([
        dbc.Col(html.H4("Latent Reasoning VLM Analysis",
                className="text-primary m-0"), width=8),
        dbc.Col(html.Div("System Active",
                className="text-right text-muted small"), width=4),
    ], className="py-2 border-bottom bg-light", style={'height': '5vh'})


def create_sidebar():
    return html.Div([
        # Level 1 Landscape Graph
        dbc.Card([
            dbc.CardHeader("Level 1: Sample Landscape", className="py-1 small font-weight-bold"),
            dbc.CardBody([
                dcc.Graph(
                    id='level1-scatter', 
                    figure=create_level1_landscape(MOCK_DATA), 
                    style={'height': '60vh', 'width': '100%'},
                    config={'responsive': True}
                )
            ], className="p-0")
        ], className="mb-2"),
        
        # UMAP Tuner Controls
        dbc.Card([
            dbc.CardHeader("UMAP Parameter Tuner", className="py-1 small font-weight-bold"),
            dbc.CardBody([
                html.Div([
                    html.Label("Nearest Neighbors (n_neighbors)", className="small text-muted mb-0"),
                    dcc.Slider(
                        id='umap-neighbors-slider',
                        min=2, max=30, step=1, value=5,
                        marks={2: '2', 15: '15', 30: '30'},
                        className="p-0"
                    ),
                ], className="mb-3"),
                
                html.Div([
                    html.Label("Minimum Distance (min_dist)", className="small text-muted mb-0"),
                    dcc.Slider(
                        id='umap-dist-slider',
                        min=0.0, max=1.0, step=0.05, value=0.3,
                        marks={0: '0', 0.5: '0.5', 1: '1'},
                        className="p-0"
                    ),
                ], className="mb-2"),

                dbc.Checkbox(
                    id="umap-pca-toggle",
                    label="PCA Denoising (Pre-process)",
                    value=False,
                    className="small text-muted mb-2"
                ),

                html.Div([
                    html.Label("Color Metric", className="small text-muted mb-0"),
                    dcc.Dropdown(
                        id='umap-color-dropdown',
                        options=[
                            {'label': 'Reasoning Intensity (KL)', 'value': 'avg_kl'},
                            {'label': 'Correctness', 'value': 'correctness'},
                            {'label': 'Level ID', 'value': 'level_id'},
                            {'label': 'Sequence Length', 'value': 'seq_len'},
                            {'label': 'Num Latent Tokens', 'value': 'num_latent'}
                        ],
                        value='avg_kl',
                        clearable=False,
                        className="dash-bootstrap"
                    )
                ])
            ], className="px-3 py-2")
        ], style={'height': '32vh'})
    ])


def create_main_content():
    return html.Div([
        # Level 2: Main Reasoning Path
        dbc.Card([
            dbc.CardHeader("Level 2: Reasoning Path Analysis",
                           className="py-1 small font-weight-bold"),
            dbc.CardBody(create_level2_path(), className="p-1")
        ], style={'height': '43vh'}, className="mb-2"),

        # Level 3: Token Specifics
        dbc.Card([
            dbc.CardHeader(html.Div(id='level3-instructions', children="Level 3: Token Details"),
                           className="py-1 small font-weight-bold"),
            dbc.CardBody(create_level3_detail(), className="p-1")
        ], style={'height': '43vh'})
    ])


def create_layout():
    return dbc.Container([
        # Main Dashboard Wrapper - 100vh, no scroll
        html.Div([
            create_header(),

            # Main Body (95vh)
            dbc.Row([
                # Left Column: Selection (25% width)
                dbc.Col(create_sidebar(), width=3, className="pr-1 py-2"),

                # Right Column: Analysis (75% width)
                dbc.Col(create_main_content(), width=9, className="pl-1 py-2")
            ], style={'height': '95vh'})
        ], style={'height': '100vh', 'overflow': 'hidden'})
    ], fluid=True, className="p-0")
