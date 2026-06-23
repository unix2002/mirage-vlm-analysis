import dash_bootstrap_components as dbc
from dash import html, dcc
from .components.level1_landscape import create_level1_landscape
from .components.level2_path import create_level2_top, create_level2_bottom, create_level2_probing_row
from .components.level3_detail import create_level3_detail
from .data_loader import LOADER


def create_header():
    return dbc.Row([
        dbc.Col(html.H4("Latent Reasoning VLM Analysis",
                className="text-primary m-0", style={'paddingLeft': '10px'}), width=12),
    ], className="py-2 border-bottom bg-light", style={'height': '5vh'})


_HEADER_CLASS = "py-1 small font-weight-bold"


def create_sidebar():
    return html.Div([
        dbc.Card([
            dbc.CardHeader(
                html.Div([
                    html.Span("Step 1: Sample Selection & UMAP Tuner", className="align-self-center"),
                    dbc.Button("HELP", id="help-btn-step1", size="sm", color="info", outline=True, className="ms-auto py-0 px-2", style={'fontSize': '0.75rem'})
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}),
                className=_HEADER_CLASS
            ),
            dbc.CardBody([
                dcc.Graph(
                    id='level1-scatter', 
                    figure=create_level1_landscape(LOADER.get_data()),
                    style={'height': '55vh', 'width': '100%'},
                    config={'responsive': True}
                ),
                html.Hr(className="my-2"),
                html.Div([
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
                        id="umap-flippers-toggle",
                        label="Highlight Plan Flippers",
                        value=False,
                        className="small text-muted mb-2"
                    ),

                    html.Div([
                        html.Label("Color Metric", className="small text-muted mb-0"),
                        dcc.Dropdown(
                            id='umap-color-dropdown',
                            options=[
                                {'label': 'Reasoning Intensity (KL)', 'value': 'avg_kl'},
                                {'label': 'Level ID', 'value': 'level_id'},
                                {'label': 'Solution Length / Walking length', 'value': 'solution_length'},
                                {'label': 'Projection Error', 'value': 'umap_uncertainty'}
                            ],
                            value='level_id',
                            clearable=False,
                            className="dash-bootstrap"
                        )
                    ])
                ], className="px-3 py-0")
            ], className="p-0", style={'display': 'flex', 'flexDirection': 'column', 'height': '100%'})
        ], style={'height': '100%'})
    ], style={'height': '100%'})


def create_main_content():
    return html.Div([
        dbc.Card([
            dbc.CardHeader(
                html.Div([
                    html.Span("Step 2: Reasoning Path Analysis", id='level2-header-title', className="align-self-center small font-weight-bold"),
                    dbc.RadioItems(
                        id='level2-tab-selector',
                        options=[
                            {'label': 'Probing', 'value': 'probing'},
                            {'label': 'Ablation', 'value': 'ablation'},
                        ],
                        value='probing',
                        inline=True,
                        className='ms-auto',
                        inputClassName='btn-check',
                        labelClassName='btn btn-outline-secondary btn-sm px-2 py-0',
                    ),
                    dbc.Button("HELP", id="help-btn-step2", size="sm", color="info", outline=True, className="ms-2 py-0 px-2", style={'fontSize': '0.75rem'})
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}),
                className="py-1"
            ),
            dbc.CardBody(html.Div([
                create_level2_top(),
                create_level2_probing_row(),
                create_level2_bottom(),
            ], style={'display': 'flex', 'flexDirection': 'column', 'height': '100%'}), className="p-1")
        ], style={'height': '55vh', 'flexShrink': 0}, className="mb-2"),

        # Level 3: Token Specifics
        dbc.Card([
            dbc.CardHeader(
                html.Div([
                    html.Span("Step 3: Token Details", id='level3-instructions', className="align-self-center"),
                    dbc.Button("HELP", id="help-btn-step3", size="sm", color="info", outline=True, className="ms-auto py-0 px-2", style={'fontSize': '0.75rem'})
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}),
                className=_HEADER_CLASS
            ),
            dbc.CardBody(create_level3_detail(), className="p-1", style={'display': 'flex', 'flexDirection': 'column', 'flex': 1, 'minHeight': 0})
        ], style={'flex': 1, 'minHeight': 0, 'display': 'flex', 'flexDirection': 'column'})
    ], style={'height': '100%', 'display': 'flex', 'flexDirection': 'column'})


def create_help_modals():
    return html.Div([
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Help: Sample Selection & UMAP Tuner")),
            dbc.ModalBody("Use the scatter plot to select samples for detailed analysis. The UMAP projection clusters samples based on their latent reasoning representations. You can tune the UMAP parameters and use the color metric dropdown to highlight different aspects of the data."),
        ], id="help-modal-step1", is_open=False, size="lg", centered=True),
        
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Help: Reasoning Path Analysis")),
            dbc.ModalBody("This section lets you inspect the internal reasoning path. Use 'Probing' to see which maze cells the model focuses on at each step. Use 'Ablation' to see what happens when specific reasoning paths are altered. Click on a sample in Step 1 to load it here."),
        ], id="help-modal-step2", is_open=False, size="lg", centered=True),

        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Help: Token Details")),
            dbc.ModalBody("Here you can inspect the specifics of an individual reasoning token. This includes attention maps over the visual input, and other token-level metrics that help explain why the model made a specific move. Click on a token in Step 2 to view details."),
        ], id="help-modal-step3", is_open=False, size="lg", centered=True),
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
        ], style={'height': '100vh', 'overflow': 'hidden'}),
        create_help_modals()
    ], fluid=True, className="p-0")
