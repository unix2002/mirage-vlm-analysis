import dash_bootstrap_components as dbc
from dash import html, dcc
from .components.level1_landscape import create_level1_landscape
from .components.level2_path import create_level2_top, create_level2_bottom, create_level2_probing_row
from .components.level3_detail import create_level3_detail
from .data_loader import LOADER
from .help_content import help_page, HELP_OVERLAY_STYLE


def create_header():
    return dbc.Row([
        dbc.Col(
            html.Div([
                html.H4("Visual Analytics System for Latent Reasoning",
                        className="m-0", style={'color': '#1f2937', 'fontSize': '18px'}),
                dbc.Button("?", id="help-btn-global", color="info", outline=True, size="sm",
                           title="About this dashboard",
                           className="ms-auto",
                           style={'borderRadius': '50%', 'width': '30px', 'height': '30px',
                                  'padding': 0, 'fontWeight': 'bold', 'fontSize': '16px', 'lineHeight': '1'}),
            ], style={'display': 'flex', 'alignItems': 'center',
                      'paddingLeft': '10px', 'paddingRight': '14px'}),
            width=12),
    ], className="py-2 border-bottom bg-light", style={'height': '5vh'})


_HEADER_CLASS = "py-1 font-weight-bold"


def create_sidebar():
    return html.Div([
        dbc.Card([
            dbc.CardHeader(
                html.Div([
                    html.Span("Step 1: Sample Selection & UMAP Tuner", className="align-self-center", style={'color': '#1f2937', 'fontSize': '14px'}),
                    dbc.Button("HELP", id="help-btn-step1", size="sm", color="info", outline=True, className="ms-auto py-0 px-2", style={'fontSize': '14px'})
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
                        html.Label("Nearest Neighbors (n_neighbors)", className="mb-0", style={'color': '#1f2937', 'fontSize': '14px'}),
                        dcc.Slider(
                            id='umap-neighbors-slider',
                            min=2, max=30, step=1, value=12,
                            marks={2: '2', 15: '15', 30: '30'},
                            className="p-0"
                        ),
                    ], className="mb-3"),

                    html.Div([
                        html.Label("Minimum Distance (min_dist)", className="mb-0", style={'color': '#1f2937', 'fontSize': '14px'}),
                        dcc.Slider(
                            id='umap-dist-slider',
                            min=0.0, max=1.0, step=0.05, value=0.8,
                            marks={0: '0', 0.5: '0.5', 1: '1'},
                            className="p-0"
                        ),
                    ], className="mb-2"),

                    dbc.Checkbox(
                        id="umap-flippers-toggle",
                        label="Highlight Plan Flippers",
                        value=False,
                        className="mb-2",
                        style={'color': '#1f2937', 'fontSize': '14px'}
                    ),

                    html.Div([
                        html.Label("Color Metric", className="mb-0", style={'color': '#1f2937', 'fontSize': '14px'}),
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
                    html.Span("Step 2: Reasoning Path Analysis", id='level2-header-title', className="align-self-center font-weight-bold", style={'color': '#1f2937', 'fontSize': '14px'}),
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
                    dbc.Button("HELP", id="help-btn-step2", size="sm", color="info", outline=True, className="ms-2 py-0 px-2", style={'fontSize': '14px'})
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}),
                className="py-1"
            ),
            dbc.CardBody(
                html.Div([
                    html.Div([
                        create_level2_top(),
                        create_level2_probing_row(),
                        create_level2_bottom(),
                    ], style={'display': 'flex', 'flexDirection': 'column', 'height': '100%'}),
                    # Full-card help overlay shown until a sample is selected (toggle_help_overlay).
                    html.Div(help_page(), id='level2-help-overlay', style=HELP_OVERLAY_STYLE),
                ], style={'position': 'relative', 'height': '100%'}),
                className="p-1")
        ], style={'height': '55vh', 'flexShrink': 0}, className="mb-2"),

        # Level 3: Token Specifics
        dbc.Card([
            dbc.CardHeader(
                html.Div([
                    html.Span("Step 3: Token Details", id='level3-instructions', className="align-self-center", style={'color': '#1f2937', 'fontSize': '14px'}),
                    dbc.Button("HELP", id="help-btn-step3", size="sm", color="info", outline=True, className="ms-auto py-0 px-2", style={'fontSize': '14px'})
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}),
                className=_HEADER_CLASS
            ),
            dbc.CardBody(create_level3_detail(), className="p-1", style={'display': 'flex', 'flexDirection': 'column', 'flex': 1, 'minHeight': 0})
        ], style={'flex': 1, 'minHeight': 0, 'display': 'flex', 'flexDirection': 'column'})
    ], style={'height': '100%', 'display': 'flex', 'flexDirection': 'column'})


def create_help_modals():
    return html.Div([
        # Hidden target for the header "?" clientside reload callback.
        html.Div(id="help-reload-dummy", style={'display': 'none'}),

        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Help: Sample Selection & UMAP Tuner")),
            dbc.ModalBody(dcc.Markdown("""
This interactive scatter plot projects high-dimensional latent reasoning representations from the VLM into a 2D space. Each point represents a specific maze-solving sample, allowing you to explore how the model's internal reasoning clusters across different scenarios.

## Zooming (Macro vs. Micro View)
The plot dynamically transitions between two views based on your zoom level:

#### 1. Macro View (Zoomed out)
* **Glyphs (Next Move):** Shapes represent the model's predicted next move:
  * `▲` UP
  * `▼` DOWN
  * `◀` LEFT
  * `▶` RIGHT
  * `●` UNKNOWN
* **Glyph Size:** Scales with Reasoning Intensity (KL divergence).
* **Convex Hulls:** At very high-level zoom level, shaded background regions enclose clusters of the same directional decision.
* **Projection Error Aura:** Starting when zoomed a bit in, a gray circular aura fades in around points. The size of this aura indicates UMAP placement uncertainty (projection error). It fades out as you approach the Micro View.

#### 2. Micro View (Zoomed in)
As the zoom level becomes more fine-grained, the macro triangles and auras fade out, revealing miniature mazes directly on the map.
* **Light Gray Lines:** The maze grid topology.
* **Black Squares:** Walls and obstacles.
* **Blue Circle:** Start point.
* **Green Circle:** End goal.
* **Red Line:** The VLM's predicted solution path.

## UMAP Tuner & Color Metrics
* **Nearest Neighbors**: Controls the balance between local fine-grained clusters (low) and global structure (high).
* **Minimum Distance**: Controls how tightly points are packed. Lower values create tighter clumps.
* **Color Dropdown**: Recolor points by metrics like **Reasoning Intensity** (average KL divergence), **Level ID**, or **Projection Error**.
* **Plan Flippers**: Toggle to outline samples with a red border where altering latent tokens caused the model to change its planned path.
            """)),
        ], id="help-modal-step1", is_open=False, size="lg", centered=True),

        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Help: Reasoning Path Analysis")),
            dbc.ModalBody(html.Div([
                html.P("Step 2 shows the reasoning path for the sample you selected in Step 1. It has two tabs:"),
                html.Ul([
                    html.Li([html.B("Probing"), " — how decodable the next move is from the latent tokens (RQ2)."]),
                    html.Li([html.B("Ablation"), " — what happens to the answer when latent tokens are switched off (RQ3)."]),
                ]),
                html.P("Click a sample point in Step 1 to load it here, then click any latent-token "
                       "heatmap to drill into per-token detail in Step 3."),
            ])),
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