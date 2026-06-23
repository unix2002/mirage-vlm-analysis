from dash import html, dcc
from . import info_tip, tip_body

_HIDDEN = {'display': 'none'}
_VISIBLE = {'display': 'block', 'overflow': 'hidden', 'height': '100%'}


def create_level3_detail():
    return html.Div([
        dcc.Store(id='current-token-state', data={}),
        dcc.Store(id='ablation-state', data={'ablated_ranks': []}),
        # Hidden until a sample is selected in Step 1 (see reset_level3_on_sample);
        # then shown as the "select a latent token" prompt until a token is clicked.
        html.Div("select a latent token (step 2)", id='level3-placeholder',
                 className="justify-content-center align-items-center h-100",
                 style={'color': '#1f2937', 'fontSize': '14px', 'display': 'none'}),
        html.Div([
            html.Div([
                html.Div([html.Div(info_tip(tip_body(
                    "RQ1: Latent Token Heatmap",
                    "This heatmap shows the spatial latent token attention to the original input image of the model, with a slider to select a specific layer (0 to 26).")),
                                   style={'position': 'absolute', 'top': 2, 'left': 4, 'zIndex': 10}),
                          html.Div(id='token-detail-heatmap-bg', style={'display': 'none'}),
                          dcc.Graph(id='token-detail-heatmap',
                                    style={'height': '100%', 'position': 'relative', 'zIndex': 1,
                                           'backgroundColor': 'transparent'})],
                         style={'position': 'relative', 'flex': 1, 'minWidth': 0}),
                html.Div([html.Div(info_tip(tip_body(
                    "RQ2: Direction Probe",
                    "Asks a simple classifier: just from this one token, which way does the model want to move first?",
                    [
                        "One bar per direction (LEFT, DOWN, RIGHT, UP); taller = more probable.",
                        "The true first move is highlighted in cyan.",
                        "The scale is logarithmic, so even very small probabilities stay visible.",
                        "This reads the move out of the token; it is not the model's own output.",
                    ])),
                                   style={'position': 'absolute', 'top': 2, 'left': 4, 'zIndex': 10}),
                          dcc.Graph(id='token-detail-probe-bar', style={'height': '100%'})],
                         style={'position': 'relative', 'flex': 1, 'minWidth': 0}),
                html.Div([html.Div(info_tip(tip_body(
                    "RQ3: Token Ablation Contributions",
                    "How much each latent token matters to the answer, with two bars per token.",
                    [
                        [html.B("Individual (grey): "), "the change (KL) from zeroing only that token on its own."],
                        [html.B("Marginal (cyan): "), "the extra change it adds, averaged over all the combinations that don't already include it."],
                        "The token you clicked is highlighted in cyan.",
                        "A token can look weak on its own yet still matter once others are removed.",
                    ])),
                                   style={'position': 'absolute', 'top': 2, 'left': 4, 'zIndex': 10}),
                          dcc.Graph(id='token-detail-dependency-curve', style={'height': '100%'})],
                         style={'position': 'relative', 'flex': 1, 'minWidth': 0}),
            ], style={'display': 'flex', 'flex': 1, 'minHeight': 0}),
            html.Div(dcc.Slider(
                id='layer-slider',
                min=0, max=26, step=1, value=26,
                marks={0: '0', 6: '6', 13: '13', 20: '20', 26: '26'},
                tooltip=dict(placement='bottom'),
            ), style={'width': '33.33%', 'marginTop': '0.25rem', 'flexShrink': 0}),
        ], id='level3-content', style=_HIDDEN),
    ], style={'position': 'relative', 'height': '100%'})
