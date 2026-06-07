from dash import html, dcc
import dash_bootstrap_components as dbc

def create_level3_detail():
    return html.Div([
        dbc.Row([
            dbc.Col(dcc.Graph(id='token-heatmap', style={'height': '30vh'}), width=4),
            dbc.Col(dcc.Graph(id='token-probe-bar', style={'height': '30vh'}), width=4),
            dbc.Col(dcc.Graph(id='token-dependency-curve', style={'height': '30vh'}), width=4),
        ], className="g-0"),
        dbc.Row([
            dbc.Col([
                dbc.ButtonGroup([
                    dbc.Button("Ablate Token", id="ablate-btn", color="warning", size="sm"),
                    html.Div(id="ablate-output", className="ml-3 small font-weight-bold text-danger d-inline-block", style={'paddingTop': '5px'})
                ])
            ], width=12)
        ], className="mt-1")
    ])
