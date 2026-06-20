from dash import html, dcc
import dash_bootstrap_components as dbc


def create_level3_detail():
    return html.Div([
        dcc.Store(id='current-token-state', data={}),
        dcc.Store(id='ablation-state', data={'ablated_ranks': []}),
        dbc.Row([
            dbc.Col(dcc.Graph(id='token-detail-heatmap',
                    style={'height': '30vh'}), width=4),
            dbc.Col(dcc.Graph(id='token-detail-probe-bar',
                    style={'height': '30vh'}), width=4),
            dbc.Col(dcc.Graph(id='token-detail-dependency-curve',
                    style={'height': '30vh'}), width=4),
        ], className="g-0"),
    ])
