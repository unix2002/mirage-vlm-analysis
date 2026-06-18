from dash import html, dcc
import dash_bootstrap_components as dbc


def create_level3_detail():
    return html.Div([
        dcc.Store(id='current-token-state', data={}),
        dcc.Store(id='ablation-state', data={'ablated_ranks': []}),
        dbc.Row([
            dbc.Col(dcc.Graph(id='token-detail-heatmap',
                    style={'height': '35vh'}), width=3),
            dbc.Col(dcc.Graph(id='rq2-static-bar',
                    style={'height': '35vh'}), width=3),
            dbc.Col(dcc.Graph(id='rq2-dynamic-grid',
                    style={'height': '35vh'}), width=3),
            dbc.Col(dcc.Graph(id='token-detail-dependency-curve',
                    style={'height': '35vh'}), width=3),
        ], className="g-0"),
    ])
