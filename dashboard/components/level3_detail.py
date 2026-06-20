from dash import html, dcc
import dash_bootstrap_components as dbc


def create_level3_detail():
    return html.Div([
        dcc.Store(id='current-token-state', data={}),
        dcc.Store(id='ablation-state', data={'ablated_ranks': []}),
        html.Div(id='level3-detail-content'),
    ])
