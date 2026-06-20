from dash import html, dcc
import dash_bootstrap_components as dbc


def create_level3_detail():
    return html.Div([
        dcc.Store(id='current-token-state', data={}),
        dcc.Store(id='ablation-state', data={'ablated_ranks': []}),
        html.Div("Select a Sample (Level 1)", id='level3-detail-content', className="text-muted small d-flex justify-content-center align-items-center h-100"),
    ])
