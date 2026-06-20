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
        dbc.Row([
            dbc.Col(dcc.Slider(
                id='layer-slider',
                min=0, max=26, step=1, value=26,
                marks={0: '0', 6: '6', 13: '13', 20: '20', 26: '26'},
                tooltip=dict(placement='bottom'),
            ), width=4),
        ], className="mt-1"),
    ])
