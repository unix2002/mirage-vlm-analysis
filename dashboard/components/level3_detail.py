from dash import html, dcc

_HIDDEN = {'display': 'none'}
_VISIBLE = {'display': 'block', 'overflow': 'hidden', 'height': '100%'}


def create_level3_detail():
    return html.Div([
        dcc.Store(id='current-token-state', data={}),
        dcc.Store(id='ablation-state', data={'ablated_ranks': []}),
        html.Div("select a level 2 latent token", id='level3-placeholder',
                 className="text-muted small d-flex justify-content-center align-items-center h-100"),
        html.Div([
            html.Div([
                html.Div(dcc.Graph(id='token-detail-heatmap', style={'height': '24vh'}),
                         style={'flex': 1, 'minWidth': 0}),
                html.Div(dcc.Graph(id='token-detail-probe-bar', style={'height': '24vh'}),
                         style={'flex': 1, 'minWidth': 0}),
                html.Div(dcc.Graph(id='token-detail-dependency-curve', style={'height': '24vh'}),
                         style={'flex': 1, 'minWidth': 0}),
            ], style={'display': 'flex', 'flex': 1, 'minHeight': 0}),
            html.Div(dcc.Slider(
                id='layer-slider',
                min=0, max=26, step=1, value=26,
                marks={0: '0', 6: '6', 13: '13', 20: '20', 26: '26'},
                tooltip=dict(placement='bottom'),
            ), style={'width': '33.33%', 'marginTop': '0.25rem', 'flexShrink': 0}),
        ], id='level3-content', style=_HIDDEN),
    ], style={'position': 'relative', 'height': '100%'})
