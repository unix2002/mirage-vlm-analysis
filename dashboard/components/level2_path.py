from dash import html, dcc
import dash_bootstrap_components as dbc


def create_level2_path():
    return dbc.Row([
        dbc.Col(
            html.Div(id='level2-maze-pane', className='h-100',
                     style={'overflow': 'hidden'}),
            width=2, className='h-100',
            style={'paddingRight': '32px'},
        ),
        dbc.Col(
            html.Div(id='level2-token-grid', className='h-100',
                     style={'overflow': 'hidden'}),
            width=10, className='h-100',
            style={'paddingLeft': '32px'}),
    ], className="g-0", style={'flexShrink': 0, 'height': '22vh'})


def create_level2_bottom():
    return dbc.Row([
        dbc.Col(
            html.Div(id='level2-output-pane',
                     style={'height': '100%', 'overflowY': 'auto'}),
            width=6, className='h-100'),
        dbc.Col(
            html.Div(id='level2-kl-pane',
                     style={'height': '100%', 'overflow': 'auto'}),
            width=6, className='h-100',
            style={'borderLeft': '1px solid #e9ecef'}),
    ], className="g-0 mt-1", style={'flex': 1, 'minHeight': 0})
