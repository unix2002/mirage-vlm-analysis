from dash import html
import dash_bootstrap_components as dbc


def create_level2_path():
    return dbc.Row([
        dbc.Col(
            html.Div(id='level2-ablation-pane', className='h-100'),
            width=3,
            className='h-100'
        ),
        dbc.Col(
            html.Div(id='level2-maze-pane', className='h-100'),
            width=4,
            className='h-100'
        ),
        dbc.Col(
            html.Div(id='level2-token-grid', className='h-100'),
            width=5,
            className='h-100'
        ),
    ], className="g-0", style={'height': '65%'})


def create_level2_bottom():
    return dbc.Row([
        dbc.Col(html.Div(id='level2-output-pane', className='h-100'), width=12, className='h-100')
    ], className="g-0 mt-1", style={'height': '27%'})
