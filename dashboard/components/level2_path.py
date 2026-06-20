from dash import html
import dash_bootstrap_components as dbc


def create_level2_top():
    """Pinned top section: plan row above maze (left) + token heatmap grid (right)."""
    return dbc.Row([
        dbc.Col(
            html.Div([
                html.Div(id='level2-plan-status-row', style={'flexShrink': 0}),
                html.Div(id='level2-maze-pane',
                         style={'flex': 1, 'minHeight': 0, 'overflow': 'hidden'}),
            ], style={'display': 'flex', 'flexDirection': 'column', 'height': '100%'}),
            width=3, className='h-100',
            style={'paddingRight': '8px'},
        ),
        dbc.Col(
            html.Div(id='level2-token-grid', className='h-100',
                     style={'overflow': 'hidden'}),
            width=9, className='h-100',
            style={'paddingLeft': '8px'}),
    ], className="g-0", style={'flexShrink': 0, 'height': '15vh', 'paddingBottom': '4px'})


def create_level2_bottom():
    """Bottom content: dose-response graph (left) + KL chart & token strip (right)."""
    return dbc.Row([
        dbc.Col(
            html.Div(id='level2-output-pane',
                     style={'height': '100%', 'overflowY': 'auto'}),
            width=7, className='h-100'),
        dbc.Col(
            html.Div(id='level2-kl-pane',
                     style={'height': '100%', 'overflow': 'auto'}),
            width=5, className='h-100'),
    ], className="g-0", style={'flex': 1, 'minHeight': 0, 'overflow': 'hidden'})


def create_level2_probing_row():
    """Full-width probing row: RQ2 bar chart + decodability grid side by side."""
    return dbc.Row([
        dbc.Col(html.Div(id='level2-probing-pane'), width=12),
    ], className="g-0", style={'flexShrink': 0})
