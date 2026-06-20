from dash import html, dcc
import dash_bootstrap_components as dbc


def create_level2_top():
    """Pinned top section: maze + token heatmap grid."""
    return dbc.Row([
        dbc.Col(
            html.Div(id='level2-maze-pane', className='h-100',
                     style={'overflow': 'hidden'}),
            width=2, className='h-100',
            style={'paddingRight': '8px'},
        ),
        dbc.Col(
            html.Div(id='level2-token-grid', className='h-100',
                     style={'overflow': 'hidden'}),
            width=10, className='h-100',
            style={'paddingLeft': '8px'}),
    ], className="g-0", style={'flexShrink': 0, 'height': '15vh'})


def create_level2_plan_row():
    """Pinned plan + status row: predicted plan arrows + flip readout side by side."""
    return dbc.Row([
        dbc.Col(html.Div(id='level2-plan-status-row'), width=12),
    ], className="g-0", style={'flexShrink': 0})


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
    ], className="g-0", style={'flex': 1, 'minHeight': 0})
