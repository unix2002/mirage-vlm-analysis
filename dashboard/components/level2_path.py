from dash import html, dcc
import dash_bootstrap_components as dbc

def create_level2_path():
    return dbc.Row([
        dbc.Col([
            html.Div(id='level2-image-pane', style={
                'height': '38vh', 'backgroundColor': '#f8f9fa', 
                'border': '1px solid #dee2e6', 'position': 'relative',
                'backgroundImage': 'linear-gradient(45deg, #eee 25%, transparent 25%, transparent 75%, #eee 75%, #eee), linear-gradient(45deg, #eee 25%, transparent 25%, transparent 75%, #eee 75%, #eee)',
                'backgroundSize': '15px 15px',
                'backgroundPosition': '0 0, 7.5px 7.5px'
            })
        ], width=7),
        dbc.Col([
            dcc.Graph(id='level2-flow-pane', style={'height': '38vh'})
        ], width=5)
    ], className="g-0")
