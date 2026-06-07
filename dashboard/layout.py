import dash_bootstrap_components as dbc
from dash import html, dcc
from .components.level1_landscape import create_level1_landscape
from .components.level2_path import create_level2_path
from .components.level3_detail import create_level3_detail

def create_layout():
    return dbc.Container([
        # Main Dashboard Wrapper - 100vh, no scroll
        html.Div([
            # Top Header Bar (Compact)
            dbc.Row([
                dbc.Col(html.H4("Latent Reasoning VLM Analysis", className="text-primary m-0"), width=8),
                dbc.Col(html.Div(id="status-indicator", children="System Ready", className="text-right text-muted small"), width=4),
            ], className="py-2 border-bottom bg-light", style={'height': '5vh'}),

            # Main Body (95vh)
            dbc.Row([
                # Left Column: Selection & Sensitivity (20% width)
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Level 1: Samples", className="py-1 small font-weight-bold"),
                        dbc.CardBody(create_level1_landscape(), className="p-0")
                    ], style={'height': '45vh'}, className="mb-2"),
                    
                    dbc.Card([
                        dbc.CardHeader("Phase 4: Sensitivity", className="py-1 small font-weight-bold"),
                        dbc.CardBody(dcc.Graph(id='sensitivity-subplots', style={'height': '100%'}), className="p-0")
                    ], style={'height': '43vh'})
                ], width=3, className="pr-1 py-2"),

                # Right Column: Analysis (80% width)
                dbc.Col([
                    # Level 2: Main Reasoning Path
                    dbc.Card([
                        dbc.CardHeader("Level 2: Reasoning Path Analysis", className="py-1 small font-weight-bold"),
                        dbc.CardBody(create_level2_path(), className="p-1")
                    ], style={'height': '45vh'}, className="mb-2"),

                    # Level 3: Token Specifics
                    dbc.Card([
                        dbc.CardHeader(html.Div(id='level3-instructions', children="Level 3: Token Details"), className="py-1 small font-weight-bold"),
                        dbc.CardBody(create_level3_detail(), className="p-1")
                    ], style={'height': '43vh'})
                ], width=9, className="pl-1 py-2")
            ], style={'height': '95vh'})
        ], style={'height': '100vh', 'overflow': 'hidden'})
    ], fluid=True, className="p-0")
