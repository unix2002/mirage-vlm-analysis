from dash import html
import dash_bootstrap_components as dbc


def info_tip(text, placement="right"):
    """Small ⓘ icon with Bootstrap tooltip on hover."""
    tip_id = f"tooltip-{hash(text) & 0xFFFFFFFF}"
    return html.Span([
        html.Span("ⓘ", id=tip_id, style={
            'fontSize': '0.55rem', 'color': '#94a3b8', 'cursor': 'help',
            'marginLeft': '4px', 'verticalAlign': 'super',
        }),
        dbc.Tooltip(text, target=tip_id, placement=placement, style={'fontSize': '0.65rem'}),
    ])
