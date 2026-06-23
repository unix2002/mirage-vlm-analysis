from dash import html


def info_tip(text):
    """Small ⓘ icon with hover tooltip."""
    return html.Span("ⓘ", title=text, style={
        'fontSize': '0.55rem', 'color': '#94a3b8', 'cursor': 'help',
        'marginLeft': '4px', 'verticalAlign': 'super',
    })
