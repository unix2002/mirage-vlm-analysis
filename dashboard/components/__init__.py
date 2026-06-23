from dash import html
import dash_bootstrap_components as dbc


def info_tip(text):
    """Small blue ⓘ button that shows a popover on click, matching HELP button style."""
    tip_id = f"tip-{abs(hash(text)) % 1000000}"
    return html.Span([
        dbc.Button("ⓘ", id=tip_id, size="sm", color="info", outline=True,
                   className="me-1 py-0 px-1", style={'fontSize': '11px'}),
        dbc.Popover(text, target=tip_id, trigger="click", placement="bottom",
                    body=True, style={'fontSize': '13px', 'maxWidth': '300px'}),
    ])
