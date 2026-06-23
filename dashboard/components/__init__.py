from dash import html
import dash_bootstrap_components as dbc
import itertools

# Unique, collision-free ids for each tooltip (the button and its popover share one).
_tip_counter = itertools.count()


def info_tip(content):
    """Small blue ⓘ button that shows a popover on click, matching HELP button style.

    `content` may be a plain string or any Dash component(s) for richer, formatted help.
    """
    tip_id = f"tip-{next(_tip_counter)}"
    return html.Span([
        dbc.Button("ⓘ", id=tip_id, size="sm", color="info", outline=True,
                   className="me-1 py-0 px-1", style={'fontSize': '11px'}),
        dbc.Popover(content, target=tip_id, trigger="click", placement="bottom",
                    body=True, style={'fontSize': '13px', 'maxWidth': '330px'}),
    ])


def tip_body(title, intro=None, items=None):
    """Build readable popover content for new users: a bold title, an optional plain
    intro line, and an optional bullet list of the key things to look at.

    `intro` and each item may be a string or a list of Dash components (e.g. with html.B).
    """
    children = [html.Div(title, style={
        'fontWeight': 'bold', 'marginBottom': '5px', 'color': '#0e7490'})]
    if intro:
        children.append(html.Div(intro, style={
            'marginBottom': '7px' if items else '0', 'lineHeight': '1.45'}))
    if items:
        children.append(html.Ul(
            [html.Li(it, style={'marginBottom': '3px'}) for it in items],
            style={'margin': '0', 'paddingLeft': '18px', 'lineHeight': '1.4'}))
    return children
