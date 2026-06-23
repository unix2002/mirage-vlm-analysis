"""Shared global help / project-overview content.

Rendered both as the main-pane empty state (before a sample is selected) and inside
the header '?' help modal, so the two always stay in sync. Kept left-aligned with
tight margins so the full text is readable in the narrow main pane.
"""
from dash import html

_ACCENT = '#06b6d4'

# Style for the full-card help overlay that covers the entire Step 2 area before a
# sample is selected. Toggled (display none/block) by toggle_help_overlay.
HELP_OVERLAY_STYLE = {
    'position': 'absolute', 'top': 0, 'left': 0, 'right': 0, 'bottom': 0,
    'backgroundColor': '#ffffff', 'overflowY': 'auto',
    'padding': '10px 16px', 'zIndex': 5,
}


def help_page():
    """Project overview + dashboard-layout guide as a Dash component."""
    def _section(title, body):
        return html.Div([
            html.Div(title, style={
                'fontSize': '13px', 'fontWeight': 'bold', 'color': _ACCENT,
                'textTransform': 'uppercase', 'letterSpacing': '0.06em', 'marginBottom': '4px',
            }),
            html.Div(body, style={'fontSize': '13px', 'color': '#374151', 'lineHeight': '1.55'}),
        ], style={'marginBottom': '11px'})

    def _row(tag, name, desc):
        children = [html.Span(tag, style={
            'display': 'inline-block', 'minWidth': '56px', 'fontWeight': 'bold',
            'color': _ACCENT if name else '#1f2937', 'fontFamily': 'monospace', 'fontSize': '12px',
        })]
        if name:
            children.append(html.B(name, style={'color': '#1f2937', 'fontSize': '13px'}))
            children.append(html.Span(' — ' + desc, style={'color': '#374151', 'fontSize': '13px'}))
        else:
            children.append(html.Span(desc, style={'color': '#374151', 'fontSize': '13px'}))
        return html.Div(children, style={'marginBottom': '5px'})

    return html.Div([
        html.Div('Visual Analytics System for Latent Reasoning', style={
            'fontSize': '19px', 'fontWeight': 'bold', 'color': '#111827', 'marginBottom': '2px'}),
        html.Div('Interactive exploration of latent visual reasoning in a vision-language model',
                 style={'fontSize': '13px', 'color': '#6b7280', 'marginBottom': '11px'}),

        _section('About', [
            "The Mirage model extends Qwen2.5-VL with ", html.B("latent visual tokens"),
            " — compressed image features the model reasons over instead of text. These tokens "
            "carry visual reasoning but are opaque: they cannot be read directly. This dashboard "
            "makes them inspectable across 996 VSP maze-navigation samples, combining spatial "
            "attention, linear probing, and causal ablation in one linked view.",
        ]),

        _section('Getting started', [
            "Click any point in the ", html.B("Step 1 landscape"), " (left) to load a sample. "
            "Then click a latent-token heatmap in Step 2 to drill into per-token detail in Step 3.",
        ]),

        _section('Finding help', [
            html.Div([
                html.B("HELP buttons"), " sit in each step's header (Steps 1, 2 and 3). Click one for a "
                "plain-language overview of that whole section.",
            ], style={'marginBottom': '6px'}),
            html.Div([
                html.B("ⓘ buttons"), " sit on each individual graph. Click one for a popover explaining "
                "how to read that specific figure. Every graph has one except the Step 1 UMAP landscape, "
                "whose HELP button covers it instead.",
            ]),
        ]),

        _section('Dashboard layout', [
            _row('Step 1', 'Sample Landscape',
                 'UMAP map of all 996 samples (left sidebar); recolor, zoom, and re-project with the tuner below it.'),
            _row('Step 2', 'Reasoning Path',
                 'Maze, predicted plan, and latent-token heatmaps, with two tabs:'),
            html.Div([
                html.Div([html.B('Probing'), ' — layer-wise and per-step decodability (RQ2).'],
                         style={'fontSize': '13px', 'color': '#374151', 'marginLeft': '56px'}),
                html.Div([html.B('Ablation'), ' — dose-response curve and the KL fingerprint of all 63 token subsets (RQ3).'],
                         style={'fontSize': '13px', 'color': '#374151', 'marginLeft': '56px'}),
            ], style={'marginBottom': '5px'}),
            _row('Step 3', 'Token Details',
                 'Per-token spatial focus across layers, direction probe, and ablation contributions.'),
        ]),

        _section('Research questions', [
            _row('RQ1', '', 'Do latent tokens focus on key visual regions? (spatial attention heatmaps)'),
            _row('RQ2', '', 'Do they hold enough information to predict the answer? (linear probing)'),
            _row('RQ3', '', 'Does the answer causally depend on them? (token ablation → KL divergence)'),
        ]),
    ], style={'maxWidth': '900px', 'margin': 0, 'padding': 0})
