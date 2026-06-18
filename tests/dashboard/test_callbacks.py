import pytest
import json
from dashboard.callbacks import update_level2_logic, update_level3_logic
from dashboard.mock_data import MOCK_DATA
import plotly.graph_objects as go
from dash import html


def test_update_level2_logic_no_data():
<<<<<<< HEAD
    ablation_summary, maze_view, ablation_tab = update_level2_logic(None)
    assert isinstance(ablation_summary, html.Div)
    assert "Select a Sample (Level 1)" in ablation_summary.children
    assert isinstance(maze_view, html.Div)
    assert isinstance(ablation_tab, html.Div)

=======
    out = update_level2_logic(None)
    assert len(out) == 4
    assert "Select a Sample (Level 1)" in out[0].children
>>>>>>> 77f8af0 (Improve UMAP projections and dashboard visualization)

def test_update_level2_logic_valid_click():
    valid_id = MOCK_DATA[0]['sample_id']
    clickData = {'points': [{'hovertext': valid_id}]}
<<<<<<< HEAD
    ablation_summary, maze_view, ablation_tab = update_level2_logic(clickData)

    assert ablation_summary is not None
    assert isinstance(maze_view, html.Div)
    assert isinstance(ablation_tab, html.Div)

=======
    out = update_level2_logic(clickData)
    
    assert len(out) == 4
    # Check if ablation summary is first
    assert hasattr(out[0], 'figure') or isinstance(out[0], html.Div)
>>>>>>> 77f8af0 (Improve UMAP projections and dashboard visualization)

def test_update_level3_logic_no_clicks():
    fig1, fig2, fig3, text, store = update_level3_logic([0], None, None)
    assert text == "Level 3: Token Details"
    assert isinstance(store, dict)


def test_update_level3_logic_valid_token_click():
    valid_id = MOCK_DATA[0]['sample_id']
    clickData = {'points': [{'hovertext': valid_id}]}
    triggered_id = '{"index":"T0","type":"token-heatmap"}.n_clicks'
    n_clicks = [1]
<<<<<<< HEAD
=======
    
    fig1, fig2, fig3, text, ablate = update_level3_logic(n_clicks, clickData, triggered_id)
    
    assert f"Details: T0 ({valid_id})" in text
    assert isinstance(fig1, go.Figure) # Heatmap
    assert isinstance(fig2, go.Figure) # Bar
    assert isinstance(fig3, go.Figure) # Line
    
    assert fig1.layout.title.text == f"RQ1: Spatial Focus Heatmap (Token T0)"
>>>>>>> 77f8af0 (Improve UMAP projections and dashboard visualization)

    fig1, fig2, fig3, text, store = update_level3_logic(n_clicks, clickData, triggered_id)

    assert f"Details: T0 ({valid_id})" in text
    assert isinstance(fig1, go.Figure)  # Heatmap
    assert isinstance(fig2, go.Figure)  # Bar
    assert isinstance(fig3, go.Figure)  # Line

    assert fig1.layout.title.text == "RQ1: Spatial Focus Heatmap (Token T0)"
