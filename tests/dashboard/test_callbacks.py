import pytest
import json
from dashboard.callbacks import update_level2_logic, update_level3_logic, ablate_token_logic
from dashboard.mock_data import MOCK_DATA
import plotly.graph_objects as go
from dash import html

def test_update_level2_logic_no_data():
    children, figure = update_level2_logic(None)
    assert isinstance(children, html.Div)
    assert "Select a Sample (Level 1)" in children.children
    assert isinstance(figure, go.Figure)

def test_update_level2_logic_valid_click():
    clickData = {'points': [{'hovertext': 'sample_0'}]}
    children, figure = update_level2_logic(clickData)
    
    assert isinstance(children, html.Div)
    # Check if glyphs are generated (should have 6 tokens)
    assert len(children.children) == 6
    assert isinstance(figure, go.Figure)
    assert figure.data[0].type == 'heatmap'

def test_update_level3_logic_no_clicks():
    fig1, fig2, fig3, text, ablate = update_level3_logic([0], None, None)
    assert text == "Level 3: Token Details"

def test_update_level3_logic_valid_token_click():
    clickData = {'points': [{'hovertext': 'sample_0'}]}
    triggered_id = '{"index":"T0","type":"token-glyph"}.n_clicks'
    n_clicks = [1]
    
    fig1, fig2, fig3, text, ablate = update_level3_logic(n_clicks, clickData, triggered_id)
    
    assert "Details: T0 (sample_0)" in text
    assert isinstance(fig1, go.Figure) # Heatmap
    assert isinstance(fig2, go.Figure) # Bar
    assert isinstance(fig3, go.Figure) # Line
    
    assert fig1.layout.title.text == "RQ1: Focus T0"

def test_ablate_token_logic():
    assert ablate_token_logic(0) == ""
    assert "Ablation Result" in ablate_token_logic(1)
