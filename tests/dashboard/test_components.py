import pytest
from dashboard.components.level1_landscape import create_level1_landscape
from dashboard.components.level2_path import create_level2_top, create_level2_bottom, create_level2_probing_row
from dashboard.components.level3_detail import create_level3_detail
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dashboard.mock_data import MOCK_DATA

def test_level1_landscape_returns_graph():
    component = create_level1_landscape(MOCK_DATA)
    # The component is now returning a go.Figure instead of dcc.Graph directly
    assert isinstance(component, go.Figure)
    assert component.layout.title.text == "UMAP Latent Reasoning Landscape"

def test_level2_path_structure():
    top = create_level2_top()
    bottom = create_level2_bottom()
    probing = create_level2_probing_row()
    
    assert isinstance(top, dbc.Row)
    assert isinstance(bottom, dbc.Row)
    assert isinstance(probing, dbc.Row)

    ids = []
    def find_ids(node):
        if hasattr(node, 'id') and node.id:
            ids.append(node.id)
        if hasattr(node, 'children'):
            if isinstance(node.children, list):
                for child in node.children:
                    find_ids(child)
            else:
                find_ids(node.children)

    find_ids(top)
    find_ids(bottom)
    find_ids(probing)
    
    assert 'level2-maze-pane' in ids
    assert 'level2-token-grid' in ids
    assert 'level2-output-pane' in ids
    assert 'level2-kl-pane' in ids
    assert 'level2-probing-pane' in ids

def test_level3_detail_structure():
    component = create_level3_detail()
    assert isinstance(component, html.Div)

    ids = []
    def find_ids(node):
        if hasattr(node, 'id') and node.id:
            ids.append(node.id)
        if hasattr(node, 'children'):
            if isinstance(node.children, list):
                for child in node.children:
                    find_ids(child)
            else:
                find_ids(node.children)

    find_ids(component)
    assert 'token-detail-heatmap' in ids
    assert 'token-detail-probe-bar' in ids
    assert 'current-token-state' in ids
