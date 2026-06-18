import pytest
from dashboard.components.level1_landscape import create_level1_landscape
from dashboard.components.level2_path import create_level2_path
from dashboard.components.level3_detail import create_level3_detail
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dashboard.mock_data import MOCK_DATA

def test_level1_landscape_returns_figure():
    component = create_level1_landscape(MOCK_DATA)
    assert isinstance(component, go.Figure)

def test_level2_path_structure():
    component = create_level2_path()
    assert isinstance(component, dbc.Row)
<<<<<<< HEAD

=======
    
>>>>>>> 77f8af0 (Improve UMAP projections and dashboard visualization)
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
<<<<<<< HEAD
    assert 'level2-tabs' in ids
    assert 'level2-ablation-pane' in ids
    assert 'level2-ablation-tab' in ids
=======
    assert 'level2-ablation-pane' in ids
>>>>>>> 77f8af0 (Improve UMAP projections and dashboard visualization)
    assert 'level2-maze-pane' in ids
    assert 'level2-token-grid' in ids

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
<<<<<<< HEAD
    assert 'current-token-state' in ids
=======
    assert 'token-detail-dependency-curve' in ids
    assert 'ablate-btn' in ids
>>>>>>> 77f8af0 (Improve UMAP projections and dashboard visualization)
