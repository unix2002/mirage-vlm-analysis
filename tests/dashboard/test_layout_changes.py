import pytest
from dash import html
import dash_bootstrap_components as dbc
from dashboard.layout import create_sidebar, create_main_content
from dashboard.components.level3_detail import create_level3_detail

def test_sidebar_is_flex_coupled():
    sidebar = create_sidebar()
    assert isinstance(sidebar, html.Div)
    # The sidebar wrapper itself should have height: 100%
    assert sidebar.style.get('height') == '100%'
    
    children = sidebar.children
    assert isinstance(children, list)
    assert len(children) == 1
    
    card = children[0]
    assert isinstance(card, dbc.Card)
    # The card itself should have height: 100%
    assert card.style.get('height') == '100%'
    
    card_children = card.children
    assert len(card_children) == 2
    header, body = card_children
    assert header.children == "Step 1: Sample Selection & UMAP Tuner"

def test_level3_flex_coupling():
    main_content = create_main_content()
    # Main content wrapper should use flex column and height 100%
    assert main_content.style.get('height') == '100%'
    assert main_content.style.get('display') == 'flex'
    assert main_content.style.get('flexDirection') == 'column'

    assert len(main_content.children) == 2
    level2_card, level3_card = main_content.children
    
    # Step 3 block should use flex: 1 to fill remaining space
    assert level3_card.style.get('flex') == 1
    assert level3_card.style.get('display') == 'flex'
    
    # Check graph heights in level3_detail are decoupled from vh
    level3_detail = create_level3_detail()
    content_div = level3_detail.children[3]
    graphs_container = content_div.children[0]
    
    for graph_div in graphs_container.children:
        graph = graph_div.children
        # The graph should have height: 100%
        assert graph.style.get('height') == '100%'
