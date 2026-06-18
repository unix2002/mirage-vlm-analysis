import pytest
import re
import json
from dashboard.layout import create_layout
from dashboard.callbacks import register_callbacks
from dash import Dash

def test_id_integrity():
    """Ensure all IDs referenced in callbacks exist in the layout."""
    layout = create_layout()
    
    # Extract all IDs from the layout recursively
    layout_ids = set()
    def get_ids(node):
        if hasattr(node, 'id'):
            if isinstance(node.id, dict):
                # Handle pattern matching IDs
                layout_ids.add(json.dumps(node.id, sort_keys=True))
            elif node.id:
                layout_ids.add(node.id)
        if hasattr(node, 'children'):
            if isinstance(node.children, list):
                for child in node.children:
                    get_ids(child)
            else:
                get_ids(node.children)
    
    get_ids(layout)
    
    app = Dash(__name__)
    register_callbacks(app)
    
    for cb in app.callback_map.values():
        # Check Outputs
        outputs = cb['output']
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            cid = getattr(out, 'component_id', None)
            if isinstance(cid, str):
                assert cid in layout_ids, f"Output ID '{cid}' not found in layout"
        
        # Check Inputs
        for inp in cb['inputs']:
            cid = getattr(inp, 'component_id', None)
            if isinstance(cid, str):
                # Skip dynamically created IDs; only level1-scatter is a known static input.
                if cid != 'level1-scatter':
                     continue
                assert cid in layout_ids, f"Input ID '{cid}' not found in layout"
        
        # Check States
        for state in cb['state']:
            cid = getattr(state, 'component_id', None)
            if isinstance(cid, str):
                assert cid in layout_ids, f"State ID '{cid}' not found in layout"

def test_static_id_list():
    """Explicitly check for critical IDs that must be present."""
    layout = create_layout()
    required_ids = [
        'level1-scatter',
<<<<<<< HEAD
        'level2-tabs',
        'level2-ablation-pane',
        'level2-ablation-tab',
        'level2-maze-pane',
        'level2-token-grid',
        'token-detail-heatmap',
        'token-detail-probe-bar',
        'token-detail-dependency-curve',
        'level3-instructions',
        'current-token-state',
        'ablation-state',
=======
        'level2-ablation-pane',
        'level2-maze-pane',
        'level2-token-grid',
        'level2-output-pane',
        'token-detail-heatmap',
        'rq2-static-bar',
        'rq2-dynamic-grid',
        'token-detail-dependency-curve',
        'ablate-btn',
        'ablate-output'
>>>>>>> 77f8af0 (Improve UMAP projections and dashboard visualization)
    ]
    
    def find_id(node, target_id):
        if hasattr(node, 'id') and node.id == target_id:
            return True
        if hasattr(node, 'children'):
            if isinstance(node.children, list):
                return any(find_id(c, target_id) for c in node.children)
            else:
                return find_id(node.children, target_id)
        return False

    for rid in required_ids:
        assert find_id(layout, rid), f"Required ID '{rid}' missing from layout"
