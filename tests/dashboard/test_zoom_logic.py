import pytest
from dashboard.components.level1_landscape import create_level1_landscape
import numpy as np

def test_zoom_culling_and_rendering():
    """Verify Phase 4 zoom-aware logic with aura fading."""
    
    test_data = [
        {
            'sample_id': 's1', 'umap_x': 0, 'umap_y': 0, 
            'umap_uncertainty': 0.1, 'correctness': True, 
            'move_direction': 'UP', 'tokens': [{'kl_divergence': 0.5}],
            'level_id': 1, 'seq_len': 100, 'num_latent': 6,
            'map_desc': [[1, 0], [0, 2]], 'full_path': 'DOWN, RIGHT'
        }
    ]
    
    # 1. Macro view, zoom 1.0 (aura opacity should be 0)
    fig_macro1 = create_level1_landscape(test_data, zoom_level=1.0)
    trace_names1 = [t.name for t in fig_macro1.data]
    # At opacity 0, the trace is technically still added, but let's check its opacity
    if "Uncertainty Aura" in trace_names1:
        aura_trace = next(t for t in fig_macro1.data if t.name == "Uncertainty Aura")
        assert aura_trace.marker.opacity == 0.0
    
    # 2. Mid view, zoom 5.0 (aura opacity should be > 0)
    fig_macro2 = create_level1_landscape(test_data, zoom_level=5.0)
    aura_trace2 = next(t for t in fig_macro2.data if t.name == "Uncertainty Aura")
    assert aura_trace2.marker.opacity > 0.0
    assert aura_trace2.marker.opacity <= 0.15
    
    # 3. Micro view, zoom 20.0 (macro should be hidden/faded out)
    viewport = {'x_min': -2, 'x_max': 2, 'y_min': -2, 'y_max': 2}
    fig_micro = create_level1_landscape(test_data, zoom_level=20.0, viewport=viewport)
    trace_names_micro = [t.name for t in fig_micro.data]
    
    # At zoom 20.0, macro_opacity is 0, so the aura trace shouldn't even be added 
    # (since our logic says if aura_opacity > 0: add trace)
    assert "Uncertainty Aura" not in trace_names_micro
    assert "Maze Grid" in trace_names_micro

