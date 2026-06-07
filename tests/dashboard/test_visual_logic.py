import pytest
from dashboard.callbacks import update_level3_logic, update_level2_logic
from dashboard.mock_data import MOCK_DATA
import numpy as np

def test_rq1_visual_heatmap_data():
    """Verify RQ1 spatial focus heatmap data bounds."""
    sample = MOCK_DATA[0]
    clickData = {'points': [{'hovertext': sample['sample_id']}]}
    triggered_id = '{"index":"T0","type":"token-glyph"}.n_clicks'
    
    fig_heatmap, _, _, _, _ = update_level3_logic([1], clickData, triggered_id)
    
    z_data = fig_heatmap.data[0].z
    assert np.max(z_data) <= 1.0
    assert np.min(z_data) >= 0.0
    assert np.array(z_data).shape == (14, 14)

def test_rq2_visual_bar_logic():
    """Verify RQ2 probe accuracy bar chart logic."""
    # Find a correct sample
    sample = next(s for s in MOCK_DATA if s['correctness'])
    clickData = {'points': [{'hovertext': sample['sample_id']}]}
    triggered_id = '{"index":"T0","type":"token-glyph"}.n_clicks'
    
    _, fig_bar, _, _, _ = update_level3_logic([1], clickData, triggered_id)
    
    y_vals = fig_bar.data[0].y
    x_vals = fig_bar.data[0].x
    
    # The correct direction should have the highest accuracy in my mock logic
    correct_dir = sample['move_direction']
    correct_idx = list(x_vals).index(correct_dir)
    
    assert y_vals[correct_idx] == max(y_vals)

def test_rq3_visual_dependency_curve():
    """Verify RQ3 causal dependency curve trend."""
    sample = MOCK_DATA[0]
    clickData = {'points': [{'hovertext': sample['sample_id']}]}
    triggered_id = '{"index":"T0","type":"token-glyph"}.n_clicks'
    
    _, _, fig_curve, _, _ = update_level3_logic([1], clickData, triggered_id)
    
    y_vals = fig_curve.data[0].y
    # In mock logic, it's exponentially decaying: kls = [token['kl_divergence'] * np.exp(-0.2 * s) for s in steps]
    assert y_vals[0] > y_vals[-1]

def test_rq3_visual_flow_heatmap():
    """Verify RQ3 sequential flow attention matrix size."""
    sample = MOCK_DATA[0]
    clickData = {'points': [{'hovertext': sample['sample_id']}]}
    
    _, fig_flow = update_level2_logic(clickData)
    
    z_data = fig_flow.data[0].z
    assert np.array(z_data).shape == (6, 6)
