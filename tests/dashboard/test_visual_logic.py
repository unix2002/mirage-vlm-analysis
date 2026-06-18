import pytest
from dashboard.callbacks import update_level3_logic, update_level2_logic
from dashboard.mock_data import MOCK_DATA
import numpy as np

def test_rq1_visual_heatmap_data():
    """Verify RQ1 spatial focus heatmap data bounds."""
    sample = MOCK_DATA[0]
    clickData = {'points': [{'hovertext': sample['sample_id']}]}
    triggered_id = '{"index":"T0","type":"token-heatmap"}.n_clicks'
    
    fig_heatmap, _, _, _, _ = update_level3_logic([1], clickData, triggered_id)
    
    z_data = fig_heatmap.data[0].z
    assert np.max(z_data) <= 1.0
    assert np.min(z_data) >= 0.0
    
    # Grid size should match the data (dynamic)
    expected_grid = len(sample['tokens'][0]['spatial_focus'])
    assert np.array(z_data).shape == (expected_grid, expected_grid)

def test_rq2_visual_bar_logic():
    """Verify RQ2 probe accuracy bar chart logic."""
    # Find a correct sample
    sample = next(s for s in MOCK_DATA if s['correctness'])
    clickData = {'points': [{'hovertext': sample['sample_id']}]}
    triggered_id = '{"index":"T0","type":"token-heatmap"}.n_clicks'
    
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
    triggered_id = '{"index":"T0","type":"token-heatmap"}.n_clicks'
    
    _, _, fig_curve, _, _ = update_level3_logic([1], clickData, triggered_id)
    
    y_vals = fig_curve.data[0].y
    # In mock logic, it's exponentially decaying: kls = [token['kl_divergence'] * np.exp(-0.2 * s) for s in steps]
    assert y_vals[0] > y_vals[-1]

<<<<<<< HEAD
def test_level2_ablation_tab():
    """Verify Level 2 now returns an ablation tab as the third element."""
    sample = MOCK_DATA[0]
    clickData = {'points': [{'hovertext': sample['sample_id']}]}
    
    ablation_summary, maze_view, ablation_tab = update_level2_logic(clickData)
    
    # Ablation summary is a graph for real data; for mock data it may be a placeholder.
    assert ablation_summary is not None
    # Maze view is an html.Div containing the maze image.
    assert maze_view is not None
    # Ablation tab is an html.Div with the ablation landscape.
    assert ablation_tab is not None
=======
from dashboard.components.level1_landscape import create_level1_landscape

def test_level1_static_glyphs_logic():
    """Verify Phase 2 visual logic for Auras and Velocity glyphs."""
    test_data = [
        {
            'sample_id': 's1', 'umap_x': 0, 'umap_y': 0, 
            'umap_uncertainty': 0.1, 'correctness': True, 
            'move_direction': 'UP', 'tokens': [{'kl_divergence': 0.5}, {'kl_divergence': 0.5}],
            'level_id': 1, 'seq_len': 100, 'num_latent': 6
        },
        {
            'sample_id': 's2', 'umap_x': 1, 'umap_y': 1, 
            'umap_uncertainty': 0.9, 'correctness': False, 
            'move_direction': 'DOWN', 'tokens': [{'kl_divergence': 0.1}, {'kl_divergence': 0.1}],
            'level_id': 1, 'seq_len': 100, 'num_latent': 6
        }
    ]
    
    # We now default to zoom_level=1.0 where auras are hidden. 
    # To test the auras trace being present we use zoom_level=5.0
    fig = create_level1_landscape(test_data, color_metric='avg_kl', zoom_level=5.0)
    
    aura_trace = next((t for t in fig.data if t.name == "Uncertainty Aura"), None)
    velocity_trace = next((t for t in fig.data if t.mode == 'markers' and t.name != "Uncertainty Aura" and t.name != 'Maze Hover' and 'Cluster' not in str(t.name)), None)
    
    assert aura_trace is not None
    assert velocity_trace is not None
    
    # 1. Test Uncertainty Auras (size mapping: 10 + uncertainty * 60)
    # s1 uncertainty 0.1 -> 10 + 6 = 16
    # s2 uncertainty 0.9 -> 10 + 54 = 64
    np.testing.assert_allclose(aura_trace.marker.size, [16, 64])
    
    # 2. Test Velocity Glyphs (size mapping: 8 + norm_kl * 14)
    # s1 avg_kl=0.5 (max, norm=1) -> 8 + 14 = 22
    # s2 avg_kl=0.1 (min, norm=0) -> 8 + 0 = 8
    np.testing.assert_allclose(velocity_trace.marker.size, [22, 8], atol=1e-5)
    
    # 3. Test Correctness outline was removed and replaced with standard black outline
    # Now it should be rgba(0,0,0,0.5)
    assert velocity_trace.marker.line.color == 'rgba(0,0,0,0.5)'

>>>>>>> 77f8af0 (Improve UMAP projections and dashboard visualization)
