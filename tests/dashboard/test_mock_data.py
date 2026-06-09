import pytest
from dashboard.mock_data import generate_mock_data

def test_generate_mock_data_structure():
    n_samples = 10
    data = generate_mock_data(n_samples=n_samples)
    
    assert len(data) == n_samples
    for sample in data:
        assert 'sample_id' in sample
        assert 'correctness' in sample
        assert 'move_direction' in sample
        assert 'umap_x' in sample
        assert 'umap_y' in sample
        assert 'tokens' in sample
        assert 'attention_weights' in sample
        
        # Check directions
        assert sample['move_direction'] in ['UP', 'DOWN', 'LEFT', 'RIGHT', 'UNKNOWN']
        
        # Check tokens
        assert len(sample['tokens']) == 6
        for token in sample['tokens']:
            assert 'token_id' in token
            assert 'spatial_focus' in token
            assert 'probe_accuracy' in token
            assert 'kl_divergence' in token
            
            # RQ1: Spatial focus should be 14x14
            spatial = token['spatial_focus']
            assert len(spatial) == 14
            assert len(spatial[0]) == 14
            
            # RQ2: Probe accuracy bounds
            assert 0.0 <= token['probe_accuracy'] <= 1.0
            
        # Check attention weights (6x6)
        attn = sample['attention_weights']
        assert len(attn) == 6
        assert len(attn[0]) == 6

def test_mock_data_determinism():
    data1 = generate_mock_data(n_samples=5)
    data2 = generate_mock_data(n_samples=5)
    
    # Since seed is set inside generate_mock_data, they should be identical
    assert data1 == data2
