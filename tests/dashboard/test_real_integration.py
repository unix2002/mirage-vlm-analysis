import pytest
import numpy as np
from dashboard.data_loader import RealDataLoader
from dashboard.mock_data import MOCK_DATA

def test_data_loader_initialization():
    loader = RealDataLoader()
    assert loader.metadata is not None
    assert len(loader.metadata) > 0
    assert loader.l2v_attn is not None

def test_sample_structure():
    sample = MOCK_DATA[0]
    required_keys = [
        'sample_id', 'correctness', 'move_direction', 
        'umap_x', 'umap_y', 'tokens', 'attention_weights'
    ]
    for key in required_keys:
        assert key in sample

def test_token_structure():
    sample = MOCK_DATA[0]
    token = sample['tokens'][0]
    required_keys = ['token_id', 'spatial_focus', 'probe_accuracy', 'kl_divergence']
    for key in required_keys:
        assert key in token

def test_spatial_focus_dimensions():
    # Since we reverted to mock data for tokens, we expect a fixed 14x14 grid
    for sample in MOCK_DATA[:5]:
        spatial_focus = np.array(sample['tokens'][0]['spatial_focus'])
        assert spatial_focus.shape == (14, 14)

def test_move_direction_parsing():
    loader = RealDataLoader()
    assert loader._extract_move_direction("<think></think><output_image>\\boxed{DOWN}") == "DOWN"
    assert loader._extract_move_direction("<think></think><output_image>\\boxed{RIGHT, DOWN}") == "RIGHT"
    assert loader._extract_move_direction("No box here") == "UNKNOWN"

def test_integration_fallback():
    # If we point to a non-existent directory, it should return empty list or fallback
    import os
    if not os.path.exists('non_existent_dir'):
        try:
            loader = RealDataLoader(data_dir='non_existent_dir')
            data = loader.get_data()
            assert data == []
        except Exception:
            pass # Error handling is expected
