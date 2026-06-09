
import pytest
import numpy as np
import os
from dashboard.data_loader import RealDataLoader
from dashboard.mock_data import MOCK_DATA

def test_data_loader_initialization():
    loader = RealDataLoader()
    assert loader.metadata is not None
    assert len(loader.metadata) > 0
    # loader.l2v_attn was removed in favor of per-sample loading

def test_sample_structure():
    assert len(MOCK_DATA) > 0
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
    # Spatial focus should be a square grid (e.g., 11x11 or 14x14 depending on maze size)
    for sample in MOCK_DATA[:5]:
        spatial_focus = np.array(sample['tokens'][0]['spatial_focus'])
        assert spatial_focus.ndim == 2
        assert spatial_focus.shape[0] == spatial_focus.shape[1]

def test_move_direction_parsing():
    loader = RealDataLoader()
    assert loader._extract_move_direction("<think></think><output_image>\\boxed{DOWN}") == "DOWN"
    assert loader._extract_move_direction("<think></think><output_image>\\boxed{RIGHT, DOWN}") == "RIGHT"
    assert loader._extract_move_direction("No box here") == "UNKNOWN"

def test_integration_path_handling():
    # Verify it can handle both real and fallback data dirs
    loader = RealDataLoader(data_dir='../../mirage_data/extracted')
    valid_paths = [
        '../mirage_data/extracted', 
        '../../mirage_data/extracted', 
        '/gpfs/home1/scur0241/mirage_data/extracted', 
        'data/reference'
    ]
    assert loader.data_dir in valid_paths
    assert len(loader.get_data()) > 0

