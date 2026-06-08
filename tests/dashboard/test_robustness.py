import pytest
from dashboard.callbacks import update_level2_logic, update_level3_logic
from dashboard.mock_data import MOCK_DATA

def test_callback_robustness_exhaustive():
    """Run every mock sample through Level 2 and Level 3 logic to ensure no crashes."""
    for sample in MOCK_DATA:
        sample_id = sample['sample_id']
        clickData = {'points': [{'hovertext': sample_id}]}
        
        # Test Level 2
        children, fig_flow = update_level2_logic(clickData)
        assert children is not None
        assert fig_flow is not None
        
        # Test Level 3 for every token in this sample
        for token in sample['tokens']:
            token_id = token['token_id']
            triggered_id = f'{{"index":"{token_id}","type":"token-glyph"}}.n_clicks'
            
            fig_heatmap, fig_bar, fig_curve, text, ablate = update_level3_logic([1], clickData, triggered_id)
            
            assert token_id in text
            assert fig_heatmap.data[0].type == 'heatmap'
            assert fig_bar.data[0].type == 'bar'
            assert fig_curve.data[0].type == 'scatter' # line is scatter in plotly go

def test_data_schema_strict():
    """Ensure mock data adheres to exact research specs (RQ1, RQ2, RQ3)."""
    for sample in MOCK_DATA:
        # RQ1: Spatial focus 14x14
        for token in sample['tokens']:
            grid = token['spatial_focus']
            assert len(grid) == 14, f"RQ1 grid height mismatch in {sample['sample_id']}"
            assert all(len(row) == 14 for row in grid), f"RQ1 grid width mismatch in {sample['sample_id']}"
            
        # RQ2: Information content 0-1
        for token in sample['tokens']:
            assert 0.0 <= token['probe_accuracy'] <= 1.0
            
        # RQ3: Causal dependence (KL div) positive
        for token in sample['tokens']:
            assert token['kl_divergence'] > 0
            
        # RQ3: Attention weights 6x6
        attn = sample['attention_weights']
        assert len(attn) == 6
        assert all(len(row) == 6 for row in attn)
