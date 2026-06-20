import pytest
from pathlib import Path
import plotly.graph_objects as go
from dashboard.rq2_viz import load_rq2_data, build_rq2_static_bar, build_rq2_dynamic_grid

NEEDS_RQ2 = pytest.mark.skipif(
    not Path("data/processed/rq2/probe_results.json").exists(),
    reason="RQ2 probe data not found"
)


@NEEDS_RQ2
@NEEDS_RQ2
def test_load_rq2_data():
    """Verify that RQ2 JSON files exist and load correctly."""
    static, per_sample = load_rq2_data()
    assert "layers" in static
    assert "per_sample" in per_sample
    assert len(static["layers"]) > 0
    assert len(per_sample["per_sample"]) > 0

@NEEDS_RQ2
def test_build_rq2_static_bar():
    """Verify static bar chart generation."""
    fig = build_rq2_static_bar()
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "bar"
    assert "Decodability" in fig.layout.title.text

@NEEDS_RQ2
def test_build_rq2_dynamic_grid():
    """Verify dynamic grid generation for a valid sample."""
    # Test with numeric ID (per teammate's internal format)
    fig = build_rq2_dynamic_grid("0")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    assert "Decodability" in fig.layout.title.text
    
    # Test with dashboard-style ID (sample_XXX)
    fig2 = build_rq2_dynamic_grid("sample_000")
    assert "Decodability" in fig2.layout.title.text
    assert len(fig2.data) == 2

@NEEDS_RQ2
def test_build_rq2_dynamic_grid_invalid():
    """Verify handling of non-existent sample IDs."""
    fig = build_rq2_dynamic_grid("non_existent_id")
    assert "not found" in fig.layout.title.text
