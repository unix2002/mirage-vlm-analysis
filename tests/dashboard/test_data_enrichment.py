import pytest
from dashboard.data_loader import RealDataLoader

def test_data_enrichment():
    loader = RealDataLoader()
    data = loader.get_data()
    
    assert len(data) > 0, "No data loaded"
    
    samples_with_maze = [s for s in data if s.get('map_desc') is not None]
    assert len(samples_with_maze) > 0, "No samples were enriched with maze data"
    
    print(f"Total samples: {len(data)}, Enriched: {len(samples_with_maze)}")
    
    # Check alignment for the first few enriched samples
    for sample in samples_with_maze[:10]:
        assert 'map_desc' in sample
        assert 'full_path' in sample
        assert sample['full_path'] is not None
        
        # Verify if full path starts with the move direction, if it was correct
        if sample['correctness'] and sample['move_direction'] != "UNKNOWN":
            assert sample['move_direction'] in sample['full_path'], "Move direction doesn't match full path"

if __name__ == "__main__":
    test_data_enrichment()
