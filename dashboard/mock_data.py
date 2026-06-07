import numpy as np

def generate_mock_data(n_samples=50):
    np.random.seed(42)
    directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    
    samples = []
    for i in range(n_samples):
        sample_id = f"sample_{i}"
        correctness = bool(np.random.choice([True, False], p=[0.8, 0.2]))
        move_dir = np.random.choice(directions)
        umap_x = float(np.random.normal(loc=directions.index(move_dir), scale=0.5))
        umap_y = float(np.random.normal(loc=correctness, scale=0.5))
        
        tokens = []
        for t in range(6):
            # RQ1: Spatial Focus - 14x14 attention matrix
            spatial_focus = np.random.rand(14, 14).tolist()
            
            # RQ2: Information Content - probe accuracy (0.0 to 1.0)
            probe_acc = float(np.random.uniform(0.4, 1.0) if correctness else np.random.uniform(0.1, 0.6))
            
            # RQ3: Causal Dependence - KL divergence
            kl_div = float(np.random.uniform(0.1, 2.0))
            
            tokens.append({
                'token_id': f"T{t}",
                'spatial_focus': spatial_focus,
                'probe_accuracy': probe_acc,
                'kl_divergence': kl_div
            })
            
        # Attention weights across tokens (6x6 matrix)
        attention_weights = np.random.rand(6, 6).tolist()
        
        samples.append({
            'sample_id': sample_id,
            'correctness': correctness,
            'move_direction': move_dir,
            'umap_x': umap_x,
            'umap_y': umap_y,
            'tokens': tokens,
            'attention_weights': attention_weights
        })
        
    return samples

MOCK_DATA = generate_mock_data()
