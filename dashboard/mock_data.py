
import numpy as np
from .data_loader import REAL_DATA

def generate_mock_data(n_samples=50):
    np.random.seed(42)
    
    # Define cluster centers for each direction to test Convex Hulls
    centers = {
        'UP': [2, 8],
        'DOWN': [2, 2],
        'LEFT': [8, 2],
        'RIGHT': [8, 8],
        'UNKNOWN': [5, 5]
    }

    samples = []
    for i in range(n_samples):
        sample_id = f"mock_{i:03d}"
        correctness = bool(np.random.choice([True, False], p=[0.8, 0.2]))
        move_dir = np.random.choice(list(centers.keys()))
        
        center = centers[move_dir]
        umap_x = float(np.random.normal(loc=center[0], scale=1.0))
        umap_y = float(np.random.normal(loc=center[1], scale=1.0))

        tokens = []
        for t in range(6):
            tokens.append({
                'token_id': f"T{t}",
                'spatial_focus': np.random.rand(11, 11).tolist(),
                'probe_accuracy': float(np.random.uniform(0.4, 1.0) if correctness else np.random.uniform(0.1, 0.6)),
                'kl_divergence': float(np.random.uniform(0.1, 2.0))
            })

        samples.append({
            'sample_id': sample_id,
            'correctness': correctness,
            'move_direction': move_dir,
            'umap_x': umap_x,
            'umap_y': umap_y,
            'tokens': tokens,
            'attention_weights': np.random.rand(6, 6).tolist(),
            'level_id': np.random.randint(1, 7),
            'seq_len': np.random.randint(400, 500),
            'num_latent': 6
        })
    return samples

# Use real data if available, otherwise fallback
if REAL_DATA:
    MOCK_DATA = REAL_DATA
else:
    MOCK_DATA = generate_mock_data()
