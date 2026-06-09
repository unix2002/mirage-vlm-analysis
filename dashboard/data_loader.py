import json
import os
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class RealDataLoader:
    def __init__(self, data_dir='data/reference'):
        self.data_dir = data_dir
        self.metadata_path = os.path.join(data_dir, 'metadata.json')
        self.coords_path = os.path.join(data_dir, 'optimal_landscape_coords.json')
        
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        self.optimal_coords = {}
        if os.path.exists(self.coords_path):
            with open(self.coords_path, 'r') as f:
                self.optimal_coords = json.load(f)
            
        # For this integration, we use the reference tensors for all samples 
        # as a starting point, but we could easily extend this to load from 
        # sample-specific directories if they were available.
        self.l2v_attn = torch.load(os.path.join(data_dir, 'latent_to_visual_attn.pt'), map_location='cpu')
        self.embeddings = torch.load(os.path.join(data_dir, 'embeddings.pt'), map_location='cpu')
        self.hidden_states = torch.load(os.path.join(data_dir, 'hidden_states.pt'), map_location='cpu')
        
        self.processed_samples = self._process_all_samples()

    def _extract_move_direction(self, text):
        import re
        match = re.search(r'\\boxed{([^}]+)}', text)
        if match:
            # Take the first direction if multiple are listed
            return match.group(1).split(',')[0].strip().upper()
        return "UNKNOWN"

    def _process_all_samples(self):
        # We'll use the first 50 samples for the dashboard
        samples_to_process = self.metadata[:50]
        
        processed = []
        for i, meta in enumerate(samples_to_process):
            raw_sid = str(meta['sample_id'])
            sample_id = f"sample_{raw_sid}"
            move_dir = self._extract_move_direction(meta['text_output_short'])
            
            # Simple heuristic for correctness: if it has a boxed direction
            correctness = move_dir != "UNKNOWN"
            
            # Use pre-computed analytical coordinates if available
            if raw_sid in self.optimal_coords:
                umap_x, umap_y = self.optimal_coords[raw_sid]
            else:
                # Fallback to mock cluster if not computed
                directions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'UNKNOWN']
                dir_idx = directions.index(move_dir)
                umap_x = float(dir_idx + np.random.normal(0, 0.2))
                umap_y = float((1 if correctness else 0) + np.random.normal(0, 0.2))

            # Process tokens (RQ1, RQ2, RQ3)
            # REVERTED TO MOCK DATA per user request to avoid "prototype" duplication
            tokens = []
            num_latent = meta.get('num_latent', 6)
            
            for t in range(num_latent):
                # RQ1: Spatial Focus - Mock 14x14 grid as placeholder
                spatial_focus = np.random.rand(14, 14).tolist()
                
                # RQ2: Information Content (Mocked)
                probe_acc = float(0.5 + 0.4 * np.random.rand())
                
                # RQ3: Causal Dependence (Mocked)
                kl_div = float(0.1 + 1.5 * np.random.rand())
                
                tokens.append({
                    'token_id': f"T{t}",
                    'spatial_focus': spatial_focus,
                    'probe_accuracy': probe_acc,
                    'kl_divergence': kl_div
                })

            # Level 2 Path: Token-to-token attention (Mocked)
            attention_weights = np.random.rand(num_latent, num_latent).tolist()

            processed.append({
                'sample_id': sample_id,
                'correctness': correctness,
                'move_direction': move_dir,
                'umap_x': umap_x,
                'umap_y': umap_y,
                'tokens': tokens,
                'attention_weights': attention_weights,
                'metadata': meta # Keep original meta for path access
            })
            
        return processed

    def get_data(self):
        return self.processed_samples

# Singleton instance
try:
    LOADER = RealDataLoader()
    REAL_DATA = LOADER.get_data()
except Exception as e:
    print(f"Error loading real data: {e}")
    REAL_DATA = []
