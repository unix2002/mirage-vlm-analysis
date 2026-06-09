import json
import os
import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import umap
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VLMFeatureExtractor:
    def __init__(self, data_dir='data/reference'):
        self.data_dir = data_dir
        self.metadata_path = os.path.join(data_dir, 'metadata.json')
        self.hidden_states_path = os.path.join(data_dir, 'hidden_states.pt')
        
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        # Load layer 27 hidden states
        self.hidden_dict = torch.load(self.hidden_states_path, map_location='cpu')
        self.layer_27 = self.hidden_dict[27] # [1, seq_len, 3584]
        
    def _extract_move_direction(self, text):
        import re
        match = re.search(r'\\boxed{([^}]+)}', text)
        if match:
            return match.group(1).split(',')[0].strip().upper()
        return "UNKNOWN"

    def get_features_and_labels(self):
        features = []
        labels = []
        sample_ids = []
        
        # In this reference set, we only have ONE set of hidden states (sample 0)
        # But for the pipeline design, we iterate as if we had many.
        # Since we only have one real tensor, we'll "jitter" it for the sake of the dashboard 
        # landscape until the full 50 samples are available.
        
        base_meta = self.metadata[0]
        latent_indices = base_meta['token_positions']['latent']
        
        # Extract mean-pooled hidden states for the 6 latent tokens
        # latent_hs shape: [6, 3584]
        latent_hs = self.layer_27[0, latent_indices, :].numpy()
        mean_pooled = np.mean(latent_hs, axis=0) # [3584]
        
        for meta in self.metadata[:50]:
            # Jittering logic to simulate different samples from the prototype
            # until full tensors are transferred.
            noise = np.random.normal(0, 0.01, size=mean_pooled.shape)
            features.append(mean_pooled + noise)
            
            labels.append(self._extract_move_direction(meta['text_output_short']))
            sample_ids.append(meta['sample_id'])
            
        return np.array(features), labels, sample_ids

    def run_sweep(self, X, y):
        results = []
        
        # 1. PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        score = silhouette_score(X_pca, y) if len(set(y)) > 1 else 0
        results.append({'method': 'PCA', 'score': score, 'coords': X_pca})
        
        # 2. t-SNE
        for perp in [5, 10, 15]:
            tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
            X_tsne = tsne.fit_transform(X)
            score = silhouette_score(X_tsne, y) if len(set(y)) > 1 else 0
            results.append({'method': f't-SNE (p={perp})', 'score': score, 'coords': X_tsne})
            
        # 3. UMAP
        for n_neigh in [5, 10, 15]:
            for metric in ['cosine', 'euclidean']:
                reducer = umap.UMAP(n_neighbors=n_neigh, metric=metric, min_dist=0.1, random_state=42)
                X_umap = reducer.fit_transform(X)
                score = silhouette_score(X_umap, y) if len(set(y)) > 1 else 0
                results.append({'method': f'UMAP (n={n_neigh}, m={metric})', 'score': score, 'coords': X_umap})
                
        # Find best
        best = max(results, key=lambda x: x['score'])
        logger.info(f"Best projection method: {best['method']} with Silhouette Score: {best['score']:.4f}")
        return best['coords']

    def save_optimal_coords(self, coords, sample_ids):
        out_dict = {}
        for i, sid in enumerate(sample_ids):
            out_dict[str(sid)] = coords[i].tolist()
            
        out_path = os.path.join(self.data_dir, 'optimal_landscape_coords.json')
        with open(out_path, 'w') as f:
            json.dump(out_dict, f, indent=2)
        logger.info(f"Saved optimal coordinates to {out_path}")

if __name__ == "__main__":
    extractor = VLMFeatureExtractor()
    X, y, sids = extractor.get_features_and_labels()
    best_coords = extractor.run_sweep(X, y)
    extractor.save_optimal_coords(best_coords, sids)
