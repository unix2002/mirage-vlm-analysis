
import json
import os
import torch
import numpy as np
import traceback
import warnings
import re
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress repetitive UMAP/Sklearn warnings
warnings.filterwarnings("ignore", message="n_jobs value 1 overridden")
warnings.filterwarnings("ignore", message="Graph is not fully connected")

# Strict UMAP requirement
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    logging.warning("umap-learn is not installed. Clustering projections will fail.")

class RealDataLoader:
    def __init__(self, data_dir=None):
        self.data_dir = None
        self.metadata = []
        self.processed_samples = []
        
        # Caching for re-projection
        self.X_raw = None # Raw cached hidden states (float32)
        self.valid_indices = []
        
        try:
            self._initialize_paths(data_dir)
            if self.data_dir:
                self.processed_samples = self._process_all_samples()
                if self.processed_samples:
                    logging.info(f"Successfully loaded {len(self.processed_samples)} samples from {self.data_dir}")
                else:
                    logging.warning("No samples processed.")
            else:
                logging.error("No real data found. Dashboard will fallback to synthetic data.")
        except Exception as e:
            logging.error(f"Critical error during data initialization: {str(e)}")
            logging.debug(traceback.format_exc())

    def _initialize_paths(self, override_dir=None):
        possible_dirs = [
            override_dir,
            'real_data/mirage_extracted',    # Local real dataset
            '../mirage_data/extracted',      # From remco/
            '../../mirage_data/extracted',   # From remco/mirage-vlm-analysis/
            '/gpfs/home1/scur0241/mirage_data/extracted', # Absolute path
            'data/reference'                 # Local fallback
        ]
        
        for d in possible_dirs:
            if d is None: continue
            meta_path = os.path.join(d, 'metadata.json')
            if os.path.exists(meta_path):
                self.data_dir = d
                with open(meta_path, 'r') as f:
                    self.metadata = json.load(f)
                return

    def _extract_move_direction(self, text):
        if not text: return "UNKNOWN"
        match = re.search(r'\\boxed{([^}]+)}', text)
        if match:
            return match.group(1).split(',')[0].strip().upper()
        return "UNKNOWN"

    def _extract_level(self, image_path):
        if not image_path: return 0
        match = re.search(r'level_(\d+)', image_path)
        if match:
            return int(match.group(1))
        return 0

    def _process_all_samples(self):
        if not self.metadata:
            return []

        samples_to_process = self.metadata[:1000]
        processed = []
        all_hidden_states = []
        self.valid_indices = []

        for i, meta in enumerate(samples_to_process):
            try:
                raw_sid = meta.get('sample_id', i)
                sample_id = f"sample_{raw_sid:03d}"
                move_dir = self._extract_move_direction(meta.get('text_output_short', ''))
                correctness = move_dir != "UNKNOWN"
                level_id = self._extract_level(meta.get('image_input', ''))
                seq_len = meta.get('seq_len', 0)

                sample_tensor_dir = os.path.join(self.data_dir, 'tensors', sample_id)
                hs_path = os.path.join(sample_tensor_dir, 'hidden_states.pt')
                attn_path = os.path.join(sample_tensor_dir, 'latent_to_visual_attn.pt')

                latent_pos = meta.get('token_positions', {}).get('latent', [])
                num_latent = len(latent_pos) if latent_pos else 6

                # 1. Load Attention
                real_attn = None
                if os.path.exists(attn_path):
                    try:
                        attn_dict = torch.load(attn_path, map_location='cpu')
                        if isinstance(attn_dict, dict):
                            last_layer = max(attn_dict.keys())
                            real_attn = attn_dict[last_layer]
                        else:
                            real_attn = attn_dict
                    except Exception as e:
                        logging.debug(f"Failed to load attn for {sample_id}: {e}")

                tokens = []
                token_vectors = []
                for t in range(num_latent):
                    spatial_focus = None
                    if real_attn is not None:
                        try:
                            token_attn = real_attn[t, :].numpy()
                            n_vis = len(token_attn)
                            side = int(np.sqrt(n_vis))
                            if side * side == n_vis:
                                spatial_focus = token_attn.reshape(side, side).tolist()
                        except: pass

                    if spatial_focus is None:
                        spatial_focus = np.zeros((11, 11), dtype=np.float32).tolist()

                    focus_arr = np.asarray(spatial_focus, dtype=np.float32)
                    flat_focus = focus_arr.reshape(-1)
                    total = float(flat_focus.sum())
                    if total > 0:
                        prob = flat_focus / total
                    else:
                        prob = np.full_like(flat_focus, 1.0 / len(flat_focus))
                    uniform = np.full_like(prob, 1.0 / len(prob))
                    probe_accuracy = float(np.clip(prob.max() * len(prob), 0.0, 1.0))
                    kl_divergence = float(np.sum(prob * np.log((prob + 1e-8) / uniform)))

                    token_vectors.append(flat_focus)

                    tokens.append({
                        'token_id': f"T{t}",
                        'spatial_focus': spatial_focus,
                        'probe_accuracy': probe_accuracy,
                        'kl_divergence': kl_divergence
                    })

                # 2. Load Hidden States
                if os.path.exists(hs_path):
                    try:
                        hs_data = torch.load(hs_path, map_location='cpu')
                        if isinstance(hs_data, dict):
                            last_layer = max(hs_data.keys())
                            hs_tensor = hs_data[last_layer]
                        else:
                            hs_tensor = hs_data
                        
                        if hs_tensor.dim() == 3:
                            if latent_pos:
                                last_latent_pos = latent_pos[-1]
                                if hs_tensor.shape[1] > last_latent_pos:
                                    vec = hs_tensor[0, last_latent_pos, :].to(torch.float32).numpy()
                                else:
                                    vec = hs_tensor[0, -1, :].to(torch.float32).numpy()
                            else:
                                vec = hs_tensor[0, -1, :].to(torch.float32).numpy()
                        else:
                            vec = hs_tensor[-1, :].to(torch.float32).numpy()
                        
                        all_hidden_states.append(vec)
                        self.valid_indices.append(i)
                    except Exception as e:
                        logging.debug(f"Error loading HS for {sample_id}: {e}")

                if token_vectors:
                    token_matrix = cosine_similarity(np.asarray(token_vectors, dtype=np.float32))
                else:
                    token_matrix = np.zeros((num_latent, num_latent), dtype=np.float32)

                processed.append({
                    'sample_id': sample_id,
                    'correctness': correctness,
                    'move_direction': move_dir,
                    'level_id': level_id,
                    'seq_len': seq_len,
                    'num_latent': num_latent,
                    'umap_x': 0.0,
                    'umap_y': 0.0,
                    'tokens': tokens,
                    'attention_weights': token_matrix.tolist(),
                    'metadata': meta
                })
            except Exception as e:
                logging.warning(f"Failed processing sample {i}: {e}")

        # Cache hidden states
        if len(all_hidden_states) > 5:
            self.X_raw = np.array(all_hidden_states)
            
            # Pre-compute L2 Normalization (Standard for Cosine)
            norms = np.linalg.norm(self.X_raw, axis=1, keepdims=True)
            self.X_norm = self.X_raw / (norms + 1e-8)
            
            # Pre-compute PCA Denoising to avoid latency during sweeps
            n_comp = min(32, self.X_norm.shape[0], self.X_norm.shape[1])
            pca = PCA(n_components=n_comp, random_state=42)
            self.X_pca = pca.fit_transform(self.X_norm)

            # Initial projection with defaults
            self.recompute_umap(n_neighbors=5, min_dist=0.3, use_pca=False, processed_override=processed)
        else:
            if not HAS_UMAP:
                logging.error("UMAP library not installed. Points will remain at origin.")
            elif len(all_hidden_states) <= 5:
                logging.warning(f"Insufficient hidden states ({len(all_hidden_states)}) for UMAP projection.")

        return processed

    def recompute_umap(self, n_neighbors, min_dist, use_pca=False, processed_override=None):
        """Dynamic re-projection using pre-computed features."""
        target_processed = processed_override if processed_override is not None else self.processed_samples
        
        if self.X_raw is not None and HAS_UMAP:
            try:
                # Select pre-computed input feature set
                X_input = self.X_pca if use_pca else self.X_norm
                
                # Scientific Stabilizer: Cosine Metric
                reducer = umap.UMAP(
                    n_neighbors=int(n_neighbors), 
                    min_dist=float(min_dist), 
                    metric='cosine',
                    n_components=2, 
                    random_state=42
                )
                coords = reducer.fit_transform(X_input)
                
                for idx, coord_idx in enumerate(self.valid_indices):
                    target_processed[coord_idx]['umap_x'] = float(coords[idx, 0])
                    target_processed[coord_idx]['umap_y'] = float(coords[idx, 1])
                
                return target_processed
            except Exception as e:
                logging.error(f"Re-projection failed: {e}")
        return target_processed

    def get_data(self):
        return self.processed_samples

# Singleton instance
LOADER = RealDataLoader()
REAL_DATA = LOADER.get_data()
