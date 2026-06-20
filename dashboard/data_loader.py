
import json
import os
import torch
import numpy as np
import traceback
import warnings
import re
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
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

def _enrich_has_plan_flip(data_dir, processed):
    """Set has_plan_flip=True on samples where any ablation combo changes the plan."""
    if not data_dir or not processed:
        return
    path = os.path.join(data_dir, 'train_plans_gen.jsonl')
    if not os.path.exists(path):
        return
    try:
        has_flip = {}
        with open(path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                subsets = entry.get('subsets', {})
                if any(s.get('changed') for s in subsets.values()):
                    has_flip[str(entry.get('sample_id', ''))] = True
        for s in processed:
            key = s['sample_id'].replace('sample_', '').lstrip('0') or '0'
            s['has_plan_flip'] = has_flip.get(str(int(key) if key.isdigit() else key), False)
    except Exception as e:
        logging.warning(f"Failed to compute plan-flip flag: {e}")

class RealDataLoader:
    def __init__(self, data_dir=None):
        self.data_dir = None
        self.metadata = []
        self.processed_samples = []
        self.maze_dict = {}
        self.X_pca = None
        self.X_raw = None
        self.X_norm = None
        self.attn_cache = None

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
                logging.error("No data directory found. Dashboard will have no sample data.")
        except Exception as e:
            logging.error(f"Critical error during data initialization: {str(e)}")
            logging.debug(traceback.format_exc())

    def _load_maze_dict(self, data_dir):
        # Try to find train_direct.jsonl
        base_dir = os.path.dirname(data_dir) if data_dir.endswith('extracted') else data_dir
        jsonl_path = os.path.join(base_dir, 'vsp_spatial_planning', 'train_direct.jsonl')
        
        if not os.path.exists(jsonl_path):
            jsonl_path = '/gpfs/home1/scur0241/mirage_data/vsp_spatial_planning/train_direct.jsonl'
            
        if os.path.exists(jsonl_path):
            try:
                with open(jsonl_path, 'r') as f:
                    for line in f:
                        if not line.strip(): continue
                        item = json.loads(line)
                        if 'image_input' in item:
                            img_path = item['image_input']
                            match = re.search(r'level_(\d+)/(\d+)/', img_path)
                            if match:
                                level = int(match.group(1))
                                map_id = int(match.group(2))
                                self.maze_dict[(level, map_id)] = {
                                    'map_desc': item.get('map_desc'),
                                    'full_path': item.get('text_output')
                                }
                logging.info(f"Loaded {len(self.maze_dict)} maze descriptions from {jsonl_path}")
            except Exception as e:
                logging.warning(f"Failed to load maze dictionary: {e}")

    def _initialize_paths(self, override_dir=None):
        possible_dirs = [
            override_dir,
            '/gpfs/home1/scur0241/mirage_data', # Central root data directory
            'data/',    # Local real dataset
            '../mirage_data/extracted',      # From remco/
            '../../mirage_data/extracted',   # From remco/mirage-vlm-analysis/
            '/gpfs/home1/scur0241/mirage_data/extracted', # Absolute path fallback
            'data/reference'                 # Local fallback
        ]

        for d in possible_dirs:
            if d is None: continue
            meta_path = os.path.join(d, 'metadata.json')
            if os.path.exists(meta_path):
                self.data_dir = d
                with open(meta_path, 'r') as f:
                    self.metadata = json.load(f)
                self._load_maze_dict(d)
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

        # Load all available samples; UMAP projection is deferred until first use
        # to avoid a startup RAM spike. Override with DASHBOARD_MAX_SAMPLES if needed.
        max_samples = int(os.environ.get('DASHBOARD_MAX_SAMPLES', 1000))
        samples_to_process = self.metadata[:max_samples]
        processed = []
        all_hidden_states = []
        self.valid_indices = []
        self.has_raw_hs = False

        # Try loading pre-computed PCA cache (removes 10 GB hidden-states dependency)
        pca_cache = os.path.join('data', 'processed', 'pca_vectors.npy')
        if os.path.exists(pca_cache):
            self.X_pca = np.load(pca_cache).astype(np.float64)
            self.X_raw = None
            self.X_norm = None
            logging.info("Loaded pre-computed PCA vectors (%d × %d) — hidden states not needed.",
                         self.X_pca.shape[0], self.X_pca.shape[1])

        # Try loading pre-computed attention cache (per-layer heatmap slider)
        attn_cache_path = os.path.join('data', 'processed', 'attn_full.npz')
        if os.path.exists(attn_cache_path):
            self.attn_cache = dict(np.load(attn_cache_path, allow_pickle=False))
            logging.info("Loaded attention cache (%d layers) — per-sample tensors not needed.",
                         len(self.attn_cache))

        for i, meta in enumerate(samples_to_process):
            try:
                raw_sid = meta.get('sample_id', i)
                sample_id = f"sample_{raw_sid:03d}"
                move_dir = self._extract_move_direction(meta.get('text_output_short', ''))
                correctness = move_dir != "UNKNOWN"
                
                image_input = meta.get('image_input', '')
                level_id = self._extract_level(image_input)
                
                # Extract map_id
                map_id = 0
                map_match = re.search(r'level_\d+/(\d+)/', image_input)
                if map_match:
                    map_id = int(map_match.group(1))

                seq_len = meta.get('seq_len', 0)

                sample_tensor_dir = os.path.join(self.data_dir, 'tensors', sample_id)
                hs_path = os.path.join(sample_tensor_dir, 'hidden_states.pt')
                attn_path = os.path.join(sample_tensor_dir, 'latent_to_visual_attn.pt')

                latent_pos = meta.get('token_positions', {}).get('latent', [])
                num_latent = len(latent_pos) if latent_pos else 6

                # Fetch maze data
                maze_info = self.maze_dict.get((level_id, map_id), {})
                map_desc = maze_info.get('map_desc')
                full_path = maze_info.get('full_path')

                # 1. Load Attention
                real_attn = None
                last_layer = None
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

                # Fallback to pre-computed attention cache (last layer for initial view)
                if real_attn is None and self.attn_cache is not None:
                    try:
                        layer_keys = sorted(int(k) for k in self.attn_cache.keys() if k != '_nvis')
                        if layer_keys:
                            last_layer = layer_keys[-1]
                            arr = self.attn_cache[str(last_layer)]
                            if i < arr.shape[0]:
                                n_vis = int(self.attn_cache.get('_nvis', [arr.shape[2]] * arr.shape[0])[i])
                                real_attn = torch.from_numpy(arr[i, :, :n_vis])
                    except Exception as e:
                        logging.debug(f"Failed attn cache fallback for {sample_id}: {e}")

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
                        except Exception:
                            pass

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

                # 2. Load Hidden States (skip if PCA cache is available)
                if self.X_pca is None and os.path.exists(hs_path):
                    self.has_raw_hs = True
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
                    'map_id': map_id,
                    'seq_len': seq_len,
                    'num_latent': num_latent,
                    'umap_x': 0.0,
                    'umap_y': 0.0,
                    'tokens': tokens,
                    'attention_weights': token_matrix.tolist(),
                    'metadata': meta,
                    'map_desc': map_desc,
                    'full_path': full_path
                })
            except Exception as e:
                logging.warning(f"Failed processing sample {i}: {e}")

        # Compute per-sample plan-change flag from train_plans_gen.jsonl
        _enrich_has_plan_flip(self.data_dir, processed)

        # PCA cache path — skip hidden-state loading entirely
        if self.X_pca is not None:
            if len(processed) < self.X_pca.shape[0]:
                self.X_pca = self.X_pca[:len(processed)]
            self.valid_indices = list(range(len(processed)))
        # Legacy path — build feature matrices from hidden states
        elif len(all_hidden_states) > 5:
            self.X_raw = np.array(all_hidden_states)

            # Pre-compute L2 Normalization
            norms = np.linalg.norm(self.X_raw, axis=1, keepdims=True)
            self.X_norm = self.X_raw / (norms + 1e-8)

            # Pre-compute PCA Denoising
            n_comp = min(32, self.X_norm.shape[0], self.X_norm.shape[1])
            pca = PCA(n_components=n_comp, random_state=42)
            self.X_pca = pca.fit_transform(self.X_norm)

        # Eagerly compute default UMAP at startup from PCA input
        if self.X_pca is not None and HAS_UMAP and self.valid_indices:
            reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='cosine',
                                n_components=2, random_state=42)
            coords = reducer.fit_transform(self.X_pca)
            for idx, coord_idx in enumerate(self.valid_indices):
                if coord_idx >= len(processed):
                    continue
                target = processed[coord_idx]
                target['umap_x'] = float(coords[idx, 0])
                target['umap_y'] = float(coords[idx, 1])
            logging.info("Default UMAP projection ready (PCA input).")

        if self.X_pca is not None:
            logging.info("UMAP ready — PCA cache loaded, hidden states not needed.")
        elif not HAS_UMAP:
            logging.error("UMAP library not installed. Points will remain at origin.")
        elif len(all_hidden_states) <= 5:
            logging.warning(f"Insufficient hidden states ({len(all_hidden_states)}) for UMAP projection.")

        return processed

    def recompute_umap(self, n_neighbors, min_dist, use_pca=False, processed_override=None):
        """Dynamic re-projection using pre-computed features.

        Falls back to PCA input when raw hidden states are unavailable
        (i.e., when using the pre-computed PCA cache).
        """
        target_processed = processed_override if processed_override is not None else self.processed_samples

        if not target_processed:
            logging.warning("No processed samples available for UMAP re-projection")
            return target_processed

        if self.X_pca is not None and HAS_UMAP:
            try:
                # Use PCA input as the default; raw 4096-dim only when available
                if use_pca or self.X_norm is None:
                    X_input = self.X_pca
                else:
                    X_input = self.X_norm

                # Scientific Stabilizer: Cosine Metric
                reducer = umap.UMAP(
                    n_neighbors=int(n_neighbors),
                    min_dist=float(min_dist),
                    metric='cosine',
                    n_components=2,
                    random_state=42
                )
                coords = reducer.fit_transform(X_input)

                # Vectorized uncertainty calculation (Average Local Error)
                # 1. High-D distances (Cosine) normalized by max
                dist_high = 1.0 - cosine_similarity(X_input)
                max_high = np.max(dist_high)
                if max_high > 0: dist_high /= max_high

                # 2. 2D distances (Euclidean) normalized by max
                dist_2d = euclidean_distances(coords)
                max_2d = np.max(dist_2d)
                if max_2d > 0: dist_2d /= max_2d

                # 3. Average Local Error per point
                uncertainty = np.mean(np.abs(dist_high - dist_2d), axis=1)

                for idx, coord_idx in enumerate(self.valid_indices):
                    target_processed[coord_idx]['umap_x'] = float(coords[idx, 0])
                    target_processed[coord_idx]['umap_y'] = float(coords[idx, 1])
                    target_processed[coord_idx]['umap_uncertainty'] = float(uncertainty[idx])

                return target_processed
            except Exception as e:
                logging.error(f"Re-projection failed: {e}")
        return target_processed

    def get_data(self):
        return self.processed_samples

# Singleton instance
LOADER = RealDataLoader()
REAL_DATA = LOADER.get_data()


def get_layer_heatmap(sample_id, token_idx, layer):
    """Load attention for (sample, token_idx, layer) → 11×11 spatial focus grid.
    
    Uses per-sample tensor files when available; falls back to the pre-computed
    attention cache (attn_full.npz) for fresh clones without tensor data.
    """
    sid = sample_id if isinstance(sample_id, str) else f"sample_{sample_id:03d}"
    data_dir = getattr(LOADER, 'data_dir', 'data')
    attn_path = os.path.join(data_dir, 'tensors', sid, 'latent_to_visual_attn.pt')

    token_attn = None

    # Primary: per-sample file
    if os.path.exists(attn_path):
        attn = torch.load(attn_path, map_location='cpu')
        if isinstance(attn, dict):
            keys = sorted(attn.keys())
            layer_key = layer if layer in attn else (keys[-1] if keys else 0)
            attn = attn[layer_key]
        token_attn = attn[token_idx, :].numpy()

    # Fallback: pre-computed attention cache
    if token_attn is None and LOADER.attn_cache is not None:
        try:
            layer_key = str(layer)
            if layer_key in LOADER.attn_cache:
                arr = LOADER.attn_cache[layer_key]
                sample_idx = int(sid.replace('sample_', '').lstrip('0') or '0')
                if sample_idx < arr.shape[0]:
                    n_vis = int(LOADER.attn_cache.get('_nvis', [arr.shape[2]] * arr.shape[0])[sample_idx])
                    token_attn = arr[sample_idx, token_idx, :n_vis]
        except Exception as e:
            logging.debug(f"Attn cache fallback failed for {sid} layer {layer}: {e}")

    if token_attn is None:
        return None

    n_vis = len(token_attn)
    side = int(np.sqrt(n_vis))
    if side * side == n_vis:
        return token_attn.reshape(side, side).tolist()
    return None
