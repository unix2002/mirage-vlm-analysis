#!/usr/bin/env python3
"""Precompute PCA vectors from hidden states for compact UMAP projection.

Reads all sample_*/hidden_states.pt files under data/tensors/, extracts the
last-latent-token hidden state from the last layer, stacks them into a
(996, 4096) matrix, fits PCA(32), and saves:

    data/processed/pca_vectors.npy   — 996 × 32 float32 (~127 KB)
    data/processed/pca_model.pkl     — fitted PCA model (for transform)

After running, the dashboard no longer needs hidden_states.pt files for UMAP.
"""

import json, os, pickle, sys, time
import numpy as np

DATA_DIR = 'data'
TENSORS_DIR = os.path.join(DATA_DIR, 'tensors')
METADATA_PATH = os.path.join(DATA_DIR, 'metadata.json')
OUT_DIR = os.path.join(DATA_DIR, 'processed')
PCA_VECTORS_PATH = os.path.join(OUT_DIR, 'pca_vectors.npy')
PCA_MODEL_PATH = os.path.join(OUT_DIR, 'pca_model.pkl')

try:
    import torch
except ImportError:
    print("PyTorch not available — cannot load hidden states.", file=sys.stderr)
    sys.exit(1)

try:
    from sklearn.decomposition import PCA
except ImportError:
    print("scikit-learn not available.", file=sys.stderr)
    sys.exit(1)


def load_hidden_state(sample_dir):
    hs_path = os.path.join(sample_dir, 'hidden_states.pt')
    if not os.path.exists(hs_path):
        return None
    try:
        hs_data = torch.load(hs_path, map_location='cpu')
        if isinstance(hs_data, dict):
            last_layer = max(hs_data.keys())
            hs_tensor = hs_data[last_layer]
        else:
            hs_tensor = hs_data

        if hs_tensor.dim() == 3:
            # Use last position (last latent token)
            vec = hs_tensor[0, -1, :].to(torch.float32).numpy()
        else:
            vec = hs_tensor[-1, :].to(torch.float32).numpy()
        return vec
    except Exception as e:
        print(f"  Warning: failed to load {hs_path}: {e}")
        return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load metadata for sample ordering
    if not os.path.exists(METADATA_PATH):
        print(f"metadata.json not found at {METADATA_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    print(f"Processing {len(metadata)} samples from {METADATA_PATH}...")
    t0 = time.time()

    vectors = []
    valid_count = 0

    for i, meta in enumerate(metadata):
        raw_sid = meta.get('sample_id', i)
        sample_id = f"sample_{raw_sid:03d}"
        sample_dir = os.path.join(TENSORS_DIR, sample_id)

        vec = load_hidden_state(sample_dir)
        if vec is not None:
            vectors.append(vec)
            valid_count += 1
        else:
            vectors.append(np.zeros(4096, dtype=np.float32))

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(metadata)} samples loaded ({valid_count} valid)")

    elapsed = time.time() - t0
    print(f"Loaded {valid_count}/{len(metadata)} hidden states in {elapsed:.1f}s")

    # Stack into matrix
    X = np.array(vectors, dtype=np.float32)
    print(f"Input matrix: {X.shape} ({X.nbytes / 1e6:.1f} MB)")

    # L2 normalize (matching data_loader.py behavior)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = X / (norms + 1e-8)

    # Fit PCA
    n_comp = min(32, X_norm.shape[0], X_norm.shape[1])
    print(f"Fitting PCA({n_comp})...")
    pca = PCA(n_components=n_comp, random_state=42)
    X_pca = pca.fit_transform(X_norm)
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # Save
    np.save(PCA_VECTORS_PATH, X_pca.astype(np.float32))
    with open(PCA_MODEL_PATH, 'wb') as f:
        pickle.dump(pca, f)

    size_kb = os.path.getsize(PCA_VECTORS_PATH) / 1024
    print(f"\nSaved {PCA_VECTORS_PATH} ({size_kb:.0f} KB)")
    print(f"Saved {PCA_MODEL_PATH}")
    print("Done — dashboard can now run without hidden_states.pt files.")


if __name__ == '__main__':
    main()
