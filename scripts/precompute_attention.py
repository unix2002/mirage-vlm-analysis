#!/usr/bin/env python3
"""Build a merged attention cache for the per-layer heatmap slider.

Reads all 996 sample_XXX/latent_to_visual_attn.pt files, stacks them into
a single file keyed by (layer, sample_idx) → tensor[6, N_vis]:

    data/processed/attn_full.npz  — {str(layer): array[996, 6, 196]} float16

File is ~40 MB.  Used by data_loader.py as a fallback when the
per-sample tensor directory is unavailable (e.g., fresh clone without the
full 16 GB data).
"""

import json, os, sys, time
import numpy as np

DATA_DIR = 'data'
TENSORS_DIR = os.path.join(DATA_DIR, 'tensors')
METADATA_PATH = os.path.join(DATA_DIR, 'metadata.json')
OUT_PATH = os.path.join(DATA_DIR, 'processed', 'attn_full.npz')

try:
    import torch
except ImportError:
    print("PyTorch required.", file=sys.stderr)
    sys.exit(1)


def load_sample_attention(sample_dir):
    path = os.path.join(sample_dir, 'latent_to_visual_attn.pt')
    if not os.path.exists(path):
        return None
    try:
        data = torch.load(path, map_location='cpu')
        if isinstance(data, dict):
            out = {}
            for layer, tensor in data.items():
                t = tensor.numpy() if hasattr(tensor, 'numpy') else np.asarray(tensor)
                out[int(layer)] = t.astype(np.float16)
            return out
        return None
    except Exception as e:
        print(f"  Warning: {path}: {e}")
        return None


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    if not os.path.exists(METADATA_PATH):
        print(f"metadata.json not found at {METADATA_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    n_samples = len(metadata)
    print(f"Building attention cache for {n_samples} samples...")
    t0 = time.time()

    # layer → {sample_idx → array[6, N_vis]}
    cache = {}
    layers_found = set()
    valid = 0

    for i, meta in enumerate(metadata):
        raw_sid = meta.get('sample_id', i)
        sample_id = f"sample_{raw_sid:03d}"
        sample_dir = os.path.join(TENSORS_DIR, sample_id)
        sample_attn = load_sample_attention(sample_dir)

        if sample_attn is None:
            continue

        layers_found.update(sample_attn.keys())
        for layer, arr in sample_attn.items():
            cache.setdefault(layer, {})[i] = arr

        valid += 1
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_samples} samples ({valid} valid, {len(layers_found)} layers)")

    elapsed = time.time() - t0
    n_layers = len(layers_found)
    print(f"Done in {elapsed:.1f}s — {valid} samples, {n_layers} layers "
          f"({min(layers_found)}–{max(layers_found)})")

    # Find max N_vis for uniform padding
    max_vis = max(arr.shape[1] for layer in cache.values() for arr in layer.values())
    print(f"Max visual patches: {max_vis}")

    # Convert to uniform numpy arrays per layer
    packed = {}
    for layer in range(n_layers):
        samples = cache.get(layer, {})
        arr = np.zeros((n_samples, 6, max_vis), dtype=np.float16)
        for idx in sorted(samples.keys()):
            a = samples[idx]
            arr[idx, :, :a.shape[1]] = a
        packed[str(layer)] = arr

    # Store per-sample N_vis for correct reshaping on load
    nvis = np.zeros(n_samples, dtype=np.int16)
    first_layer = cache.get(0, {})
    for idx in sorted(first_layer.keys()):
        nvis[idx] = first_layer[idx].shape[1]
    packed['_nvis'] = nvis

    np.savez_compressed(OUT_PATH, **packed)
    size_mb = os.path.getsize(OUT_PATH) / (1024 * 1024)
    print(f"Saved {OUT_PATH} ({size_mb:.1f} MB)")

    print("Ready — upload to GitHub Release or Hugging Face.")


if __name__ == '__main__':
    main()
