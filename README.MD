# Mirage VLM — Latent Reasoning Analysis

Interactive visual analytics for inspecting latent reasoning in vision-language models.  
4-week project, 4 roles (A/B/C/D). See `report/session_report.tex` for the Week 1 session log.

**Compute:** Snellius (`scur0259@snellius.surf.nl`)  
**Data root:** `/home/scur0259/mirage/data/extracted/`  
**Model:** Qwen2.5-VL-7B-Instruct + latent tokens (= Mirage)

---

## Project Overview

Vision-language models increasingly solve complex VQA through **latent visual reasoning** —
generating continuous "visual thoughts" rather than compressing everything into discrete text.
These latent tokens are opaque: they cannot be read, inspected, or debugged.

This project builds an interactive Dash dashboard that makes latent reasoning tangible across
three research questions:

| RQ | Question | Method | Output |
|----|----------|--------|--------|
| RQ1 | Do latent tokens focus on key visual regions? | Cross-attention → heatmap overlay | Per-token 2D attention maps |
| RQ2 | Do latent tokens contain enough information to predict the answer? | Linear probing classifiers | Probe accuracy per token |
| RQ3 | Does the final answer causally depend on latent tokens? | Token ablation → KL divergence | Dependency score per token |
| RQ4 | How can the reasoning path be visualized? | Integrated Dash dashboard | Multi-level interactive view |

---

## Repository Structure
```
.
├── README.md                         ← You are here
├── src/
│   ├── hooks.py                      ← MirageAnalyzer class (forward hooks)
│   ├── extract.py                    ← Batch extraction pipeline (samples → .pt + JSON)
│   ├── main.py                       ← Mirage training entry point
│   ├── task.py                       ← Dataset preprocessing
│   ├── trainer.py                    ← Training loop wrapper
│   ├── utils.py                      ← Token manipulation, batch processing
│   └── test.py                       ← Debug utilities
├── data_reference/                   ← One sample for validation (no large logits)
│   ├── metadata.json                 ← Sample index with token positions
│   ├── latent_to_visual_attn.pt      ← 28-layer attention submatrices
│   ├── hidden_states.pt              ← Layers 25–27 hidden states
│   └── embeddings.pt                 ← Post-scatter embeddings
└── report/
    └── session_report.tex            ← Week 1 session log (LaTeX)
```

---

## Dataset: VSP

The project plan originally specified GQA, but Mirage's latent token mechanism requires
**paired output images** that GQA does not provide. We use **VSP (Visual Spatial Planning)** —
maze-based spatial reasoning with step-by-step board-state images. This is the dataset Mirage
was trained on. The analytical framework (hooks → RQ1/RQ2/RQ3 → Dash) is identical regardless
of dataset.

| Split | Samples | Location |
|-------|---------|----------|
| Examples | 9 | `data/examples/sample.jsonl` |
| Full training | 1,000 | `data/vsp_spatial_planning/train_direct.jsonl` |
| Test | tarball | `data/vsp_spatial_planning/` |

---

## Token Sequence Structure

A typical 771-token sequence contains (in order):

1. Text tokens — system prompt, maze description
2. `<|vision_start|>` + `<|image_pad|>` × N_visual + `<|vision_end|>` — input image patches
3. Text tokens — question, action plan
4. `<|im_start|>assistant`
5. `<|latent_start|>` + `<|latent_pad|>` × latent_size + `<|latent_end|>` — **latent reasoning**
6. Answer text tokens — `` + `\boxed{...}`

---

## Extracted Data Format

Full data lives at `/home/scur0259/mirage/data/extracted/tensors/sample_NNN/`.  
Load with: `torch.load(path, map_location='cpu', weights_only=True)`.

### `metadata.json` — Everyone

```json
{
  "sample_id": 0,
  "seq_len": 771,
  "token_positions": {
    "latent": [587, 588, 589, 590, 591, 592],
    "visual": [29, 30, 31, ...]
  },
  "num_latent": 6,
  "num_visual": 196,
  "tensor_dir": "tensors/sample_000",
  "image_input": "./data/examples/images/...",
  "image_output": "./data/examples/images/..."
}
```

### `latent_to_visual_attn.pt` → RQ1 (Role B)

```
Dict[int, Tensor]: {0: [N_latent, N_visual], ..., 27: [N_latent, N_visual]}  float16
```

Attention from latent tokens (rows) to visual patches (cols), averaged over 28 heads.  
`N_visual` maps to a square grid: 49 → 7×7, 81 → 9×9, 121 → 11×11, 196 → 14×14.

### `hidden_states.pt` → RQ2 (Role C)

```
Dict[str, Tensor]: {
    'layer_25': [1, seq_len, 3584],
    'layer_26': [1, seq_len, 3584],
    'layer_27': [1, seq_len, 3584]
}  float16
```

Final 3 decoder layers. Latent token indices are in `metadata.json → token_positions.latent`.

### `embeddings.pt` → RQ2 alternative (Role C)

```
Tensor[1, seq_len, 3584]  float16
```

Post-scatter embeddings — after latent token values replace the `<|latent_pad|>` placeholders.

### `logits.pt` → RQ3 (Role A), answer display (Role D)

```
Tensor[1, ≤200, 151668]  float16
```

Answer-region vocabulary distributions (region after the last latent token).  
**Size:** 160–260 MB per sample. Week 2 plan: add `logits_topk.pt` with top-100 indices + values
per position (~1 MB/sample).


**Getting started:**
```python
import torch, numpy as np
attn = torch.load("data_reference/latent_to_visual_attn.pt", map_location="cpu", weights_only=True)
layer_14 = attn[14]                        # [6, 196]
grid = int(np.sqrt(layer_14.shape[1]))     # 14
heatmap = layer_14[0].numpy().reshape(grid, grid)
# Overlay on image via matplotlib
```

## Environment

```bash
ssh scur0259@snellius.surf.nl
cd /home/scur0259/mirage
source ~/mirage_setup.sh
```

| Component | Detail |
|-----------|--------|
| Python | 3.11.3 |
| PyTorch | 2.3.0 + CUDA 12.1 |
| GPU | 1× A100-SXM4-40GB (∼16.6 GB VRAM used) |
| HF cache | `/scratch-shared/scur0259/hf_cache` |
| Source code | `/home/scur0259/mirage/src/` |
| Extracted data | `/home/scur0259/mirage/data/extracted/` |

---

## Getting the Data

**On Snellius** (everyone can read from `/home/scur0259/`):
```bash
cp -r /home/scur0259/mirage/data/extracted ~/mirage_data/
```

**From your laptop** (small reference files only):
```bash
scp scur0259@snellius.surf.nl:/home/scur0259/mirage/data/extracted/metadata.json .
scp -r scur0259@snellius.surf.nl:/home/scur0259/mirage/data/extracted/tensors/sample_000/latent_to_visual_attn.pt .
```
