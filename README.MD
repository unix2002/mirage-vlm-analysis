# Mirage VLM — Latent Reasoning Analysis

Interactive visual analytics for inspecting latent reasoning in vision-language models.  
4-week project, 4 roles (A/B/C/D).

**Compute:** Snellius (`scur0259@snellius.surf.nl`)  
**Data root:** `/home/scur0259/mirage/data/extracted/` (50 samples, 7.9 GB)  
**Model:** Qwen2.5-VL-7B-Instruct + latent tokens (= Mirage)  
**Paper:** [Machine Mental Imagery (arXiv 2506.17218)](https://arxiv.org/abs/2506.17218)

---

## Project Overview

Vision-language models increasingly solve complex VQA through **latent visual reasoning** —
generating continuous "visual thoughts" rather than compressing everything into discrete text.
These latent tokens are opaque: they cannot be read, inspected, or debugged.

This project builds an interactive Dash dashboard that makes latent reasoning tangible across
four research questions:

| RQ | Question | Method | Output |
|----|----------|--------|--------|
| **RQ1** | Do latent tokens focus on key visual regions? | Self-attention latent → visual patches, heatmap overlay | Per-token 2D attention maps (28 layers) |
| **RQ2** | Do latent tokens contain enough information to predict the answer? | Linear probing classifiers on token vectors | Probe accuracy per token per layer |
| **RQ3** | Does the final answer causally depend on latent tokens? | Token ablation → KL divergence | Dependency score per token |
| **RQ4** | How can the reasoning path be visualized? | Integrated Dash dashboard | Multi-level interactive view |

---

## Repository Structure

```
.
├── README.md
├── SCHEDULE.md                           ← 4-week schedule & role assignments
├── data_reference/                       ← One sample for offline dev (no logits)
│   ├── metadata.json
│   ├── latent_to_visual_attn.pt
│   ├── hidden_states.pt
│   └── embeddings.pt
├── src/
│   ├── hooks.py                          ← MirageAnalyzer class (forward hooks)
│   ├── extract.py                        ← Batch extraction pipeline (samples → .pt + JSON)
│   ├── main.py                           ← Training entry point (original Mirage)
│   ├── task.py                           ← Dataset preprocessing (original)
│   ├── trainer.py                        ← Training loop wrapper (original)
│   ├── utils.py                          ← Token manipulation, batch processing (original)
│   └── test.py                           ← Inference / debug utilities (original)
└── report/
    └── session_report.tex                ← Week 1 session log (LaTeX)
```

**Original codebase:** [UMass-Embodied-AGI/Mirage](https://github.com/UMass-Embodied-AGI/Mirage)  
**Our additions:** `hooks.py`, `extract.py`, `data_reference/`, this README, `SCHEDULE.md`.

---

## Dataset: VSP (Visual Spatial Planning)

The Mirage model's latent token mechanism **requires paired output images** that GQA does not
provide. We use **VSP** — maze-based spatial reasoning with step-by-step board-state images.
This is the dataset Mirage was designed and trained for.

| Split | Samples | Location (Snellius) |
|-------|---------|---------------------|
| Full training | 1,000 | `/home/scur0259/mirage/data/vsp_spatial_planning/train_direct.jsonl` |
| Preprocessed | 1,000 | `.../train_direct_with_oi.jsonl` (fixed with `<output_image>` tags) |
| Extracted (ours) | 50 | `/home/scur0259/mirage/data/extracted/` |

### Data Format

One JSON object per line:

```json
{
  "text_input":   "... maze description and task prompt ...",
  "image_input":  "./data/vsp_spatial_planning/img/.../map_*_step_0.png",
  "image_output": "./data/vsp_spatial_planning/img/.../map_reasoning_path.png",
  "text_output":  "<output_image>\\boxed{DOWN}",
  "map_id":       "1454",
  "map_desc":     "[[0,0,0,0,0],[-1,0,-1,-1,-1],...]"
}
```

### Data Preparation Fix

The original `train_direct.jsonl` has bare `\boxed{DOWN}` without `<output_image>` or
`<think>` tags. Mirage's `place_output_image()` utility expects `<think>...</think>` wrapping.
Missing tags cause the processor to miscount image placeholders (3 image_pads for 2 images →
`IndexError` in `image_grid_thw`).

**Fix:** `<think></think><output_image>` prepended before `\boxed{...}` in all 1,000 samples.
Output: `train_direct_with_oi.jsonl`.

---

## Token Sequence Structure

A typical ~475-token sequence contains (in order):

1. **System prompt** + maze description (text tokens)
2. **`<|vision_start|>`** + `<|image_pad|>` × N_visual + **`<|vision_end|>`** — input image patches
3. **Task prompt** (text tokens)
4. **`<|im_start|>assistant`**
5. **`<|latent_start|>`** + `<|latent_pad|>` × 6 + **`<|latent_end|>`** — **latent reasoning tokens**
6. **`<think></think>`** + answer text + **`<|im_end|>`**

The 6 latent tokens are always consecutive and at the same relative position. This
determinism is critical for ablation (RQ3): the latent region can be reliably targeted.

---

## Extracted Data (50 Samples)

### Location

```
/home/scur0259/mirage/data/extracted/
├── metadata.json              ← 50-sample index (133 KB)
└── tensors/
    ├── sample_000/             ← 4 .pt files, ~158 MB
    ├── sample_001/
    ├── ...
    └── sample_053/             ← (46–49 skipped: square images)
```

**Access for teammates:**
```bash
cp -r /home/scur0259/mirage/data/extracted ~/mirage_data/
```

### Dataset Summary

| Metric | Min | Max | Avg |
|--------|-----|-----|-----|
| Sequence length | 374 | 537 | 475 |
| Latent tokens | 6 | 6 | 6.0 |
| Visual patches | 49 | 196 | 145 |
| **Total** | | | **7.9 GB** (~158 MB/sample) |

### Size Breakdown

| Tensor type | Total size | % |
|-------------|-----------|----|
| `logits.pt` | 7,206 MB | 91% |
| `hidden_states.pt` | 511 MB | 6% |
| `embeddings.pt` | 170 MB | 2% |
| `latent_to_visual_attn.pt` | 3 MB | <1% |

The logits dominate due to the 151,668-token vocabulary. **Week 2 plan:** top-*k* compression
(top-100 indices + values per position, ~1 MB/sample instead of ~144 MB).

### Per-Sample Files

| File | Shape | dtype | Size | Use |
|------|-------|-------|------|-----|
| `embeddings.pt` | `[1, seq_len, 3584]` | float16 | ~3.4 MB | RQ2 alt |
| `hidden_states.pt` | 3 layers × `[1, seq_len, 3584]` | float16 | ~10 MB | RQ2 |
| `latent_to_visual_attn.pt` | 28 layers × `[6, N_visual]` | float16 | ~60 KB | RQ1 |
| `logits.pt` | `[1, ≤200, 151668]` | float16 | ~144 MB | RQ3 |

---

## Data Handoff Specification

| Recipient | File(s) | Contents |
|-----------|---------|----------|
| **All** | `metadata.json` | 50-sample index: `sample_id`, `seq_len`, `token_positions` (latent & visual indices), `num_latent`, `num_visual`, `tensor_dir` |
| **B (RQ1)** | `latent_to_visual_attn.pt` | Dict `{layer_idx: [6, N_visual]}` for all 28 layers, float16. N_visual ∈ {49, 81, 121, 196} maps to square grids (7×7 through 14×14). |
| **C (RQ2)** | `hidden_states.pt` | Dict `{25, 26, 27: [1, seq_len, 3584]}`, float16. Latent token indices in metadata. |
| **C (RQ2)** | `embeddings.pt` | `[1, seq_len, 3584]`, float16, post-scatter. |
| **A (RQ3) / D** | `logits.pt` | `[1, ≤200, 151668]`, float16, answer-region vocabulary slice. |

---

## Week 1 Status

| Task | Status |
|------|--------|
| Environment setup (Mirage + venv on A100) | ✅ Done |
| `MirageAnalyzer` forward hooks (embeddings, attention, hidden states, logits) | ✅ Done |
| Smoke tests (plain image + latent-token forward pass) | ✅ Done |
| Data preparation (1,000 VSP samples fixed) | ✅ Done |
| Extraction at scale (50 samples, 7.9 GB) | ✅ Done |
| GitHub repo with code + reference data | ✅ Done |
| **Friday milestone:** ≥10 samples with all tensor types | ✅ Met (50 samples) |

## Schedule (Weeks 2–4)

| Week | Role | Tasks |
|------|------|-------|
| **Week 2** | A (RQ3) | Zero-out latent token ablation + KL divergence on all 50 samples |
| | B (RQ1) | Attention heatmaps from `latent_to_visual_attn.pt` |
| | C (RQ2) | Linear probe training on `hidden_states.pt` |
| **Week 3** | A (RQ3) | Systematic ablation: zero-out, shuffle, noise injection |
| | B (RQ1) | Aggregate attention patterns across 50 samples |
| | C (RQ2) | Probe evaluation, per-layer accuracy comparison |
| | D (RQ4) | Dash skeleton with real data, single-sample view |
| **Week 4** | All | Integration: all RQs into Dash dashboard |
| | D (RQ4) | Multi-sample comparison, interactive controls |
| | All | Final report + presentation |

---

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
| GPU | 1× A100-SXM4-40GB (~16.6 GB VRAM used) |
| HF cache | `/scratch-shared/scur0259/hf_cache` (shared scratch, 14 GB) |
| Source code | `/home/scur0259/mirage/src/` |
| Extracted data | `/home/scur0259/mirage/data/extracted/` |

---

## Getting the Data

**On Snellius** (readable by all):
```bash
cp -r /home/scur0259/mirage/data/extracted ~/mirage_data/
```

**From your laptop** (reference files only — no logits due to size):
```bash
scp scur0259@snellius.surf.nl:/home/scur0259/mirage/data/extracted/metadata.json .
scp -r scur0259@snellius.surf.nl:/home/scur0259/mirage/data/extracted/tensors/sample_000/latent_to_visual_attn.pt .
```

**Quick start (Python):**
```python
import torch, numpy as np

# Load attention submatrix for one layer
attn = torch.load("latent_to_visual_attn.pt", map_location="cpu", weights_only=True)
layer_14 = attn[14]                        # [6, N_visual] — 6 latent tokens × N visual patches
grid = int(np.sqrt(layer_14.shape[1]))     # e.g. 14 for 196 patches
heatmap = layer_14[0].numpy().reshape(grid, grid)  # first latent token's attention

# Overlay on input image with matplotlib
import matplotlib.pyplot as plt
plt.imshow(heatmap, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()
```

---

## Citation

```bibtex
@article{yang2025machine,
  title={Machine Mental Imagery: Empower Multimodal Reasoning with Latent Visual Tokens},
  author={Zeyuan Yang and Xueyang Yu and Delin Chen and Maohao Shen and Chuang Gan},
  year={2025},
  eprint={2506.17218},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2506.17218},
}
```
