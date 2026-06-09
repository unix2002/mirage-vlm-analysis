# Mirage VLM — Latent Reasoning Analysis

Interactive visual analytics for inspecting latent reasoning in vision-language models.  
4-week project, 4 roles (A/B/C/D).

**Compute:** Snellius (`scur0259@snellius.surf.nl`)  
**Data root:** `/home/scur0259/mirage/data/extracted/` (50 samples, 7.9 GB)  
**Model:** Qwen2.5-VL-7B-Instruct + latent tokens (= Mirage)  
**Base codebase:** [UMass-Embodied-AGI/Mirage](https://github.com/UMass-Embodied-AGI/Mirage)

---

## 🚀 Quick Start: Running the Dashboard

To run the interactive Dash dashboard with simulated data (50 samples):

1. **Ensure dependencies are installed:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the dashboard:**
   ```bash
   python3 run_dashboard.py
   ```

3. **Access the UI:**
   Open your browser and navigate to `http://127.0.0.1:8050`.

---

## Project Overview

The Mirage model (Yang et al., 2025) extends Qwen2.5-VL with **latent visual tokens** —
compressed image features scattered between the user's question and the assistant's answer.
These tokens carry visual reasoning information but are opaque: they cannot be read, inspected,
or debugged directly.

This project builds an interactive Dash dashboard that makes latent reasoning tangible across
four research questions:

| RQ | Question | Method | Output |
|----|----------|--------|--------|
| **RQ1** | Do latent tokens focus on key visual regions? | Self-attention latent → visual patches, heatmap overlay | Per-token 2D attention maps (28 layers) |
| **RQ2** | Do latent tokens contain enough information to predict the answer? | Linear probing classifiers on token vectors | Probe accuracy per token per layer |
| **RQ3** | Does the final answer causally depend on latent tokens? | Token ablation → KL divergence | Dependency score per token |
| **RQ4** | How can the reasoning path be visualized? | Integrated Dash dashboard | Multi-level interactive view |

---

## 🏗️ Repository Structure

The codebase is organized into a modular package structure to ensure research reproducibility and scalability:

```
.
├── mirage_vlm/                           ← Core Analysis Package
│   ├── extraction/                       ← Logic for VLM tensor extraction and hooks
│   ├── inference/                        ← Main entry points for model inference
│   ├── tasks/                            ← Task definitions (e.g., Maze/VSP)
│   ├── training/                         ← Evaluation and probing scripts
│   └── utils/                            ← Shared utility functions
├── dashboard/                            ← Interactive Visualization App
│   ├── components/                       ← UI Levels 1, 2, 3
│   ├── app.py                            ← Main Dash entry point
│   ├── callbacks.py                      ← Interactive logic & RQ mapping
│   ├── layout.py                         ← Full-screen responsive layout
│   └── mock_data.py                      ← Simulated data generator
├── data/                                 ← Data Management
│   ├── reference/                        ← Raw reference tensors
│   └── processed/                        ← Processed analysis results
├── tests/                                ← "Bulletproof" Test Suite
├── run_dashboard.py                      ← Unified App Entry Point
├── requirements.txt                      ← Project Dependencies
├── SCHEDULE.md                           ← 4-week schedule & role assignments
└── README.md
```

---

## Dataset: VSP (Visual Spatial Planning)

The Mirage model's latent token mechanism **requires paired output images** — something GQA
does not provide. We use **VSP** — maze-based spatial reasoning with step-by-step board-state
images. This is the dataset Mirage was designed and trained for.

| Split | Samples | Location (Snellius) |
|-------|---------|---------------------|
| Full training | 1,000 | `/home/scur0259/mirage/data/vsp_spatial_planning/train_direct.jsonl` |
| Preprocessed | 1,000 | `.../train_direct_with_oi.jsonl` (fixed with `<output_image>` tags) |
| Extracted (ours) | 50 | `/home/scur0259/mirage/data/extracted/` |

### Data Preparation Fix

The original `train_direct.jsonl` has bare `\boxed{DOWN}` without `<output_image>` or
`<think>` tags. Mirage's `place_output_image()` utility expects `<think>...</think>` wrapping
to correctly count image placeholders. Missing tags cause an `IndexError` in the processor's
`image_grid_thw` indexing (3 image_pads for 2 images → crash).

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

## 📊 Dashboard Views (Research Analysis)

The dashboard is structured into three interactive levels mapped to the project RQs:

- **Level 1: Sample Landscape**: UMAP clustering of 50 samples by correctness and move direction.
- **Level 2: Reasoning Path**: 
  - **Image-Anchored View**: Latent token "glyphs" overlaid on the maze. (Glyph Size = RQ3 Causal Dependence; Color = RQ2 Information Content).
  - **Sequential Flow**: Attention weight matrix between tokens.
- **Level 3: Token Detail**: 
  - **RQ1**: High-res spatial focus heatmap.
  - **RQ2**: Probe accuracy bar charts.
  - **RQ3**: Dependency decay curves and a "What-If" ablation sandbox.

---

## 🧪 Testing

The framework includes 18+ rigorous tests to ensure stability:
```bash
PYTHONPATH=. python3 -m pytest tests/dashboard
```
Tests cover **ID Integrity**, **Exhaustive Data Robustness**, and **Visual Logic Verification**.

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
    └── sample_053/             ← (indices 46–49 skipped: square images incompatible with processor)
```

### Dataset Summary

| Metric | Min | Max | Avg |
|--------|-----|-----|-----|
| Sequence length | 374 | 537 | 475 |
| Latent tokens | 6 | 6 | 6.0 |
| Visual patches | 49 | 196 | 145 |
| **Total size** | | | **7.9 GB** (~158 MB/sample) |

### Size Breakdown

| Tensor type | Total size | % |
|-------------|-----------|----|
| `logits.pt` | 7,206 MB | 91% |
| `hidden_states.pt` | 511 MB | 6% |
| `embeddings.pt` | 170 MB | 2% |
| `latent_to_visual_attn.pt` | 3 MB | <1% |

---

## Schedule & Status

### Week 1 Status

| Task | Status |
|------|--------|
| Environment setup (Mirage + venv on A100) | ✅ Done |
| `MirageAnalyzer` forward hooks (embeddings, attention, hidden states, logits) | ✅ Done |
| Smoke tests (plain image + latent-token forward pass) | ✅ Done |
| Data preparation (1,000 VSP samples fixed) | ✅ Done |
| Extraction at scale (50 samples, 7.9 GB) | ✅ Done |
| GitHub repo with code + reference data | ✅ Done |

### Schedule (Weeks 2–4)

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
