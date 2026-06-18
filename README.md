# Mirage VLM — Latent Reasoning Analysis

Interactive visual analytics for inspecting latent reasoning in vision-language models.  
4-week project, 4 roles (A/B/C/D).

**Compute:** Snellius (`scur0259@snellius.surf.nl`)  
**Data root:** `/home/scur0259/mirage/data/extracted/` (996 samples, 16 GB)  
**Model:** Qwen2.5-VL-7B-Instruct + latent tokens (= Mirage)  
**Base codebase:** [UMass-Embodied-AGI/Mirage](https://github.com/UMass-Embodied-AGI/Mirage)

---

## 🚀 Quick Start: Running the Dashboard

### Prerequisites

```bash
pip install -r requirements.txt
```

### Data Setup

The dashboard needs two groups of files in `data/`:

**From git** (cloned automatically):
```
data/ablation_results.json
data/train_direct.jsonl
```

**From Google Drive** (shared privately — ask the team for the link) and place in `data/`:

| File | Size | Notes |
|------|------|-------|
| `data/metadata.json` | 2.7 MB | Sample index (996 entries) |
| `data/tensors/` | 16 GB total | Per-sample `.pt` files |
| `data/vsp_spatial_planning.tar.gz` | 42 MB | Maze images — **do not unpack**, read directly by the dashboard |

The final `data/` directory should look like:
```
data/
├── ablation_results.json          (git)
├── train_direct.jsonl             (git)
├── metadata.json                  (Drive)
├── tensors/                       (Drive — 996 sample dirs)
│   ├── sample_000/
│   │   ├── hidden_states.pt
│   │   └── latent_to_visual_attn.pt
│   └── ...
└── vsp_spatial_planning.tar.gz    (Drive — keep as .tar.gz)
```

### Run

```bash
python3 run_dashboard.py
```

Open `http://127.0.0.1:8050` in your browser.

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
├── data/                                 ← Data (flat structure)
│   ├── ablation_results.json             ← Tracked in git
│   ├── train_direct.jsonl                ← Tracked in git
│   ├── metadata.json                     ← Shared via Drive (gitignored)
│   ├── tensors/                          ← Shared via Drive (gitignored)
│   └── vsp_spatial_planning.tar.gz       ← Shared via Drive (gitignored)
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

- **Level 1: Sample Landscape**: UMAP clustering of 996 samples by correctness and move direction.
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

## Extracted Data (996 Samples)

### Location

```
/home/scur0259/mirage/data/extracted/
├── metadata.json              ← 996-sample index (2.7 MB)
└── tensors/
    ├── sample_000/             ← .pt files, ~14 MB
    ├── sample_001/
    ├── ...
    └── sample_999/
```

### Dataset Summary

| Metric | Value |
|--------|-------|
| Samples | 996 |
| Latent tokens per sample | 6 |
| **Total size** | **16 GB** (~14 MB/sample) |

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

The data is shared privately via Google Drive — ask the team for the download link.

**From git** (cloned automatically):
```bash
git clone <repo-url>
```

Then download `metadata.json`, `tensors/`, and `vsp_spatial_planning.tar.gz` from Drive and place them in `data/` as shown in the Quick Start section above.

**On Snellius** (readable by all):
```bash
cp -r /home/scur0259/mirage/data/extracted ~/mirage_data/
```
