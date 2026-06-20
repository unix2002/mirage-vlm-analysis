# Mirage VLM — Latent Reasoning Analysis

Interactive visual analytics for inspecting latent reasoning in vision-language models.

**Model:** Qwen2.5-VL-7B-Instruct + latent tokens (= Mirage)
**Base codebase:** [UMass-Embodied-AGI/Mirage](https://github.com/UMass-Embodied-AGI/Mirage)

---

## Quick Start: Running the Dashboard

### Prerequisites

```bash
pip install -r requirements.txt
```

### Data Setup

The dashboard needs the following files in `data/`:

| File | Size | Source |
|------|------|--------|
| `ablation_results.json` | 5 MB | Git |
| `ablation_v2/subsets.json` | 1 MB | Git |
| `train_plans_gen.jsonl` | 12 MB | Git |
| `test_plans_gen.jsonl` | 5 MB | Git |
| `processed/rq2/probe_results.json` | 1 KB | Git |
| `processed/rq2/probe_results_per_sample.json` | 18 MB | Git |
| `metadata.json` | 2.7 MB | External (Drive) |
| `tensors/` (996 sample dirs) | 16 GB | External (Drive) |
| `vsp_spatial_planning/` (maze images) | 111 MB | External (Drive) |

The final structure:

```
data/
├── ablation_results.json
├── ablation_v2/subsets.json
├── train_plans_gen.jsonl
├── test_plans_gen.jsonl
├── processed/rq2/probe_results.json
├── processed/rq2/probe_results_per_sample.json
├── metadata.json
├── tensors/
│   ├── sample_000/{hidden_states.pt, latent_to_visual_attn.pt, ...}
│   └── ...
└── vsp_spatial_planning/
    ├── img/ (maze step images)
    └── train_direct.jsonl
```

The large files (`metadata.json`, `tensors/`, `vsp_spatial_planning/`) are shared privately via Google Drive — ask the team for the download link. The smaller JSON/JSONL files are tracked in git.

### Run

```bash
python3 run_dashboard.py
```

Open `http://127.0.0.1:8055` in your browser.

---

## Project Overview

The Mirage model (Yang et al., 2025) extends Qwen2.5-VL with **latent visual tokens** — compressed image features scattered between the user's question and the assistant's answer. These tokens carry visual reasoning information but are opaque: they cannot be read, inspected, or debugged directly.

This project builds an interactive Dash dashboard that makes latent reasoning tangible across four research questions:

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
├── dashboard/                            ← Interactive Visualization App
│   ├── components/                       ← UI Levels 1, 2, 3
│   │   ├── level1_landscape.py           ← UMAP landscape with maze overlays
│   │   ├── level2_path.py               ← Level 2 layout structure
│   │   └── level3_detail.py             ← Level 3 detail container
│   ├── callbacks/                        ← Dash callback logic
│   │   ├── level1.py                    ← UMAP controls, zoom, color, flippers
│   │   ├── level2.py                    ← Probing/ablation tab, KL fingerprint, dose response
│   │   ├── level3.py                    ← Token detail heatmap, probe bar, dependency curve
│   │   └── ablation_v2.py               ← Combinatorial ablation data loading
│   ├── app.py                            ← Dash app entry point
│   ├── layout.py                         ← Full-screen responsive layout
│   ├── data_loader.py                    ← Real data loader with eager UMAP
│   ├── gen_data.py                       ← Free-run reroute data (plan flippers)
│   ├── rq2_viz.py                        ← RQ2 probe visualization
│   └── mock_data.py                      ← Test data generator
├── mirage_vlm/                           ← Core analysis package
│   └── utils/maze_renderer.py            ← Maze grid/path trace generation
├── data/                                 ← Data (large files gitignored)
├── tests/dashboard/                      ← Test suite
├── run_dashboard.py                      ← Dashboard entry point
├── requirements.txt                      ← Python dependencies
└── README.md
```

---

## Dataset: VSP (Visual Spatial Planning)

The Mirage model's latent token mechanism requires paired output images — something standard VQA benchmarks don't provide. We use **VSP** — maze-based spatial reasoning with step-by-step board-state images.

| Split | Samples |
|-------|---------|
| Full training | 1,000 |
| Extracted (ours) | 996 |

### Token Sequence Structure

A typical ~475-token sequence contains (in order):

1. **System prompt** + maze description (text tokens)
2. **`<|vision_start|>`** + `<|image_pad|>` × N_visual + **`<|vision_end|>`** — input image patches
3. **Task prompt** (text tokens)
4. **`<|im_start|>assistant`**
5. **`<|latent_start|>`** + `<|latent_pad|>` × 6 + **`<|latent_end|>`** — **latent reasoning tokens**
6. **`<think></think>`** + answer text + **`<|im_end|>`**

The 6 latent tokens are always consecutive and at the same relative position. This determinism is critical for ablation (RQ3): the latent region can be reliably targeted.

---

## Dashboard Views

The dashboard is structured into three interactive levels:

- **Level 1: Sample Landscape** — UMAP projection of 996 samples, colored by reasoning intensity (KL), with maze overlays on zoom.
- **Level 2: Reasoning Path Analysis** — 
  - **Probing tab**: RQ2 decodability grid + layer-wise probe accuracy.
  - **Ablation tab**: Dose-response curve (KL vs tokens zeroed), KL fingerprint of all 63 subset combinations, plan status with free-run rerouting.
- **Level 3: Token Details** — Per-token spatial focus heatmap (RQ1), directional probe accuracy (RQ2), per-position KL decay curve (RQ3).

---

## Testing

```bash
PYTHONPATH=. python3 -m pytest tests/dashboard
```

Tests cover component structure, layout integrity, callback logic, maze rendering, zoom behavior, mock data generation, and RQ2 visualization. Data-dependent tests gracefully skip when external data is unavailable.

---

## Environment

| Component | Detail |
|-----------|--------|
| Python | 3.11+ |
| PyTorch | 2.3+ |
| GPU | 1× A100 (16 GB VRAM) |
| Dependencies | See `requirements.txt` |

---

## Getting the Data

The large data files (`metadata.json`, `tensors/`, `vsp_spatial_planning/`) are shared privately via Google Drive — ask the team for the download link. Place them in `data/` as shown in the Quick Start section.

The smaller files (`ablation_results.json`, `subsets.json`, `*_plans_gen.jsonl`, `probe_results*.json`) are tracked in git and available on clone.
