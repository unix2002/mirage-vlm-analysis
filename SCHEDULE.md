# Mirage Project Schedule — 4 Weeks

| Week | Role | Tasks | Deliverable |
|------|------|-------|-------------|
| **Week 1** | A | Environment setup, forward hooks (`MirageAnalyzer`), smoke tests, data pipeline, extraction at scale | 50 samples extracted (7.9 GB), GitHub repo |
| | B, C, D | Familiarize with codebase, set up local dev environments | — |
| **Week 2** | A (RQ3) | Zero-out latent token ablation, KL divergence on all 50 samples | Ablation results, dependency scores |
| | B (RQ1) | Attention heatmap generation from `latent_to_visual_attn.pt` | Per-sample & aggregate heatmaps |
| | C (RQ2) | Linear probe training on `hidden_states.pt` | Per-layer probe accuracy |
| **Week 3** | A (RQ3) | Systematic ablation: zero-out, shuffle, noise injection | Ablation report with visualizations |
| | B (RQ1) | Aggregate attention patterns across 50 samples, identify common focus regions | Aggregate heatmap analysis |
| | C (RQ2) | Probe evaluation, per-layer accuracy comparison, cross-sample analysis | Probe accuracy report |
| | D (RQ4) | Dash skeleton with real data from Snellius, single-sample view | Working Dash prototype |
| **Week 4** | All | Integration: all RQs into Dash dashboard | Unified dashboard |
| | D (RQ4) | Multi-sample comparison, interactive controls, polish | Final Dash app |
| | All | Final report writing + presentation slides | Report + slides |

## Role Assignments

| Role | Primary RQ | Key Files |
|------|-----------|-----------|
| **A** — Pipeline & RQ3 | Causal dependence (ablation) | `src/hooks.py`, `src/extract.py`, `logits.pt` |
| **B** — RQ1 | Spatial focus (attention) | `latent_to_visual_attn.pt` |
| **C** — RQ2 | Information content (probing) | `hidden_states.pt`, `embeddings.pt` |
| **D** — RQ4 | Dashboard integration | All tensor types + `metadata.json` |

## Data Location

All extracted data: `/home/scur0259/mirage/data/extracted/` (50 samples, 7.9 GB)  
Access: `cp -r /home/scur0259/mirage/data/extracted ~/mirage_data/`
