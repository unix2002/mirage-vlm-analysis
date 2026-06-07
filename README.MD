# Latent Reasoning in Vision-Language Models (VLM Analysis)

This project provides a multi-level visual analytics dashboard and a modular analysis framework for investigating latent reasoning paths in Vision-Language Models (VLMs), specifically focused on the Visual Spatial Prediction (VSP) task.

## 🚀 Quick Start: Running the Mock Dashboard

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

## 🏗️ Project Structure

The codebase is organized into a modular package structure to ensure research reproducibility and scalability:

- **`mirage_vlm/`**: Core analysis package.
  - `extraction/`: Logic for VLM tensor extraction and hooks.
  - `inference/`: Main entry points for model inference.
  - `tasks/`: Task definitions (e.g., Maze/VSP).
  - `training/`: Evaluation and probing scripts.
- **`dashboard/`**: Interactive visual analytics application.
  - `components/`: UI levels (Landscape, Path, Detail).
  - `mock_data.py`: High-fidelity data simulation for 50 samples.
- **`data/`**: Standardized data management (`reference/` for raw tensors, `processed/` for analysis).
- **`tests/`**: "Bulletproof" testing suite.
- **`run_dashboard.py`**: Root entry point for the application.

---

## 📊 Dashboard Views (Research Questions)

The dashboard is structured into three interactive levels mapped to specific Research Questions (RQs):

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
