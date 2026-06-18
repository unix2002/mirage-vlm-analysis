import json
from pathlib import Path
import numpy as np
import plotly.graph_objects as go

# Paths provided by teammate
PROBE_RESULTS_PATH = Path("/home/scur0241/gerrit/mirage-vlm-analysis-fresh/data/processed/rq2/probe_results.json")
PER_SAMPLE_RESULTS_PATH = Path("/home/scur0241/gerrit/mirage-vlm-analysis-fresh/data/processed/rq2/probe_results_per_sample.json")

RQ2_GRID_DIRECTIONS = ["LEFT", "DOWN", "RIGHT", "UP"]

def load_rq2_data():
    """Load both static and per-sample probe results."""
    with open(PROBE_RESULTS_PATH, "r", encoding="utf-8") as f:
        static = json.load(f)
    with open(PER_SAMPLE_RESULTS_PATH, "r", encoding="utf-8") as f:
        per_sample = json.load(f)
    return static, per_sample

def build_rq2_static_bar(probe_results=None):
    """Build the static layer-wise decodability bar chart."""
    if probe_results is None:
        probe_results, _ = load_rq2_data()
        
    layers = [int(x) for x in probe_results["layers"]]
    y = [float(probe_results["per_layer"][str(layer)]["all_tokens_concat"]) for layer in layers]
    chance = float(probe_results.get("chance_baseline", 0.25))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[str(layer) for layer in layers],
            y=y,
            name="Concat Probe Accuracy",
            marker_color="#1f77b4",
            text=[f"{v:.3f}" for v in y],
            textposition="outside",
        )
    )
    fig.add_hline(
        y=chance,
        line_dash="dash",
        line_color="#b22222",
        annotation_text=f"Chance = {chance:.3f}",
        annotation_position="top left",
    )
    fig.update_layout(
        title=dict(text="RQ2: Layer-wise Decodability (Global)", font=dict(size=12)),
        xaxis_title="Layer",
        yaxis_title="Accuracy",
        template="plotly_white",
        height=300,
        margin=dict(l=40, r=20, t=40, b=40)
    )
    return fig

def build_rq2_dynamic_grid(sample_id, layer=26, per_sample_payload=None):
    """Build the sequence-wise decodability grid for a specific sample."""
    if per_sample_payload is None:
        _, per_sample_payload = load_rq2_data()
        
    samples = per_sample_payload.get("per_sample", {})
    
    # Handle sample_id mapping (numeric string vs sample_XXX)
    lookup_id = str(sample_id)
    if lookup_id.startswith("sample_"):
        lookup_id = str(int(lookup_id.split("_")[-1]))
    
    sample_blob = samples.get(lookup_id)
    if not sample_blob:
        return go.Figure().update_layout(title=f"Sample {sample_id} not found in RQ2 data")

    true_moves = sample_blob.get("true_move_sequence", [])
    layer_key = str(layer)
    token_map = sample_blob.get("per_layer", {}).get(layer_key, {}).get("token_step_direction_probs", {})
    token_ids = sorted(token_map.keys(), key=lambda x: int(x))
    
    max_steps = 9 # Per teammate requirement
    means = []
    stds = []
    row_labels = []
    true_dirs = []

    for step in range(max_steps):
        true_dir = true_moves[step] if step < len(true_moves) else "n/a"
        true_dirs.append(true_dir)
        row_labels.append(f"Step {step + 1}")

        row_mu = []
        row_sigma = []
        for direction in RQ2_GRID_DIRECTIONS:
            vals = []
            for tok in token_ids:
                probs_step = token_map.get(tok, {}).get(str(step), {})
                vals.append(float(probs_step.get(direction, 0.0)))
            
            if vals:
                arr = np.array(vals)
                row_mu.append(float(arr.mean()))
                row_sigma.append(float(arr.std()))
            else:
                row_mu.append(0.0)
                row_sigma.append(0.0)
        
        means.append(row_mu)
        stds.append(row_sigma)

    means = np.array(means)
    stds = np.array(stds)

    z_true = []
    z_other = []
    text_vals = []

    for r in range(max_steps):
        row_true = []
        row_other = []
        row_text = []
        for c, direction in enumerate(RQ2_GRID_DIRECTIONS):
            mu = means[r, c]
            sigma = stds[r, c]
            label = f"{mu:.2f}±{sigma:.2f}"
            row_text.append(label)
            
            if true_dirs[r] != "n/a" and direction == true_dirs[r]:
                row_true.append(mu)
                row_other.append(np.nan)
            else:
                row_true.append(np.nan)
                row_other.append(mu)
        
        z_true.append(row_true)
        z_other.append(row_other)
        text_vals.append(row_text)

    fig = go.Figure()
    
    # Background for non-true directions
    fig.add_trace(go.Heatmap(
        z=z_other,
        x=RQ2_GRID_DIRECTIONS,
        y=row_labels,
        colorscale=[[0, "#fff7f7"], [1, "#e8b4b4"]],
        showscale=False,
        xgap=1, ygap=1,
        hoverinfo='skip'
    ))
    
    # Foreground for true directions
    fig.add_trace(go.Heatmap(
        z=z_true,
        x=RQ2_GRID_DIRECTIONS,
        y=row_labels,
        colorscale=[[0, "#f7fcf7"], [1, "#a7dca7"]],
        text=text_vals,
        texttemplate="%{text}",
        textfont={"size": 8},
        showscale=False,
        xgap=1, ygap=1,
        hovertemplate="Step %{y}<br>%{x}<br>Prob: %{z:.3f}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text=f"RQ2: Decodability Grid (Sample {sample_id})", font=dict(size=12)),
        xaxis_title="Direction",
        yaxis_title="Sequence Step",
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=20, t=40, b=40)
    )
    fig.update_yaxes(autorange="reversed")
    return fig
