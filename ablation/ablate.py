"""
RQ3 Zero-Out Ablation
=====================
Re-runs the Mirage forward pass with latent token embeddings zeroed out,
then computes KL divergence between original and ablated answer distributions.

Output: data/ablation/zero_out/
    sample_000/
        ablated_logits.pt       — [{answer_start}:, :] logits with latent zeroed
        kl_divergence.pt         — per-position KL(P_orig || P_abl)
        kl_summary.json          — {"kl_mean": 0.42, "kl_positions": [...]}
    ...
    all_kl_scores.json           — aggregate: {sample_id: kl_mean} for all samples
"""

import json
import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List

import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

# Path setup
sys.path.insert(0, '/home/scur0259/mirage/src')
sys.path.insert(0, '/home/scur0259/mirage/transformers/src')

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen2_5_VLConfig
from qwen_vl_utils import process_vision_info
from utils import (
    place_input_image, place_output_image, replace_visual_spectial_tokens,
    process_batch, remove_assistant_images, remove_user_images
)

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Configuration (overridable via command line)
# ---------------------------------------------------------------------------
MODEL_PATH = "Miiche/vsp_spatial_planning_direct_sft"
HF_CACHE = "/scratch-shared/scur0259/hf_cache"
DATA_DIR = Path("/home/scur0259/mirage/data")
EXTRACTED_DIR = Path("/scratch-shared/scur0259/mirage_extracted")
SOURCE_JSONL = DATA_DIR / "vsp_spatial_planning" / "train_direct_with_oi.jsonl"

# Defaults — overridden by command-line args
NUM_SAMPLES = 1000
START_SAMPLE = 0
NOISE_STD = 0.1

DEVICE = torch.device("cuda:0")  # NOT model.device
# ---------------------------------------------------------------------------
MODEL_PATH = "Miiche/vsp_spatial_planning_direct_sft"
HF_CACHE = "/scratch-shared/scur0259/hf_cache"
DATA_DIR = Path("/home/scur0259/mirage/data")
EXTRACTED_DIR = Path("/scratch-shared/scur0259/mirage_extracted")
OUTPUT_DIR = DATA_DIR / "ablation" / "zero_out"
SOURCE_JSONL = DATA_DIR / "vsp_spatial_planning" / "train_direct_with_oi.jsonl"

ABLATION_TYPE = "zero_out"  # "zero_out" | "shuffle" | "noise"
NUM_SAMPLES = 1000            # Start with 50; set to 1000 for full run
START_SAMPLE = 0

DEVICE = torch.device("cuda:0")  # NOT model.device


# ---------------------------------------------------------------------------
# KL Divergence
# ---------------------------------------------------------------------------
def compute_kl_divergence(
    orig_logits: torch.Tensor,   # [N_answer, vocab_size]
    abl_logits: torch.Tensor,    # [N_answer, vocab_size]
    eps: float = 1e-10,
) -> Dict:
    """
    Compute KL(P_orig || P_abl) per position and aggregate.

    Returns:
        kl_positions: [N_answer] KL at each answer position
        kl_mean:       scalar, mean KL across positions
        kl_max:        scalar, max KL across positions
        kl_sum:        scalar, total KL across positions
    """
    # Truncate to same length (in case of mismatched answer regions)
    min_len = min(orig_logits.shape[0], abl_logits.shape[0])
    orig = orig_logits[:min_len].float()
    abl = abl_logits[:min_len].float()

    orig_probs = torch.softmax(orig, dim=-1)
    abl_probs = torch.softmax(abl, dim=-1)

    log_orig = torch.log(orig_probs + eps)
    log_abl = torch.log(abl_probs + eps)

    # KL(P||Q) = Σ P(i) * (log P(i) - log Q(i))
    kl_per_pos = torch.sum(orig_probs * (log_orig - log_abl), dim=-1)

    return {
        "kl_positions": kl_per_pos.tolist(),
        "kl_mean": kl_per_pos.mean().item(),
        "kl_max": kl_per_pos.max().item(),
        "kl_sum": kl_per_pos.sum().item(),
    }


# ---------------------------------------------------------------------------
# Ablation Hook
# ---------------------------------------------------------------------------
def make_ablation_hook(latent_positions: List[int], mode: str = "zero_out",
                       noise_std: float = 0.1):
    """
    Create a pre-forward hook that modifies latent token embeddings.

    Args:
        latent_positions: indices of latent tokens in the sequence
        mode: "zero_out" | "shuffle" | "noise"
        noise_std: standard deviation for Gaussian noise (relative to embedding std)
    """
    def hook(module, args, kwargs):
        embeds = kwargs.get("inputs_embeds", None)
        if embeds is None or len(latent_positions) == 0:
            return

        # embeds: [B, seq_len, hidden_dim]
        seq_len = embeds.shape[1]
        valid_positions = [p for p in latent_positions if 0 <= p < seq_len]

        if not valid_positions:
            return

        if mode == "zero_out":
            embeds[0, valid_positions, :] = 0.0

        elif mode == "shuffle":
            # Randomly permute embeddings among latent positions
            original = embeds[0, valid_positions, :].clone()
            perm = torch.randperm(len(valid_positions))
            embeds[0, valid_positions, :] = original[perm]

        elif mode == "noise":
            # Add Gaussian noise scaled to embedding magnitude
            latent_embeds = embeds[0, valid_positions, :]
            emb_std = latent_embeds.std().item()
            noise = torch.randn_like(latent_embeds) * noise_std * emb_std
            embeds[0, valid_positions, :] = latent_embeds + noise

        kwargs["inputs_embeds"] = embeds

    return hook


def make_logits_hook(collector: dict):
    """Capture logits from lm_head."""
    def hook(module, input, output):
        collector["logits"] = output.detach().cpu()
    return hook


# ---------------------------------------------------------------------------
# Sample Processing
# ---------------------------------------------------------------------------
def process_sample(processor, sample: dict) -> dict:
    """
    Reconstruct the full input batch for one VSP sample.
    Mirrors extract.py's reconstruction logic exactly.
    """
    img_in = Image.open(sample["image_input"]).convert("RGB")
    img_out = Image.open(sample["image_output"]).convert("RGB")

    conversations = [{
        "role": "user", "content": [
            {"type": "image", "image": img_in},
            {"type": "text", "text": sample["text_input"]},
        ],
    }, {
        "role": "assistant", "content": [
            {"type": "image", "image": img_out},
            {"type": "text", "text": sample["text_output"]},
        ],
    }]

    # Full batch with both images
    text = processor.apply_chat_template(conversations, tokenize=False)
    text = place_input_image(text)
    text = place_output_image(text)
    texts = replace_visual_spectial_tokens([text])
    image_inputs, _ = process_vision_info(conversations)

    # User-only batch (input image)
    user_examples = remove_assistant_images([conversations])
    user_text = [processor.apply_chat_template(e, tokenize=False) for e in user_examples]
    user_text = replace_visual_spectial_tokens(user_text)
    ui, _ = process_vision_info(user_examples[0])
    user_batch = processor(text=user_text, images=ui, return_tensors="pt", padding=True)

    # Assistant-only batch (output image → latent)
    assistant_examples = remove_user_images([conversations])
    assistant_text = [processor.apply_chat_template(e, tokenize=False) for e in assistant_examples]
    assistant_text = replace_visual_spectial_tokens(assistant_text)
    ai, _ = process_vision_info(assistant_examples[0])
    assistant_batch = processor(text=assistant_text, images=ai, return_tensors="pt", padding=True)

    # Combine
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    batch["pixel_values"] = user_batch["pixel_values"]
    batch["image_grid_thw"] = user_batch["image_grid_thw"]
    batch["pixel_values_latent"] = assistant_batch["pixel_values"]
    batch["image_grid_thw_latent"] = assistant_batch["image_grid_thw"]

    # Apply process_batch (latent token scattering)
    lt_pad = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0]
    lt_start = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0]
    lt_end = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0]
    pad_tok = processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"][0]

    new_input_ids, new_attention_mask = process_batch(
        batch["input_ids"], batch["attention_mask"],
        lt_start, lt_end, lt_pad, 4, pad_tok
    )

    batch["input_ids"] = new_input_ids
    batch["attention_mask"] = new_attention_mask

    return batch


def find_latent_positions(input_ids: torch.Tensor, processor) -> List[int]:
    """Find positions of latent tokens in the input_ids sequence."""
    lt_pad_id = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0].item()
    lt_start_id = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0].item()
    lt_end_id = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0].item()

    latent_ids = {lt_pad_id, lt_start_id, lt_end_id}
    ids = input_ids[0]
    positions = [i for i in range(len(ids)) if ids[i].item() in latent_ids]
    return positions


# ---------------------------------------------------------------------------
# Main Ablation Loop
# ---------------------------------------------------------------------------
def atomic_torch_save(obj, path):
    """Save to temp file then copy to destination — avoids NFS corruption."""
    import tempfile, shutil
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(dir="/tmp", suffix=".pt", delete=False)
    try:
        torch.save(obj, tmp.name)
        tmp.close()
        # shutil.copy2 instead of move — handles cross-device (tmp is local SSD, dest may be NFS/scratch)
        shutil.copy2(tmp.name, str(path))
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


def run_ablation(
    model, processor,
    samples: List[dict],
    metadata: List[dict],
    ablation_mode: str = "zero_out",
    noise_std: float = 0.1,
):
    """Run ablation on all samples and save results."""
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    orig_impl = model.config._attn_implementation
    model.config._attn_implementation = "eager"

    all_kl_scores = {}
    handle = None
    logits_handle = None

    for s in tqdm(metadata, desc="Ablating"):
        sid = s["sample_id"]
        sample = samples[sid]

        # --- Load original logits ---
        orig_logits_path = EXTRACTED_DIR / "tensors" / f"sample_{sid:03d}" / "logits.pt"
        if not orig_logits_path.exists():
            logging.warning(f"Sample {sid}: no original logits, skipping")
            continue

        try:
            orig_logits = torch.load(orig_logits_path, map_location="cpu", weights_only=True)
        except Exception as e:
            logging.warning(f"Sample {sid}: failed to load logits ({e})")
            continue

        # --- Reconstruct inputs ---
        try:
            batch = process_sample(processor, sample)
        except Exception as e:
            logging.warning(f"Sample {sid}: preprocessing failed ({e})")
            continue

        batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # --- Find latent positions ---
        latent_positions = find_latent_positions(batch["input_ids"], processor)

        if not latent_positions:
            logging.warning(f"Sample {sid}: no latent tokens found in input!")
            continue

        # --- Register hooks ---
        # 1. Zero-out hook (pre-hook on model.model)
        abl_hook = make_ablation_hook(latent_positions, mode=ablation_mode,
                                      noise_std=noise_std)
        # 2. Logits capture
        logits_collector = {}
        logits_hook_fn = make_logits_hook(logits_collector)

        try:
            handle = model.model.register_forward_pre_hook(abl_hook, with_kwargs=True)
            logits_handle = model.lm_head.register_forward_hook(logits_hook_fn)

            with torch.no_grad():
                _ = model(**batch, output_attentions=True)

        except Exception as e:
            logging.warning(f"Sample {sid}: forward pass failed ({e})")
            continue
        finally:
            if handle:
                handle.remove()
            if logits_handle:
                logits_handle.remove()

        # --- Extract ablated logits ---
        if "logits" not in logits_collector or logits_collector["logits"] is None:
            logging.warning(f"Sample {sid}: no logits captured")
            continue

        abl_logits_full = logits_collector["logits"]  # [1, seq_len, vocab_size]

        # Slice answer region: from (last latent + 1) onwards
        answer_start = max(latent_positions) + 1
        abl_logits = abl_logits_full[0, answer_start:, :]  # [N_answer, vocab_size]

        # --- Compute KL divergence ---
        try:
            kl_result = compute_kl_divergence(orig_logits[0], abl_logits)
        except Exception as e:
            logging.warning(f"Sample {sid}: KL computation failed ({e})")
            continue

        # --- Save ---
        sample_out = output_dir / f"sample_{sid:03d}"
        sample_out.mkdir(parents=True, exist_ok=True)

        atomic_torch_save(abl_logits.half(), sample_out / "ablated_logits.pt")
        atomic_torch_save(torch.tensor(kl_result["kl_positions"]), sample_out / "kl_divergence.pt")

        with open(sample_out / "kl_summary.json", "w") as f:
            json.dump({
                "sample_id": sid,
                "ablation_mode": ablation_mode,
                "kl_mean": kl_result["kl_mean"],
                "kl_max": kl_result["kl_max"],
                "kl_sum": kl_result["kl_sum"],
                "num_answer_positions": len(kl_result["kl_positions"]),
                "latent_positions": latent_positions,
            }, f, indent=2)

        all_kl_scores[sid] = {
            "kl_mean": kl_result["kl_mean"],
            "kl_max": kl_result["kl_max"],
        }

        # Free memory
        del batch, abl_logits_full, abl_logits
        torch.cuda.empty_cache()

    # --- Aggregate ---
    model.config._attn_implementation = orig_impl

    with open(output_dir / "all_kl_scores.json", "w") as f:
        json.dump(all_kl_scores, f, indent=2)

    # Summary statistics
    kl_values = [v["kl_mean"] for v in all_kl_scores.values()]
    if kl_values:
        print(f"\n=== Ablation Summary ({ablation_mode}) ===")
        print(f"  Samples:         {len(kl_values)}")
        print(f"  KL mean:         {np.mean(kl_values):.4f}")
        print(f"  KL std:          {np.std(kl_values):.4f}")
        print(f"  KL min:          {np.min(kl_values):.4f}")
        print(f"  KL max:          {np.max(kl_values):.4f}")
        print(f"\n  High dependency  (KL > 2× mean):")
        threshold = 2 * np.mean(kl_values)
        high = {k: v for k, v in all_kl_scores.items() if v["kl_mean"] > threshold}
        for sid, scores in sorted(high.items(), key=lambda x: -x[1]["kl_mean"]):
            print(f"    sample_{sid:03d}: KL={scores['kl_mean']:.4f}")
        print(f"\n  Low dependency   (KL < 0.5× mean):")
        threshold = 0.5 * np.mean(kl_values)
        low = {k: v for k, v in all_kl_scores.items() if v["kl_mean"] < threshold}
        for sid, scores in sorted(low.items(), key=lambda x: x[1]["kl_mean"]):
            print(f"    sample_{sid:03d}: KL={scores['kl_mean']:.6f}")

    return all_kl_scores


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ3 Ablation Runner")
    parser.add_argument("--type", dest="ablation_type", default="zero_out",
                        choices=["zero_out", "shuffle", "noise"],
                        help="Ablation mode")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES,
                        help="Number of samples to ablate")
    parser.add_argument("--start", type=int, default=START_SAMPLE,
                        help="Starting sample index")
    parser.add_argument("--noise-std", type=float, default=NOISE_STD,
                        help="Gaussian noise std (noise mode only)")
    args = parser.parse_args()

    ABLATION_TYPE = args.ablation_type
    NUM_SAMPLES = args.num_samples
    START_SAMPLE = args.start
    NOISE_STD = args.noise_std

    # Output to type-specific directory
    OUTPUT_DIR = Path("/scratch-shared/scur0259/mirage_ablation") / ABLATION_TYPE

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    print("=" * 60)
    print(f"RQ3 Ablation: {ABLATION_TYPE}")
    print(f"Model: {MODEL_PATH}")
    print(f"Samples: {START_SAMPLE}–{START_SAMPLE + NUM_SAMPLES - 1}")
    if ABLATION_TYPE == "noise":
        print(f"Noise std: {NOISE_STD}")
    print("=" * 60)

    # --- Load data ---
    with open(SOURCE_JSONL) as f:
        all_samples = [json.loads(l) for l in f]

    with open(EXTRACTED_DIR / "metadata.json") as f:
        metadata = json.load(f)

    metadata = [s for s in metadata
                if START_SAMPLE <= s["sample_id"] < START_SAMPLE + NUM_SAMPLES]
    print(f"Loaded {len(metadata)} samples from metadata")

    # --- Load model ---
    print("Loading model...")
    config = Qwen2_5_VLConfig.from_pretrained(MODEL_PATH, cache_dir=HF_CACHE)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, config=config, device_map="auto",
        torch_dtype=torch.float16, cache_dir=HF_CACHE
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(MODEL_PATH, cache_dir=HF_CACHE)
    processor.tokenizer.add_tokens("<|latent_pad|>", special_tokens=True)
    processor.tokenizer.add_tokens("<|latent_start|>", special_tokens=True)
    processor.tokenizer.add_tokens("<|latent_end|>", special_tokens=True)

    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    # --- Run ---
    t0 = time.time()
    results = run_ablation(
        model, processor,
        all_samples, metadata,
        ablation_mode=ABLATION_TYPE,
        noise_std=NOISE_STD,
    )
    elapsed = time.time() - t0

    print(f"\nDone in {elapsed:.0f}s ({elapsed/len(metadata):.1f}s per sample)")
    print(f"Results saved to {OUTPUT_DIR}")
