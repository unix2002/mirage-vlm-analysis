"""Test-set heatmap + hidden-state extraction (multi-pass aware).

The Mirage forward generates the latents INCREMENTALLY, so self_attn fires several
times per layer with the latent tokens split across passes (a prefill square, a few
single-token steps, and a final block). A single-pass capture (MirageAnalyzer) misses
them. Here we hook every self_attn call, record its query offset (q_start = k - q),
and for each latent position pull its attention row from whichever call contains it,
slicing the visual keys -> a clean [n_latent, n_visual] latent->visual matrix. Hidden
states are stitched back into a full [1, seq, H] so the UMAP indexing works.

Forward matches train extract.py: eager attention + dummy pixel_values_latent
(proven no-op on the self-computed latents, but needed for the forward to return
attention weights).

Output (dashboard RealDataLoader reads it):
  <out>/metadata.json
  <out>/tensors/sample_XXX/latent_to_visual_attn.pt   # {layer: [n_latent, n_visual]}
  <out>/tensors/sample_XXX/hidden_states.pt           # {layer: [1, seq, hidden]}
"""
import argparse
import gc
import json
import sys
from collections import defaultdict
from pathlib import Path

for _p in ('/home/scur0259/mirage/src', '/home/scur0259/mirage/transformers/src',
           '/home/scur0259/mirage/ablation', '/home/scur0259/mirage/ablation/old'):
    sys.path.insert(0, _p)

import torch
from PIL import Image
from tqdm import tqdm
from qwen_vl_utils import process_vision_info

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen2_5_VLConfig
from utils import (place_input_image, place_output_image, replace_visual_spectial_tokens,
                   process_batch, remove_assistant_images, remove_user_images)

try:
    from ablate import MODEL_PATH, HF_CACHE, LATENT_SIZE
except ModuleNotFoundError:
    MODEL_PATH = "Miiche/vsp_spatial_planning_direct_sft"
    HF_CACHE = "/scratch-shared/scur0259/hf_cache"
    LATENT_SIZE = 4

DEFAULT_TEST = Path("/home/scur0259/mirage/data/vsp_spatial_planning/test_direct_with_oi.jsonl")


def build_batch_test(processor, sample, tok_ids):
    img = Image.open(sample["image_input"]).convert("RGB")
    conversations = [
        {"role": "user", "content": [{"type": "image", "image": img},
                                     {"type": "text", "text": sample["text_input"]}]},
        {"role": "assistant", "content": [{"type": "image", "image": img},
                                          {"type": "text", "text": sample["text_output"]}]},
    ]
    text = processor.apply_chat_template(conversations, tokenize=False)
    text = place_input_image(text)
    text = place_output_image(text)
    texts = replace_visual_spectial_tokens([text])
    image_inputs, _ = process_vision_info(conversations)

    ue = remove_assistant_images([conversations])
    ut = replace_visual_spectial_tokens([processor.apply_chat_template(e, tokenize=False) for e in ue])
    ui, _ = process_vision_info(ue[0])
    ub = processor(text=ut, images=ui, return_tensors="pt", padding=True)

    ae = remove_user_images([conversations])
    at = replace_visual_spectial_tokens([processor.apply_chat_template(e, tokenize=False) for e in ae])
    ai, _ = process_vision_info(ae[0])
    ab = processor(text=at, images=ai, return_tensors="pt", padding=True)

    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    batch["pixel_values"] = ub["pixel_values"]
    batch["image_grid_thw"] = ub["image_grid_thw"]
    batch["pixel_values_latent"] = ab["pixel_values"]
    batch["image_grid_thw_latent"] = ab["image_grid_thw"]

    new_ids, new_mask = process_batch(
        batch["input_ids"], batch["attention_mask"],
        tok_ids["lt_start"], tok_ids["lt_end"], tok_ids["lt_pad"], LATENT_SIZE, tok_ids["pad"])
    batch["input_ids"] = new_ids
    batch["attention_mask"] = new_mask
    return batch


def latent_to_visual(att_calls, latent_pos, visual_pos):
    """For each latent query position, pull its row from whichever self_attn call
    contains it (q_start <= p < q_start+q), average over heads, slice visual keys.
    Returns [n_latent, n_visual] or None if any latent isn't found."""
    rows = []
    for p in latent_pos:
        row = None
        for q_start, attn in att_calls:
            q = attn.shape[-2]
            if q_start <= p < q_start + q:
                vis = [v for v in visual_pos if v < attn.shape[-1]]
                row = attn[0, :, p - q_start, vis].float().mean(0)  # [n_visual]
                break
        if row is None:
            return None
        rows.append(row)
    return torch.stack(rows)  # [n_latent, n_visual]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, default=400)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--sample-ids", type=int, nargs="*", default=None)
    ap.add_argument("--source-jsonl", type=Path, default=DEFAULT_TEST)
    ap.add_argument("--output-dir", type=Path,
                    default=Path("/scratch-shared/scur0259/mirage_test_extracted"))
    args = ap.parse_args()

    with open(args.source_jsonl) as f:
        all_samples = [json.loads(l) for l in f]
    if args.sample_ids is not None:
        sids = [s for s in args.sample_ids if 0 <= s < len(all_samples)]
    else:
        hi = min(args.start + args.num_samples, len(all_samples))
        sids = list(range(args.start, hi))
    print(f"{len(sids)} test samples | source={args.source_jsonl}")

    processor = AutoProcessor.from_pretrained(MODEL_PATH, cache_dir=HF_CACHE)
    for t in ("<|latent_pad|>", "<|latent_start|>", "<|latent_end|>"):
        processor.tokenizer.add_tokens(t, special_tokens=True)
    config = Qwen2_5_VLConfig.from_pretrained(MODEL_PATH, cache_dir=HF_CACHE)
    config._attn_implementation = "eager"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, config=config, device_map="auto",
        torch_dtype=torch.float16, cache_dir=HF_CACHE)
    model.resize_token_embeddings(len(processor.tokenizer))
    model.eval()

    tk = processor.tokenizer
    tok_ids = {k: tk(v, return_tensors="pt")["input_ids"][0] for k, v in {
        "lt_pad": "<|latent_pad|>", "lt_start": "<|latent_start|>",
        "lt_end": "<|latent_end|>", "pad": "<|endoftext|>"}.items()}
    model.config.latent_token_id = int(tok_ids["lt_pad"])
    model.config.latent_start_id = int(tok_ids["lt_start"])
    model.config.latent_end_id = int(tok_ids["lt_end"])
    img_pad_id = tk.convert_tokens_to_ids("<|image_pad|>")
    n_layers = len(model.model.layers)

    # per-call capture hooks (cleared each sample)
    hid_calls = defaultdict(list)   # layer -> [hidden [1,q,H]]
    att_calls = defaultdict(list)   # layer -> [(q_start, attn [1,heads,q,k])]
    handles = []
    for i, layer in enumerate(model.model.layers):
        def mk_hid(idx):
            def h(m, inp, out):
                hs = out[0] if isinstance(out, (tuple, list)) else out
                hid_calls[idx].append(hs.detach())
            return h

        def mk_att(idx):
            def h(m, inp, out):
                aw = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else None
                if torch.is_tensor(aw):
                    att_calls[idx].append((aw.shape[-1] - aw.shape[-2], aw.detach()))
            return h
        handles.append(layer.register_forward_hook(mk_hid(i)))
        handles.append(layer.self_attn.register_forward_hook(mk_att(i)))

    out_dir = args.output_dir
    tensor_dir = out_dir / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.json"
    records, existing = [], set()
    if meta_path.exists():
        try:
            records = json.load(open(meta_path))
            existing = {r["sample_id"] for r in records}
            print(f"resuming: {len(records)} records present")
        except Exception:
            pass

    done = attn_saved = 0
    for sid in tqdm(sids, desc="TestExtract"):
        if sid in existing:
            done += 1
            continue
        try:
            hid_calls.clear()
            att_calls.clear()
            sample = all_samples[sid]
            batch = build_batch_test(processor, sample, tok_ids)
            ids = batch["input_ids"][0]
            seq_len = int(ids.shape[0])
            latent = ((ids == tok_ids["lt_pad"]) | (ids == tok_ids["lt_start"])
                      | (ids == tok_ids["lt_end"])).nonzero(as_tuple=True)[0].tolist()
            visual = (ids == img_pad_id).nonzero(as_tuple=True)[0].tolist()
            token_positions = {"latent": latent, "visual": visual}

            with torch.no_grad():
                model(input_ids=batch["input_ids"].to(model.device),
                      attention_mask=batch["attention_mask"].to(model.device),
                      pixel_values=batch["pixel_values"].to(model.device),
                      image_grid_thw=batch["image_grid_thw"].to(model.device),
                      pixel_values_latent=batch["pixel_values_latent"].to(model.device),
                      image_grid_thw_latent=batch["image_grid_thw_latent"].to(model.device),
                      output_attentions=True)

            sample_dir = tensor_dir / f"sample_{sid:03d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            # latent -> visual attention, all layers
            l2v = {}
            if latent and visual:
                for layer_idx in range(n_layers):
                    m = latent_to_visual(att_calls.get(layer_idx, []), latent, visual)
                    if m is not None:
                        l2v[layer_idx] = m.half().cpu()
            if l2v:
                torch.save(l2v, sample_dir / "latent_to_visual_attn.pt")
                attn_saved += 1
            else:
                tqdm.write(f"[WARN] sample {sid}: no attention assembled")

            # hidden states: stitch per-call hidden back to full [1, seq, H], last 3 layers
            hidden = {}
            for layer_idx in (n_layers - 3, n_layers - 2, n_layers - 1):
                parts = hid_calls.get(layer_idx, [])
                if parts:
                    full = torch.cat(parts, dim=1)  # call order == position order
                    if full.shape[1] == seq_len:
                        hidden[layer_idx] = full.half().cpu()
            if hidden:
                torch.save(hidden, sample_dir / "hidden_states.pt")

            records.append({
                "sample_id": sid,
                "image_input": sample.get("image_input"),
                "image_output": None,
                "map_desc": sample.get("map_desc"),
                "text_input_short": sample.get("text_input", "")[:200],
                "text_output_short": sample.get("text_output", "")[:200],
                "seq_len": seq_len,
                "token_positions": token_positions,
                "num_latent": len(latent),
                "num_visual": len(visual),
                "tensor_dir": f"tensors/sample_{sid:03d}",
            })
            existing.add(sid)
            done += 1

            del l2v, hidden
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            tqdm.write(f"[FAIL] sample {sid}: {e}")
            import traceback
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()

    for h in handles:
        h.remove()
    records.sort(key=lambda r: r["sample_id"])
    with open(meta_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nwrote {len(records)} records to {meta_path}  ({done}/{len(sids)} this run)")
    print(f"latent_to_visual_attn.pt saved for {attn_saved} samples this run")
    if attn_saved == 0 and done > 0:
        print("WARNING: no attention assembled -- check the self_attn call structure")


if __name__ == "__main__":
    main()
