"""Diagnose why latent->visual attention isn't captured. One sample. Prints:
  - the attn implementation actually in effect + the self_attn module type,
  - the RAW structure of layer-0 self_attn's output (is attn_weights at [1]? None?),
  - whether the model's forward returns .attentions (the standard route),
  - the MirageAnalyzer capture count.
Run:
  srun --partition=gpu_a100 --gpus=1 --time=00:15:00 --pty bash -c \
    "source venv/bin/activate; export HF_HOME=/scratch-shared/scur0259/hf_cache; \
     python3 ablation/probe_heat.py"
"""
import json
import sys
from pathlib import Path

for _p in ('/home/scur0259/mirage/src', '/home/scur0259/mirage/transformers/src',
           '/home/scur0259/mirage/ablation', '/home/scur0259/mirage/ablation/old'):
    sys.path.insert(0, _p)

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen2_5_VLConfig
from hooks import MirageAnalyzer
from utils import (place_input_image, place_output_image, replace_visual_spectial_tokens,
                   process_batch, remove_assistant_images, remove_user_images)

try:
    from ablate import MODEL_PATH, HF_CACHE, LATENT_SIZE
except ModuleNotFoundError:
    MODEL_PATH = "Miiche/vsp_spatial_planning_direct_sft"
    HF_CACHE = "/scratch-shared/scur0259/hf_cache"
    LATENT_SIZE = 4

SRC = Path("/home/scur0259/mirage/data/vsp_spatial_planning/test_direct_with_oi.jsonl")
sample = json.loads(open(SRC).readline())

processor = AutoProcessor.from_pretrained(MODEL_PATH, cache_dir=HF_CACHE)
for t in ("<|latent_pad|>", "<|latent_start|>", "<|latent_end|>"):
    processor.tokenizer.add_tokens(t, special_tokens=True)
config = Qwen2_5_VLConfig.from_pretrained(MODEL_PATH, cache_dir=HF_CACHE)
config._attn_implementation = "eager"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, config=config, device_map="auto", torch_dtype=torch.float16, cache_dir=HF_CACHE)
model.resize_token_embeddings(len(processor.tokenizer))
model.eval()
tk = processor.tokenizer
tid = {k: tk(v, return_tensors="pt")["input_ids"][0] for k, v in {
    "lt_pad": "<|latent_pad|>", "lt_start": "<|latent_start|>",
    "lt_end": "<|latent_end|>", "pad": "<|endoftext|>"}.items()}
model.config.latent_token_id = int(tid["lt_pad"])
model.config.latent_start_id = int(tid["lt_start"])
model.config.latent_end_id = int(tid["lt_end"])

print("top config _attn_implementation :", model.config._attn_implementation)
print("text_config _attn_implementation:",
      getattr(getattr(model.config, "text_config", None), "_attn_implementation", "n/a"))
print("layer0 self_attn type           :", type(model.model.layers[0].self_attn).__name__)

# ---- build batch (extract.py style, dummy output image) ----
img = Image.open(sample["image_input"]).convert("RGB")
conv = [{"role": "user", "content": [{"type": "image", "image": img},
                                     {"type": "text", "text": sample["text_input"]}]},
        {"role": "assistant", "content": [{"type": "image", "image": img},
                                          {"type": "text", "text": sample["text_output"]}]}]
text = replace_visual_spectial_tokens([place_output_image(place_input_image(
    processor.apply_chat_template(conv, tokenize=False)))])
ii, _ = process_vision_info(conv)
ue = remove_assistant_images([conv])
ut = replace_visual_spectial_tokens([processor.apply_chat_template(ue[0], tokenize=False)])
ui, _ = process_vision_info(ue[0])
ub = processor(text=ut, images=ui, return_tensors="pt", padding=True)
ae = remove_user_images([conv])
at = replace_visual_spectial_tokens([processor.apply_chat_template(ae[0], tokenize=False)])
ai, _ = process_vision_info(ae[0])
ab = processor(text=at, images=ai, return_tensors="pt", padding=True)
batch = processor(text=text, images=ii, return_tensors="pt", padding=True)
batch["pixel_values"] = ub["pixel_values"]
batch["image_grid_thw"] = ub["image_grid_thw"]
batch["pixel_values_latent"] = ab["pixel_values"]
batch["image_grid_thw_latent"] = ab["image_grid_thw"]
nid, nm = process_batch(batch["input_ids"], batch["attention_mask"],
                        tid["lt_start"], tid["lt_end"], tid["lt_pad"], LATENT_SIZE, tid["pad"])
batch["input_ids"], batch["attention_mask"] = nid, nm


def describe(o):
    if isinstance(o, (tuple, list)):
        return f"{type(o).__name__}[{len(o)}](" + ", ".join(describe(x) for x in o) + ")"
    if torch.is_tensor(o):
        return f"T{tuple(o.shape)}"
    return "None" if o is None else type(o).__name__


rec = {}


def sa_hook(m, i, out):
    rec.setdefault("o", describe(out))


h = model.model.layers[0].self_attn.register_forward_hook(sa_hook)
analyzer = MirageAnalyzer(model)
analyzer.register_all_hooks()
kw = dict(
    input_ids=batch["input_ids"].to(model.device),
    attention_mask=batch["attention_mask"].to(model.device),
    pixel_values=batch["pixel_values"].to(model.device),
    image_grid_thw=batch["image_grid_thw"].to(model.device),
    pixel_values_latent=batch["pixel_values_latent"].to(model.device),
    image_grid_thw_latent=batch["image_grid_thw_latent"].to(model.device),
)
with torch.no_grad():
    out = model(**kw, output_attentions=True)
h.remove()
analyzer.remove_all_hooks()

print("layer0 self_attn raw output     :", rec.get("o"))
attns = getattr(out, "attentions", None)
print("model output type               :", type(out).__name__)
print("out.attentions present          :", attns is not None,
      "| n:", len(attns) if attns else 0,
      "| [0] shape:", tuple(attns[0].shape) if attns else None)
print("MirageAnalyzer all_attn count   :", len(analyzer.all_attention_weights))
print("DONE")
