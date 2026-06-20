"""One-sample probe to decide how to free-run ablated generation. Prints:
  1. model/module structure + where hidden_states arrives at layer 0 (hook target),
  2. APPROACH A: prefill-then-generate (prompt already contains the latent scaffold) -- what regen_gen does,
  3. APPROACH B: test.py-style generate (prompt ends at assistant; model produces the latents).
Run:
  srun --partition=gpu_a100 --gpus=1 --time=00:15:00 --pty bash -c \
    "source venv/bin/activate; export HF_HOME=/scratch-shared/scur0259/hf_cache; \
     python3 ablation/probe_gen.py"
"""
import json
import sys
import traceback
from pathlib import Path

for p in ('/home/scur0259/mirage/src',
          '/home/scur0259/mirage/transformers/src',
          '/home/scur0259/mirage/ablation',
          '/home/scur0259/mirage/ablation/old'):
    sys.path.insert(0, p)

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from utils import (place_input_image, place_output_image, replace_visual_spectial_tokens,
                   process_batch, remove_assistant_images)

# ablate.py was moved under ablation/old/; we only need these three constants.
try:
    from ablate import MODEL_PATH, HF_CACHE, LATENT_SIZE
except ModuleNotFoundError:
    MODEL_PATH = "Miiche/vsp_spatial_planning_direct_sft"
    HF_CACHE = "/scratch-shared/scur0259/hf_cache"
    LATENT_SIZE = 4
    print("[note] ablate not importable; using hardcoded MODEL_PATH / HF_CACHE / LATENT_SIZE")

def banner(t):
    print("\n" + "=" * 18 + " " + t + " " + "=" * 18)


def find_first(cands):
    for c in cands:
        if Path(c).exists():
            return c
    return None


banner("PATHS (discovered)")
SRC = find_first([
    "/home/scur0259/mirage/data/vsp_spatial_planning/test_direct_with_oi.jsonl",
    "/home/scur0259/test_direct_with_oi.jsonl",
    "/home/scur0259/mirage/data/vsp_spatial_planning/test_direct.jsonl",
])
KL_TEST = find_first(["/home/scur0259/test_plans_dist.jsonl",
                      "/home/scur0259/mirage/data/test_plans_dist.jsonl"])
KL_TRAIN = find_first(["/home/scur0259/ablated_plans_dist.jsonl",
                       "/home/scur0259/mirage/data/ablated_plans_dist.jsonl"])
print("source jsonl (test):", SRC)
print("kl-source (test)   :", KL_TEST)
print("kl-source (train)  :", KL_TRAIN)
if SRC is None:
    print("\nERROR: could not find test_direct_with_oi.jsonl — paste its real path and I'll fix it.")
    sys.exit(1)
if "with_oi" not in str(SRC):
    print("WARNING: using test_direct.jsonl (no text_output) — build needs the _with_oi file; "
          "structure probe may differ.")

sample = json.loads(open(SRC).readline())

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, device_map="auto", torch_dtype=torch.float16,
    cache_dir=HF_CACHE, attn_implementation="sdpa")
processor = AutoProcessor.from_pretrained(MODEL_PATH, cache_dir=HF_CACHE)
for t in ("<|latent_pad|>", "<|latent_start|>", "<|latent_end|>"):
    processor.tokenizer.add_tokens(t, special_tokens=True)
model.resize_token_embeddings(len(processor.tokenizer))
model.eval()
tk = processor.tokenizer
tid = {k: tk(v, return_tensors="pt")["input_ids"][0] for k, v in {
    "lt_pad": "<|latent_pad|>", "lt_start": "<|latent_start|>",
    "lt_end": "<|latent_end|>", "pad": "<|endoftext|>"}.items()}
model.config.latent_token_id = int(tid["lt_pad"])
model.config.latent_start_id = int(tid["lt_start"])
model.config.latent_end_id = int(tid["lt_end"])
eos = tk.convert_tokens_to_ids("<|im_end|>")

banner("STRUCTURE")
print("model:", type(model).__name__, "| model.model:", type(model.model).__name__)
has_layers = hasattr(model.model, "layers")
print("model.model.layers?:", has_layers, "| n:", len(model.model.layers) if has_layers else "-")
if has_layers:
    print("layer0:", type(model.model.layers[0]).__name__,
          "| self_attn:", type(model.model.layers[0].self_attn).__name__)
print("config attn impl:", model.config._attn_implementation)

# ---- build the regen_test batch + latent positions ----
img = Image.open(sample["image_input"]).convert("RGB")
conv_full = [
    {"role": "user", "content": [{"type": "image", "image": img},
                                 {"type": "text", "text": sample["text_input"]}]},
    {"role": "assistant", "content": [{"type": "image", "image": img},
                                      {"type": "text", "text": sample["text_output"]}]},
]
text = replace_visual_spectial_tokens([place_output_image(place_input_image(
    processor.apply_chat_template(conv_full, tokenize=False)))])
image_inputs, _ = process_vision_info(conv_full)
ue = remove_assistant_images([conv_full])
ut = replace_visual_spectial_tokens([processor.apply_chat_template(ue[0], tokenize=False)])
ui, _ = process_vision_info(ue[0])
ub = processor(text=ut, images=ui, return_tensors="pt", padding=True)
batch = processor(text=text, images=image_inputs, return_tensors="pt", padding=True)
batch["pixel_values"] = ub["pixel_values"]
batch["image_grid_thw"] = ub["image_grid_thw"]
nid, nm = process_batch(batch["input_ids"], batch["attention_mask"],
                        tid["lt_start"], tid["lt_end"], tid["lt_pad"], LATENT_SIZE, tid["pad"])
batch["input_ids"], batch["attention_mask"] = nid, nm
ids = batch["input_ids"][0]
lat = ((ids == tid["lt_pad"]) | (ids == tid["lt_start"]) | (ids == tid["lt_end"])).nonzero(as_tuple=True)[0].tolist()
answer_start = max(lat) + 1
print("\ninput_ids:", tuple(batch["input_ids"].shape),
      "| pixel_values:", tuple(batch["pixel_values"].shape),
      "| image_grid_thw:", batch["image_grid_thw"].tolist())
print("latent positions:", lat, "| answer_start:", answer_start)
print("decoded around latents:", repr(tk.decode(ids[min(lat) - 2: answer_start + 3].tolist())))

# ---- record where hidden_states arrives at layer 0 ----
rec = {}


def sig_hook(m, a, kw):
    if "sig" not in rec:
        rec["sig"] = ("hidden_states" in kw, len(a),
                      tuple(a[0].shape) if a and torch.is_tensor(a[0]) else None,
                      tuple(kw["hidden_states"].shape) if "hidden_states" in kw else None)


banner("APPROACH A: prefill-then-generate (regen_gen)")
prompt = batch["input_ids"][:, :answer_start].to(model.device)
pmask = batch["attention_mask"][:, :answer_start].to(model.device)
h = model.model.layers[0].register_forward_pre_hook(sig_hook, with_kwargs=True)
try:
    with torch.no_grad():
        out = model.generate(input_ids=prompt, attention_mask=pmask,
                             pixel_values=batch["pixel_values"].to(model.device),
                             image_grid_thw=batch["image_grid_thw"].to(model.device),
                             max_new_tokens=64, do_sample=False,
                             eos_token_id=eos, pad_token_id=int(tid["pad"]))
    print("OK -> decoded:", repr(tk.decode(out[0, prompt.shape[1]:], skip_special_tokens=True)))
except Exception:
    print("FAILED:")
    traceback.print_exc()
finally:
    h.remove()
print("layer0 hidden_states signature (in_kwargs, n_args, args0_shape, kw_shape):", rec.get("sig"))

banner("APPROACH B: test.py-style generate (model produces latents)")
try:
    conv_u = [{"role": "user", "content": [{"type": "image", "image": img},
                                           {"type": "text", "text": sample["text_input"]}]}]
    tb = place_input_image(processor.apply_chat_template(conv_u, tokenize=False), sep_token=None)
    iib, _ = process_vision_info(conv_u)
    inp = processor(text=[tb + "<|im_start|>assistant"], images=iib, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=64, do_sample=False,
                             eos_token_id=eos, pad_token_id=int(tid["pad"]))
    print("OK -> decoded:", repr(tk.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)))
except Exception:
    print("FAILED:")
    traceback.print_exc()

print("\nDONE.")
