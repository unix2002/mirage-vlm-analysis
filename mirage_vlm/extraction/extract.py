"""Extract hook data → lean JSON metadata + .pt tensors."""
import sys, json, torch, logging, os, gc
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, '/home/scur0259/mirage/src')
sys.path.insert(0, '/home/scur0259/mirage/transformers/src')

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen2_5_VLConfig
from hooks import MirageAnalyzer
from utils import (
    place_input_image, place_output_image, replace_visual_spectial_tokens,
    process_batch, remove_assistant_images, remove_user_images
)
from qwen_vl_utils import process_vision_info

logging.basicConfig(level=logging.WARNING)

CACHE = '/scratch-shared/scur0259/hf_cache'
MODEL_PATH = 'Qwen/Qwen2.5-VL-7B-Instruct'
DATA_PATH = '/home/scur0259/mirage/data/examples/sample.jsonl'
OUT_DIR = '/home/scur0259/mirage/data/extracted/'
LATENT_SIZE = 4

os.makedirs(OUT_DIR, exist_ok=True)
TENSOR_DIR = os.path.join(OUT_DIR, "tensors")
os.makedirs(TENSOR_DIR, exist_ok=True)

with open(DATA_PATH) as f:
    samples = [json.loads(l) for l in f]
print(f"Loaded {len(samples)} samples")

processor = AutoProcessor.from_pretrained(MODEL_PATH, cache_dir=CACHE)
processor.tokenizer.add_tokens("<|latent_pad|>", special_tokens=True)
processor.tokenizer.add_tokens("<|latent_start|>", special_tokens=True)
processor.tokenizer.add_tokens("<|latent_end|>", special_tokens=True)

config = Qwen2_5_VLConfig.from_pretrained(MODEL_PATH, cache_dir=CACHE)
config.compress_strategy = 'average'
config.latent_size = LATENT_SIZE
config.stage = 'stage1'
config._attn_implementation = "eager"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, config=config, device_map='auto',
    torch_dtype=torch.float16, cache_dir=CACHE
)
model.resize_token_embeddings(len(processor.tokenizer))
model.eval()

lt_pad = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0]
lt_start = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0]
lt_end = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0]
pad_tok = processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"][0]
model.config.latent_token_id = int(lt_pad)
model.config.latent_start_id = int(lt_start)
model.config.latent_end_id = int(lt_end)

records = []

for idx, sample in enumerate(tqdm(samples, desc="Extracting")):
    try:
        img_in = Image.open(sample['image_input']).convert('RGB')
        img_out = Image.open(sample['image_output']).convert('RGB')

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

        text = processor.apply_chat_template(conversations, tokenize=False)
        text = place_input_image(text)
        text = place_output_image(text)
        texts = replace_visual_spectial_tokens([text])
        image_inputs, _ = process_vision_info(conversations)

        user_examples = remove_assistant_images([conversations])
        user_text = [processor.apply_chat_template(e, tokenize=False) for e in user_examples]
        user_text = replace_visual_spectial_tokens(user_text)
        ui, _ = process_vision_info(user_examples[0])
        user_batch = processor(text=user_text, images=ui, return_tensors="pt", padding=True)

        assistant_examples = remove_user_images([conversations])
        assistant_text = [processor.apply_chat_template(e, tokenize=False) for e in assistant_examples]
        assistant_text = replace_visual_spectial_tokens(assistant_text)
        ai, _ = process_vision_info(assistant_examples[0])
        assistant_batch = processor(text=assistant_text, images=ai, return_tensors="pt", padding=True)

        batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
        batch['pixel_values'] = user_batch['pixel_values']
        batch['image_grid_thw'] = user_batch['image_grid_thw']
        batch['pixel_values_latent'] = assistant_batch['pixel_values']
        batch['image_grid_thw_latent'] = assistant_batch['image_grid_thw']

        new_input_ids, new_attention_mask = process_batch(
            batch["input_ids"], batch["attention_mask"],
            lt_start, lt_end, lt_pad, LATENT_SIZE, pad_tok
        )
        batch["input_ids"] = new_input_ids
        batch["attention_mask"] = new_attention_mask

        analyzer = MirageAnalyzer(model)
        analyzer.register_all_hooks()

        with torch.no_grad():
            out = model(
                input_ids=batch['input_ids'].to(model.device),
                attention_mask=batch['attention_mask'].to(model.device),
                pixel_values=batch['pixel_values'].to(model.device),
                image_grid_thw=batch['image_grid_thw'].to(model.device),
                pixel_values_latent=batch['pixel_values_latent'].to(model.device),
                image_grid_thw_latent=batch['image_grid_thw_latent'].to(model.device),
                output_attentions=True,
            )

        # ---- Save tensors to .pt ----
        sample_dir = os.path.join(TENSOR_DIR, f"sample_{idx:03d}")
        os.makedirs(sample_dir, exist_ok=True)

        ids = batch['input_ids'][0]
        latent_mask = (ids == lt_pad) | (ids == lt_start) | (ids == lt_end)
        img_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        visual_mask = (ids == img_pad_id)

        token_positions = {
            "latent": latent_mask.nonzero(as_tuple=True)[0].tolist(),
            "visual": visual_mask.nonzero(as_tuple=True)[0].tolist(),
        }

        # Latent→visual attention matrices (small)
        l2v_dict = {}
        for layer_idx, attn in analyzer.all_attention_weights.items():
            if token_positions["latent"] and token_positions["visual"]:
                l2v = analyzer.get_latent_to_visual_attention(
                    layer_idx, token_positions["latent"], token_positions["visual"]
                )
                l2v_dict[layer_idx] = l2v.half().cpu()
        if l2v_dict:
            torch.save(l2v_dict, os.path.join(sample_dir, "latent_to_visual_attn.pt"))

        # Hidden states (last 3 layers)
        max_layer = max(analyzer.hidden_states.keys())
        hidden = {l: analyzer.hidden_states[l].half().cpu() for l in [max_layer-2, max_layer-1, max_layer] if l in analyzer.hidden_states}
        torch.save(hidden, os.path.join(sample_dir, "hidden_states.pt"))

        # Logits: answer region only
        latent_positions = token_positions["latent"]
        answer_start = max(latent_positions) + 1 if latent_positions else 0
        answer_end = min(answer_start + 200, analyzer.logits.shape[1])
        logits_slice = analyzer.logits[:, answer_start:answer_end, :].half().cpu()
        torch.save(logits_slice, os.path.join(sample_dir, "logits.pt"))

        # Embeddings
        emb = analyzer.post_insertion_embeddings.half().cpu()
        torch.save(emb, os.path.join(sample_dir, "embeddings.pt"))

        analyzer.remove_all_hooks()

        # Store only metadata in JSON
        record = {
            "sample_id": idx,
            "image_input": sample["image_input"],
            "image_output": sample["image_output"],
            "text_input_short": sample["text_input"][:200],
            "text_output_short": sample["text_output"][:200],
            "seq_len": len(ids),
            "token_positions": token_positions,
            "num_latent": len(token_positions["latent"]),
            "num_visual": len(token_positions["visual"]),
            "tensor_dir": f"tensors/sample_{idx:03d}",
        }
        records.append(record)

        # Free memory
        del out, analyzer, l2v_dict, hidden, logits_slice, emb
        for key in list(batch.keys()):
            if isinstance(batch[key], torch.Tensor):
                del batch[key]
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"\n[ERROR] Sample {idx}: {e}")
        import traceback; traceback.print_exc()
        gc.collect()
        torch.cuda.empty_cache()

# ---- Export metadata JSON ----
out_file = os.path.join(OUT_DIR, "metadata.json")
with open(out_file, 'w') as f:
    json.dump(records, f, indent=2)

# Report sizes
total_mb = 0
for root, _, files in os.walk(OUT_DIR):
    for f in files:
        total_mb += os.path.getsize(os.path.join(root, f)) / 1e6

print(f"\n✅ Exported {len(records)} samples to {OUT_DIR}")
print(f"   Metadata: {os.path.getsize(out_file)/1e3:.1f} KB")
print(f"   Total size: {total_mb:.1f} MB")
