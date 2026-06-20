"""Free-running (greedy) ablation: GENERATE the model's plan under each latent-token
ablation subset, so the plan can actually reroute. Teacher-forcing (regen_test.py)
structurally hides this -- the gold token is fed at every step, so an early deviation
can never compound. Here we ablate the (self-computed) latents during prefill, then
let the model free-run, so a real path change is attributable to the ablation.

Method: prefill-then-greedy-generate.
  * build the batch like regen_test (input image, latent region formed by process_batch),
    truncate input_ids at answer_start (right after latent_end) -> the prompt,
  * zero the chosen latent slots at layer-0 input during prefill (same intervention
    point as the combo job); those ablated latents are KV-cached, so the answer free-runs,
  * greedy decode (do_sample=False) -> deterministic, so a change is due to the ablation.

Optimisations:
  * subset-batching: the subsets of ONE sample share an identical prompt (only WHICH
    latent slots are zeroed differs), so they run as one batched generate with no padding
    and per-row ablation. (Biggest win: batch=1 barely uses the A100.)
  * KL-pruning: a subset whose teacher-forced KL is ~0 didn't move the step-0 distribution,
    so free-running can't diverge there. Skip it (recorded, evaluated=false). Clean (k=0)
    and the full-ablation subset are always run.
  * tight greedy decode (--max-new-tokens, stop at <|im_end|>), sdpa (no attention weights
    needed here).

Per-sample output: <output-dir>/sample_XXX.json
  {sample_id, n, positions, token_labels, kl_threshold, clean_plan_gen, clean_text_gen,
   clean_plan_tf,                     # teacher-forced clean plan from the combo file (if present)
   subsets: {bitmask: {label, kl_mean, evaluated,
                       ablated_plan_gen, ablated_text_gen, changed, diverge_step}}}
Pruned subsets: {label, kl_mean, evaluated:false, changed:false}.

Usage (test):
  python3 ablation/regen_gen.py \
      --source-jsonl /home/scur0259/mirage/data/vsp_spatial_planning/test_direct_with_oi.jsonl \
      --kl-source    /home/scur0259/mirage/data/test_plans_dist.jsonl \
      --output-dir   /scratch-shared/scur0259/mirage_test_plans_gen \
      --num-samples 400 --batch-size 8 --kl-threshold 1e-3
Train: swap --source-jsonl train_direct_with_oi.jsonl, --kl-source ablated_plans_dist.jsonl.
"""
import argparse
import itertools
import json
import re
import sys
from pathlib import Path

for _p in ('/home/scur0259/mirage/src',
           '/home/scur0259/mirage/transformers/src',
           '/home/scur0259/mirage/ablation',
           '/home/scur0259/mirage/ablation/old'):
    sys.path.insert(0, _p)

import torch
from PIL import Image
from tqdm import tqdm
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


def token_label(i, n):
    if i == 0:
        return "latent_start"
    if i == n - 1:
        return "latent_end"
    return f"pad_{i}"


def all_subsets(n):
    """All non-empty bitmasks, ordered by (k, mask)."""
    out = []
    for k in range(1, n + 1):
        for c in itertools.combinations(range(n), k):
            key = "".join("1" if i in c else "0" for i in range(n))
            out.append((key, set(c), "zero " + "+".join(token_label(i, n) for i in c)))
    return out


def parse_plan(text):
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if not m:
        return []
    return [t.strip().upper() for t in re.split(r"[,\s]+", m.group(1)) if t.strip()]


def first_divergence(a, b):
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else None


def build_batch(processor, sample, tok_ids):
    """Form the 6-latent region (input image as a dummy for the missing output image),
    same as regen_test. The gold text only fixes the structure; it is truncated away
    before generation (never teacher-forced)."""
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
    ut = [processor.apply_chat_template(e, tokenize=False) for e in ue]
    ut = replace_visual_spectial_tokens(ut)
    ui, _ = process_vision_info(ue[0])
    ub = processor(text=ut, images=ui, return_tensors="pt", padding=True)

    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    batch["pixel_values"] = ub["pixel_values"]
    batch["image_grid_thw"] = ub["image_grid_thw"]
    nid, nm = process_batch(batch["input_ids"], batch["attention_mask"],
                            tok_ids["lt_start"], tok_ids["lt_end"], tok_ids["lt_pad"],
                            LATENT_SIZE, tok_ids["pad"])
    batch["input_ids"] = nid
    batch["attention_mask"] = nm
    return batch


def make_batched_zero_hook(row_positions):
    """layer-0 pre-hook: zero hidden_states[r, p, :] for each row r and its target
    positions p (the ablated latent slots). In-place on the same tensor."""
    def hook(module, args, kwargs):
        hs = kwargs.get("hidden_states", None)
        if hs is None and args:
            hs = args[0]
        if hs is None:
            return
        seq = hs.shape[1]
        for r, pos in enumerate(row_positions):
            for p in pos:
                if 0 <= p < seq:
                    hs[r, p, :] = 0
    return hook


@torch.no_grad()
def gen_for_positions(model, processor, layer0, prompt_ids, prompt_mask, pv, thw,
                      position_lists, batch_size, max_new, eos_id, pad_id):
    """Greedy-generate one answer per entry in position_lists (each a list of latent
    positions to zero). Same-sample => identical prompt => batch with no padding."""
    texts = []
    for s in range(0, len(position_lists), batch_size):
        chunk = position_lists[s:s + batch_size]
        bc = len(chunk)
        in_ids = prompt_ids.repeat(bc, 1)
        in_mask = prompt_mask.repeat(bc, 1)
        pvr = pv.repeat(bc, 1) if pv is not None else None
        thwr = thw.repeat(bc, 1) if thw is not None else None
        h = layer0.register_forward_pre_hook(make_batched_zero_hook(chunk), with_kwargs=True)
        try:
            out = model.generate(
                input_ids=in_ids, attention_mask=in_mask,
                pixel_values=pvr, image_grid_thw=thwr,
                max_new_tokens=max_new, do_sample=False,
                eos_token_id=eos_id, pad_token_id=pad_id)
        finally:
            h.remove()
        new = out[:, in_ids.shape[1]:]
        for r in range(bc):
            texts.append(processor.tokenizer.decode(new[r], skip_special_tokens=True))
    return texts


def load_kl_source(path):
    """jsonl (test_plans_dist / ablated_plans_dist) -> {sample_id: ({mask: kl}, clean_plan_tf)}."""
    out = {}
    if not path or not Path(path).exists():
        return out
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            kls = {m: s.get("kl_mean", 0.0) for m, s in e.get("subsets", {}).items()}
            out[int(e["sample_id"])] = (kls, e.get("clean_plan"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-jsonl", type=Path, required=True,
                    help="*_direct_with_oi.jsonl (input mazes + structure)")
    ap.add_argument("--kl-source", type=Path, default=None,
                    help="combo jsonl (test_plans_dist / ablated_plans_dist) for KL + pruning")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--num-samples", type=int, default=400)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--sample-ids", type=int, nargs="*", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--kl-threshold", type=float, default=1e-3,
                    help="skip subsets with teacher-forced kl_mean below this (always keep full ablation)")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--attn-impl", default="sdpa")
    args = ap.parse_args()

    with open(args.source_jsonl) as f:
        all_samples = [json.loads(l) for l in f]
    if args.sample_ids is not None:
        sids = [s for s in args.sample_ids if 0 <= s < len(all_samples)]
    else:
        hi = min(args.start + args.num_samples, len(all_samples))
        sids = list(range(args.start, hi))
    kl_src = load_kl_source(args.kl_source)
    print(f"{len(sids)} samples | batch={args.batch_size} | kl_threshold={args.kl_threshold} "
          f"| kl_source={'yes' if kl_src else 'NONE (running all 63)'}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, device_map="auto", torch_dtype=torch.float16,
        cache_dir=HF_CACHE, attn_implementation=args.attn_impl)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, cache_dir=HF_CACHE)
    for t in ("<|latent_pad|>", "<|latent_start|>", "<|latent_end|>"):
        processor.tokenizer.add_tokens(t, special_tokens=True)
    model.resize_token_embeddings(len(processor.tokenizer))
    model.eval()

    tk = processor.tokenizer
    tok_ids = {k: tk(v, return_tensors="pt")["input_ids"][0] for k, v in {
        "lt_pad": "<|latent_pad|>", "lt_start": "<|latent_start|>",
        "lt_end": "<|latent_end|>", "pad": "<|endoftext|>"}.items()}
    model.config.latent_token_id = int(tok_ids["lt_pad"])
    model.config.latent_start_id = int(tok_ids["lt_start"])
    model.config.latent_end_id = int(tok_ids["lt_end"])
    eos_id = tk.convert_tokens_to_ids("<|im_end|>")
    pad_id = int(tok_ids["pad"])
    layer0 = model.model.layers[0]

    outdir = args.output_dir / "gen"
    outdir.mkdir(parents=True, exist_ok=True)
    done = failed = 0
    for sid in tqdm(sids, desc="RegenGen"):
        outpath = outdir / f"sample_{sid:03d}.json"
        if outpath.exists():
            done += 1
            continue
        try:
            batch = build_batch(processor, all_samples[sid], tok_ids)
            ids = batch["input_ids"][0]
            lat_all = ((ids == tok_ids["lt_pad"]) | (ids == tok_ids["lt_start"])
                       | (ids == tok_ids["lt_end"])).nonzero(as_tuple=True)[0].tolist()
            combo_pos = sorted(lat_all)
            n = len(combo_pos)
            answer_start = max(lat_all) + 1

            prompt_ids = batch["input_ids"][:, :answer_start].to(model.device)
            prompt_mask = batch["attention_mask"][:, :answer_start].to(model.device)
            pv = batch["pixel_values"].to(model.device)
            thw = batch["image_grid_thw"].to(model.device)

            kls, clean_tf = kl_src.get(sid, ({}, None))

            # decide which subsets to free-run (clean is always first)
            subsets = all_subsets(n)
            full_mask = "1" * n
            to_run, pruned = [], []
            for key, idxset, label in subsets:
                kl = kls.get(key, None)
                if kls and key != full_mask and kl is not None and kl < args.kl_threshold:
                    pruned.append((key, idxset, label, kl))
                else:
                    to_run.append((key, idxset, label, kl))

            position_lists = [[]] + [[combo_pos[i] for i in sorted(idxset)]
                                     for (_, idxset, _, _) in to_run]
            gen_texts = gen_for_positions(
                model, processor, layer0, prompt_ids, prompt_mask, pv, thw,
                position_lists, args.batch_size, args.max_new_tokens, eos_id, pad_id)

            clean_text = gen_texts[0]
            clean_plan = parse_plan(clean_text)

            out = {
                "sample_id": sid, "n": n, "positions": combo_pos,
                "token_labels": {str(i): token_label(i, n) for i in range(n)},
                "kl_threshold": args.kl_threshold,
                "clean_plan_gen": clean_plan, "clean_text_gen": clean_text,
                "clean_plan_tf": clean_tf,
                "subsets": {},
            }
            for (key, idxset, label, kl), text in zip(to_run, gen_texts[1:]):
                plan = parse_plan(text)
                out["subsets"][key] = {
                    "label": label, "kl_mean": kl, "evaluated": True,
                    "ablated_plan_gen": plan, "ablated_text_gen": text,
                    "changed": plan != clean_plan,
                    "diverge_step": first_divergence(clean_plan, plan),
                }
            for key, idxset, label, kl in pruned:
                out["subsets"][key] = {"label": label, "kl_mean": kl,
                                       "evaluated": False, "changed": False}

            with open(outpath, "w") as f:
                json.dump(out, f)
            done += 1
        except Exception as e:
            failed += 1
            import traceback
            tqdm.write(f"[FAIL] sample {sid}: {e}")
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache()

    print(f"\n{done}/{len(sids)} present in {outdir}")
    if failed:
        print(f"{failed} failed")


if __name__ == "__main__":
    main()
