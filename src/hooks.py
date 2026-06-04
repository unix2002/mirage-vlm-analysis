"""
Forward hooks for Mirage VLM analysis.
Captures embeddings, attention weights, hidden states, and logits.
"""

import torch
from typing import Dict, List, Optional


class MirageAnalyzer:
    """
    Registers forward hooks on a Mirage (Qwen2.5-VL) model to extract:
    
    1. post_insertion_embeddings — token embeddings AFTER latent token scattering
    2. all_attention_weights — {layer_idx: attn_weights [B, heads, seq, seq]}
    3. hidden_states — {layer_idx: [B, seq_len, hidden_dim]}
    4. logits — [B, seq_len, vocab_size]
    """

    def __init__(self, model):
        self.model = model
        self.handles = []
        self.post_insertion_embeddings = None
        self.all_attention_weights = {}
        self.hidden_states = {}
        self.logits = None

    # ------------------------------------------------------------------
    #  Register / remove hooks
    # ------------------------------------------------------------------
    def register_all_hooks(self):
        self._remove_all()

        # -- Hook 1: EMBEDDINGS --
        # model.model (Qwen2_5_VLModel) receives inputs_embeds as a kwarg.
        # Use a pre-hook with kwargs to capture it.
        def capture_embeddings_pre(module, args, kwargs):
            embeds = kwargs.get("inputs_embeds", None)
            if embeds is not None:
                self.post_insertion_embeddings = embeds.detach().cpu()

        self.handles.append(
            self.model.model.register_forward_pre_hook(
                capture_embeddings_pre, with_kwargs=True
            )
        )

        # -- Hook 2: ATTENTION WEIGHTS & HIDDEN STATES (per layer) --
        for i, layer in enumerate(self.model.model.layers):

            def make_attn_hook(idx):
                def hook(module, input, output):
                    # output = (attn_output, attn_weights, past_key_value)
                    if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                        self.all_attention_weights[idx] = output[1].detach().cpu()
                return hook

            self.handles.append(
                layer.self_attn.register_forward_hook(make_attn_hook(i))
            )

            def make_hidden_hook(idx):
                def hook(module, input, output):
                    h = output[0] if isinstance(output, tuple) else output
                    self.hidden_states[idx] = h.detach().cpu()
                return hook

            self.handles.append(
                layer.register_forward_hook(make_hidden_hook(i))
            )

        # -- Hook 3: LOGITS --
        def capture_logits(module, input, output):
            self.logits = output.detach().cpu()

        self.handles.append(
            self.model.lm_head.register_forward_hook(capture_logits)
        )

    def remove_all_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    _remove_all = remove_all_hooks

    # ------------------------------------------------------------------
    #  Convenience: latent → visual attention submatrix
    # ------------------------------------------------------------------
    def get_latent_to_visual_attention(
        self,
        layer_idx: int,
        latent_positions: List[int],
        visual_positions: List[int],
        head_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Extract attn weights: latent queries → visual keys."""
        if layer_idx not in self.all_attention_weights:
            raise KeyError(
                f"Layer {layer_idx} not found. Available: "
                f"{list(self.all_attention_weights.keys())}"
            )
        attn = self.all_attention_weights[layer_idx]  # [B, heads, seq, seq]
        sub = attn[:, :, latent_positions, :][:, :, :, visual_positions]
        if head_idx is not None:
            sub = sub[:, head_idx, :, :]
        else:
            sub = sub.mean(dim=1)
        return sub.squeeze(0)


# ------------------------------------------------------------------
#  Quick test
# ------------------------------------------------------------------
def test_hooks(model, processor):
    from PIL import Image
    import numpy as np

    analyzer = MirageAnalyzer(model)
    analyzer.register_all_hooks()

    img = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=[img], return_tensors="pt"
    ).to(model.device)

    # Force eager attention to get weights
    old_impl = model.config._attn_implementation
    model.config._attn_implementation = "eager"

    with torch.no_grad():
        _ = model(**inputs, output_attentions=True)

    model.config._attn_implementation = old_impl

    print("\n=== HOOK TEST RESULTS ===")
    if analyzer.post_insertion_embeddings is not None:
        print(f"Embeddings shape:    {analyzer.post_insertion_embeddings.shape}")
    else:
        print("Embeddings:          None (not captured)")
    print(f"Attention layers:    {list(analyzer.all_attention_weights.keys())}")
    if analyzer.all_attention_weights:
        k0 = list(analyzer.all_attention_weights.keys())[0]
        print(f"Attn weights shape:  {analyzer.all_attention_weights[k0].shape}")
    if analyzer.logits is not None:
        print(f"Logits shape:        {analyzer.logits.shape}")
    else:
        print("Logits:              None")
    print(f"Hidden state layers: {list(analyzer.hidden_states.keys())}")
    if analyzer.hidden_states:
        k0 = list(analyzer.hidden_states.keys())[0]
        print(f"Hidden state shape:  {analyzer.hidden_states[k0].shape}")

    analyzer.remove_all_hooks()
    return analyzer


if __name__ == "__main__":
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    cache = "/scratch-shared/scur0259/hf_cache"

    print("Loading model + processor ...")
    processor = AutoProcessor.from_pretrained(model_path, cache_dir=cache)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto", cache_dir=cache
    )
    model.eval()

    test_hooks(model, processor)
    print("\nDone.")
