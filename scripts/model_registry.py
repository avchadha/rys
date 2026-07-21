"""Model registry: load a vision encoder + matching RYSScanner accessors.

Each entry knows how to load the model, which image processor to use, how to
call its transformer blocks, and how to pool. Verified facts for eva18b come
from inspecting the BAAI/EVA-CLIP-18B remote code (modeling_evaclip.py):

  - The repo has NO preprocessor_config.json; the model card itself uses the
    openai/clip-vit-large-patch14 processor.
  - EvaCLIPEncoderLayer.forward(hidden_states, attention_mask,
    causal_attention_mask) has two REQUIRED positional mask args -> the
    scanner must call layer(hs, None, None).
  - vision_model.embeddings prepends CLS at index 0 (patch 14, 224px,
    257 tokens); vision_model.post_layernorm follows the encoder. We apply
    post_layernorm in pool_fn so probed features live in the model's trained
    output space (uniformly for baseline and all configs).
"""

import gc

import torch


MODEL_SPECS = {
    "eva18b": {
        "name": "BAAI/EVA-CLIP-18B",
        "processor_name": "openai/clip-vit-large-patch14",
        "image_size": 224,
        "expected_layers": 48,
        "expected_hidden": 5120,
    },
    "internvit6b": {
        "name": "OpenGVLab/InternViT-6B-224px",
        "processor_name": "OpenGVLab/InternViT-6B-224px",
        "image_size": 224,
        "expected_layers": 45,
        "expected_hidden": 3200,
    },
    # DINOv3 ViT-7B/16 (Meta, 2025): largest open-weights SELF-SUPERVISED
    # ViT — no text supervision anywhere, making it the objective-ablation
    # arm against CLIP-trained EVA. Gated repo: accept license on HF first.
    # Interface (verified against transformers>=4.56 DINOv3ViTModel with a
    # random-init config; scanner path reproduces model.forward exactly):
    #   embeddings(px) -> (B, 201, 4096): CLS + 4 register tokens + 196
    #   patches; rope_embeddings(px) -> (cos, sin) shared by every layer;
    #   blocks at model.layer taking (h, attention_mask, position_embeddings)
    #   and returning a Tensor; final model.norm before CLS pooling.
    "dinov3": {
        "name": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        "processor_name": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        "image_size": 224,
        "expected_layers": 40,
        "expected_hidden": 4096,
        "num_prefix_tokens": 5,  # CLS + 4 register tokens
    },
}


def load_model(key: str, device: str = "cuda", dtype=torch.float16):
    """Load a registered model and wrap it in a RYSScanner.

    Returns:
        (scanner, processor, spec) — spec is the MODEL_SPECS entry with
        'name' etc.
    """
    from transformers import AutoImageProcessor, AutoModel, CLIPImageProcessor

    from scanner.layer_loop import RYSScanner

    if key not in MODEL_SPECS:
        raise ValueError(f"Unknown model '{key}'. Options: {list(MODEL_SPECS)}")
    spec = dict(MODEL_SPECS[key])

    print(f"Loading {spec['name']}...")
    full_model = AutoModel.from_pretrained(
        spec["name"],
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=(key != "dinov3"),
    )
    if key == "dinov3":
        processor = AutoImageProcessor.from_pretrained(spec["processor_name"])
    else:
        processor = CLIPImageProcessor.from_pretrained(spec["processor_name"])

    if key == "eva18b":
        vision_model = full_model.vision_model
        del full_model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        post_ln = vision_model.post_layernorm

        scanner = RYSScanner(
            vision_model,
            device=device,
            get_layers=lambda m: list(m.encoder.layers),
            embed_fn=lambda m, x: m.embeddings(x),
            # EvaCLIPEncoderLayer requires the two mask args positionally.
            layer_fn=lambda layer, h: layer(h, None, None),
            # Trained output space: post-LN CLS token.
            pool_fn=lambda h: post_ln(h[:, 0]),
        )
    elif key == "dinov3":
        # RoPE is computed from the input once per batch and shared by all
        # layers; embed_fn stashes it for layer_fn. Safe with the cached
        # path: cache_baseline_states always calls embed_fn on a batch
        # before any config re-runs layers on that batch (and rope depends
        # only on the fixed 224x224 grid anyway).
        model = full_model
        norm = model.norm
        rope_cache = {}

        def _dino_embed(m, x):
            rope_cache["pe"] = m.rope_embeddings(x)
            return m.embeddings(x)

        scanner = RYSScanner(
            model,
            device=device,
            get_layers=lambda m: list(m.model.layer),
            embed_fn=_dino_embed,
            layer_fn=lambda layer, h: layer(h, None, rope_cache["pe"]),
            # Trained output space: final norm, CLS at index 0 (matches
            # model.forward's pooler_output exactly — verified).
            pool_fn=lambda h: norm(h[:, 0]),
        )
    elif key == "internvit6b":
        # InternViT is a bare vision model (no text tower). NOTE: layer call
        # signature not yet verified against its remote code — run
        # scripts/test_model_loading.py before a full scan.
        vision_model = full_model
        scanner = RYSScanner(
            vision_model,
            device=device,
            get_layers=lambda m: list(m.encoder.layers),
            embed_fn=lambda m, x: m.embeddings(x),
            pool_fn=lambda h: h[:, 0],
        )

    scanner.num_prefix_tokens = spec.get("num_prefix_tokens", 1)
    n = scanner.num_layers
    if n != spec["expected_layers"]:
        print(f"  WARNING: expected {spec['expected_layers']} layers, got {n}")
    print(f"  {n} layers, device={device}, dtype={dtype}")
    return scanner, processor, spec
