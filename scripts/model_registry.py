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
}


def load_model(key: str, device: str = "cuda", dtype=torch.float16):
    """Load a registered model and wrap it in a RYSScanner.

    Returns:
        (scanner, processor, spec) — spec is the MODEL_SPECS entry with
        'name' etc.
    """
    from transformers import AutoModel, CLIPImageProcessor

    from scanner.layer_loop import RYSScanner

    if key not in MODEL_SPECS:
        raise ValueError(f"Unknown model '{key}'. Options: {list(MODEL_SPECS)}")
    spec = dict(MODEL_SPECS[key])

    print(f"Loading {spec['name']}...")
    full_model = AutoModel.from_pretrained(
        spec["name"],
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
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

    n = scanner.num_layers
    if n != spec["expected_layers"]:
        print(f"  WARNING: expected {spec['expected_layers']} layers, got {n}")
    print(f"  {n} layers, device={device}, dtype={dtype}")
    return scanner, processor, spec
