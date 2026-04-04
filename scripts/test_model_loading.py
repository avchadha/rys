"""Verify EVA-CLIP-18B vision encoder loads correctly and produces expected outputs."""

import gc

import torch
from transformers import AutoModel, CLIPImageProcessor
from PIL import Image
import numpy as np


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load full CLIP model, then extract vision encoder
    model_name = "BAAI/EVA-CLIP-18B"
    print(f"Loading {model_name}...")
    full_model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    processor = CLIPImageProcessor.from_pretrained(model_name)

    # Extract vision encoder and free text model
    vision_model = full_model.vision_model
    del full_model
    gc.collect()

    vision_model = vision_model.to(device).eval()
    print(f"\nVision encoder type: {type(vision_model).__name__}")

    # Check layers
    layers = list(vision_model.encoder.layers)
    num_layers = len(layers)
    print(f"Transformer layers: {num_layers}")

    # Check hidden size
    hidden_size = layers[0].self_attn.out_proj.out_features
    print(f"Hidden size: {hidden_size}")
    print(f"Vision params: {sum(p.numel() for p in vision_model.parameters()):,}")

    # Run inference on a test image
    print("\nRunning inference on a test image...")
    test_image = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    inputs = processor(images=test_image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device=device, dtype=torch.float16)
    print(f"Input shape: {pixel_values.shape}")

    with torch.no_grad():
        # Test embeddings
        hidden_states = vision_model.embeddings(pixel_values)
        print(f"Embeddings output shape: {hidden_states.shape}")

        # Test full forward through layers
        for layer in layers:
            out = layer(hidden_states)
            hidden_states = out[0] if isinstance(out, (tuple, list)) else out
        cls_token = hidden_states[:, 0]
        print(f"Final CLS shape: {cls_token.shape}")
        print(f"CLS norm: {cls_token.float().norm().item():.4f}")

    # Verify expected shapes
    assert num_layers == 48, f"Expected 48 layers, got {num_layers}"
    assert hidden_size == 5120, f"Expected hidden_size 5120, got {hidden_size}"
    assert cls_token.shape == (1, 5120), f"Expected (1, 5120), got {cls_token.shape}"

    total_configs = num_layers * (num_layers + 1) // 2
    print(f"\nTotal RYS configs: {total_configs}")
    assert total_configs == 1176

    print("\nAll checks passed!")


if __name__ == "__main__":
    main()
