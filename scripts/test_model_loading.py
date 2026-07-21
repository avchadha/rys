"""Verify the registered model loads and the scanner's forward path works.

Run this on the GPU box BEFORE launching a full scan — it exercises the
exact embed -> layers -> pool path the scanner uses, including the
EVA-CLIP layer-call signature fix (layers need two positional mask args).
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.model_registry import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="eva18b")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    scanner, processor, spec = load_model(args.model, device=device)
    print(f"Layers: {scanner.num_layers} "
          f"(expected {spec['expected_layers']})")
    n_params = sum(p.numel() for p in scanner.model.parameters())
    print(f"Vision params: {n_params:,}")

    test_image = Image.fromarray(
        np.random.randint(0, 255, (spec["image_size"], spec["image_size"], 3),
                          dtype=np.uint8)
    )
    inputs = processor(images=test_image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    print(f"Input shape: {tuple(pixel_values.shape)}")

    t0 = time.time()
    baseline = scanner.get_baseline_embeddings(pixel_values)
    print(f"Baseline embedding: {tuple(baseline.shape)} "
          f"({time.time() - t0:.1f}s)")
    assert baseline.shape == (1, spec["expected_hidden"]), baseline.shape

    cached = scanner.cache_baseline_states(pixel_values)
    assert len(cached) == scanner.num_layers + 1

    # Cached-path pooling must equal the direct baseline
    cached_pooled = scanner._pool_fn(cached[-1])
    assert torch.allclose(baseline.float(), cached_pooled.float(), atol=1e-4)

    # A mid-block duplication must change the embedding
    L = scanner.num_layers
    modified = scanner.run_config_from_cache(L // 4, L // 2, cached)
    assert not torch.allclose(baseline.float(), modified.float(), atol=1e-3)

    # Cache path must equal the direct path
    direct = scanner.run_config(L // 4, L // 2, pixel_values)
    assert torch.allclose(direct.float(), modified.float(), atol=1e-4)

    # Throughput probe (batch 8)
    batch = pixel_values.repeat(8, 1, 1, 1)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    scanner.get_baseline_embeddings(batch)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"Throughput (batch 8, full forward): {8 / dt:.1f} img/s")

    n = scanner.num_configs()
    print(f"Total RYS configs: {n}")
    print("\nAll checks passed!")


if __name__ == "__main__":
    main()
