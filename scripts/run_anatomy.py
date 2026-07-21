"""Run the layer-anatomy analysis (Sapir-Whorf ViT analog).

Cheap: one cached forward pass over 64 synthetic images. Produces the
three-curve anatomy plot, per-layer PCA panels, and anatomy.json with the
content-dominant ("reasoning phase") layer range — a prediction for where
the RYS duplication scan should find tolerance.

Usage:
    python scripts/run_anatomy.py --output-dir results/anatomy
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.style_content import StyleContentGrid
from benchmarks.suite import get_transform
from scanner.anatomy import (
    centered_similarity_curves,
    extract_layerwise_features,
    plot_anatomy,
    plot_pca_grid,
    save_anatomy_results,
)
from scripts.model_registry import load_model


def main():
    parser = argparse.ArgumentParser(description="ViT layer anatomy")
    parser.add_argument("--model", type=str, default="eva18b")
    parser.add_argument("--output-dir", type=str, default="results/anatomy")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scanner, processor, spec = load_model(args.model, device=device)
    transform = get_transform(processor, image_size=spec["image_size"])

    grid = StyleContentGrid(transform=transform)
    loader = DataLoader(grid, batch_size=args.batch_size, shuffle=False,
                        num_workers=2)

    import numpy as np

    results = {}
    for pool in ("mean_patch", "cls"):
        print(f"Extracting layerwise features (pool={pool})...")
        feats = extract_layerwise_features(scanner, loader, pool=pool)

        # Persist raw per-layer features (fp16) so any curve, PCA view, or
        # alternative pairing analysis can be regenerated without GPU time.
        np.savez_compressed(
            os.path.join(args.output_dir, f"anatomy_features_{pool}.npz"),
            features=feats.astype(np.float16),
            content_labels=grid.content_labels,
            style_labels=grid.style_labels,
        )

        curves = centered_similarity_curves(
            feats, grid.content_labels, grid.style_labels
        )
        results[pool] = curves

        rl = curves["reasoning_layers"]
        span = f"{min(rl)}..{max(rl)}" if rl else "none"
        print(f"  Content-dominant layers ({pool}): {span}")

        plot_anatomy(
            curves,
            os.path.join(args.output_dir, f"anatomy_{pool}.png"),
            title=f"{spec['name']} layer anatomy (pool={pool})",
        )
        plot_pca_grid(
            feats, grid.content_labels, grid.style_labels,
            os.path.join(args.output_dir, f"anatomy_pca_{pool}.png"),
        )
        del feats

    save_anatomy_results(results, args.output_dir)


if __name__ == "__main__":
    main()
