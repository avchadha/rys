"""Heatmap visualization for RYS accuracy/MRR matrices.

Generates per-dataset and aggregated heatmaps for both metrics.
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_heatmap(
    delta_matrix: np.ndarray,
    title: str,
    output_path: str,
    top_k: int = 5,
):
    """Plot a 2D heatmap of metric deltas.

    Args:
        delta_matrix: (L, L+1) matrix of deltas. NaN for invalid entries.
        title: Plot title.
        output_path: Where to save.
        top_k: Number of top configs to annotate.
    """
    num_layers = delta_matrix.shape[0]

    # Mask invalid entries
    masked = np.copy(delta_matrix)
    for i in range(num_layers):
        for j in range(num_layers + 1):
            if not (0 <= i < j <= num_layers):
                masked[i, j] = np.nan

    valid = masked[~np.isnan(masked)]
    if len(valid) == 0:
        print(f"  No valid entries for {output_path}")
        return

    vmax = max(abs(valid.min()), abs(valid.max()))
    if vmax == 0:
        vmax = 0.01

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(
        masked, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        origin="upper", aspect="auto",
    )
    plt.colorbar(im, ax=ax, label="Δ vs Baseline")

    # Annotate top-k
    valid_coords = [
        (i, j)
        for i in range(num_layers)
        for j in range(i + 1, num_layers + 1)
        if not np.isnan(delta_matrix[i, j])
    ]
    scored = [(c, delta_matrix[c[0], c[1]]) for c in valid_coords]
    scored.sort(key=lambda x: x[1], reverse=True)

    for rank, ((ci, cj), delta) in enumerate(scored[:top_k]):
        ax.annotate(
            f"({ci},{cj})\n{delta:+.2%}",
            xy=(cj, ci), fontsize=6, ha="center", va="center",
            color="black", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
        )

    ax.set_xlabel("j (end layer, exclusive)")
    ax.set_ylabel("i (start layer)")
    ax.set_title(title)
    ax.set_xticks(range(0, num_layers + 1, 5))
    ax.set_yticks(range(0, num_layers, 5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def generate_all_heatmaps(
    results_dir: str = "results",
    output_dir: str = "outputs",
    top_k: int = 5,
):
    """Generate per-dataset and aggregated heatmaps from scan results."""
    os.makedirs(output_dir, exist_ok=True)

    baseline_path = os.path.join(results_dir, "baseline_results.json")
    if not os.path.exists(baseline_path):
        print("No baseline_results.json found. Run the scan first.")
        return

    with open(baseline_path) as f:
        baseline_results = json.load(f)

    # Per-dataset heatmaps
    for ds_name, bl in baseline_results.items():
        for metric in ["accuracy", "mrr"]:
            matrix_path = os.path.join(results_dir, f"{ds_name}_{metric}_matrix.npy")
            if not os.path.exists(matrix_path):
                continue

            matrix = np.load(matrix_path)
            bl_val = bl[metric]
            delta = matrix - bl_val

            plot_heatmap(
                delta,
                title=(f"{ds_name} — Δ {metric} vs baseline ({bl_val:.3f})"),
                output_path=os.path.join(
                    output_dir, f"heatmap_{ds_name}_{metric}.png"
                ),
                top_k=top_k,
            )

    # Aggregated heatmaps
    for metric in ["accuracy", "mrr"]:
        agg_path = os.path.join(results_dir, f"aggregated_{metric}_delta.npy")
        if not os.path.exists(agg_path):
            continue

        agg_delta = np.load(agg_path)
        plot_heatmap(
            agg_delta,
            title=f"Aggregated Δ {metric} (mean across {len(baseline_results)} datasets)",
            output_path=os.path.join(output_dir, f"heatmap_aggregated_{metric}.png"),
            top_k=top_k,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    generate_all_heatmaps(args.results_dir, args.output_dir, args.top_k)
