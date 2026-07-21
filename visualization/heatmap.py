"""Heatmap + auxiliary visualization for RYS scan results.

Generates, per dataset and per split (hard/rand):
  - Δ-vs-baseline heatmaps for accuracy and MRR
Plus:
  - aggregated raw-delta and z-scored heatmaps
  - Pareto scatter (Δacc vs duplicated block size = compute overhead)
  - single-layer k-repeat curves (if repeat scan results exist)
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SPLITS = ("hard", "rand")
METRICS = ("accuracy", "mrr")


def plot_heatmap(delta_matrix, title, output_path, top_k=5, unit="pp"):
    """Plot a 2D upper-triangular heatmap of metric deltas."""
    num_layers = delta_matrix.shape[0]

    masked = np.copy(delta_matrix)
    for i in range(num_layers):
        for j in range(num_layers + 1):
            if not i < j:
                masked[i, j] = np.nan

    valid = masked[~np.isnan(masked)]
    if len(valid) == 0:
        print(f"  No valid entries for {output_path}")
        return

    vmax = max(abs(valid.min()), abs(valid.max())) or 0.01

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(
        masked, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        origin="upper", aspect="auto",
    )
    plt.colorbar(im, ax=ax, label="Δ vs baseline")

    coords = [
        (i, j)
        for i in range(num_layers)
        for j in range(i + 1, num_layers + 1)
        if not np.isnan(delta_matrix[i, j])
    ]
    scored = sorted(
        ((c, delta_matrix[c[0], c[1]]) for c in coords),
        key=lambda x: x[1], reverse=True,
    )
    for (ci, cj), delta in scored[:top_k]:
        label = f"({ci},{cj})\n{delta:+.2%}" if unit == "pp" else f"({ci},{cj})\n{delta:+.2f}"
        ax.annotate(
            label, xy=(cj, ci), fontsize=6, ha="center", va="center",
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


def plot_pareto(agg_delta, output_path, title):
    """Δ vs duplicated block size (compute-overhead Pareto view, as RYS-II)."""
    num_layers = agg_delta.shape[0]
    sizes, deltas = [], []
    for i in range(num_layers):
        for j in range(i + 1, num_layers + 1):
            if not np.isnan(agg_delta[i, j]):
                sizes.append(j - i)
                deltas.append(agg_delta[i, j])
    if not sizes:
        return
    sizes = np.array(sizes)
    deltas = np.array(deltas)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(sizes, deltas, s=8, alpha=0.35, color="tab:blue")

    # Pareto frontier: best delta at each block size, then running max
    best = {}
    for s, d in zip(sizes, deltas):
        best[s] = max(best.get(s, -np.inf), d)
    xs = sorted(best)
    running, front = -np.inf, []
    for s in xs:
        running = max(running, best[s])
        front.append(running)
    ax.plot(xs, [best[s] for s in xs], color="tab:orange", lw=1,
            alpha=0.6, label="best at block size")
    ax.plot(xs, front, color="tab:red", lw=2, label="Pareto frontier")

    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("duplicated block size (j - i) ∝ extra compute")
    ax.set_ylabel("aggregated Δ")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_repeats(results_dir, output_dir):
    """Single-layer k-repeat curves per dataset (RYS-II style)."""
    repeat_files = [
        f for f in os.listdir(results_dir) if f.endswith("_repeats.json")
    ]
    for fname in repeat_files:
        ds_name = fname[: -len("_repeats.json")]
        with open(os.path.join(results_dir, fname)) as f:
            results = json.load(f)
        if not results:
            continue

        ks = sorted({int(key.split(",")[1]) for key in results})
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, split in zip(axes, SPLITS):
            for k in ks:
                layers, accs = [], []
                for key, entry in results.items():
                    i, kk = map(int, key.split(","))
                    if kk == k and split in entry:
                        layers.append(i)
                        accs.append(entry[split]["accuracy"])
                order = np.argsort(layers)
                ax.plot(np.array(layers)[order], np.array(accs)[order],
                        marker="o", ms=3, lw=1, label=f"k={k}")
            ax.set_xlabel("layer index")
            ax.set_ylabel("accuracy")
            ax.set_title(f"{ds_name} — {split} split")
            ax.legend()
        fig.suptitle(f"{ds_name}: single layer repeated k times")
        plt.tight_layout()
        out = os.path.join(output_dir, f"repeats_{ds_name}.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  Saved: {out}")


def generate_all_heatmaps(results_dir="results", output_dir="outputs", top_k=5):
    """Generate all plots from scan results."""
    os.makedirs(output_dir, exist_ok=True)

    baseline_path = os.path.join(results_dir, "baseline_results.json")
    if not os.path.exists(baseline_path):
        print("No baseline_results.json found. Run the scan first.")
        return

    with open(baseline_path) as f:
        baseline_results = json.load(f)

    for ds_name, bl in baseline_results.items():
        for split in SPLITS:
            for metric in METRICS:
                matrix_path = os.path.join(
                    results_dir, f"{ds_name}_{split}_{metric}_matrix.npy"
                )
                if not os.path.exists(matrix_path):
                    continue
                matrix = np.load(matrix_path)
                bl_val = bl[split][metric]
                plot_heatmap(
                    matrix - bl_val,
                    title=f"{ds_name} [{split}] — Δ {metric} vs baseline ({bl_val:.3f})",
                    output_path=os.path.join(
                        output_dir, f"heatmap_{ds_name}_{split}_{metric}.png"
                    ),
                    top_k=top_k,
                )

    for split in SPLITS:
        for metric in METRICS:
            for kind, unit in (("delta", "pp"), ("zscore", "z")):
                agg_path = os.path.join(
                    results_dir, f"aggregated_{split}_{metric}_{kind}.npy"
                )
                if not os.path.exists(agg_path):
                    continue
                agg = np.load(agg_path)
                plot_heatmap(
                    agg,
                    title=f"Aggregated [{split}] Δ {metric} ({kind})",
                    output_path=os.path.join(
                        output_dir, f"heatmap_aggregated_{split}_{metric}_{kind}.png"
                    ),
                    top_k=top_k,
                    unit=unit,
                )

    agg_path = os.path.join(results_dir, "aggregated_rand_accuracy_delta.npy")
    if os.path.exists(agg_path):
        plot_pareto(
            np.load(agg_path),
            os.path.join(output_dir, "pareto_rand_accuracy.png"),
            "Δ accuracy vs duplicated block size (random split)",
        )

    plot_repeats(results_dir, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    generate_all_heatmaps(args.results_dir, args.output_dir, args.top_k)
