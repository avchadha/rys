"""Generate a final analysis report from RYS scan results."""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_report(
    results_dir: str = "results",
    output_path: str = "outputs/report.md",
    top_k: int = 10,
):
    """Analyze scan results and produce a markdown report."""
    baseline_path = os.path.join(results_dir, "baseline_results.json")
    if not os.path.exists(baseline_path):
        print("No baseline_results.json found. Run the scan first.")
        return

    with open(baseline_path) as f:
        baseline_results = json.load(f)

    # Load probe metadata
    meta_path = os.path.join(results_dir, "probe_meta.json")
    probe_meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            probe_meta = json.load(f)

    # Load per-dataset matrices
    per_dataset = {}
    for ds_name, bl in baseline_results.items():
        acc_path = os.path.join(results_dir, f"{ds_name}_accuracy_matrix.npy")
        mrr_path = os.path.join(results_dir, f"{ds_name}_mrr_matrix.npy")
        if os.path.exists(acc_path) and os.path.exists(mrr_path):
            per_dataset[ds_name] = {
                "baseline_acc": bl["accuracy"],
                "baseline_mrr": bl["mrr"],
                "acc_matrix": np.load(acc_path),
                "mrr_matrix": np.load(mrr_path),
                "acc_delta": np.load(acc_path) - bl["accuracy"],
                "mrr_delta": np.load(mrr_path) - bl["mrr"],
            }

    if not per_dataset:
        print("No per-dataset matrices found.")
        return

    num_layers = next(iter(per_dataset.values()))["acc_matrix"].shape[0]

    # Recompute aggregated delta from per-dataset matrices
    agg_acc_delta = np.nanmean(
        np.stack([d["acc_delta"] for d in per_dataset.values()]), axis=0
    )
    agg_mrr_delta = np.nanmean(
        np.stack([d["mrr_delta"] for d in per_dataset.values()]), axis=0
    )

    # Valid configs
    valid_configs = [
        (i, j)
        for i in range(num_layers)
        for j in range(i + 1, num_layers + 1)
        if not np.isnan(agg_acc_delta[i, j])
    ]
    ranked = sorted(valid_configs, key=lambda c: agg_acc_delta[c[0], c[1]], reverse=True)

    # ── Build report ──
    lines = []
    lines.append("# RYS Experiment Report — EVA-CLIP-18B\n")

    lines.append("## Model\n")
    lines.append("- **Model:** `BAAI/EVA-CLIP-18B` (vision encoder)")
    lines.append(f"- **Transformer layers:** {num_layers}")
    lines.append(f"- **Total RYS configs evaluated:** {len(valid_configs)}")
    lines.append("- **Probe method:** Nearest-centroid classification "
                  "(cosine similarity, zero training)\n")

    # Probe set details
    lines.append("## Probe Sets\n")
    lines.append("| Dataset | Classes | Reference imgs | Hard test imgs | "
                 "Baseline acc | Baseline MRR |")
    lines.append("|---------|---------|---------------|---------------|"
                 "-------------|-------------|")
    for ds_name in sorted(per_dataset):
        d = per_dataset[ds_name]
        meta = probe_meta.get(ds_name, {})
        lines.append(
            f"| {ds_name} | {meta.get('n_classes', '?')} | "
            f"{meta.get('n_ref', '?')} | {meta.get('n_hard', '?')} | "
            f"{d['baseline_acc']:.3f} | {d['baseline_mrr']:.3f} |"
        )
    mean_bl_acc = np.mean([d["baseline_acc"] for d in per_dataset.values()])
    mean_bl_mrr = np.mean([d["baseline_mrr"] for d in per_dataset.values()])
    lines.append(
        f"| **Mean** | | | | **{mean_bl_acc:.3f}** | **{mean_bl_mrr:.3f}** |"
    )
    lines.append("")

    # Top configs
    lines.append("## Top Configs (by Aggregated Mean Δ Accuracy)\n")
    ds_names_sorted = sorted(per_dataset)
    header = ("| Rank | Config (i, j) | Block Size | Avg Δacc | Avg ΔMRR | "
              + " | ".join(f"{dn}" for dn in ds_names_sorted) + " |")
    sep = "|" + "|".join(["------"] * (5 + len(per_dataset))) + "|"
    lines.append(header)
    lines.append(sep)
    for rank, (ci, cj) in enumerate(ranked[:top_k]):
        d_acc = agg_acc_delta[ci, cj]
        d_mrr = agg_mrr_delta[ci, cj]
        block_size = cj - ci
        per_ds = " | ".join(
            f"{per_dataset[dn]['acc_delta'][ci, cj]:+.3f}"
            for dn in ds_names_sorted
        )
        lines.append(
            f"| {rank+1} | ({ci}, {cj}) | {block_size} | "
            f"{d_acc:+.4f} | {d_mrr:+.4f} | {per_ds} |"
        )
    lines.append("")

    # Worst configs
    lines.append("## Worst Configs\n")
    lines.append(header)
    lines.append(sep)
    for rank, (ci, cj) in enumerate(ranked[-5:]):
        d_acc = agg_acc_delta[ci, cj]
        d_mrr = agg_mrr_delta[ci, cj]
        block_size = cj - ci
        per_ds = " | ".join(
            f"{per_dataset[dn]['acc_delta'][ci, cj]:+.3f}"
            for dn in ds_names_sorted
        )
        lines.append(
            f"| {rank+1} | ({ci}, {cj}) | {block_size} | "
            f"{d_acc:+.4f} | {d_mrr:+.4f} | {per_ds} |"
        )
    lines.append("")

    # Key finding
    best_cfg = ranked[0] if ranked else None
    lines.append("## Key Finding\n")
    if best_cfg:
        ci, cj = best_cfg
        best_acc_delta = agg_acc_delta[ci, cj]
        best_mrr_delta = agg_mrr_delta[ci, cj]

        # Compute mean absolute accuracy for best config
        best_mean_acc = np.mean([
            per_dataset[dn]["acc_matrix"][ci, cj] for dn in per_dataset
        ])

        if best_acc_delta > 0:
            lines.append(
                f"**Yes — layer duplication improved performance.** "
                f"Config ({ci}, {cj}) (block size {cj-ci}) achieved a mean "
                f"accuracy improvement of {best_acc_delta:+.4f} "
                f"(MRR {best_mrr_delta:+.4f}) across {len(per_dataset)} datasets, "
                f"reaching mean accuracy {best_mean_acc:.3f} "
                f"vs baseline {mean_bl_acc:.3f}.\n"
            )
        else:
            lines.append(
                f"**No — no config improved over baseline on average.** "
                f"The best config ({ci}, {cj}) had mean Δacc={best_acc_delta:+.4f}, "
                f"ΔMRR={best_mrr_delta:+.4f}. Mean accuracy {best_mean_acc:.3f} "
                f"vs baseline {mean_bl_acc:.3f}.\n"
            )

        lines.append("### Best Config — Per-Dataset Performance\n")
        lines.append("| Dataset | Baseline Acc | Config Acc | Δ Acc | "
                     "Baseline MRR | Config MRR | Δ MRR |")
        lines.append("|---------|-------------|-----------|-------|"
                     "-------------|-----------|-------|")
        for dn in ds_names_sorted:
            d = per_dataset[dn]
            lines.append(
                f"| {dn} | {d['baseline_acc']:.3f} | "
                f"{d['acc_matrix'][ci, cj]:.3f} | "
                f"{d['acc_delta'][ci, cj]:+.3f} | "
                f"{d['baseline_mrr']:.3f} | "
                f"{d['mrr_matrix'][ci, cj]:.3f} | "
                f"{d['mrr_delta'][ci, cj]:+.3f} |"
            )
        lines.append(
            f"| **Mean** | **{mean_bl_acc:.3f}** | **{best_mean_acc:.3f}** | "
            f"**{best_acc_delta:+.3f}** | **{mean_bl_mrr:.3f}** | "
            f"**{np.mean([per_dataset[dn]['mrr_matrix'][ci, cj] for dn in per_dataset]):.3f}** | "
            f"**{best_mrr_delta:+.3f}** |"
        )
        lines.append("")

    # Summary statistics
    lines.append("## Summary Statistics\n")
    valid_acc = agg_acc_delta[~np.isnan(agg_acc_delta)]
    valid_mrr = agg_mrr_delta[~np.isnan(agg_mrr_delta)]
    lines.append(f"- **Mean Δacc across all configs:** {valid_acc.mean():+.4f}")
    lines.append(f"- **Std Δacc:** {valid_acc.std():.4f}")
    lines.append(f"- **Configs with positive Δacc:** "
                 f"{int((valid_acc > 0).sum())} / {len(valid_acc)}")
    lines.append(f"- **Max Δacc:** {valid_acc.max():+.4f}")
    lines.append(f"- **Min Δacc:** {valid_acc.min():+.4f}")
    lines.append(f"- **Mean ΔMRR:** {valid_mrr.mean():+.4f}")
    lines.append(f"- **Max ΔMRR:** {valid_mrr.max():+.4f}")
    lines.append("")

    # Block size analysis
    lines.append("## Block Size Analysis\n")
    lines.append("Mean aggregated Δ by duplicated block size (j - i):\n")
    lines.append("| Block Size | Mean Δacc | Mean ΔMRR | Best Config | "
                 "Best Δacc | Num Configs |")
    lines.append("|-----------|----------|----------|-------------|"
                 "----------|-------------|")
    for block_size in range(1, num_layers + 1):
        cfgs = [(i, j) for (i, j) in valid_configs if j - i == block_size]
        if not cfgs:
            continue
        acc_d = [agg_acc_delta[c[0], c[1]] for c in cfgs]
        mrr_d = [agg_mrr_delta[c[0], c[1]] for c in cfgs]
        best_idx = int(np.argmax(acc_d))
        lines.append(
            f"| {block_size} | {np.mean(acc_d):+.4f} | {np.mean(mrr_d):+.4f} | "
            f"{cfgs[best_idx]} | {acc_d[best_idx]:+.4f} | {len(cfgs)} |"
        )
    lines.append("")

    # Heatmap references
    lines.append("## Heatmaps\n")
    for ds_name in ds_names_sorted:
        lines.append(f"### {ds_name}\n![{ds_name}](heatmap_{ds_name}_accuracy.png)\n")
    lines.append("### Aggregated\n![Aggregated](heatmap_aggregated_accuracy.png)\n")

    # Write
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)

    print(f"Report saved to {output_path}")
    print(f"\n{'='*60}")
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-path", default="outputs/report.md")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    generate_report(args.results_dir, args.output_path, args.top_k)
