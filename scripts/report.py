"""Generate the final analysis report from RYS scan results.

Ranks configs on the RANDOM control split (unbiased) using z-scored
aggregation, reports paired-bootstrap confidence intervals from persisted
per-image outcomes, and compares top configs against the pure-noise null
from Phase 1's noise control.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probes.nearest_centroid import (
    compute_centroids,
    score_nearest_centroid_detailed,
)

SPLITS = ("hard", "rand")


def load_dataset_bundle(results_dir, ds_name, baseline):
    """Load matrices, per-image outcomes, and baseline per-image outcomes."""
    bundle = {"baseline": baseline}
    for split in SPLITS:
        for metric in ("accuracy", "mrr"):
            path = os.path.join(
                results_dir, f"{ds_name}_{split}_{metric}_matrix.npy"
            )
            if os.path.exists(path):
                bundle[(split, metric)] = np.load(path)

    percase_path = os.path.join(results_dir, f"{ds_name}_percase.npz")
    if os.path.exists(percase_path):
        z = np.load(percase_path)
        bundle["percase"] = {k: z[k] for k in z.files}

    feats_path = os.path.join(results_dir, f"{ds_name}_baseline_features.npz")
    if os.path.exists(feats_path):
        z = np.load(feats_path, allow_pickle=False)
        centroids = compute_centroids(z["ref"], z["ref_labels"])
        for split in SPLITS:
            d = score_nearest_centroid_detailed(
                z[split], z[f"{split}_labels"], centroids
            )
            bundle[f"baseline_correct_{split}"] = d["correct"]
    return bundle


def paired_bootstrap_delta(cfg_correct_by_ds, bl_correct_by_ds,
                           n_boot=2000, seed=0):
    """CI for the across-dataset mean of per-dataset Δaccuracy.

    Images are resampled independently per dataset (paired: the same image
    indices are used for config and baseline, so image difficulty cancels).
    """
    rng = np.random.default_rng(seed)
    ds_list = list(cfg_correct_by_ds)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        deltas = []
        for ds in ds_list:
            c = cfg_correct_by_ds[ds]
            bl = bl_correct_by_ds[ds]
            idx = rng.integers(0, len(c), len(c))
            deltas.append(c[idx].mean() - bl[idx].mean())
        boots[b] = np.mean(deltas)
    point = np.mean([
        cfg_correct_by_ds[ds].mean() - bl_correct_by_ds[ds].mean()
        for ds in ds_list
    ])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi


def generate_report(results_dir="results", output_path="outputs/report.md",
                    top_k=10):
    baseline_path = os.path.join(results_dir, "baseline_results.json")
    if not os.path.exists(baseline_path):
        print("No baseline_results.json found. Run the scan first.")
        return

    with open(baseline_path) as f:
        baseline_results = json.load(f)

    probe_meta = {}
    meta_path = os.path.join(results_dir, "probe_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            probe_meta = json.load(f)

    per_dataset = {}
    for ds_name, bl in baseline_results.items():
        bundle = load_dataset_bundle(results_dir, ds_name, bl)
        if ("rand", "accuracy") in bundle:
            per_dataset[ds_name] = bundle
    if not per_dataset:
        print("No per-dataset matrices found.")
        return

    ds_names = sorted(per_dataset)
    num_layers = per_dataset[ds_names[0]][("rand", "accuracy")].shape[0]
    mask = np.array([[i < j for j in range(num_layers + 1)]
                     for i in range(num_layers)])
    valid_configs = [(i, j) for i in range(num_layers)
                     for j in range(i + 1, num_layers + 1)]
    cfg_row = {cfg: r for r, cfg in enumerate(valid_configs)}

    # Aggregations (rand split is the honest one; hard shown for contrast)
    agg = {}
    for split in SPLITS:
        deltas, zs = [], []
        for ds in ds_names:
            key = (split, "accuracy")
            if key not in per_dataset[ds]:
                continue
            d = per_dataset[ds][key] - per_dataset[ds]["baseline"][split]["accuracy"]
            deltas.append(d)
            v = d[mask]
            mu, sd = np.nanmean(v), np.nanstd(v)
            zs.append((d - mu) / (sd if sd > 1e-9 else 1.0))
        agg[split] = {
            "delta": np.nanmean(np.stack(deltas), axis=0),
            "zscore": np.nanmean(np.stack(zs), axis=0),
        }

    rank_matrix = agg["rand"]["zscore"]
    ranked = sorted(
        (c for c in valid_configs if not np.isnan(rank_matrix[c[0], c[1]])),
        key=lambda c: rank_matrix[c[0], c[1]], reverse=True,
    )

    lines = []
    lines.append("# RYS Scan Report\n")
    lines.append(f"- **Transformer layers:** {num_layers}")
    lines.append(f"- **Configs evaluated:** {len(ranked)} / {len(valid_configs)}")
    lines.append("- **Probe:** nearest-centroid (cosine), zero training")
    lines.append("- **Ranking basis:** z-scored Δaccuracy on the RANDOM control "
                 "split, averaged across datasets (the borderline split is "
                 "reported for contrast but is selection-sensitive)\n")

    # Probe sets
    lines.append("## Probe Sets\n")
    lines.append("| Dataset | Classes | Refs | Borderline n | Random n | "
                 "BL acc (border) | BL acc (rand) | Max noise Δacc |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for ds in ds_names:
        meta = probe_meta.get(ds, {})
        bl = per_dataset[ds]["baseline"]
        nc = meta.get("noise_control", {})
        worst_noise = max(
            (v["acc_delta_mean"]
             for s in SPLITS for k, v in nc.get(s, {}).items()
             if isinstance(v, dict) and k.startswith("alpha_")),
            default=float("nan"),
        )
        lines.append(
            f"| {ds} | {meta.get('n_classes', '?')} | {meta.get('n_ref', '?')} | "
            f"{meta.get('n_hard', '?')} | {meta.get('n_rand', '?')} | "
            f"{bl['hard']['accuracy']:.3f} | {bl['rand']['accuracy']:.3f} | "
            f"{worst_noise:+.3f} |"
        )
    lines.append("")
    lines.append("*Max noise Δacc = largest mean Δaccuracy produced by pure "
                 "Gaussian feature noise across tested magnitudes — the null "
                 "any real effect must beat.*\n")

    # Top configs with bootstrap CIs on the random split
    lines.append(f"## Top {top_k} Configs (z-scored Δacc, random split)\n")
    lines.append("| Rank | (i, j) | Block | z̄ | Δacc (rand) | 95% CI | "
                 "Δacc (border) |")
    lines.append("|---|---|---|---|---|---|---|")

    def bootstrap_for(cfg, split):
        cfg_by_ds, bl_by_ds = {}, {}
        for ds in ds_names:
            b = per_dataset[ds]
            if "percase" not in b or f"baseline_correct_{split}" not in b:
                continue
            row = b["percase"][f"{split}_correct"][cfg_row[cfg]]
            if (row < 0).any():
                continue
            cfg_by_ds[ds] = row.astype(float)
            bl_by_ds[ds] = b[f"baseline_correct_{split}"].astype(float)
        if not cfg_by_ds:
            return None
        return paired_bootstrap_delta(cfg_by_ds, bl_by_ds)

    for rank, cfg in enumerate(ranked[:top_k]):
        i, j = cfg
        boot = bootstrap_for(cfg, "rand")
        ci = f"[{boot[1]:+.3f}, {boot[2]:+.3f}]" if boot else "n/a"
        pt = f"{boot[0]:+.4f}" if boot else f"{agg['rand']['delta'][i, j]:+.4f}"
        lines.append(
            f"| {rank + 1} | ({i}, {j}) | {j - i} | "
            f"{rank_matrix[i, j]:+.3f} | {pt} | {ci} | "
            f"{agg['hard']['delta'][i, j]:+.4f} |"
        )
    lines.append("")

    # Worst configs (correct ordering: worst first)
    lines.append("## Worst 5 Configs\n")
    lines.append("| Rank | (i, j) | Block | z̄ | Δacc (rand) |")
    lines.append("|---|---|---|---|---|")
    for rank, cfg in enumerate(reversed(ranked[-5:])):
        i, j = cfg
        lines.append(
            f"| {rank + 1} | ({i}, {j}) | {j - i} | "
            f"{rank_matrix[i, j]:+.3f} | {agg['rand']['delta'][i, j]:+.4f} |"
        )
    lines.append("")

    # Key finding
    lines.append("## Key Finding\n")
    if ranked:
        bi, bj = ranked[0]
        boot = bootstrap_for((bi, bj), "rand")
        if boot and boot[1] > 0:
            verdict = ("**Layer duplication improved performance**: config "
                       f"({bi}, {bj}) has a random-split Δacc 95% CI of "
                       f"[{boot[1]:+.3f}, {boot[2]:+.3f}], excluding zero.")
        elif boot:
            verdict = ("**No config shows a significant improvement** on the "
                       f"unbiased random split — best config ({bi}, {bj}) CI "
                       f"[{boot[1]:+.3f}, {boot[2]:+.3f}] includes zero. "
                       "NOTE: with 1176 configs, the selected maximum is "
                       "inflated (winner's curse); confirm with "
                       "scripts/confirm_top.py on fresh images before "
                       "claiming an effect.")
        else:
            verdict = f"Best config: ({bi}, {bj}). Per-image data unavailable for CI."
        lines.append(verdict + "\n")

        lines.append("### Best Config — Per-Dataset Δacc (random split)\n")
        lines.append("| Dataset | Baseline | Config | Δ |")
        lines.append("|---|---|---|---|")
        for ds in ds_names:
            b = per_dataset[ds]
            m = b.get(("rand", "accuracy"))
            if m is None or np.isnan(m[bi, bj]):
                continue
            blv = b["baseline"]["rand"]["accuracy"]
            lines.append(f"| {ds} | {blv:.3f} | {m[bi, bj]:.3f} | "
                         f"{m[bi, bj] - blv:+.3f} |")
        lines.append("")

    # Summary statistics (rand split)
    v = agg["rand"]["delta"][mask]
    v = v[~np.isnan(v)]
    lines.append("## Summary Statistics (random split)\n")
    lines.append(f"- Mean Δacc across configs: {v.mean():+.4f}")
    lines.append(f"- Std: {v.std():.4f}")
    lines.append(f"- Positive configs: {int((v > 0).sum())} / {len(v)}")
    lines.append(f"- Max: {v.max():+.4f}   Min: {v.min():+.4f}\n")

    # Block size analysis
    lines.append("## Block Size Analysis (random split)\n")
    lines.append("| Block size | Mean Δacc | Best config | Best Δacc | N |")
    lines.append("|---|---|---|---|---|")
    d = agg["rand"]["delta"]
    for size in range(1, num_layers + 1):
        cfgs = [(i, i + size) for i in range(num_layers - size + 1)
                if not np.isnan(d[i, i + size])]
        if not cfgs:
            continue
        vals = [d[c[0], c[1]] for c in cfgs]
        b = int(np.argmax(vals))
        lines.append(f"| {size} | {np.mean(vals):+.4f} | {cfgs[b]} | "
                     f"{vals[b]:+.4f} | {len(cfgs)} |")
    lines.append("")

    # Anatomy cross-reference
    anatomy_path = os.path.join(results_dir, "anatomy", "anatomy.json")
    if os.path.exists(anatomy_path):
        with open(anatomy_path) as f:
            anatomy = json.load(f)
        rl = anatomy.get("mean_patch", {}).get("reasoning_layers", [])
        lines.append("## Layer Anatomy Cross-Reference\n")
        if rl:
            lines.append(
                f"The style/content anatomy analysis found content-dominant "
                f"(candidate 'reasoning') layers **{min(rl)}–{max(rl)}**. "
                "The Sapir-Whorf/RYS hypothesis predicts duplication "
                "tolerance inside this range and damage outside it — "
                "compare with the heatmaps.\n"
            )
        else:
            lines.append("Anatomy analysis found no content-dominant region — "
                         "under the RYS hypothesis, duplication should mostly "
                         "hurt this model.\n")

    # Heatmap references
    lines.append("## Heatmaps\n")
    for ds in ds_names:
        lines.append(f"### {ds}\n"
                     f"![{ds} rand](heatmap_{ds}_rand_accuracy.png)\n")
    lines.append("### Aggregated (rand, z-scored)\n"
                 "![aggregated](heatmap_aggregated_rand_accuracy_zscore.png)\n")
    lines.append("### Pareto\n![pareto](pareto_rand_accuracy.png)\n")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"Report saved to {output_path}")
    print(f"\n{'=' * 60}")
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-path", default="outputs/report.md")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    generate_report(args.results_dir, args.output_path, args.top_k)
