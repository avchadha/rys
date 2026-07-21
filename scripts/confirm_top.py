"""Stage-2 confirmation: re-evaluate top configs on FRESH test images.

The sweep ranks 1176 configs on ~100-image splits, so the top of the
leaderboard is inflated by selection (winner's curse). This script takes
the top-K configs from the sweep plus K random control configs, and
re-scores them on a larger, never-before-used test sample per dataset.
An effect is real only if it survives here — and beats the random-config
distribution.

Mirrors the RYS-II protocol: small probes guide search, large probes judge
final candidates.

Usage (after run_scan.py):
    python scripts/confirm_top.py --data-dir /path/to/data
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.suite import get_labels, load_benchmark
from probes.nearest_centroid import (
    compute_centroids,
    score_nearest_centroid_detailed,
)
from scripts.model_registry import load_model
from scripts.report import paired_bootstrap_delta
from scripts.run_scan import (
    DEFAULT_DATASETS,
    extract_baseline_features,
    extract_features_for_configs,
)


def resample_reference_indices(train_ds, meta, seed=1337):
    """Fresh reference images for the confirmation stage.

    Sweep-stage configs were selected partly for favorable interaction with
    the sweep's specific few-shot centroids; reusing those refs would carry
    that component of the winner's curse into stage 2. Prefer images the
    sweep never used; fall back to sweep refs only when a class is too small.
    """
    labels = get_labels(train_ds)
    sweep_refs = set(meta["ref_indices"])
    n_per_class = meta["n_ref_per_class"]
    rng = np.random.default_rng(seed)

    indices = []
    for label in meta["class_subset"]:
        class_idx = np.where(labels == label)[0]
        fresh = np.array([i for i in class_idx if i not in sweep_refs])
        take = min(n_per_class, len(fresh))
        chosen = rng.choice(fresh, take, replace=False).tolist() if take else []
        if take < n_per_class:  # tiny class: top up from sweep refs
            used = [i for i in class_idx if i in sweep_refs]
            chosen += used[: n_per_class - take]
        indices.extend(int(i) for i in chosen)
    return indices, labels[indices]


def pick_configs(results_dir, num_layers, top_k, n_random, seed=99):
    """Top-K configs by z-scored rand-split aggregate + random controls."""
    agg = np.load(os.path.join(
        results_dir, "aggregated_rand_accuracy_zscore.npy"
    ))
    valid = [(i, j) for i in range(num_layers)
             for j in range(i + 1, num_layers + 1)
             if not np.isnan(agg[i, j])]
    ranked = sorted(valid, key=lambda c: agg[c[0], c[1]], reverse=True)
    top = ranked[:top_k]

    rng = np.random.default_rng(seed)
    pool = [c for c in valid if c not in set(top)]
    controls = [pool[k] for k in rng.choice(len(pool), min(n_random, len(pool)),
                                            replace=False)]
    return top, controls


def main():
    parser = argparse.ArgumentParser(description="RYS stage-2 confirmation")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="eva18b")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--n-random-configs", type=int, default=20)
    parser.add_argument("--n-test", type=int, default=500,
                        help="Fresh test images per dataset")
    parser.add_argument("--datasets", type=str, nargs="*", default=None)
    parser.add_argument("--allow-missing-datasets", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scanner, processor, spec = load_model(args.model, device=device)
    num_layers = scanner.num_layers

    top, controls = pick_configs(
        args.results_dir, num_layers, args.top_k, args.n_random_configs
    )
    configs = top + controls
    print(f"Confirming {len(top)} top + {len(controls)} control configs")

    with open(os.path.join(args.results_dir, "probe_sets.json")) as f:
        probe_sets_meta = json.load(f)

    dataset_names = args.datasets or [
        d for d in DEFAULT_DATASETS if d in probe_sets_meta
    ]
    benchmark = load_benchmark(
        args.data_dir, processor, dataset_names=dataset_names,
        image_size=spec["image_size"],
        allow_missing=args.allow_missing_datasets,
    )

    loader_kwargs = dict(batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)

    results = {"top": [list(c) for c in top],
               "controls": [list(c) for c in controls],
               "per_config": {}}
    correct_by_cfg = {cfg: {} for cfg in configs}
    bl_correct = {}
    percase_arrays = {}  # raw per-image outcomes, persisted for reanalysis
    percase_arrays["configs"] = np.array([[c[0], c[1]] for c in configs],
                                         dtype=np.int32)

    for ds_name, (train_ds, test_ds) in benchmark.items():
        meta = probe_sets_meta.get(ds_name)
        if meta is None:
            print(f"  [{ds_name}] no probe set from sweep — skipping")
            continue

        ref_indices, ref_labels_arr = resample_reference_indices(train_ds, meta)
        ref_ds = Subset(train_ds, ref_indices)

        # Fresh test sample: exclude every image the sweep touched
        labels_test = get_labels(test_ds)
        class_subset = np.array(meta["class_subset"])
        pool = np.where(np.isin(labels_test, class_subset))[0]
        pool = np.setdiff1d(pool, np.array(meta["cand_indices"]))
        rng = np.random.default_rng(777)
        n = min(args.n_test, len(pool))
        fresh_idx = rng.choice(pool, n, replace=False)
        fresh_ds = Subset(test_ds, fresh_idx.tolist())
        fresh_labels = labels_test[fresh_idx]
        print(f"\n  [{ds_name}] {len(ref_ds)} refs, {n} fresh test images")

        ref_loader = DataLoader(ref_ds, **loader_kwargs)
        fresh_loader = DataLoader(fresh_ds, **loader_kwargs)

        # Baseline
        bl_ref, _ = extract_baseline_features(scanner, ref_loader)
        bl_test, _ = extract_baseline_features(scanner, fresh_loader)
        centroids = compute_centroids(bl_ref, ref_labels_arr)
        d = score_nearest_centroid_detailed(bl_test, fresh_labels, centroids)
        bl_correct[ds_name] = d["correct"].astype(float)
        print(f"  [{ds_name}] baseline acc={d['correct'].mean():.3f}")

        # Configs (grouped to bound memory)
        ds_correct = np.full((len(configs), n), -1, dtype=np.int8)
        ds_rank = np.full((len(configs), n), -1, dtype=np.int32)
        for k in range(0, len(configs), 25):
            group = configs[k: k + 25]
            ref_feats = extract_features_for_configs(scanner, ref_loader, group)
            test_feats = extract_features_for_configs(scanner, fresh_loader, group)
            for cfg in group:
                cents = compute_centroids(ref_feats[cfg], ref_labels_arr)
                dd = score_nearest_centroid_detailed(
                    test_feats[cfg], fresh_labels, cents
                )
                correct_by_cfg[cfg][ds_name] = dd["correct"].astype(float)
                row = configs.index(cfg)
                ds_correct[row] = dd["correct"].astype(np.int8)
                ds_rank[row] = dd["rank"].astype(np.int32)
            del ref_feats, test_feats
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        percase_arrays[f"{ds_name}__correct"] = ds_correct
        percase_arrays[f"{ds_name}__rank"] = ds_rank
        percase_arrays[f"{ds_name}__bl_correct"] = d["correct"].astype(np.int8)
        percase_arrays[f"{ds_name}__bl_rank"] = d["rank"].astype(np.int32)
        percase_arrays[f"{ds_name}__fresh_indices"] = fresh_idx.astype(np.int64)
        percase_arrays[f"{ds_name}__fresh_labels"] = fresh_labels
        percase_arrays[f"{ds_name}__ref_indices"] = np.array(ref_indices,
                                                             dtype=np.int64)
        np.savez_compressed(
            os.path.join(args.results_dir, "confirm_percase.npz"),
            **percase_arrays,
        )

    # Analysis: paired bootstrap per config across datasets
    rows = []
    for cfg in configs:
        if not correct_by_cfg[cfg]:
            continue
        pt, lo, hi = paired_bootstrap_delta(correct_by_cfg[cfg], bl_correct)
        rows.append({
            "config": list(cfg), "kind": "top" if cfg in set(top) else "control",
            "delta_acc": pt, "ci_lo": lo, "ci_hi": hi,
            "per_dataset": {
                ds: float(c.mean() - bl_correct[ds].mean())
                for ds, c in correct_by_cfg[cfg].items()
            },
        })
    control_deltas = [r["delta_acc"] for r in rows if r["kind"] == "control"]
    ctrl_mu = float(np.mean(control_deltas)) if control_deltas else float("nan")
    ctrl_sd = float(np.std(control_deltas)) if control_deltas else float("nan")

    rows.sort(key=lambda r: r["delta_acc"], reverse=True)
    results["per_config"] = rows
    results["control_mean"] = ctrl_mu
    results["control_std"] = ctrl_sd

    out = os.path.join(args.results_dir, "confirm_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Control configs: mean Δacc={ctrl_mu:+.4f} (sd {ctrl_sd:.4f})")
    print(f"{'config':>10} {'kind':>8} {'Δacc':>8} {'95% CI':>20} {'beats null':>10}")
    for r in rows:
        beats = (r["ci_lo"] > 0 and
                 r["delta_acc"] > ctrl_mu + 2 * ctrl_sd)
        print(f"  {tuple(r['config'])!s:>9} {r['kind']:>8} "
              f"{r['delta_acc']:+.4f} "
              f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]   "
              f"{'YES' if beats else 'no'}")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
