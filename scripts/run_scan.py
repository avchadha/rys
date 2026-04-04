"""Run the full RYS scan over all (i, j) configs for EVA-CLIP-18B vision encoder.

Two-phase approach matching the original RYS blog post methodology:
  Phase 1: Select small probe sets (reference images + hard test images)
  Phase 2: Evaluate all 1176 configs via nearest-centroid classification
"""

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, CLIPImageProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.suite import (
    get_labels,
    load_benchmark,
    select_candidate_subset,
    select_reference_subset,
)
from probes.nearest_centroid import (
    compute_centroids,
    score_nearest_centroid,
    select_hard_images,
)
from scanner.layer_loop import RYSScanner


# ── Feature extraction ──


def extract_baseline_features(
    scanner: RYSScanner,
    dataloader: DataLoader,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract CLS embeddings using the unmodified model."""
    all_features = []
    all_labels = []

    for images, labels in dataloader:
        emb = scanner.get_baseline_embeddings(images)
        all_features.append(emb.float().cpu())
        all_labels.append(labels)

    return (
        torch.cat(all_features).numpy(),
        torch.cat(all_labels).numpy(),
    )


def extract_features_for_configs(
    scanner: RYSScanner,
    dataloader: DataLoader,
    configs: list[tuple[int, int]],
) -> tuple[dict[tuple[int, int] | None, np.ndarray], np.ndarray]:
    """Extract CLS embeddings for baseline + a list of configs in one pass.

    Builds a cache of intermediate hidden states per batch, then runs
    each config's partial forward from the cache.

    Returns:
        features: dict mapping None (baseline) and (i,j) → (N, D) float32.
        labels: (N,) int array.
    """
    accum = {None: []}
    for cfg in configs:
        accum[cfg] = []
    label_accum = []

    for images, labels in dataloader:
        cached = scanner.cache_baseline_states(images)

        # Baseline CLS from final cached state (use scanner's pool_fn)
        accum[None].append(scanner._pool_fn(cached[-1]).float().cpu())

        # Each config's CLS from partial forward
        for cfg in configs:
            cls = scanner.run_config_from_cache(cfg[0], cfg[1], cached)
            accum[cfg].append(cls.float().cpu())

        label_accum.append(labels)
        del cached
        torch.cuda.empty_cache()

    labels = torch.cat(label_accum).numpy()
    features = {}
    for key, chunks in accum.items():
        features[key] = torch.cat(chunks).numpy()

    return features, labels


# ── Main ──


def main():
    parser = argparse.ArgumentParser(
        description="RYS Scanner — Nearest-Centroid Probes"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Root directory for dataset downloads",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--n-per-class", type=int, default=5,
        help="Reference images per class for centroid computation",
    )
    parser.add_argument(
        "--n-candidates", type=int, default=1000,
        help="Candidate pool size for hard-image selection",
    )
    parser.add_argument(
        "--n-hard", type=int, default=50,
        help="Number of hard test images per dataset",
    )
    parser.add_argument(
        "--config-batch-size", type=int, default=50,
        help="Configs to process per dataset pass (memory vs passes tradeoff)",
    )
    parser.add_argument(
        "--datasets", type=str, nargs="*", default=None,
        help="Subset of datasets (default: all). "
             "Options: imagenet dtd eurosat places365 stanford40 fgvc_aircraft "
             "counting same_different spatial_relation symmetry inside_outside",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model — EVA-CLIP-18B vision encoder (~17.5B params, 48 layers)
    model_name = "BAAI/EVA-CLIP-18B"
    image_size = 224
    print(f"Loading {model_name}...")
    full_model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    processor = CLIPImageProcessor.from_pretrained(model_name)

    # Extract vision encoder and free the text model (~9B params)
    vision_model = full_model.vision_model
    del full_model
    gc.collect()
    torch.cuda.empty_cache()

    scanner = RYSScanner(vision_model, device=device)
    num_layers = scanner.num_layers
    all_configs = list(scanner.all_configs())
    total_configs = scanner.num_configs()
    print(f"Model: {num_layers} layers, {total_configs} configs to scan.\n")

    # Load datasets
    print("Loading benchmark datasets:")
    benchmark = load_benchmark(
        args.data_dir, processor,
        dataset_names=args.datasets, image_size=image_size,
    )
    if not benchmark:
        print("No datasets loaded. Exiting.")
        return

    loader_kwargs = dict(
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ════════════════════════════════════════════════════════════════
    # Phase 1: Build probe sets
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Phase 1: Building probe sets")
    print(f"{'='*60}")

    probe_sets = {}
    probe_meta = {}

    for ds_name, (train_ds, test_ds) in benchmark.items():
        print(f"\n  [{ds_name}] Selecting reference images "
              f"({args.n_per_class}/class)...", end=" ", flush=True)
        ref_ds, ref_labels = select_reference_subset(
            train_ds, n_per_class=args.n_per_class
        )
        n_classes = len(np.unique(ref_labels))
        print(f"{len(ref_ds)} images, {n_classes} classes")

        print(f"  [{ds_name}] Selecting candidate pool "
              f"({args.n_candidates})...", end=" ", flush=True)
        cand_ds, cand_labels = select_candidate_subset(
            test_ds, n_candidates=args.n_candidates
        )
        print(f"{len(cand_ds)} images")

        # Extract baseline features for reference and candidates
        print(f"  [{ds_name}] Extracting baseline features...", end=" ", flush=True)
        t0 = time.time()
        ref_features, _ = extract_baseline_features(
            scanner, DataLoader(ref_ds, **loader_kwargs)
        )
        cand_features, _ = extract_baseline_features(
            scanner, DataLoader(cand_ds, **loader_kwargs)
        )
        print(f"{time.time() - t0:.1f}s")

        # Compute baseline centroids and find hard images
        centroids = compute_centroids(ref_features, ref_labels)
        baseline_scores = score_nearest_centroid(cand_features, cand_labels, centroids)
        print(f"  [{ds_name}] Baseline nearest-centroid on candidates: "
              f"acc={baseline_scores['accuracy']:.3f}, "
              f"mrr={baseline_scores['mrr']:.3f}")

        hard_indices = select_hard_images(
            cand_features, cand_labels, centroids, n_hard=args.n_hard
        )
        hard_ds = torch.utils.data.Subset(cand_ds, hard_indices)
        hard_labels = cand_labels[hard_indices]
        print(f"  [{ds_name}] Selected {len(hard_indices)} hard test images")

        probe_sets[ds_name] = {
            "ref_ds": ref_ds, "ref_labels": ref_labels,
            "test_ds": hard_ds, "test_labels": hard_labels,
        }
        probe_meta[ds_name] = {
            "n_ref": len(ref_ds), "n_classes": n_classes,
            "n_hard": len(hard_indices),
            "baseline_accuracy": baseline_scores["accuracy"],
            "baseline_mrr": baseline_scores["mrr"],
        }

    # Save probe metadata
    meta_path = os.path.join(args.output_dir, "probe_meta.json")
    with open(meta_path, "w") as f:
        json.dump(probe_meta, f, indent=2)

    # ════════════════════════════════════════════════════════════════
    # Phase 2: Scan all configs
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Phase 2: Scanning all configs")
    print(f"{'='*60}")

    baseline_path = os.path.join(args.output_dir, "baseline_results.json")
    if args.resume and os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline_results = json.load(f)
        print(f"  Loaded baseline results for: {list(baseline_results.keys())}")
    else:
        baseline_results = {}
    log_path = os.path.join(args.output_dir, "scan_log.jsonl")

    config_groups = [
        all_configs[k : k + args.config_batch_size]
        for k in range(0, len(all_configs), args.config_batch_size)
    ]

    for ds_name, probe in probe_sets.items():
        acc_path = os.path.join(args.output_dir, f"{ds_name}_accuracy_matrix.npy")
        mrr_path = os.path.join(args.output_dir, f"{ds_name}_mrr_matrix.npy")

        # Resume: skip fully completed datasets
        if args.resume and os.path.exists(acc_path):
            acc_matrix = np.load(acc_path)
            if not np.any(np.isnan(acc_matrix[
                np.array([[0 <= i < j <= num_layers
                           for j in range(num_layers + 1)]
                          for i in range(num_layers)])
            ])):
                print(f"\n  [{ds_name}] Already complete, skipping.")
                continue
            print(f"\n  [{ds_name}] Resuming partial scan...")
            if os.path.exists(mrr_path):
                mrr_matrix = np.load(mrr_path)
            else:
                mrr_matrix = np.full((num_layers, num_layers + 1), np.nan)
        else:
            acc_matrix = np.full((num_layers, num_layers + 1), np.nan)
            mrr_matrix = np.full((num_layers, num_layers + 1), np.nan)

        ref_loader = DataLoader(probe["ref_ds"], **loader_kwargs)
        test_loader = DataLoader(probe["test_ds"], **loader_kwargs)
        ref_labels = probe["ref_labels"]
        test_labels = probe["test_labels"]

        print(f"\n  [{ds_name}] {len(probe['ref_ds'])} ref + "
              f"{len(probe['test_ds'])} hard test images, "
              f"{len(config_groups)} config groups")

        for g_idx, config_group in enumerate(config_groups):
            # Skip configs already done (resume)
            remaining = [c for c in config_group
                         if np.isnan(acc_matrix[c[0], c[1]])]
            if not remaining:
                continue

            t0 = time.time()

            # Extract features for baseline + this group
            ref_feats, _ = extract_features_for_configs(
                scanner, ref_loader, remaining
            )
            test_feats, _ = extract_features_for_configs(
                scanner, test_loader, remaining
            )

            # Score baseline (once)
            if ds_name not in baseline_results:
                centroids_bl = compute_centroids(ref_feats[None], ref_labels)
                bl_scores = score_nearest_centroid(
                    test_feats[None], test_labels, centroids_bl
                )
                baseline_results[ds_name] = bl_scores
                print(f"  [{ds_name}] Baseline on hard set: "
                      f"acc={bl_scores['accuracy']:.3f}, "
                      f"mrr={bl_scores['mrr']:.3f}")
                # Save baseline results incrementally for resume safety
                with open(baseline_path, "w") as f:
                    json.dump(baseline_results, f, indent=2)

            # Score each config
            for cfg in remaining:
                i, j = cfg
                centroids_cfg = compute_centroids(ref_feats[cfg], ref_labels)
                scores = score_nearest_centroid(
                    test_feats[cfg], test_labels, centroids_cfg
                )
                acc_matrix[i, j] = scores["accuracy"]
                mrr_matrix[i, j] = scores["mrr"]

                with open(log_path, "a") as f:
                    f.write(json.dumps({
                        "dataset": ds_name, "i": i, "j": j,
                        "accuracy": scores["accuracy"],
                        "mrr": scores["mrr"],
                    }) + "\n")

            # Save after each group
            np.save(acc_path, acc_matrix)
            np.save(mrr_path, mrr_matrix)

            elapsed = time.time() - t0
            done = int(np.sum(~np.isnan(acc_matrix[
                np.array([[0 <= i < j <= num_layers
                           for j in range(num_layers + 1)]
                          for i in range(num_layers)])
            ])))
            print(
                f"  [{ds_name}] Group {g_idx+1}/{len(config_groups)} "
                f"({len(remaining)} configs) — {elapsed:.0f}s — "
                f"{done}/{total_configs} done"
            )

            del ref_feats, test_feats
            torch.cuda.empty_cache()

    # ════════════════════════════════════════════════════════════════
    # Aggregate
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Aggregating results")
    print(f"{'='*60}")

    # Save baseline results
    baseline_path = os.path.join(args.output_dir, "baseline_results.json")
    with open(baseline_path, "w") as f:
        json.dump(baseline_results, f, indent=2)

    # Compute aggregated delta matrices
    acc_deltas, mrr_deltas = [], []
    for ds_name in probe_sets:
        acc_path = os.path.join(args.output_dir, f"{ds_name}_accuracy_matrix.npy")
        mrr_path = os.path.join(args.output_dir, f"{ds_name}_mrr_matrix.npy")
        if not os.path.exists(acc_path):
            continue
        bl = baseline_results[ds_name]
        acc_deltas.append(np.load(acc_path) - bl["accuracy"])
        mrr_deltas.append(np.load(mrr_path) - bl["mrr"])

    if acc_deltas:
        agg_acc = np.nanmean(np.stack(acc_deltas), axis=0)
        agg_mrr = np.nanmean(np.stack(mrr_deltas), axis=0)
        np.save(os.path.join(args.output_dir, "aggregated_accuracy_delta.npy"), agg_acc)
        np.save(os.path.join(args.output_dir, "aggregated_mrr_delta.npy"), agg_mrr)

    # Summary
    print(f"\nBaseline results (on hard test sets):")
    for ds_name, bl in baseline_results.items():
        print(f"  {ds_name}: acc={bl['accuracy']:.3f}, mrr={bl['mrr']:.3f}")

    if acc_deltas:
        valid = agg_acc[~np.isnan(agg_acc)]
        print(f"\nAggregated accuracy Δ: "
              f"best={valid.max():+.4f}, worst={valid.min():+.4f}, "
              f"mean={valid.mean():+.4f}")

        # Top-5 configs
        best_configs = []
        for cfg in all_configs:
            i, j = cfg
            if not np.isnan(agg_acc[i, j]):
                best_configs.append((cfg, agg_acc[i, j]))
        best_configs.sort(key=lambda x: x[1], reverse=True)
        print("\nTop-5 configs (by aggregated accuracy Δ):")
        for (ci, cj), d in best_configs[:5]:
            print(f"  ({ci:2d}, {cj:2d}): Δacc={d:+.4f}")

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
