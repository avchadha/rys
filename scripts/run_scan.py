"""Run the full RYS scan over all (i, j) configs for a registered ViT.

Two-phase design following the RYS blog methodology, with bias controls:

  Phase 1: Build probe sets per dataset —
    - reference images (centroids), count scaled to class count
    - BORDERLINE test set (stratified barely-wrong / barely-right, so the
      baseline sits near 50% on it and deltas can move both directions;
      selecting only hardest images guarantees fake positive deltas)
    - RANDOM control test set (unbiased deltas)
    - noise-control: Δacc under pure Gaussian feature noise (the null)
    Probe sets, baseline features, and baseline scores are persisted so
    resume never silently switches test sets.

  Phase 2: Evaluate all L*(L+1)/2 configs per dataset from cached hidden
    states, on both test splits, persisting per-image outcomes for
    bootstrap confidence intervals.

Optionally (--repeat-scan) also scans single-layer k-repeats (RYS-II).
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.suite import (
    BENCHMARK_DATASETS,
    choose_class_subset,
    get_labels,
    load_benchmark,
    select_candidate_subset,
    select_reference_subset,
)
from probes.nearest_centroid import (
    compute_centroids,
    score_nearest_centroid,
    score_nearest_centroid_detailed,
    select_borderline_images,
)
from scripts.model_registry import load_model

SPLITS = ("hard", "rand")

# ImageNet is opt-in: gated on HF, ~155GB, needs datasets<3 + HF token.
DEFAULT_DATASETS = [name for name, _ in BENCHMARK_DATASETS if name != "imagenet"]


# ── Feature extraction ──


@torch.inference_mode()
def extract_baseline_features(scanner, dataloader) -> tuple[np.ndarray, np.ndarray]:
    """Extract pooled embeddings using the unmodified model."""
    all_features, all_labels = [], []
    for images, labels in dataloader:
        emb = scanner.get_baseline_embeddings(images)
        all_features.append(emb.float().cpu())
        all_labels.append(labels)
    return torch.cat(all_features).numpy(), torch.cat(all_labels).numpy()


@torch.inference_mode()
def extract_features_for_configs(
    scanner, dataloader, configs: list[tuple]
) -> dict[tuple, np.ndarray]:
    """Extract pooled embeddings for a list of configs in one cached pass.

    Configs are (i, j) or (i, j, repeats) tuples.
    """
    accum = {cfg: [] for cfg in configs}
    for images, _labels in dataloader:
        cached = scanner.cache_baseline_states(images)
        for cfg in configs:
            i, j = cfg[0], cfg[1]
            repeats = cfg[2] if len(cfg) > 2 else 2
            emb = scanner.run_config_from_cache(i, j, cached, repeats=repeats)
            accum[cfg].append(emb.float().cpu())
        del cached
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {cfg: torch.cat(chunks).numpy() for cfg, chunks in accum.items()}


# ── Noise control ──


def noise_control(
    ref_feats, ref_labels, test_feats, test_labels,
    alphas=(0.005, 0.01, 0.02, 0.05, 0.1), n_seeds=20, seed=0,
) -> dict:
    """Δaccuracy under pure Gaussian noise applied to ref AND test features.

    This is the null hypothesis for the scan: a config whose Δacc does not
    beat matched-magnitude noise has shown nothing. alpha is the noise std
    relative to the mean feature norm (per-dimension std = alpha * ||f|| / sqrt(D)).
    """
    base = score_nearest_centroid(
        test_feats, test_labels, compute_centroids(ref_feats, ref_labels)
    )
    dim = ref_feats.shape[1]
    ref_scale = float(np.linalg.norm(ref_feats, axis=1).mean())
    test_scale = float(np.linalg.norm(test_feats, axis=1).mean())

    out = {"baseline_accuracy": base["accuracy"], "baseline_mrr": base["mrr"]}
    for alpha in alphas:
        accs, mrrs = [], []
        for s in range(n_seeds):
            rng = np.random.default_rng(seed * 100003 + s)
            r = ref_feats + rng.standard_normal(ref_feats.shape).astype(np.float32) \
                * (alpha * ref_scale / math.sqrt(dim))
            t = test_feats + rng.standard_normal(test_feats.shape).astype(np.float32) \
                * (alpha * test_scale / math.sqrt(dim))
            sc = score_nearest_centroid(t, test_labels, compute_centroids(r, ref_labels))
            accs.append(sc["accuracy"])
            mrrs.append(sc["mrr"])
        out[f"alpha_{alpha}"] = {
            "acc_delta_mean": float(np.mean(accs) - base["accuracy"]),
            "acc_delta_std": float(np.std(accs)),
            "mrr_delta_mean": float(np.mean(mrrs) - base["mrr"]),
        }
    return out


# ── Phase 1: probe sets ──


def build_or_load_probe_sets(benchmark, scanner, args, loader_kwargs):
    """Build probe sets (or reload persisted ones on --resume).

    Returns dict {ds_name: probe} where probe holds datasets, labels, and
    baseline features/scores for both splits.
    """
    sets_path = os.path.join(args.output_dir, "probe_sets.json")
    persisted = {}
    if args.resume and os.path.exists(sets_path):
        with open(sets_path) as f:
            persisted = json.load(f)
        print(f"  Loaded persisted probe sets for: {list(persisted.keys())}")

    probe_sets = {}
    probe_meta = {}
    meta_path = os.path.join(args.output_dir, "probe_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            probe_meta = json.load(f)

    for ds_name, (train_ds, test_ds) in benchmark.items():
        feats_path = os.path.join(args.output_dir, f"{ds_name}_baseline_features.npz")

        if ds_name in persisted and os.path.exists(feats_path):
            # Rebuild datasets from persisted indices (deterministic loaders)
            p = persisted[ds_name]
            ref_ds = Subset(train_ds, p["ref_indices"])
            cand_ds = Subset(test_ds, p["cand_indices"])
            hard_ds = Subset(cand_ds, p["hard_sel"])
            rand_ds = Subset(cand_ds, p["rand_sel"])
            z = np.load(feats_path)
            probe_sets[ds_name] = {
                "ref_ds": ref_ds, "ref_labels": z["ref_labels"],
                "hard_ds": hard_ds, "hard_labels": z["hard_labels"],
                "rand_ds": rand_ds, "rand_labels": z["rand_labels"],
                "baseline": {
                    s: {k: float(v) for k, v in json.loads(str(z[f"baseline_{s}"])).items()}
                    for s in SPLITS
                },
            }
            print(f"  [{ds_name}] Restored probe sets "
                  f"({len(ref_ds)} ref, {len(hard_ds)} hard, {len(rand_ds)} rand)")
            continue

        labels_train = get_labels(train_ds)
        class_subset = choose_class_subset(labels_train, args.max_classes)
        n_classes = len(class_subset)
        n_ref_pc = max(args.n_per_class, min(50, math.ceil(args.ref_budget / n_classes)))

        print(f"\n  [{ds_name}] {n_classes} classes, "
              f"{n_ref_pc} reference images per class")
        ref_ds, ref_labels = select_reference_subset(
            train_ds, n_per_class=n_ref_pc, class_subset=class_subset
        )
        cand_ds, cand_labels = select_candidate_subset(
            test_ds, n_candidates=args.n_candidates, class_subset=class_subset
        )

        print(f"  [{ds_name}] Extracting baseline features "
              f"({len(ref_ds)} ref + {len(cand_ds)} candidates)...",
              end=" ", flush=True)
        t0 = time.time()
        ref_features, _ = extract_baseline_features(
            scanner, DataLoader(ref_ds, **loader_kwargs)
        )
        cand_features, _ = extract_baseline_features(
            scanner, DataLoader(cand_ds, **loader_kwargs)
        )
        print(f"{time.time() - t0:.1f}s")

        centroids = compute_centroids(ref_features, ref_labels)
        cand_scores = score_nearest_centroid(cand_features, cand_labels, centroids)
        print(f"  [{ds_name}] Baseline on candidates: "
              f"acc={cand_scores['accuracy']:.3f}, mrr={cand_scores['mrr']:.3f}")

        # Borderline (stratified) test set + disjoint random control set
        hard_sel = select_borderline_images(
            cand_features, cand_labels, centroids, n_select=args.n_hard
        )
        remaining_pool = np.setdiff1d(np.arange(len(cand_ds)), np.array(hard_sel))
        rng = np.random.default_rng(4242)
        n_rand = min(args.n_rand, len(remaining_pool))
        rand_sel = rng.choice(remaining_pool, n_rand, replace=False).tolist()

        hard_ds = Subset(cand_ds, hard_sel)
        rand_ds = Subset(cand_ds, rand_sel)
        hard_labels = cand_labels[hard_sel]
        rand_labels = cand_labels[rand_sel]
        hard_feats = cand_features[hard_sel]
        rand_feats = cand_features[rand_sel]

        baseline = {
            "hard": score_nearest_centroid(hard_feats, hard_labels, centroids),
            "rand": score_nearest_centroid(rand_feats, rand_labels, centroids),
        }
        print(f"  [{ds_name}] Baseline hard(borderline)={baseline['hard']['accuracy']:.3f} "
              f"rand={baseline['rand']['accuracy']:.3f}")

        # Null model: Δacc under pure feature noise
        print(f"  [{ds_name}] Noise control...", end=" ", flush=True)
        nc = {
            s: noise_control(ref_features, ref_labels, f, l)
            for s, f, l in (("hard", hard_feats, hard_labels),
                            ("rand", rand_feats, rand_labels))
        }
        worst = max(v["acc_delta_mean"]
                    for s in SPLITS for k, v in nc[s].items()
                    if k.startswith("alpha_"))
        print(f"max noise Δacc={worst:+.3f}")

        np.savez_compressed(
            feats_path,
            ref=ref_features, ref_labels=ref_labels,
            hard=hard_feats, hard_labels=hard_labels,
            rand=rand_feats, rand_labels=rand_labels,
            baseline_hard=json.dumps(baseline["hard"]),
            baseline_rand=json.dumps(baseline["rand"]),
        )

        probe_sets[ds_name] = {
            "ref_ds": ref_ds, "ref_labels": ref_labels,
            "hard_ds": hard_ds, "hard_labels": hard_labels,
            "rand_ds": rand_ds, "rand_labels": rand_labels,
            "baseline": baseline,
        }
        persisted[ds_name] = {
            "ref_indices": list(map(int, ref_ds.indices)),
            "cand_indices": list(map(int, cand_ds.indices)),
            "hard_sel": list(map(int, hard_sel)),
            "rand_sel": list(map(int, rand_sel)),
            "n_ref_per_class": int(n_ref_pc),
            "class_subset": [int(c) for c in class_subset],
        }
        probe_meta[ds_name] = {
            "n_ref": len(ref_ds), "n_classes": int(n_classes),
            "n_hard": len(hard_sel), "n_rand": int(n_rand),
            "baseline_candidates": cand_scores,
            "baseline_hard": baseline["hard"],
            "baseline_rand": baseline["rand"],
            "noise_control": nc,
        }

        with open(sets_path, "w") as f:
            json.dump(persisted, f)
        with open(meta_path, "w") as f:
            json.dump(probe_meta, f, indent=2)

    return probe_sets


# ── Phase 2: config sweep ──


def valid_mask(num_layers):
    return np.array([
        [i < j for j in range(num_layers + 1)] for i in range(num_layers)
    ])


def matrix_paths(output_dir, ds_name):
    return {
        (split, metric): os.path.join(
            output_dir, f"{ds_name}_{split}_{metric}_matrix.npy"
        )
        for split in SPLITS for metric in ("accuracy", "mrr")
    }


def scan_dataset(scanner, ds_name, probe, all_configs, args, loader_kwargs):
    """Sweep all configs for one dataset, both splits, with resume."""
    num_layers = scanner.num_layers
    paths = matrix_paths(args.output_dir, ds_name)
    percase_path = os.path.join(args.output_dir, f"{ds_name}_percase.npz")
    log_path = os.path.join(args.output_dir, "scan_log.jsonl")
    mask = valid_mask(num_layers)

    matrices = {}
    for key, path in paths.items():
        if args.resume and os.path.exists(path):
            matrices[key] = np.load(path)
        else:
            matrices[key] = np.full((num_layers, num_layers + 1), np.nan)

    if all(not np.any(np.isnan(m[mask])) for m in matrices.values()):
        print(f"\n  [{ds_name}] Already complete, skipping.")
        return

    n_hard, n_rand = len(probe["hard_labels"]), len(probe["rand_labels"])
    cfg_row = {cfg: r for r, cfg in enumerate(all_configs)}
    if args.resume and os.path.exists(percase_path):
        z = np.load(percase_path)
        percase = {k: z[k] for k in z.files}
    else:
        percase = {
            "hard_correct": np.full((len(all_configs), n_hard), -1, dtype=np.int8),
            "rand_correct": np.full((len(all_configs), n_rand), -1, dtype=np.int8),
            "hard_rank": np.full((len(all_configs), n_hard), -1, dtype=np.int32),
            "rand_rank": np.full((len(all_configs), n_rand), -1, dtype=np.int32),
            "configs": np.array([[c[0], c[1]] for c in all_configs], dtype=np.int32),
        }

    ref_loader = DataLoader(probe["ref_ds"], **loader_kwargs)
    test_ds = ConcatDataset([probe["hard_ds"], probe["rand_ds"]])
    test_loader = DataLoader(test_ds, **loader_kwargs)
    ref_labels = probe["ref_labels"]

    config_groups = [
        all_configs[k: k + args.config_batch_size]
        for k in range(0, len(all_configs), args.config_batch_size)
    ]
    total = len(all_configs)

    print(f"\n  [{ds_name}] {len(probe['ref_ds'])} ref + {n_hard} hard "
          f"+ {n_rand} rand images, {len(config_groups)} config groups")

    for g_idx, group in enumerate(config_groups):
        remaining = [
            cfg for cfg in group
            if any(np.isnan(matrices[key][cfg[0], cfg[1]]) for key in matrices)
        ]
        if not remaining:
            continue

        t0 = time.time()
        ref_feats = extract_features_for_configs(scanner, ref_loader, remaining)
        test_feats = extract_features_for_configs(scanner, test_loader, remaining)

        for cfg in remaining:
            i, j = cfg
            centroids = compute_centroids(ref_feats[cfg], ref_labels)
            tf = test_feats[cfg]
            split_feats = {"hard": tf[:n_hard], "rand": tf[n_hard:]}
            record = {"dataset": ds_name, "i": int(i), "j": int(j)}
            for split in SPLITS:
                d = score_nearest_centroid_detailed(
                    split_feats[split], probe[f"{split}_labels"], centroids
                )
                acc = float(d["correct"].mean())
                mrr = float((1.0 / d["rank"]).mean())
                matrices[(split, "accuracy")][i, j] = acc
                matrices[(split, "mrr")][i, j] = mrr
                r = cfg_row[cfg]
                percase[f"{split}_correct"][r] = d["correct"].astype(np.int8)
                percase[f"{split}_rank"][r] = d["rank"].astype(np.int32)
                record[f"{split}_accuracy"] = acc
                record[f"{split}_mrr"] = mrr
            with open(log_path, "a") as f:
                f.write(json.dumps(record) + "\n")

        # percase first: matrices are the completion signal for resume, so
        # they must be the LAST write (a crash between the two would
        # otherwise mark configs complete with per-image rows still at -1)
        np.savez_compressed(percase_path, **percase)
        for key, path in paths.items():
            np.save(path, matrices[key])

        done = int(np.sum(~np.isnan(matrices[("hard", "accuracy")][mask])))
        print(f"  [{ds_name}] Group {g_idx + 1}/{len(config_groups)} "
              f"({len(remaining)} configs) — {time.time() - t0:.0f}s — "
              f"{done}/{total} done")

        del ref_feats, test_feats
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def scan_repeats(scanner, ds_name, probe, args, loader_kwargs):
    """RYS-II style single-layer k-repeat scan (k=3,4)."""
    out_path = os.path.join(args.output_dir, f"{ds_name}_repeats.json")
    results = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)

    configs = [
        cfg for cfg in scanner.single_layer_repeat_configs(k_values=(3, 4))
        if f"{cfg[0]},{cfg[2]}" not in results
    ]
    if not configs:
        print(f"  [{ds_name}] Repeat scan already complete.")
        return

    ref_loader = DataLoader(probe["ref_ds"], **loader_kwargs)
    test_ds = ConcatDataset([probe["hard_ds"], probe["rand_ds"]])
    test_loader = DataLoader(test_ds, **loader_kwargs)
    n_hard = len(probe["hard_labels"])

    groups = [
        configs[k: k + args.config_batch_size]
        for k in range(0, len(configs), args.config_batch_size)
    ]
    print(f"  [{ds_name}] Repeat scan: {len(configs)} configs, {len(groups)} groups")
    for g_idx, group in enumerate(groups):
        t0 = time.time()
        ref_feats = extract_features_for_configs(scanner, ref_loader, group)
        test_feats = extract_features_for_configs(scanner, test_loader, group)
        for cfg in group:
            i, _j, k = cfg
            centroids = compute_centroids(ref_feats[cfg], probe["ref_labels"])
            tf = test_feats[cfg]
            entry = {}
            for split, feats in (("hard", tf[:n_hard]), ("rand", tf[n_hard:])):
                entry[split] = score_nearest_centroid(
                    feats, probe[f"{split}_labels"], centroids
                )
            results[f"{i},{k}"] = entry
        with open(out_path, "w") as f:
            json.dump(results, f)
        print(f"  [{ds_name}] Repeat group {g_idx + 1}/{len(groups)} "
              f"— {time.time() - t0:.0f}s")
        del ref_feats, test_feats
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── Aggregation ──


def aggregate(probe_sets, args, num_layers):
    """Raw-mean and z-scored aggregated delta matrices per split/metric."""
    mask = valid_mask(num_layers)
    for split in SPLITS:
        for metric in ("accuracy", "mrr"):
            deltas = []
            for ds_name, probe in probe_sets.items():
                path = os.path.join(
                    args.output_dir, f"{ds_name}_{split}_{metric}_matrix.npy"
                )
                if not os.path.exists(path):
                    continue
                bl = probe["baseline"][split][metric]
                deltas.append(np.load(path) - bl)
            if not deltas:
                continue
            stack = np.stack(deltas)
            agg = np.nanmean(stack, axis=0)
            np.save(os.path.join(
                args.output_dir, f"aggregated_{split}_{metric}_delta.npy"), agg)

            # z-score each dataset's deltas across valid configs, then mean:
            # datasets with different class counts / chance levels become
            # commensurable, so binary probes cannot dominate the average.
            zstack = []
            for d in deltas:
                v = d[mask]
                mu, sd = np.nanmean(v), np.nanstd(v)
                zstack.append((d - mu) / (sd if sd > 1e-9 else 1.0))
            aggz = np.nanmean(np.stack(zstack), axis=0)
            np.save(os.path.join(
                args.output_dir, f"aggregated_{split}_{metric}_zscore.npy"), aggz)
    print("  Saved aggregated delta + z-score matrices.")


# ── Main ──


def main():
    parser = argparse.ArgumentParser(description="RYS scanner — two-split probes")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="eva18b")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--n-per-class", type=int, default=5,
                        help="Minimum reference images per class")
    parser.add_argument("--ref-budget", type=int, default=150,
                        help="Target total reference images; per-class count "
                             "= clamp(budget/classes, n-per-class, 50)")
    parser.add_argument("--n-candidates", type=int, default=1000)
    parser.add_argument("--n-hard", type=int, default=100,
                        help="Borderline test images (stratified around the "
                             "decision boundary)")
    parser.add_argument("--n-rand", type=int, default=100,
                        help="Random control test images (disjoint from hard)")
    parser.add_argument("--max-classes", type=int, default=100,
                        help="Cap classes per dataset (0 = unlimited); only "
                             "affects ImageNet/Places365")
    parser.add_argument("--config-batch-size", type=int, default=50)
    parser.add_argument("--repeat-scan", action="store_true",
                        help="Also scan single-layer k-repeats (k=3,4)")
    parser.add_argument("--allow-missing-datasets", action="store_true")
    parser.add_argument("--datasets", type=str, nargs="*", default=None,
                        help=f"Datasets to scan (default: {DEFAULT_DATASETS}; "
                             "pass 'all' to include imagenet)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    scanner, processor, spec = load_model(args.model, device=device)
    num_layers = scanner.num_layers
    all_configs = list(scanner.all_configs())
    print(f"Model: {num_layers} layers, {len(all_configs)} configs to scan.\n")

    if args.datasets is None:
        dataset_names = DEFAULT_DATASETS
    elif args.datasets == ["all"]:
        dataset_names = None
    else:
        dataset_names = args.datasets

    print("Loading benchmark datasets:")
    benchmark = load_benchmark(
        args.data_dir, processor,
        dataset_names=dataset_names, image_size=spec["image_size"],
        allow_missing=args.allow_missing_datasets,
    )
    if not benchmark:
        print("No datasets loaded. Exiting.")
        return

    loader_kwargs = dict(
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    print(f"\n{'=' * 60}\nPhase 1: Building probe sets\n{'=' * 60}")
    probe_sets = build_or_load_probe_sets(benchmark, scanner, args, loader_kwargs)

    # Persist baseline scores in the legacy location too
    baseline_results = {
        ds: probe["baseline"] for ds, probe in probe_sets.items()
    }
    with open(os.path.join(args.output_dir, "baseline_results.json"), "w") as f:
        json.dump(baseline_results, f, indent=2)

    print(f"\n{'=' * 60}\nPhase 2: Scanning all configs\n{'=' * 60}")
    # Cheapest datasets first: crash-resilient ordering
    order = sorted(
        probe_sets,
        key=lambda ds: len(probe_sets[ds]["ref_ds"])
        + len(probe_sets[ds]["hard_labels"]) + len(probe_sets[ds]["rand_labels"]),
    )
    for ds_name in order:
        scan_dataset(scanner, ds_name, probe_sets[ds_name],
                     all_configs, args, loader_kwargs)

    if args.repeat_scan:
        print(f"\n{'=' * 60}\nPhase 2b: Single-layer k-repeat scan\n{'=' * 60}")
        for ds_name in order:
            scan_repeats(scanner, ds_name, probe_sets[ds_name], args, loader_kwargs)

    print(f"\n{'=' * 60}\nAggregating results\n{'=' * 60}")
    aggregate(probe_sets, args, num_layers)

    print("\nBaseline results:")
    for ds_name, bl in baseline_results.items():
        print(f"  {ds_name}: hard(borderline) acc={bl['hard']['accuracy']:.3f}, "
              f"rand acc={bl['rand']['accuracy']:.3f}")

    agg_path = os.path.join(args.output_dir, "aggregated_rand_accuracy_delta.npy")
    if os.path.exists(agg_path):
        agg = np.load(agg_path)
        valid = agg[valid_mask(num_layers)]
        valid = valid[~np.isnan(valid)]
        if len(valid):
            print(f"\nRandom-split aggregated Δacc: best={valid.max():+.4f}, "
                  f"worst={valid.min():+.4f}, mean={valid.mean():+.4f}")

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
