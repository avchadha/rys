"""Layer-anatomy analysis: locate the ViT's encode/reason/decode phases.

Vision analog of https://dnhkng.github.io/posts/sapir-whorf/ — there, hidden
states of 8 languages x 8 topics showed LLM middle layers organize by CONTENT
(topic) while early/late layers organize by FORMAT (language), explaining why
RYS layer duplication only works mid-stack (middle layers map a shared
semantic space onto itself, so re-entering them is a near-distribution
operation).

Here we use an 8-shape x 8-rendering-style grid (benchmarks/style_content.py)
and measure, at every layer, per-layer-centered pairwise cosine similarity
for three pair categories:

  - same content, different style  ("same topic, different language")
  - same style, different content  ("same language, different topic")
  - different content and style

Layers where the same-content curve dominates are the candidate "reasoning"
phase where duplication should be tolerated; this prediction is then testable
against the RYS scan heatmap.
"""

import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Feature extraction ──


@torch.no_grad()
def extract_layerwise_features(scanner, dataloader, pool="mean_patch"):
    """Pooled representation of every image at every layer.

    Args:
        scanner: RYSScanner (used for its cached forward).
        dataloader: yields (images, labels); iterated in order.
        pool: 'mean_patch' (mean over non-CLS tokens, as in the blog post's
              mean pooling over content tokens) or 'cls'.

    Returns:
        (num_layers + 1, N, D) float32 numpy array; index 0 is the
        embedding output, index k is the state after block k-1.
    """
    # Patch tokens start after the model's prefix tokens (CLS for EVA;
    # CLS + 4 register tokens for DINOv3) — set by the model registry.
    n_prefix = getattr(scanner, "num_prefix_tokens", 1)
    per_layer = None
    for images, _ in dataloader:
        cached = scanner.cache_baseline_states(images)
        if per_layer is None:
            per_layer = [[] for _ in cached]
        for k, state in enumerate(cached):
            if pool == "mean_patch":
                pooled = state[:, n_prefix:].mean(dim=1)
            elif pool == "cls":
                pooled = state[:, 0]
            else:
                raise ValueError(f"Unknown pool: {pool}")
            per_layer[k].append(pooled.float().cpu())
        del cached
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.stack([torch.cat(chunks).numpy() for chunks in per_layer])


# ── Similarity structure ──


def pair_categories(content_labels, style_labels):
    """Index arrays (iu, ju) and a category id per pair.

    Categories: 0 = same content / diff style, 1 = same style / diff content,
    2 = diff content / diff style, 3 = same content AND same style
    (different jitter instances — a ceiling reference, excluded from the
    content-vs-style comparison).
    """
    n = len(content_labels)
    iu, ju = np.triu_indices(n, k=1)
    same_c = content_labels[iu] == content_labels[ju]
    same_s = style_labels[iu] == style_labels[ju]
    cat = np.full(len(iu), 2)
    cat[same_c & ~same_s] = 0
    cat[~same_c & same_s] = 1
    cat[same_c & same_s] = 3
    return iu, ju, cat


def centered_similarity_curves(features, content_labels, style_labels):
    """Per-layer mean centered cosine similarity for each pair category.

    Args:
        features: (L+1, N, D) array from extract_layerwise_features.

    Returns:
        dict with 'same_content', 'same_style', 'different' — each (L+1,)
        lists — plus 'reasoning_layers': layers where content > style.
    """
    iu, ju, cat = pair_categories(
        np.asarray(content_labels), np.asarray(style_labels)
    )

    has_same_both = bool((cat == 3).any())
    cats = (0, 1, 2, 3) if has_same_both else (0, 1, 2)
    curves = {c: [] for c in cats}
    for layer_feats in features:
        norms = np.linalg.norm(layer_feats, axis=1, keepdims=True)
        normed = layer_feats / (norms + 1e-8)
        sims = (normed[iu] * normed[ju]).sum(axis=1)
        sims = sims - sims.mean()  # per-layer centering
        for c in cats:
            curves[c].append(float(sims[cat == c].mean()))

    same_content = np.array(curves[0])
    same_style = np.array(curves[1])
    reasoning = np.where(same_content > same_style)[0]

    out = {
        "same_content": curves[0],
        "same_style": curves[1],
        "different": curves[2],
        "reasoning_layers": reasoning.tolist(),
    }
    if has_same_both:
        out["same_both"] = curves[3]
    return out


def plot_anatomy(curves, output_path, title="ViT layer anatomy"):
    """Three-curve plot with the content-dominant region shaded."""
    n_points = len(curves["same_content"])
    x = np.arange(n_points)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, curves["same_content"], color="tab:red", lw=2,
            label="same content, different style")
    ax.plot(x, curves["same_style"], color="tab:green", lw=2,
            label="same style, different content")
    ax.plot(x, curves["different"], color="tab:gray", lw=1.5, ls="--",
            label="different content and style")
    if "same_both" in curves:
        ax.plot(x, curves["same_both"], color="tab:purple", lw=1, ls=":",
                label="same content and style (ceiling)")

    reasoning = curves.get("reasoning_layers", [])
    if reasoning:
        ax.axvspan(min(reasoning) - 0.5, max(reasoning) + 0.5,
                   color="tab:red", alpha=0.08,
                   label="content-dominant (candidate reasoning phase)")

    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("layer (0 = embeddings, k = after block k-1)")
    ax.set_ylabel("centered mean cosine similarity")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_pca_grid(features, content_labels, style_labels, output_path,
                  n_panels=6):
    """PCA scatter of selected layers, colored by content, marker by style."""
    L = features.shape[0]
    layer_ids = np.unique(np.linspace(0, L - 1, n_panels).astype(int))
    markers = ["o", "s", "^", "v", "D", "P", "X", "*"]
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(1, len(layer_ids),
                             figsize=(4 * len(layer_ids), 4.2))
    for ax, lid in zip(np.atleast_1d(axes), layer_ids):
        feats = features[lid]
        feats = feats - feats.mean(axis=0)
        # PCA via SVD (no sklearn dependency here)
        _, _, vt = np.linalg.svd(feats, full_matrices=False)
        xy = feats @ vt[:2].T
        for c in np.unique(content_labels):
            for s in np.unique(style_labels):
                m = (content_labels == c) & (style_labels == s)
                ax.scatter(xy[m, 0], xy[m, 1], color=cmap(int(c) % 10),
                           marker=markers[int(s) % len(markers)], s=36,
                           edgecolors="none", alpha=0.85)
        ax.set_title(f"layer {lid}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("PCA per layer — color = content (shape), marker = style")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def save_anatomy_results(curves_by_pool, output_dir):
    path = os.path.join(output_dir, "anatomy.json")
    with open(path, "w") as f:
        json.dump(curves_by_pool, f, indent=2)
    print(f"  Saved: {path}")
