"""Nearest-centroid probe for RYS experiments.

Lightweight evaluation: compute class centroids from a small reference set,
then classify test images by cosine similarity. No training required.
"""

import numpy as np


def compute_centroids(features: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    """Compute mean embedding per class.

    Args:
        features: (N, D) array of embeddings.
        labels: (N,) array of integer class labels.

    Returns:
        dict mapping label → (D,) centroid vector.
    """
    centroids = {}
    for label in np.unique(labels):
        mask = labels == label
        centroids[int(label)] = features[mask].mean(axis=0)
    return centroids


def score_nearest_centroid(
    test_features: np.ndarray,
    test_labels: np.ndarray,
    centroids: dict[int, np.ndarray],
) -> dict[str, float]:
    """Score test set via nearest-centroid classification with cosine similarity.

    Returns:
        dict with keys: accuracy, mrr, mean_correct_sim.
    """
    label_list = sorted(centroids.keys())
    label_to_idx = {l: i for i, l in enumerate(label_list)}
    centroid_matrix = np.stack([centroids[l] for l in label_list])

    eps = 1e-8
    test_norm = test_features / (np.linalg.norm(test_features, axis=1, keepdims=True) + eps)
    cent_norm = centroid_matrix / (np.linalg.norm(centroid_matrix, axis=1, keepdims=True) + eps)

    sims = test_norm @ cent_norm.T  # (N_test, N_classes)

    # Top-1 accuracy
    pred_indices = sims.argmax(axis=1)
    predictions = np.array([label_list[i] for i in pred_indices])
    accuracy = float((predictions == test_labels).mean())

    # Mean reciprocal rank
    mrr_sum = 0.0
    correct_sim_sum = 0.0
    for idx in range(len(test_labels)):
        true_idx = label_to_idx[int(test_labels[idx])]
        true_sim = sims[idx, true_idx]
        rank = int((sims[idx] >= true_sim).sum())  # 1-based rank
        mrr_sum += 1.0 / rank
        correct_sim_sum += true_sim

    mrr = mrr_sum / len(test_labels)
    mean_correct_sim = correct_sim_sum / len(test_labels)

    return {
        "accuracy": accuracy,
        "mrr": mrr,
        "mean_correct_sim": float(mean_correct_sim),
    }


def select_hard_images(
    features: np.ndarray,
    labels: np.ndarray,
    centroids: dict[int, np.ndarray],
    n_hard: int = 50,
) -> list[int]:
    """Select the hardest images by cosine similarity margin.

    Margin = sim(correct class) - sim(best incorrect class).
    Lower margin = harder image.

    Returns:
        List of indices into features/labels for the hardest images.
    """
    label_list = sorted(centroids.keys())
    label_to_idx = {l: i for i, l in enumerate(label_list)}
    centroid_matrix = np.stack([centroids[l] for l in label_list])

    eps = 1e-8
    feat_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + eps)
    cent_norm = centroid_matrix / (np.linalg.norm(centroid_matrix, axis=1, keepdims=True) + eps)

    sims = feat_norm @ cent_norm.T  # (N, C)

    margins = np.full(len(features), np.inf)
    for idx in range(len(features)):
        true_idx = label_to_idx[int(labels[idx])]
        true_sim = sims[idx, true_idx]
        other_sims = np.concatenate([sims[idx, :true_idx], sims[idx, true_idx + 1:]])
        if len(other_sims) > 0:
            margins[idx] = true_sim - other_sims.max()

    n = min(n_hard, len(features))
    return np.argsort(margins)[:n].tolist()
