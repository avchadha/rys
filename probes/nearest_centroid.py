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
        dict mapping label -> (D,) centroid vector.
    """
    centroids = {}
    for label in np.unique(labels):
        mask = labels == label
        centroids[int(label)] = features[mask].mean(axis=0)
    return centroids


def _similarity_matrix(features, centroids):
    """Cosine similarity of each feature to each centroid.

    Returns (sims, label_list, label_to_idx)."""
    label_list = sorted(centroids.keys())
    label_to_idx = {l: i for i, l in enumerate(label_list)}
    centroid_matrix = np.stack([centroids[l] for l in label_list])

    eps = 1e-8
    feat_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + eps)
    cent_norm = centroid_matrix / (
        np.linalg.norm(centroid_matrix, axis=1, keepdims=True) + eps
    )
    return feat_norm @ cent_norm.T, label_list, label_to_idx


def _check_labels_covered(labels, label_to_idx):
    missing = set(int(l) for l in labels) - set(label_to_idx)
    if missing:
        raise ValueError(
            f"Test labels {sorted(missing)} have no reference centroid; "
            "filter the test set to centroid classes first."
        )


def score_nearest_centroid_detailed(
    test_features: np.ndarray,
    test_labels: np.ndarray,
    centroids: dict[int, np.ndarray],
) -> dict:
    """Per-image nearest-centroid outcomes.

    Returns dict with:
        correct: (N,) bool — top-1 correct
        rank: (N,) int — 1-based rank of the true class (ties pessimistic)
        margin: (N,) float — sim(true) - max sim(other)
        true_sim: (N,) float
    """
    sims, label_list, label_to_idx = _similarity_matrix(test_features, centroids)
    _check_labels_covered(test_labels, label_to_idx)

    n = len(test_labels)
    true_idx = np.array([label_to_idx[int(l)] for l in test_labels])
    true_sim = sims[np.arange(n), true_idx]

    # Pessimistic 1-based rank: ties count against the true class
    rank = (sims >= true_sim[:, None]).sum(axis=1)

    other = sims.copy()
    other[np.arange(n), true_idx] = -np.inf
    margin = true_sim - other.max(axis=1)

    pred_idx = sims.argmax(axis=1)
    correct = pred_idx == true_idx

    return {
        "correct": correct,
        "rank": rank,
        "margin": margin,
        "true_sim": true_sim,
    }


def score_nearest_centroid(
    test_features: np.ndarray,
    test_labels: np.ndarray,
    centroids: dict[int, np.ndarray],
) -> dict[str, float]:
    """Aggregate nearest-centroid scores.

    Returns:
        dict with keys: accuracy, mrr, mean_correct_sim.
    """
    d = score_nearest_centroid_detailed(test_features, test_labels, centroids)
    return {
        "accuracy": float(d["correct"].mean()),
        "mrr": float((1.0 / d["rank"]).mean()),
        "mean_correct_sim": float(d["true_sim"].mean()),
    }


def select_borderline_images(
    features: np.ndarray,
    labels: np.ndarray,
    centroids: dict[int, np.ndarray],
    n_select: int = 100,
) -> list[int]:
    """Select borderline images, stratified around the decision boundary.

    Half are the barely-WRONG images (misclassified, margin closest to 0)
    and half the barely-RIGHT (correct, margin closest to 0). This keeps the
    baseline near 50% accuracy on the selected set so configs can move the
    score in BOTH directions.

    Selecting only the most-confidently-wrong images (the previous approach)
    puts the baseline at ~0% on the selected set, so any perturbation —
    including pure noise — scores a positive delta (regression to the mean).

    Returns:
        List of indices into features/labels.
    """
    d = score_nearest_centroid_detailed(features, labels, centroids)
    margin = d["margin"]

    wrong = np.where(margin < 0)[0]
    right = np.where(margin >= 0)[0]

    n_half = n_select // 2
    # closest to the boundary on each side
    wrong_sorted = wrong[np.argsort(-margin[wrong])]   # least negative first
    right_sorted = right[np.argsort(margin[right])]    # least positive first

    take_wrong = wrong_sorted[:n_half]
    take_right = right_sorted[: n_select - len(take_wrong)]
    selected = np.concatenate([take_wrong, take_right])

    # If the right side ran short, top up from the remaining wrong side
    if len(selected) < n_select:
        extra = wrong_sorted[len(take_wrong): len(take_wrong) + n_select - len(selected)]
        selected = np.concatenate([selected, extra])

    return selected[: min(n_select, len(features))].astype(int).tolist()


def select_hard_images(
    features: np.ndarray,
    labels: np.ndarray,
    centroids: dict[int, np.ndarray],
    n_hard: int = 50,
) -> list[int]:
    """DEPRECATED: lowest-margin selection; kept for backwards compatibility.

    Biased — see select_borderline_images. Use that instead.
    """
    d = score_nearest_centroid_detailed(features, labels, centroids)
    n = min(n_hard, len(features))
    return np.argsort(d["margin"])[:n].tolist()
