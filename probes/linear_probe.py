"""Linear probe evaluation for RYS experiments."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def train_linear_probe(
    features: np.ndarray,
    labels: np.ndarray,
    max_iter: int = 1000,
    C: float = 1.0,
) -> Pipeline:
    """Train a logistic regression probe on frozen features.

    Args:
        features: (N, D) array of embeddings.
        labels: (N,) array of integer class labels.
        max_iter: Maximum iterations for solver.
        C: Inverse regularization strength.

    Returns:
        Fitted sklearn Pipeline (scaler + logistic regression).
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        # NOTE: no multi_class kwarg — deprecated in sklearn 1.5, removed in
        # 1.7; multinomial is already the default for the lbfgs solver.
        ("clf", LogisticRegression(
            max_iter=max_iter,
            C=C,
            solver="lbfgs",
            n_jobs=-1,
        )),
    ])
    pipe.fit(features, labels)
    return pipe


def evaluate_probe(
    pipe: Pipeline,
    features: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Evaluate a trained probe and return top-1 accuracy."""
    return pipe.score(features, labels)


def train_and_evaluate(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    max_iter: int = 1000,
    C: float = 1.0,
) -> float:
    """Train a linear probe and return validation accuracy.

    This is the main entry point for evaluating a single RYS config.
    """
    pipe = train_linear_probe(train_features, train_labels, max_iter=max_iter, C=C)
    return evaluate_probe(pipe, val_features, val_labels)
