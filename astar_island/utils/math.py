import numpy as np


def floor_renorm(arr: np.ndarray, floor: float = 0.005) -> np.ndarray:
    """Apply minimum floor to all class probabilities and renormalise to sum=1.

    Works on any shape [..., 6] (or any last-dim).
    """
    arr = np.maximum(arr, floor)
    return arr / arr.sum(axis=-1, keepdims=True)


def entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Shannon entropy (nats). Shape of probs: [..., n_classes] → returns [...]."""
    p = np.clip(probs, eps, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


def kl_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL divergence KL(p || q). Both 1-D arrays of length n_classes.

    Returns inf if q has a zero where p > eps.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    mask = p > eps
    if np.any((q < eps) & mask):
        return float("inf")
    terms = np.where(mask, p * np.log(p / np.maximum(q, eps)), 0.0)
    return float(np.sum(terms))


def weighted_kl_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Competition metric: entropy-weighted KL divergence.

    Args:
        pred: [H, W, 6] predicted probabilities
        gt:   [H, W, 6] ground truth probabilities

    Returns:
        score in [0, 100] — higher is better
    """
    H, W, _ = pred.shape
    total_weight = 0.0
    total_wkl = 0.0
    for row in range(H):
        for col in range(W):
            p = gt[row, col]
            q = pred[row, col]
            h = float(entropy(p))
            if h < 1e-8:
                continue  # static cell, excluded
            kl = kl_div(p, q)
            total_wkl += h * kl
            total_weight += h
    if total_weight < 1e-12:
        return 100.0
    wkl = total_wkl / total_weight
    import math
    return max(0.0, min(100.0, 100.0 * math.exp(-3.0 * wkl)))
