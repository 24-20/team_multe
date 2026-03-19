"""Local implementation of the official Astar Island scoring formula.

From scoring docs:
    KL(p || q)   = Σ pᵢ × log(pᵢ / qᵢ)
    entropy(cell) = -Σ pᵢ × log(pᵢ)
    weighted_kl  = Σ entropy(cell) × KL(p[cell], q[cell])
                   ──────────────────────────────────────
                   Σ entropy(cell)
    score        = max(0, min(100, 100 × exp(-3 × weighted_kl)))

Key pitfalls encoded here:
- Never assign q = 0.  The KL term pᵢ × log(pᵢ / 0) = +∞.
  We raise ValueError if any q = 0 is found.
- Static cells have entropy ≈ 0 and are excluded from scoring
  (they receive negligible weight).
- If no dynamic cells exist, score defaults to 100 (perfect on a static map).
"""
from __future__ import annotations

import math
import logging

import numpy as np

logger = logging.getLogger(__name__)

_EPS = 1e-12  # numerical guard for log(0) in entropy calculation


def entropy(p: np.ndarray) -> np.ndarray:
    """Shannon entropy along last axis.  Shape [..., K] → [...]."""
    safe_p = np.where(p > _EPS, p, _EPS)
    return -(p * np.log(safe_p)).sum(axis=-1)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """KL(p || q) along last axis.  Shape [..., K] → [...].

    Raises ValueError if any q == 0 where p > 0 (infinite divergence).
    """
    if np.any((q == 0) & (p > 0)):
        raise ValueError(
            "Prediction q contains 0.0 where ground truth p > 0. "
            "KL divergence is infinite. Apply probability floor first."
        )
    # Where p == 0, the term is 0 regardless of q (0 × log(0/q) = 0).
    mask = p > _EPS
    result = np.zeros(p.shape[:-1], dtype=np.float64)
    # Compute only where p > 0
    safe_q = np.where(q > _EPS, q, _EPS)
    kl_terms = np.where(mask, p * np.log(p / safe_q), 0.0)
    result = kl_terms.sum(axis=-1)
    return result


def compute_cell_score(p_cell: np.ndarray, q_cell: np.ndarray) -> float:
    """Score a single cell: entropy-weighted KL contribution.

    Returns (entropy_weight, kl) for use in aggregation.
    """
    h = float(entropy(p_cell))
    kl = float(kl_divergence(p_cell.reshape(1, -1), q_cell.reshape(1, -1))[0])
    return h, kl


def compute_seed_score(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> float:
    """Compute the score [0, 100] for one seed.

    Args:
        ground_truth: [H, W, 6] true distribution (each cell sums to 1)
        prediction:   [H, W, 6] predicted distribution (each cell sums to 1)
    """
    assert ground_truth.shape == prediction.shape, (
        f"Shape mismatch: {ground_truth.shape} vs {prediction.shape}"
    )
    assert ground_truth.ndim == 3 and ground_truth.shape[-1] == 6

    h = entropy(ground_truth)           # [H, W]
    kl = kl_divergence(ground_truth, prediction)  # [H, W]

    total_entropy = h.sum()
    if total_entropy < _EPS:
        # Fully static map — nothing to predict
        return 100.0

    weighted_kl = (h * kl).sum() / total_entropy
    score = 100.0 * math.exp(-3.0 * weighted_kl)
    return float(np.clip(score, 0.0, 100.0))


def compute_round_score(seed_scores: list[float]) -> float:
    """Average per-seed scores into a round score.

    Missing seeds (None) are treated as 0, matching the competition rule
    that not submitting for a seed scores 0.
    """
    scores = [s if s is not None else 0.0 for s in seed_scores]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def local_score_report(
    ground_truth_per_seed: list[np.ndarray],
    predictions_per_seed: list[np.ndarray],
) -> dict:
    """Compute a full diagnostic report for a round.

    Returns:
        {
            "seed_scores": [...],
            "round_score": float,
            "per_seed_details": [{"weighted_kl": ..., "dynamic_cell_count": ...}, ...]
        }
    """
    seed_scores = []
    details = []

    for i, (gt, pred) in enumerate(zip(ground_truth_per_seed, predictions_per_seed)):
        h = entropy(gt)
        kl = kl_divergence(gt, pred)
        dynamic = int((h > 0.01).sum())
        total_entropy = float(h.sum())
        weighted_kl = float((h * kl).sum() / total_entropy) if total_entropy > _EPS else 0.0
        score = float(np.clip(100.0 * math.exp(-3.0 * weighted_kl), 0.0, 100.0))
        seed_scores.append(score)
        details.append({
            "seed": i,
            "score": score,
            "weighted_kl": weighted_kl,
            "dynamic_cell_count": dynamic,
            "total_entropy": total_entropy,
        })
        logger.info(
            "Seed %d: score=%.2f  weighted_kl=%.4f  dynamic_cells=%d",
            i, score, weighted_kl, dynamic,
        )

    round_score = compute_round_score(seed_scores)
    logger.info("Round score: %.2f", round_score)

    return {
        "seed_scores": seed_scores,
        "round_score": round_score,
        "per_seed_details": details,
    }


def validate_prediction_tensor(pred: np.ndarray, min_floor: float = 0.001) -> list[str]:
    """Return a list of validation errors (empty = valid).

    Checks:
    - Shape is (H, W, 6)
    - All values are finite
    - No zeros
    - Each cell sums to 1.0 ± 1e-4
    """
    errors: list[str] = []
    if pred.ndim != 3 or pred.shape[-1] != 6:
        errors.append(f"Shape must be (H, W, 6), got {pred.shape}")
        return errors  # can't check further
    if not np.isfinite(pred).all():
        errors.append("Prediction contains non-finite values (inf or nan)")
    if (pred <= 0).any():
        n = int((pred <= 0).sum())
        errors.append(f"Prediction has {n} cells with value <= 0 (will cause infinite KL)")
    if (pred < min_floor).any():
        n = int((pred < min_floor).sum())
        errors.append(f"Prediction has {n} values below floor {min_floor} (risky)")
    row_sums = pred.sum(axis=-1)
    off = np.abs(row_sums - 1.0) > 1e-4
    if off.any():
        errors.append(f"{int(off.sum())} cells do not sum to 1.0")
    return errors
