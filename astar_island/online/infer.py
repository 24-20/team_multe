"""Full-map inference.

Baseline path (no ML model):
    1. Start from state.posterior_mean (Bayesian blend of prior + observations)
    2. Apply static constraints from raw_grids (ocean / mountain)
    3. floor_renorm(floor=0.005)

LightGBM path (model available):
    1. Build feature matrix for all cells
    2. Blend: 0.4 * p_lgbm + 0.6 * posterior_mean
    3. Static constraints + floor_renorm
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..utils.math import floor_renorm

if TYPE_CHECKING:
    from .round_state import RoundState

# Static-cell override distributions (applied after blending)
_OCEAN_PRIOR = np.array([0.97, 0.005, 0.005, 0.005, 0.005, 0.005], dtype=np.float64)
_MOUNTAIN_PRIOR = np.array([0.005, 0.005, 0.005, 0.005, 0.005, 0.97], dtype=np.float64)


def infer_full_map(
    state: "RoundState",
    lgbm_model: Any | None = None,
    lgbm_blend_weight: float = 0.4,
    prob_floor: float = 0.005,
) -> np.ndarray:
    """Infer prediction tensor for all seeds.

    Returns:
        np.ndarray[float64] of shape [seeds, H, W, 6], floor-renormed
    """
    seeds = state.seeds_count
    H = state.height
    W = state.width

    posterior = state.posterior_mean.copy()  # [seeds, H, W, 6]

    if lgbm_model is not None:
        posterior = _blend_lgbm(state, lgbm_model, posterior, lgbm_blend_weight)

    # Apply static constraints from raw grids (NOT encoded — raw is authoritative)
    ocean_prior = _OCEAN_PRIOR
    mtn_prior = _MOUNTAIN_PRIOR
    for s in range(seeds):
        ocean_mask = state.raw_grids[s] == 10
        mtn_mask = state.raw_grids[s] == 5
        posterior[s, ocean_mask] = ocean_prior
        posterior[s, mtn_mask] = mtn_prior

    return floor_renorm(posterior, floor=prob_floor)


def _blend_lgbm(
    state: "RoundState",
    model: Any,
    posterior: np.ndarray,
    blend_weight: float,
) -> np.ndarray:
    """Blend LightGBM predictions with Bayesian posterior."""
    from ..map.features import build_feature_matrix

    seeds = state.seeds_count
    H, W = state.height, state.width

    blended = posterior.copy()
    for s in range(seeds):
        feat_matrix = build_feature_matrix(
            state.raw_grids[s],
            round_features=state.round_features(s),
        )  # [H*W, n_feats]

        try:
            p_lgbm = model.predict_proba(feat_matrix)  # [H*W, 6]
        except Exception:
            continue  # fall back to posterior-only if model fails

        if p_lgbm.shape[1] < 6:
            # Model missing some classes (e.g. ruins never seen in training).
            # Fill missing classes with the prob_floor so KL stays finite,
            # then renormalise — this keeps the model's signal for known
            # classes while avoiding zero probability on unseen classes.
            from ..utils.math import floor_renorm
            full = np.full((H * W, 6), 0.005, dtype=np.float64)
            for ci, cls in enumerate(model.classes_):
                full[:, int(cls)] = p_lgbm[:, ci]
            p_lgbm = floor_renorm(full, floor=0.005)

        p_lgbm = p_lgbm.reshape(H, W, 6)
        blended[s] = blend_weight * p_lgbm + (1.0 - blend_weight) * posterior[s]

    return blended
