"""Terrain encoding: raw API codes → 6 prediction classes.

Raw codes (from API):
    0  = Empty
    1  = Settlement
    2  = Port
    3  = Ruin
    4  = Forest
    5  = Mountain  ← static
    10 = Ocean     ← static
    11 = Plains

Prediction classes:
    0 = empty / ocean / plains  (all map to 0)
    1 = settlement
    2 = port
    3 = ruin
    4 = forest
    5 = mountain

IMPORTANT: static detection MUST use raw codes (5, 10), not encoded class,
because class 0 covers multiple raw types with different dynamic behaviour.
"""
import os
from functools import lru_cache

import numpy as np
from omegaconf import OmegaConf

RAW_TO_CLASS: dict[int, int] = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}

# Only these raw codes are truly static (never change during simulation)
STATIC_RAW_CODES: frozenset[int] = frozenset({5, 10})

# Vectorised lookup array for fast encode_grid
_LOOKUP = np.zeros(12, dtype=np.int8)
for _raw, _cls in RAW_TO_CLASS.items():
    _LOOKUP[_raw] = _cls


def encode_grid(raw_grid: list[list[int]] | np.ndarray) -> np.ndarray:
    """Convert raw terrain grid to prediction-class grid.

    Args:
        raw_grid: height x width array of raw terrain codes

    Returns:
        np.ndarray[int8] of shape (height, width) with values 0-5
    """
    arr = np.asarray(raw_grid, dtype=np.int16)
    return _LOOKUP[arr].astype(np.int8)


def is_raw_static(raw_code: int) -> bool:
    """Return True iff this raw terrain code never changes during simulation."""
    return raw_code in STATIC_RAW_CODES


def _load_priors() -> dict[int, np.ndarray]:
    """Load prior distributions from configs/base.yaml, fall back to hard-coded defaults."""
    defaults: dict[str, list[float]] = {
        "ocean_raw10":      [0.97, 0.005, 0.005, 0.005, 0.005, 0.005],
        "mountain_raw5":    [0.005, 0.005, 0.005, 0.005, 0.005, 0.97],
        "forest_raw4":      [0.10, 0.05, 0.02, 0.05, 0.72, 0.02],
        "settlement_raw1":  [0.05, 0.60, 0.10, 0.15, 0.05, 0.005],
        "port_raw2":        [0.005, 0.10, 0.75, 0.08, 0.05, 0.005],
        "ruin_raw3":        [0.10, 0.15, 0.05, 0.55, 0.10, 0.005],
        "empty_raw0":       [0.50, 0.15, 0.05, 0.10, 0.15, 0.005],
        "plains_raw11":     [0.45, 0.20, 0.05, 0.10, 0.15, 0.005],
    }
    cfg_values = defaults.copy()
    try:
        cfg_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "base.yaml"
        )
        cfg = OmegaConf.load(cfg_path)
        if "priors" in cfg:
            for key in defaults:
                if key in cfg.priors:
                    cfg_values[key] = list(cfg.priors[key])
    except Exception:
        pass

    key_to_raw = {
        "ocean_raw10": 10, "mountain_raw5": 5, "forest_raw4": 4,
        "settlement_raw1": 1, "port_raw2": 2, "ruin_raw3": 3,
        "empty_raw0": 0, "plains_raw11": 11,
    }
    out: dict[int, np.ndarray] = {}
    for key, raw in key_to_raw.items():
        p = np.array(cfg_values[key], dtype=np.float64)
        p = p / p.sum()
        out[raw] = p
    return out


@lru_cache(maxsize=1)
def _get_priors() -> dict[int, np.ndarray]:
    return _load_priors()


def class_prior(raw_code: int) -> np.ndarray:
    """Return a [6] soft prior probability vector for a given raw terrain code."""
    priors = _get_priors()
    if raw_code in priors:
        return priors[raw_code].copy()
    # Unknown code: uniform
    return np.full(6, 1.0 / 6.0, dtype=np.float64)


def build_prior_map(raw_grid: np.ndarray) -> np.ndarray:
    """Build per-cell prior probability map from raw grid.

    Args:
        raw_grid: [H, W] raw terrain codes

    Returns:
        np.ndarray[float64] of shape [H, W, 6]
    """
    H, W = raw_grid.shape
    out = np.zeros((H, W, 6), dtype=np.float64)
    priors = _get_priors()
    for raw_code, prior in priors.items():
        mask = raw_grid == raw_code
        out[mask] = prior
    # Any cell not covered by known codes → uniform
    covered = np.zeros((H, W), dtype=bool)
    for raw_code in priors:
        covered |= raw_grid == raw_code
    out[~covered] = 1.0 / 6.0
    return out
