"""Feature builder for the prediction model.

Coordinate convention: (row, col) = (y, x) throughout.
build_cell_features() accepts (row, col) directly.
build_feature_matrix() iterates all cells and returns [H*W, n_features].
"""
from __future__ import annotations

import numpy as np

from .masks import build_masks, connected_land_sizes, distance_map


# Class indices used in density counting
_CLASS_INDICES = list(range(6))

# Radii for density counts
_RADII = (1, 2, 3, 5)

# Feature column names (order must match build_cell_features output)
FEATURE_NAMES: list[str] = []
_scalar = [
    "row_rel", "col_rel",
    "is_coastal", "is_adjacent_forest", "is_adjacent_mountain",
    "connected_land_size_norm",
    "dist_to_coast", "dist_to_forest", "dist_to_mountain",
    "local_entropy_3x3",
]
_density = [f"class{c}_r{r}" for r in _RADII for c in _CLASS_INDICES]
_round = ["ruin_rate", "port_rate", "settlement_survival", "dynamism_rate", "settlement_density", "empty_rate"]

FEATURE_NAMES = _scalar + _density + _round


def build_cell_features(
    raw_grid: np.ndarray,
    row: int,
    col: int,
    masks: dict[str, np.ndarray] | None = None,
    dist_coast: np.ndarray | None = None,
    dist_forest: np.ndarray | None = None,
    dist_mountain: np.ndarray | None = None,
    land_sizes: np.ndarray | None = None,
    encoded_grid: np.ndarray | None = None,
    round_features: dict | None = None,
) -> np.ndarray:
    """Build feature vector for a single cell (row, col).

    Pre-computed arrays (masks, dist_*, land_sizes, encoded_grid) should be
    passed in for efficiency when calling over many cells. If None, they are
    computed on the fly (expensive).

    Returns:
        np.ndarray[float32] of shape (n_features,)
    """
    H, W = raw_grid.shape

    if masks is None:
        masks = build_masks(raw_grid)
    if encoded_grid is None:
        from .encoding import encode_grid
        encoded_grid = encode_grid(raw_grid)

    # --- Scalar features ---
    row_rel = row / max(H - 1, 1)
    col_rel = col / max(W - 1, 1)
    is_coastal = float(masks["coastal_mask"][row, col])
    adj_forest = _any_adjacent(masks["forest_mask"], row, col, H, W)
    adj_mountain = _any_adjacent(masks["mountain_mask"], row, col, H, W)

    if land_sizes is None:
        land_sizes = connected_land_sizes(raw_grid)
    connected = land_sizes[row, col] / max(H * W, 1)

    if dist_coast is None:
        dist_coast = distance_map(masks["ocean_mask"])
    if dist_forest is None:
        dist_forest = distance_map(masks["forest_mask"])
    if dist_mountain is None:
        dist_mountain = distance_map(masks["mountain_mask"])

    d_coast = float(dist_coast[row, col])
    d_forest = float(dist_forest[row, col])
    d_mountain = float(dist_mountain[row, col])

    # Local entropy of encoded terrain in 3x3 patch
    r0, r1 = max(row - 1, 0), min(row + 2, H)
    c0, c1 = max(col - 1, 0), min(col + 2, W)
    patch = encoded_grid[r0:r1, c0:c1].ravel()
    counts = np.bincount(patch.astype(np.int64), minlength=6).astype(np.float32)
    counts /= max(counts.sum(), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = float(-np.nansum(counts * np.log(np.where(counts > 0, counts, 1))))

    scalars = np.array(
        [row_rel, col_rel, is_coastal, adj_forest, adj_mountain, connected,
         d_coast, d_forest, d_mountain, ent],
        dtype=np.float32,
    )

    # --- Density features ---
    density = _density_features(encoded_grid, row, col, H, W)

    # --- Round-level features ---
    if round_features is not None:
        rf = np.array([
            round_features.get("ruin_rate", 0.0),
            round_features.get("port_rate", 0.0),
            round_features.get("settlement_survival", 1.0),
            round_features.get("dynamism_rate", 0.0),
            round_features.get("settlement_density", 0.0),
            round_features.get("empty_rate", 1.0),
        ], dtype=np.float32)
    else:
        rf = np.zeros(6, dtype=np.float32)

    return np.concatenate([scalars, density, rf])


def _any_adjacent(mask: np.ndarray, row: int, col: int, H: int, W: int) -> float:
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = row + dr, col + dc
        if 0 <= nr < H and 0 <= nc < W and mask[nr, nc]:
            return 1.0
    return 0.0


def _density_features(
    encoded: np.ndarray, row: int, col: int, H: int, W: int
) -> np.ndarray:
    feats = []
    for r in _RADII:
        r0 = max(row - r, 0); r1 = min(row + r + 1, H)
        c0 = max(col - r, 0); c1 = min(col + r + 1, W)
        patch = encoded[r0:r1, c0:c1].ravel()
        counts = np.bincount(patch.astype(np.int64), minlength=6).astype(np.float32)
        area = (r1 - r0) * (c1 - c0)
        feats.append(counts / max(area, 1))
    return np.concatenate(feats)


def build_feature_matrix(
    raw_grid: np.ndarray,
    round_features: dict | None = None,
) -> np.ndarray:
    """Build [H*W, n_features] feature matrix for all cells in one pass.

    Efficient: pre-computes all expensive maps once.
    """
    from .encoding import encode_grid
    H, W = raw_grid.shape
    masks = build_masks(raw_grid)
    enc = encode_grid(raw_grid)
    land_sizes = connected_land_sizes(raw_grid)
    dist_coast = distance_map(masks["ocean_mask"])
    dist_forest = distance_map(masks["forest_mask"])
    dist_mountain = distance_map(masks["mountain_mask"])

    n_feats = len(FEATURE_NAMES)
    matrix = np.zeros((H * W, n_feats), dtype=np.float32)
    for row in range(H):
        for col in range(W):
            idx = row * W + col
            matrix[idx] = build_cell_features(
                raw_grid, row, col,
                masks=masks,
                dist_coast=dist_coast,
                dist_forest=dist_forest,
                dist_mountain=dist_mountain,
                land_sizes=land_sizes,
                encoded_grid=enc,
                round_features=round_features,
            )
    return matrix
