"""NxN patch extraction from grid arrays.

Coordinate convention: (row, col) = (y, x) throughout.
Functions that accept user-facing (x, y) convert immediately.
"""
import numpy as np


def extract_patch(
    grid: np.ndarray,
    row: int,
    col: int,
    size: int = 15,
) -> np.ndarray:
    """Extract a zero-padded square patch centred at (row, col).

    Args:
        grid: [H, W] or [H, W, C] array
        row:  centre row (y)
        col:  centre column (x)
        size: patch side length (should be odd for clean centring)

    Returns:
        np.ndarray of shape (size, size) or (size, size, C), same dtype as grid
    """
    H, W = grid.shape[:2]
    half = size // 2
    r0, r1 = row - half, row - half + size
    c0, c1 = col - half, col - half + size

    if grid.ndim == 2:
        patch = np.zeros((size, size), dtype=grid.dtype)
    else:
        patch = np.zeros((size, size, grid.shape[2]), dtype=grid.dtype)

    # Clamp to grid bounds
    gr0 = max(r0, 0)
    gr1 = min(r1, H)
    gc0 = max(c0, 0)
    gc1 = min(c1, W)

    pr0 = gr0 - r0
    pr1 = pr0 + (gr1 - gr0)
    pc0 = gc0 - c0
    pc1 = pc0 + (gc1 - gc0)

    if gr0 < gr1 and gc0 < gc1:
        patch[pr0:pr1, pc0:pc1] = grid[gr0:gr1, gc0:gc1]

    return patch


def extract_patch_xy(
    grid: np.ndarray,
    x: int,
    y: int,
    size: int = 15,
) -> np.ndarray:
    """Convenience wrapper: accepts (x, y) user-facing coords, converts to (row, col)."""
    return extract_patch(grid, row=y, col=x, size=size)
