"""Central validation for tensors, viewports, and API response shapes.

All validation logic lives here — never spread across callers.
"""
import numpy as np


def validate_prediction_tensor(tensor: np.ndarray, height: int, width: int) -> None:
    """Raise ValueError with a descriptive message if the tensor is invalid.

    Checks performed (in order):
      1. Shape == (height, width, 6)
      2. No NaN or Inf values
      3. All values >= 0
      4. Each cell sums to ~1.0 (atol=1e-4)
      5. No exact zeros (floor must have been applied)
    """
    if tensor.shape != (height, width, 6):
        raise ValueError(
            f"Tensor shape {tensor.shape} != expected ({height}, {width}, 6)"
        )
    if np.any(np.isnan(tensor)):
        raise ValueError("Tensor contains NaN values")
    if np.any(np.isinf(tensor)):
        raise ValueError("Tensor contains Inf values")
    if np.any(tensor < 0):
        bad = np.argwhere(tensor < 0)
        raise ValueError(f"Tensor has {len(bad)} negative values (first: {bad[0]})")
    sums = tensor.sum(axis=-1)
    if not np.allclose(sums, 1.0, atol=1e-4):
        bad = np.argwhere(np.abs(sums - 1.0) > 1e-4)
        raise ValueError(
            f"Tensor has {len(bad)} cells not summing to 1.0 "
            f"(first: row={bad[0][0]}, col={bad[0][1]}, sum={sums[bad[0][0], bad[0][1]]:.6f})"
        )
    if np.any(tensor == 0.0):
        zeros = np.argwhere(tensor == 0.0)
        raise ValueError(
            f"Tensor has {len(zeros)} exact zeros — apply floor_renorm before submitting"
        )


def validate_viewport(
    x: int, y: int, w: int, h: int, map_width: int, map_height: int
) -> None:
    """Raise ValueError if viewport coordinates are invalid."""
    if not (1 <= w <= 15):
        raise ValueError(f"Viewport width {w} out of range [1, 15]")
    if not (1 <= h <= 15):
        raise ValueError(f"Viewport height {h} out of range [1, 15]")
    if x < 0 or x + w > map_width:
        raise ValueError(
            f"Viewport x={x}, w={w} exceeds map width {map_width}"
        )
    if y < 0 or y + h > map_height:
        raise ValueError(
            f"Viewport y={y}, h={h} exceeds map height {map_height}"
        )


def validate_round_detail(detail: dict) -> None:
    """Raise ValueError if required keys are missing from round detail response."""
    required = ["id", "map_width", "map_height", "seeds_count", "initial_states", "status"]
    missing = [k for k in required if k not in detail]
    if missing:
        raise ValueError(f"Round detail missing required keys: {missing}")
    if not isinstance(detail["initial_states"], list):
        raise ValueError("initial_states must be a list")
    if len(detail["initial_states"]) != detail["seeds_count"]:
        raise ValueError(
            f"initial_states length {len(detail['initial_states'])} "
            f"!= seeds_count {detail['seeds_count']}"
        )
    for i, state in enumerate(detail["initial_states"]):
        if "grid" not in state:
            raise ValueError(f"initial_states[{i}] missing 'grid'")
        if "settlements" not in state:
            raise ValueError(f"initial_states[{i}] missing 'settlements'")
