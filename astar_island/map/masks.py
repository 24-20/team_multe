"""Build boolean masks from the RAW terrain grid.

All masks are derived from raw codes, not encoded prediction classes,
because class 0 conflates ocean, plains, and empty which have different
static/dynamic behaviour.
"""
import numpy as np
from scipy.ndimage import label as _label


def build_masks(raw_grid: np.ndarray) -> dict[str, np.ndarray]:
    """Compute all terrain masks from the raw grid.

    Args:
        raw_grid: [H, W] int array of raw terrain codes

    Returns:
        dict of bool arrays, each [H, W]:
            ocean_mask       — raw == 10
            mountain_mask    — raw == 5
            static_mask      — ocean | mountain
            plains_mask      — raw == 11
            empty_mask       — raw == 0
            forest_mask      — raw == 4
            settlement_mask  — raw == 1
            port_mask        — raw == 2
            ruin_mask        — raw == 3
            water_mask       — ocean (alias for ocean_mask)
            land_mask        — ~ocean
            buildable_mask   — land & ~mountain
            coastal_mask     — land cell adjacent (4-connected) to ocean
    """
    r = np.asarray(raw_grid, dtype=np.int16)

    ocean   = r == 10
    mtn     = r == 5
    plains  = r == 11
    empty   = r == 0
    forest  = r == 4
    sett    = r == 1
    port    = r == 2
    ruin    = r == 3

    land = ~ocean
    buildable = land & ~mtn

    # Coastal: land cell with at least one ocean neighbour (4-connected)
    # Pad ocean map by 1, then check adjacency
    H, W = r.shape
    coastal = np.zeros((H, W), dtype=bool)
    coastal[:-1, :] |= ocean[1:, :]   # cell above ocean
    coastal[1:,  :] |= ocean[:-1, :]  # cell below ocean
    coastal[:,  :-1] |= ocean[:, 1:]  # cell right of ocean
    coastal[:,   1:] |= ocean[:, :-1] # cell left of ocean
    coastal &= land  # only land cells can be coastal

    return {
        "ocean_mask":      ocean,
        "mountain_mask":   mtn,
        "static_mask":     ocean | mtn,
        "plains_mask":     plains,
        "empty_mask":      empty,
        "forest_mask":     forest,
        "settlement_mask": sett,
        "port_mask":       port,
        "ruin_mask":       ruin,
        "water_mask":      ocean,
        "land_mask":       land,
        "buildable_mask":  buildable,
        "coastal_mask":    coastal,
    }


def connected_land_sizes(raw_grid: np.ndarray) -> np.ndarray:
    """Return a [H, W] array where each land cell holds the size of its connected component.

    Ocean cells are 0.
    """
    land = (np.asarray(raw_grid, dtype=np.int16) != 10).astype(np.int32)
    labeled, n_components = _label(land)
    if n_components == 0:
        return np.zeros_like(land, dtype=np.int32)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # background
    return sizes[labeled]


def distance_map(mask: np.ndarray) -> np.ndarray:
    """Return Manhattan-distance map to nearest True cell in mask.

    Uses BFS for correctness. Returns float32 [H, W], normalised to [0, 1]
    by dividing by max(H, W).
    """
    from collections import deque
    H, W = mask.shape
    dist = np.full((H, W), np.inf, dtype=np.float32)
    queue: deque = deque()
    for r in range(H):
        for c in range(W):
            if mask[r, c]:
                dist[r, c] = 0.0
                queue.append((r, c))
    while queue:
        row, col = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + dr, col + dc
            if 0 <= nr < H and 0 <= nc < W and dist[nr, nc] == np.inf:
                dist[nr, nc] = dist[row, col] + 1.0
                queue.append((nr, nc))
    max_dist = max(H, W)
    return np.clip(dist / max_dist, 0.0, 1.0)
