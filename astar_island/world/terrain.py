"""Deterministic world reconstruction from round details.

Terrain code → prediction class mapping (from API docs):
  10 = Ocean    → class 0 (Empty)
  11 = Plains   → class 0 (Empty)
   0 = Empty    → class 0 (Empty)
   1 = Settlement → class 1
   2 = Port     → class 2
   3 = Ruin     → class 3
   4 = Forest   → class 4
   5 = Mountain → class 5

Static cells (never change): Ocean (10), Mountain (5).
Mostly static: Forest (4) — can be reclaimed by nearby thriving settlements.
Dynamic: Settlement (1), Port (2), Ruin (3), Plains/Empty (0/11).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..api.client import Round, InitialState


# --- Terrain code constants ---
class TerrainCode:
    EMPTY = 0
    SETTLEMENT = 1
    PORT = 2
    RUIN = 3
    FOREST = 4
    MOUNTAIN = 5
    OCEAN = 10
    PLAINS = 11


# Prediction class 0 includes ocean, plains, and empty
TERRAIN_TO_CLASS: dict[int, int] = {
    TerrainCode.OCEAN: 0,
    TerrainCode.PLAINS: 0,
    TerrainCode.EMPTY: 0,
    TerrainCode.SETTLEMENT: 1,
    TerrainCode.PORT: 2,
    TerrainCode.RUIN: 3,
    TerrainCode.FOREST: 4,
    TerrainCode.MOUNTAIN: 5,
}

CLASS_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]

# 4-directional neighbors
_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
# 8-directional neighbors (including diagonals)
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _terrain_to_class_array(grid: list[list[int]]) -> np.ndarray:
    """Convert a 2D terrain-code grid to a 2D class-index array."""
    arr = np.array(grid, dtype=np.int32)
    out = np.zeros_like(arr)
    for code, cls in TERRAIN_TO_CLASS.items():
        out[arr == code] = cls
    return out


@dataclass
class SeedWorld:
    """All deterministic information for one seed."""
    seed_index: int
    height: int
    width: int

    # Raw terrain codes [height, width]
    terrain: np.ndarray        # dtype int32

    # Prediction class per cell [height, width]
    initial_class: np.ndarray  # dtype int32

    # Boolean masks [height, width]
    ocean_mask: np.ndarray     # True where ocean (code 10)
    mountain_mask: np.ndarray  # True where mountain (code 5)
    static_mask: np.ndarray    # True where cell cannot change (ocean or mountain)
    coast_mask: np.ndarray     # True where land cell adjacent to ocean
    land_mask: np.ndarray      # True where not ocean

    # Settlement positions from initial state
    settlement_positions: list[tuple[int, int]]   # (x, y)
    port_positions: list[tuple[int, int]]          # (x, y) of initial ports

    # Distance from nearest initial settlement (BFS), inf for unreachable
    settlement_distance: np.ndarray  # dtype float32, shape [height, width]

    # Number of initial settlements within radius-3 neighborhood of each cell
    local_settlement_density: np.ndarray  # dtype int32, shape [height, width]

    def cell_terrain(self, x: int, y: int) -> int:
        return int(self.terrain[y, x])

    def cell_class(self, x: int, y: int) -> int:
        return int(self.initial_class[y, x])

    def is_coastal(self, x: int, y: int) -> bool:
        return bool(self.coast_mask[y, x])

    def dynamic_cells(self) -> list[tuple[int, int]]:
        """Return (x,y) positions of all non-static cells."""
        ys, xs = np.where(~self.static_mask)
        return list(zip(xs.tolist(), ys.tolist()))


@dataclass
class WorldState:
    """All deterministic world information for a round (all seeds)."""
    round_id: str
    map_width: int
    map_height: int
    seeds_count: int
    seeds: list[SeedWorld]

    @classmethod
    def from_round(cls, round_obj: Round) -> "WorldState":
        seeds = [
            _build_seed_world(i, state, round_obj.map_width, round_obj.map_height)
            for i, state in enumerate(round_obj.initial_states)
        ]
        return cls(
            round_id=round_obj.id,
            map_width=round_obj.map_width,
            map_height=round_obj.map_height,
            seeds_count=round_obj.seeds_count,
            seeds=seeds,
        )

    def seed(self, idx: int) -> SeedWorld:
        return self.seeds[idx]


def _build_seed_world(
    seed_index: int,
    state: InitialState,
    width: int,
    height: int,
) -> SeedWorld:
    terrain = np.array(state.grid, dtype=np.int32)  # [height, width]
    assert terrain.shape == (height, width), (
        f"Terrain shape {terrain.shape} != expected ({height}, {width})"
    )

    initial_class = _terrain_to_class_array(state.grid)

    ocean_mask = terrain == TerrainCode.OCEAN
    mountain_mask = terrain == TerrainCode.MOUNTAIN
    static_mask = ocean_mask | mountain_mask
    land_mask = ~ocean_mask

    # Coast: land cell adjacent (4-dir) to ocean
    coast_mask = _compute_coast(ocean_mask, height, width)

    # Settlement/port positions from initial state
    settlement_positions = []
    port_positions = []
    for s in state.settlements:
        if s.alive:
            settlement_positions.append((s.x, s.y))
            if s.has_port:
                port_positions.append((s.x, s.y))

    # BFS distance from initial settlements
    settlement_distance = _bfs_distance(settlement_positions, height, width, land_mask)

    # Local settlement density within radius 3 (Chebyshev distance)
    local_settlement_density = _local_settlement_density(settlement_positions, height, width, radius=3)

    return SeedWorld(
        seed_index=seed_index,
        height=height,
        width=width,
        terrain=terrain,
        initial_class=initial_class,
        ocean_mask=ocean_mask,
        mountain_mask=mountain_mask,
        static_mask=static_mask,
        coast_mask=coast_mask,
        land_mask=land_mask,
        settlement_positions=settlement_positions,
        port_positions=port_positions,
        settlement_distance=settlement_distance,
        local_settlement_density=local_settlement_density,
    )


def _compute_coast(ocean_mask: np.ndarray, height: int, width: int) -> np.ndarray:
    coast = np.zeros((height, width), dtype=bool)
    for dy, dx in _DIRS4:
        shifted = np.roll(ocean_mask, shift=(dy, dx), axis=(0, 1))
        # Zero out wraparound edges
        if dy > 0:
            shifted[:dy, :] = False
        elif dy < 0:
            shifted[dy:, :] = False
        if dx > 0:
            shifted[:, :dx] = False
        elif dx < 0:
            shifted[:, dx:] = False
        coast |= shifted
    # A coastal cell is land that is adjacent to ocean
    return coast & ~ocean_mask


def _bfs_distance(
    sources: list[tuple[int, int]],
    height: int,
    width: int,
    passable: np.ndarray,
) -> np.ndarray:
    """BFS distance from any source cell. inf where unreachable."""
    dist = np.full((height, width), np.inf, dtype=np.float32)
    queue: deque[tuple[int, int]] = deque()

    for x, y in sources:
        if 0 <= y < height and 0 <= x < width:
            dist[y, x] = 0.0
            queue.append((x, y))

    while queue:
        x, y = queue.popleft()
        for dy, dx in _DIRS4:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and passable[ny, nx] and dist[ny, nx] == np.inf:
                dist[ny, nx] = dist[y, x] + 1
                queue.append((nx, ny))

    return dist


def _local_settlement_density(
    positions: list[tuple[int, int]],
    height: int,
    width: int,
    radius: int = 3,
) -> np.ndarray:
    """Count how many initial settlements fall within Chebyshev radius of each cell."""
    density = np.zeros((height, width), dtype=np.int32)
    for x, y in positions:
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        density[y0:y1, x0:x1] += 1
    return density
