"""Query policy: decide which viewports to query during an active round.

Phase A (baseline, no ML model):
    - 8 fixed 15x15 windows per seed = 40 queries total
    - Same schedule every round; covers ~90% of a 40x40 map
    - Remaining 10 queries used adaptively based on posterior entropy

Phase B (with LightGBM model):
    - Adaptive scoring: entropy + terrain diversity + coastal bonus
"""
from __future__ import annotations

from collections import namedtuple
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .round_state import RoundState


QuerySpec = namedtuple("QuerySpec", ["seed_idx", "x", "y", "w", "h"])

# Fixed 15x15 coverage windows for a 40x40 map
# Stride ~13 gives 3x3 = 9 positions; we pick 8 best-spread ones
_COVERAGE_POSITIONS_40x40 = [
    (0, 0), (0, 13), (0, 25),
    (13, 0),          (13, 25),
    (25, 0), (25, 13), (25, 25),
]  # (x, y) in API convention (col, row)


def _coverage_positions(width: int, height: int, vp: int = 15) -> list[tuple[int, int]]:
    """Generate 8 spread-out viewport positions for arbitrary map size.

    Returns list of (x, y) in API convention.
    """
    stride_x = max((width - vp) // 2, 1)
    stride_y = max((height - vp) // 2, 1)
    positions = []
    for row in range(0, height - vp + 1, stride_y):
        for col in range(0, width - vp + 1, stride_x):
            positions.append((col, row))  # API: x=col, y=row
    # Ensure last column/row is included
    if (width - vp, 0) not in positions:
        positions.append((width - vp, 0))
    if (0, height - vp) not in positions:
        positions.append((0, height - vp))
    if (width - vp, height - vp) not in positions:
        positions.append((width - vp, height - vp))
    # Deduplicate and clamp
    seen = set()
    out = []
    for x, y in positions:
        x = min(max(x, 0), width - vp)
        y = min(max(y, 0), height - vp)
        if (x, y) not in seen:
            seen.add((x, y))
            out.append((x, y))
    return out[:8]  # cap at 8


class QueryPolicy:
    """Determines which viewports to query given current round state.

    Args:
        state:  current RoundState
        budget: total query budget remaining (default 50)
        vp_w/h: viewport dimensions (default 15x15)
    """

    def __init__(
        self,
        state: "RoundState",
        budget: int = 50,
        vp_w: int = 15,
        vp_h: int = 15,
    ) -> None:
        self._state = state
        self._budget = budget
        self._vp_w = min(vp_w, 15)
        self._vp_h = min(vp_h, 15)
        self._positions = _coverage_positions(state.width, state.height, max(vp_w, vp_h))

    def coverage_queries(self) -> list[QuerySpec]:
        """Phase A: 8 fixed windows per seed, round-robin across seeds.

        Returns up to (8 * seeds_count) QuerySpec objects ordered to spread
        across seeds evenly (seed 0 q0, seed 1 q0, ..., seed 4 q0, seed 0 q1, ...).
        """
        specs: list[QuerySpec] = []
        n_seeds = self._state.seeds_count
        n_pos = min(len(self._positions), self._budget // n_seeds)

        for pos_idx in range(n_pos):
            for seed_idx in range(n_seeds):
                x, y = self._positions[pos_idx]
                # Clamp to map bounds
                x = min(x, self._state.width - self._vp_w)
                y = min(y, self._state.height - self._vp_h)
                specs.append(QuerySpec(seed_idx=seed_idx, x=x, y=y,
                                       w=self._vp_w, h=self._vp_h))
        return specs

    def adaptive_queries(self, remaining: int) -> list[QuerySpec]:
        """Phase B / top-up: score windows by entropy + dynamism and select best.

        Scoring (per window):
            score = 0.55 * entropy_mean + 0.30 * dynamism_mean + 0.15 * unobserved_fraction

        - entropy_mean:        mean posterior entropy in window (high = uncertain)
        - dynamism_mean:       fraction of observed cells in window that differ from
                               initial state (high = active/changing region)
        - unobserved_fraction: fraction of cells never queried (high = unexplored)

        Args:
            remaining: number of queries still available

        Returns:
            Up to `remaining` QuerySpec objects.
        """
        if remaining <= 0:
            return []

        posterior = self._state.posterior_mean  # [seeds, H, W, 6]
        from ..utils.math import entropy as _entropy
        ent_map = _entropy(posterior)  # [seeds, H, W]

        specs: list[QuerySpec] = []
        per_seed = max(1, remaining // self._state.seeds_count)

        for seed_idx in range(self._state.seeds_count):
            dyn_map = self._state.dynamism_map(seed_idx)  # [H, W]
            unobs_map = (self._state.observed_count[seed_idx] == 0).astype(np.float64)
            candidates = self._score_all_windows(
                ent_map[seed_idx], dyn_map, unobs_map
            )
            for score, x, y in candidates[:per_seed]:
                specs.append(QuerySpec(seed_idx=seed_idx, x=x, y=y,
                                       w=self._vp_w, h=self._vp_h))

        return specs[:remaining]

    def _score_all_windows(
        self,
        ent: np.ndarray,
        dyn: np.ndarray | None = None,
        unobs: np.ndarray | None = None,
    ) -> list[tuple[float, int, int]]:
        """Return (score, x, y) sorted descending for all valid windows.

        Args:
            ent:   [H, W] entropy map
            dyn:   [H, W] dynamism map (optional; 0 if not provided)
            unobs: [H, W] unobserved mask as float (optional; 0 if not provided)
        """
        H, W = ent.shape
        vw, vh = self._vp_w, self._vp_h

        if dyn is None:
            dyn = np.zeros_like(ent)
        if unobs is None:
            unobs = np.zeros_like(ent)

        # Normalise each component to [0, 1] range for fair weighting
        def _norm(arr: np.ndarray) -> np.ndarray:
            lo, hi = arr.min(), arr.max()
            if hi - lo < 1e-9:
                return np.zeros_like(arr)
            return (arr - lo) / (hi - lo)

        ent_n = _norm(ent)
        dyn_n = _norm(dyn)
        unobs_n = _norm(unobs)

        results = []
        for y in range(0, H - vh + 1, 3):
            for x in range(0, W - vw + 1, 3):
                e = float(ent_n[y:y + vh, x:x + vw].mean())
                d = float(dyn_n[y:y + vh, x:x + vw].mean())
                u = float(unobs_n[y:y + vh, x:x + vw].mean())
                score = 0.55 * e + 0.30 * d + 0.15 * u
                results.append((score, x, y))
        results.sort(reverse=True)
        return results
