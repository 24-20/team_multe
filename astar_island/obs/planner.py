"""Observation engine: store, query budget accounting, and smart viewport planning.

Strategy (blueprint §5 + improvements):
- Split budget into reconnaissance (broad structural coverage) and adaptive exploitation.
- Prioritise viewports that maximise:
    1. Settlement density (most dynamic cells → highest scoring impact)
    2. Coverage novelty (fraction of cells not yet seen)
    3. Coastal richness (port formation zones)
- Because hidden parameters are shared across all seeds, observations from any seed
  inform the parameter posterior used for all seeds.  The store accumulates
  cross-seed statistics (survival rates, expansion rates) that shift the predictor's
  calibration for the entire round.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

import numpy as np

from ..world.terrain import WorldState, SeedWorld

logger = logging.getLogger(__name__)

MAX_VIEWPORT = 15
MIN_VIEWPORT = 5


@dataclass
class Observation:
    """One raw API response from /simulate, persisted to disk."""
    round_id: str
    seed_index: int
    viewport_x: int
    viewport_y: int
    viewport_w: int
    viewport_h: int
    grid: list[list[int]]         # [h][w] terrain codes after 50 years
    settlements: list[dict]       # settlement stats from API

    def cell_terrain(self, world_x: int, world_y: int) -> int | None:
        """Return the post-simulation terrain code at world coordinate (x,y), or None if outside viewport."""
        lx = world_x - self.viewport_x
        ly = world_y - self.viewport_y
        if 0 <= lx < self.viewport_w and 0 <= ly < self.viewport_h:
            return self.grid[ly][lx]
        return None

    def iter_cells(self) -> Iterator[tuple[int, int, int]]:
        """Yield (world_x, world_y, terrain_code) for every observed cell."""
        for ly in range(self.viewport_h):
            for lx in range(self.viewport_w):
                yield (
                    self.viewport_x + lx,
                    self.viewport_y + ly,
                    self.grid[ly][lx],
                )


class ObservationStore:
    """Persists observations to JSON files and provides fast lookups.

    Directory structure:
        data/{round_id}/obs/{seed_index}_{query_num:03d}.json
        data/{round_id}/round_detail.json
    """

    def __init__(self, data_dir: str | Path = "data"):
        self._root = Path(data_dir)
        self._cache: dict[str, list[Observation]] = {}  # round_id -> list

    def _round_dir(self, round_id: str) -> Path:
        d = self._root / round_id / "obs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(self, obs: Observation) -> None:
        """Persist observation to disk."""
        d = self._round_dir(obs.round_id)
        existing = len(list(d.glob(f"{obs.seed_index}_*.json")))
        path = d / f"{obs.seed_index}_{existing:03d}.json"
        path.write_text(json.dumps(asdict(obs), indent=2))
        logger.debug("Saved observation to %s", path)
        # Invalidate cache
        self._cache.pop(obs.round_id, None)

    def load_all(self, round_id: str) -> list[Observation]:
        """Load all observations for a round (cached)."""
        if round_id not in self._cache:
            d = self._root / round_id / "obs"
            obs_list = []
            if d.exists():
                for p in sorted(d.glob("*.json")):
                    try:
                        data = json.loads(p.read_text())
                        obs_list.append(Observation(**data))
                    except Exception as e:
                        logger.warning("Failed to load %s: %s", p, e)
            self._cache[round_id] = obs_list
        return self._cache[round_id]

    def load_for_seed(self, round_id: str, seed_index: int) -> list[Observation]:
        return [o for o in self.load_all(round_id) if o.seed_index == seed_index]

    def save_round_detail(self, round_id: str, detail: dict) -> None:
        path = self._root / round_id / "round_detail.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(detail, indent=2))

    def load_round_detail(self, round_id: str) -> dict | None:
        path = self._root / round_id / "round_detail.json"
        if path.exists():
            return json.loads(path.read_text())
        return None

    def query_count(self, round_id: str) -> int:
        return len(self.load_all(round_id))

    def observed_cells(self, round_id: str, seed_index: int) -> np.ndarray:
        """Return a boolean mask [height, width] of cells observed for this seed.

        Requires world_state for dimensions — use coverage_mask instead for a
        size-agnostic variant.
        """
        obs_list = self.load_for_seed(round_id, seed_index)
        if not obs_list:
            return np.zeros((0, 0), dtype=bool)
        # Infer grid size from observations — not ideal but workable
        max_y = max(o.viewport_y + o.viewport_h for o in obs_list)
        max_x = max(o.viewport_x + o.viewport_w for o in obs_list)
        mask = np.zeros((max_y, max_x), dtype=bool)
        for obs in obs_list:
            mask[
                obs.viewport_y : obs.viewport_y + obs.viewport_h,
                obs.viewport_x : obs.viewport_x + obs.viewport_w,
            ] = True
        return mask

    def round_level_stats(self, round_id: str) -> dict:
        """Compute cross-seed statistics useful for parameter inference.

        Returns a dict with:
        - total_initial_settlements: count seen across all viewports/seeds
        - alive_count: those still alive post-simulation
        - survival_rate: fraction alive
        - ruined_count: settlements that turned to ruin
        - new_settlements: settlements visible post-sim that weren't initial
        """
        all_obs = self.load_all(round_id)
        total_initial = alive = ruined = new_count = 0

        for obs in all_obs:
            for s in obs.settlements:
                if s.get("alive", True):
                    alive += 1
                else:
                    # Presence in viewport but not alive = ruin/destroyed
                    ruined += 1
                total_initial += 1

        survival_rate = alive / total_initial if total_initial > 0 else 0.5
        return {
            "total_initial_seen": total_initial,
            "alive_count": alive,
            "ruined_count": ruined,
            "survival_rate": survival_rate,
        }


class ViewportPlanner:
    """Selects which viewport to query next given remaining budget.

    Scoring function (per candidate viewport):
        score = w_coverage * novelty
              + w_settlement * settlement_density
              + w_coast * coastal_density
              + w_dynamic * dynamic_density

    Blueprint improvement: also weight by cross-seed leverage — viewports
    covering structurally diverse terrain are worth more early in the round
    because hidden parameters are shared and observations generalise.
    """

    def __init__(
        self,
        world: WorldState,
        store: ObservationStore,
        total_budget: int = 50,
        recon_fraction: float = 0.4,
    ):
        self._world = world
        self._store = store
        self._total_budget = total_budget
        # Queries reserved for broad reconnaissance before switching to exploitation
        self._recon_budget = max(1, int(total_budget * recon_fraction))

    def next_viewport(
        self,
        seed_index: int,
        queries_used: int,
        vp_w: int = MAX_VIEWPORT,
        vp_h: int = MAX_VIEWPORT,
    ) -> tuple[int, int, int, int] | None:
        """Return (x, y, w, h) for the best next viewport, or None if budget spent."""
        if queries_used >= self._total_budget:
            return None

        sw = self._world.seed(seed_index)
        vp_w, vp_h = self._clamp_viewport(sw, vp_w, vp_h)
        in_recon = queries_used < self._recon_budget

        candidates = list(self._enumerate_candidates(sw, vp_w, vp_h))
        if not candidates:
            return None

        # Build observed mask for this seed
        all_obs = self._store.load_for_seed(self._world.round_id, seed_index)
        observed = np.zeros((sw.height, sw.width), dtype=bool)
        for obs in all_obs:
            y0 = obs.viewport_y
            x0 = obs.viewport_x
            observed[y0:y0+obs.viewport_h, x0:x0+obs.viewport_w] = True

        scored = [
            (self._score(x, y, vp_w, vp_h, sw, observed, in_recon), x, y)
            for x, y in candidates
        ]
        scored.sort(key=lambda t: -t[0])
        _, best_x, best_y = scored[0]
        return best_x, best_y, vp_w, vp_h

    def _clamp_viewport(self, sw: SeedWorld, vp_w: int, vp_h: int) -> tuple[int, int]:
        """Clamp viewport dimensions to map size and allowed range."""
        vp_w = max(MIN_VIEWPORT, min(vp_w, sw.width))
        vp_h = max(MIN_VIEWPORT, min(vp_h, sw.height))
        return vp_w, vp_h

    def _enumerate_candidates(
        self,
        sw: SeedWorld,
        vp_w: int,
        vp_h: int,
    ) -> Iterator[tuple[int, int]]:
        """Yield (x, y) top-left corners that fit within the map."""
        vp_w, vp_h = self._clamp_viewport(sw, vp_w, vp_h)
        step = max(1, MAX_VIEWPORT // 2)  # 50% overlap grid
        for y in range(0, sw.height - vp_h + 1, step):
            for x in range(0, sw.width - vp_w + 1, step):
                yield x, y

    def _score(
        self,
        x: int,
        y: int,
        vp_w: int,
        vp_h: int,
        sw: SeedWorld,
        observed: np.ndarray,
        in_recon: bool,
    ) -> float:
        vp_slice_y = slice(y, y + vp_h)
        vp_slice_x = slice(x, x + vp_w)
        total_cells = vp_w * vp_h

        # Coverage novelty: fraction of cells not yet seen
        already_seen = observed[vp_slice_y, vp_slice_x].sum()
        novelty = 1.0 - already_seen / total_cells

        # Settlement density: initial settlements in this viewport
        s_in_vp = sum(
            1 for sx, sy in sw.settlement_positions
            if x <= sx < x + vp_w and y <= sy < y + vp_h
        )
        settlement_score = s_in_vp / max(1, len(sw.settlement_positions))

        # Coastal richness: coast cells in viewport (port formation zones)
        coast_in_vp = sw.coast_mask[vp_slice_y, vp_slice_x].sum()
        coast_score = coast_in_vp / total_cells

        # Dynamic cell density: non-static cells (excludes ocean + mountain)
        dynamic_in_vp = (~sw.static_mask[vp_slice_y, vp_slice_x]).sum()
        dynamic_score = dynamic_in_vp / total_cells

        if in_recon:
            # Reconnaissance: prioritise coverage + structural diversity
            return 0.45 * novelty + 0.25 * settlement_score + 0.15 * coast_score + 0.15 * dynamic_score
        else:
            # Exploitation: prioritise settlement density + novelty
            return 0.30 * novelty + 0.45 * settlement_score + 0.15 * coast_score + 0.10 * dynamic_score

    def plan_round(self, budget_per_seed: int | None = None) -> list[tuple[int, int, int, int, int]]:
        """Plan an ordered sequence of (seed_idx, x, y, w, h) queries for a full round.

        budget_per_seed: if None, distribute evenly across seeds.
        Returns list ordered by seed in round-robin fashion to spread information gain.
        """
        seeds = self._world.seeds_count
        total = self._total_budget
        per_seed = budget_per_seed or (total // seeds)
        queries: list[tuple[int, int, int, int, int]] = []

        # Round-robin planning: alternating seeds maximises cross-seed information
        seed_counts = [0] * seeds
        round_id = self._world.round_id
        # Pre-populate coverage with already-stored observations so we don't re-query
        simulated_observed = []
        for s in range(seeds):
            mask = np.zeros((self._world.map_height, self._world.map_width), dtype=bool)
            for obs in self._store.load_for_seed(round_id, s):
                mask[obs.viewport_y:obs.viewport_y+obs.viewport_h,
                     obs.viewport_x:obs.viewport_x+obs.viewport_w] = True
                seed_counts[s] += 1
            simulated_observed.append(mask)

        for _ in range(total):
            # Pick seed with fewest queries, preferring lower indices on tie
            seed = min(range(seeds), key=lambda s: seed_counts[s])
            if seed_counts[seed] >= per_seed:
                break

            sw = self._world.seed(seed)
            candidates = list(self._enumerate_candidates(sw, MAX_VIEWPORT, MAX_VIEWPORT))
            if not candidates:
                seed_counts[seed] = per_seed  # no more to query
                continue

            in_recon = seed_counts[seed] < max(1, int(per_seed * 0.4))
            vw, vh = self._clamp_viewport(sw, MAX_VIEWPORT, MAX_VIEWPORT)
            scored = [
                (self._score(x, y, vw, vh, sw, simulated_observed[seed], in_recon), x, y)
                for x, y in candidates
            ]
            scored.sort(key=lambda t: -t[0])
            _, bx, by = scored[0]

            queries.append((seed, bx, by, vw, vh))
            simulated_observed[seed][by:by+vh, bx:bx+vw] = True
            seed_counts[seed] += 1

        return queries
